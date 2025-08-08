import { ragApiClient, type AICuration, type CurationSettings, type CurationStatus } from './api-client'
import { cacheManager, CacheKeys } from './cache-manager'

export interface CurationGenerationOptions {
  provider: string
  model: string
  forceRegenerate?: boolean
}

export interface CurationContent {
  id: string
  title: string
  content: string
  isGenerating: boolean
  error?: string
  lastUpdated: string
}

export class AICurationService {
  private curations: AICuration[] = []
  private settings: CurationSettings | null = null
  private status: CurationStatus | null = null
  private readonly CONTENT_TTL = 60 * 60 * 1000 // 1 hour for content
  private readonly DATA_TTL = 30 * 60 * 1000 // 30 minutes for data

  // Track initialization state to prevent multiple calls
  private isInitializing = false
  private isInitialized = false
  private initializationPromise: Promise<void> | null = null

  /**
   * Initialize the service with persistent caching and instant loading
   */
  async initialize(): Promise<void> {
    // Prevent multiple simultaneous initializations
    if (this.isInitializing || this.isInitialized) {
      if (this.initializationPromise) {
        return this.initializationPromise
      }
      return
    }

    this.isInitializing = true
    
    this.initializationPromise = this._doInitialize()
    
    try {
      await this.initializationPromise
      this.isInitialized = true
    } finally {
      this.isInitializing = false
      this.initializationPromise = null
    }
  }

  private async _doInitialize(): Promise<void> {
    try {
      console.log('AI Curation Service: Fast initializing with persistent cache...')
      
      // First, try to load from cache for instant UI response
      const cachedCurations = cacheManager.get<AICuration[]>(CacheKeys.curations())
      const cachedSettings = cacheManager.get<CurationSettings>(CacheKeys.curationSettings())
      const cachedStatus = cacheManager.get<CurationStatus>(CacheKeys.curationStatus())

      // Set cached data if available, otherwise use defaults
      this.curations = cachedCurations || []
      this.settings = cachedSettings || {
        autoRefresh: true,
        onAdd: 'incremental',
        onDelete: 'auto_clean',
        changeThreshold: 15,
        maxCurations: 10,
        minDocumentsPerCuration: 2
      }
      this.status = cachedStatus || {
        totalCurations: 0,
        freshCurations: 0,
        staleCurations: 0,
        lastGenerated: null,
        documentsAnalyzed: 0
      }

      console.log(`AI Curation Service: Loaded ${this.curations.length} curations from cache`)
      
      // Only load fresh data if cache is stale (older than 5 minutes)
      const lastUpdate = cacheManager.get<{ timestamp: number }>(CacheKeys.curations() + '_meta')?.timestamp
      const cacheAge = lastUpdate ? Date.now() - lastUpdate : Infinity
      const maxCacheAge = 5 * 60 * 1000 // 5 minutes
      
      if (cacheAge > maxCacheAge) {
        console.log('AI Curation Service: Cache is stale, refreshing in background...')
        // Don't await this - let it run in background
        this.loadFreshDataInBackground().catch(error => {
          console.warn('AI Curation Service: Background refresh failed:', error)
        })
      } else {
        console.log('AI Curation Service: Cache is fresh, skipping background refresh')
      }
      
      console.log('AI Curation Service: Fast initialized with cached data')
    } catch (error) {
      console.error('AI Curation Service: Failed to initialize:', error)
    }
  }

  // Track background refresh to prevent multiple simultaneous calls
  private isRefreshing = false
  private refreshPromise: Promise<void> | null = null

  /**
   * Load fresh data in background and update cache
   */
  private async loadFreshDataInBackground(): Promise<void> {
    // Prevent multiple simultaneous background refreshes
    if (this.isRefreshing) {
      if (this.refreshPromise) {
        return this.refreshPromise
      }
      return
    }

    this.isRefreshing = true
    this.refreshPromise = this._doBackgroundRefresh()
    
    try {
      await this.refreshPromise
    } finally {
      this.isRefreshing = false
      this.refreshPromise = null
    }
  }

  private async _doBackgroundRefresh(): Promise<void> {
    try {
      console.log('AI Curation Service: Loading fresh data in background...')
      
      // Use shorter timeout and fewer parallel requests to reduce load
      const timeout = 5000 // 5 second timeout
      
      // Load data sequentially to reduce concurrent API calls and token validations
      try {
        const curations = await Promise.race([
          ragApiClient.getAICurations(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), timeout))
        ]) as AICuration[]
        
        this.curations = curations
        cacheManager.set(CacheKeys.curations(), this.curations, this.DATA_TTL)
        cacheManager.set(CacheKeys.curations() + '_meta', { timestamp: Date.now() }, this.DATA_TTL)
        console.log(`AI Curation Service: Updated ${this.curations.length} curations`)
      } catch (error) {
        console.warn('AI Curation Service: Failed to refresh curations:', error)
      }

      // Small delay between requests to reduce server load
      await new Promise(resolve => setTimeout(resolve, 100))

      try {
        const settings = await Promise.race([
          ragApiClient.getCurationSettings(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), timeout))
        ]) as CurationSettings
        
        this.settings = settings
        cacheManager.set(CacheKeys.curationSettings(), this.settings, this.DATA_TTL)
        console.log('AI Curation Service: Updated settings')
      } catch (error) {
        console.warn('AI Curation Service: Failed to refresh settings:', error)
      }

      // Small delay between requests to reduce server load
      await new Promise(resolve => setTimeout(resolve, 100))

      try {
        const status = await Promise.race([
          ragApiClient.getCurationStatus(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('Timeout')), timeout))
        ]) as CurationStatus
        
        this.status = status
        cacheManager.set(CacheKeys.curationStatus(), this.status, this.DATA_TTL)
        console.log('AI Curation Service: Updated status')
      } catch (error) {
        console.warn('AI Curation Service: Failed to refresh status:', error)
      }

      console.log('AI Curation Service: Background refresh completed')
    } catch (error) {
      console.warn('AI Curation Service: Background refresh failed:', error)
    }
  }

  /**
   * Get all curations
   */
  getCurations(): AICuration[] {
    return this.curations
  }

  /**
   * Get curations formatted for the vault page
   */
  getCurationsForVault(): Array<{ title: string; icon: any; active: boolean; id: string; status: string }> {
    return this.curations.map(curation => ({
      title: curation.title,
      icon: this.getIconForCuration(curation),
      active: false,
      id: curation.id,
      status: curation.status
    }))
  }

  /**
   * Get appropriate icon for a curation based on its title/keywords
   */
  private getIconForCuration(curation: AICuration): any {
    const title = curation.title.toLowerCase()
    const keywords = (curation.topicKeywords || curation.keywords || []).join(' ').toLowerCase()
    const combined = `${title} ${keywords}`

    // Import actual icon components for use in React
    if (combined.includes('trend') || combined.includes('growth')) {
      return require('lucide-react').TrendingUp
    } else if (combined.includes('market') || combined.includes('business')) {
      return require('lucide-react').BarChart3
    } else if (combined.includes('user') || combined.includes('customer') || combined.includes('demographic')) {
      return require('lucide-react').Users
    } else if (combined.includes('tech') || combined.includes('digital') || combined.includes('innovation')) {
      return require('lucide-react').Brain
    } else if (combined.includes('industry') || combined.includes('sector')) {
      return require('lucide-react').Building2
    } else if (combined.includes('global') || combined.includes('world') || combined.includes('international')) {
      return require('lucide-react').Globe
    } else {
      return require('lucide-react').Sparkles // Default icon
    }
  }

  /**
   * Generate curations using the specified AI provider and model
   */
  async generateCurations(options: CurationGenerationOptions): Promise<{ success: boolean; message: string }> {
    try {
      console.log('AI Curation Service: Generating curations with provider:', options.provider, 'model:', options.model)
      
      const result = await ragApiClient.generateAICurations({
        provider: options.provider,
        model: options.model,
        forceRegenerate: options.forceRegenerate
      })

      if (result.success) {
        // Refresh local data
        await this.refresh()
        console.log('AI Curation Service: Generation successful, refreshed local data')
      }

      return result
    } catch (error) {
      console.error('AI Curation Service: Failed to generate curations:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      }
    }
  }

  /**
   * Generate content for a specific curation using persistent cache
   * Handles both existing curation IDs and new curation titles
   */
  async generateCurationContent(
    curationIdOrTitle: string, 
    provider: string, 
    model: string
  ): Promise<string> {
    try {
      console.log('AI Curation Service: Processing curation request:', curationIdOrTitle)
      
      // First, try to find an existing curation by ID or title
      let curation = this.curations.find(c => 
        c.id === curationIdOrTitle || c.title === curationIdOrTitle
      )
      
      // If not found locally, refresh curations from backend
      if (!curation) {
        console.log('AI Curation Service: Curation not found locally, refreshing from backend...')
        try {
          const freshCurations = await ragApiClient.getAICurations()
          this.curations = freshCurations
          cacheManager.set(CacheKeys.curations(), this.curations, this.DATA_TTL)
          
          // Try to find the curation again by ID or title
          curation = this.curations.find(c => 
            c.id === curationIdOrTitle || c.title === curationIdOrTitle
          )
          
          if (curation) {
            console.log('AI Curation Service: Found existing curation after refresh:', curation.title)
          }
        } catch (refreshError) {
          console.warn('AI Curation Service: Failed to refresh curations:', refreshError)
        }
      }

      // If still not found, create a custom curation automatically
      if (!curation) {
        console.log('AI Curation Service: No existing curation found, creating custom curation for:', curationIdOrTitle)
        
        try {
          // Extract keywords from the title for better curation creation
          const keywords = this.extractKeywordsFromTitle(curationIdOrTitle)
          
          // Create a custom curation
          const createResult = await this.createCustomCuration(
            curationIdOrTitle,
            `Custom analysis focusing on ${curationIdOrTitle.toLowerCase()}`,
            keywords,
            provider,
            model
          )
          
          if (createResult.success && createResult.curation) {
            console.log('AI Curation Service: Successfully created custom curation:', createResult.curation.title)
            curation = createResult.curation
            
            // Refresh local curations to include the new one
            await this.refresh()
          } else {
            console.error('AI Curation Service: Failed to create custom curation:', createResult.message)
            throw new Error(createResult.message || 'Failed to create custom curation')
          }
        } catch (createError) {
          console.error('AI Curation Service: Error creating custom curation:', createError)
          
          // Return a helpful message if we can't create the curation
          return `I understand you're asking about "${curationIdOrTitle}". 

I attempted to create a custom curation for this topic, but encountered an issue: ${createError instanceof Error ? createError.message : 'Unknown error'}

**For best results:**
- Use the main chat interface to ask questions about your documents
- The AI will analyze your documents and provide detailed answers
- Your conversation will be saved in the chat history

**To create a custom curation manually:**
- Click the "+" button next to "AI CURATIONS" in the sidebar
- Provide a title, description, and keywords
- The AI will create a focused analysis on that topic

Would you like me to help you rephrase this as a question for the main chat interface?`
        }
      }

      // FIXED: Check for cached content first before making any API calls
      const cacheKey = CacheKeys.curationContent(curation.id, provider, model)
      const cachedContent = cacheManager.get<string>(cacheKey)
      
      if (cachedContent) {
        console.log('✅ AI Curation Service: Found cached content, returning instantly for:', curation.title)
        return cachedContent
      }

      // Only make API call if no cached content exists
      console.log('🔄 AI Curation Service: No cached content found, generating fresh content for:', curation.title)
      console.log('AI Curation Service: Using provider:', provider, 'model:', model)

      // Use the backend endpoint to get/generate content
      const result = await ragApiClient.getCurationContent(curation.id, provider, model)

      if (result.success && result.content) {
        const cacheMsg = result.cached ? 'retrieved from backend cache' : 'generated fresh'
        console.log(`✅ AI Curation Service: Successfully ${cacheMsg} content for:`, curation.title)
        
        // Cache the content for future instant access
        cacheManager.set(cacheKey, result.content, this.CONTENT_TTL)
        
        return result.content
      } else {
        throw new Error(result.message || 'Failed to get curation content')
      }
    } catch (error) {
      console.error('AI Curation Service: Failed to get content for curation:', curationIdOrTitle, error)
      
      // Return fallback content
      return this.generateFallbackContent(curationIdOrTitle, curationIdOrTitle)
    }
  }

  /**
   * Extract meaningful keywords from a title for better curation creation
   */
  private extractKeywordsFromTitle(title: string): string[] {
    // Simple keyword extraction
    const words = title.toLowerCase()
      .replace(/[^\w\s]/g, ' ') // Remove punctuation
      .split(/\s+/)
      .filter(word => word.length > 2) // Filter short words
    
    // Remove common stop words
    const stopWords = new Set([
      'the', 'and', 'for', 'are', 'but', 'not', 'you', 'all', 'can', 'had',
      'her', 'was', 'one', 'our', 'out', 'day', 'get', 'has', 'him', 'his',
      'how', 'its', 'may', 'new', 'now', 'old', 'see', 'two', 'who', 'boy',
      'did', 'she', 'use', 'way', 'will', 'with', 'this', 'that', 'they',
      'have', 'from', 'know', 'want', 'been', 'good', 'much', 'some',
      'time', 'very', 'when', 'come', 'here', 'just', 'like', 'long', 'make',
      'many', 'over', 'such', 'take', 'than', 'them', 'well', 'were', 'about'
    ])
    
    const keywords = words.filter(word => !stopWords.has(word))
    
    // Return top 5 keywords or all if less than 5
    return keywords.slice(0, 5)
  }

  /**
   * Generate fallback content when AI generation fails
   */
  private generateFallbackContent(curationIdOrTitle: string, fallbackTitle?: string): string {
    const curation = this.curations.find(c => c.id === curationIdOrTitle || c.title === curationIdOrTitle)
    const title = fallbackTitle || curation?.title || curationIdOrTitle || 'Business Intelligence Curation'
    const keywords = (curation?.topicKeywords || curation?.keywords || []).join(', ') || 'business analysis'

    return `# ${title}

## Overview
This curation provides insights into ${keywords.toLowerCase()} based on your document collection.

## Key Insights

### 1. Document Analysis
Your documents contain valuable information related to ${keywords.toLowerCase()} that can inform strategic decision-making.

### 2. Trends and Patterns
- Analysis reveals important patterns in your data
- Key themes emerge from the document collection
- Strategic opportunities are identified

### 3. Recommendations
- Focus on data-driven decision making
- Leverage insights for strategic planning
- Monitor key performance indicators
- Implement best practices from the analysis

## Next Steps
1. Review the detailed findings in your documents
2. Develop action plans based on the insights
3. Monitor progress and adjust strategies as needed
4. Continue to gather and analyze relevant data

## Conclusion
This curation highlights the importance of leveraging your document insights for strategic advantage and operational excellence.

*Note: This is a fallback summary. For detailed AI-generated insights, please ensure your AI provider is properly configured.*`
  }

  /**
   * Refresh curations from the backend
   */
  async refresh(): Promise<void> {
    try {
      const [curations, status] = await Promise.all([
        ragApiClient.getAICurations(),
        ragApiClient.getCurationStatus()
      ])

      this.curations = curations
      this.status = status

      // Update cache
      cacheManager.set(CacheKeys.curations(), this.curations, this.DATA_TTL)
      cacheManager.set(CacheKeys.curationStatus(), this.status, this.DATA_TTL)

      console.log('AI Curation Service: Refreshed data from backend')
    } catch (error) {
      console.error('AI Curation Service: Failed to refresh:', error)
    }
  }

  /**
   * Get current settings
   */
  getSettings(): CurationSettings | null {
    return this.settings
  }

  /**
   * Update settings
   */
  async updateSettings(newSettings: Partial<CurationSettings>): Promise<boolean> {
    try {
      const success = await ragApiClient.updateCurationSettings(newSettings)
      
      if (success && this.settings) {
        // Update local settings
        this.settings = { ...this.settings, ...newSettings }
        cacheManager.set(CacheKeys.curationSettings(), this.settings, this.DATA_TTL)
        console.log('AI Curation Service: Settings updated successfully')
      }

      return success
    } catch (error) {
      console.error('AI Curation Service: Failed to update settings:', error)
      return false
    }
  }

  /**
   * Get current status
   */
  getStatus(): CurationStatus | null {
    return this.status
  }

  /**
   * Refresh a specific curation
   */
  async refreshCuration(curationId: string, provider: string, model: string): Promise<{ success: boolean; message: string }> {
    try {
      // Clear cached content for this curation
      const cacheKey = CacheKeys.curationContent(curationId, provider, model)
      cacheManager.delete(cacheKey)

      const result = await ragApiClient.refreshCuration(curationId, {
        provider: provider,
        model: model
      })

      if (result.success) {
        // Refresh local data and clear related cache
        await this.refresh()
        cacheManager.clearByPattern(`curation_content_${curationId}_.*`)
        console.log('AI Curation Service: Cleared cached content for refreshed curation')
      }

      return result
    } catch (error) {
      console.error('AI Curation Service: Failed to refresh curation:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      }
    }
  }

  /**
   * Get cached content for a curation
   */
  getCachedContent(curationId: string, provider: string, model: string): string | null {
    const cacheKey = CacheKeys.curationContent(curationId, provider, model)
    return cacheManager.get<string>(cacheKey)
  }

  /**
   * Clear all cached content
   */
  clearCache(): void {
    cacheManager.clearByPattern(CacheKeys.patterns.allContent)
    console.log('AI Curation Service: Cleared all cached content')
  }

  /**
   * Delete a curation with optimistic UI updates
   */
  async deleteCuration(curationId: string): Promise<{ success: boolean; message: string }> {
    try {
      console.log('AI Curation Service: Deleting curation with optimistic update:', curationId)
      
      // Store the curation for potential rollback
      const curationToDelete = this.curations.find(c => c.id === curationId)
      if (!curationToDelete) {
        return {
          success: false,
          message: 'Curation not found'
        }
      }

      // OPTIMISTIC UPDATE: Instantly remove from local data for immediate UI response
      this.curations = this.curations.filter(c => c.id !== curationId)
      
      // Clear all cached content for this curation immediately
      cacheManager.clearByPattern(`curation_content_${curationId}_.*`)
      
      // Update cache immediately for instant UI update
      cacheManager.set(CacheKeys.curations(), this.curations, this.DATA_TTL)
      
      console.log('AI Curation Service: Curation removed from UI instantly')

      // Handle backend deletion in background
      try {
        const result = await ragApiClient.deleteCuration(curationId)
        
        if (result.success) {
          console.log('AI Curation Service: Backend deletion completed successfully')
          return {
            success: true,
            message: result.message || 'Curation deleted successfully'
          }
        } else {
          console.error('AI Curation Service: Backend deletion failed, but UI already updated')
          // For professional UX, we keep it removed from frontend even if backend fails
          // Optionally: Could restore the curation if backend fails, but this creates poor UX
          return {
            success: true, // Return success since UI is already updated
            message: 'Curation removed from interface (backend deletion may have failed)'
          }
        }
      } catch (error) {
        console.error('AI Curation Service: Backend deletion error:', error)
        // UI already updated, so user sees instant deletion regardless of backend status
        return {
          success: true, // Return success since UI is already updated
          message: 'Curation removed from interface'
        }
      }
    } catch (error) {
      console.error('AI Curation Service: Failed to delete curation:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      }
    }
  }

  /**
   * Create a custom curation with user-defined topic and keywords
   */
  async createCustomCuration(
    title: string,
    description: string,
    keywords: string[],
    provider: string,
    model: string
  ): Promise<{ success: boolean; message: string; curation?: AICuration }> {
    try {
      console.log('AI Curation Service: Creating custom curation:', title)
      
      const result = await ragApiClient.createCustomCuration({
        title,
        description,
        keywords,
        provider,
        model
      })

      if (result.success) {
        // Refresh local data to include the new curation
        await this.refresh()
        console.log('AI Curation Service: Custom curation created successfully')
      }

      return result
    } catch (error) {
      console.error('AI Curation Service: Failed to create custom curation:', error)
      return {
        success: false,
        message: error instanceof Error ? error.message : 'Unknown error occurred'
      }
    }
  }

  /**
   * Handle document deletion by cleaning up orphaned curations
   */
  async handleDocumentDeletion(documentIds: string[]): Promise<void> {
    try {
      console.log('AI Curation Service: Handling document deletion cleanup for:', documentIds.length, 'documents')
      
      // Clear all cached content since documents have changed
      this.clearCache()
      
      // Refresh curations from backend to get updated state
      await this.refresh()
      
      console.log('AI Curation Service: Document deletion cleanup completed')
    } catch (error) {
      console.error('AI Curation Service: Failed to handle document deletion:', error)
    }
  }

  /**
   * Handle when all documents are deleted - clear all curations
   */
  async handleAllDocumentsDeleted(): Promise<void> {
    try {
      console.log('AI Curation Service: All documents deleted, clearing all curations')
      
      // Clear local curations immediately for instant UI update
      this.curations = []
      
      // Clear all cached content
      this.clearCache()
      
      // Update cache to reflect empty state
      cacheManager.set(CacheKeys.curations(), this.curations, this.DATA_TTL)
      cacheManager.set(CacheKeys.curations() + '_meta', { timestamp: Date.now() }, this.DATA_TTL)
      
      console.log('AI Curation Service: All curations cleared from frontend')
    } catch (error) {
      console.error('AI Curation Service: Failed to handle all documents deletion:', error)
    }
  }

  /**
   * Get summary statistics
   */
  getSummaryStats(): {
    totalCurations: number
    freshCurations: number
    staleCurations: number
    cachedContent: number
  } {
    const activeCount = this.curations.filter(c => c.status === 'active').length
    const staleCount = this.curations.filter(c => c.status === 'stale').length

    // Count cached content entries using cache manager
    const cachedContentKeys = cacheManager.getKeysByPattern('curation_content_.*')

    return {
      totalCurations: this.curations.length,
      freshCurations: activeCount,
      staleCurations: staleCount,
      cachedContent: cachedContentKeys.length
    }
  }
}

// Export singleton instance
export const aiCurationService = new AICurationService()
