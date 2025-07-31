import { ragApiClient, type AICuration, type CurationSettings, type CurationStatus } from './api-client'

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
}

export class AICurationService {
  private curations: AICuration[] = []
  private settings: CurationSettings | null = null
  private status: CurationStatus | null = null
  private contentCache: Map<string, CurationContent> = new Map()

  /**
   * Initialize the service by loading curations and settings
   */
  async initialize(): Promise<void> {
    try {
      console.log('AI Curation Service: Initializing...')
      
      // Load curations, settings, and status in parallel
      const [curations, settings, status] = await Promise.all([
        ragApiClient.getAICurations(),
        ragApiClient.getCurationSettings(),
        ragApiClient.getCurationStatus()
      ])

      this.curations = curations
      this.settings = settings
      this.status = status

      console.log('AI Curation Service: Initialized successfully')
      console.log(`  - Curations: ${curations.length}`)
      console.log(`  - Settings: Auto-refresh ${settings.autoRefresh ? 'enabled' : 'disabled'}`)
      console.log(`  - Status: ${status.totalCurations} total, ${status.freshCurations} fresh`)
    } catch (error) {
      console.error('AI Curation Service: Failed to initialize:', error)
      // Set default values on error
      this.curations = []
      this.settings = {
        autoRefresh: true,
        onAdd: 'incremental',
        onDelete: 'auto_clean',
        changeThreshold: 15,
        maxCurations: 10,
        minDocumentsPerCuration: 2
      }
      this.status = {
        totalCurations: 0,
        freshCurations: 0,
        staleCurations: 0,
        lastGenerated: null,
        documentsAnalyzed: 0
      }
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
    const keywords = curation.topicKeywords.join(' ').toLowerCase()
    const combined = `${title} ${keywords}`

    // Import icons dynamically based on content
    if (combined.includes('trend') || combined.includes('growth')) {
      return 'TrendingUp'
    } else if (combined.includes('market') || combined.includes('business')) {
      return 'BarChart3'
    } else if (combined.includes('user') || combined.includes('customer') || combined.includes('demographic')) {
      return 'Users'
    } else if (combined.includes('tech') || combined.includes('digital') || combined.includes('innovation')) {
      return 'Brain'
    } else if (combined.includes('industry') || combined.includes('sector')) {
      return 'Building2'
    } else if (combined.includes('global') || combined.includes('world') || combined.includes('international')) {
      return 'Globe'
    } else {
      return 'Sparkles' // Default icon
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
   * Generate content for a specific curation using the selected AI provider
   */
  async generateCurationContent(
    curationId: string, 
    provider: string, 
    model: string
  ): Promise<string> {
    try {
      // Check if content is already cached and not generating
      const cached = this.contentCache.get(curationId)
      if (cached && !cached.isGenerating && !cached.error) {
        console.log('AI Curation Service: Returning cached content for curation:', curationId)
        return cached.content
      }

      // Find the curation
      const curation = this.curations.find(c => c.id === curationId)
      if (!curation) {
        throw new Error('Curation not found')
      }

      // Mark as generating
      this.contentCache.set(curationId, {
        id: curationId,
        title: curation.title,
        content: '',
        isGenerating: true
      })

      console.log('AI Curation Service: Generating content for curation:', curation.title)
      console.log('AI Curation Service: Using provider:', provider, 'model:', model)

      // Use the RAG query endpoint to generate content based on the curation
      const prompt = this.buildCurationPrompt(curation)
      
      const response = await ragApiClient.queryDocuments(prompt, {
        provider: provider,
        model: model,
        hybrid: false, // Use pure RAG for curations
        maxContext: true, // Use maximum context for better insights
        documentIds: curation.documentIds.length > 0 ? curation.documentIds : undefined
      })

      const content = response.answer || 'Failed to generate curation content.'

      // Cache the generated content
      this.contentCache.set(curationId, {
        id: curationId,
        title: curation.title,
        content: content,
        isGenerating: false
      })

      console.log('AI Curation Service: Successfully generated content for:', curation.title)
      return content

    } catch (error) {
      console.error('AI Curation Service: Failed to generate content for curation:', curationId, error)
      
      // Cache the error
      this.contentCache.set(curationId, {
        id: curationId,
        title: 'Error',
        content: '',
        isGenerating: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      })

      // Return fallback content
      return this.generateFallbackContent(curationId)
    }
  }

  /**
   * Build a prompt for generating curation content
   */
  private buildCurationPrompt(curation: AICuration): string {
    const keywords = curation.topicKeywords.join(', ')
    
    return `Generate a comprehensive business intelligence curation titled "${curation.title}".

Focus on these key topics and keywords: ${keywords}

Please provide:
1. An executive summary of the key insights
2. Main findings and trends
3. Strategic implications and recommendations
4. Supporting data and evidence from the documents
5. Actionable next steps

Format the response using markdown with clear headings, bullet points, and professional structure. Make it visually appealing and easy to read.

Base your analysis on the uploaded documents and provide specific, actionable insights that would be valuable for business decision-making.`
  }

  /**
   * Generate fallback content when AI generation fails
   */
  private generateFallbackContent(curationId: string): string {
    const curation = this.curations.find(c => c.id === curationId)
    const title = curation?.title || 'Business Intelligence Curation'
    const keywords = curation?.topicKeywords.join(', ') || 'business analysis'

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
      // Clear cached content
      this.contentCache.delete(curationId)

      const result = await ragApiClient.refreshCuration(curationId, {
        provider: provider,
        model: model
      })

      if (result.success) {
        // Refresh local data
        await this.refresh()
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
   * Check if a curation is currently generating content
   */
  isGeneratingContent(curationId: string): boolean {
    const cached = this.contentCache.get(curationId)
    return cached?.isGenerating || false
  }

  /**
   * Get cached content for a curation
   */
  getCachedContent(curationId: string): CurationContent | null {
    return this.contentCache.get(curationId) || null
  }

  /**
   * Clear all cached content
   */
  clearCache(): void {
    this.contentCache.clear()
    console.log('AI Curation Service: Cleared content cache')
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
   * Get summary statistics
   */
  getSummaryStats(): {
    totalCurations: number
    freshCurations: number
    staleCurations: number
    cachedContent: number
  } {
    const freshCount = this.curations.filter(c => c.status === 'fresh').length
    const staleCount = this.curations.filter(c => c.status === 'stale').length

    return {
      totalCurations: this.curations.length,
      freshCurations: freshCount,
      staleCurations: staleCount,
      cachedContent: this.contentCache.size
    }
  }
}

// Export singleton instance
export const aiCurationService = new AICurationService()
