/**
 * Focused Cache Manager - Caching ONLY for AI Curation and Chat History
 * Implements optimized caching specifically for these two features only
 * Documents upload/delete and other functionalities are excluded from caching
 */

export interface CacheEntry<T> {
  data: T
  timestamp: number
  ttl: number // Time to live in milliseconds
  key: string
  version: string
}

export interface CacheStats {
  totalEntries: number
  totalSize: number
  hitRate: number
  missRate: number
  hits: number
  misses: number
}

class CacheManager {
  private static instance: CacheManager
  private cache: Map<string, CacheEntry<any>> = new Map()
  private stats = {
    hits: 0,
    misses: 0,
    totalEntries: 0
  }
  private readonly STORAGE_KEY = 'sixthvault_cache'
  private readonly VERSION = '2.0.0' // Updated for focused caching
  private readonly DEFAULT_TTL = 30 * 60 * 1000 // 30 minutes
  
  // Focused TTL settings for AI Curation and Chat History ONLY
  private readonly CURATION_CONTENT_TTL = 2 * 60 * 60 * 1000 // 2 hours for curation content
  private readonly CURATION_DATA_TTL = 60 * 60 * 1000 // 1 hour for curation data
  private readonly CHAT_HISTORY_TTL = 24 * 60 * 60 * 1000 // 24 hours for chat history
  private readonly CONVERSATION_TTL = 12 * 60 * 60 * 1000 // 12 hours for conversations

  private constructor() {
    this.loadFromStorage()
    this.startCleanupInterval()
    this.setupStorageSync()
  }

  static getInstance(): CacheManager {
    if (!CacheManager.instance) {
      CacheManager.instance = new CacheManager()
    }
    return CacheManager.instance
  }

  /**
   * Store data in cache with automatic persistence
   */
  set<T>(key: string, data: T, ttl: number = this.DEFAULT_TTL): void {
    try {
      const entry: CacheEntry<T> = {
        data,
        timestamp: Date.now(),
        ttl,
        key,
        version: this.VERSION
      }

      this.cache.set(key, entry)
      this.stats.totalEntries = this.cache.size
      this.persistToStorage()
      
      console.log(`Cache: Stored entry '${key}' (TTL: ${ttl}ms)`)
    } catch (error) {
      console.error(`Cache: Failed to store entry '${key}':`, error)
    }
  }

  /**
   * Retrieve data from cache
   */
  get<T>(key: string): T | null {
    try {
      const entry = this.cache.get(key)
      
      if (!entry) {
        this.stats.misses++
        console.log(`Cache: Miss for key '${key}'`)
        return null
      }

      // Check if entry has expired
      if (Date.now() - entry.timestamp > entry.ttl) {
        this.cache.delete(key)
        this.stats.misses++
        console.log(`Cache: Expired entry '${key}' removed`)
        return null
      }

      // Check version compatibility
      if (entry.version !== this.VERSION) {
        this.cache.delete(key)
        this.stats.misses++
        console.log(`Cache: Version mismatch for '${key}', removed`)
        return null
      }

      this.stats.hits++
      console.log(`Cache: Hit for key '${key}'`)
      return entry.data as T
    } catch (error) {
      console.error(`Cache: Failed to retrieve entry '${key}':`, error)
      this.stats.misses++
      return null
    }
  }

  /**
   * Check if cache has a valid entry for key
   */
  has(key: string): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false
    
    // Check expiration
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key)
      return false
    }
    
    // Check version
    if (entry.version !== this.VERSION) {
      this.cache.delete(key)
      return false
    }
    
    return true
  }

  /**
   * Remove specific cache entry
   */
  delete(key: string): boolean {
    const deleted = this.cache.delete(key)
    if (deleted) {
      this.stats.totalEntries = this.cache.size
      this.persistToStorage()
      console.log(`Cache: Deleted entry '${key}'`)
    }
    return deleted
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear()
    this.stats = { hits: 0, misses: 0, totalEntries: 0 }
    this.persistToStorage()
    console.log('Cache: Cleared all entries')
  }

  /**
   * Clear cache entries by pattern
   */
  clearByPattern(pattern: string): number {
    let deletedCount = 0
    const regex = new RegExp(pattern)
    
    for (const key of this.cache.keys()) {
      if (regex.test(key)) {
        this.cache.delete(key)
        deletedCount++
      }
    }
    
    if (deletedCount > 0) {
      this.stats.totalEntries = this.cache.size
      this.persistToStorage()
      console.log(`Cache: Deleted ${deletedCount} entries matching pattern '${pattern}'`)
    }
    
    return deletedCount
  }

  /**
   * Update TTL for existing entry
   */
  updateTTL(key: string, newTTL: number): boolean {
    const entry = this.cache.get(key)
    if (!entry) return false
    
    entry.ttl = newTTL
    entry.timestamp = Date.now() // Reset timestamp
    this.cache.set(key, entry)
    this.persistToStorage()
    
    console.log(`Cache: Updated TTL for '${key}' to ${newTTL}ms`)
    return true
  }

  /**
   * Get cache statistics
   */
  getStats(): CacheStats {
    const totalRequests = this.stats.hits + this.stats.misses
    return {
      totalEntries: this.cache.size,
      totalSize: this.calculateCacheSize(),
      hitRate: totalRequests > 0 ? (this.stats.hits / totalRequests) * 100 : 0,
      missRate: totalRequests > 0 ? (this.stats.misses / totalRequests) * 100 : 0,
      hits: this.stats.hits,
      misses: this.stats.misses
    }
  }

  /**
   * Get all cache keys
   */
  getKeys(): string[] {
    return Array.from(this.cache.keys())
  }

  /**
   * Get cache keys by pattern
   */
  getKeysByPattern(pattern: string): string[] {
    const regex = new RegExp(pattern)
    return Array.from(this.cache.keys()).filter(key => regex.test(key))
  }

  /**
   * Load cache from localStorage
   */
  private loadFromStorage(): void {
    try {
      const stored = localStorage.getItem(this.STORAGE_KEY)
      if (!stored) return

      const parsed = JSON.parse(stored)
      if (!Array.isArray(parsed)) return

      for (const entry of parsed) {
        if (this.isValidEntry(entry)) {
          this.cache.set(entry.key, entry)
        }
      }

      this.stats.totalEntries = this.cache.size
      console.log(`Cache: Loaded ${this.cache.size} entries from storage`)
    } catch (error) {
      console.error('Cache: Failed to load from storage:', error)
      // Clear corrupted storage
      localStorage.removeItem(this.STORAGE_KEY)
    }
  }

  /**
   * Persist cache to localStorage
   */
  private persistToStorage(): void {
    try {
      const entries = Array.from(this.cache.values())
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(entries))
      console.log(`Cache: Persisted ${entries.length} entries to storage`)
    } catch (error) {
      console.error('Cache: Failed to persist to storage:', error)
      // If storage is full, clear old entries and try again
      this.cleanupExpiredEntries()
      try {
        const entries = Array.from(this.cache.values())
        localStorage.setItem(this.STORAGE_KEY, JSON.stringify(entries))
      } catch (retryError) {
        console.error('Cache: Failed to persist after cleanup:', retryError)
      }
    }
  }

  /**
   * Validate cache entry structure
   */
  private isValidEntry(entry: any): boolean {
    return (
      entry &&
      typeof entry === 'object' &&
      typeof entry.key === 'string' &&
      typeof entry.timestamp === 'number' &&
      typeof entry.ttl === 'number' &&
      typeof entry.version === 'string' &&
      entry.data !== undefined
    )
  }

  /**
   * Calculate approximate cache size in bytes
   */
  private calculateCacheSize(): number {
    try {
      const serialized = JSON.stringify(Array.from(this.cache.values()))
      return new Blob([serialized]).size
    } catch {
      return 0
    }
  }

  /**
   * Clean up expired entries
   */
  private cleanupExpiredEntries(): void {
    const now = Date.now()
    let deletedCount = 0

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > entry.ttl || entry.version !== this.VERSION) {
        this.cache.delete(key)
        deletedCount++
      }
    }

    if (deletedCount > 0) {
      this.stats.totalEntries = this.cache.size
      console.log(`Cache: Cleaned up ${deletedCount} expired entries`)
    }
  }

  /**
   * Start automatic cleanup interval with enhanced focused cleanup
   */
  private startCleanupInterval(): void {
    // Enhanced cleanup every 3 minutes for better performance
    setInterval(() => {
      this.cleanupExpiredEntries()
      this.persistToStorage()
      
      // Log focused cache statistics periodically
      const focusedStats = this.getFocusedStats()
      console.log('🔄 Auto Cleanup - Focused Cache Stats:', focusedStats)
    }, 3 * 60 * 1000) // Cleanup every 3 minutes
    
    // Additional cleanup for very old entries every 30 minutes
    setInterval(() => {
      this.cleanupVeryOldEntries()
    }, 30 * 60 * 1000) // Deep cleanup every 30 minutes
  }

  /**
   * Clean up very old entries (older than 24 hours) for memory optimization
   */
  private cleanupVeryOldEntries(): void {
    const now = Date.now()
    const maxAge = 24 * 60 * 60 * 1000 // 24 hours
    let deletedCount = 0

    for (const [key, entry] of this.cache.entries()) {
      if (now - entry.timestamp > maxAge) {
        this.cache.delete(key)
        deletedCount++
      }
    }

    if (deletedCount > 0) {
      this.stats.totalEntries = this.cache.size
      console.log(`🧹 Deep Cleanup: Removed ${deletedCount} very old cache entries`)
      this.persistToStorage()
    }
  }

  /**
   * Setup storage event listener for cross-tab sync
   */
  private setupStorageSync(): void {
    window.addEventListener('storage', (event) => {
      if (event.key === this.STORAGE_KEY) {
        console.log('Cache: Detected storage change, reloading cache')
        this.cache.clear()
        this.loadFromStorage()
      }
    })
  }

  /**
   * Preload data with background fetch
   */
  async preload<T>(
    key: string,
    fetchFunction: () => Promise<T>,
    ttl: number = this.DEFAULT_TTL
  ): Promise<T> {
    // Check if we have cached data
    const cached = this.get<T>(key)
    if (cached) {
      // Return cached data immediately
      console.log(`Cache: Serving cached data for '${key}'`)
      
      // Optionally refresh in background if data is getting old
      const entry = this.cache.get(key)
      if (entry && Date.now() - entry.timestamp > ttl * 0.7) {
        console.log(`Cache: Background refresh triggered for '${key}'`)
        fetchFunction().then(data => {
          this.set(key, data, ttl)
        }).catch(error => {
          console.error(`Cache: Background refresh failed for '${key}':`, error)
        })
      }
      
      return cached
    }

    // Fetch fresh data
    console.log(`Cache: Fetching fresh data for '${key}'`)
    try {
      const data = await fetchFunction()
      this.set(key, data, ttl)
      return data
    } catch (error) {
      console.error(`Cache: Failed to fetch data for '${key}':`, error)
      throw error
    }
  }

  /**
   * Batch operations for multiple cache entries
   */
  setMultiple<T>(entries: Array<{ key: string; data: T; ttl?: number }>): void {
    for (const entry of entries) {
      this.set(entry.key, entry.data, entry.ttl)
    }
  }

  getMultiple<T>(keys: string[]): Record<string, T | null> {
    const result: Record<string, T | null> = {}
    for (const key of keys) {
      result[key] = this.get<T>(key)
    }
    return result
  }

  /**
   * Clear document-related caches
   */
  clearDocumentCaches(): number {
    console.log('Cache: Clearing all document-related caches')
    let deletedCount = 0
    
    // Clear documents list cache
    if (this.delete(CacheKeys.documents())) {
      deletedCount++
    }
    
    // Clear all document content caches
    deletedCount += this.clearByPattern('document_.*')
    
    // Clear any other document-related patterns
    deletedCount += this.clearByPattern('documents_.*')
    
    console.log(`Cache: Cleared ${deletedCount} document-related cache entries`)
    return deletedCount
  }

  /**
   * Clear all vault-related caches (documents, curations, summaries)
   */
  clearVaultCaches(): number {
    console.log('Cache: Clearing all vault-related caches')
    let deletedCount = 0
    
    // Clear documents
    deletedCount += this.clearDocumentCaches()
    
    // Clear curations
    deletedCount += this.clearByPattern(CacheKeys.patterns.allCurations)
    
    // Clear summaries
    deletedCount += this.clearByPattern(CacheKeys.patterns.allSummaries)
    
    // Clear analytics
    if (this.delete(CacheKeys.analytics())) {
      deletedCount++
    }
    
    console.log(`Cache: Cleared ${deletedCount} vault-related cache entries`)
    return deletedCount
  }

  /**
   * FOCUSED CACHING METHODS - AI Curation and Chat History ONLY
   */

  /**
   * Cache AI Curation content with optimized TTL
   */
  setCurationContent(curationId: string, content: string, provider: string, model: string): void {
    const key = CacheKeys.curationContent(curationId, provider, model)
    this.set(key, content, this.CURATION_CONTENT_TTL)
    console.log(`✅ Focused Cache: Stored AI curation content for ${curationId}`)
  }

  /**
   * Get cached AI Curation content
   */
  getCurationContent(curationId: string, provider: string, model: string): string | null {
    const key = CacheKeys.curationContent(curationId, provider, model)
    const content = this.get<string>(key)
    if (content) {
      console.log(`⚡ Focused Cache: Retrieved AI curation content for ${curationId}`)
    }
    return content
  }

  /**
   * Cache AI Curation data with optimized TTL
   */
  setCurationData(curations: any[]): void {
    this.set(CacheKeys.curations(), curations, this.CURATION_DATA_TTL)
    console.log(`✅ Focused Cache: Stored ${curations.length} AI curations`)
  }

  /**
   * Get cached AI Curation data
   */
  getCurationData(): any[] | null {
    const curations = this.get<any[]>(CacheKeys.curations())
    if (curations) {
      console.log(`⚡ Focused Cache: Retrieved ${curations.length} AI curations`)
    }
    return curations
  }

  /**
   * Cache Chat History/Conversations with optimized TTL
   */
  setChatHistory(conversations: any[]): void {
    this.set('chat_conversations', conversations, this.CHAT_HISTORY_TTL)
    console.log(`✅ Focused Cache: Stored ${conversations.length} chat conversations`)
  }

  /**
   * Get cached Chat History/Conversations
   */
  getChatHistory(): any[] | null {
    const conversations = this.get<any[]>('chat_conversations')
    if (conversations) {
      console.log(`⚡ Focused Cache: Retrieved ${conversations.length} chat conversations`)
    }
    return conversations
  }

  /**
   * Cache individual conversation content
   */
  setConversationContent(conversationId: string, conversation: any, messages: any[]): void {
    const key = `conversation_content_${conversationId}`
    const data = { conversation, messages }
    this.set(key, data, this.CONVERSATION_TTL)
    console.log(`✅ Focused Cache: Stored conversation content for ${conversationId}`)
  }

  /**
   * Get cached conversation content
   */
  getConversationContent(conversationId: string): { conversation: any; messages: any[] } | null {
    const key = `conversation_content_${conversationId}`
    const data = this.get<{ conversation: any; messages: any[] }>(key)
    if (data) {
      console.log(`⚡ Focused Cache: Retrieved conversation content for ${conversationId}`)
    }
    return data
  }

  /**
   * Clear only AI Curation caches (focused clearing)
   */
  clearCurationCaches(): number {
    console.log('🎯 Focused Cache: Clearing AI Curation caches only')
    let deletedCount = 0
    
    // Clear curations list
    if (this.delete(CacheKeys.curations())) {
      deletedCount++
    }
    
    // Clear all curation content
    deletedCount += this.clearByPattern('curation_content_.*')
    
    // Clear curation settings
    if (this.delete(CacheKeys.curationSettings())) {
      deletedCount++
    }
    
    console.log(`✅ Focused Cache: Cleared ${deletedCount} AI curation cache entries`)
    return deletedCount
  }

  /**
   * Clear only Chat History caches (focused clearing)
   */
  clearChatHistoryCaches(): number {
    console.log('🎯 Focused Cache: Clearing Chat History caches only')
    let deletedCount = 0
    
    // Clear conversations list
    if (this.delete('chat_conversations')) {
      deletedCount++
    }
    
    // Clear all conversation content
    deletedCount += this.clearByPattern('conversation_content_.*')
    
    console.log(`✅ Focused Cache: Cleared ${deletedCount} chat history cache entries`)
    return deletedCount
  }

  /**
   * Clear ONLY focused caches (AI Curation + Chat History)
   */
  clearFocusedCaches(): number {
    console.log('🎯 Focused Cache: Clearing ONLY AI Curation and Chat History caches')
    let deletedCount = 0
    
    deletedCount += this.clearCurationCaches()
    deletedCount += this.clearChatHistoryCaches()
    
    console.log(`✅ Focused Cache: Cleared ${deletedCount} focused cache entries total`)
    return deletedCount
  }

  /**
   * Get focused cache statistics (AI Curation + Chat History only)
   */
  getFocusedStats(): {
    curationEntries: number
    chatHistoryEntries: number
    totalFocusedEntries: number
    focusedCacheSize: number
  } {
    const curationKeys = this.getKeysByPattern('(ai_curations|curation_.*)')
    const chatKeys = this.getKeysByPattern('(chat_conversations|conversation_.*)')
    
    return {
      curationEntries: curationKeys.length,
      chatHistoryEntries: chatKeys.length,
      totalFocusedEntries: curationKeys.length + chatKeys.length,
      focusedCacheSize: this.calculateFocusedCacheSize(curationKeys.concat(chatKeys))
    }
  }

  /**
   * Calculate size of focused cache entries only
   */
  private calculateFocusedCacheSize(keys: string[]): number {
    try {
      const focusedEntries = keys.map(key => this.cache.get(key)).filter(Boolean)
      const serialized = JSON.stringify(focusedEntries)
      return new Blob([serialized]).size
    } catch {
      return 0
    }
  }

  /**
   * Export cache for debugging
   */
  export(): any {
    return {
      entries: Array.from(this.cache.entries()),
      stats: this.getStats(),
      focusedStats: this.getFocusedStats(),
      version: this.VERSION
    }
  }
}

// Export singleton instance
export const cacheManager = CacheManager.getInstance()

// Cache key generators for consistency
export const CacheKeys = {
  // AI Curations
  curations: () => 'ai_curations',
  curationContent: (id: string, provider: string, model: string) => 
    `curation_content_${id}_${provider}_${model}`,
  curationSettings: () => 'curation_settings',
  curationStatus: () => 'curation_status',
  deletedCurations: () => 'deleted_curations_list',

  // AI Summaries  
  summaries: () => 'ai_summaries',
  summaryContent: (id: string) => `summary_content_${id}`,
  summarySettings: () => 'summary_settings',
  summaryStatus: () => 'summary_status',

  // Documents
  documents: () => 'documents_list',
  documentContent: (id: string) => `document_${id}`,

  // User preferences
  userProvider: () => 'user_preferred_provider',
  userModel: () => 'user_preferred_model',

  // Analytics
  analytics: () => 'vault_analytics',
  
  // Patterns for bulk operations
  patterns: {
    allCurations: 'ai_curations|curation_.*',
    allSummaries: 'ai_summaries|summary_.*',
    allDocuments: 'documents_.*',
    userSettings: 'user_.*',
    allContent: 'curation_content_.*|summary_content_.*'
  }
} as const

export default cacheManager
