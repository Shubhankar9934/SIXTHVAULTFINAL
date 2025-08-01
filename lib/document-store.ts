import { ragApiClient, type BackendDocument } from './api-client'

export interface DocumentData {
  id: string
  name: string
  size: number
  type: string
  uploadDate: string
  language: string
  content: string
  summary?: string
  themes: string[]
  keywords: string[]
  demographics?: string[]
  mainTopics?: string[]
  sentiment?: string
  readingLevel?: string
  keyInsights?: string[]
}

class DocumentStore {
  // Remove all localStorage - make everything backend-first
  private cache: DocumentData[] = []
  private cacheTimestamp: number = 0
  private readonly CACHE_TTL = 60000 // Increased to 60 seconds for ngrok stability
  private isLoading: boolean = false
  private loadPromise: Promise<DocumentData[]> | null = null
  
  // Real-time update system
  private updateListeners: Set<(documents: DocumentData[]) => void> = new Set()
  private activeWebSockets: Map<string, any> = new Map()

  /**
   * Convert backend document to frontend format
   */
  private convertBackendDocument(doc: BackendDocument): DocumentData {
    // Handle insights conversion - backend returns 'insights' as string, frontend expects 'keyInsights' as array
    let keyInsights: string[] = []
    if (doc.keyInsights && Array.isArray(doc.keyInsights)) {
      keyInsights = doc.keyInsights
    } else if (doc.insights && typeof doc.insights === 'string') {
      // Convert string insights to array by splitting on common delimiters
      keyInsights = doc.insights
        .split(/[•\n\r-]/)
        .map(insight => insight.trim())
        .filter(insight => insight.length > 0)
        .slice(0, 5) // Limit to 5 key insights
    }

    return {
      id: doc.id,
      name: doc.name,
      size: doc.size,
      type: doc.type,
      uploadDate: doc.uploadDate,
      language: doc.language,
      content: "", // Content not needed for most operations
      summary: doc.summary || "",
      themes: doc.themes || [],
      keywords: doc.keywords || [],
      demographics: doc.demographics || [],
      mainTopics: doc.mainTopics || doc.themes || [], // Use mainTopics from backend, fallback to themes
      sentiment: doc.sentiment || "neutral",
      readingLevel: doc.readingLevel || "intermediate",
      keyInsights: keyInsights
    }
  }

  /**
   * Check if cache is valid
   */
  private isCacheValid(): boolean {
    return Date.now() - this.cacheTimestamp < this.CACHE_TTL
  }

  /**
   * Load documents from backend with enhanced error handling
   */
  private async loadFromBackend(): Promise<DocumentData[]> {
    try {
      console.log('DocumentStore: Fetching documents from backend (ngrok-enhanced)')
      const backendDocs = await ragApiClient.getUserDocuments()
      
      // Convert and cache
      this.cache = backendDocs.map(doc => this.convertBackendDocument(doc))
      this.cacheTimestamp = Date.now()
      
      console.log('DocumentStore: Successfully cached', this.cache.length, 'documents')
      
      // Notify all listeners of the update
      this.notifyListeners()
      
      return this.cache
    } catch (error) {
      console.error('DocumentStore: Failed to get documents:', error)
      
      // Return cached data if available, even if stale (ngrok fallback)
      if (this.cache.length > 0) {
        console.log('DocumentStore: Returning stale cache due to error (ngrok fallback)')
        return this.cache
      }
      
      // If no cache available, throw the error for proper handling
      throw error
    }
  }

  /**
   * Notify all listeners of document updates
   */
  private notifyListeners(): void {
    this.updateListeners.forEach(listener => {
      try {
        listener([...this.cache])
      } catch (error) {
        console.error('DocumentStore: Error notifying listener:', error)
      }
    })
  }

  /**
   * Subscribe to real-time document updates
   */
  subscribeToUpdates(callback: (documents: DocumentData[]) => void): () => void {
    this.updateListeners.add(callback)
    
    // Immediately call with current data if available
    if (this.cache.length > 0) {
      callback([...this.cache])
    }
    
    // Return unsubscribe function
    return () => {
      this.updateListeners.delete(callback)
    }
  }

  /**
   * Connect to WebSocket for real-time updates during upload processing
   */
  connectToProcessingUpdates(batchId: string): () => void {
    console.log('DocumentStore: Connecting to processing updates for batch:', batchId)
    
    // Create reliable WebSocket connection
    const reliableWS = ragApiClient.createReliableWebSocket(batchId, (message: any) => {
      this.handleWebSocketMessage(message, batchId)
    })

    // Store the WebSocket for cleanup
    this.activeWebSockets.set(batchId, reliableWS)

    // Connect the WebSocket
    reliableWS.connect().then(() => {
      console.log('DocumentStore: WebSocket connected for batch:', batchId)
    }).catch((error) => {
      console.error('DocumentStore: Failed to connect WebSocket for batch:', batchId, error)
    })

    // Return cleanup function
    return () => {
      console.log('DocumentStore: Cleaning up WebSocket for batch:', batchId)
      reliableWS.disconnect()
      this.activeWebSockets.delete(batchId)
    }
  }

  /**
   * Handle WebSocket messages and update document state in real-time
   */
  private handleWebSocketMessage(message: any, batchId: string): void {
    try {
      console.log('DocumentStore: Processing WebSocket message:', message.type, message.data)

      switch (message.type) {
        case 'completed':
        case 'file_processing_completed':
          this.handleDocumentCompletion(message.data, batchId)
          break

        case 'processing':
          this.handleProcessingUpdate(message.data, batchId)
          break

        case 'error':
        case 'processing_error':
          this.handleProcessingError(message.data, batchId)
          break

        case 'batch_completed':
          this.handleBatchCompletion(message.data, batchId)
          break

        default:
          // Handle other message types if needed
          console.log('DocumentStore: Unhandled message type:', message.type)
          break
      }
    } catch (error) {
      console.error('DocumentStore: Error handling WebSocket message:', error)
    }
  }

  /**
   * Handle document completion with AI data
   */
  private handleDocumentCompletion(data: any, batchId: string): void {
    console.log('DocumentStore: Handling document completion:', data)

    // Extract filename from path or use provided filename
    const filename = this.extractFilename(data.file || data.filename || '')
    
    if (!filename) {
      console.warn('DocumentStore: No filename found in completion data')
      return
    }

    // Update cache with completed document data
    let documentUpdated = false
    this.cache = this.cache.map(doc => {
      if (this.matchesDocument(doc, filename, data)) {
        documentUpdated = true
        console.log('DocumentStore: Updating document with AI data:', doc.name)
        
        return {
          ...doc,
          id: data.doc_id || doc.id,
          summary: data.summary || doc.summary || '',
          themes: data.themes || doc.themes || [],
          keywords: data.keywords || data.themes || doc.keywords || [],
          demographics: data.demographics || doc.demographics || [],
          mainTopics: data.themes || doc.mainTopics || [],
          keyInsights: data.insights ? [data.insights] : (doc.keyInsights || []),
          language: data.language || doc.language || 'English',
          sentiment: data.sentiment || doc.sentiment || 'neutral',
          readingLevel: data.readingLevel || doc.readingLevel || 'intermediate'
        }
      }
      return doc
    })

    if (documentUpdated) {
      this.cacheTimestamp = Date.now()
      this.notifyListeners()
      console.log('DocumentStore: Document updated and listeners notified')
    } else {
      // Document not found in cache, refresh from backend
      console.log('DocumentStore: Document not found in cache, refreshing from backend')
      this.refreshDocuments()
    }
  }

  /**
   * Handle processing updates
   */
  private handleProcessingUpdate(data: any, batchId: string): void {
    // For now, we don't update cache during processing
    // This could be extended to show processing status if needed
    console.log('DocumentStore: Processing update received:', data)
  }

  /**
   * Handle processing errors
   */
  private handleProcessingError(data: any, batchId: string): void {
    console.error('DocumentStore: Processing error received:', data)
    // Could update document status to error if needed
  }

  /**
   * Handle batch completion
   */
  private handleBatchCompletion(data: any, batchId: string): void {
    console.log('DocumentStore: Batch completion received:', data)
    
    // CRITICAL FIX: Immediately refresh documents to show completed processing
    this.refreshDocuments().then(() => {
      console.log('DocumentStore: Documents refreshed after batch completion')
    }).catch((error) => {
      console.error('DocumentStore: Failed to refresh documents after batch completion:', error)
    })
    
    // Clean up WebSocket
    const ws = this.activeWebSockets.get(batchId)
    if (ws) {
      ws.disconnect()
      this.activeWebSockets.delete(batchId)
    }
  }

  /**
   * Extract filename from path
   */
  private extractFilename(filePath: string): string {
    if (!filePath) return ''
    
    // Remove UUID prefix if present (32 hex chars + underscore)
    let filename = filePath.split('/').pop() || filePath.split('\\').pop() || filePath
    
    if (filename.length > 33 && filename[32] === '_') {
      filename = filename.substring(33)
    }
    
    return filename
  }

  /**
   * Check if a document matches the completion data
   */
  private matchesDocument(doc: DocumentData, filename: string, data: any): boolean {
    // Try multiple matching strategies
    return (
      doc.name === filename ||
      doc.name.includes(filename) ||
      filename.includes(doc.name) ||
      (data.filename && doc.name === data.filename) ||
      (data.doc_id && doc.id === data.doc_id)
    )
  }

  /**
   * Add a new document to cache (for upload scenarios)
   */
  addDocumentToCache(document: DocumentData): void {
    // Check if document already exists
    const existingIndex = this.cache.findIndex(doc => doc.id === document.id || doc.name === document.name)
    
    if (existingIndex >= 0) {
      // Update existing document
      this.cache[existingIndex] = document
    } else {
      // Add new document
      this.cache.push(document)
    }
    
    this.cacheTimestamp = Date.now()
    this.notifyListeners()
  }

  /**
   * Update a document in cache
   */
  updateDocumentInCache(documentId: string, updates: Partial<DocumentData>): void {
    const index = this.cache.findIndex(doc => doc.id === documentId)
    
    if (index >= 0) {
      this.cache[index] = { ...this.cache[index], ...updates }
      this.cacheTimestamp = Date.now()
      this.notifyListeners()
    }
  }

  /**
   * Get all documents from backend (with enhanced caching and race condition prevention)
   */
  async getDocuments(): Promise<DocumentData[]> {
    try {
      // If already loading, return the same promise to prevent race conditions
      if (this.isLoading && this.loadPromise) {
        console.log('DocumentStore: Returning existing load promise (race condition prevention)')
        return this.loadPromise
      }

      // Use cache if valid and not empty
      if (this.isCacheValid() && this.cache.length > 0) {
        console.log('DocumentStore: Using cached documents (valid cache)')
        return this.cache
      }

      // Start loading with race condition protection
      this.isLoading = true
      this.loadPromise = this.loadFromBackend()

      try {
        const result = await this.loadPromise
        return result
      } finally {
        this.isLoading = false
        this.loadPromise = null
      }
    } catch (error) {
      console.error('DocumentStore: Critical error in getDocuments:', error)
      
      // Reset loading state
      this.isLoading = false
      this.loadPromise = null
      
      // Return empty array as fallback
      return []
    }
  }

  /**
   * Get documents by IDs
   */
  async getDocumentsByIds(ids: string[]): Promise<DocumentData[]> {
    const documents = await this.getDocuments()
    return documents.filter(doc => ids.includes(doc.id))
  }

  /**
   * Get documents by tags/themes
   */
  async getDocumentsByTags(tags: string[]): Promise<DocumentData[]> {
    if (tags.length === 0) {
      return await this.getDocuments()
    }
    
    const documents = await this.getDocuments()
    return documents.filter(doc =>
      doc.themes.some(theme => tags.includes(theme)) ||
      doc.keywords.some(keyword => tags.includes(keyword)) ||
      (doc.demographics && doc.demographics.some(demo => tags.includes(demo))) ||
      (doc.mainTopics && doc.mainTopics.some(topic => tags.includes(topic)))
    )
  }

  /**
   * Search documents by query
   */
  async searchDocuments(query: string): Promise<DocumentData[]> {
    const documents = await this.getDocuments()
    const lowerQuery = query.toLowerCase()
    
    return documents.filter(doc =>
      doc.name.toLowerCase().includes(lowerQuery) ||
      doc.themes.some(theme => theme.toLowerCase().includes(lowerQuery)) ||
      doc.keywords.some(keyword => keyword.toLowerCase().includes(lowerQuery)) ||
      (doc.demographics && doc.demographics.some(demo => demo.toLowerCase().includes(lowerQuery))) ||
      (doc.mainTopics && doc.mainTopics.some(topic => topic.toLowerCase().includes(lowerQuery)))
    )
  }

  /**
   * Search documents by language
   */
  async searchDocumentsByLanguage(language: string): Promise<DocumentData[]> {
    const documents = await this.getDocuments()
    return documents.filter(doc => doc.language.toLowerCase() === language.toLowerCase())
  }

  /**
   * Delete a document
   */
  async deleteDocument(id: string): Promise<boolean> {
    try {
      const success = await ragApiClient.deleteDocument(id)
      if (success) {
        // Remove from cache
        this.cache = this.cache.filter(doc => doc.id !== id)
        this.notifyListeners()
        console.log('DocumentStore: Document deleted and removed from cache')
      }
      return success
    } catch (error) {
      console.error('DocumentStore: Failed to delete document:', error)
      return false
    }
  }

  /**
   * Refresh cache - force reload from backend
   */
  async refreshDocuments(): Promise<DocumentData[]> {
    this.cache = []
    this.cacheTimestamp = 0
    return await this.getDocuments()
  }

  /**
   * Clear cache
   */
  clearCache(): void {
    this.cache = []
    this.cacheTimestamp = 0
    this.notifyListeners()
    console.log('DocumentStore: Cache cleared')
  }

  /**
   * Get cache status
   */
  getCacheStatus(): { size: number; age: number; valid: boolean } {
    return {
      size: this.cache.length,
      age: Date.now() - this.cacheTimestamp,
      valid: this.isCacheValid()
    }
  }

  /**
   * Cleanup all WebSocket connections
   */
  cleanup(): void {
    console.log('DocumentStore: Cleaning up all WebSocket connections')
    this.activeWebSockets.forEach((ws, batchId) => {
      console.log('DocumentStore: Disconnecting WebSocket for batch:', batchId)
      ws.disconnect()
    })
    this.activeWebSockets.clear()
    this.updateListeners.clear()
  }

  // REMOVED METHODS (no longer needed):
  // - addDocument() - handled by upload API
  // - removeDocument() - use deleteDocument()
  // - updateDocument() - handled by backend
  // - clearDocuments() - use clearCache()
  // - saveToStorage() - NO LOCAL STORAGE
  // - loadFromStorage() - NO LOCAL STORAGE
}

export const documentStore = new DocumentStore()
