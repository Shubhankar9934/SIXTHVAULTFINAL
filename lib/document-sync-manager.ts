/**
 * Document Sync Manager - Real-time document synchronization across pages
 * Ensures immediate document updates in vault page when uploads complete on documents page
 */

export interface DocumentSyncEvent {
  type: 'upload_completed' | 'document_deleted' | 'batch_completed' | 'processing_update'
  data: {
    documentId?: string
    documentName?: string
    batchId?: string
    documentCount?: number
    timestamp: number
    source: 'documents_page' | 'vault_page' | 'background'
  }
}

export interface DocumentSyncListener {
  id: string
  callback: (event: DocumentSyncEvent) => void
  source: 'vault' | 'documents' | 'background'
}

class DocumentSyncManager {
  private static instance: DocumentSyncManager
  private listeners: Map<string, DocumentSyncListener> = new Map()
  private eventQueue: DocumentSyncEvent[] = []
  private isProcessingQueue = false
  
  // Storage keys for cross-tab communication
  private readonly UPLOAD_EVENT_KEY = 'sixthvault_document_upload_event'
  private readonly DELETE_EVENT_KEY = 'sixthvault_document_delete_event'
  private readonly CACHE_INVALIDATE_KEY = 'sixthvault_cache_invalidate_documents'
  private readonly SYNC_EVENT_KEY = 'sixthvault_document_sync_event'
  
  private constructor() {
    if (typeof window !== 'undefined') {
      this.setupStorageListener()
      this.startEventProcessor()
    }
  }

  static getInstance(): DocumentSyncManager {
    if (!DocumentSyncManager.instance) {
      DocumentSyncManager.instance = new DocumentSyncManager()
    }
    return DocumentSyncManager.instance
  }

  /**
   * Register a listener for document sync events
   */
  addListener(
    id: string, 
    callback: (event: DocumentSyncEvent) => void, 
    source: 'vault' | 'documents' | 'background' = 'vault'
  ): () => void {
    const listener: DocumentSyncListener = { id, callback, source }
    this.listeners.set(id, listener)
    
    console.log(`📡 DocumentSync: Registered listener '${id}' from ${source}`)
    
    // Return unsubscribe function
    return () => {
      this.listeners.delete(id)
      console.log(`📡 DocumentSync: Unregistered listener '${id}'`)
    }
  }

  /**
   * Emit a document sync event
   */
  emit(event: DocumentSyncEvent): void {
    console.log(`📡 DocumentSync: Emitting event '${event.type}' from ${event.data.source}`)
    
    // Add to queue for processing
    this.eventQueue.push(event)
    
    // Trigger cross-tab sync
    this.triggerCrossTabSync(event)
    
    // Process queue
    this.processEventQueue()
  }

  /**
   * Emit upload completion event
   */
  emitUploadCompleted(documentId: string, documentName: string, source: DocumentSyncEvent['data']['source']): void {
    this.emit({
      type: 'upload_completed',
      data: {
        documentId,
        documentName,
        timestamp: Date.now(),
        source
      }
    })
  }

  /**
   * Emit batch completion event
   */
  emitBatchCompleted(batchId: string, documentCount: number, source: DocumentSyncEvent['data']['source']): void {
    this.emit({
      type: 'batch_completed',
      data: {
        batchId,
        documentCount,
        timestamp: Date.now(),
        source
      }
    })
  }

  /**
   * Emit document deletion event
   */
  emitDocumentDeleted(documentId: string, documentName: string, source: DocumentSyncEvent['data']['source']): void {
    this.emit({
      type: 'document_deleted',
      data: {
        documentId,
        documentName,
        timestamp: Date.now(),
        source
      }
    })
  }

  /**
   * Trigger immediate vault refresh (for critical updates)
   */
  triggerImmediateVaultRefresh(): void {
    console.log('🚀 DocumentSync: Triggering immediate vault refresh')
    
    // Emit to all vault listeners immediately
    const vaultListeners = Array.from(this.listeners.values()).filter(l => l.source === 'vault')
    
    const immediateEvent: DocumentSyncEvent = {
      type: 'processing_update',
      data: {
        timestamp: Date.now(),
        source: 'background'
      }
    }
    
    vaultListeners.forEach(listener => {
      try {
        listener.callback(immediateEvent)
      } catch (error) {
        console.error(`📡 DocumentSync: Error in immediate refresh for listener '${listener.id}':`, error)
      }
    })
    
    // Also trigger cross-tab sync
    this.triggerCrossTabSync(immediateEvent)
  }

  /**
   * Setup storage event listener for cross-tab communication
   */
  private setupStorageListener(): void {
    window.addEventListener('storage', (event) => {
      if (event.key === this.SYNC_EVENT_KEY && event.newValue) {
        try {
          const syncEvent: DocumentSyncEvent = JSON.parse(event.newValue)
          console.log(`📡 DocumentSync: Received cross-tab event '${syncEvent.type}'`)
          
          // Add to queue for processing
          this.eventQueue.push(syncEvent)
          this.processEventQueue()
          
        } catch (error) {
          console.error('📡 DocumentSync: Failed to parse cross-tab sync event:', error)
        }
      }
      
      // Handle legacy events for backward compatibility
      if (event.key === this.UPLOAD_EVENT_KEY && event.newValue) {
        try {
          const eventData = JSON.parse(event.newValue)
          this.emit({
            type: 'upload_completed',
            data: {
              documentId: eventData.documentId,
              documentName: eventData.documentName,
              timestamp: eventData.timestamp || Date.now(),
              source: 'documents_page'
            }
          })
        } catch (error) {
          console.error('📡 DocumentSync: Failed to parse legacy upload event:', error)
        }
      }
      
      if (event.key === this.DELETE_EVENT_KEY && event.newValue) {
        try {
          const eventData = JSON.parse(event.newValue)
          this.emit({
            type: 'document_deleted',
            data: {
              documentId: eventData.documentId,
              documentName: eventData.documentName,
              timestamp: eventData.timestamp || Date.now(),
              source: 'documents_page'
            }
          })
        } catch (error) {
          console.error('📡 DocumentSync: Failed to parse legacy delete event:', error)
        }
      }
      
      if (event.key === this.CACHE_INVALIDATE_KEY) {
        this.emit({
          type: 'processing_update',
          data: {
            timestamp: Date.now(),
            source: 'background'
          }
        })
      }
    })
  }

  /**
   * Trigger cross-tab synchronization
   */
  private triggerCrossTabSync(event: DocumentSyncEvent): void {
    try {
      // Use the new unified sync event
      localStorage.setItem(this.SYNC_EVENT_KEY, JSON.stringify(event))
      
      // Remove after a short delay to trigger the storage event
      setTimeout(() => {
        localStorage.removeItem(this.SYNC_EVENT_KEY)
      }, 100)
      
      // Also trigger legacy events for backward compatibility
      if (event.type === 'upload_completed' || event.type === 'batch_completed') {
        localStorage.setItem(this.UPLOAD_EVENT_KEY, JSON.stringify(event.data))
        setTimeout(() => {
          localStorage.removeItem(this.UPLOAD_EVENT_KEY)
        }, 100)
      }
      
      if (event.type === 'document_deleted') {
        localStorage.setItem(this.DELETE_EVENT_KEY, JSON.stringify(event.data))
        setTimeout(() => {
          localStorage.removeItem(this.DELETE_EVENT_KEY)
        }, 100)
      }
      
      // Always trigger cache invalidation for immediate updates
      localStorage.setItem(this.CACHE_INVALIDATE_KEY, Date.now().toString())
      setTimeout(() => {
        localStorage.removeItem(this.CACHE_INVALIDATE_KEY)
      }, 100)
      
      console.log(`📡 DocumentSync: Triggered cross-tab sync for '${event.type}'`)
      
    } catch (error) {
      console.error('📡 DocumentSync: Failed to trigger cross-tab sync:', error)
    }
  }

  /**
   * Process the event queue
   */
  private processEventQueue(): void {
    if (this.isProcessingQueue || this.eventQueue.length === 0) {
      return
    }
    
    this.isProcessingQueue = true
    
    try {
      // Process all events in the queue
      const eventsToProcess = [...this.eventQueue]
      this.eventQueue = []
      
      eventsToProcess.forEach(event => {
        this.notifyListeners(event)
      })
      
    } catch (error) {
      console.error('📡 DocumentSync: Error processing event queue:', error)
    } finally {
      this.isProcessingQueue = false
    }
  }

  /**
   * Notify all listeners of an event
   */
  private notifyListeners(event: DocumentSyncEvent): void {
    const relevantListeners = Array.from(this.listeners.values())
    
    console.log(`📡 DocumentSync: Notifying ${relevantListeners.length} listeners of '${event.type}' event`)
    
    relevantListeners.forEach(listener => {
      try {
        listener.callback(event)
      } catch (error) {
        console.error(`📡 DocumentSync: Error notifying listener '${listener.id}':`, error)
      }
    })
  }

  /**
   * Start the event processor
   */
  private startEventProcessor(): void {
    // Process events every 100ms for responsive updates
    setInterval(() => {
      if (this.eventQueue.length > 0) {
        this.processEventQueue()
      }
    }, 100)
  }

  /**
   * Get sync statistics
   */
  getStats(): {
    activeListeners: number
    queuedEvents: number
    listenersBySource: Record<string, number>
  } {
    const listenersBySource: Record<string, number> = {}
    
    this.listeners.forEach(listener => {
      listenersBySource[listener.source] = (listenersBySource[listener.source] || 0) + 1
    })
    
    return {
      activeListeners: this.listeners.size,
      queuedEvents: this.eventQueue.length,
      listenersBySource
    }
  }

  /**
   * Clear all listeners and events
   */
  cleanup(): void {
    console.log('📡 DocumentSync: Cleaning up all listeners and events')
    this.listeners.clear()
    this.eventQueue = []
    this.isProcessingQueue = false
  }
}

export const documentSyncManager = DocumentSyncManager.getInstance()
export default documentSyncManager
