"use client"

import React from 'react'

// Upload State Manager - Global state for tracking document uploads
// Prevents navigation during active uploads to ensure data integrity

interface UploadDocument {
  id: string
  name: string
  status: 'uploading' | 'processing' | 'completed' | 'error' | 'waiting'
  progress: number
  batchId?: string
}

interface UploadState {
  isUploading: boolean
  documents: UploadDocument[]
  totalFiles: number
  completedFiles: number
  hasActiveUploads: boolean
  uploadStartTime: number | null
  lastActivity: number
}

type UploadStateListener = (state: UploadState) => void

class UploadStateManager {
  private state: UploadState = {
    isUploading: false,
    documents: [],
    totalFiles: 0,
    completedFiles: 0,
    hasActiveUploads: false,
    uploadStartTime: null,
    lastActivity: Date.now()
  }

  private listeners: Set<UploadStateListener> = new Set()
  private persistenceKey = 'sixthvault_upload_state'
  private activityTimeout: NodeJS.Timeout | null = null

  constructor() {
    // Load persisted state on initialization
    this.loadPersistedState()
    
    // Set up activity monitoring
    this.startActivityMonitoring()
    
    // Listen for cross-tab upload events
    if (typeof window !== 'undefined') {
      window.addEventListener('storage', this.handleStorageEvent.bind(this))
      window.addEventListener('beforeunload', this.handleBeforeUnload.bind(this))
    }
  }

  private loadPersistedState() {
    if (typeof window === 'undefined') return
    
    try {
      const persistedState = localStorage.getItem(this.persistenceKey)
      if (persistedState) {
        const parsed = JSON.parse(persistedState)
        
        // Only restore if the state is recent (within 30 minutes)
        const stateAge = Date.now() - (parsed.lastActivity || 0)
        if (stateAge < 30 * 60 * 1000) {
          this.state = { ...this.state, ...parsed }
          console.log('📁 UploadStateManager: Restored persisted upload state:', this.state)
        } else {
          console.log('📁 UploadStateManager: Persisted state too old, starting fresh')
          this.clearPersistedState()
        }
      }
    } catch (error) {
      console.error('📁 UploadStateManager: Failed to load persisted state:', error)
      this.clearPersistedState()
    }
  }

  private persistState() {
    if (typeof window === 'undefined') return
    
    try {
      localStorage.setItem(this.persistenceKey, JSON.stringify(this.state))
    } catch (error) {
      console.error('📁 UploadStateManager: Failed to persist state:', error)
    }
  }

  private clearPersistedState() {
    if (typeof window === 'undefined') return
    
    try {
      localStorage.removeItem(this.persistenceKey)
    } catch (error) {
      console.error('📁 UploadStateManager: Failed to clear persisted state:', error)
    }
  }

  private startActivityMonitoring() {
    // Clear existing timeout
    if (this.activityTimeout) {
      clearTimeout(this.activityTimeout)
    }

    // Set up activity timeout (5 minutes of inactivity)
    this.activityTimeout = setTimeout(() => {
      if (this.state.hasActiveUploads) {
        console.log('📁 UploadStateManager: Activity timeout reached, clearing stale upload state')
        this.clearUploadState()
      }
    }, 5 * 60 * 1000)
  }

  private handleStorageEvent(event: StorageEvent) {
    if (event.key === this.persistenceKey && event.newValue) {
      try {
        const newState = JSON.parse(event.newValue)
        this.state = { ...this.state, ...newState }
        this.notifyListeners()
        console.log('📁 UploadStateManager: Received cross-tab state update')
      } catch (error) {
        console.error('📁 UploadStateManager: Failed to parse cross-tab state:', error)
      }
    }
  }

  private handleBeforeUnload = (event: BeforeUnloadEvent) => {
    if (this.state.hasActiveUploads) {
      const message = 'Documents are currently uploading. Are you sure you want to leave?'
      event.preventDefault()
      event.returnValue = message
      return message
    }
  }

  private notifyListeners() {
    this.listeners.forEach(listener => {
      try {
        listener(this.state)
      } catch (error) {
        console.error('📁 UploadStateManager: Listener error:', error)
      }
    })
  }

  private updateState(updates: Partial<UploadState>) {
    this.state = {
      ...this.state,
      ...updates,
      lastActivity: Date.now()
    }
    
    // Update hasActiveUploads based on document states
    const activeUploads = this.state.documents.filter(doc => 
      doc.status === 'uploading' || doc.status === 'processing' || doc.status === 'waiting'
    )
    
    this.state.hasActiveUploads = activeUploads.length > 0
    this.state.isUploading = this.state.hasActiveUploads
    
    // Persist state and notify listeners
    this.persistState()
    this.notifyListeners()
    
    // Reset activity monitoring
    this.startActivityMonitoring()
    
    console.log('📁 UploadStateManager: State updated:', {
      hasActiveUploads: this.state.hasActiveUploads,
      totalFiles: this.state.totalFiles,
      completedFiles: this.state.completedFiles,
      documentsCount: this.state.documents.length
    })
  }

  // Public methods
  startUpload(files: File[]) {
    console.log('📁 UploadStateManager: Starting upload for', files.length, 'files')
    
    const uploadDocuments: UploadDocument[] = files.map((file, index) => ({
      id: `upload_${Date.now()}_${index}_${Math.random().toString(36).substr(2, 9)}`,
      name: file.name,
      status: index === 0 ? 'uploading' : 'waiting',
      progress: 0
    }))

    this.updateState({
      isUploading: true,
      documents: [...this.state.documents, ...uploadDocuments],
      totalFiles: this.state.totalFiles + files.length,
      hasActiveUploads: true,
      uploadStartTime: this.state.uploadStartTime || Date.now()
    })

    return uploadDocuments.map(doc => doc.id)
  }

  updateDocumentProgress(documentId: string, progress: number, status?: UploadDocument['status'], batchId?: string) {
    const updatedDocuments = this.state.documents.map(doc => {
      if (doc.id === documentId) {
        return {
          ...doc,
          progress: Math.max(doc.progress, progress), // Ensure progress never decreases
          status: status || doc.status,
          batchId: batchId || doc.batchId
        }
      }
      return doc
    })

    // Count completed files
    const completedFiles = updatedDocuments.filter(doc => 
      doc.status === 'completed' || doc.status === 'error'
    ).length

    this.updateState({
      documents: updatedDocuments,
      completedFiles
    })
  }

  completeDocument(documentId: string) {
    console.log('📁 UploadStateManager: Completing document:', documentId)
    
    const updatedDocuments = this.state.documents.map(doc => {
      if (doc.id === documentId) {
        return {
          ...doc,
          status: 'completed' as const,
          progress: 100
        }
      }
      return doc
    })

    const completedFiles = updatedDocuments.filter(doc => 
      doc.status === 'completed' || doc.status === 'error'
    ).length

    this.updateState({
      documents: updatedDocuments,
      completedFiles
    })

    // Check if all uploads are complete
    if (completedFiles >= this.state.totalFiles) {
      console.log('📁 UploadStateManager: All uploads completed')
      setTimeout(() => {
        this.clearUploadState()
      }, 2000) // Clear state after 2 seconds
    }
  }

  errorDocument(documentId: string, error: string) {
    console.log('📁 UploadStateManager: Document error:', documentId, error)
    
    const updatedDocuments = this.state.documents.map(doc => {
      if (doc.id === documentId) {
        return {
          ...doc,
          status: 'error' as const,
          progress: 0
        }
      }
      return doc
    })

    const completedFiles = updatedDocuments.filter(doc => 
      doc.status === 'completed' || doc.status === 'error'
    ).length

    this.updateState({
      documents: updatedDocuments,
      completedFiles
    })
  }

  clearUploadState() {
    console.log('📁 UploadStateManager: Clearing upload state')
    
    this.updateState({
      isUploading: false,
      documents: [],
      totalFiles: 0,
      completedFiles: 0,
      hasActiveUploads: false,
      uploadStartTime: null
    })

    this.clearPersistedState()
  }

  // Force clear for emergency situations
  forceReset() {
    console.log('📁 UploadStateManager: Force resetting all upload state')
    
    if (this.activityTimeout) {
      clearTimeout(this.activityTimeout)
      this.activityTimeout = null
    }

    this.state = {
      isUploading: false,
      documents: [],
      totalFiles: 0,
      completedFiles: 0,
      hasActiveUploads: false,
      uploadStartTime: null,
      lastActivity: Date.now()
    }

    this.clearPersistedState()
    this.notifyListeners()
  }

  // Getters
  getCurrentState(): UploadState {
    return { ...this.state }
  }

  hasActiveUploads(): boolean {
    return this.state.hasActiveUploads
  }

  getUploadProgress(): { completed: number; total: number; percentage: number } {
    const completed = this.state.completedFiles
    const total = this.state.totalFiles
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0
    
    return { completed, total, percentage }
  }

  getActiveDocuments(): UploadDocument[] {
    return this.state.documents.filter(doc => 
      doc.status === 'uploading' || doc.status === 'processing' || doc.status === 'waiting'
    )
  }

  // Subscription methods
  subscribe(listener: UploadStateListener): () => void {
    this.listeners.add(listener)
    
    // Immediately call listener with current state
    listener(this.state)
    
    return () => {
      this.listeners.delete(listener)
    }
  }

  // Navigation blocking logic
  shouldBlockNavigation(): { blocked: boolean; reason?: string; details?: any } {
    if (!this.state.hasActiveUploads) {
      return { blocked: false }
    }

    const activeDocuments = this.getActiveDocuments()
    const uploadProgress = this.getUploadProgress()
    
    return {
      blocked: true,
      reason: 'Documents are currently uploading',
      details: {
        activeDocuments: activeDocuments.length,
        totalProgress: uploadProgress.percentage,
        documentsInProgress: activeDocuments.map(doc => ({
          name: doc.name,
          status: doc.status,
          progress: doc.progress
        }))
      }
    }
  }

  // Get user-friendly status message
  getStatusMessage(): string {
    if (!this.state.hasActiveUploads) {
      return 'No active uploads'
    }

    const activeDocuments = this.getActiveDocuments()
    const uploadProgress = this.getUploadProgress()
    
    if (activeDocuments.length === 1) {
      const doc = activeDocuments[0]
      return `Uploading "${doc.name}" (${doc.progress}%)`
    } else {
      return `Uploading ${activeDocuments.length} documents (${uploadProgress.percentage}% overall)`
    }
  }
}

// Create singleton instance
export const uploadStateManager = new UploadStateManager()

// React hook for components
export function useUploadState() {
  const [state, setState] = React.useState<UploadState>(uploadStateManager.getCurrentState())

  React.useEffect(() => {
    const unsubscribe = uploadStateManager.subscribe(setState)
    return unsubscribe
  }, [])

  return {
    ...state,
    shouldBlockNavigation: uploadStateManager.shouldBlockNavigation(),
    getStatusMessage: uploadStateManager.getStatusMessage(),
    getUploadProgress: () => uploadStateManager.getUploadProgress(),
    getActiveDocuments: () => uploadStateManager.getActiveDocuments(),
    clearUploadState: () => uploadStateManager.clearUploadState(),
    forceReset: () => uploadStateManager.forceReset()
  }
}

// Export types for use in other components
export type { UploadState, UploadDocument }
