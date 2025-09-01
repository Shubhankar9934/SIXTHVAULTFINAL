"use client"

import { Brain, FileText, MessageSquare, Settings, Sparkles, Database } from 'lucide-react'

export interface LoadingStep {
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  completed: boolean
  inProgress: boolean
  description?: string
}

export interface LoadingState {
  isVisible: boolean
  currentStep: string
  progress: number
  steps: LoadingStep[]
  message?: string
}

export class VaultLoadingManager {
  private state: LoadingState
  private listeners: ((state: LoadingState) => void)[] = []
  private stepTimeouts: Map<string, NodeJS.Timeout> = new Map()

  constructor() {
    this.state = {
      isVisible: false,
      currentStep: '',
      progress: 0,
      steps: [
        {
          id: 'auth',
          label: 'Authenticating User',
          icon: Settings,
          completed: false,
          inProgress: false,
          description: 'Verifying credentials and permissions'
        },
        {
          id: 'documents',
          label: 'Loading Documents',
          icon: FileText,
          completed: false,
          inProgress: false,
          description: 'Fetching your document library'
        },
        {
          id: 'curations',
          label: 'Preparing AI Curations',
          icon: Sparkles,
          completed: false,
          inProgress: false,
          description: 'Setting up intelligent content analysis'
        },
        {
          id: 'conversations',
          label: 'Loading Chat History',
          icon: MessageSquare,
          completed: false,
          inProgress: false,
          description: 'Retrieving previous conversations'
        },
        {
          id: 'providers',
          label: 'Configuring AI Models',
          icon: Brain,
          completed: false,
          inProgress: false,
          description: 'Connecting to AI providers'
        },
        {
          id: 'finalize',
          label: 'Finalizing Setup',
          icon: Database,
          completed: false,
          inProgress: false,
          description: 'Completing workspace initialization'
        }
      ],
      message: 'Preparing your secure workspace...'
    }
  }

  subscribe(listener: (state: LoadingState) => void) {
    this.listeners.push(listener)
    return () => {
      this.listeners = this.listeners.filter(l => l !== listener)
    }
  }

  private notify() {
    this.listeners.forEach(listener => listener({ ...this.state }))
  }

  show() {
    this.state.isVisible = true
    this.state.progress = 0
    this.state.currentStep = 'auth'
    this.state.message = 'Preparing your secure workspace...'
    
    // Reset all steps
    this.state.steps = this.state.steps.map(step => ({
      ...step,
      completed: false,
      inProgress: false
    }))
    
    this.notify()
  }

  hide() {
    this.state.isVisible = false
    this.clearAllTimeouts()
    this.notify()
  }

  startStep(stepId: string, message?: string) {
    const stepIndex = this.state.steps.findIndex(s => s.id === stepId)
    if (stepIndex === -1) return

    // Complete previous steps
    this.state.steps = this.state.steps.map((step, index) => ({
      ...step,
      completed: index < stepIndex,
      inProgress: index === stepIndex
    }))

    this.state.currentStep = stepId
    this.state.progress = (stepIndex / this.state.steps.length) * 100
    
    if (message) {
      this.state.message = message
    }

    this.notify()

    // Auto-complete step after timeout if not manually completed
    const timeout = setTimeout(() => {
      this.completeStep(stepId)
    }, 10000) // 10 second timeout

    this.stepTimeouts.set(stepId, timeout)
  }

  completeStep(stepId: string, message?: string) {
    const stepIndex = this.state.steps.findIndex(s => s.id === stepId)
    if (stepIndex === -1) return

    // Clear timeout for this step
    const timeout = this.stepTimeouts.get(stepId)
    if (timeout) {
      clearTimeout(timeout)
      this.stepTimeouts.delete(stepId)
    }

    // Mark step as completed
    this.state.steps = this.state.steps.map((step, index) => ({
      ...step,
      completed: index <= stepIndex,
      inProgress: false
    }))

    this.state.progress = ((stepIndex + 1) / this.state.steps.length) * 100

    if (message) {
      this.state.message = message
    }

    this.notify()

    // If this is the last step, hide after a brief delay
    if (stepIndex === this.state.steps.length - 1) {
      setTimeout(() => {
        this.hide()
      }, 1500)
    }
  }

  updateMessage(message: string) {
    this.state.message = message
    this.notify()
  }

  setProgress(progress: number) {
    this.state.progress = Math.max(0, Math.min(100, progress))
    this.notify()
  }

  private clearAllTimeouts() {
    this.stepTimeouts.forEach(timeout => clearTimeout(timeout))
    this.stepTimeouts.clear()
  }

  // Convenience methods for common loading scenarios
  startAuthentication() {
    this.show()
    this.startStep('auth', 'Verifying your credentials...')
  }

  completeAuthentication() {
    this.completeStep('auth', 'Authentication successful!')
  }

  startDocumentLoading() {
    this.startStep('documents', 'Loading your document library...')
  }

  completeDocumentLoading(count: number) {
    this.completeStep('documents', `Loaded ${count} documents successfully`)
  }

  startCurationLoading() {
    this.startStep('curations', 'Preparing AI-powered content analysis...')
  }

  completeCurationLoading(count: number) {
    this.completeStep('curations', `${count} AI curations ready`)
  }

  startConversationLoading() {
    this.startStep('conversations', 'Loading your chat history...')
  }

  completeConversationLoading(count: number) {
    this.completeStep('conversations', `${count} conversations loaded`)
  }

  startProviderLoading() {
    this.startStep('providers', 'Connecting to AI models...')
  }

  completeProviderLoading(count: number) {
    this.completeStep('providers', `${count} AI providers configured`)
  }

  startFinalization() {
    this.startStep('finalize', 'Finalizing your workspace...')
  }

  completeFinalization() {
    this.completeStep('finalize', 'Workspace ready! Welcome to SixthVault.')
  }

  // Error handling
  showError(stepId: string, error: string) {
    const stepIndex = this.state.steps.findIndex(s => s.id === stepId)
    if (stepIndex !== -1) {
      this.state.steps[stepIndex].inProgress = false
      this.state.message = `Error: ${error}`
      this.notify()
    }
  }

  getCurrentState(): LoadingState {
    return { ...this.state }
  }
}

// Singleton instance
export const vaultLoadingManager = new VaultLoadingManager()
