export interface AuthLoadingState {
  stage: 'idle' | 'initializing' | 'validating' | 'navigating' | 'complete' | 'error'
  message: string
  progress: number
  canRetry: boolean
  showAlternatives: boolean
  timeElapsed: number
  isStuck: boolean
}

export interface AuthLoadingConfig {
  maxLoadingTime: number
  maxTransitionTime: number
  retryDelay: number
  maxRetries: number
  networkTimeoutThreshold: number
}

export const DEFAULT_AUTH_CONFIG: AuthLoadingConfig = {
  maxLoadingTime: 25000, // 25 seconds - more reasonable for slow networks
  maxTransitionTime: 15000, // 15 seconds for navigation
  retryDelay: 2000, // 2 seconds between retries
  maxRetries: 3,
  networkTimeoutThreshold: 5000 // 5 seconds to detect slow network
}

export const AUTH_STAGE_MESSAGES = {
  idle: 'Ready',
  initializing: 'Initializing secure session...',
  validating: 'Validating your credentials...',
  navigating: 'Preparing your workspace...',
  complete: 'Authentication complete',
  error: 'Authentication failed'
} as const

export const AUTH_STAGE_PROGRESS = {
  idle: 0,
  initializing: 25,
  validating: 50,
  navigating: 75,
  complete: 100,
  error: 0
} as const

export class AuthLoadingManager {
  private startTime: number = 0
  private stage: AuthLoadingState['stage'] = 'idle'
  private config: AuthLoadingConfig
  private retryCount: number = 0
  private networkSpeed: 'fast' | 'slow' | 'unknown' = 'unknown'
  private sessionId: string = ''
  private isActive: boolean = false
  private timeoutId: NodeJS.Timeout | null = null

  constructor(config: Partial<AuthLoadingConfig> = {}) {
    this.config = { ...DEFAULT_AUTH_CONFIG, ...config }
    this.sessionId = this.generateSessionId()
  }

  private generateSessionId(): string {
    return `auth_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  }

  start(stage: AuthLoadingState['stage'] = 'initializing'): AuthLoadingState {
    // Clear any existing timeout
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }

    this.startTime = Date.now()
    this.stage = stage
    this.retryCount = 0
    this.isActive = true
    this.sessionId = this.generateSessionId()

    // Set automatic timeout to prevent infinite loading
    this.timeoutId = setTimeout(() => {
      if (this.isActive && this.stage !== 'complete' && this.stage !== 'error') {
        console.warn('Auth loading timeout reached, forcing error state')
        this.stage = 'error'
      }
    }, this.config.maxLoadingTime)

    return this.getCurrentState()
  }

  updateStage(stage: AuthLoadingState['stage']): AuthLoadingState {
    if (!this.isActive) {
      console.warn('Attempted to update inactive loading manager')
      return this.getCurrentState()
    }

    this.stage = stage

    // Clear timeout on completion or error
    if ((stage === 'complete' || stage === 'error') && this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
      this.isActive = false
    }

    return this.getCurrentState()
  }

  getCurrentState(): AuthLoadingState {
    const timeElapsed = Date.now() - this.startTime
    const isStuck = this.isOperationStuck(timeElapsed)
    
    return {
      stage: this.stage,
      message: this.getStageMessage(),
      progress: AUTH_STAGE_PROGRESS[this.stage],
      canRetry: this.canRetry(timeElapsed),
      showAlternatives: this.shouldShowAlternatives(timeElapsed),
      timeElapsed,
      isStuck
    }
  }

  private getStageMessage(): string {
    const baseMessage = AUTH_STAGE_MESSAGES[this.stage]
    
    // Add network-aware messaging
    if (this.networkSpeed === 'slow' && this.stage !== 'error' && this.stage !== 'complete') {
      return `${baseMessage} (Slow connection detected)`
    }
    
    return baseMessage
  }

  private isOperationStuck(timeElapsed: number): boolean {
    const threshold = this.stage === 'navigating' 
      ? this.config.maxTransitionTime 
      : this.config.maxLoadingTime
    
    return timeElapsed > threshold
  }

  private canRetry(timeElapsed: number): boolean {
    return this.retryCount < this.config.maxRetries && 
           timeElapsed > this.config.retryDelay &&
           this.stage === 'error'
  }

  private shouldShowAlternatives(timeElapsed: number): boolean {
    // Only show alternatives if we're truly stuck, not during normal operation
    return this.isOperationStuck(timeElapsed) && this.retryCount >= 2
  }

  incrementRetry(): void {
    this.retryCount++
  }

  detectNetworkSpeed(responseTime: number): void {
    if (responseTime > this.config.networkTimeoutThreshold) {
      this.networkSpeed = 'slow'
    } else {
      this.networkSpeed = 'fast'
    }
  }

  reset(): void {
    // Clear any existing timeout
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }

    this.startTime = 0
    this.stage = 'idle'
    this.retryCount = 0
    this.networkSpeed = 'unknown'
    this.isActive = false
    this.sessionId = this.generateSessionId()
  }

  // Get current session ID for debugging
  getSessionId(): string {
    return this.sessionId
  }

  // Check if manager is active
  isManagerActive(): boolean {
    return this.isActive
  }

  // Force complete the loading state
  forceComplete(): AuthLoadingState {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }
    this.stage = 'complete'
    this.isActive = false
    return this.getCurrentState()
  }

  // Force error state
  forceError(): AuthLoadingState {
    if (this.timeoutId) {
      clearTimeout(this.timeoutId)
      this.timeoutId = null
    }
    this.stage = 'error'
    this.isActive = false
    return this.getCurrentState()
  }
}
