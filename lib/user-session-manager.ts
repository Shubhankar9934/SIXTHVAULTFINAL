/**
 * User Session Manager - Handles complete session isolation between different accounts
 * Ensures no cache leakage between user accounts when switching in the same tab
 */

export interface UserSession {
  userId: string
  email: string
  sessionId: string
  timestamp: number
}

export interface SessionIsolationConfig {
  clearAllCaches: boolean
  clearLocalStorage: boolean
  clearSessionStorage: boolean
  clearIndexedDB: boolean
  clearCookies: boolean
}

class UserSessionManager {
  private static instance: UserSessionManager
  private currentSession: UserSession | null = null
  private readonly SESSION_KEY = 'sixthvault_current_session'
  private readonly USER_CACHE_PREFIX = 'user_'
  
  private constructor() {}

  static getInstance(): UserSessionManager {
    if (!UserSessionManager.instance) {
      UserSessionManager.instance = new UserSessionManager()
    }
    return UserSessionManager.instance
  }

  /**
   * Initialize a new user session with complete isolation
   */
  initializeSession(userId: string, email: string, preserveAuthToken: boolean = false): UserSession {
    console.log('🔐 SessionManager: Initializing new user session for:', email)
    
    // Clear any existing session data first, but preserve auth token if requested
    this.clearCurrentSession({
      clearAllCaches: true,
      clearLocalStorage: true,
      clearSessionStorage: true,
      clearIndexedDB: true,
      clearCookies: !preserveAuthToken // Don't clear cookies if we want to preserve the auth token
    })
    
    // Create new session
    const newSession: UserSession = {
      userId,
      email,
      sessionId: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      timestamp: Date.now()
    }
    
    this.currentSession = newSession
    
    // Store session info
    if (typeof window !== 'undefined') {
      sessionStorage.setItem(this.SESSION_KEY, JSON.stringify(newSession))
    }
    
    console.log('✅ SessionManager: New session initialized:', newSession.sessionId)
    return newSession
  }

  /**
   * Get current session
   */
  getCurrentSession(): UserSession | null {
    if (this.currentSession) {
      return this.currentSession
    }
    
    // Try to restore from session storage
    if (typeof window !== 'undefined') {
      try {
        const stored = sessionStorage.getItem(this.SESSION_KEY)
        if (stored) {
          this.currentSession = JSON.parse(stored)
          return this.currentSession
        }
      } catch (error) {
        console.error('SessionManager: Failed to restore session:', error)
      }
    }
    
    return null
  }

  /**
   * Clear current session with complete data isolation
   */
  clearCurrentSession(config: Partial<SessionIsolationConfig> = {}): void {
    const fullConfig: SessionIsolationConfig = {
      clearAllCaches: true,
      clearLocalStorage: true,
      clearSessionStorage: true,
      clearIndexedDB: true,
      clearCookies: true,
      ...config
    }
    
    console.log('🧹 SessionManager: Clearing current session with config:', fullConfig)
    
    if (typeof window === 'undefined') return
    
    try {
      // Clear session storage
      if (fullConfig.clearSessionStorage) {
        sessionStorage.clear()
        console.log('✅ SessionManager: Session storage cleared')
      }
      
      // Clear user-specific localStorage items
      if (fullConfig.clearLocalStorage) {
        this.clearUserSpecificLocalStorage()
      }
      
      // Clear all caches
      if (fullConfig.clearAllCaches) {
        this.clearAllUserCaches()
      }
      
      // Clear cookies
      if (fullConfig.clearCookies) {
        this.clearAuthCookies()
      }
      
      // Clear IndexedDB (if used)
      if (fullConfig.clearIndexedDB) {
        this.clearIndexedDB()
      }
      
      // Reset current session
      this.currentSession = null
      
      console.log('✅ SessionManager: Session cleared completely')
      
    } catch (error) {
      console.error('❌ SessionManager: Error clearing session:', error)
    }
  }

  /**
   * Switch to a different user account with complete isolation
   */
  switchUser(newUserId: string, newEmail: string): UserSession {
    console.log('🔄 SessionManager: Switching user from', this.currentSession?.email, 'to', newEmail)
    
    // Clear current session completely
    this.clearCurrentSession()
    
    // Initialize new session
    return this.initializeSession(newUserId, newEmail)
  }

  /**
   * Generate user-specific cache key
   */
  getUserCacheKey(baseKey: string, userId?: string): string {
    const currentUserId = userId || this.currentSession?.userId
    if (!currentUserId) {
      return baseKey // Fallback to non-user-specific key
    }
    return `${this.USER_CACHE_PREFIX}${currentUserId}_${baseKey}`
  }

  /**
   * Check if current session belongs to specific user
   */
  isCurrentUser(userId: string): boolean {
    return this.currentSession?.userId === userId
  }

  /**
   * Get session age in milliseconds
   */
  getSessionAge(): number {
    if (!this.currentSession) return 0
    return Date.now() - this.currentSession.timestamp
  }

  /**
   * Validate session integrity
   */
  validateSession(): boolean {
    if (!this.currentSession) return false
    
    // Check if session is too old (24 hours)
    const maxAge = 24 * 60 * 60 * 1000
    if (this.getSessionAge() > maxAge) {
      console.log('⚠️ SessionManager: Session expired, clearing')
      this.clearCurrentSession()
      return false
    }
    
    return true
  }

  /**
   * Clear user-specific localStorage items
   */
  private clearUserSpecificLocalStorage(): void {
    try {
      const keysToRemove: string[] = []
      
      // Collect all localStorage keys
      for (let i = 0; i < localStorage.length; i++) {
        const key = localStorage.key(i)
        if (key) {
          // Remove user-specific keys and app-specific keys
          if (key.startsWith('sixthvault') || 
              key.startsWith('vault_') || 
              key.startsWith('user_') ||
              key.startsWith('auth-') ||
              key.includes('cache') ||
              key.includes('session') ||
              key.includes('conversation') ||
              key.includes('curation') ||
              key.includes('document')) {
            keysToRemove.push(key)
          }
        }
      }
      
      // Remove collected keys
      keysToRemove.forEach(key => {
        localStorage.removeItem(key)
      })
      
      console.log(`✅ SessionManager: Removed ${keysToRemove.length} localStorage items`)
      
    } catch (error) {
      console.error('❌ SessionManager: Error clearing localStorage:', error)
    }
  }

  /**
   * Clear all user-specific caches
   */
  private clearAllUserCaches(): void {
    try {
      // Import cache manager dynamically to avoid circular dependencies
      import('./cache-manager').then(({ cacheManager }) => {
        // Clear all cache entries
        cacheManager.clear()
        console.log('✅ SessionManager: All caches cleared')
      }).catch(error => {
        console.error('❌ SessionManager: Error clearing caches:', error)
      })
      
    } catch (error) {
      console.error('❌ SessionManager: Error accessing cache manager:', error)
    }
  }

  /**
   * Clear authentication cookies
   */
  private clearAuthCookies(): void {
    try {
      // Clear auth token cookie
      document.cookie = 'auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax'
      
      // Clear any other auth-related cookies
      const cookiesToClear = ['auth-token', 'session-id', 'user-id', 'refresh-token']
      cookiesToClear.forEach(cookieName => {
        document.cookie = `${cookieName}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax`
        document.cookie = `${cookieName}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; samesite=lax; secure`
      })
      
      console.log('✅ SessionManager: Auth cookies cleared')
      
    } catch (error) {
      console.error('❌ SessionManager: Error clearing cookies:', error)
    }
  }

  /**
   * Clear IndexedDB data
   */
  private clearIndexedDB(): void {
    try {
      if ('indexedDB' in window) {
        // Clear common IndexedDB databases used by the app
        const dbNames = ['sixthvault', 'vault-cache', 'user-data']
        
        dbNames.forEach(dbName => {
          const deleteReq = indexedDB.deleteDatabase(dbName)
          deleteReq.onsuccess = () => {
            console.log(`✅ SessionManager: IndexedDB ${dbName} cleared`)
          }
          deleteReq.onerror = () => {
            console.log(`⚠️ SessionManager: IndexedDB ${dbName} not found or already cleared`)
          }
        })
      }
    } catch (error) {
      console.error('❌ SessionManager: Error clearing IndexedDB:', error)
    }
  }

  /**
   * Force clear all browser storage (nuclear option)
   */
  forceCompleteCleanup(): void {
    console.log('💥 SessionManager: Performing complete browser storage cleanup')
    
    if (typeof window === 'undefined') return
    
    try {
      // Clear all storage types
      localStorage.clear()
      sessionStorage.clear()
      
      // Clear all cookies
      document.cookie.split(";").forEach(cookie => {
        const eqPos = cookie.indexOf("=")
        const name = eqPos > -1 ? cookie.substr(0, eqPos).trim() : cookie.trim()
        document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT`
        document.cookie = `${name}=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT; secure`
      })
      
      // Clear IndexedDB
      this.clearIndexedDB()
      
      // Clear cache API if available
      if ('caches' in window) {
        caches.keys().then(names => {
          names.forEach(name => {
            caches.delete(name)
          })
        })
      }
      
      console.log('✅ SessionManager: Complete cleanup finished')
      
    } catch (error) {
      console.error('❌ SessionManager: Error in complete cleanup:', error)
    }
  }

  /**
   * Get session statistics for debugging
   */
  getSessionStats(): {
    hasSession: boolean
    sessionAge: number
    userId?: string
    email?: string
    sessionId?: string
  } {
    const session = this.getCurrentSession()
    return {
      hasSession: !!session,
      sessionAge: this.getSessionAge(),
      userId: session?.userId,
      email: session?.email,
      sessionId: session?.sessionId
    }
  }
}

// Export singleton instance
export const userSessionManager = UserSessionManager.getInstance()

export default userSessionManager
