"use client"

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react'
import { AuthLoadingManager, AuthLoadingState } from './auth-loading-state'
import { userSessionManager } from './user-session-manager'
import { vaultLoadingManager } from './vault-loading-manager'

interface User {
  id: number
  email: string
  username?: string
  first_name: string
  last_name: string
  company?: string
  company_id?: string
  verified: boolean
  role: string
  is_admin: boolean
  is_active: boolean
  created_at: string
  last_login?: string
}

interface RegisterData {
  email: string
  password: string
  first_name: string
  last_name: string
  company?: string
}

interface RegisterResponse {
  email: string
  verificationCode: string
}

interface AuthContextType {
  user: User | null
  login: (email: string, password: string) => Promise<boolean>
  register: (userData: RegisterData) => Promise<RegisterResponse>
  logout: () => Promise<void>
  forceRefresh: () => void
  clearAuthState: () => void
  isAuthenticated: boolean
  isLoading: boolean
  authError: string | null
  loadingState: AuthLoadingState | null
  retryAuth: () => Promise<void>
}

interface AuthState {
  user: User | null
  isAuthenticated: boolean
  isLoading: boolean
  authError: string | null
  lastStateChange: number
  sessionId: string
}

const AuthContext = createContext<AuthContextType | undefined>(undefined)

// Auth state persistence key
const AUTH_STATE_KEY = 'sixthvault-auth-state'

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AuthState>({
    user: null,
    isAuthenticated: false,
    isLoading: true,
    authError: null,
    lastStateChange: Date.now(),
    sessionId: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
  })

  const [loadingState, setLoadingState] = useState<AuthLoadingState | null>(null)
  const loadingManagerRef = useRef<AuthLoadingManager>(new AuthLoadingManager())
  const apiUrl = process.env.NEXT_PUBLIC_RAG_API_URL || 'https://sixth-vault.com/api'
  const initializationRef = useRef(false)
  const authOperationRef = useRef<AbortController | null>(null)
  const cleanupTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Function to set auth cookie with enhanced debugging
  const setAuthCookie = (token: string) => {
    if (typeof window === 'undefined') return
    
    console.log('🍪 AUTH: Setting auth cookie with token:', token.substring(0, 20) + '...')
    console.log('🍪 AUTH: Token length:', token.length)
    console.log('🍪 AUTH: Current domain:', window.location.hostname)
    console.log('🍪 AUTH: Current protocol:', window.location.protocol)
    
    // Clear any existing auth token first
    document.cookie = 'auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT'
    
    // Use secure cookie settings for production, lax for development
    const isProduction = window.location.protocol === 'https:'
    const cookieString = isProduction 
      ? `auth-token=${encodeURIComponent(token)}; path=/; max-age=604800; samesite=lax; secure`
      : `auth-token=${encodeURIComponent(token)}; path=/; max-age=604800; samesite=lax`
    
    console.log('🍪 AUTH: Cookie string:', cookieString.substring(0, 100) + '...')
    document.cookie = cookieString
    
    if (isProduction) {
      console.log('🍪 AUTH: Auth cookie set for production environment')
    } else {
      console.log('🍪 AUTH: Auth cookie set for development environment')
    }
    
    // Verify cookie was set with multiple attempts
    let cookieCheck = getAuthCookie()
    let attempts = 0
    const maxAttempts = 3
    
    while (!cookieCheck && attempts < maxAttempts) {
      attempts++
      console.log(`🍪 AUTH: Cookie verification attempt ${attempts}/${maxAttempts}`)
      setTimeout(() => {
        cookieCheck = getAuthCookie()
      }, 100 * attempts) // Progressive delay
    }
    
    if (cookieCheck) {
      console.log('✅ AUTH: Auth cookie set and verified successfully')
      console.log('🍪 AUTH: Verified token preview:', cookieCheck.substring(0, 20) + '...')
    } else {
      console.error('❌ AUTH: Auth cookie not available after setting')
      console.log('🍪 AUTH: All cookies after setting:', document.cookie)
    }
  }

  // Function to remove auth cookie
  const removeAuthCookie = () => {
    if (typeof window === 'undefined') return
    document.cookie = 'auth-token=; path=/; expires=Thu, 01 Jan 1970 00:00:00 GMT'
  }

  // Function to get auth cookie
  const getAuthCookie = () => {
    if (typeof window === 'undefined') return null
    return document.cookie.split('; ').find(row => row.startsWith('auth-token='))?.split('=')[1]
  }

  // Clear all auth state function with complete session isolation
  const clearAuthState = useCallback(() => {
    console.log('🧹 AuthContext: Clearing all auth state with complete session isolation')
    
    // Cancel any ongoing operations
    if (authOperationRef.current) {
      authOperationRef.current.abort()
      authOperationRef.current = null
    }
    
    // Clear timeouts
    if (cleanupTimeoutRef.current) {
      clearTimeout(cleanupTimeoutRef.current)
      cleanupTimeoutRef.current = null
    }
    
    // Use session manager for complete cleanup
    userSessionManager.clearCurrentSession({
      clearAllCaches: true,
      clearLocalStorage: true,
      clearSessionStorage: true,
      clearIndexedDB: true,
      clearCookies: true
    })
    
    // Generate new session ID for complete isolation
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    setState({
      user: null,
      isAuthenticated: false,
      isLoading: false,
      authError: null,
      lastStateChange: Date.now(),
      sessionId: newSessionId
    })
    
    // Reset loading manager
    loadingManagerRef.current.reset()
    setLoadingState(null)
    
    console.log('✅ AuthContext: Complete auth state cleared with session isolation')
  }, []) // No dependencies to prevent infinite loops

  // Force refresh function - stable version
  const forceRefresh = useCallback(() => {
    console.log('Force refresh triggered')
    clearAuthState()
    window.location.reload()
  }, [clearAuthState])

  // Update loading state periodically with better cleanup - simplified
  useEffect(() => {
    if (!loadingState) return

    const interval = setInterval(() => {
      // Check if loading manager is still active
      if (!loadingManagerRef.current.isManagerActive()) {
        setLoadingState(null)
        return
      }

      const currentState = loadingManagerRef.current.getCurrentState()
      
      // Only update if state actually changed to prevent infinite loops
      setLoadingState(prevState => {
        if (!prevState || 
            prevState.stage !== currentState.stage || 
            prevState.progress !== currentState.progress ||
            prevState.isStuck !== currentState.isStuck) {
          return currentState
        }
        return prevState
      })

      // Handle stuck states with automatic recovery
      if (currentState.isStuck && !currentState.showAlternatives) {
        console.warn('Auth operation stuck, initiating recovery')
        loadingManagerRef.current.forceError()
        setState(prev => ({
          ...prev,
          authError: 'Connection timeout. Please try refreshing the page.',
          lastStateChange: Date.now()
        }))
      }
    }, 1000)

    return () => clearInterval(interval)
  }, []) // Empty dependency array to prevent infinite loop

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      // Cancel any ongoing auth operations
      if (authOperationRef.current) {
        authOperationRef.current.abort()
      }
      
      // Clear any pending timeouts
      if (cleanupTimeoutRef.current) {
        clearTimeout(cleanupTimeoutRef.current)
      }
      
      // Reset loading manager
      loadingManagerRef.current.reset()
    }
  }, [])

  // Check for existing token on mount - simplified version
  useEffect(() => {
    // Prevent multiple initializations
    if (initializationRef.current) return
    initializationRef.current = true

    const initAuth = async () => {
      try {
        setState(prev => ({ ...prev, isLoading: true, authError: null, lastStateChange: Date.now() }))
        
        const token = getAuthCookie();
        if (!token) {
          // No token found - this is normal for public pages
          console.log('No auth token found, user is not authenticated');
          setState(prev => ({
            ...prev,
            user: null,
            isAuthenticated: false,
            authError: null,
            isLoading: false,
            lastStateChange: Date.now()
          }));
          return;
        }

        // Verify token with timeout
        const controller = new AbortController()
        const timeoutId = setTimeout(() => {
          controller.abort()
        }, 5000)

        let response: Response
        
        try {
          response = await fetch(`${apiUrl}/auth/me`, {
            headers: {
              'Authorization': `Bearer ${token}`,
              'Cache-Control': 'no-cache',
              'Pragma': 'no-cache',
            },
            credentials: 'include',
            signal: controller.signal
          });

          // Clear timeout on successful response
          clearTimeout(timeoutId)
        } catch (fetchError: any) {
          // Clear timeout in case of error
          clearTimeout(timeoutId)
          
          // Handle abort errors specifically
          if (fetchError.name === 'AbortError') {
            console.log('Auth token verification timed out');
            throw new Error('Connection timeout during authentication');
          }
          
          // Handle actual network errors only
          if (fetchError.name === 'TypeError' && 
              (fetchError.message.includes('Failed to fetch') || 
               fetchError.message.includes('Network request failed') ||
               fetchError.message.includes('fetch is not defined'))) {
            throw new Error('Network connection failed during token verification');
          }
          
          // Re-throw other errors
          throw fetchError;
        }

        if (!response.ok) {
          throw new Error(`Token validation failed: ${response.status}`);
        }

        const userData = await response.json();
        setState(prev => ({
          ...prev,
          user: userData,
          isAuthenticated: true,
          authError: null,
          lastStateChange: Date.now()
        }));

      } catch (error) {
        console.error('Auth initialization error:', error);
        // Clear invalid auth state
        removeAuthCookie();
        setState(prev => ({
          ...prev,
          user: null,
          isAuthenticated: false,
          authError: null, // Don't show error for missing tokens
          lastStateChange: Date.now()
        }));
        
        // Only redirect if we're on a protected route and in browser
        if (typeof window !== 'undefined') {
          const protectedRoutes = ['/vault', '/upload', '/documents', '/admin']
          if (protectedRoutes.some(route => window.location.pathname.startsWith(route))) {
            window.location.href = '/login';
          }
        }
      } finally {
        setState(prev => ({
          ...prev,
          isLoading: false,
          lastStateChange: Date.now()
        }));
      }
    };

    // Start auth initialization
    initAuth();

    // Cleanup function
    return () => {
      initializationRef.current = false
    };
  }, [apiUrl]); // Only depend on apiUrl

  const login = async (email: string, password: string) => {
    try {
      console.log('AuthContext: Starting login process for:', email)
      
      // Cancel any ongoing operations
      if (authOperationRef.current) {
        authOperationRef.current.abort()
      }
      
      // Create new abort controller for this login attempt
      authOperationRef.current = new AbortController()
      
      // Clear any existing auth state completely
      removeAuthCookie()
      if (typeof window !== 'undefined') {
        sessionStorage.clear()
        localStorage.removeItem('auth-state')
        localStorage.removeItem(AUTH_STATE_KEY)
      }
      
      // Set clean loading state
      setState(prev => ({ 
        ...prev, 
        user: null, 
        isAuthenticated: false,
        authError: null,
        isLoading: true,
        lastStateChange: Date.now()
      }))
      
      // Normalize email
      const normalizedEmail = email.toLowerCase().trim()
      
      console.log('AuthContext: Making login request to backend')
      let response: Response
      
      try {
        console.log('AuthContext: Making request to:', `${apiUrl}/auth/login`)
        console.log('AuthContext: Request headers:', {
          'Content-Type': 'application/json',
          'Accept': 'application/json',
          'Cache-Control': 'no-cache',
          'Pragma': 'no-cache'
        })
        console.log('AuthContext: Request body:', JSON.stringify({ email: normalizedEmail, password: '***' }))
        
        response = await fetch(`${apiUrl}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify({ email: normalizedEmail, password }),
          mode: 'cors',
          signal: authOperationRef.current?.signal
        })
        console.log('AuthContext: Response received, status:', response.status, 'ok:', response.ok)
        console.log('AuthContext: Response headers:', Object.fromEntries(response.headers.entries()))
      } catch (fetchError: any) {
        console.error('AuthContext: Fetch error details:', fetchError)
        console.error('AuthContext: Error name:', fetchError.name)
        console.error('AuthContext: Error message:', fetchError.message)
        console.error('AuthContext: Error stack:', fetchError.stack)
        
        // Only handle actual network/connection errors here
        if (fetchError.name === 'AbortError') {
          throw new Error('Request was cancelled')
        }
        if (fetchError.name === 'TypeError' && 
            (fetchError.message.includes('Failed to fetch') || 
             fetchError.message.includes('Network request failed') ||
             fetchError.message.includes('fetch is not defined') ||
             fetchError.message.includes('NetworkError') ||
             fetchError.message.includes('ERR_NETWORK'))) {
          throw new Error('Network connection failed. Please check your internet connection and try again.')
        }
        // For other fetch errors, provide a generic connection error
        throw new Error(`Connection failed: ${fetchError.message}`)
      }

      if (!response.ok) {
        let errorMessage = 'Login failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
          // If response is not JSON, use status-based message
          if (response.status === 401) {
            errorMessage = 'Invalid email or password'
          } else if (response.status === 403) {
            errorMessage = 'Account not verified. Please check your email.'
          } else if (response.status >= 500) {
            errorMessage = 'Server error. Please try again later.'
          } else {
            errorMessage = 'Network error. Please check your connection.'
          }
        }
        throw new Error(errorMessage)
      }

      let data
      try {
        const responseText = await response.text()
        console.log('AuthContext: Raw response text length:', responseText.length)
        console.log('AuthContext: Response text preview:', responseText.substring(0, 200))
        
        if (!responseText.trim()) {
          throw new Error('Empty response from server')
        }
        
        data = JSON.parse(responseText)
        console.log('AuthContext: Login response parsed successfully:', data)
      } catch (parseError) {
        console.error('AuthContext: Failed to parse login response:', parseError)
        throw new Error('Invalid response from server. Please try again.')
      }
      
      // Validate response structure
      if (!data.access_token) {
        console.error('AuthContext: No access token in response:', data)
        throw new Error('Invalid login response - missing access token')
      }
      
      if (!data.user) {
        console.error('AuthContext: No user data in response:', data)
        throw new Error('Invalid login response - missing user data')
      }
      
      console.log('AuthContext: Login response received, setting auth state')
      
      // Store token in cookie immediately
      setAuthCookie(data.access_token)
      
      // Small delay to ensure cookie is properly set before making subsequent requests
      await new Promise(resolve => setTimeout(resolve, 100))
      
      // Initialize new user session with complete isolation, preserving the auth token
      const userSession = userSessionManager.initializeSession(
        data.user.id.toString(),
        data.user.email,
        true // Preserve auth token that was just set
      )
      
      // Set the user data and authenticated state atomically
      setState({
        user: data.user,
        isAuthenticated: true,
        authError: null,
        isLoading: false,
        lastStateChange: Date.now(),
        sessionId: userSession.sessionId
      })
      
      // Start the vault loading process immediately after successful login
      console.log('🚀 AuthContext: Starting vault initialization loading sequence')
      vaultLoadingManager.startAuthentication()
      
      // Complete authentication step quickly since login is done
      setTimeout(() => {
        vaultLoadingManager.completeAuthentication()
      }, 500)
      
      // Clear the operation ref since we're done
      authOperationRef.current = null
      
      console.log('✅ AuthContext: Login successful with isolated user session:', userSession.sessionId)
      return true
      
    } catch (error) {
      console.error('AuthContext: Login error:', error)
      
      // Clear the operation ref
      authOperationRef.current = null
      
      // Don't handle aborted requests as errors
      if (error instanceof Error && error.name === 'AbortError') {
        console.log('AuthContext: Login request was aborted')
        return false
      }
      
      // Ensure complete cleanup on any error
      removeAuthCookie()
      if (typeof window !== 'undefined') {
        sessionStorage.clear()
        localStorage.removeItem('auth-state')
        localStorage.removeItem(AUTH_STATE_KEY)
      }
      
      setState(prev => ({
        ...prev,
        user: null,
        isAuthenticated: false,
        authError: error instanceof Error ? error.message : 'Login failed',
        isLoading: false,
        lastStateChange: Date.now()
      }))
      
      // Re-throw the error for the login page to handle
      if (error instanceof Error) {
        throw error
      } else {
        throw new Error('Network error. Please check your connection and try again.')
      }
    }
  }

  const register = async (userData: RegisterData) => {
    try {
      console.log('AuthContext: Starting registration process for:', userData.email)
      
      let response: Response
      
      try {
        console.log('AuthContext: Making registration request to:', `${apiUrl}/auth/register`)
        response = await fetch(`${apiUrl}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Accept': 'application/json'
          },
          body: JSON.stringify(userData),
          mode: 'cors'
        })
        console.log('AuthContext: Registration response received, status:', response.status, 'ok:', response.ok)
      } catch (fetchError: any) {
        console.error('AuthContext: Registration fetch error:', fetchError)
        
        // Handle actual network errors only
        if (fetchError.name === 'TypeError' && 
            (fetchError.message.includes('Failed to fetch') || 
             fetchError.message.includes('Network request failed') ||
             fetchError.message.includes('fetch is not defined') ||
             fetchError.message.includes('NetworkError') ||
             fetchError.message.includes('ERR_NETWORK'))) {
          throw new Error('Network connection failed. Please check your internet connection and try again.')
        }
        
        // For other fetch errors, be more specific
        throw new Error(`Registration request failed: ${fetchError.message}`)
      }

      if (!response.ok) {
        console.log('AuthContext: Registration failed with status:', response.status)
        let errorMessage = 'Registration failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
          console.log('AuthContext: Registration error details:', errorData)
        } catch (parseError) {
          console.error('AuthContext: Failed to parse error response:', parseError)
          // If response is not JSON, use status-based message
          if (response.status === 409) {
            errorMessage = 'Email already exists'
          } else if (response.status >= 500) {
            errorMessage = 'Server error. Please try again later.'
          } else {
            errorMessage = 'Registration failed. Please try again.'
          }
        }
        throw new Error(errorMessage)
      }

      let data
      try {
        const responseText = await response.text()
        console.log('AuthContext: Registration response text length:', responseText.length)
        
        if (!responseText.trim()) {
          throw new Error('Empty response from server')
        }
        
        data = JSON.parse(responseText)
        console.log('AuthContext: Registration response parsed successfully:', data)
      } catch (parseError) {
        console.error('AuthContext: Failed to parse registration response:', parseError)
        throw new Error('Invalid response from server. Please try again.')
      }

      // Validate response structure
      if (!data.email) {
        console.error('AuthContext: No email in registration response:', data)
        throw new Error('Invalid registration response - missing email')
      }

      console.log('AuthContext: Registration successful for:', data.email)
      return {
        email: data.email,
        verificationCode: data.verification_code || null
      }
    } catch (error) {
      console.error('AuthContext: Registration error:', error)
      
      // Don't re-wrap errors that are already properly formatted
      if (error instanceof Error) {
        throw error
      } else {
        throw new Error('Registration failed. Please try again.')
      }
    }
  }

  const logout = async () => {
    const currentSessionId = state.sessionId
    const currentUser = state.user
    console.log('🚪 AuthContext: Starting logout with complete session isolation for:', currentUser?.email)
    
    try {
      // Cancel any ongoing auth operations immediately
      if (authOperationRef.current) {
        authOperationRef.current.abort()
        authOperationRef.current = null
      }
      
      // Clear any pending timeouts
      if (cleanupTimeoutRef.current) {
        clearTimeout(cleanupTimeoutRef.current)
        cleanupTimeoutRef.current = null
      }
      
      // Get the current token before clearing
      const token = getAuthCookie()
      
      // Use session manager for complete session cleanup
      userSessionManager.clearCurrentSession({
        clearAllCaches: true,
        clearLocalStorage: true,
        clearSessionStorage: true,
        clearIndexedDB: true,
        clearCookies: true
      })
      
      // Generate new session ID for complete isolation
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      setState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        authError: null,
        lastStateChange: Date.now(),
        sessionId: newSessionId
      })
      
      // Try to revoke token on backend (don't block on failure)
      if (token) {
        try {
          const logoutController = new AbortController()
          const logoutTimeout = setTimeout(() => logoutController.abort(), 3000)
          
          fetch(`${apiUrl}/auth/logout`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${token}`,
            },
            signal: logoutController.signal
          }).then(() => {
            clearTimeout(logoutTimeout)
            console.log('✅ AuthContext: Backend logout successful')
          }).catch(error => {
            clearTimeout(logoutTimeout)
            console.warn('⚠️ AuthContext: Backend logout failed (non-blocking):', error)
          })
        } catch (error) {
          console.warn('⚠️ AuthContext: Backend logout setup failed (non-blocking):', error)
        }
      }

      console.log('✅ AuthContext: Logout completed with session isolation')
      console.log('   - Previous session:', currentSessionId)
      console.log('   - New session:', newSessionId)
      console.log('   - User cache cleared for:', currentUser?.email)
      
      // Use replace to avoid middleware redirect and ensure clean navigation
      window.location.replace('/')
      
    } catch (error) {
      console.error('❌ AuthContext: Logout error for session:', currentSessionId, error)
      
      // Force complete cleanup even on error using session manager
      userSessionManager.forceCompleteCleanup()
      
      const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      
      setState({
        user: null,
        isAuthenticated: false,
        isLoading: false,
        authError: null,
        lastStateChange: Date.now(),
        sessionId: newSessionId
      })
      
      setTimeout(() => {
        window.location.href = '/'
      }, 500)
    }
  }

  const retryAuth = async () => {
    try {
      setState(prev => ({ ...prev, authError: null, isLoading: true }))
      
      const token = getAuthCookie()
      if (!token) {
        throw new Error('No authentication token found')
      }

      let response: Response
      
      try {
        response = await fetch(`${apiUrl}/auth/me`, {
          headers: {
            'Authorization': `Bearer ${token}`,
            'Cache-Control': 'no-cache'
          }
        })
      } catch (fetchError: any) {
        // Handle actual network errors
        if (fetchError.name === 'TypeError' && 
            (fetchError.message.includes('Failed to fetch') || 
             fetchError.message.includes('Network request failed') ||
             fetchError.message.includes('fetch is not defined'))) {
          throw new Error('Network connection failed during retry')
        }
        throw fetchError
      }

      if (!response.ok) {
        throw new Error('Token validation failed')
      }

      const userData = await response.json()
      setState(prev => ({
        ...prev,
        user: userData,
        isAuthenticated: true,
        isLoading: false,
        authError: null
      }))
    } catch (error) {
      console.error('Retry auth failed:', error)
      setState(prev => ({
        ...prev,
        authError: error instanceof Error ? error.message : 'Authentication retry failed',
        isLoading: false
      }))
    }
  }

  const value: AuthContextType = {
    user: state.user,
    login,
    register,
    logout,
    forceRefresh,
    clearAuthState,
    isAuthenticated: state.isAuthenticated,
    isLoading: state.isLoading,
    authError: state.authError,
    loadingState,
    retryAuth
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

export function useAuth() {
  const context = useContext(AuthContext)
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}
