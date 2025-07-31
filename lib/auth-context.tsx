"use client"

import React, { createContext, useContext, useState, useEffect, useRef, useCallback } from 'react'
import { AuthLoadingManager, AuthLoadingState } from './auth-loading-state'

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
  const apiUrl = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'
  const initializationRef = useRef(false)
  const authOperationRef = useRef<AbortController | null>(null)
  const cleanupTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Function to set auth cookie - Updated for ngrok compatibility and reliability
  const setAuthCookie = (token: string) => {
    if (typeof window === 'undefined') return
    
    console.log('Setting auth cookie with token:', token.substring(0, 20) + '...')
    
    // Check if we're using ngrok (contains .ngrok-free.app or .ngrok.io)
    const isNgrok = window.location.hostname.includes('ngrok') || 
                   process.env.NEXT_PUBLIC_RAG_API_URL?.includes('ngrok')
    
    // Use different cookie settings for ngrok vs localhost
    if (isNgrok) {
      // For ngrok tunnels, use lax samesite and secure flag
      document.cookie = `auth-token=${token}; path=/; max-age=604800; samesite=lax; secure`
      console.log('Auth cookie set for ngrok environment')
    } else {
      // For localhost development, use lax samesite for better compatibility
      document.cookie = `auth-token=${token}; path=/; max-age=604800; samesite=lax`
      console.log('Auth cookie set for localhost environment')
    }
    
    // Verify cookie was set immediately and after a delay
    const immediateCheck = getAuthCookie()
    if (immediateCheck) {
      console.log('Auth cookie set successfully (immediate check)')
    } else {
      console.warn('Auth cookie not immediately available')
    }
    
    // Double-check after a delay
    setTimeout(() => {
      const cookieValue = getAuthCookie()
      if (cookieValue) {
        console.log('Auth cookie verification successful (delayed check)')
      } else {
        console.error('Auth cookie verification failed - cookie not found after setting')
        // Try setting again with different settings
        document.cookie = `auth-token=${token}; path=/; max-age=604800`
        console.log('Retried setting auth cookie with minimal settings')
      }
    }, 100)
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

  // Clear all auth state function - stable version with no dependencies
  const clearAuthState = useCallback(() => {
    console.log('Clearing all auth state')
    
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
    
    // Clear storage (with browser checks)
    removeAuthCookie()
    if (typeof window !== 'undefined') {
      sessionStorage.clear()
      localStorage.removeItem(AUTH_STATE_KEY)
      localStorage.removeItem('auth-state')
    }
    
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
              'ngrok-skip-browser-warning': 'true',
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
        
        // Only redirect if we're on a protected route
        const protectedRoutes = ['/vault', '/upload', '/documents', '/admin']
        if (protectedRoutes.some(route => window.location.pathname.startsWith(route))) {
          window.location.href = '/login';
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
        response = await fetch(`${apiUrl}/auth/login`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify({ email: normalizedEmail, password }),
          credentials: 'include',
          signal: authOperationRef.current?.signal
        })
      } catch (fetchError: any) {
        // Only handle actual network/connection errors here
        if (fetchError.name === 'AbortError') {
          throw new Error('Request was cancelled')
        }
        if (fetchError.name === 'TypeError' && 
            (fetchError.message.includes('Failed to fetch') || 
             fetchError.message.includes('Network request failed') ||
             fetchError.message.includes('fetch is not defined'))) {
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

      const data = await response.json()
      console.log('AuthContext: Login response received, setting auth state')
      
      // Store token in cookie immediately
      setAuthCookie(data.access_token)
      
      // Set the user data and authenticated state atomically
      setState({
        user: data.user,
        isAuthenticated: true,
        authError: null,
        isLoading: false,
        lastStateChange: Date.now(),
        sessionId: `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
      })
      
      // Clear the operation ref since we're done
      authOperationRef.current = null
      
      console.log('AuthContext: Login successful, auth state updated')
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
      let response: Response
      
      try {
        response = await fetch(`${apiUrl}/auth/register`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'ngrok-skip-browser-warning': 'true'
          },
          body: JSON.stringify(userData)
        })
      } catch (fetchError: any) {
        // Handle actual network errors
        if (fetchError.name === 'TypeError' && 
            (fetchError.message.includes('Failed to fetch') || 
             fetchError.message.includes('Network request failed') ||
             fetchError.message.includes('fetch is not defined'))) {
          throw new Error('Network connection failed. Please check your internet connection and try again.')
        }
        throw new Error(`Connection failed: ${fetchError.message}`)
      }

      if (!response.ok) {
        let errorMessage = 'Registration failed'
        try {
          const errorData = await response.json()
          errorMessage = errorData.detail || errorMessage
        } catch {
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

      const data = await response.json()
      return {
        email: data.email,
        verificationCode: data.verification_code || null
      }
    } catch (error) {
      console.error('Registration error:', error)
      throw error
    }
  }

  const logout = async () => {
    const currentSessionId = state.sessionId
    console.log('Starting logout for session:', currentSessionId)
    
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
      
      // Immediately clear local state to prevent UI confusion
      removeAuthCookie()
      if (typeof window !== 'undefined') {
        sessionStorage.clear()
        localStorage.removeItem('auth-state')
        localStorage.removeItem(AUTH_STATE_KEY)
      }
      
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
          }).catch(error => {
            clearTimeout(logoutTimeout)
            console.warn('Backend logout failed (non-blocking):', error)
          })
        } catch (error) {
          console.warn('Backend logout setup failed (non-blocking):', error)
        }
      }

      console.log('Logout completed for session:', currentSessionId, 'new session:', newSessionId)
      
      // Use replace to avoid middleware redirect and ensure clean navigation
      window.location.replace('/')
      
    } catch (error) {
      console.error('Logout error for session:', currentSessionId, error)
      
      // Force complete cleanup even on error
      removeAuthCookie()
      if (typeof window !== 'undefined') {
        sessionStorage.clear()
        localStorage.removeItem('auth-state')
        localStorage.removeItem(AUTH_STATE_KEY)
      }
      
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
            'Cache-Control': 'no-cache',
            'ngrok-skip-browser-warning': 'true'
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
