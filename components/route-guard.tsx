"use client"

import { useEffect, useState, useRef } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/lib/auth-context'
import { LoadingOverlay } from '@/components/ui/loading-overlay'

// Route categorization (matching middleware.ts)
const PUBLIC_ROUTES = ['/login', '/register', '/forgot-password']
const VERIFICATION_ROUTES = ['/verify']
const PROTECTED_ROUTES = ['/vault', '/upload', '/documents', '/admin']
const CONDITIONAL_ROUTES = ['/reset-password']

interface RouteGuardProps {
  children: React.ReactNode
}

export function RouteGuard({ children }: RouteGuardProps) {
  const { 
    isAuthenticated, 
    isLoading, 
    user, 
    authError,
    forceRefresh,
    clearAuthState,
    loadingState,
    retryAuth
  } = useAuth()
  
  const router = useRouter()
  const pathname = usePathname()
  const [isTransitioning, setIsTransitioning] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const transitionTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const hasCheckedAccessRef = useRef(false)

  useEffect(() => {
    // Clear any existing timeout
    if (transitionTimeoutRef.current) {
      clearTimeout(transitionTimeoutRef.current)
      transitionTimeoutRef.current = null
    }

    const checkAccess = () => {
      console.log('RouteGuard: Checking access for:', pathname, {
        isLoading,
        isAuthenticated,
        userVerified: user?.verified
      })

      // Always allow public routes
      if (PUBLIC_ROUTES.some(route => pathname?.startsWith(route))) {
        console.log('RouteGuard: Public route access granted')
        setIsTransitioning(false)
        setError(null)
        return true
      }

      // Handle authenticated users on login page with delay to prevent race conditions
      if (pathname === '/login' && isAuthenticated && user?.verified) {
        console.log('RouteGuard: Authenticated user on login page, redirecting to vault')
        // Add a small delay to ensure auth state is stable
        setTimeout(() => {
          window.location.href = '/vault'
        }, 100)
        return false
      }

      // Check verification routes
      if (VERIFICATION_ROUTES.some(route => pathname?.startsWith(route))) {
        const params = new URLSearchParams(window.location.search)
        const hasToken = params.get('token')
        const hasEmail = params.get('email')

        if (!hasToken || !hasEmail) {
          console.log('RouteGuard: Invalid verification link')
          setError('Invalid verification link')
          router.replace('/login')
          return false
        }
        console.log('RouteGuard: Verification route access granted')
        setIsTransitioning(false)
        setError(null)
        return true
      }

      // Check protected routes
      if (PROTECTED_ROUTES.some(route => pathname?.startsWith(route))) {
        if (!isAuthenticated) {
          console.log('RouteGuard: Authentication required for protected route')
          setError('Authentication required')
          router.replace(`/login?redirect=${encodeURIComponent(pathname || '')}`)
          return false
        }
        if (!user?.verified) {
          console.log('RouteGuard: Email verification required')
          setError('Email verification required')
          router.replace('/verify')
          return false
        }
        console.log('RouteGuard: Protected route access granted')
        setIsTransitioning(false)
        setError(null)
        return true
      }

      // Check conditional routes
      if (CONDITIONAL_ROUTES.some(route => pathname?.startsWith(route))) {
        const params = new URLSearchParams(window.location.search)
        const hasToken = params.get('token')

        if (!hasToken) {
          console.log('RouteGuard: Invalid reset link')
          setError('Invalid reset link')
          router.replace('/login')
          return false
        }
        console.log('RouteGuard: Conditional route access granted')
        setIsTransitioning(false)
        setError(null)
        return true
      }

      // Default: require authentication for unknown routes
      if (!isAuthenticated) {
        console.log('RouteGuard: Authentication required for unknown route')
        setError('Authentication required')
        router.replace('/login')
        return false
      }

      console.log('RouteGuard: Default access granted')
      setIsTransitioning(false)
      setError(null)
      return true
    }

    // Only check access when not loading and haven't checked for this specific state
    if (!isLoading) {
      // Create a unique key for this state combination
      const stateKey = `${pathname}-${isAuthenticated}-${user?.verified}-${Date.now()}`
      
      // Prevent multiple rapid checks for the same state
      if (hasCheckedAccessRef.current) {
        // Allow recheck after a short delay for state changes
        const timeSinceLastCheck = Date.now() - (hasCheckedAccessRef.current as any)
        if (timeSinceLastCheck < 100) {
          return
        }
      }
      
      hasCheckedAccessRef.current = Date.now() as any

      const hasAccess = checkAccess()
      if (hasAccess) {
        // Immediate transition for better UX
        setIsTransitioning(false)
      }
    }

    // Reset check flag when loading state changes or route changes
    return () => {
      if (isLoading) {
        hasCheckedAccessRef.current = false
      }
    }
  }, [isAuthenticated, isLoading, pathname, user?.verified, router])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (transitionTimeoutRef.current) {
        clearTimeout(transitionTimeoutRef.current)
        transitionTimeoutRef.current = null
      }
    }
  }, [])

  // Show loading state during initial load or transition
  if (isLoading || isTransitioning) {
    // Use the new LoadingOverlay if we have a loading state from auth context
    if (loadingState) {
      return (
        <LoadingOverlay 
          loadingState={loadingState}
          onRetry={retryAuth}
          onForceRefresh={forceRefresh}
          onGoToLogin={() => window.location.href = '/login'}
        />
      )
    }

    // Fallback to simple loading for route transitions with timeout protection
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
        <div className="bg-white/95 p-6 rounded-lg shadow-xl flex flex-col items-center space-y-4 backdrop-blur-sm max-w-md w-full">
          <div className="relative">
            <div className="h-8 w-8 border-4 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
            <div className="absolute inset-0 blur-sm animate-pulse bg-blue-400/30 rounded-full"></div>
          </div>
          
          <div className="text-center space-y-2">
            <p className="text-slate-700 font-medium text-lg">
              {error || authError || "Verifying access..."}
            </p>
            
            {(error || authError) && (
              <p className="text-sm text-slate-500">
                Please wait while we resolve this issue.
              </p>
            )}
          </div>

          {(error || authError) && (
            <div className="flex flex-col space-y-2 w-full">
              <button 
                onClick={() => {
                  console.log('RouteGuard: Manual page refresh triggered')
                  window.location.reload()
                }} 
                className="w-full px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 transition-colors"
              >
                Refresh Page
              </button>
              
              {forceRefresh && (
                <button 
                  onClick={() => {
                    console.log('RouteGuard: Force refresh triggered')
                    forceRefresh()
                  }}
                  className="w-full px-4 py-2 bg-red-600 text-white rounded-md hover:bg-red-700 transition-colors"
                >
                  Clear Session & Restart
                </button>
              )}
              
              <button 
                onClick={() => {
                  console.log('RouteGuard: Manual navigation to login')
                  window.location.href = '/login'
                }}
                className="w-full px-4 py-2 bg-slate-600 text-white rounded-md hover:bg-slate-700 transition-colors"
              >
                Go to Login
              </button>
            </div>
          )}

          {!error && !authError && (
            <div className="flex items-center space-x-2 text-sm text-slate-500">
              <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
              <span>Establishing secure connection...</span>
            </div>
          )}
        </div>
      </div>
    )
  }

  // If we're not loading and we have access, show the protected content
  console.log('RouteGuard: Rendering protected content')
  return <>{children}</>
}
