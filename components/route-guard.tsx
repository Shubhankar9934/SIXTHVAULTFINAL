"use client"

import { useEffect, useState, useRef } from 'react'
import { useRouter, usePathname } from 'next/navigation'
import { useAuth } from '@/lib/auth-context'
import { LoadingOverlay } from '@/components/ui/loading-overlay'
import { uploadStateManager } from '@/lib/upload-state-manager'
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import {
  Upload,
  AlertTriangle,
  Clock,
  FileText,
  Loader2,
  CheckCircle,
  XCircle,
  LogOut
} from "lucide-react"

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
    retryAuth,
    logout
  } = useAuth()
  
  const router = useRouter()
  const pathname = usePathname()
  const [isTransitioning, setIsTransitioning] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const transitionTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const hasCheckedAccessRef = useRef(false)
  
  // Upload blocking state
  const [showUploadBlockDialog, setShowUploadBlockDialog] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null)
  const [isLogoutAttempt, setIsLogoutAttempt] = useState(false)
  const [uploadState, setUploadState] = useState(uploadStateManager.getCurrentState())

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

  // Subscribe to upload state changes
  useEffect(() => {
    const unsubscribe = uploadStateManager.subscribe(setUploadState)
    return unsubscribe
  }, [])

  // Block navigation to vault page during uploads
  useEffect(() => {
    const originalPush = router.push
    const originalReplace = router.replace

    // Override router.push to check for upload blocks
    router.push = (href: string, options?: any) => {
      // Check if trying to navigate to vault during upload from documents page
      if (pathname === '/documents' && href === '/vault' && uploadState.hasActiveUploads) {
        console.log('🚫 RouteGuard: Blocking navigation to vault during upload')
        setShowUploadBlockDialog(true)
        setPendingNavigation(href)
        setIsLogoutAttempt(false)
        return Promise.resolve()
      }
      return originalPush.call(router, href, options)
    }

    // Override router.replace to check for upload blocks
    router.replace = (href: string, options?: any) => {
      // Check if trying to navigate to vault during upload from documents page
      if (pathname === '/documents' && href === '/vault' && uploadState.hasActiveUploads) {
        console.log('🚫 RouteGuard: Blocking navigation to vault during upload')
        setShowUploadBlockDialog(true)
        setPendingNavigation(href)
        setIsLogoutAttempt(false)
        return Promise.resolve()
      }
      return originalReplace.call(router, href, options)
    }

    return () => {
      // Restore original methods
      router.push = originalPush
      router.replace = originalReplace
    }
  }, [router, pathname, uploadState.hasActiveUploads])

  // Block logout during uploads
  useEffect(() => {
    const originalLogout = logout

    // Override logout to check for uploads
    const wrappedLogout = async () => {
      if (uploadState.hasActiveUploads) {
        console.log('🚫 RouteGuard: Blocking logout during upload')
        setShowUploadBlockDialog(true)
        setIsLogoutAttempt(true)
        setPendingNavigation('/login')
        return
      }
      return originalLogout()
    }

    // Store the wrapped logout for use in dialogs
    ;(window as any).wrappedLogout = wrappedLogout

    return () => {
      delete (window as any).wrappedLogout
    }
  }, [logout, uploadState.hasActiveUploads])

  // Handle browser back/forward navigation during uploads
  useEffect(() => {
    const handlePopState = (event: PopStateEvent) => {
      if (pathname === '/documents' && uploadState.hasActiveUploads) {
        event.preventDefault()
        setShowUploadBlockDialog(true)
        setPendingNavigation(window.location.pathname)
        setIsLogoutAttempt(false)
        // Push current state back to prevent navigation
        window.history.pushState(null, '', pathname)
      }
    }

    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (pathname === '/documents' && uploadState.hasActiveUploads) {
        const message = 'Documents are currently uploading. Leaving now may interrupt the process.'
        event.preventDefault()
        event.returnValue = message
        return message
      }
    }

    window.addEventListener('popstate', handlePopState)
    window.addEventListener('beforeunload', handleBeforeUnload)

    return () => {
      window.removeEventListener('popstate', handlePopState)
      window.removeEventListener('beforeunload', handleBeforeUnload)
    }
  }, [pathname, uploadState.hasActiveUploads])

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

  // Helper functions for upload blocking dialog
  const handleForceNavigation = async () => {
    console.log('🔓 RouteGuard: Force navigation requested, clearing upload state')
    
    // Clear upload state to allow navigation
    uploadStateManager.forceReset()
    
    setShowUploadBlockDialog(false)
    
    if (isLogoutAttempt) {
      // Perform logout
      await logout()
    } else if (pendingNavigation) {
      // Navigate to pending destination
      window.location.href = pendingNavigation
    }
    
    setPendingNavigation(null)
    setIsLogoutAttempt(false)
  }

  const handleCancelNavigation = () => {
    setShowUploadBlockDialog(false)
    setPendingNavigation(null)
    setIsLogoutAttempt(false)
  }

  const getStatusIcon = (status: string) => {
    switch (status) {
      case 'uploading':
        return <Upload className="w-4 h-4 text-blue-500" />
      case 'processing':
        return <Loader2 className="w-4 h-4 text-blue-500 animate-spin" />
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-500" />
      case 'error':
        return <XCircle className="w-4 h-4 text-red-500" />
      case 'waiting':
        return <Clock className="w-4 h-4 text-yellow-500" />
      default:
        return <FileText className="w-4 h-4 text-gray-500" />
    }
  }

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'uploading':
        return 'bg-blue-100 text-blue-800'
      case 'processing':
        return 'bg-blue-100 text-blue-800'
      case 'completed':
        return 'bg-green-100 text-green-800'
      case 'error':
        return 'bg-red-100 text-red-800'
      case 'waiting':
        return 'bg-yellow-100 text-yellow-800'
      default:
        return 'bg-gray-100 text-gray-800'
    }
  }

  const getUploadProgress = () => {
    const completed = uploadState.completedFiles
    const total = uploadState.totalFiles
    const percentage = total > 0 ? Math.round((completed / total) * 100) : 0
    return { completed, total, percentage }
  }

  const getActiveDocuments = () => {
    return uploadState.documents.filter(doc => 
      doc.status === 'uploading' || doc.status === 'processing' || doc.status === 'waiting'
    )
  }

  // If we're not loading and we have access, show the protected content
  console.log('RouteGuard: Rendering protected content')
  return (
    <>
      {children}
      
      {/* Upload Navigation Block Dialog */}
      <AlertDialog open={showUploadBlockDialog} onOpenChange={setShowUploadBlockDialog}>
        <AlertDialogContent className="max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-orange-100 rounded-full flex items-center justify-center">
                <AlertTriangle className="w-6 h-6 text-orange-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">
                  {isLogoutAttempt ? 'Logout Blocked' : 'Navigation Blocked'}
                </h3>
                <p className="text-sm text-gray-500 mt-1">Upload in progress</p>
              </div>
            </AlertDialogTitle>
          </AlertDialogHeader>
          
          <AlertDialogDescription>
            {isLogoutAttempt ? (
              "You cannot logout while documents are uploading. This could interrupt the upload process and cause data loss."
            ) : (
              "You cannot navigate to the vault while documents are uploading. This could interrupt the upload process and cause data loss."
            )}
          </AlertDialogDescription>
          
          <div className="space-y-4">
            {/* Upload Progress Overview */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Upload Progress</span>
                <Badge className="bg-blue-100 text-blue-800">
                  {getUploadProgress().completed} / {getUploadProgress().total} files
                </Badge>
              </div>
              
              <Progress 
                value={getUploadProgress().percentage} 
                className="h-2"
              />
              
              <div className="text-xs text-gray-500 text-center">
                {getUploadProgress().percentage}% complete
              </div>
            </div>

            {/* Active Documents List */}
            {getActiveDocuments().length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Active Uploads:</h4>
                <div className="max-h-32 overflow-y-auto space-y-2">
                  {getActiveDocuments().slice(0, 3).map((doc) => (
                    <div key={doc.id} className="flex items-center justify-between bg-white rounded-lg p-2 border border-gray-200">
                      <div className="flex items-center space-x-2 flex-1 min-w-0">
                        {getStatusIcon(doc.status)}
                        <span className="text-sm text-gray-700 truncate" title={doc.name}>
                          {doc.name}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2 flex-shrink-0">
                        <Badge variant="outline" className={getStatusColor(doc.status)}>
                          {doc.status}
                        </Badge>
                        <span className="text-xs text-gray-500 min-w-[3rem] text-right">
                          {doc.progress}%
                        </span>
                      </div>
                    </div>
                  ))}
                  
                  {getActiveDocuments().length > 3 && (
                    <div className="text-xs text-gray-500 text-center">
                      +{getActiveDocuments().length - 3} more documents
                    </div>
                  )}
                </div>
              </div>
            )}

            <div className="bg-blue-50 border border-blue-200 rounded-lg p-3">
              <div className="flex items-start space-x-2">
                <Clock className="w-4 h-4 text-blue-600 mt-0.5 flex-shrink-0" />
                <div className="text-sm text-blue-800">
                  <strong>Please wait</strong> for all uploads to complete before navigating away. 
                  You can continue using this page while uploads are in progress.
                </div>
              </div>
            </div>
          </div>

          <AlertDialogFooter className="space-x-3">
            <AlertDialogCancel 
              onClick={handleCancelNavigation}
              className="bg-gray-100 hover:bg-gray-200 text-gray-700"
            >
              Stay on Page
            </AlertDialogCancel>
            
            <AlertDialogAction
              onClick={handleForceNavigation}
              className="bg-red-600 hover:bg-red-700 text-white"
            >
              {isLogoutAttempt ? (
                <div className="flex items-center space-x-2">
                  <LogOut className="w-4 h-4" />
                  <span>Force Logout</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <AlertTriangle className="w-4 h-4" />
                  <span>Leave Anyway</span>
                </div>
              )}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  )
}
