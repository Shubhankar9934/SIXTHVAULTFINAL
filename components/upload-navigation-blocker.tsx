"use client"

import React, { useEffect, useState } from 'react'
import { useRouter, usePathname } from 'next/navigation'
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
import { Button } from "@/components/ui/button"
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
import { useUploadState } from '@/lib/upload-state-manager'
import { useAuth } from '@/lib/auth-context'

interface NavigationBlockerProps {
  children: React.ReactNode
}

export function UploadNavigationBlocker({ children }: NavigationBlockerProps) {
  const router = useRouter()
  const pathname = usePathname()
  const { logout } = useAuth()
  const uploadState = useUploadState()
  
  const [showBlockDialog, setShowBlockDialog] = useState(false)
  const [pendingNavigation, setPendingNavigation] = useState<string | null>(null)
  const [isLogoutAttempt, setIsLogoutAttempt] = useState(false)

  // Override browser navigation during uploads
  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (uploadState.hasActiveUploads) {
        const message = 'Documents are currently uploading. Leaving now may interrupt the process.'
        event.preventDefault()
        event.returnValue = message
        return message
      }
    }

    const handlePopState = (event: PopStateEvent) => {
      if (uploadState.hasActiveUploads) {
        event.preventDefault()
        setShowBlockDialog(true)
        setPendingNavigation(window.location.pathname)
        // Push current state back to prevent navigation
        window.history.pushState(null, '', pathname)
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    window.addEventListener('popstate', handlePopState)

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      window.removeEventListener('popstate', handlePopState)
    }
  }, [uploadState.hasActiveUploads, pathname])

  // Intercept Link clicks and programmatic navigation
  useEffect(() => {
    const originalPush = router.push
    const originalReplace = router.replace

    // Override router.push
    router.push = (href: string, options?: any) => {
      if (uploadState.hasActiveUploads && pathname !== href) {
        console.log('🚫 Navigation blocked: Upload in progress, target:', href)
        setShowBlockDialog(true)
        setPendingNavigation(href)
        setIsLogoutAttempt(false)
        return Promise.resolve()
      }
      return originalPush.call(router, href, options)
    }

    // Override router.replace
    router.replace = (href: string, options?: any) => {
      if (uploadState.hasActiveUploads && pathname !== href) {
        console.log('🚫 Navigation blocked: Upload in progress, target:', href)
        setShowBlockDialog(true)
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
  }, [router, uploadState.hasActiveUploads, pathname])

  // Intercept logout attempts
  useEffect(() => {
    const handleLogoutAttempt = (event: CustomEvent) => {
      if (uploadState.hasActiveUploads) {
        event.preventDefault()
        setShowBlockDialog(true)
        setIsLogoutAttempt(true)
        setPendingNavigation('/login')
      }
    }

    window.addEventListener('logout-attempt' as any, handleLogoutAttempt)

    return () => {
      window.removeEventListener('logout-attempt' as any, handleLogoutAttempt)
    }
  }, [uploadState.hasActiveUploads])

  const handleForceNavigation = async () => {
    console.log('🔓 Force navigation requested, clearing upload state')
    
    // Clear upload state to allow navigation
    uploadState.forceReset()
    
    setShowBlockDialog(false)
    
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
    setShowBlockDialog(false)
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

  return (
    <>
      {children}
      
      {/* Navigation Block Dialog */}
      <AlertDialog open={showBlockDialog} onOpenChange={setShowBlockDialog}>
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
          
          <AlertDialogDescription className="space-y-4">
            <div className="text-gray-600">
              {isLogoutAttempt ? (
                <>
                  You cannot logout while documents are uploading. This could interrupt the upload process and cause data loss.
                </>
              ) : (
                <>
                  You cannot navigate away from this page while documents are uploading. This could interrupt the upload process and cause data loss.
                </>
              )}
            </div>

            {/* Upload Progress Overview */}
            <div className="bg-gray-50 rounded-lg p-4 space-y-3">
              <div className="flex items-center justify-between">
                <span className="text-sm font-medium text-gray-700">Upload Progress</span>
                <Badge className="bg-blue-100 text-blue-800">
                  {uploadState.completedFiles} / {uploadState.totalFiles} files
                </Badge>
              </div>
              
              <Progress 
                value={uploadState.getUploadProgress().percentage} 
                className="h-2"
              />
              
              <div className="text-xs text-gray-500 text-center">
                {uploadState.getUploadProgress().percentage}% complete
              </div>
            </div>

            {/* Active Documents List */}
            {uploadState.getActiveDocuments().length > 0 && (
              <div className="space-y-2">
                <h4 className="text-sm font-medium text-gray-700">Active Uploads:</h4>
                <div className="max-h-32 overflow-y-auto space-y-2">
                  {uploadState.getActiveDocuments().slice(0, 3).map((doc) => (
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
                  
                  {uploadState.getActiveDocuments().length > 3 && (
                    <div className="text-xs text-gray-500 text-center">
                      +{uploadState.getActiveDocuments().length - 3} more documents
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
          </AlertDialogDescription>

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

// Higher-order component to wrap pages that need upload protection
export function withUploadProtection<T extends object>(
  Component: React.ComponentType<T>
) {
  return function ProtectedComponent(props: T) {
    return (
      <UploadNavigationBlocker>
        <Component {...props} />
      </UploadNavigationBlocker>
    )
  }
}
