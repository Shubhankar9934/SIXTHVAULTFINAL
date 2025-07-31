"use client"

import { useState, useEffect, useRef } from 'react'
import { Loader2, Shield, RefreshCcw, CheckCircle, AlertCircle, Wifi, WifiOff, X } from "lucide-react"
import { Button } from "./button"
import { Progress } from "./progress"
import { AuthLoadingState } from "@/lib/auth-loading-state"

interface LoadingOverlayProps {
  loadingState?: AuthLoadingState
  onRetry?: () => void
  onForceRefresh?: () => void
  onGoToLogin?: () => void
}

export function LoadingOverlay({ 
  loadingState,
  onRetry,
  onForceRefresh,
  onGoToLogin
}: LoadingOverlayProps) {
  const [isVisible, setIsVisible] = useState(true)
  const [showEmergencyExit, setShowEmergencyExit] = useState(false)
  const emergencyTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const autoHideTimeoutRef = useRef<NodeJS.Timeout | null>(null)

  // Auto-hide on completion with better timing
  useEffect(() => {
    if (autoHideTimeoutRef.current) {
      clearTimeout(autoHideTimeoutRef.current)
      autoHideTimeoutRef.current = null
    }

    if (loadingState?.stage === 'complete') {
      autoHideTimeoutRef.current = setTimeout(() => {
        setIsVisible(false)
      }, 300) // Faster hide for better UX
    }

    return () => {
      if (autoHideTimeoutRef.current) {
        clearTimeout(autoHideTimeoutRef.current)
        autoHideTimeoutRef.current = null
      }
    }
  }, [loadingState?.stage])

  // Show emergency exit after extended loading
  useEffect(() => {
    if (emergencyTimeoutRef.current) {
      clearTimeout(emergencyTimeoutRef.current)
      emergencyTimeoutRef.current = null
    }

    if (loadingState && loadingState.stage !== 'complete' && loadingState.stage !== 'error') {
      emergencyTimeoutRef.current = setTimeout(() => {
        console.warn('LoadingOverlay: Emergency exit activated after extended loading')
        setShowEmergencyExit(true)
      }, 10000) // Show emergency exit after 10 seconds
    } else {
      setShowEmergencyExit(false)
    }

    return () => {
      if (emergencyTimeoutRef.current) {
        clearTimeout(emergencyTimeoutRef.current)
        emergencyTimeoutRef.current = null
      }
    }
  }, [loadingState])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (emergencyTimeoutRef.current) {
        clearTimeout(emergencyTimeoutRef.current)
        emergencyTimeoutRef.current = null
      }
      if (autoHideTimeoutRef.current) {
        clearTimeout(autoHideTimeoutRef.current)
        autoHideTimeoutRef.current = null
      }
    }
  }, [])

  if (!isVisible || !loadingState) return null

  const getIcon = () => {
    switch (loadingState.stage) {
      case 'complete':
        return <CheckCircle className="h-8 w-8 text-green-600" />
      case 'error':
        return <AlertCircle className="h-8 w-8 text-red-600" />
      default:
        return (
          <div className="relative">
            <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
            <div className="absolute inset-0 blur-sm animate-pulse bg-blue-400/20 rounded-full" />
          </div>
        )
    }
  }

  const getStatusColor = () => {
    switch (loadingState.stage) {
      case 'complete':
        return 'text-green-700'
      case 'error':
        return 'text-red-700'
      case 'navigating':
        return 'text-purple-700'
      default:
        return 'text-blue-700'
    }
  }

  const getProgressColor = () => {
    switch (loadingState.stage) {
      case 'complete':
        return 'bg-green-500'
      case 'error':
        return 'bg-red-500'
      case 'navigating':
        return 'bg-purple-500'
      default:
        return 'bg-blue-500'
    }
  }

  const formatTime = (ms: number) => {
    return `${Math.round(ms / 1000)}s`
  }

  return (
    <div className="fixed inset-0 bg-slate-900/50 backdrop-blur-sm z-[9999] flex items-center justify-center transition-opacity duration-300 ease-in-out">
      <div className="bg-white/95 p-8 rounded-xl shadow-2xl flex flex-col items-center space-y-6 backdrop-blur-sm max-w-md mx-4 border border-slate-200/50 relative">
        {/* Emergency Exit Button */}
        {showEmergencyExit && (
          <button
            onClick={() => {
              console.log('LoadingOverlay: Emergency exit triggered')
              setIsVisible(false)
              window.location.href = '/login'
            }}
            className="absolute top-4 right-4 p-2 text-slate-400 hover:text-slate-600 hover:bg-slate-100 rounded-full transition-colors"
            title="Force close and return to login"
          >
            <X className="h-4 w-4" />
          </button>
        )}

        {/* Icon */}
        <div className="relative">
          {getIcon()}
        </div>

        {/* Main Content */}
        <div className="text-center space-y-4 w-full">
          <div className="space-y-2">
            <h3 className={`font-semibold text-lg ${getStatusColor()}`}>
              {loadingState.stage === 'complete' ? 'Success!' : 
               loadingState.stage === 'error' ? 'Connection Issue' : 
               'Authenticating'}
            </h3>
            <p className="text-slate-700 font-medium">
              {loadingState.message}
            </p>
          </div>

          {/* Progress Bar */}
          {loadingState.stage !== 'error' && (
            <div className="w-full space-y-2">
              <Progress 
                value={loadingState.progress} 
                className="w-full h-2"
              />
              <div className="flex justify-between text-xs text-slate-500">
                <span>Progress</span>
                <span>{loadingState.progress}%</span>
              </div>
            </div>
          )}

          {/* Network Status */}
          {loadingState.timeElapsed > 5000 && loadingState.stage !== 'complete' && (
            <div className="flex items-center justify-center space-x-2 text-sm text-slate-500 bg-slate-50 rounded-lg p-2">
              {loadingState.message.includes('Slow connection') ? (
                <WifiOff className="h-4 w-4 text-orange-500" />
              ) : (
                <Wifi className="h-4 w-4 text-green-500" />
              )}
              <span>
                {loadingState.message.includes('Slow connection') 
                  ? 'Slow connection detected' 
                  : 'Connection stable'}
              </span>
              <span>• {formatTime(loadingState.timeElapsed)}</span>
            </div>
          )}

          {/* Emergency Exit Warning */}
          {showEmergencyExit && loadingState.stage !== 'error' && (
            <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <p className="text-sm text-yellow-800">
                Taking longer than expected? Click the × button above to force exit.
              </p>
            </div>
          )}

          {/* Error State Actions */}
          {loadingState.stage === 'error' && (
            <div className="space-y-3 w-full">
              <p className="text-sm text-slate-600">
                We're having trouble connecting. This could be due to network issues or server maintenance.
              </p>
              
              {loadingState.canRetry && onRetry && (
                <Button
                  onClick={() => {
                    console.log('LoadingOverlay: Retry button clicked')
                    onRetry()
                  }}
                  className="w-full bg-blue-600 hover:bg-blue-700 text-white"
                >
                  <RefreshCcw className="h-4 w-4 mr-2" />
                  Try Again
                </Button>
              )}
            </div>
          )}

          {/* Stuck State Actions */}
          {loadingState.showAlternatives && (
            <div className="space-y-3 w-full border-t border-slate-200 pt-4">
              <p className="text-sm text-slate-600">
                This is taking longer than expected. You can try these options:
              </p>
              
              <div className="space-y-2">
                <Button
                  onClick={() => {
                    console.log('LoadingOverlay: Page refresh triggered')
                    window.location.reload()
                  }}
                  variant="outline"
                  className="w-full border-slate-300 hover:bg-slate-50"
                >
                  <RefreshCcw className="h-4 w-4 mr-2" />
                  Refresh Page
                </Button>
                
                {onForceRefresh && (
                  <Button
                    onClick={() => {
                      console.log('LoadingOverlay: Force refresh triggered')
                      onForceRefresh()
                    }}
                    variant="outline"
                    className="w-full border-orange-300 text-orange-700 hover:bg-orange-50"
                  >
                    <Shield className="h-4 w-4 mr-2" />
                    Clear Session & Restart
                  </Button>
                )}
                
                <Button
                  onClick={() => {
                    console.log('LoadingOverlay: Return to login triggered')
                    const loginHandler = onGoToLogin || (() => window.location.href = '/login')
                    loginHandler()
                  }}
                  variant="ghost"
                  className="w-full text-slate-600 hover:text-slate-800 hover:bg-slate-100"
                >
                  Return to Login
                </Button>
              </div>
            </div>
          )}

          {/* Success State */}
          {loadingState.stage === 'complete' && (
            <p className="text-sm text-green-600">
              Redirecting to your workspace...
            </p>
          )}
        </div>
      </div>
    </div>
  )
}
