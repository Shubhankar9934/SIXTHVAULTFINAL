import { useState, useEffect, useCallback, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { AuthLoadingManager, AuthLoadingState } from '@/lib/auth-loading-state'

interface UsePageTransitionProps {
  onBeforeNavigate?: () => Promise<boolean> | boolean
  onNavigationStart?: () => void
  onNavigationComplete?: () => void
}

export function usePageTransition({
  onBeforeNavigate,
  onNavigationStart,
  onNavigationComplete
}: UsePageTransitionProps = {}) {
  const router = useRouter()
  const [isNavigating, setIsNavigating] = useState(false)
  const [loadingState, setLoadingState] = useState<AuthLoadingState | null>(null)
  const loadingManagerRef = useRef<AuthLoadingManager>(new AuthLoadingManager())
  const navigationTimeoutRef = useRef<NodeJS.Timeout | null>(null)
  const isNavigationActiveRef = useRef(false)

  useEffect(() => {
    // Cleanup function to handle unmounting during navigation
    return () => {
      if (navigationTimeoutRef.current) {
        clearTimeout(navigationTimeoutRef.current)
        navigationTimeoutRef.current = null
      }
      
      if (isNavigating) {
        onNavigationComplete?.()
        loadingManagerRef.current.reset()
        setLoadingState(null)
        isNavigationActiveRef.current = false
      }
    }
  }, [isNavigating, onNavigationComplete])

  const navigate = useCallback(async (path: string) => {
    // Prevent concurrent navigation attempts
    if (isNavigationActiveRef.current) {
      console.warn('Navigation already in progress, ignoring duplicate request')
      return false
    }

    isNavigationActiveRef.current = true
    const navigationId = `nav_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`
    
    try {
      console.log('Starting navigation:', navigationId, 'to:', path)
      
      // Clear any existing timeout
      if (navigationTimeoutRef.current) {
        clearTimeout(navigationTimeoutRef.current)
        navigationTimeoutRef.current = null
      }

      // Start loading state with timeout protection
      const initialState = loadingManagerRef.current.start('validating')
      setLoadingState(initialState)
      setIsNavigating(true)

      // Set navigation timeout to prevent infinite loading
      navigationTimeoutRef.current = setTimeout(() => {
        console.warn('Navigation timeout reached for:', navigationId)
        loadingManagerRef.current.forceError()
        setIsNavigating(false)
        isNavigationActiveRef.current = false
      }, 15000) // 15 second timeout

      // Check if navigation is allowed
      if (onBeforeNavigate) {
        const validatingState = loadingManagerRef.current.updateStage('validating')
        setLoadingState(validatingState)

        const startTime = Date.now()
        const canNavigate = await onBeforeNavigate()
        const responseTime = Date.now() - startTime
        
        // Detect network speed for better UX
        loadingManagerRef.current.detectNetworkSpeed(responseTime)
        
        if (!canNavigate) {
          console.log('Navigation blocked by onBeforeNavigate for:', navigationId)
          const errorState = loadingManagerRef.current.updateStage('error')
          setLoadingState(errorState)
          
          // Clear navigation state after short delay
          setTimeout(() => {
            setIsNavigating(false)
            setLoadingState(null)
            isNavigationActiveRef.current = false
          }, 1000)
          
          return false
        }
      }

      // Start navigation sequence
      const navigatingState = loadingManagerRef.current.updateStage('navigating')
      setLoadingState(navigatingState)
      onNavigationStart?.()

      console.log('Executing router.push for:', navigationId)
      
      // Start navigation with better error handling
      await router.push(path)

      // Trigger Next.js page prefetch for future navigations
      try {
        // @ts-ignore - router.prefetch exists but is not in the types
        await router.prefetch(path)
      } catch (e) {
        // Prefetch might fail in development, but that's okay
        console.debug('Prefetch failed:', e)
      }

      // Complete navigation
      const completeState = loadingManagerRef.current.forceComplete()
      setLoadingState(completeState)
      
      console.log('Navigation completed successfully for:', navigationId)
      
      // Auto-hide loading state after completion with shorter delay
      navigationTimeoutRef.current = setTimeout(() => {
        setIsNavigating(false)
        setLoadingState(null)
        isNavigationActiveRef.current = false
        onNavigationComplete?.()
      }, 300) // Shorter delay for better UX

      return true
    } catch (error) {
      console.error('Navigation failed for:', navigationId, error)
      
      // Set error state
      const errorState = loadingManagerRef.current.forceError()
      setLoadingState(errorState)
      
      // Clear navigation state after error display
      navigationTimeoutRef.current = setTimeout(() => {
        setIsNavigating(false)
        setLoadingState(null)
        isNavigationActiveRef.current = false
      }, 2000)
      
      throw error // Re-throw to let the login page handle the error
    } finally {
      // Clear timeout if navigation completed normally
      if (navigationTimeoutRef.current) {
        clearTimeout(navigationTimeoutRef.current)
        navigationTimeoutRef.current = null
      }
    }
  }, [router, onBeforeNavigate, onNavigationStart, onNavigationComplete])

  // Force reset function for stuck states
  const resetNavigation = useCallback(() => {
    console.log('Force resetting navigation state')
    
    if (navigationTimeoutRef.current) {
      clearTimeout(navigationTimeoutRef.current)
      navigationTimeoutRef.current = null
    }
    
    loadingManagerRef.current.reset()
    setLoadingState(null)
    setIsNavigating(false)
    isNavigationActiveRef.current = false
  }, [])

  return {
    isNavigating,
    navigate,
    loadingState,
    resetNavigation
  }
}
