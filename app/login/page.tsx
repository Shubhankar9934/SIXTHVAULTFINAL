"use client"

import { useState, useEffect, useRef } from 'react'
import { useRouter } from 'next/navigation'
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Eye, EyeOff, Shield, Lock, Mail, Building2, ArrowRight, ArrowLeft, AlertCircle, CheckCircle } from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { useAuth } from "@/lib/auth-context"

export default function LoginPage() {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [isUnverified, setIsUnverified] = useState(false)
  const [resendSuccess, setResendSuccess] = useState(false)
  const [loginSuccess, setLoginSuccess] = useState(false)
  const initRef = useRef(false)
  const loginAttemptRef = useRef(false)
  
  const { login, clearAuthState, isAuthenticated } = useAuth()
  const router = useRouter()
  
  // Single initialization effect - runs once on mount
  useEffect(() => {
    if (initRef.current) return
    initRef.current = true
    
    console.log('Login page: Initializing - clearing auth state')
    clearAuthState()
    
    // Clear any existing errors and states
    setError('')
    setIsUnverified(false)
    setResendSuccess(false)
    setLoginSuccess(false)
    setIsLoading(false)
  }, [clearAuthState])

  // Handle successful authentication - separate effect with proper cleanup
  useEffect(() => {
    if (isAuthenticated && loginSuccess && !loginAttemptRef.current) {
      console.log('Login page: Authentication confirmed, navigating to vault')
      
      // Set a flag to prevent multiple navigation attempts
      loginAttemptRef.current = true
      
      // Add a small delay to ensure auth state is fully propagated
      setTimeout(() => {
        // Use Next.js router for proper navigation
        router.push('/vault')
      }, 500)
    }
  }, [isAuthenticated, loginSuccess, router])

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    
    // Prevent double submission
    if (isLoading || loginAttemptRef.current) {
      console.warn('Login page: Submission blocked - already in progress')
      return
    }
    
    console.log('Login page: Starting login process')
    setIsLoading(true)
    setError('')
    setIsUnverified(false)
    setResendSuccess(false)
    setLoginSuccess(false)

    // Client-side validation
    if (!email.trim()) {
      setError('Email is required')
      setIsLoading(false)
      return
    }

    if (!password) {
      setError('Password is required')
      setIsLoading(false)
      return
    }

    // Basic email format validation
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email.trim())) {
      setError('Please enter a valid email address')
      setIsLoading(false)
      return
    }

    try {
      console.log('Login page: Calling login function')
      const success = await login(email.trim().toLowerCase(), password)
      
      if (success) {
        console.log('Login page: Login successful')
        setLoginSuccess(true)
        // Keep loading state - navigation will happen in useEffect
      } else {
        console.log('Login page: Login failed - no success response')
        setError('Login failed. Please check your credentials.')
        setIsLoading(false)
      }
    } catch (error: any) {
      console.error('Login page: Login error:', error)
      
      // Handle specific error types
      if (error.message.includes('verify') || error.message.includes('not verified')) {
        setIsUnverified(true)
        setError('Account not verified. Please verify your email before signing in.')
      } else if (error.message.includes('Invalid email or password')) {
        setError('Invalid email or password. Please check your credentials.')
      } else if (error.message.includes('Server error')) {
        setError('Server error. Please try again in a moment.')
      } else {
        setError(error instanceof Error ? error.message : 'Login failed. Please try again.')
      }
      
      setIsLoading(false)
      setLoginSuccess(false)
    }
  }

  const handleResendVerification = async () => {
    setIsLoading(true)
    setResendSuccess(false)
    try {
      const response = await fetch('/api/auth/resend-verification', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      if (response.ok) {
        setResendSuccess(true)
        setError('')
      } else {
        const data = await response.json()
        setError(data.error || 'Failed to resend verification code')
      }
    } catch (error) {
      setError('Failed to resend verification code. Please try again.')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-white flex items-center justify-center mobile-container relative overflow-hidden mobile-safe-top mobile-safe-bottom">
      {/* Beautiful flowing wave background */}
      <div className="absolute inset-0">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="waveGradient1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
              <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
              <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
            </linearGradient>
            <linearGradient id="waveGradient2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
              <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
              <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
            </linearGradient>
            <linearGradient id="waveGradient3" x1="50%" y1="0%" x2="50%" y2="100%">
              <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
              <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
              <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
            </linearGradient>
          </defs>
          
          {/* Main flowing wave patterns */}
          <g stroke="url(#waveGradient1)" strokeWidth="1.5" fill="none" opacity="0.8">
            <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
            <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
            <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
            <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
          </g>
          
          <g stroke="url(#waveGradient2)" strokeWidth="1.2" fill="none" opacity="0.7">
            <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
            <path d="M0,320 Q360,240 720,300 T1440,280"/>
            <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
            <path d="M0,360 Q420,280 840,340 T1440,320"/>
          </g>
          
          <g stroke="url(#waveGradient3)" strokeWidth="1.0" fill="none" opacity="0.6">
            <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
            <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
            <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
          </g>
          
          {/* Filled wave areas for depth */}
          <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1)" opacity="0.1"/>
          <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2)" opacity="0.08"/>
          <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3)" opacity="0.06"/>
          
          {/* Additional flowing lines for complexity */}
          <g stroke="url(#waveGradient1)" strokeWidth="0.8" fill="none" opacity="0.5">
            <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
            <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
          </g>
          
          <g stroke="url(#waveGradient2)" strokeWidth="0.6" fill="none" opacity="0.4">
            <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
            <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
          </g>
        </svg>
      </div>

      <div className="w-full max-w-md mx-auto relative z-10">
        <div className="flex justify-center mb-8 lg:hidden">
          <SixthvaultLogo size="large" />
        </div>

        <Card className="shadow-2xl bg-white/95 backdrop-blur-sm border-0 rounded-xl sm:rounded-2xl">
          <CardHeader className="space-y-4 pb-6 sm:pb-8 px-4 sm:px-6">
            <Link
              href="/"
              className="touch-target inline-flex items-center text-responsive-sm text-blue-600 hover:text-blue-500 transition-colors mb-2"
            >
              <ArrowLeft className="w-4 h-4 mr-1" />
              Back to Home
            </Link>
            <div className="text-center">
              <div className="hidden lg:flex lg:justify-center mb-2">
                <SixthvaultLogo size="large" />
              </div>
              <CardTitle className="text-responsive-xl font-bold text-slate-900">Sign In to Your Account</CardTitle>
              <CardDescription className="text-responsive-sm text-slate-600 mt-2">
                Access your secure document vault
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 sm:space-y-6 px-4 sm:px-6 pb-6 sm:pb-8">
            <form onSubmit={handleSubmit} className="space-y-5">
              {error && (
                <Alert variant="destructive" className="border-red-200 bg-red-50">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription className="text-red-800">{error}</AlertDescription>
                </Alert>
              )}

              {isUnverified && (
                <Alert className="border-blue-200 bg-blue-50">
                  <AlertCircle className="h-4 w-4 text-blue-600" />
                  <div className="flex flex-col space-y-2">
                    <AlertDescription className="text-blue-800">
                      Please verify your email to access your account.
                    </AlertDescription>
                    <Button 
                      type="button"
                      variant="outline"
                      className="w-full border-blue-200 hover:bg-blue-100 text-blue-700"
                      onClick={handleResendVerification}
                      disabled={isLoading}
                    >
                      {isLoading ? (
                        <>
                          <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                          Sending verification code...
                        </>
                      ) : (
                        'Resend Verification Code'
                      )}
                    </Button>
                  </div>
                </Alert>
              )}

              {resendSuccess && (
                <Alert className="border-green-200 bg-green-50">
                  <CheckCircle className="h-4 w-4 text-green-600" />
                  <AlertDescription className="text-green-800">
                    Verification code sent! Please check your email.
                  </AlertDescription>
                </Alert>
              )}

              <div className="space-y-2">
                <Label htmlFor="email" className="text-slate-700 font-medium text-responsive-sm">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={isLoading}
                    className="pl-10 h-12 sm:h-14 border-slate-200 focus:border-blue-500 focus:ring-blue-500 text-responsive-base rounded-lg"
                    autoComplete="email"
                    inputMode="email"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="password" className="text-slate-700 font-medium text-responsive-sm">Password</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                  <Input
                    id="password"
                    type={showPassword ? "text" : "password"}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    required
                    disabled={isLoading}
                    className="pl-10 pr-12 h-12 sm:h-14 border-slate-200 focus:border-blue-500 focus:ring-blue-500 text-responsive-base rounded-lg"
                    autoComplete="current-password"
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="touch-target absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                    onClick={() => setShowPassword(!showPassword)}
                    disabled={isLoading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-5 w-5 text-slate-400" />
                    ) : (
                      <Eye className="h-5 w-5 text-slate-400" />
                    )}
                  </Button>
                </div>
              </div>

              <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between space-y-3 sm:space-y-0">
                <div className="flex items-center space-x-2">
                  <input
                    id="remember"
                    type="checkbox"
                    className="h-5 w-5 sm:h-4 sm:w-4 text-blue-600 focus:ring-blue-500 border-slate-300 rounded"
                  />
                  <Label htmlFor="remember" className="text-responsive-sm text-slate-600">
                    Remember me
                  </Label>
                </div>
                <Link 
                  href="/forgot-password" 
                  className="touch-target text-responsive-sm text-blue-600 hover:text-blue-500 font-medium transition-colors"
                >
                  Forgot password?
                </Link>
              </div>

              <Button 
                type="submit" 
                className="touch-target w-full h-12 sm:h-14 bg-gray-100 hover:bg-gray-200 text-gray-800 font-semibold shadow-sm border border-gray-300 transition-all duration-200 text-responsive-base rounded-lg"
                disabled={isLoading}
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-2 h-5 w-5 animate-spin" />
                    {isAuthenticated ? 'Redirecting to Vault...' : 'Signing in...'}
                  </>
                ) : (
                  <>
                    Sign In
                    <ArrowRight className="ml-2 h-5 w-5" />
                  </>
                )}
              </Button>
            </form>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-slate-200" />
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 text-slate-500">New to SixthVault?</span>
              </div>
            </div>

            <div className="text-center">
              <Link 
                href="/register" 
                className="inline-flex items-center justify-center w-full h-12 px-4 py-2 border border-slate-300 rounded-md shadow-sm bg-white text-slate-700 hover:bg-slate-50 font-medium transition-colors"
              >
                Create Enterprise Account
                <ArrowRight className="ml-2 h-4 w-4" />
              </Link>
            </div>

          </CardContent>
        </Card>

      </div>
    </div>
  )
}
