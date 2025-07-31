"use client"

import type React from "react"

import { useState, useEffect } from "react"
import { usePageTransition } from "@/hooks/use-page-transition"
import { useAuth } from "@/lib/auth-context"
import { LoadingOverlay } from "@/components/ui/loading-overlay"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, CheckCircle, AlertCircle, Mail, RefreshCw, Lock, ArrowLeft } from "lucide-react"
import { motion } from "framer-motion"
import Link from "next/link"
import { useSearchParams } from "next/navigation"
import { useRouter } from "next/navigation"
import SixthvaultLogo from "@/components/SixthvaultLogo"

export default function VerifyPage() {
  const [email, setEmail] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isResending, setIsResending] = useState(false)
  const [error, setError] = useState("")
  const [success, setSuccess] = useState(false)
  const [resendMessage, setResendMessage] = useState("")

  const searchParams = useSearchParams()
  const router = useRouter()
  const { navigate, isNavigating } = usePageTransition()
  const { user } = useAuth()

  useEffect(() => {
    // Check URL parameters
    const token = searchParams.get("token")
    const emailParam = searchParams.get("email")

    // Only redirect if we don't have an email or if user is verified
    if (!emailParam || user?.verified) {
      router.replace('/login')
      return
    }

    // Set initial values if parameters are valid
    setEmail(emailParam)
    if (token) {
      setVerificationCode(token)
    }
  }, [searchParams, router, user])

  const handleVerify = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email.trim() || !verificationCode.trim()) {
      setError("Please enter both email and verification code")
      return
    }

    if (verificationCode.length !== 6) {
      setError("Verification code must be 6 characters")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      const response = await fetch("/api/auth/verify", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          email: email.trim(),
          verificationCode: verificationCode.trim().toUpperCase(),
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || "Verification failed")
        setVerificationCode("")  // Clear the code field on error
        return
      }

      setSuccess(true)
    } catch (error) {
      console.error("Verification error:", error)
      setError("Network error. Please check your connection and try again.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleResendCode = async () => {
    if (!email.trim()) {
      setError("Please enter your email address")
      return
    }

    setIsResending(true)
    setError("")
    setResendMessage("")

    try {
      const response = await fetch("/api/auth/resend-verification", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ email: email.trim() }),
      })

      const data = await response.json()

      if (!response.ok) {
        setError(data.error || "Failed to resend verification code")
        return
      }

      setResendMessage("New verification code sent to your email!")
    } catch (error) {
      console.error("Resend error:", error)
      setError("Failed to resend verification code. Please try again.")
    } finally {
      setIsResending(false)
    }
  }

  if (success) {
    return (
      <div className="min-h-screen bg-white flex items-center justify-center p-4 relative overflow-hidden">
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

        {/* Loading Overlay */}
        {isLoading && (
          <LoadingOverlay />
        )}

        <Card className="w-full max-w-md shadow-2xl bg-white/95 backdrop-blur-sm border-0">
          <CardHeader className="text-center">
            <div className="mx-auto mb-4">
              <SixthvaultLogo size="medium" />
            </div>
            <motion.div
              initial={{ scale: 0.5, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              transition={{ duration: 0.5, type: "spring" }}
            >
              <CardTitle className="text-2xl text-green-600 flex items-center justify-center">
                <CheckCircle className="w-6 h-6 mr-2" />
                Email Verified!
              </CardTitle>
            </motion.div>
            <CardDescription>Your account has been successfully verified</CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="text-center p-4 bg-green-50 rounded-lg">
              <p className="text-green-800 mb-2">Welcome to SIXTHVAULT! Your account is now active.</p>
              <p className="text-sm text-green-700">You can now sign in and access all AI-powered features.</p>
            </div>

            <motion.div
              initial={{ y: 20, opacity: 0 }}
              animate={{ y: 0, opacity: 1 }}
              transition={{ delay: 0.3, duration: 0.5 }}
            >
              <Button
                onClick={async () => {
                  setIsLoading(true)
                  try {
                    await navigate("/login")
                  } catch (error) {
                    setIsLoading(false)
                    console.error('Navigation failed:', error)
                  }
                }}
                className="w-full h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold shadow-lg transition-all duration-200 transform hover:scale-[1.02]"
                disabled={isLoading}
              >
                <Lock className="w-4 h-4 mr-2" />
                Continue to Sign In
              </Button>
            </motion.div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
      <div className="min-h-screen bg-white flex items-center justify-center p-4 relative overflow-hidden">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-verify" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient2-verify" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient3-verify" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            {/* Main flowing wave patterns */}
            <g stroke="url(#waveGradient1-verify)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
              <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
              <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
            </g>
            
            <g stroke="url(#waveGradient2-verify)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              <path d="M0,320 Q360,240 720,300 T1440,280"/>
              <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
              <path d="M0,360 Q420,280 840,340 T1440,320"/>
            </g>
            
            <g stroke="url(#waveGradient3-verify)" strokeWidth="1.0" fill="none" opacity="0.6">
              <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
              <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
              <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
            </g>
            
            {/* Filled wave areas for depth */}
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-verify)" opacity="0.1"/>
            <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-verify)" opacity="0.08"/>
            <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-verify)" opacity="0.06"/>
            
            {/* Additional flowing lines for complexity */}
            <g stroke="url(#waveGradient1-verify)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
              <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
            </g>
            
            <g stroke="url(#waveGradient2-verify)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
              <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
            </g>
          </svg>
        </div>

        {/* Loading Overlay */}
        {(isLoading || isNavigating) && (
          <LoadingOverlay />
        )}
        
        <Card className="w-full max-w-md shadow-2xl bg-white/95 backdrop-blur-sm border-0 relative z-10">
        <CardHeader className="pb-8">
          <div className="text-center space-y-2">
            <div className="mx-auto mb-6 transform hover:scale-105 transition-transform duration-200">
              <SixthvaultLogo size="large" />
            </div>
            <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
              Verify Your Email
            </CardTitle>
            <CardDescription className="text-slate-600 text-lg">
              Enter the verification code to get started
            </CardDescription>
          </div>
        </CardHeader>
        <CardContent className="space-y-6">
          <form onSubmit={handleVerify} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {resendMessage && (
              <Alert className="border-green-200 bg-green-50">
                <CheckCircle className="h-4 w-4 text-green-600" />
                <AlertDescription className="text-green-800">{resendMessage}</AlertDescription>
              </Alert>
            )}

            <div className="p-4 bg-blue-50 rounded-lg border border-blue-100">
              <div className="flex items-start space-x-3">
                <div className="p-2 bg-blue-100 rounded-full">
                  <Mail className="h-5 w-5 text-blue-600" />
                </div>
                <div className="flex-1">
                  <h4 className="font-semibold text-blue-900 mb-1">Check your inbox</h4>
                  <p className="text-blue-700 text-sm leading-relaxed">
                    We've sent a verification code to your email. It may take a few minutes to arrive. 
                    Don't forget to check your spam folder.
                  </p>
                </div>
              </div>
            </div>

            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-semibold text-slate-700">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                  <Input
                    id="email"
                    type="email"
                    value={email}
                    onChange={(e) => setEmail(e.target.value)}
                    required
                    disabled={isLoading}
                    placeholder="Enter your email"
                    className="pl-10 h-12 border-slate-200 bg-white/50 focus:bg-white focus:border-blue-500 focus:ring-blue-500 transition-colors"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="verificationCode" className="text-sm font-semibold text-slate-700">Verification Code</Label>
                <div className="relative">
                  <Lock className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                  <Input
                    id="verificationCode"
                    type="text"
                    value={verificationCode}
                    onChange={(e) => setVerificationCode(e.target.value.toUpperCase())}
                    required
                    disabled={isLoading}
                    placeholder="Enter 6-digit code"
                    className="pl-10 h-12 text-center text-lg tracking-widest font-mono border-slate-200 bg-white/50 focus:bg-white focus:border-blue-500 focus:ring-blue-500 transition-colors"
                    maxLength={6}
                  />
                </div>
                <p className="text-xs text-slate-500">The verification code expires in 24 hours</p>
              </div>
            </div>

            <Button 
              type="submit" 
              className="w-full h-12 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg transition-all duration-200 transform hover:scale-[1.02] rounded-lg"
              disabled={isLoading || verificationCode.length !== 6}
            >
              {isLoading ? (
                <>
                  <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                  Verifying...
                </>
              ) : (
                "Verify Email"
              )}
            </Button>

            <div className="relative">
              <div className="absolute inset-0 flex items-center">
                <span className="w-full border-t border-slate-200"></span>
              </div>
              <div className="relative flex justify-center text-xs uppercase">
                <span className="bg-white px-2 text-slate-500">or</span>
              </div>
            </div>

            <div className="space-y-4">
              <Button
                type="button"
                variant="outline"
                onClick={handleResendCode}
                disabled={isResending || !email.trim()}
                className="w-full h-11 border-slate-200 hover:bg-slate-50 text-slate-700 font-medium transition-colors"
              >
                {isResending ? (
                  <>
                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                    Sending new code...
                  </>
                ) : (
                  <>
                    <RefreshCw className="w-4 h-4 mr-2" />
                    Resend Verification Code
                  </>
                )}
              </Button>

              <div className="text-center">
                <Link 
                  href="/login" 
                  className="inline-flex items-center text-sm text-blue-600 hover:text-blue-700 font-medium"
                >
                  <ArrowLeft className="w-4 h-4 mr-1" />
                  Back to Sign In
                </Link>
              </div>
            </div>
          </form>
        </CardContent>
      </Card>
    </div>
  )
}
