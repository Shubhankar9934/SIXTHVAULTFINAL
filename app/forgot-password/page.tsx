"use client"

import { useState } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Mail, CheckCircle, ArrowLeft, Shield, KeyRound, Lock, RefreshCw } from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"

export default function ForgotPasswordPage() {
  const [email, setEmail] = useState("")
  const [verificationCode, setVerificationCode] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isResending, setIsResending] = useState(false)
  const [error, setError] = useState("")
  const [step, setStep] = useState<"email" | "verify">("email")
  const [resendMessage, setResendMessage] = useState("")
  
  const router = useRouter()

  const handleEmailSubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!email.trim()) {
      setError("Please enter your email address")
      return
    }

    if (!email.includes("@")) {
      setError("Please enter a valid email address")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      // Call the backend API for password reset verification code
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.message || 'Failed to send verification code')
      }

      setStep("verify")
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to send verification code. Please try again.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleVerifySubmit = async (e: React.FormEvent) => {
    e.preventDefault()

    if (!verificationCode.trim()) {
      setError("Please enter the verification code")
      return
    }

    if (verificationCode.length !== 6) {
      setError("Verification code must be 6 characters")
      return
    }

    setIsLoading(true)
    setError("")

    try {
      // Verify the code and proceed to reset password
      const response = await fetch('/api/auth/verify-reset-code', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          email: email.trim(),
          verificationCode: verificationCode.trim().toUpperCase()
        }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.message || 'Invalid verification code')
      }

      // Redirect to reset password page with token
      router.push(`/reset-password?token=${data.resetToken}&email=${encodeURIComponent(email)}`)
    } catch (error) {
      setError(error instanceof Error ? error.message : "Invalid verification code. Please try again.")
      setVerificationCode("")
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
      const response = await fetch('/api/auth/forgot-password', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ email }),
      })

      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.message || 'Failed to resend verification code')
      }

      setResendMessage("New verification code sent to your email!")
    } catch (error) {
      setError(error instanceof Error ? error.message : "Failed to resend verification code. Please try again.")
    } finally {
      setIsResending(false)
    }
  }

  if (step === "verify") {
    return (
      <div className="min-h-screen bg-white relative overflow-hidden">
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
        
        <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
          <Card className="w-full max-w-md shadow-2xl bg-white/95 backdrop-blur-sm border-0 relative z-10">
          <CardHeader className="pb-8">
            <div className="text-center space-y-2">
              <div className="mx-auto mb-6">
                <SixthvaultLogo size="large" />
              </div>
              <CardTitle className="text-3xl font-bold bg-gradient-to-r from-blue-600 to-indigo-600 bg-clip-text text-transparent">
                Verify Reset Code
              </CardTitle>
              <CardDescription className="text-slate-600 text-lg">
                Enter the verification code sent to your email
              </CardDescription>
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            <form onSubmit={handleVerifySubmit} className="space-y-4">
              {error && (
                <Alert variant="destructive">
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
                      We've sent a verification code to <strong>{email}</strong>. 
                      It may take a few minutes to arrive.
                    </p>
                  </div>
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
                <p className="text-xs text-slate-500">The verification code expires in 15 minutes</p>
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
                  "Verify & Continue"
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
                  disabled={isResending}
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

                <div className="text-center space-y-2">
                  <Button
                    type="button"
                    variant="ghost"
                    onClick={() => setStep("email")}
                    className="text-sm text-blue-600 hover:text-blue-700 font-medium"
                  >
                    <ArrowLeft className="w-4 h-4 mr-1" />
                    Change Email Address
                  </Button>
                  <div>
                    <Link 
                      href="/login" 
                      className="inline-flex items-center text-sm text-slate-600 hover:text-slate-700 font-medium"
                    >
                      <ArrowLeft className="w-4 h-4 mr-1" />
                      Back to Sign In
                    </Link>
                  </div>
                </div>
              </div>
            </form>
          </CardContent>
        </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Beautiful flowing wave background */}
      <div className="absolute inset-0">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="waveGradient1-email" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
              <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
              <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
            </linearGradient>
            <linearGradient id="waveGradient2-email" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
              <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
              <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
            </linearGradient>
            <linearGradient id="waveGradient3-email" x1="50%" y1="0%" x2="50%" y2="100%">
              <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
              <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
              <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
              <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
              <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
            </linearGradient>
          </defs>
          
          {/* Main flowing wave patterns */}
          <g stroke="url(#waveGradient1-email)" strokeWidth="1.5" fill="none" opacity="0.8">
            <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
            <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
            <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
            <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
          </g>
          
          <g stroke="url(#waveGradient2-email)" strokeWidth="1.2" fill="none" opacity="0.7">
            <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
            <path d="M0,320 Q360,240 720,300 T1440,280"/>
            <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
            <path d="M0,360 Q420,280 840,340 T1440,320"/>
          </g>
          
          <g stroke="url(#waveGradient3-email)" strokeWidth="1.0" fill="none" opacity="0.6">
            <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
            <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
            <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
          </g>
          
          {/* Filled wave areas for depth */}
          <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-email)" opacity="0.1"/>
          <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-email)" opacity="0.08"/>
          <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-email)" opacity="0.06"/>
          
          {/* Additional flowing lines for complexity */}
          <g stroke="url(#waveGradient1-email)" strokeWidth="0.8" fill="none" opacity="0.5">
            <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
            <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
          </g>
          
          <g stroke="url(#waveGradient2-email)" strokeWidth="0.6" fill="none" opacity="0.4">
            <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
            <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
          </g>
        </svg>
      </div>

      <div className="relative z-10 flex items-center justify-center min-h-screen p-4">
        <div className="w-full max-w-6xl grid lg:grid-cols-2 gap-8 items-center">
          {/* Left Side - Security Message */}
          <div className="hidden lg:block text-gray-900 space-y-8">
            <div className="space-y-6">
              <div className="flex items-center space-x-3">
                <Shield className="h-8 w-8 text-blue-600" />
                <span className="text-2xl font-bold">Secure Recovery</span>
              </div>
              <h1 className="text-5xl font-bold leading-tight">
                Reset Your Password
                <span className="bg-gradient-to-r from-blue-600 to-cyan-600 bg-clip-text text-transparent block">
                  Securely & Safely
                </span>
              </h1>
              <p className="text-xl text-gray-600 leading-relaxed">
                Our enterprise-grade security ensures your password reset process is completely secure. 
                We'll send you a time-limited, encrypted reset link.
              </p>
            </div>

            <div className="grid grid-cols-1 gap-4">
              <div className="flex items-center space-x-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
                <div className="p-2 bg-blue-100 rounded-lg">
                  <KeyRound className="h-6 w-6 text-blue-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Encrypted Reset Links</h3>
                  <p className="text-sm text-gray-600">Time-limited and single-use tokens</p>
                </div>
              </div>
              <div className="flex items-center space-x-4 p-4 bg-purple-50 rounded-lg border border-purple-100">
                <div className="p-2 bg-purple-100 rounded-lg">
                  <Mail className="h-6 w-6 text-purple-600" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">Email Verification</h3>
                  <p className="text-sm text-gray-600">Sent only to verified email addresses</p>
                </div>
              </div>
            </div>
          </div>

        {/* Right Side - Reset Form */}
        <div className="w-full max-w-md mx-auto lg:mx-0">
          <div className="text-center mb-8 lg:hidden">
            <SixthvaultLogo size="large" />
          </div>

          <Card className="shadow-2xl bg-white/95 backdrop-blur-sm border-0">
            <CardHeader className="space-y-4 pb-8">
              <div className="hidden lg:block text-center">
                <SixthvaultLogo size="medium" />
              </div>
              <div className="text-center">
                <CardTitle className="text-2xl font-bold text-slate-900">Reset Your Password</CardTitle>
                <CardDescription className="text-slate-600 mt-2">
                  Enter your email address and we'll send you a secure reset link
                </CardDescription>
              </div>
            </CardHeader>
            <CardContent className="space-y-6">
              <form onSubmit={handleEmailSubmit} className="space-y-5">
                {error && (
                  <Alert variant="destructive" className="border-red-200 bg-red-50">
                    <AlertDescription className="text-red-800">{error}</AlertDescription>
                  </Alert>
                )}

                <div className="space-y-2">
                  <Label htmlFor="email" className="text-slate-700 font-medium">Email Address</Label>
                  <div className="relative">
                    <Mail className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-slate-400" />
                    <Input
                      id="email"
                      type="email"
                      value={email}
                      onChange={(e) => setEmail(e.target.value)}
                      required
                      disabled={isLoading}
                      placeholder="Enter your email address"
                      className="pl-10 h-12 border-slate-200 focus:border-blue-500 focus:ring-blue-500"
                    />
                  </div>
                </div>

                <Button 
                  type="submit" 
                  className="w-full h-12 bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 text-white font-semibold shadow-lg transition-all duration-200 transform hover:scale-[1.02]" 
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <>
                      <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                      Sending Verification Code...
                    </>
                  ) : (
                    <>
                      <Mail className="w-4 h-4 mr-2" />
                      Send Verification Code
                    </>
                  )}
                </Button>

                <div className="text-center">
                  <Link 
                    href="/login" 
                    className="inline-flex items-center text-sm text-blue-600 hover:text-blue-500 font-medium transition-colors"
                  >
                    <ArrowLeft className="w-4 h-4 mr-1" />
                    Back to Sign In
                  </Link>
                </div>
              </form>

              <div className="mt-6 p-4 bg-slate-50 rounded-lg border border-slate-200">
                <div className="text-center">
                  <h4 className="text-sm font-semibold text-slate-900 mb-2">Security Notice</h4>
                  <p className="text-xs text-slate-600">
                    Verification codes expire in 15 minutes for your security. 
                    If you don't receive an email, check your spam folder.
                  </p>
                </div>
              </div>
            </CardContent>
          </Card>

          {/* Trust Indicators */}
          <div className="mt-8 text-center">
            <p className="text-sm text-slate-300 mb-4">Your security is our priority</p>
            <div className="flex items-center justify-center space-x-6 opacity-60">
              <div className="text-xs text-slate-400">256-bit Encryption</div>
              <div className="w-1 h-1 bg-slate-400 rounded-full"></div>
              <div className="text-xs text-slate-400">SOC 2 Compliant</div>
              <div className="w-1 h-1 bg-slate-400 rounded-full"></div>
              <div className="text-xs text-slate-400">GDPR Ready</div>
            </div>
          </div>
        </div>
        </div>
      </div>
    </div>
  )
}
