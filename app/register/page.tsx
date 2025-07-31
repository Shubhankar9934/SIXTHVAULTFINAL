"use client"

import { useState } from 'react'
import { useRouter } from 'next/navigation'
import { LoadingOverlay } from "@/components/ui/loading-overlay"
import { usePageTransition } from "@/hooks/use-page-transition"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Loader2, Eye, EyeOff, Shield, Users, Zap, CheckCircle, Building2, Mail, Lock, User, ArrowLeft } from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { useAuth } from "@/lib/auth-context"

export default function RegisterPage() {
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    confirmPassword: '',
    first_name: '',
    last_name: '',
    company: ''
  })
  const [showPassword, setShowPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState('')
  const [passwordStrength, setPasswordStrength] = useState(0)
  
  const { register } = useAuth()
  const router = useRouter()
  const { navigate, isNavigating } = usePageTransition()

  const calculatePasswordStrength = (password: string) => {
    let strength = 0
    if (password.length >= 8) strength += 1
    if (/[A-Z]/.test(password)) strength += 1
    if (/[a-z]/.test(password)) strength += 1
    if (/[0-9]/.test(password)) strength += 1
    if (/[^A-Za-z0-9]/.test(password)) strength += 1
    return strength
  }

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = e.target
    setFormData(prev => ({
      ...prev,
      [name]: value
    }))
    
    if (name === 'password') {
      setPasswordStrength(calculatePasswordStrength(value))
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    setIsLoading(true)
    setError('')

    // Validate passwords match
    if (formData.password !== formData.confirmPassword) {
      setError('Passwords do not match')
      setIsLoading(false)
      return
    }

    // Validate password strength
    if (formData.password.length < 8) {
      setError('Password must be at least 8 characters long')
      setIsLoading(false)
      return
    }

    try {
      const registrationResult = await register({
        email: formData.email,
        password: formData.password,
        first_name: formData.first_name,
        last_name: formData.last_name,
        company: formData.company || undefined
      })
      
      // After successful registration, use page transition to verify page
      // Don't clear loading state yet since we're transitioning
      let verifyUrl = `/verify?email=${encodeURIComponent(registrationResult.email)}`;
      if (registrationResult.verificationCode) {
        verifyUrl += `&token=${encodeURIComponent(registrationResult.verificationCode)}`;
      }
      
      await navigate(verifyUrl).catch((error) => {
        // Only clear loading if navigation fails
        setIsLoading(false)
        throw error
      })
    } catch (error) {
      setError(error instanceof Error ? error.message : 'Registration failed')
      setIsLoading(false)
    }
  }

  const getPasswordStrengthColor = () => {
    if (passwordStrength <= 2) return 'bg-red-500'
    if (passwordStrength <= 3) return 'bg-yellow-500'
    return 'bg-green-500'
  }

  const getPasswordStrengthText = () => {
    if (passwordStrength <= 2) return 'Weak'
    if (passwordStrength <= 3) return 'Medium'
    return 'Strong'
  }

  return (
    <div className="h-screen bg-white relative overflow-hidden">
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
        <LoadingOverlay 
          loadingState={{
            stage: isNavigating ? 'navigating' : 'validating',
            message: isNavigating ? "Redirecting to verification..." : "Creating your account...",
            progress: isNavigating ? 75 : 50,
            canRetry: false,
            showAlternatives: false,
            timeElapsed: 0,
            isStuck: false
          }}
        />
      )}

      <div className="relative z-10 flex h-screen">
        {/* Left Side - Features */}
        <div className="hidden lg:flex lg:w-1/2 p-6">
          <div className="w-full bg-white p-8 flex-col justify-center relative overflow-hidden rounded-2xl flex border border-gray-200 shadow-lg">
            {/* Wavy lines background for main container */}
            <div className="absolute inset-0">
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 600 800" preserveAspectRatio="xMidYMid slice">
                <defs>
                  <linearGradient id="containerWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
                    <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
                    <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
                    <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
                    <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.15"/>
                  </linearGradient>
                  <linearGradient id="containerWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.5"/>
                    <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.4"/>
                    <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.3"/>
                    <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.2"/>
                    <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.15"/>
                  </linearGradient>
                  <linearGradient id="containerWave3" x1="50%" y1="0%" x2="50%" y2="100%">
                    <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.4"/>
                    <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.3"/>
                    <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.25"/>
                    <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.2"/>
                    <stop offset="100%" stopColor="#ffa726" stopOpacity="0.15"/>
                  </linearGradient>
                </defs>
                
                {/* Main flowing wave patterns for container */}
                <g stroke="url(#containerWave1)" strokeWidth="1.5" fill="none" opacity="0.8">
                  <path d="M0,150 Q150,100 300,130 T600,120"/>
                  <path d="M0,180 Q180,130 360,160 T600,150"/>
                  <path d="M0,210 Q120,160 240,190 T480,180 Q540,175 600,185"/>
                  <path d="M0,240 Q200,190 400,220 T600,210"/>
                </g>
                
                <g stroke="url(#containerWave2)" strokeWidth="1.2" fill="none" opacity="0.7">
                  <path d="M0,300 Q200,250 400,280 T600,270"/>
                  <path d="M0,330 Q240,280 480,310 T600,300"/>
                  <path d="M0,360 Q160,310 320,340 T600,330"/>
                  <path d="M0,390 Q280,340 560,370 T600,360"/>
                </g>
                
                <g stroke="url(#containerWave3)" strokeWidth="1.0" fill="none" opacity="0.6">
                  <path d="M0,450 Q120,400 240,430 T480,420 Q540,415 600,425"/>
                  <path d="M0,480 Q180,430 360,460 T600,450"/>
                  <path d="M0,510 Q150,460 300,490 T600,480"/>
                  <path d="M0,540 Q220,490 440,520 T600,510"/>
                </g>
                
                {/* Filled wave areas for depth in container */}
                <path d="M0,200 Q150,150 300,180 T600,170 L600,800 L0,800 Z" fill="url(#containerWave1)" opacity="0.1"/>
                <path d="M0,350 Q200,300 400,330 T600,320 L600,800 L0,800 Z" fill="url(#containerWave2)" opacity="0.08"/>
                <path d="M0,500 Q180,450 360,480 T600,470 L600,800 L0,800 Z" fill="url(#containerWave3)" opacity="0.06"/>
                
                {/* Additional flowing lines for complexity in container */}
                <g stroke="url(#containerWave1)" strokeWidth="0.8" fill="none" opacity="0.5">
                  <path d="M0,120 Q80,70 160,100 T320,90 Q400,85 480,95 T600,85"/>
                  <path d="M0,600 Q100,550 200,580 T400,570 Q500,565 600,575"/>
                  <path d="M0,650 Q120,600 240,630 T480,620 Q540,615 600,625"/>
                  <path d="M0,700 Q160,650 320,680 T600,670"/>
                </g>
                
                <g stroke="url(#containerWave2)" strokeWidth="0.6" fill="none" opacity="0.4">
                  <path d="M0,100 Q60,50 120,80 T240,70 Q320,65 400,75 T560,65 Q580,63 600,68"/>
                  <path d="M0,620 Q90,570 180,600 T360,590 Q450,585 540,595 T600,585"/>
                  <path d="M0,670 Q110,620 220,650 T440,640 Q520,635 600,645"/>
                  <path d="M0,720 Q140,670 280,700 T600,690"/>
                </g>
              </svg>
            </div>
          
          <div className="max-w-lg mx-auto text-gray-900 relative z-10 space-y-6">
            {/* Header Block */}
            <div className="bg-gray-50 rounded-2xl p-8 border border-gray-200 shadow-lg relative overflow-hidden">
              {/* Wavy lines background inside block */}
              <div className="absolute inset-0">
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 300" preserveAspectRatio="xMidYMid slice">
                  <defs>
                    <linearGradient id="blockWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                      <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                    </linearGradient>
                    <linearGradient id="blockWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.3"/>
                      <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
                    </linearGradient>
                  </defs>
                  <g stroke="url(#blockWave1)" strokeWidth="1" fill="none" opacity="0.6">
                    <path d="M0,80 Q100,60 200,75 T400,70"/>
                    <path d="M0,100 Q120,80 240,95 T400,90"/>
                  </g>
                  <g stroke="url(#blockWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
                    <path d="M0,120 Q150,100 300,115 T400,110"/>
                    <path d="M0,140 Q180,120 360,135 T400,130"/>
                  </g>
                  <path d="M0,110 Q100,90 200,105 T400,100 L400,300 L0,300 Z" fill="url(#blockWave1)" opacity="0.05"/>
                </svg>
              </div>
              <div className="relative z-10 text-center">
                <div className="mb-6 mt-4 flex justify-center">
                  <SixthvaultLogo size="large" />
                </div>
                <h1 className="text-3xl font-bold mb-4 bg-gradient-to-r from-gray-900 via-blue-600 to-indigo-600 bg-clip-text text-transparent leading-tight">
                  Welcome to SixthVault
                </h1>
                <p className="text-base text-blue-600 mb-3 font-medium">
                  The Future of Document Intelligence
                </p>
                <p className="text-sm text-gray-600 leading-relaxed">
                  Start your journey with enterprise-grade document intelligence solutions
                </p>
              </div>
            </div>

            {/* Features Line */}
            <div className="text-center bg-gray-50 rounded-2xl p-6 border border-gray-200 shadow-lg relative overflow-hidden">
              {/* Wavy lines background inside features section */}
              <div className="absolute inset-0">
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 120" preserveAspectRatio="xMidYMid slice">
                  <defs>
                    <linearGradient id="featuresWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.4"/>
                      <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
                    </linearGradient>
                    <linearGradient id="featuresWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                      <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                    </linearGradient>
                  </defs>
                  <g stroke="url(#featuresWave1)" strokeWidth="1" fill="none" opacity="0.6">
                    <path d="M0,40 Q100,30 200,35 T400,30"/>
                    <path d="M0,50 Q120,40 240,45 T400,40"/>
                  </g>
                  <g stroke="url(#featuresWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
                    <path d="M0,60 Q150,50 300,55 T400,50"/>
                    <path d="M0,70 Q180,60 360,65 T400,60"/>
                  </g>
                  <path d="M0,55 Q100,45 200,50 T400,45 L400,120 L0,120 Z" fill="url(#featuresWave1)" opacity="0.04"/>
                </svg>
              </div>
              <div className="relative z-10">
                <p className="text-lg text-gray-700 font-medium">
                  AI-Powered Analysis • 99.9% Uptime Guarantee • Enterprise-Grade Security
                </p>
              </div>
            </div>

            {/* Quote Block */}
            <div className="bg-gray-50 rounded-2xl p-8 border border-gray-200 shadow-lg relative overflow-hidden">
              {/* Wavy lines background inside quote block */}
              <div className="absolute inset-0">
                <svg className="absolute inset-0 w-full h-full" viewBox="0 0 400 250" preserveAspectRatio="xMidYMid slice">
                  <defs>
                    <linearGradient id="quoteWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                      <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.4"/>
                      <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#ffa726" stopOpacity="0.1"/>
                    </linearGradient>
                    <linearGradient id="quoteWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                      <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                      <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                      <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                    </linearGradient>
                  </defs>
                  <g stroke="url(#quoteWave1)" strokeWidth="1" fill="none" opacity="0.6">
                    <path d="M0,70 Q100,50 200,65 T400,60"/>
                    <path d="M0,90 Q120,70 240,85 T400,80"/>
                  </g>
                  <g stroke="url(#quoteWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
                    <path d="M0,110 Q150,90 300,105 T400,100"/>
                    <path d="M0,130 Q180,110 360,125 T400,120"/>
                  </g>
                  <path d="M0,100 Q100,80 200,95 T400,90 L400,250 L0,250 Z" fill="url(#quoteWave1)" opacity="0.04"/>
                </svg>
              </div>
              <div className="relative z-10 text-center">
                <div className="flex justify-center space-x-2 mb-6">
                  <div className="w-3 h-3 bg-blue-400 rounded-full animate-pulse"></div>
                  <div className="w-3 h-3 bg-gray-400 rounded-full animate-pulse" style={{animationDelay: '0.5s'}}></div>
                  <div className="w-3 h-3 bg-indigo-400 rounded-full animate-pulse" style={{animationDelay: '1s'}}></div>
                </div>
                <blockquote className="text-gray-700 text-lg leading-relaxed italic font-medium mb-4">
                  "Transform your document workflow with cutting-edge AI technology and unlock unprecedented insights from your data"
                </blockquote>
                <div className="text-gray-600 text-sm font-medium">
                  — Enterprise Document Intelligence Platform
                </div>
              </div>
            </div>
          </div>
          </div>
        </div>

        {/* Right Side - Registration Form */}
        <div className="w-full lg:w-1/2 flex items-center justify-center p-4">
          <div className="w-full max-w-md">
            {/* Mobile Header */}
            <div className="lg:hidden mb-6">
              <div className="bg-white/90 backdrop-blur-md rounded-2xl p-6 border border-gray-200 shadow-xl">
                <div className="text-center">
                  <div className="mb-4">
                    <SixthvaultLogo size="medium" />
                  </div>
                  <h1 className="text-2xl font-bold text-gray-900 mb-2">Welcome to SixthVault</h1>
                  <p className="text-base text-blue-600 font-medium mb-1">The Future of Document Intelligence</p>
                  <p className="text-sm text-gray-600 leading-relaxed">
                    Start your journey with enterprise-grade document intelligence solutions
                  </p>
                </div>
              </div>
            </div>

            <Card className="shadow-2xl border-0 bg-white/80 backdrop-blur-sm">
              <CardHeader className="text-center pb-3">
                <Link
                  href="/"
                  className="inline-flex items-center text-sm text-blue-600 hover:text-blue-500 transition-colors mb-2"
                >
                  <ArrowLeft className="w-4 h-4 mr-1" />
                  Back to Home
                </Link>
                <CardTitle className="text-xl font-bold bg-gradient-to-r from-slate-900 to-slate-700 bg-clip-text text-transparent">
                  Create Your Account
                </CardTitle>
                <CardDescription className="text-sm text-slate-600">
                  Start your journey with enterprise-grade document intelligence
                </CardDescription>
              </CardHeader>
              
              <CardContent className="space-y-3">
                <form onSubmit={handleSubmit} className="space-y-3">
                  {error && (
                    <Alert variant="destructive" className="border-red-200 bg-red-50">
                      <AlertDescription className="text-red-800">{error}</AlertDescription>
                    </Alert>
                  )}

                  {/* Name Fields */}
                  <div className="grid grid-cols-2 gap-3">
                    <div className="space-y-1">
                      <Label htmlFor="first_name" className="text-xs font-medium text-slate-700">
                        First Name
                      </Label>
                      <div className="relative">
                        <User className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                        <Input
                          id="first_name"
                          name="first_name"
                          type="text"
                          placeholder="John"
                          value={formData.first_name}
                          onChange={handleChange}
                          required
                          disabled={isLoading}
                          className="pl-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                        />
                      </div>
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="last_name" className="text-xs font-medium text-slate-700">
                        Last Name
                      </Label>
                      <div className="relative">
                        <User className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                        <Input
                          id="last_name"
                          name="last_name"
                          type="text"
                          placeholder="Doe"
                          value={formData.last_name}
                          onChange={handleChange}
                          required
                          disabled={isLoading}
                          className="pl-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                        />
                      </div>
                    </div>
                  </div>

                  {/* Email Field */}
                  <div className="space-y-1">
                    <Label htmlFor="email" className="text-xs font-medium text-slate-700">
                      Business Email
                    </Label>
                    <div className="relative">
                      <Mail className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                      <Input
                        id="email"
                        name="email"
                        type="email"
                        placeholder="john@company.com"
                        value={formData.email}
                        onChange={handleChange}
                        required
                        disabled={isLoading}
                        className="pl-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                      />
                    </div>
                  </div>

                  {/* Company Field */}
                  <div className="space-y-1">
                    <Label htmlFor="company" className="text-xs font-medium text-slate-700">
                      Company <span className="text-slate-400">(Optional)</span>
                    </Label>
                    <div className="relative">
                      <Building2 className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                      <Input
                        id="company"
                        name="company"
                        type="text"
                        placeholder="Your Company"
                        value={formData.company}
                        onChange={handleChange}
                        disabled={isLoading}
                        className="pl-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                      />
                    </div>
                  </div>

                  {/* Password Field */}
                  <div className="space-y-1">
                    <Label htmlFor="password" className="text-xs font-medium text-slate-700">
                      Password
                    </Label>
                    <div className="relative">
                      <Lock className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                      <Input
                        id="password"
                        name="password"
                        type={showPassword ? "text" : "password"}
                        placeholder="Create a strong password"
                        value={formData.password}
                        onChange={handleChange}
                        required
                        disabled={isLoading}
                        className="pl-8 pr-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-8 px-2 hover:bg-transparent"
                        onClick={() => setShowPassword(!showPassword)}
                        disabled={isLoading}
                      >
                        {showPassword ? (
                          <EyeOff className="h-3 w-3 text-slate-400" />
                        ) : (
                          <Eye className="h-3 w-3 text-slate-400" />
                        )}
                      </Button>
                    </div>
                    
                    {/* Password Strength Indicator */}
                    {formData.password && (
                      <div className="space-y-1">
                        <div className="flex justify-between text-xs">
                          <span className="text-slate-500">Strength</span>
                          <span className={`font-medium ${passwordStrength <= 2 ? 'text-red-500' : passwordStrength <= 3 ? 'text-yellow-500' : 'text-green-500'}`}>
                            {getPasswordStrengthText()}
                          </span>
                        </div>
                        <div className="w-full bg-slate-200 rounded-full h-1">
                          <div 
                            className={`h-1 rounded-full transition-all duration-300 ${getPasswordStrengthColor()}`}
                            style={{ width: `${(passwordStrength / 5) * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    )}
                  </div>

                  {/* Confirm Password Field */}
                  <div className="space-y-1">
                    <Label htmlFor="confirmPassword" className="text-xs font-medium text-slate-700">
                      Confirm Password
                    </Label>
                    <div className="relative">
                      <Lock className="absolute left-2 top-2.5 h-3 w-3 text-slate-400" />
                      <Input
                        id="confirmPassword"
                        name="confirmPassword"
                        type={showConfirmPassword ? "text" : "password"}
                        placeholder="Confirm your password"
                        value={formData.confirmPassword}
                        onChange={handleChange}
                        required
                        disabled={isLoading}
                        className="pl-8 pr-8 h-8 text-sm border-slate-200 focus:border-blue-500 focus:ring-blue-500/20"
                      />
                      <Button
                        type="button"
                        variant="ghost"
                        size="sm"
                        className="absolute right-0 top-0 h-8 px-2 hover:bg-transparent"
                        onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                        disabled={isLoading}
                      >
                        {showConfirmPassword ? (
                          <EyeOff className="h-3 w-3 text-slate-400" />
                        ) : (
                          <Eye className="h-3 w-3 text-slate-400" />
                        )}
                      </Button>
                    </div>
                  </div>

                  {/* Submit Button */}
                  <Button 
                    type="submit" 
                    className="w-full h-10 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-medium shadow-lg hover:shadow-xl transition-all duration-200 transform hover:scale-[1.02]"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="mr-2 h-3 w-3 animate-spin" />
                        Creating account...
                      </>
                    ) : (
                      'Create Account'
                    )}
                  </Button>

                  {/* Terms */}
                  <p className="text-xs text-slate-500 text-center leading-tight">
                    By creating an account, you agree to our{' '}
                    <Link href="/terms" className="text-blue-600 hover:text-blue-500 font-medium underline">
                      Terms
                    </Link>{' '}
                    and{' '}
                    <Link href="/privacy" className="text-blue-600 hover:text-blue-500 font-medium underline">
                      Privacy Policy
                    </Link>
                  </p>
                </form>

                {/* Sign In Link */}
                <div className="pt-3 border-t border-slate-200">
                  <p className="text-center text-slate-600 text-sm">
                    Already have an account?{' '}
                    <Link 
                      href="/login" 
                      className="text-blue-600 hover:text-blue-500 font-medium transition-colors"
                    >
                      Sign in
                    </Link>
                  </p>
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>

      <style jsx>{`
        @keyframes blob {
          0% {
            transform: translate(0px, 0px) scale(1);
          }
          33% {
            transform: translate(30px, -50px) scale(1.1);
          }
          66% {
            transform: translate(-20px, 20px) scale(0.9);
          }
          100% {
            transform: translate(0px, 0px) scale(1);
          }
        }
        .animate-blob {
          animation: blob 7s infinite;
        }
        .animation-delay-2000 {
          animation-delay: 2s;
        }
        .animation-delay-4000 {
          animation-delay: 4s;
        }
        .bg-grid-slate-100 {
          background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='32' height='32' fill='none' stroke='rgb(148 163 184 / 0.05)'%3e%3cpath d='m0 .5h32m-32 32v-32'/%3e%3c/svg%3e");
        }
        .bg-grid-white\/\[0\.05\] {
          background-image: url("data:image/svg+xml,%3csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32' width='32' height='32' fill='none' stroke='rgb(255 255 255 / 0.05)'%3e%3cpath d='m0 .5h32m-32 32v-32'/%3e%3c/svg%3e");
        }
      `}</style>
    </div>
  )
}
