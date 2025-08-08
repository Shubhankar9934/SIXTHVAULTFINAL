'use client'

import { useEffect } from "react"
import { useRouter } from "next/navigation"
import { useAuth } from "@/lib/auth-context"
import { Button } from "@/components/ui/button"
import { 
  Loader2, 
  Search,
  Cpu,
  Shield,
  BarChart3,
  Users,
  CheckCircle,
  Cloud,
  Database,
  Lock,
  Upload,
  Brain,
  Lightbulb,
  Tags,
  UserCheck,
  Settings,
  LogOut,
  TrendingUp,
  FileText,
  Calendar,
  Instagram,
  X
} from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"

export default function HomePage() {
  const { isAuthenticated, isLoading } = useAuth()
  const router = useRouter()

  useEffect(() => {
    if (!isLoading && isAuthenticated) {
      router.push("/vault")
    }
  }, [isAuthenticated, isLoading, router])

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600 text-lg">Loading SixthVault...</p>
        </div>
      </div>
    )
  }

  if (isAuthenticated) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-white">
        <div className="text-center">
          <Loader2 className="h-12 w-12 animate-spin mx-auto mb-4 text-blue-600" />
          <p className="text-gray-600 text-lg">Redirecting to vault...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white">
      {/* Header Navigation - Mobile Responsive */}
      <header className="relative overflow-hidden z-50 mobile-safe-top">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0 z-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 200" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-header" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
              </linearGradient>
              <linearGradient id="waveGradient2-header" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.5"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.3"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.15"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
              </linearGradient>
            </defs>
            
            {/* Header wave patterns */}
            <g stroke="url(#waveGradient1-header)" strokeWidth="1" fill="none" opacity="0.6">
              <path d="M0,50 Q200,30 400,45 T800,40 Q1000,35 1200,45 T1440,40"/>
              <path d="M0,70 Q240,50 480,65 T960,60 Q1200,55 1440,65"/>
            </g>
            
            <g stroke="url(#waveGradient2-header)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,90 Q300,70 600,85 T1200,80 Q1320,75 1440,85"/>
              <path d="M0,110 Q360,90 720,105 T1440,100"/>
            </g>
            
            {/* Filled areas for subtle depth */}
            <path d="M0,80 Q200,60 400,75 T800,70 Q1000,65 1200,75 T1440,70 L1440,200 L0,200 Z" fill="url(#waveGradient1-header)" opacity="0.05"/>
            <path d="M0,120 Q300,100 600,115 T1200,110 Q1320,105 1440,115 L1440,200 L0,200 Z" fill="url(#waveGradient2-header)" opacity="0.03"/>
          </svg>
        </div>
        <div className="relative z-10 max-w-7xl mx-auto mobile-container py-4">
          <div className="flex justify-end items-center">
            {/* Action Buttons - Mobile Responsive */}
            <div className="flex items-center gap-2 sm:gap-4 z-10 relative">
              <Link 
                href="/register"
                className="touch-target inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-responsive-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-3 sm:px-4 py-2 bg-gray-300 hover:bg-gray-400 active:bg-gray-500 active:scale-95 text-gray-600 cursor-pointer relative z-20"
              >
                <span className="hidden sm:inline">Get Started</span>
                <span className="sm:hidden">Join</span>
              </Link>
              <Link 
                href="/login"
                className="touch-target inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-responsive-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 sm:px-6 py-2 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 active:scale-95 text-white rounded-lg cursor-pointer relative z-20"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Section 1: Turning knowledge to live intelligence with Dashboard - Mobile Responsive */}
      <section className="py-10 sm:py-16 lg:py-20 relative overflow-hidden">
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

        <div className="relative max-w-6xl mx-auto mobile-container text-center">
          <div className="mb-6 sm:mb-8">
            {/* Logo positioned above the main heading - Mobile Responsive */}
            <div className="flex justify-center mb-4 sm:mb-6">
              <div className="relative inline-block">
                <div className="text-responsive-3xl md:text-responsive-4xl font-bold text-blue-600 opacity-0 pointer-events-none whitespace-nowrap pb-8 sm:pb-12">
                  Turning knowledge to
                </div>
                <div className="absolute inset-0 w-full h-full flex items-center justify-center">
                  <div className="w-[300px] h-[80px] sm:w-[600px] sm:h-[160px] lg:w-[900px] lg:h-[240px]">
                    <SixthvaultLogo size="full" />
                  </div>
                </div>
              </div>
            </div>
            
            <h2 className="text-responsive-3xl md:text-responsive-4xl font-bold text-blue-600 mb-4 sm:mb-6 leading-tight">
              Turning knowledge to<br />
              live intelligence
            </h2>
            
            <p className="text-responsive-base text-gray-600 mb-8 sm:mb-12 max-w-4xl mx-auto leading-relaxed px-4">
              <span className="text-blue-600 font-semibold">Sixthvault</span> is an AI-powered retrieval-augmented engine
              <span className="hidden sm:inline"><br /></span>
              <span className="sm:hidden"> </span>
              that instantly resurfaces actionable consumer insights from your project archive,
              <span className="hidden sm:inline"><br /></span>
              <span className="sm:hidden"> </span>
              transforming dormant research knowledge into real-time strategic advantage.
            </p>
          </div>
        </div>
      </section>

      {/* Section 2: Features Grid - Mobile Responsive */}
      <section className="py-10 sm:py-16 lg:py-20 relative overflow-hidden">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-s2" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient2-s2" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient3-s2" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            {/* Main flowing wave patterns */}
            <g stroke="url(#waveGradient1-s2)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
              <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
              <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
            </g>
            
            <g stroke="url(#waveGradient2-s2)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              <path d="M0,320 Q360,240 720,300 T1440,280"/>
              <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
              <path d="M0,360 Q420,280 840,340 T1440,320"/>
            </g>
            
            <g stroke="url(#waveGradient3-s2)" strokeWidth="1.0" fill="none" opacity="0.6">
              <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
              <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
              <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
            </g>
            
            {/* Filled wave areas for depth */}
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-s2)" opacity="0.1"/>
            <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-s2)" opacity="0.08"/>
            <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-s2)" opacity="0.06"/>
            
            {/* Additional flowing lines for complexity */}
            <g stroke="url(#waveGradient1-s2)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
              <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
            </g>
            
            <g stroke="url(#waveGradient2-s2)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
              <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
            </g>
          </svg>
        </div>
        <div className="relative max-w-6xl mx-auto mobile-container">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6 sm:gap-8 lg:gap-12">
            {/* AI-Powered Analysis */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <Cpu className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">AI-Powered Analysis</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Leverage advanced natural language processing to automatically extract key insights, themes, and actionable intelligence from your documents streamlining understanding and decision making.
              </p>
            </div>

            {/* Intelligent Search */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <Search className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">Intelligent Search</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Semantic search capabilities let you quickly find relevant information by interpreting natural language queries and delivering accurate results with source citations.
              </p>
            </div>

            {/* Smart Processing */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <Brain className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">Smart Processing</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Effortlessly organize, tag, and summarize documents using AI-powered comprehension and metadata extraction making document handling faster and smarter.
              </p>
            </div>

            {/* Enterprise Security */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <Shield className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">Enterprise Security</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Ensure your data is protected with bank-grade encryption, role-based access controls, and full compliance with industry standards for robust document security.
              </p>
            </div>

            {/* Analytics Dashboard */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <BarChart3 className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">Analytics Dashboard</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Gain a clear overview of platform usage with real-time metrics, collaboration patterns, and performance indicators all presented in a user-friendly dashboard.
              </p>
            </div>

            {/* Team Collaboration */}
            <div className="text-center bg-white/50 backdrop-blur-sm rounded-xl p-6 shadow-sm border border-white/20 hover:shadow-lg transition-all duration-300">
              <div className="w-12 h-12 sm:w-16 sm:h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-4 sm:mb-6 shadow-sm">
                <Users className="w-6 h-6 sm:w-8 sm:h-8 text-gray-700" />
              </div>
              <h3 className="text-responsive-lg font-bold text-gray-900 mb-3 sm:mb-4">Team Collaboration</h3>
              <p className="text-responsive-sm text-gray-600 leading-relaxed">
                Enable seamless teamwork with features that support sharing insights, adding annotations, and managing permissions all backed by real-time controls and audit trails.
              </p>
            </div>
          </div>
        </div>
      </section>


      {/* Footer - Mobile Responsive */}
      <footer className="relative overflow-hidden py-8 sm:py-12 border-t border-gray-100 mobile-safe-bottom">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 300" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-footer" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.4"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.3"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.15"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
              </linearGradient>
              <linearGradient id="waveGradient2-footer" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.3"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.2"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.15"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.1"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.05"/>
              </linearGradient>
            </defs>
            
            {/* Footer wave patterns */}
            <g stroke="url(#waveGradient1-footer)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,80 Q200,60 400,75 T800,70 Q1000,65 1200,75 T1440,70"/>
              <path d="M0,100 Q240,80 480,95 T960,90 Q1200,85 1440,95"/>
            </g>
            
            <g stroke="url(#waveGradient2-footer)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,120 Q300,100 600,115 T1200,110 Q1320,105 1440,115"/>
              <path d="M0,140 Q360,120 720,135 T1440,130"/>
            </g>
            
            {/* Subtle filled areas */}
            <path d="M0,110 Q200,90 400,105 T800,100 Q1000,95 1200,105 T1440,100 L1440,300 L0,300 Z" fill="url(#waveGradient1-footer)" opacity="0.03"/>
            <path d="M0,150 Q300,130 600,145 T1200,140 Q1320,135 1440,145 L1440,300 L0,300 Z" fill="url(#waveGradient2-footer)" opacity="0.02"/>
          </svg>
        </div>
        <div className="max-w-7xl mx-auto mobile-container relative z-10">
          {/* Main Footer Content */}
          <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-5 gap-6 sm:gap-8 lg:gap-10">
            {/* Logo - Full width on mobile, spans 2 columns on small screens */}
            <div className="col-span-1 sm:col-span-2 md:col-span-1 lg:col-span-1 flex flex-col items-center sm:items-start">
              <div className="mb-4">
                <SixthvaultLogo size="large" />
              </div>
              <div className="flex gap-3 mt-4">
                <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="touch-target w-8 h-8 bg-gradient-to-br from-pink-500 to-purple-600 rounded-lg flex items-center justify-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                  <Instagram className="w-4 h-4 text-white" />
                </a>
                <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="touch-target w-8 h-8 bg-gradient-to-br from-gray-800 to-black rounded-lg flex items-center justify-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                  <X className="w-4 h-4 text-white" />
                </a>
              </div>
            </div>

            {/* Product */}
            <div className="flex flex-col items-center sm:items-start">
              <h3 className="font-semibold text-gray-900 mb-4 text-responsive-sm">Product</h3>
              <ul className="space-y-2 text-gray-600 text-responsive-xs text-center sm:text-left">
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Features</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Updates</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">FAQ</a></li>
              </ul>
            </div>

            {/* Company */}
            <div className="flex flex-col items-center sm:items-start">
              <h3 className="font-semibold text-gray-900 mb-4 text-responsive-sm">Company</h3>
              <ul className="space-y-2 text-gray-600 text-responsive-xs text-center sm:text-left">
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">About</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Blog</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Careers</a></li>
              </ul>
            </div>

            {/* Resources */}
            <div className="flex flex-col items-center sm:items-start">
              <h3 className="font-semibold text-gray-900 mb-4 text-responsive-sm">Resources</h3>
              <ul className="space-y-2 text-gray-600 text-responsive-xs text-center sm:text-left">
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Examples</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Community</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Guides</a></li>
              </ul>
            </div>

            {/* Legal */}
            <div className="flex flex-col items-center sm:items-start">
              <h3 className="font-semibold text-gray-900 mb-4 text-responsive-sm">Legal</h3>
              <ul className="space-y-2 text-gray-600 text-responsive-xs text-center sm:text-left">
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Privacy</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Terms</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors touch-target block">Security</a></li>
              </ul>
            </div>
          </div>

          {/* Copyright Section */}
          <div className="mt-8 pt-6 border-t border-gray-200">
            <div className="text-center">
              <p className="text-gray-500 text-responsive-xs">
                © 2025 SixthVault. All rights reserved.
              </p>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
