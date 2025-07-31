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
      {/* Header Navigation */}
      <header className="relative overflow-hidden z-50">
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
        <div className="relative z-10 max-w-7xl mx-auto px-6 py-4">
          <div className="flex justify-between items-center">
            {/* Logo */}
            <div className="flex items-center z-10">
              <SixthvaultLogo size="large" />
            </div>

            {/* Action Buttons */}
            <div className="flex items-center gap-4 z-10 relative">
              <Link 
                href="/register"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-4 py-2 bg-gray-300 hover:bg-gray-400 active:bg-gray-500 active:scale-95 text-gray-600 cursor-pointer relative z-20"
              >
                Get Started
              </Link>
              <Link 
                href="/login"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 h-10 px-6 py-2 bg-blue-600 hover:bg-blue-700 active:bg-blue-800 active:scale-95 text-white rounded-lg cursor-pointer relative z-20"
              >
                Sign In
              </Link>
            </div>
          </div>
        </div>
      </header>

      {/* Section 1: Turning knowledge to live intelligence with Dashboard */}
      <section className="py-20 relative overflow-hidden">
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

        <div className="relative max-w-6xl mx-auto px-6 text-center">
          <div className="mb-8">
            <div className="inline-block bg-blue-100 text-blue-800 px-4 py-2 rounded-full text-sm font-medium mb-6">
              Enterprise AI Documents intelligence
            </div>
            
            <h2 className="text-4xl md:text-5xl font-bold text-blue-600 mb-6">
              Turning knowledge to<br />
              live intelligence
            </h2>
            
            <p className="text-lg text-gray-600 mb-8 max-w-3xl mx-auto">
              <span className="text-blue-600 font-semibold">Sixthvault</span> is an AI-powered retrieval-augmented engine<br />
              that instantly resurfaces actionable consumer insights from your project archive,<br />
              transforming dormant research knowledge into real-time strategic advantage.
            </p>

            <div className="flex gap-4 justify-center mb-8">
              <Link 
                href="/register"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-gray-900 text-white hover:bg-gray-800 px-8 py-3 text-lg rounded-lg cursor-pointer"
              >
                Start Free Trial
              </Link>
              <Link 
                href="/login"
                className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-blue-600 text-white hover:bg-blue-700 px-8 py-3 text-lg rounded-lg cursor-pointer"
              >
                Watch demo
              </Link>
            </div>

            <div className="flex items-center justify-center gap-8 text-sm text-gray-600 mb-12">
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                no credit card required
              </div>
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                60-day free trial
              </div>
              <div className="flex items-center">
                <CheckCircle className="w-4 h-4 mr-2 text-green-500" />
                Enterprise security
              </div>
            </div>
          </div>

          {/* Exact Dashboard Mockup - Matching Your Documents Page */}
          <div className="bg-white rounded-2xl shadow-xl border border-gray-200 max-w-7xl mx-auto overflow-hidden">
            {/* Top Header - Exact Match */}
            <div className="bg-white border-b border-gray-200 p-6">
              <div className="flex items-center justify-between mb-6">
                <div className="flex items-center space-x-3">
                  <div className="w-12 h-12 bg-blue-600 rounded-xl flex items-center justify-center">
                    <Database className="w-6 h-6 text-white" />
                  </div>
                  <div>
                    <h1 className="text-2xl font-bold text-gray-900">Document Intelligence Hub</h1>
                    <p className="text-sm text-gray-500">Enterprise Document Management & AI Analysis</p>
                  </div>
                </div>
                <div className="flex items-center space-x-4">
                  <div className="flex items-center space-x-2 bg-green-50 px-3 py-2 rounded-lg">
                    <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                    <span className="text-sm font-medium text-green-700">AI Analysis Active</span>
                  </div>
                  <button className="p-2 text-gray-400 hover:text-gray-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 17h5l-5 5v-5z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 19.5A2.5 2.5 0 016.5 17H20" />
                    </svg>
                  </button>
                  <button className="p-2 text-gray-400 hover:text-gray-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                  </button>
                  <button className="p-2 text-gray-400 hover:text-gray-600">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h7" />
                    </svg>
                  </button>
                </div>
              </div>

              {/* Search and Filters Bar - Exact Match */}
              <div className="flex items-center space-x-4">
                <div className="flex items-center bg-gray-50 rounded-lg px-3 py-2 flex-1 max-w-md">
                  <Search className="w-4 h-4 text-gray-400 mr-2" />
                  <input 
                    type="text" 
                    placeholder="Search" 
                    className="bg-transparent border-none outline-none text-sm text-gray-600 flex-1"
                    readOnly
                  />
                  <svg className="w-4 h-4 text-gray-400 ml-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 4a1 1 0 011-1h16a1 1 0 011 1v2.586a1 1 0 01-.293.707l-6.414 6.414a1 1 0 00-.293.707V17l-4 4v-6.586a1 1 0 00-.293-.707L3.293 7.293A1 1 0 013 6.586V4z" />
                  </svg>
                </div>
                <select className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-600">
                  <option>All Status</option>
                </select>
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                </button>
                <select className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-600">
                  <option>Google Gemini</option>
                </select>
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                  </svg>
                </button>
                <select className="bg-gray-50 border border-gray-200 rounded-lg px-3 py-2 text-sm text-gray-600">
                  <option>Gemini 1.5 Flash</option>
                </select>
                <div className="bg-blue-50 text-blue-700 px-3 py-2 rounded-lg text-sm font-medium">
                  Using: Google Gemini / Gemini 1.5 Flash
                </div>
                <button className="p-2 text-gray-400 hover:text-gray-600">
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15" />
                  </svg>
                </button>
              </div>
            </div>

            {/* Main Layout - Exact Match */}
            <div className="flex">
              {/* Sidebar - Exact Match */}
              <div className="w-80 bg-gray-50 border-r border-gray-200 p-6">
                {/* SAXTHVAULT Logo */}
                <div className="mb-8">
                  <SixthvaultLogo size="medium" />
                </div>
                
                {/* Navigation Menu */}
                <div className="space-y-1 mb-8">
                  <div className="bg-blue-100 text-blue-700 px-4 py-3 rounded-lg flex items-center font-medium">
                    <BarChart3 className="w-4 h-4 mr-3" />
                    Dashboard
                    <svg className="w-4 h-4 ml-auto" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                  </div>
                  <div className="text-gray-600 px-4 py-3 rounded-lg hover:bg-gray-100 cursor-pointer flex items-center">
                    <Upload className="w-4 h-4 mr-3" />
                    Upload & Process
                  </div>
                  <div className="text-gray-600 px-4 py-3 rounded-lg hover:bg-gray-100 cursor-pointer flex items-center">
                    <Brain className="w-4 h-4 mr-3" />
                    AI Analysis
                  </div>
                  <div className="text-gray-600 px-4 py-3 rounded-lg hover:bg-gray-100 cursor-pointer flex items-center">
                    <Lightbulb className="w-4 h-4 mr-3" />
                    Insights
                  </div>
                  <div className="text-gray-600 px-4 py-3 rounded-lg hover:bg-gray-100 cursor-pointer flex items-center">
                    <Tags className="w-4 h-4 mr-3" />
                    Manage Tags
                  </div>
                  <div className="text-gray-600 px-4 py-3 rounded-lg hover:bg-gray-100 cursor-pointer flex items-center">
                    <UserCheck className="w-4 h-4 mr-3" />
                    Demographics
                  </div>
                </div>

                {/* Quick Stats Card - Ultra Compact Professional Design */}
                <div className="mt-8">
                  <div className="bg-white border border-gray-200 rounded-lg p-3 shadow-sm">
                    <h3 className="text-xs font-semibold text-gray-800 mb-2">Quick Stats</h3>
                    <div className="space-y-2">
                      <div className="text-center">
                        <div className="text-xl font-bold text-blue-600">3</div>
                        <div className="text-xs text-gray-500 -mt-1">Processed</div>
                      </div>
                      <div className="text-center">
                        <div className="text-base font-bold text-green-600">230.27 KB</div>
                        <div className="text-xs text-gray-500 -mt-1">Total Size</div>
                      </div>
                      <div className="pt-1 border-t border-gray-200">
                        <div className="flex justify-between text-xs">
                          <span className="text-gray-500">Languages:</span>
                          <span className="font-medium text-gray-700">1</span>
                        </div>
                      </div>
                    </div>
                  </div>
                  
                  {/* Back to Vault Button - Non-clickable demo */}
                  <div className="mt-4">
                    <div className="w-full bg-blue-600 hover:bg-blue-700 text-white py-2 px-4 rounded-lg font-medium text-center cursor-default transition-colors duration-200">
                      Back to Vault
                    </div>
                  </div>
                </div>
              </div>

              {/* Main Content Area - Exact Match */}
              <div className="flex-1 p-6">
                {/* Stats Cards - Compact Professional Design */}
                <div className="grid grid-cols-4 gap-4 mb-8">
                  <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-blue-600 text-xs font-medium mb-1">Total Documents</p>
                        <p className="text-2xl font-bold text-blue-900">3</p>
                      </div>
                      <div className="w-10 h-10 bg-blue-500 rounded-lg flex items-center justify-center">
                        <FileText className="w-5 h-5 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-green-50 border border-green-200 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-green-600 text-xs font-medium mb-1">Processed</p>
                        <p className="text-2xl font-bold text-green-900">3</p>
                      </div>
                      <div className="w-10 h-10 bg-green-500 rounded-lg flex items-center justify-center">
                        <CheckCircle className="w-5 h-5 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-purple-50 border border-purple-200 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-purple-600 text-xs font-medium mb-1">Total Size</p>
                        <p className="text-xl font-bold text-purple-900">230.27 KB</p>
                      </div>
                      <div className="w-10 h-10 bg-purple-500 rounded-lg flex items-center justify-center">
                        <Database className="w-5 h-5 text-white" />
                      </div>
                    </div>
                  </div>

                  <div className="bg-orange-50 border border-orange-200 rounded-lg p-4">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="text-orange-600 text-xs font-medium mb-1">Languages</p>
                        <p className="text-2xl font-bold text-orange-900">1</p>
                      </div>
                      <div className="w-10 h-10 bg-orange-500 rounded-lg flex items-center justify-center">
                        <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 5h12M9 3v2m1.048 9.5A18.022 18.022 0 016.412 9m6.088 9h7M11 21l5-10 5 10M12.751 5C11.783 10.77 8.07 15.61 3 18.129" />
                        </svg>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Documents Section */}
                <div className="mb-6">
                  <div className="flex items-center space-x-2 mb-2">
                    <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 10V3L4 14h7v7l9-11h-7z" />
                    </svg>
                    <h2 className="text-xl font-bold text-gray-900">Recent Documents</h2>
                  </div>
                  <p className="text-gray-500 text-sm">Latest processed documents with AI analysis</p>
                </div>

                {/* Document List - Using Actual Filenames */}
                <div className="space-y-3">
                  <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Swaad_Teausers_Pune_Female_25-35_NCCSA1.docx</h4>
                        <div className="flex items-center space-x-2 text-sm text-gray-500">
                          <span>83.84 KB</span>
                          <span>•</span>
                          <span>2025-07-25</span>
                          <span>•</span>
                          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">English</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Gypsy_Teauser_Nizamabad_Female_36-50.docx</h4>
                        <div className="flex items-center space-x-2 text-sm text-gray-500">
                          <span>111.31 KB</span>
                          <span>•</span>
                          <span>2025-07-25</span>
                          <span>•</span>
                          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">English</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>
                    </div>
                  </div>

                  <div className="flex items-center justify-between p-4 border border-gray-200 rounded-lg hover:bg-gray-50">
                    <div className="flex items-center space-x-3">
                      <div className="w-8 h-8 bg-blue-100 rounded flex items-center justify-center">
                        <FileText className="w-4 h-4 text-blue-600" />
                      </div>
                      <div>
                        <h4 className="font-medium text-gray-900">Agni2_Teauser_Kanpur_Female_25-30_NCCSB.docx</h4>
                        <div className="flex items-center space-x-2 text-sm text-gray-500">
                          <span>35.12 KB</span>
                          <span>•</span>
                          <span>2025-07-25</span>
                          <span>•</span>
                          <span className="bg-gray-100 text-gray-700 px-2 py-1 rounded text-xs">English</span>
                        </div>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <CheckCircle className="w-5 h-5 text-green-500" />
                      <button className="p-1 text-gray-400 hover:text-gray-600">
                        <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 616 0z" />
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                        </svg>
                      </button>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Section 2: Features Grid */}
      <section className="py-20 relative overflow-hidden">
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
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="grid grid-cols-3 gap-12">
            {/* AI-Powered Analysis */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Cpu className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">AI-Powered Analysis</h3>
              <p className="text-gray-600 leading-relaxed">
                Leverage advanced natural language processing to automatically extract key insights, themes, and actionable intelligence from your documents streamlining understanding and decision making.
              </p>
            </div>

            {/* Intelligent Search */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Search className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Intelligent Search</h3>
              <p className="text-gray-600 leading-relaxed">
                Semantic search capabilities let you quickly find relevant information by interpreting natural language queries and delivering accurate results with source citations.
              </p>
            </div>

            {/* Smart Processing */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Cpu className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Smart Processing</h3>
              <p className="text-gray-600 leading-relaxed">
                Effortlessly organize, tag, and summarize documents using AI-powered comprehension and metadata extraction making document handling faster and smarter.
              </p>
            </div>

            {/* Enterprise Security */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Shield className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Enterprise Security</h3>
              <p className="text-gray-600 leading-relaxed">
                Ensure your data is protected with bank-grade encryption, role-based access controls, and full compliance with industry standards for robust document security
              </p>
            </div>

            {/* Analytics Dashboard */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <BarChart3 className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Analytics Dashboard</h3>
              <p className="text-gray-600 leading-relaxed">
                Gain a clear overview of platform usage with real-time metrics, collaboration patterns, and performance indicators all presented in a user-friendly dashboard.
              </p>
            </div>

            {/* Team Collaboration */}
            <div className="text-center">
              <div className="w-16 h-16 bg-white rounded-full flex items-center justify-center mx-auto mb-6 shadow-sm">
                <Users className="w-8 h-8 text-gray-700" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-4">Team Collaboration</h3>
              <p className="text-gray-600 leading-relaxed">
                Enable seamless teamwork with features that support sharing insights, adding annotations, and managing permissions all backed by real-time controls and audit trails.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Section 3: Built on Enterprise-Grade Technology */}
      <section className="py-24 relative overflow-hidden">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-s3" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient2-s3" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient3-s3" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            {/* Main flowing wave patterns */}
            <g stroke="url(#waveGradient1-s3)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
              <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
              <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
            </g>
            
            <g stroke="url(#waveGradient2-s3)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              <path d="M0,320 Q360,240 720,300 T1440,280"/>
              <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
              <path d="M0,360 Q420,280 840,340 T1440,320"/>
            </g>
            
            <g stroke="url(#waveGradient3-s3)" strokeWidth="1.0" fill="none" opacity="0.6">
              <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
              <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
              <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
            </g>
            
            {/* Filled wave areas for depth */}
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-s3)" opacity="0.1"/>
            <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-s3)" opacity="0.08"/>
            <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-s3)" opacity="0.06"/>
            
            {/* Additional flowing lines for complexity */}
            <g stroke="url(#waveGradient1-s3)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
              <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
            </g>
            
            <g stroke="url(#waveGradient2-s3)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
              <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
            </g>
          </svg>
        </div>
        <div className="relative max-w-7xl mx-auto px-6">
          <div className="text-center mb-20">
            <div className="flex items-center justify-center mb-8">
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-400 to-transparent max-w-md"></div>
              <div className="mx-8">
                <div className="inline-block bg-blue-50 text-blue-700 px-4 py-2 rounded-full text-sm font-medium mb-4">
                  Enterprise Foundation
                </div>
                <h2 className="text-4xl md:text-5xl font-bold text-gray-900 whitespace-nowrap">
                  Built on Enterprise-Grade Technology
                </h2>
              </div>
              <div className="flex-1 h-px bg-gradient-to-r from-transparent via-gray-400 to-transparent max-w-md"></div>
            </div>
            <p className="text-lg text-gray-600 max-w-3xl mx-auto">
              Powered by cutting-edge infrastructure and security protocols that Fortune 500 companies trust
            </p>
          </div>

          <div className="grid grid-cols-4 gap-8 max-w-6xl mx-auto">
            {/* Advanced AI Models */}
            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-br from-blue-500 to-purple-600 rounded-2xl opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              <div className="relative text-center p-8 bg-white border border-gray-200 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                <div className="w-16 h-16 bg-gradient-to-br from-blue-500 to-blue-600 rounded-xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <Cpu className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 text-lg mb-2">Advanced AI Models</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  State-of-the-art language models with enterprise-grade performance and reliability
                </p>
              </div>
            </div>

            {/* Cloud Infrastructure */}
            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-br from-green-500 to-teal-600 rounded-2xl opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              <div className="relative text-center p-8 bg-white border border-gray-200 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                <div className="w-16 h-16 bg-gradient-to-br from-green-500 to-green-600 rounded-xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <Cloud className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 text-lg mb-2">Cloud Infrastructure</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  Scalable, secure cloud architecture with 99.9% uptime and global availability
                </p>
              </div>
            </div>

            {/* Vector Database */}
            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-br from-purple-500 to-pink-600 rounded-2xl opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              <div className="relative text-center p-8 bg-white border border-gray-200 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                <div className="w-16 h-16 bg-gradient-to-br from-purple-500 to-purple-600 rounded-xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <Database className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 text-lg mb-2">Vector Database</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  High-performance vector storage for lightning-fast semantic search and retrieval
                </p>
              </div>
            </div>

            {/* Zero-Trust Security */}
            <div className="group relative">
              <div className="absolute inset-0 bg-gradient-to-br from-red-500 to-orange-600 rounded-2xl opacity-0 group-hover:opacity-10 transition-opacity duration-300"></div>
              <div className="relative text-center p-8 bg-white border border-gray-200 rounded-2xl shadow-lg hover:shadow-xl transition-all duration-300 hover:-translate-y-1">
                <div className="w-16 h-16 bg-gradient-to-br from-red-500 to-red-600 rounded-xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                  <Lock className="w-8 h-8 text-white" />
                </div>
                <h3 className="font-bold text-gray-900 text-lg mb-2">Zero-Trust Security</h3>
                <p className="text-gray-600 text-sm leading-relaxed">
                  Military-grade encryption and access controls meeting SOC 2 compliance standards
                </p>
              </div>
            </div>
          </div>

        </div>
      </section>

      {/* Section 4: Delivering Measurable Results */}
      <section className="py-20 relative overflow-hidden">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-s4" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient2-s4" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient3-s4" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            {/* Main flowing wave patterns */}
            <g stroke="url(#waveGradient1-s4)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
              <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
              <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
            </g>
            
            <g stroke="url(#waveGradient2-s4)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              <path d="M0,320 Q360,240 720,300 T1440,280"/>
              <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
              <path d="M0,360 Q420,280 840,340 T1440,320"/>
            </g>
            
            <g stroke="url(#waveGradient3-s4)" strokeWidth="1.0" fill="none" opacity="0.6">
              <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
              <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
              <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
            </g>
            
            {/* Filled wave areas for depth */}
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-s4)" opacity="0.1"/>
            <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-s4)" opacity="0.08"/>
            <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-s4)" opacity="0.06"/>
            
            {/* Additional flowing lines for complexity */}
            <g stroke="url(#waveGradient1-s4)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
              <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
            </g>
            
            <g stroke="url(#waveGradient2-s4)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
              <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
            </g>
          </svg>
        </div>
        <div className="relative max-w-6xl mx-auto px-6">
          <div className="text-center mb-16">
            <h2 className="text-4xl font-bold text-gray-900 mb-8">
              Delivering Measurable Results
            </h2>
            
            <div className="grid grid-cols-4 gap-12 mb-8">
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">85%</div>
                <div className="text-gray-600">Faster Document Processing</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">92%</div>
                <div className="text-gray-600">Accuracy in AI Insights</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">60%</div>
                <div className="text-gray-600">Reduction in Manual Work</div>
              </div>
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-2">99.9%</div>
                <div className="text-gray-600">Platform Uptime</div>
              </div>
            </div>

            <p className="text-gray-600 text-lg">
              Organizations using <span className="text-blue-600 font-semibold">SixthVault</span> see immediate improvements in documents<br />
              processing efficiency and decision-making speed.
            </p>
          </div>
        </div>
      </section>

      {/* Section 5: Ready to transform your documents workflow? */}
      <section className="relative overflow-hidden py-24">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="waveGradient1-s5" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient2-s5" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="waveGradient3-s5" x1="50%" y1="0%" x2="50%" y2="100%">
                <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.6"/>
                <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ffa726" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            {/* Main flowing wave patterns */}
            <g stroke="url(#waveGradient1-s5)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
              <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
              <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
            </g>
            
            <g stroke="url(#waveGradient2-s5)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              <path d="M0,320 Q360,240 720,300 T1440,280"/>
              <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
              <path d="M0,360 Q420,280 840,340 T1440,320"/>
            </g>
            
            <g stroke="url(#waveGradient3-s5)" strokeWidth="1.0" fill="none" opacity="0.6">
              <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
              <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
              <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
            </g>
            
            {/* Filled wave areas for depth */}
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-s5)" opacity="0.1"/>
            <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-s5)" opacity="0.08"/>
            <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-s5)" opacity="0.06"/>
            
            {/* Additional flowing lines for complexity */}
            <g stroke="url(#waveGradient1-s5)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
              <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
            </g>
            
            <g stroke="url(#waveGradient2-s5)" strokeWidth="0.6" fill="none" opacity="0.4">
              <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
              <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
            </g>
          </svg>
        </div>
        <div className="relative w-full max-w-none mx-auto px-4 text-center">
          <div className="bg-gradient-to-br from-slate-50 via-blue-50 to-indigo-100 rounded-3xl border border-gray-200 shadow-2xl p-16 mx-8 md:mx-16 lg:mx-24 xl:mx-32 relative overflow-hidden">
            {/* Beautiful flowing wave background inside the card */}
            <div className="absolute inset-0 rounded-3xl overflow-hidden">
              <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 600" preserveAspectRatio="xMidYMid slice">
                <defs>
                  <linearGradient id="waveGradient1-card" x1="0%" y1="0%" x2="100%" y2="100%">
                    <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
                    <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
                    <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
                    <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
                    <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                  </linearGradient>
                  <linearGradient id="waveGradient2-card" x1="100%" y1="0%" x2="0%" y2="100%">
                    <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.5"/>
                    <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.3"/>
                    <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
                    <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.15"/>
                    <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
                  </linearGradient>
                  <linearGradient id="waveGradient3-card" x1="50%" y1="0%" x2="50%" y2="100%">
                    <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.4"/>
                    <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.3"/>
                    <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.2"/>
                    <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.15"/>
                    <stop offset="100%" stopColor="#ffa726" stopOpacity="0.1"/>
                  </linearGradient>
                </defs>
                
                {/* Main flowing wave patterns */}
                <g stroke="url(#waveGradient1-card)" strokeWidth="1.2" fill="none" opacity="0.7">
                  <path d="M0,150 Q200,100 400,130 T800,120 Q1000,110 1200,130 T1440,120"/>
                  <path d="M0,170 Q240,120 480,150 T960,140 Q1200,130 1440,150"/>
                  <path d="M0,190 Q180,140 360,170 T720,160 Q900,150 1080,170 T1440,160"/>
                </g>
                
                <g stroke="url(#waveGradient2-card)" strokeWidth="1.0" fill="none" opacity="0.6">
                  <path d="M0,220 Q300,170 600,200 T1200,190 Q1320,180 1440,200"/>
                  <path d="M0,240 Q360,190 720,220 T1440,210"/>
                  <path d="M0,260 Q240,210 480,240 T960,230 Q1200,220 1440,240"/>
                </g>
                
                <g stroke="url(#waveGradient3-card)" strokeWidth="0.8" fill="none" opacity="0.5">
                  <path d="M0,290 Q180,240 360,270 T720,260 Q900,250 1080,270 T1440,260"/>
                  <path d="M0,310 Q270,260 540,290 T1080,280 Q1260,270 1440,290"/>
                </g>
                
                {/* Filled wave areas for depth */}
                <path d="M0,180 Q200,130 400,160 T800,150 Q1000,140 1200,160 T1440,150 L1440,600 L0,600 Z" fill="url(#waveGradient1-card)" opacity="0.08"/>
                <path d="M0,250 Q300,200 600,230 T1200,220 Q1320,210 1440,230 L1440,600 L0,600 Z" fill="url(#waveGradient2-card)" opacity="0.06"/>
                <path d="M0,320 Q360,270 720,300 T1440,290 L1440,600 L0,600 Z" fill="url(#waveGradient3-card)" opacity="0.04"/>
                
                {/* Additional flowing lines for complexity */}
                <g stroke="url(#waveGradient1-card)" strokeWidth="0.6" fill="none" opacity="0.4">
                  <path d="M0,130 Q120,80 240,110 T480,100 Q600,90 720,110 T960,100 Q1080,90 1200,110 T1440,100"/>
                  <path d="M0,350 Q150,300 300,330 T600,320 Q750,310 900,330 T1200,320 Q1320,310 1440,330"/>
                </g>
                
                <g stroke="url(#waveGradient2-card)" strokeWidth="0.5" fill="none" opacity="0.3">
                  <path d="M0,110 Q90,60 180,90 T360,80 Q450,70 540,90 T720,80 Q810,70 900,90 T1080,80 Q1170,70 1260,90 T1440,80"/>
                  <path d="M0,370 Q135,320 270,350 T540,340 Q675,330 810,350 T1080,340 Q1215,330 1350,350 T1440,340"/>
                </g>
              </svg>
            </div>
            <div className="relative z-10">
              <h1 className="text-5xl md:text-6xl font-bold text-gray-800 mb-8 leading-tight">
                Ready to transform<br />
                your <span className="bg-gradient-to-r from-blue-600 via-indigo-600 to-purple-600 bg-clip-text text-transparent font-extrabold">documents</span> workflow?
              </h1>
              
              <p className="text-xl text-gray-600 mb-12 max-w-2xl mx-auto leading-relaxed">
                Join thousands of organizations already using<br />
                <span className="text-blue-600 font-semibold">SixthVault</span> to unlock the power of their documents !
              </p>

              <div className="flex gap-6 justify-center items-center">
                <Link 
                  href="/register"
                  className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-white text-gray-800 hover:bg-gray-50 border border-gray-200 shadow-lg hover:shadow-xl px-10 py-4 text-lg font-semibold rounded-xl transition-all duration-300 hover:-translate-y-1 cursor-pointer"
                >
                  Start Free Trial
                </Link>
                <Link 
                  href="/login"
                  className="inline-flex items-center justify-center gap-2 whitespace-nowrap rounded-md text-sm font-medium ring-offset-background transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2 disabled:pointer-events-none disabled:opacity-50 bg-blue-500/20 border-2 border-blue-400/50 text-white hover:bg-blue-500/30 backdrop-blur-sm px-10 py-4 text-lg font-semibold rounded-xl transition-all duration-300 hover:-translate-y-1 hover:border-blue-400/70 cursor-pointer"
                >
                  Schedule demo
                </Link>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="relative overflow-hidden py-12 border-t border-gray-100">
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
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-5 gap-8">
            {/* Logo */}
            <div className="col-span-1">
              <div className="flex items-center mb-4">
                <SixthvaultLogo size="large" />
              </div>
              <div className="flex gap-3 mt-8">
                <a href="https://instagram.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 bg-gradient-to-br from-pink-500 to-purple-600 rounded-lg flex items-center justify-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                  <Instagram className="w-4 h-4 text-white" />
                </a>
                <a href="https://twitter.com" target="_blank" rel="noopener noreferrer" className="w-8 h-8 bg-gradient-to-br from-gray-800 to-black rounded-lg flex items-center justify-center hover:shadow-lg transition-all duration-300 hover:-translate-y-1">
                  <X className="w-4 h-4 text-white" />
                </a>
              </div>
            </div>

            {/* Product */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Product</h3>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-blue-600 transition-colors">Features</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Integration</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Updates</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">FAQ</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Pricing</a></li>
              </ul>
            </div>

            {/* Company */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Company</h3>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-blue-600 transition-colors">About</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Manifesto</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Press</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Contract</a></li>
              </ul>
            </div>

            {/* Resources */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Resources</h3>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-blue-600 transition-colors">Examples</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Community</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Guides</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Docs</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Press</a></li>
              </ul>
            </div>

            {/* Legal */}
            <div>
              <h3 className="font-semibold text-gray-900 mb-4">Legal</h3>
              <ul className="space-y-2 text-gray-600">
                <li><a href="#" className="hover:text-blue-600 transition-colors">Privacy</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Terms</a></li>
                <li><a href="#" className="hover:text-blue-600 transition-colors">Security</a></li>
              </ul>
            </div>
          </div>
        </div>
      </footer>
    </div>
  )
}
