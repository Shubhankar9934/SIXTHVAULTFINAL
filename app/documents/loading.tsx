"use client"

import { Skeleton } from "@/components/ui/skeleton"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Database, FileText, BarChart3, CheckCircle, Upload, Brain } from "lucide-react"

export default function Loading() {
  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Beautiful flowing wave background - Hidden on mobile for clean look */}
      <div className="absolute inset-0 z-0 hidden md:block">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="waveGradient1-loading" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
              <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
              <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
              <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
            </linearGradient>
          </defs>
          
          <g stroke="url(#waveGradient1-loading)" strokeWidth="1.5" fill="none" opacity="0.8">
            <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
            <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
            <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
          </g>
          
          <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-loading)" opacity="0.08"/>
        </svg>
      </div>

      {/* Modern Header */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                    Document Intelligence Hub
                  </h1>
                  <p className="text-sm text-gray-500">Loading your documents...</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="flex items-center space-x-2 bg-blue-50 px-3 py-2 rounded-lg">
                <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-blue-700">Loading Documents</span>
              </div>
              <Skeleton className="h-9 w-20" />
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Loading */}
      <div className="md:hidden">
        <div className="flex items-center justify-between p-4 bg-white border-b border-gray-200">
          <Skeleton className="h-5 w-5" />
          <h1 className="text-lg font-semibold text-gray-800">Loading Documents...</h1>
          <Skeleton className="h-5 w-5" />
        </div>

        <div className="p-4 pb-24">
          <div className="space-y-4">
            {[1, 2, 3, 4, 5].map((i) => (
              <div key={i} className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm">
                <div className="flex items-center space-x-3 mb-3">
                  <Skeleton className="w-10 h-10 rounded-lg" />
                  <div className="flex-1 space-y-2">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-3 w-1/2" />
                  </div>
                  <Skeleton className="w-5 h-5 rounded-full" />
                </div>
                <div className="flex flex-wrap gap-2 mb-3">
                  <Skeleton className="h-5 w-16 rounded-full" />
                  <Skeleton className="h-5 w-20 rounded-full" />
                </div>
                <div className="flex items-center justify-between">
                  <Skeleton className="h-3 w-20" />
                  <div className="flex items-center space-x-2">
                    <Skeleton className="h-8 w-8 rounded" />
                    <Skeleton className="h-8 w-8 rounded" />
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* Desktop Loading */}
      <div className="hidden md:flex h-[calc(100vh-80px)]">
        {/* Sidebar Loading */}
        <div className="w-80 bg-white/90 backdrop-blur-xl border-r border-gray-200/50 flex flex-col">
          {/* Logo Section */}
          <div className="p-3 border-b border-gray-200/50">
            <div className="w-full h-16 flex items-center justify-center">
              <Skeleton className="h-12 w-32" />
            </div>
          </div>

          {/* Navigation Loading */}
          <div className="flex-1 p-4 space-y-2">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="flex items-center space-x-3 h-12 px-4 rounded-lg">
                <Skeleton className="w-5 h-5" />
                <Skeleton className="h-4 flex-1" />
              </div>
            ))}
          </div>

          {/* Stats Panel Loading */}
          <div className="p-4 border-t border-gray-200/50">
            <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-4 space-y-3">
              <Skeleton className="h-4 w-20" />
              <div className="grid grid-cols-2 gap-3">
                <div className="text-center space-y-1">
                  <Skeleton className="h-8 w-12 mx-auto" />
                  <Skeleton className="h-3 w-16 mx-auto" />
                </div>
                <div className="text-center space-y-1">
                  <Skeleton className="h-8 w-16 mx-auto" />
                  <Skeleton className="h-3 w-12 mx-auto" />
                </div>
              </div>
            </div>
          </div>

          {/* Back Button Loading */}
          <div className="p-4 border-t border-gray-200/50">
            <Skeleton className="w-full h-12 rounded-lg" />
          </div>
        </div>

        {/* Main Content Loading */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Search Bar Loading */}
          <div className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 p-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
              <div className="flex-1 max-w-md">
                <Skeleton className="h-10 w-full rounded-lg" />
              </div>
              <div className="flex items-center space-x-3">
                <Skeleton className="h-10 w-32" />
                <Skeleton className="h-10 w-40" />
                <Skeleton className="h-10 w-36" />
                <Skeleton className="h-10 w-24" />
              </div>
            </div>
          </div>

          {/* Content Loading */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* Loading Message */}
            <div className="text-center py-12 mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
                <Database className="w-10 h-10 text-blue-600" />
              </div>
              <h2 className="text-2xl font-bold text-gray-900 mb-3">Loading Your Documents</h2>
              <p className="text-gray-600 mb-6 max-w-md mx-auto">
                We're fetching your documents and preparing the AI analysis dashboard. This should only take a moment.
              </p>
              <div className="flex items-center justify-center space-x-6 text-sm text-gray-500">
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-bounce"></div>
                  <span>Loading documents</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-green-500 rounded-full animate-bounce" style={{ animationDelay: '0.1s' }}></div>
                  <span>Preparing AI analysis</span>
                </div>
                <div className="flex items-center space-x-2">
                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-bounce" style={{ animationDelay: '0.2s' }}></div>
                  <span>Setting up interface</span>
                </div>
              </div>
            </div>

            {/* Stats Cards Loading */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
              {[
                { color: "from-blue-50 to-blue-100", icon: FileText },
                { color: "from-green-50 to-green-100", icon: CheckCircle },
                { color: "from-purple-50 to-purple-100", icon: Database },
                { color: "from-orange-50 to-orange-100", icon: BarChart3 }
              ].map((card, index) => {
                const Icon = card.icon
                return (
                  <Card key={index} className={`bg-gradient-to-br ${card.color} border-gray-200`}>
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-24" />
                          <Skeleton className="h-8 w-16" />
                        </div>
                        <div className="w-12 h-12 bg-white/50 rounded-xl flex items-center justify-center">
                          <Icon className="w-6 h-6 text-gray-400 animate-pulse" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                )
              })}
            </div>

            {/* Document List Loading */}
            <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
              <CardHeader>
                <div className="flex items-center space-x-2">
                  <Skeleton className="w-5 h-5" />
                  <Skeleton className="h-6 w-40" />
                </div>
                <Skeleton className="h-4 w-64" />
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {[1, 2, 3, 4, 5].map((i) => (
                    <div key={i} className="flex items-center justify-between p-4 bg-gray-50/50 rounded-lg">
                      <div className="flex items-center space-x-4">
                        <Skeleton className="w-10 h-10 rounded-lg" />
                        <div className="space-y-2">
                          <Skeleton className="h-4 w-48" />
                          <div className="flex items-center space-x-2">
                            <Skeleton className="h-3 w-16" />
                            <Skeleton className="h-3 w-20" />
                            <Skeleton className="h-5 w-12 rounded-full" />
                          </div>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <Skeleton className="w-5 h-5 rounded-full" />
                        <Skeleton className="w-8 h-8 rounded" />
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </div>
      </div>
    </div>
  )
}
