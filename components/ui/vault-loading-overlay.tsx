"use client"

import React from 'react'
import { Loader2, Brain, FileText, MessageSquare, Settings, Sparkles, Database, Zap, CheckCircle } from 'lucide-react'
import { Card, CardContent } from './card'
import { Progress } from './progress'

interface LoadingStep {
  id: string
  label: string
  icon: React.ComponentType<{ className?: string }>
  completed: boolean
  inProgress: boolean
  description?: string
}

interface VaultLoadingOverlayProps {
  isVisible: boolean
  currentStep: string
  progress: number
  steps: LoadingStep[]
  message?: string
}

export function VaultLoadingOverlay({ 
  isVisible, 
  currentStep, 
  progress, 
  steps, 
  message 
}: VaultLoadingOverlayProps) {
  if (!isVisible) return null

  return (
    <div className="fixed inset-0 z-50 bg-white/95 backdrop-blur-sm flex items-center justify-center">
      {/* Beautiful flowing wave background */}
      <div className="absolute inset-0">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="loadingWave1" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
              <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
              <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
              <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
            </linearGradient>
            <linearGradient id="loadingWave2" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.5"/>
              <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.3"/>
              <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
              <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.15"/>
              <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
            </linearGradient>
          </defs>
          
          <g stroke="url(#loadingWave1)" strokeWidth="1.5" fill="none" opacity="0.6">
            <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160">
              <animate attributeName="d" dur="8s" repeatCount="indefinite"
                values="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160;
                        M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200;
                        M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
            </path>
            <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200">
              <animate attributeName="d" dur="10s" repeatCount="indefinite"
                values="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200;
                        M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240;
                        M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
            </path>
          </g>
          
          <g stroke="url(#loadingWave2)" strokeWidth="1.2" fill="none" opacity="0.5">
            <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280">
              <animate attributeName="d" dur="12s" repeatCount="indefinite"
                values="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280;
                        M0,320 Q360,240 720,300 T1440,280;
                        M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
            </path>
          </g>
          
          <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#loadingWave1)" opacity="0.08"/>
          <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#loadingWave2)" opacity="0.06"/>
        </svg>
      </div>

      <Card className="relative z-10 w-full max-w-md mx-4 shadow-2xl bg-white/95 backdrop-blur-xl border-0 rounded-2xl">
        <CardContent className="p-8">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="mx-auto w-16 h-16 bg-gradient-to-br from-blue-100 to-purple-100 rounded-2xl flex items-center justify-center mb-4">
              <div className="relative">
                <Loader2 className="w-8 h-8 text-blue-600 animate-spin" />
                <div className="absolute inset-0 w-8 h-8 border-2 border-blue-200 rounded-full animate-pulse"></div>
              </div>
            </div>
            <h2 className="text-xl font-bold text-slate-800 mb-2">
              Initializing SixthVault
            </h2>
            <p className="text-sm text-slate-600">
              Setting up your secure document workspace...
            </p>
          </div>

          {/* Progress Bar */}
          <div className="mb-6">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-slate-700">Progress</span>
              <span className="text-sm text-slate-500">{Math.round(progress)}%</span>
            </div>
            <Progress value={progress} className="h-2 bg-slate-100">
              <div 
                className="h-full bg-gradient-to-r from-blue-500 to-purple-600 rounded-full transition-all duration-500 ease-out"
                style={{ width: `${progress}%` }}
              />
            </Progress>
          </div>

          {/* Current Message */}
          {message && (
            <div className="mb-6 p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <p className="text-sm text-blue-800 font-medium">{message}</p>
            </div>
          )}

          {/* Loading Steps */}
          <div className="space-y-3">
            {steps.map((step) => {
              const IconComponent = step.icon
              const isActive = step.inProgress
              const isCompleted = step.completed
              
              return (
                <div
                  key={step.id}
                  className={`flex items-center space-x-3 p-3 rounded-lg transition-all duration-300 ${
                    isActive 
                      ? 'bg-blue-50 border border-blue-200' 
                      : isCompleted
                      ? 'bg-green-50 border border-green-200'
                      : 'bg-slate-50 border border-slate-200'
                  }`}
                >
                  <div className={`flex-shrink-0 w-8 h-8 rounded-full flex items-center justify-center ${
                    isCompleted
                      ? 'bg-green-100'
                      : isActive
                      ? 'bg-blue-100'
                      : 'bg-slate-100'
                  }`}>
                    {isCompleted ? (
                      <CheckCircle className="w-4 h-4 text-green-600" />
                    ) : isActive ? (
                      <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
                    ) : (
                      <IconComponent className={`w-4 h-4 ${
                        isCompleted ? 'text-green-600' : isActive ? 'text-blue-600' : 'text-slate-400'
                      }`} />
                    )}
                  </div>
                  
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium ${
                      isCompleted ? 'text-green-800' : isActive ? 'text-blue-800' : 'text-slate-600'
                    }`}>
                      {step.label}
                    </p>
                    {step.description && (
                      <p className={`text-xs mt-1 ${
                        isCompleted ? 'text-green-600' : isActive ? 'text-blue-600' : 'text-slate-500'
                      }`}>
                        {step.description}
                      </p>
                    )}
                  </div>
                </div>
              )
            })}
          </div>

          {/* Footer */}
          <div className="mt-6 pt-4 border-t border-slate-200">
            <div className="flex items-center justify-center space-x-2 text-xs text-slate-500">
              <Zap className="w-3 h-3" />
              <span>Enterprise-grade security and performance</span>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
