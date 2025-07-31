"use client"

import { Loader2 } from "lucide-react"

export default function VaultLoading() {
  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-blue-900 to-slate-900 flex items-center justify-center p-4">
      <div className="bg-white/95 p-6 rounded-lg shadow-xl flex flex-col items-center space-y-4 backdrop-blur-sm">
        <div className="relative">
          <Loader2 className="h-8 w-8 text-blue-600 animate-spin" />
          <div className="absolute inset-0 blur-sm animate-pulse bg-blue-400/30 rounded-full" />
        </div>
        <p className="text-slate-700 font-medium text-lg animate-fade-in">
          Loading your vault...
        </p>
      </div>
    </div>
  )
}
