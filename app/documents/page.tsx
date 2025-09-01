"use client"

import type React from "react"
import { useState, useCallback, useEffect, useMemo } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"
import { Progress } from "@/components/ui/progress"
import {
  Upload,
  FileText,
  X,
  CheckCircle,
  AlertCircle,
  Loader2,
  Tag,
  BookOpen,
  Plus,
  Trash2,
  Users,
  BarChart3,
  Languages,
  BookMarked,
  Wifi,
  Search,
  Filter,
  Download,
  Eye,
  Settings,
  TrendingUp,
  Clock,
  Database,
  Shield,
  Zap,
  Globe,
  Brain,
  Target,
  Lightbulb,
  Activity,
  PieChart,
  Calendar,
  FileCheck,
  Archive,
  Star,
  ChevronRight,
  Menu,
  Home,
  LogOut,
  RefreshCw,
  HelpCircle,
  Bell,
  User
} from "lucide-react"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger,
} from "@/components/ui/alert-dialog"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { documentStore, type DocumentData, type ProcessingDocument } from "@/lib/document-store"
import { aiService } from "@/lib/ai-service"
import { ragApiClient, type BackendDocument, type WebSocketMessage, type AvailableModel, type ModelProvider } from "@/lib/api-client"
import { RouteGuard } from "@/components/route-guard"
import { useAuth } from "@/lib/auth-context"
import { useVaultState } from "@/lib/vault-state-provider"
import { documentSyncManager } from "@/lib/document-sync-manager"
import { uploadStateManager } from "@/lib/upload-state-manager"

interface Document {
  id: string
  name: string
  size: number
  type: string
  status: "uploading" | "processing" | "completed" | "error" | "waiting" // Added "waiting" status
  progress: number
  uploadDate: string
  language?: string
  summary?: string
  themes?: string[]
  keywords?: string[]
  demographics?: string[]
  mainTopics?: string[]
  sentiment?: string
  readingLevel?: string
  keyInsights?: string[]
  content?: string
  error?: string
  batchId?: string
  processingStage?: string
  statusMessage?: string
  processingTime?: number
  processingOrder?: number // Add for sequential progress tracking
}

function DocumentsPageContent() {
  const { logout } = useAuth()
  const { refreshDocuments } = useVaultState()
  const [activeTab, setActiveTab] = useState("dashboard")
  const [documents, setDocuments] = useState<Document[]>([])
  const [processingDocuments, setProcessingDocuments] = useState<ProcessingDocument[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isInitialLoading, setIsInitialLoading] = useState(true)
  const [loadingMessage, setLoadingMessage] = useState("Initializing document system...")
  const [newTags, setNewTags] = useState<Record<string, string>>({})
  const [newKeywords, setNewKeywords] = useState<Record<string, string>>({})
  const [newDemographics, setNewDemographics] = useState<Record<string, string>>({})
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [aiStatus, setAiStatus] = useState<"checking" | "available" | "unavailable">("checking")
  const [searchQuery, setSearchQuery] = useState("")
  const [filterStatus, setFilterStatus] = useState("all")
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [selectedProvider, setSelectedProvider] = useState("")
  const [selectedModel, setSelectedModel] = useState("")
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([])
  const [modelProviders, setModelProviders] = useState<ModelProvider[]>([])
  const [isProviderLoading, setIsProviderLoading] = useState(false)
  const [deleteDialog, setDeleteDialog] = useState<{
    isOpen: boolean
    document: Document | null
    isDeleting: boolean
  }>({
    isOpen: false,
    document: null,
    isDeleting: false
  })
  const [deletingDocuments, setDeletingDocuments] = useState<Set<string>>(new Set())

  // Cleanup effect for WebSocket connections
  useEffect(() => {
    return () => {
      // Cleanup any remaining WebSocket connections
      Object.keys(window as any).forEach(key => {
        if (key.startsWith('cleanup_')) {
          const cleanup = (window as any)[key]
          if (typeof cleanup === 'function') {
            cleanup()
            delete (window as any)[key]
          }
        }
      })
    }
  }, [selectedProvider, selectedModel])

  // Helper function to extract filename from path
  const extractFilename = (filePath: string): string => {
    if (!filePath) return ''
    return filePath.split('/').pop() || filePath.split('\\').pop() || filePath
  }

  // Professional content parser for enterprise formatting
  const parseEnterpriseContent = (content: string): string => {
    if (!content) return ''

    let parsed = content

    // Parse structured headers from backend (** text **)
    parsed = parsed.replace(/\*\*(.*?)\*\*/g, (match, text) => {
      const cleanText = text.trim()
      
      // Map backend section headers to appropriate CSS classes
      if (cleanText.includes('Consumer Truths') || cleanText.includes('Core Insights')) {
        return `<h2 class="section-header consumer-truths">${cleanText}</h2>`
      } else if (cleanText.includes('Growth Opportunities') || cleanText.includes('Strategic Implications')) {
        return `<h2 class="section-header growth-opportunities">${cleanText}</h2>`
      } else if (cleanText.includes('Action Plan') || cleanText.includes('Recommended Actions')) {
        return `<h2 class="section-header action-plan">${cleanText}</h2>`
      } else if (cleanText.includes('Executive Summary')) {
        return `<h1 class="executive-title">${cleanText}</h1>`
      } else {
        return `<h3 class="subsection-header">${cleanText}</h3>`
      }
    })

    // Parse numbered lists (1. 2. 3.)
    parsed = parsed.replace(/^(\d+)\.\s+(.+)$/gm, '<div class="numbered-item"><span class="number">$1</span><span class="content">$2</span></div>')

    // Parse bullet points (- text)
    parsed = parsed.replace(/^-\s+(.+)$/gm, '<div class="bullet-item"><span class="bullet">•</span><span class="content">$1</span></div>')

    // Parse complexity ratings and opportunities
    parsed = parsed.replace(/\|\s*Complexity:\s*(Low|Med|Medium|High)/gi, '<span class="complexity-badge complexity-$1">Complexity: $1</span>')

    // Parse immediate/strategic opportunities
    parsed = parsed.replace(/^-\s*(Immediate|Strategic)\s*\(([^)]+)\):\s*(.+)$/gm, 
      '<div class="opportunity-item $1"><span class="timeframe">$1 ($2)</span><span class="opportunity">$3</span></div>')

    // Convert line breaks to proper spacing
    parsed = parsed.replace(/\n\n/g, '</div><div class="content-block">')
    parsed = parsed.replace(/\n/g, '<br>')

    // Wrap in content blocks
    if (!parsed.startsWith('<h') && !parsed.startsWith('<div')) {
      parsed = `<div class="content-block">${parsed}</div>`
    }

    return parsed
  }

  // Parse insights content - convert markdown to proper HTML formatting with respondent quote styling
  const parseInsightsContent = (insights: string[]): string => {
    if (!insights || insights.length === 0) return ''

    return insights.map((insight, index) => {
      // Handle the content from backend - convert markdown to HTML
      let content = insight.trim()
      
      // Convert markdown headers to proper HTML headers
      content = content.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-800 mb-3 mt-4">$1</h3>')
      content = content.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mb-4 mt-6">$1</h2>')
      content = content.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-gray-900 mb-4 mt-6">$1</h1>')
      
      // Convert markdown bold (**text**) to HTML strong tags
      content = content.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
      
      // Convert markdown italic (*text*) to HTML em tags
      content = content.replace(/\*([^*]+)\*/g, '<em class="italic text-gray-700">$1</em>')
      
      // Format respondent quotes in brackets with special styling
      content = content.replace(/\[([^\]]+)\]/g, '<div class="mt-3 p-3 bg-blue-50 border-l-4 border-blue-400 rounded-r-lg"><div class="flex items-start space-x-2"><svg class="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd"></path></svg><div class="text-sm"><span class="font-medium text-blue-800">Respondent Quote:</span><div class="text-blue-700 italic mt-1">"$1"</div></div></div></div>')
      
      // Convert markdown bullet points to proper HTML lists
      const lines = content.split('\n')
      let processedLines = []
      let inList = false
      
      for (let i = 0; i < lines.length; i++) {
        const line = lines[i].trim()
        
        // Check if line is a bullet point
        if (line.match(/^[-*+•]\s+/)) {
          if (!inList) {
            processedLines.push('<ul class="space-y-4 mb-6">')
            inList = true
          }
          const bulletContent = line.replace(/^[-*+•]\s+/, '')
          processedLines.push(`<li class="flex items-start space-x-3"><span class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span><div class="flex-1 text-gray-700">${bulletContent}</div></li>`)
        } else if (line.match(/^\d+\.\s+/)) {
          if (!inList) {
            processedLines.push('<ol class="space-y-4 mb-6">')
            inList = true
          }
          const numberContent = line.replace(/^\d+\.\s+/, '')
          const number = line.match(/^(\d+)\./)?.[1] || '1'
          processedLines.push(`<li class="flex items-start space-x-3"><span class="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">${number}</span><div class="flex-1 text-gray-700">${numberContent}</div></li>`)
        } else {
          if (inList) {
            processedLines.push('</ul>')
            inList = false
          }
          if (line) {
            processedLines.push(`<p class="text-gray-700 mb-3">${line}</p>`)
          }
        }
      }
      
      if (inList) {
        processedLines.push('</ul>')
      }
      
      content = processedLines.join('\n')
      
      // Check if it contains structured insights
      if (content.includes('Actionable Insight') || content.includes('Critical Risk')) {
        // For structured content, wrap in a professional container
        return `
          <div class="structured-insights-container bg-white rounded-lg p-6 border border-gray-200 shadow-sm">
            <div class="insights-content prose prose-sm max-w-none">
              ${content}
            </div>
          </div>
        `
      } else {
        // For simple insights, use the card format
        const isRisk = content.toLowerCase().includes('risk') || content.toLowerCase().includes('threat')
        
        if (isRisk) {
          return `
            <div class="critical-risk bg-red-50 border-l-4 border-red-400 p-6 rounded-r-lg">
              <div class="risk-header flex items-center mb-4">
                <span class="risk-icon text-2xl mr-3">⚠️</span>
                <span class="risk-title text-lg font-semibold text-red-800">Critical Risk ${index + 1}</span>
              </div>
              <div class="risk-content prose prose-sm max-w-none">${content}</div>
            </div>
          `
        } else {
          return `
            <div class="actionable-insight bg-blue-50 border-l-4 border-blue-400 p-6 rounded-r-lg">
              <div class="insight-header flex items-center mb-4">
                <span class="insight-icon text-2xl mr-3">💡</span>
                <span class="insight-title text-lg font-semibold text-blue-800">Strategic Insight ${index + 1}</span>
              </div>
              <div class="insight-content prose prose-sm max-w-none">${content}</div>
            </div>
          `
        }
      }
    }).join('')
  }

  // Format summary content with proper markdown to HTML conversion
  const formatSummaryContent = (summary: string) => {
    if (!summary) return null

    // Convert markdown to HTML while preserving structure
    let processedSummary = summary
    
    // Convert markdown headers to proper HTML headers
    processedSummary = processedSummary.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-800 mb-3 mt-4">$1</h3>')
    processedSummary = processedSummary.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mb-4 mt-6">$1</h2>')
    processedSummary = processedSummary.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-gray-900 mb-4 mt-6">$1</h1>')
    
    // Convert markdown bold (**text**) to HTML strong tags
    processedSummary = processedSummary.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    
    // Convert markdown italic (*text*) to HTML em tags
    processedSummary = processedSummary.replace(/\*([^*]+)\*/g, '<em class="italic text-gray-700">$1</em>')
    
    // Process bullet points and numbered lists
    const lines = processedSummary.split('\n')
    let processedLines: string[] = []
    let inList = false
    let listType = ''
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim()
      
      // Check if line is a bullet point
      if (line.match(/^[-*+]\s+/)) {
        if (!inList || listType !== 'ul') {
          if (inList) processedLines.push(`</${listType}>`)
          processedLines.push('<ul class="list-disc list-inside space-y-2 ml-4 mb-4">')
          inList = true
          listType = 'ul'
        }
        const bulletContent = line.replace(/^[-*+]\s+/, '')
        processedLines.push(`<li class="text-gray-700">${bulletContent}</li>`)
      } else if (line.match(/^\d+\.\s+/)) {
        if (!inList || listType !== 'ol') {
          if (inList) processedLines.push(`</${listType}>`)
          processedLines.push('<ol class="list-decimal list-inside space-y-2 ml-4 mb-4">')
          inList = true
          listType = 'ol'
        }
        const numberContent = line.replace(/^\d+\.\s+/, '')
        processedLines.push(`<li class="text-gray-700">${numberContent}</li>`)
      } else {
        if (inList) {
          processedLines.push(`</${listType}>`)
          inList = false
          listType = ''
        }
        if (line) {
          processedLines.push(`<p class="text-gray-700 mb-3 leading-relaxed">${line}</p>`)
        }
      }
    }
    
    if (inList) {
      processedLines.push(`</${listType}>`)
    }
    
    const htmlContent = processedLines.join('\n')
    
    return (
      <div 
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
    )
  }

  // Format insight content with proper markdown to HTML conversion and respondent quote formatting
  const formatInsightContent = (insight: string) => {
    if (!insight) return null

    // Convert markdown to HTML while preserving structure
    let processedInsight = insight.trim()
    
    // Convert markdown headers to proper HTML headers
    processedInsight = processedInsight.replace(/^### (.*$)/gm, '<h3 class="text-lg font-semibold text-gray-800 mb-3 mt-4">$1</h3>')
    processedInsight = processedInsight.replace(/^## (.*$)/gm, '<h2 class="text-xl font-bold text-gray-900 mb-4 mt-6">$1</h2>')
    processedInsight = processedInsight.replace(/^# (.*$)/gm, '<h1 class="text-2xl font-bold text-gray-900 mb-4 mt-6">$1</h1>')
    
    // Convert markdown bold (**text**) to HTML strong tags
    processedInsight = processedInsight.replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-gray-900">$1</strong>')
    
    // Convert markdown italic (*text*) to HTML em tags
    processedInsight = processedInsight.replace(/\*([^*]+)\*/g, '<em class="italic text-gray-700">$1</em>')
    
    // Format respondent quotes in brackets with special styling
    processedInsight = processedInsight.replace(/\[([^\]]+)\]/g, '<div class="mt-3 p-3 bg-blue-50 border-l-4 border-blue-400 rounded-r-lg"><div class="flex items-start space-x-2"><svg class="w-4 h-4 text-blue-500 mt-0.5 flex-shrink-0" fill="currentColor" viewBox="0 0 20 20"><path fill-rule="evenodd" d="M18 10c0 3.866-3.582 7-8 7a8.841 8.841 0 01-4.083-.98L2 17l1.338-3.123C2.493 12.767 2 11.434 2 10c0-3.866 3.582-7 8-7s8 3.134 8 7zM7 9H5v2h2V9zm8 0h-2v2h2V9zM9 9h2v2H9V9z" clip-rule="evenodd"></path></svg><div class="text-sm"><span class="font-medium text-blue-800">Respondent Quote:</span><div class="text-blue-700 italic mt-1">"$1"</div></div></div></div>')
    
    // Process bullet points and numbered lists
    const lines = processedInsight.split('\n')
    let processedLines: string[] = []
    let inList = false
    let listType = ''
    
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i].trim()
      
      // Check if line is a bullet point
      if (line.match(/^[-*+•]\s+/)) {
        if (!inList || listType !== 'ul') {
          if (inList) processedLines.push(`</${listType}>`)
          processedLines.push('<ul class="space-y-4 mb-6">')
          inList = true
          listType = 'ul'
        }
        const bulletContent = line.replace(/^[-*+•]\s+/, '')
        processedLines.push(`<li class="flex items-start space-x-3"><span class="w-2 h-2 bg-blue-500 rounded-full mt-2 flex-shrink-0"></span><div class="flex-1 text-gray-700">${bulletContent}</div></li>`)
      } else if (line.match(/^\d+\.\s+/)) {
        if (!inList || listType !== 'ol') {
          if (inList) processedLines.push(`</${listType}>`)
          processedLines.push('<ol class="space-y-4 mb-6">')
          inList = true
          listType = 'ol'
        }
        const numberContent = line.replace(/^\d+\.\s+/, '')
        const number = line.match(/^(\d+)\./)?.[1] || '1'
        processedLines.push(`<li class="flex items-start space-x-3"><span class="w-6 h-6 bg-blue-500 text-white rounded-full flex items-center justify-center text-xs font-bold flex-shrink-0">${number}</span><div class="flex-1 text-gray-700">${numberContent}</div></li>`)
      } else {
        if (inList) {
          processedLines.push(`</${listType}>`)
          inList = false
          listType = ''
        }
        if (line) {
          processedLines.push(`<p class="text-gray-700 mb-3 leading-relaxed">${line}</p>`)
        }
      }
    }
    
    if (inList) {
      processedLines.push(`</${listType}>`)
    }
    
    const htmlContent = processedLines.join('\n')
    
    return (
      <div 
        className="prose prose-sm max-w-none"
        dangerouslySetInnerHTML={{ __html: htmlContent }}
      />
    )
  }

  // Load available models on mount and restore user preferences
  useEffect(() => {
    const loadModels = async () => {
      setIsProviderLoading(true)
      try {
        const providers = await ragApiClient.getAvailableModels()
        setModelProviders(providers)
        
        // Try to restore user preferences from localStorage
        const savedProvider = localStorage.getItem('sixthvault_preferred_provider')
        const savedModel = localStorage.getItem('sixthvault_preferred_model')
        
        console.log('=== PREFERENCE RESTORATION ===')
        console.log('Saved provider:', savedProvider)
        console.log('Saved model:', savedModel)
        console.log('Available providers:', providers.map(p => p.name))
        
        // Set default provider priority: Gemini > OpenAI > Groq > Others
        const providerPriority = ['gemini', 'openai', 'groq', 'deepseek', 'ollama']
        let validProvider = savedProvider
        
        if (!validProvider || !providers.find(p => p.name === validProvider)) {
          // Find the first available provider from priority list
          validProvider = providerPriority.find(p => providers.find(provider => provider.name === p)) || 
                        (providers.length > 0 ? providers[0].name : 'gemini')
          console.log('Using priority-based provider:', validProvider)
        }
        setSelectedProvider(validProvider)
        
        // Validate and set model for the provider
        const currentProvider = providers.find(p => p.name === validProvider)
        if (currentProvider) {
          setAvailableModels(currentProvider.models)
          
          let validModel = savedModel
          if (!validModel || !currentProvider.models.find(m => m.name === validModel)) {
            // Set default model based on provider
            if (validProvider === 'gemini') {
              validModel = 'gemini-1.5-flash'
            } else if (validProvider === 'openai') {
              validModel = 'gpt-4o-mini'
            } else if (validProvider === 'groq') {
              validModel = 'llama-3.1-8b-instant'
            } else {
              validModel = currentProvider.models.length > 0 ? currentProvider.models[0].name : 'gemini-1.5-flash'
            }
            console.log('Using default model for provider:', validModel)
          }
          setSelectedModel(validModel)
          
          console.log('Final selections - Provider:', validProvider, 'Model:', validModel)
        }
        console.log('=== END PREFERENCE RESTORATION ===')
        
      } catch (error) {
        console.error('Failed to load available models:', error)
        // Robust fallback to Gemini
        setSelectedProvider('gemini')
        setSelectedModel('gemini-1.5-flash')
        console.log('Error fallback - Using Gemini as default')
      } finally {
        setIsProviderLoading(false)
      }
    }

    loadModels()
  }, [])

  // Update available models when provider changes and persist preferences
  useEffect(() => {
    const currentProvider = modelProviders.find(p => p.name === selectedProvider)
    if (currentProvider) {
      setAvailableModels(currentProvider.models)
      // Reset model selection when provider changes
      if (currentProvider.models.length > 0) {
        setSelectedModel(currentProvider.models[0].name)
        console.log('Provider changed to:', selectedProvider, 'Model set to:', currentProvider.models[0].name)
        
        // Persist user preferences
        localStorage.setItem('sixthvault_preferred_provider', selectedProvider)
        localStorage.setItem('sixthvault_preferred_model', currentProvider.models[0].name)
      } else {
        setSelectedModel("")
        console.log('Provider changed to:', selectedProvider, 'No models available')
        
        // Persist provider but clear model
        localStorage.setItem('sixthvault_preferred_provider', selectedProvider)
        localStorage.removeItem('sixthvault_preferred_model')
      }
    }
  }, [selectedProvider, modelProviders])

  // Persist model preference when manually changed
  useEffect(() => {
    if (selectedModel) {
      localStorage.setItem('sixthvault_preferred_model', selectedModel)
      console.log('Model preference saved:', selectedModel)
    }
  }, [selectedModel])

  // Cross-tab document deletion listener
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'sixthvault_document_delete_event') {
        console.log('📡 Documents: Received cross-tab document deletion event')
        
        try {
          const eventData = JSON.parse(event.newValue || '{}')
          console.log('📡 Documents: Deletion event data:', eventData)
          
          // Remove the deleted document from local state immediately
          if (eventData.documentId) {
            setDocuments(prev => prev.filter(d => d.id !== eventData.documentId))
            
            // Clear selected document if it was the deleted one
            if (selectedDocument?.id === eventData.documentId) {
              setSelectedDocument(null)
            }
          }
          
          // Force refresh documents from backend after a short delay
          setTimeout(() => {
            documentStore.refreshDocuments().then(() => {
              console.log('✅ Documents: Refreshed documents after cross-tab deletion')
            }).catch(error => {
              console.error('❌ Documents: Failed to refresh after cross-tab deletion:', error)
            })
          }, 1000)
          
        } catch (error) {
          console.error('❌ Documents: Failed to parse deletion event:', error)
        }
      }
      
      // Handle cache invalidation events
      if (event.key === 'sixthvault_cache_invalidate_documents') {
        console.log('📡 Documents: Received cache invalidation event')
        
        // Clear cache and refresh documents
        setTimeout(() => {
          documentStore.refreshDocuments().then(() => {
            console.log('✅ Documents: Refreshed documents after cache invalidation')
          }).catch(error => {
            console.error('❌ Documents: Failed to refresh after cache invalidation:', error)
          })
        }, 500)
      }
    }

    // Add storage event listener
    window.addEventListener('storage', handleStorageChange)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
    }
  }, [selectedDocument])

  // OPTIMIZED: Proper integration with document store background processing with immediate loading feedback
  useEffect(() => {
    let isMounted = true
    let unsubscribeFromUpdates: (() => void) | null = null
    let unsubscribeFromProcessing: (() => void) | null = null

    const initializeDocuments = async () => {
      try {
        console.log('Documents: Initializing with document store integration')
        setLoadingMessage("Connecting to document system...")
        
        // STEP 1: Initialize background processing in document store
        documentStore.initializeBackgroundProcessing()
        setLoadingMessage("Loading processing documents...")
        
        // STEP 2: Load processing documents from document store
        const processingDocs = documentStore.getProcessingDocuments()
        console.log('Documents: Found', processingDocs.length, 'processing documents')
        
        // Show immediate feedback if we have processing documents
        if (processingDocs.length > 0) {
          setLoadingMessage(`Found ${processingDocs.length} documents in progress...`)
          setIsInitialLoading(false) // Show processing documents immediately
        }
        
        // STEP 3: Load completed documents from document store
        setLoadingMessage("Loading completed documents...")
        const completedDocs = await documentStore.getDocuments()
        console.log('Documents: Loaded', completedDocs.length, 'completed documents')
        
        if (!isMounted) return

        // STEP 4: Convert and merge all documents
        const allDocuments: Document[] = []
        
        // Add processing documents
        processingDocs.forEach(doc => {
          allDocuments.push({
            id: doc.id,
            name: doc.name,
            size: doc.size,
            type: doc.type,
            status: doc.status,
            progress: doc.progress,
            uploadDate: doc.uploadDate,
            batchId: doc.batchId,
            processingStage: doc.processingStage,
            statusMessage: doc.statusMessage,
            processingTime: doc.processingTime,
            processingOrder: doc.processingOrder,
            language: doc.language || 'English',
            summary: doc.summary || '',
            themes: doc.themes || [],
            keywords: doc.keywords || [],
            demographics: doc.demographics || [],
            mainTopics: doc.mainTopics || [],
            sentiment: doc.sentiment || 'neutral',
            readingLevel: doc.readingLevel || 'intermediate',
            keyInsights: doc.keyInsights || [],
            content: doc.content || '',
            error: doc.error
          })
        })
        
        // Add completed documents (avoid duplicates)
        completedDocs.forEach(doc => {
          // Only add if not already in processing
          if (!allDocuments.some(existing => existing.id === doc.id)) {
            allDocuments.push({
              id: doc.id,
              name: doc.name,
              size: doc.size,
              type: doc.type,
              status: "completed" as const,
              progress: 100,
              uploadDate: doc.uploadDate,
              language: doc.language,
              summary: doc.summary,
              themes: doc.themes || [],
              keywords: doc.keywords || [],
              demographics: doc.demographics || [],
              mainTopics: doc.mainTopics || doc.themes || [],
              sentiment: doc.sentiment || "neutral",
              readingLevel: doc.readingLevel || "intermediate",
              keyInsights: doc.keyInsights || [],
              content: doc.content || ""
            })
          }
        })
        
        // Set all documents at once
        setDocuments(allDocuments)
        console.log('Documents: Set', allDocuments.length, 'total documents (', processingDocs.length, 'processing,', completedDocs.length, 'completed)')
        
        // Clear loading state now that we have documents
        setIsInitialLoading(false)
        setLoadingMessage("Documents loaded successfully!")
        
        // STEP 5: Subscribe to completed document updates
        unsubscribeFromUpdates = documentStore.subscribeToUpdates((updatedCompletedDocs) => {
          if (!isMounted) return
          
          console.log('Documents: Received completed documents update:', updatedCompletedDocs.length)
          
          setDocuments(prevDocs => {
            // Keep processing documents, update completed ones
            const processingDocs = prevDocs.filter(doc => doc.status !== "completed")
            
            // Convert updated completed documents
            const newCompletedDocs: Document[] = updatedCompletedDocs.map(doc => ({
              id: doc.id,
              name: doc.name,
              size: doc.size,
              type: doc.type,
              status: "completed" as const,
              progress: 100,
              uploadDate: doc.uploadDate,
              language: doc.language,
              summary: doc.summary,
              themes: doc.themes || [],
              keywords: doc.keywords || [],
              demographics: doc.demographics || [],
              mainTopics: doc.mainTopics || doc.themes || [],
              sentiment: doc.sentiment || "neutral",
              readingLevel: doc.readingLevel || "intermediate",
              keyInsights: doc.keyInsights || [],
              content: doc.content || ""
            }))
            
            // Remove duplicates between processing and completed
            const uniqueCompletedDocs = newCompletedDocs.filter(completedDoc => 
              !processingDocs.some(processingDoc => processingDoc.id === completedDoc.id)
            )
            
            const mergedDocs = [...processingDocs, ...uniqueCompletedDocs]
            console.log('Documents: Updated with completed documents:', mergedDocs.length, 'total')
            
            return mergedDocs
          })
        })

        // STEP 6: Subscribe to processing document updates
        unsubscribeFromProcessing = documentStore.subscribeToProcessingUpdates((updatedProcessingDocs) => {
          if (!isMounted) return
          
          console.log('Documents: Received processing documents update:', updatedProcessingDocs.length)
          
          setDocuments(prevDocs => {
            // Keep completed documents
            const completedDocs = prevDocs.filter(doc => doc.status === "completed")
            
            // Convert updated processing documents
            const newProcessingDocs: Document[] = updatedProcessingDocs.map(doc => ({
              id: doc.id,
              name: doc.name,
              size: doc.size,
              type: doc.type,
              status: doc.status,
              progress: doc.progress,
              uploadDate: doc.uploadDate,
              batchId: doc.batchId,
              processingStage: doc.processingStage,
              statusMessage: doc.statusMessage,
              processingTime: doc.processingTime,
              processingOrder: doc.processingOrder,
              language: doc.language || 'English',
              summary: doc.summary || '',
              themes: doc.themes || [],
              keywords: doc.keywords || [],
              demographics: doc.demographics || [],
              mainTopics: doc.mainTopics || [],
              sentiment: doc.sentiment || 'neutral',
              readingLevel: doc.readingLevel || 'intermediate',
              keyInsights: doc.keyInsights || [],
              content: doc.content || '',
              error: doc.error
            }))
            
            // Remove duplicates between processing and completed
            const uniqueProcessingDocs = newProcessingDocs.filter(processingDoc => 
              !completedDocs.some(completedDoc => completedDoc.id === processingDoc.id)
            )
            
            const mergedDocs = [...uniqueProcessingDocs, ...completedDocs]
            console.log('Documents: Updated with processing documents:', mergedDocs.length, 'total')
            
            return mergedDocs
          })
        })
        
      } catch (error) {
        console.error('Documents: Error initializing:', error)
        if (isMounted) {
          setDocuments([])
        }
      }
    }

    // Initialize everything
    initializeDocuments().then(() => {
      if (isMounted) {
        checkAiStatus()
      }
    })

    return () => {
      isMounted = false
      if (unsubscribeFromUpdates) {
        unsubscribeFromUpdates()
      }
      if (unsubscribeFromProcessing) {
        unsubscribeFromProcessing()
      }
    }
  }, [])

  // CRITICAL FIX: Add real-time processing updates listener
  useEffect(() => {
    let processingUpdateInterval: NodeJS.Timeout | null = null

    const startProcessingUpdatesListener = () => {
      // Check for processing documents every 2 seconds and update their progress
      processingUpdateInterval = setInterval(() => {
        setDocuments(prevDocs => {
          const hasProcessingDocs = prevDocs.some(doc => 
            doc.status === "processing" || doc.status === "uploading"
          )
          
          if (!hasProcessingDocs) {
            return prevDocs
          }

          // Update progress for processing documents
          return prevDocs.map(doc => {
            if (doc.status === "processing" && doc.progress < 90) {
              // Gradually increase progress for visual feedback
              const newProgress = Math.min(doc.progress + 2, 90)
              return {
                ...doc,
                progress: newProgress,
                statusMessage: doc.statusMessage || 'Processing with AI analysis...'
              }
            } else if (doc.status === "uploading" && doc.progress < 20) {
              // Update upload progress
              const newProgress = Math.min(doc.progress + 5, 20)
              return {
                ...doc,
                progress: newProgress,
                statusMessage: 'Uploading to server...'
              }
            }
            return doc
          })
        })
      }, 2000) // Update every 2 seconds
    }

    startProcessingUpdatesListener()

    return () => {
      if (processingUpdateInterval) {
        clearInterval(processingUpdateInterval)
      }
    }
  }, [])

  const checkAiStatus = async () => {
    try {
      const response = await fetch("/api/config")
      const data = await response.json()
      // AI is always available for all users
      setAiStatus("available")
      console.log("AI status: Available for all users")
    } catch (error) {
      console.error("Failed to check AI status:", error)
      // Even on error, assume AI is available
      setAiStatus("available")
    }
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragEnter = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    e.stopPropagation()
    setIsDragging(false)

    const droppedFiles = Array.from(e.dataTransfer.files)
    if (droppedFiles.length > 0) {
      processFiles(droppedFiles)
    }
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFiles = Array.from(e.target.files)
      processFiles(selectedFiles)
      e.target.value = ""
    }
  }

  const processFiles = useCallback(async (fileList: File[]) => {
    const allowedTypes = [
      "application/pdf",
      "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
      "text/plain",
      "application/rtf",
    ]

    const validFiles = fileList.filter((file) => {
      if (!allowedTypes.includes(file.type)) {
        alert(`File type ${file.type} is not supported. Please upload PDF, DOCX, TXT, or RTF files.`)
        return false
      }
      if (file.size > 100 * 1024 * 1024) {
        alert(`File ${file.name} is too large. Maximum size is 100MB.`)
        return false
      }
      return true
    })

    if (validFiles.length === 0) return

    // UPLOAD STATE INTEGRATION: Start tracking uploads globally
    console.log('🔄 STARTING UPLOAD STATE TRACKING for', validFiles.length, 'files')
    const uploadDocumentIds = uploadStateManager.startUpload(validFiles)
    
    // ULTIMATE FIX: True isolated file processing with distinct progress tracking
    console.log('🔄 STARTING ISOLATED FILE PROCESSING for', validFiles.length, 'files')
    
    // First, add ALL files to UI immediately with different starting progress for visual distinction
    const tempDocuments: Document[] = validFiles.map((file, index) => ({
      id: uploadDocumentIds[index], // Use upload state manager IDs
      name: file.name,
      size: file.size,
      type: file.type,
      status: index === 0 ? "uploading" : "waiting", // Only first file starts immediately
      progress: index === 0 ? 0 : 0, // All start at 0 for clear distinction
      uploadDate: new Date().toISOString().split("T")[0],
      processingOrder: index,
      statusMessage: index === 0 ? 'File 1: Starting upload...' : `File ${index + 1}: Waiting in queue...`
    }))
    
    // Add all files to UI at once
    setDocuments((prev) => [...prev, ...tempDocuments])
    
    // Process files one by one with COMPLETELY isolated progress tracking
    for (let i = 0; i < validFiles.length; i++) {
      const file = validFiles[i]
      const tempDoc = tempDocuments[i]
      
      console.log(`📁 Starting isolated processing for file ${i + 1}/${validFiles.length}: ${file.name}`)
      
      // Mark current file as active and others as waiting
      setDocuments((prev) => 
        prev.map((doc) => {
          if (doc.id === tempDoc.id) {
            return { 
              ...doc, 
              status: "uploading", 
              progress: 0, // Always start from 0 for each file
              statusMessage: `File ${i + 1}/${validFiles.length}: Upload starting...` 
            }
          } else if (prev.findIndex(d => d.id === doc.id) === tempDocuments.findIndex(t => t.id === doc.id) && tempDocuments.findIndex(t => t.id === doc.id) > i) {
            return { 
              ...doc, 
              status: "waiting", 
              statusMessage: `File ${tempDocuments.findIndex(t => t.id === doc.id) + 1}/${validFiles.length}: Waiting in queue...` 
            }
          }
          return doc
        })
      )
      
      try {
        await processFileInIsolation(file, tempDoc.id, i + 1, validFiles.length)
        console.log(`✅ Isolated file ${i + 1}/${validFiles.length} completed: ${file.name}`)
      } catch (error) {
        console.error(`❌ Isolated file ${i + 1}/${validFiles.length} failed: ${file.name}`, error)
        // Mark as error but continue with next file
        setDocuments((prev) => 
          prev.map((doc) => 
            doc.id === tempDoc.id
              ? { 
                  ...doc, 
                  status: "error", 
                  progress: 0,
                  error: `Processing failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
                  statusMessage: `File ${i + 1}/${validFiles.length}: Failed ❌`
                }
              : doc
          )
        )
      }
      
      // Brief pause between files
      if (i < validFiles.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 800))
        
        // Show next file is preparing
        if (i + 1 < tempDocuments.length) {
          setDocuments((prev) => 
            prev.map((doc) => 
              doc.id === tempDocuments[i + 1].id
                ? { ...doc, statusMessage: `File ${i + 2}/${validFiles.length}: Next in queue...` }
                : doc
            )
          )
        }
      }
    }
    
    console.log('✅ All isolated files processed')
  }, [])

  // ULTIMATE FIX: Completely isolated file processor with unique progress tracking
  const processFileInIsolation = useCallback(async (file: File, isolatedDocId: string, currentFile: number, totalFiles: number) => {
    return new Promise<void>(async (resolve, reject) => {
      // Create completely isolated progress state for THIS file only
      const isolatedProgress = {
        value: 0,
        stage: 'init',
        isComplete: false,
        lastUpdate: Date.now(),
        fileId: isolatedDocId
      }
      
      let isolatedTimer: NodeJS.Timeout | null = null
      let completionTimer: NodeJS.Timeout

      // COMPLETE FIX: Function to update progress with strict never-decrease policy
      const updateIsolatedProgress = (backendProgress: number, statusMessage: string, newStage?: string) => {
        // CRITICAL: Ensure progress never decreases - use Math.max with current value
        const safeProgress = Math.max(isolatedProgress.value, Math.min(backendProgress, 90))
        isolatedProgress.value = safeProgress
        isolatedProgress.lastUpdate = Date.now()
        if (newStage) isolatedProgress.stage = newStage
        
        // Log with both backend and safe progress values for debugging
        console.log(`🔄 PROGRESS SYNC for ${file.name}: Backend=${backendProgress}% → UI=${safeProgress}% - ${statusMessage}`)
        
        // Update upload state manager
        const status = newStage === 'processing' || newStage === 'ai_processing' || newStage === 'ws_processing' ? 'processing' : 
                     newStage === 'preparing' || newStage === 'uploaded' ? 'uploading' : 'uploading'
        uploadStateManager.updateDocumentProgress(isolatedDocId, safeProgress, status)
        
        setDocuments((prev) => 
          prev.map((doc) => {
            // CRITICAL: Only update the document with this exact ID
            if (doc.id === isolatedDocId) {
              return { 
                ...doc, 
                progress: safeProgress, // Use safe progress that never decreases
                statusMessage: `${statusMessage} (${Math.round(safeProgress)}%)` // Show actual UI progress percentage
              }
            }
            // Leave ALL other documents completely untouched
            return doc
          })
        )
      }

      const completeIsolatedFile = () => {
        if (!isolatedProgress.isComplete) {
          isolatedProgress.isComplete = true
          if (isolatedTimer) clearInterval(isolatedTimer)
          clearTimeout(completionTimer)
          console.log(`✅ ISOLATED COMPLETION for ${file.name}`)
          resolve()
        }
      }

      try {
        console.log(`📤 ISOLATED UPLOAD START: ${file.name} (ID: ${isolatedDocId})`)
        
        // Phase 1: Upload initiation (0-5%)
        updateIsolatedProgress(2, `File ${currentFile}/${totalFiles}: Preparing upload...`, 'preparing')
        
        // Upload this single file with its own batch
        const uploadResponse = await ragApiClient.uploadFiles([file])
        const batchId = uploadResponse.batch_id
        
        console.log(`✅ ISOLATED UPLOAD SUCCESS: ${file.name}, Batch: ${batchId}`)

        // Phase 2: Upload complete, starting processing (5-15%)
        updateIsolatedProgress(10, `File ${currentFile}/${totalFiles}: Upload complete, starting AI processing...`, 'uploaded')
        
        // Update to processing status
        setDocuments((prev) => 
          prev.map((doc) => 
            doc.id === isolatedDocId 
              ? { 
                  ...doc, 
                  status: "processing", 
                  progress: 15,
                  batchId: batchId,
                  statusMessage: `File ${currentFile}/${totalFiles}: AI processing initiated...`
                }
              : doc
          )
        )
        
        isolatedProgress.value = 15
        isolatedProgress.stage = 'processing'

        // Phase 3: Gradual progress simulation for this file only
        isolatedTimer = setInterval(() => {
          if (!isolatedProgress.isComplete && isolatedProgress.value < 80) {
            const increment = Math.random() * 3 + 1 // Random 1-4% increments
            const newProgress = Math.min(isolatedProgress.value + increment, 80)
            updateIsolatedProgress(
              newProgress, 
              `File ${currentFile}/${totalFiles}: AI analysis ${Math.round(newProgress)}% complete...`, 
              'ai_processing'
            )
          }
        }, 4000) // Update every 4 seconds for this file only

        // Phase 4: WebSocket connection for real-time updates for THIS file only
        const isolatedWS = ragApiClient.createReliableWebSocket(batchId, (message: any) => {
          try {
            console.log(`📨 ISOLATED WebSocket for ${file.name}:`, message.type, message.data)

            switch (message.type) {
              case 'processing':
                if (message.data?.progress && typeof message.data.progress === 'number') {
                  const wsProgress = Math.max(isolatedProgress.value, Math.min(message.data.progress, 85))
                  updateIsolatedProgress(
                    wsProgress, 
                    `File ${currentFile}/${totalFiles}: ${message.data?.message || `Processing ${Math.round(wsProgress)}%...`}`,
                    'ws_processing'
                  )
                }
                break

              case 'completed':
              case 'file_processing_completed':
                console.log(`🎉 ISOLATED COMPLETION: ${file.name}`)
                
                // Stop gradual progress
                if (isolatedTimer) {
                  clearInterval(isolatedTimer)
                  isolatedTimer = null
                }
                
                        // Update to 100% completion with AI data
                        setDocuments((prev) => 
                          prev.map((doc) => {
                            if (doc.id === isolatedDocId) {
                              const completedData = message.data || {}
                              return {
                                ...doc,
                                status: "completed" as const,
                                progress: 100,
                                summary: completedData.summary || '',
                                themes: completedData.themes || [],
                                keywords: completedData.keywords || completedData.themes || [],
                                demographics: completedData.demographics || [],
                                mainTopics: completedData.themes || [],
                                keyInsights: completedData.insights ? [completedData.insights] : [],
                                language: completedData.language || 'English',
                                statusMessage: `File ${currentFile}/${totalFiles}: Processing complete`
                              }
                            }
                            return doc
                          })
                        )

                        // ENHANCED CROSS-TAB NOTIFICATION: Use document sync manager for immediate vault updates
                        console.log('📡 Documents: Using document sync manager for immediate vault notification')
                        const completedData = message.data || {}
                        documentSyncManager.emitUploadCompleted(
                          completedData.doc_id || isolatedDocId,
                          file.name,
                          'documents_page'
                        )
                        
                        // Also trigger immediate vault refresh for instant dropdown update
                        documentSyncManager.triggerImmediateVaultRefresh()
                        
                        // Update upload state manager
                        uploadStateManager.completeDocument(isolatedDocId)
                
                isolatedProgress.value = 100
                setTimeout(completeIsolatedFile, 600)
                break

              case 'error':
                console.error(`❌ ISOLATED ERROR for ${file.name}:`, message.data)
                
                if (isolatedTimer) {
                  clearInterval(isolatedTimer)
                  isolatedTimer = null
                }
                
                // Update upload state manager with error
                uploadStateManager.errorDocument(isolatedDocId, message.data?.error || 'Processing failed')
                
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.id === isolatedDocId) {
                      return { 
                        ...doc, 
                        status: "error", 
                        progress: 0,
                        error: message.data?.error || 'Processing failed',
                        statusMessage: `File ${currentFile}/${totalFiles}: Processing failed ❌`
                      }
                    }
                    return doc
                  })
                )
                reject(new Error(message.data?.error || 'Processing failed'))
                break

              default:
                // Handle any other progress updates
                if (message.data?.progress && typeof message.data.progress === 'number') {
                  const generalProgress = Math.max(isolatedProgress.value, Math.min(message.data.progress, 85))
                  updateIsolatedProgress(
                    generalProgress, 
                    `File ${currentFile}/${totalFiles}: ${message.data?.message || `Processing ${Math.round(generalProgress)}%...`}`,
                    'general_update'
                  )
                }
                break
            }
          } catch (error) {
            console.error(`❌ ISOLATED WebSocket error for ${file.name}:`, error)
          }
        })

        // Connect the isolated WebSocket
        await isolatedWS.connect()
        
        // Timeout for this isolated file (6 minutes)
        completionTimer = setTimeout(() => {
          if (!isolatedProgress.isComplete) {
            console.warn(`⏱️ ISOLATED TIMEOUT for ${file.name}`)
            
            if (isolatedTimer) {
              clearInterval(isolatedTimer)
              isolatedTimer = null
            }
            
            isolatedWS.disconnect()
            
            // Mark as completed due to timeout
            setDocuments((prev) => 
              prev.map((doc) => {
                if (doc.id === isolatedDocId) {
                  return { 
                    ...doc, 
                    status: "completed", 
                    progress: 100,
                    statusMessage: `File ${currentFile}/${totalFiles}: Completed (timeout) ⏱️`,
                    language: 'English'
                  }
                }
                return doc
              })
            )
            
            completeIsolatedFile()
          }
        }, 360000) // 6 minutes for this isolated file

        // Store isolated cleanup function
        ;(window as any)[`cleanup_isolated_${isolatedDocId}`] = () => {
          if (isolatedTimer) clearInterval(isolatedTimer)
          isolatedWS.disconnect()
          clearTimeout(completionTimer)
        }

      } catch (error) {
        console.error(`❌ ISOLATED UPLOAD FAILED for ${file.name}:`, error)
        
        setDocuments((prev) => 
          prev.map((doc) => 
            doc.id === isolatedDocId
              ? { 
                  ...doc, 
                  status: "error", 
                  progress: 0,
                  error: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}`,
                  statusMessage: `File ${currentFile}/${totalFiles}: Upload failed ❌`
                }
              : doc
          )
        )
        
        reject(error)
      }
    })
  }, [])

  const processFilesSequentially = useCallback(async (fileList: File[], currentFile: number, totalFiles: number) => {
    const file = fileList[0] // Only one file at a time

    // PROFESSIONAL STATE DEBUGGING AND VALIDATION
    console.log('=== SEQUENTIAL PROCESSING FILE', currentFile, 'of', totalFiles, '===')
    console.log('Processing file:', file.name)
    console.log('State - selectedProvider:', selectedProvider)
    console.log('State - selectedModel:', selectedModel)
    
    // Get current values from state with fallback logic
    let finalProvider = selectedProvider
    let finalModel = selectedModel
    
    // INTELLIGENT FALLBACK SYSTEM: Always ensure we have valid provider/model
    if (!finalProvider || !finalModel) {
      console.warn('State variables empty, applying intelligent fallback system...')
      
      // Priority-based provider fallback
      const providerPriority = ['gemini', 'openai', 'groq', 'deepseek', 'ollama']
      
      if (!finalProvider) {
        finalProvider = providerPriority.find(p => modelProviders.find(provider => provider.name === p)) || 
                      (modelProviders.length > 0 ? modelProviders[0].name : 'gemini')
        console.log('Fallback provider set to:', finalProvider)
      }
      
      const currentProvider = modelProviders.find(p => p.name === finalProvider)
      if (!finalModel && currentProvider && currentProvider.models.length > 0) {
        // Set default model based on provider
        if (finalProvider === 'gemini') {
          finalModel = 'gemini-1.5-flash'
        } else if (finalProvider === 'openai') {
          finalModel = 'gpt-4o-mini'
        } else if (finalProvider === 'groq') {
          finalModel = 'llama-3.1-8b-instant'
        } else {
          finalModel = currentProvider.models[0].name
        }
        console.log('Fallback model set to:', finalModel)
      }
      
      // Update state to match fallback values
      setSelectedProvider(finalProvider)
      setSelectedModel(finalModel)
      
      // Persist the fallback selections
      localStorage.setItem('sixthvault_preferred_provider', finalProvider)
      localStorage.setItem('sixthvault_preferred_model', finalModel)
    }
    
    // FINAL FALLBACK: If still no valid selection, use hardcoded defaults
    if (!finalProvider || !finalModel) {
      console.warn('Using hardcoded fallback - Gemini Flash')
      finalProvider = 'gemini'
      finalModel = 'gemini-1.5-flash'
      
      // Update state
      setSelectedProvider(finalProvider)
      setSelectedModel(finalModel)
      
      // Persist
      localStorage.setItem('sixthvault_preferred_provider', finalProvider)
      localStorage.setItem('sixthvault_preferred_model', finalModel)
    }

    // Create temporary document for UI with unique ID
    const tempDocument: Document = {
      id: `temp_${Math.random().toString(36).substr(2, 9)}_${Date.now()}`,
      name: file.name,
      size: file.size,
      type: file.type,
      status: "uploading",
      progress: 0,
      uploadDate: new Date().toISOString().split("T")[0],
      batchId: '', // Will be set after upload
      processingOrder: currentFile - 1, // Track processing order for sequential progress
    }

    setDocuments((prev) => [...prev, tempDocument])

    try {
      console.log(`📤 Uploading file ${currentFile}/${totalFiles}: ${file.name}`)
      
      const uploadResponse = await ragApiClient.uploadFiles(fileList)
      console.log('Upload response:', uploadResponse)
      const batchId = uploadResponse.batch_id

      // Update document to processing status
      setDocuments((prev) => 
        prev.map((doc) => {
          if (doc.id === tempDocument.id) {
            return { 
              ...doc, 
              status: "processing", 
              progress: 10,
              batchId: batchId,
              statusMessage: `Processing file ${currentFile}/${totalFiles}`
            }
          }
          return doc
        })
      )

      // Sequential Progress Management System for Single File
      let isCleanedUp = false
      let progressInterval: NodeJS.Timeout | null = null
      let connectionStatus = 'connecting'
      let batchProgressTracker = {
        totalFiles: 1, // Single file processing
        completedFiles: 0,
        currentProcessingFile: 0,
        isSequentialProcessing: true,
        baseProgressPerFile: 90, // Full 90% for this single file
        priorityFileCompleted: false
      }
      
      // Sequential Progress Calculator
      const calculateSequentialProgress = (fileIndex: number, stage: string): number => {
        const baseProgress = batchProgressTracker.baseProgressPerFile * fileIndex
        let stageProgress = 0
        
        // Define stage progress within each file's allocation
        switch (stage) {
          case 'uploading':
            stageProgress = 0.1 * batchProgressTracker.baseProgressPerFile
            break
          case 'processing_started':
            stageProgress = 0.2 * batchProgressTracker.baseProgressPerFile
            break
          case 'processing':
            stageProgress = 0.6 * batchProgressTracker.baseProgressPerFile
            break
          case 'completing':
            stageProgress = 0.9 * batchProgressTracker.baseProgressPerFile
            break
          case 'completed':
            stageProgress = batchProgressTracker.baseProgressPerFile
            break
        }
        
        return Math.min(baseProgress + stageProgress, 90)
      }
      
      // Update progress for all files in batch
      const updateBatchProgress = () => {
        setDocuments((prev) => 
          prev.map((doc) => {
            if (doc.batchId === batchId && doc.status === "processing") {
              // Calculate progress based on current processing state
              const currentProgress = calculateSequentialProgress(
                doc.processingOrder || 0,
                'processing'
              )
              
              // Ensure progress never decreases
              const newProgress = Math.max(doc.progress, Math.min(currentProgress, 90))
              return { ...doc, progress: newProgress }
            }
            return doc
          })
        )
      }
      
      // Smooth progress updater for sequential processing
      const startSequentialProgressUpdater = () => {
        progressInterval = setInterval(() => {
          if (batchProgressTracker.isSequentialProcessing) {
            updateBatchProgress()
          }
        }, 1000) // Update every second for smooth progress
      }
      
      const stopProgressFallback = () => {
        if (progressInterval) {
          clearInterval(progressInterval)
          progressInterval = null
        }
      }

      // Create reliable WebSocket connection with enhanced message handling
      const reliableWS = ragApiClient.createReliableWebSocket(batchId, (message: any) => {
        try {
          console.log('Enhanced WebSocket message:', message)

          // Handle all new message types from the enhanced backend
          switch (message.type) {
            case 'connection_status':
              connectionStatus = message.data?.status || 'unknown'
              console.log('Connection status update:', message.data)
              
              // Update UI with connection status if needed
              if (message.data?.status === 'established') {
                console.log('WebSocket connection established successfully')
              } else if (message.data?.status === 'error') {
                console.error('WebSocket connection error:', message.data)
              }
              return

            case 'flush_start':
              console.log('Starting to flush queued messages:', message.data?.count || 0)
              return

            case 'flush_complete':
              console.log('Completed flushing queued messages:', message.data)
              return

            case 'upload_started':
              console.log('Upload started:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "uploading", 
                      progress: 5,
                      statusMessage: message.data?.message || 'Upload started'
                    }
                  }
                  return doc
                })
              )
              return

            case 'upload_completed':
              console.log('Upload completed:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "processing", 
                      progress: 20,
                      statusMessage: message.data?.message || 'Upload completed, starting processing'
                    }
                  }
                  return doc
                })
              )
              return

            case 'file_upload_progress':
              console.log('File upload progress:', message.data)
              if (message.data?.filename) {
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && doc.name === message.data.filename) {
                      return { 
                        ...doc, 
                        progress: Math.max(doc.progress, message.data.progress || doc.progress),
                        statusMessage: message.data?.message || 'Uploading...'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'file_uploaded':
              console.log('File uploaded:', message.data)
              if (message.data?.filename) {
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && doc.name === message.data.filename) {
                      return { 
                        ...doc, 
                        status: "processing",
                        progress: 25,
                        statusMessage: message.data?.message || 'File uploaded successfully'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'processing_started':
              console.log('Processing started:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "processing", 
                      progress: Math.max(doc.progress, 30),
                      statusMessage: message.data?.message || 'AI processing started'
                    }
                  }
                  return doc
                })
              )
              return

            case 'processing_completed':
              console.log('Processing completed:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "completed", 
                      progress: 100,
                      statusMessage: message.data?.message || 'Processing completed successfully'
                    }
                  }
                  return doc
                })
              )
              stopProgressFallback()
              return

            case 'processing_error':
              console.error('Processing error:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "error", 
                      error: message.data?.error || 'Processing failed',
                      statusMessage: message.data?.message || 'Processing failed'
                    }
                  }
                  return doc
                })
              )
              stopProgressFallback()
              return

            case 'queued':
              console.log('Files queued:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      status: "processing", 
                      progress: Math.max(doc.progress, 15),
                      statusMessage: message.data?.message || 'Queued for processing'
                    }
                  }
                  return doc
                })
              )
              return

            case 'processing':
              // Enhanced processing updates with detailed progress
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      const progress = Math.max(doc.progress, message.data.progress || 50)
                      const stage = message.data.stage || 'processing'
                      const statusMessage = message.data.message || 'Processing...'
                      
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: progress,
                        processingStage: stage,
                        statusMessage: statusMessage
                      }
                    }
                    return doc
                  })
                )
              } else {
                // If no specific file mentioned, update all documents in batch
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId) {
                      const progress = Math.max(doc.progress, message.data?.progress || doc.progress)
                      return {
                        ...doc,
                        status: "processing",
                        progress: progress,
                        processingStage: message.data?.stage || doc.processingStage,
                        statusMessage: message.data?.message || doc.statusMessage
                      }
                    }
                    return doc
                  })
                )
              }
              return

            // CRITICAL FIX: Add missing real-time progress message handlers
            case 'parallel_progress':
              console.log('Parallel progress update:', message.data)
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: Math.max(doc.progress, message.data.overall_progress || doc.progress),
                        statusMessage: message.data.message || 'AI processing in parallel...'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'task_completed':
              console.log('Task completed:', message.data)
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: Math.max(doc.progress, 70), // Increment progress for each task
                        statusMessage: message.data.message || `${message.data.task} completed`
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'task_started':
              console.log('Task started:', message.data)
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        statusMessage: message.data.message || `Starting ${message.data.task}...`
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'task_progress':
              console.log('Task progress:', message.data)
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: Math.max(doc.progress, message.data.progress || doc.progress),
                        statusMessage: message.data.message || `${message.data.task} ${message.data.progress}%`
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'batch_progress':
              console.log('Batch progress:', message.data)
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: Math.max(doc.progress, message.data.progress || doc.progress),
                        statusMessage: message.data.message || 'Processing batch...'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'completed':
              // CRITICAL FIX: Handle individual file completion with proper deduplication
              if (message.data?.file || message.data?.filename) {
                const messageFileName = extractFilename(message.data.file || message.data.filename || '')
                console.log('🎯 INDIVIDUAL FILE COMPLETED:', messageFileName, 'Data:', message.data)
                
                setDocuments((prev) => {
                  // Find all matching documents (both processing and potentially duplicated)
                  const matchingDocs = prev.filter(doc => 
                    doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name) ||
                      (message.data.filename && doc.name === message.data.filename)
                    )
                  )
                  
                  if (matchingDocs.length === 0) return prev
                  
                  // DEDUPLICATION: Keep only the first matching document, remove duplicates
                  const primaryDoc = matchingDocs[0]
                  const duplicateIds = matchingDocs.slice(1).map(d => d.id)
                  
                  console.log('🔄 DEDUPLICATING DOCUMENTS:', {
                    filename: messageFileName,
                    totalMatches: matchingDocs.length,
                    primaryDocId: primaryDoc.id,
                    duplicatesToRemove: duplicateIds
                  })
                  
                  // Remove duplicates and update the primary document
                  const updatedDocs = prev
                    .filter(doc => !duplicateIds.includes(doc.id)) // Remove duplicates
                    .map((doc) => {
                      if (doc.id === primaryDoc.id) {
                        const completedData = message.data
                        
                        console.log('✅ UPDATING PRIMARY DOCUMENT WITH AI DATA:', {
                          filename: doc.name,
                          docId: doc.id,
                          summary: completedData.summary ? 'Present' : 'Missing',
                          insights: completedData.insights ? 'Present' : 'Missing',
                          themes: completedData.themes?.length || 0,
                          demographics: completedData.demographics?.length || 0
                        })
                        
                        return {
                          ...doc,
                          id: completedData.doc_id || doc.id,
                          status: "completed" as const,
                          progress: 100,
                          // CRITICAL: Immediately apply ALL AI-generated data
                          summary: completedData.summary || doc.summary || '',
                          themes: completedData.themes || doc.themes || [],
                          keywords: completedData.keywords || completedData.themes || doc.keywords || [],
                          demographics: completedData.demographics || doc.demographics || [],
                          mainTopics: completedData.themes || doc.mainTopics || [],
                          keyInsights: completedData.insights ? [completedData.insights] : (doc.keyInsights || []),
                          language: completedData.language || doc.language || 'English',
                          sentiment: doc.sentiment || 'neutral',
                          readingLevel: doc.readingLevel || 'intermediate',
                          processingTime: completedData.processing_time || doc.processingTime,
                          statusMessage: completedData.message || 'AI analysis completed - Summary & insights ready!'
                        }
                      }
                      return doc
                    })
                  
                  // Check if all documents in this batch are completed
                  const batchDocs = updatedDocs.filter(d => d.batchId === batchId)
                  const completedCount = batchDocs.filter(d => d.status === "completed").length
                  if (completedCount === batchDocs.length) {
                    setTimeout(stopProgressFallback, 1000)
                  }
                  
                  return updatedDocs
                })
              }
              return

            case 'file_processing_completed':
              // ADDITIONAL HANDLER: Handle file_processing_completed messages with AI data
              if (message.data?.filename) {
                const messageFileName = message.data.filename
                console.log('🎯 FILE PROCESSING COMPLETED:', messageFileName, 'Data:', message.data)
                
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      const completedData = message.data
                      
                      console.log('✅ UPDATING DOCUMENT STATE WITH FILE PROCESSING DATA:', {
                        filename: doc.name,
                        summary: completedData.summary ? 'Present' : 'Missing',
                        insights: completedData.insights ? 'Present' : 'Missing',
                        themes: completedData.themes?.length || 0,
                        demographics: completedData.demographics?.length || 0,
                        ai_analysis_complete: completedData.ai_analysis_complete
                      })
                      
                      return {
                        ...doc,
                        id: completedData.doc_id || doc.id,
                        status: "completed",
                        progress: 100,
                        // CRITICAL: Apply AI data if available
                        summary: completedData.summary || doc.summary || '',
                        themes: completedData.themes || doc.themes || [],
                        keywords: completedData.keywords || completedData.themes || doc.keywords || [],
                        demographics: completedData.demographics || doc.demographics || [],
                        mainTopics: completedData.themes || doc.mainTopics || [],
                        keyInsights: completedData.insights ? [completedData.insights] : (doc.keyInsights || []),
                        language: completedData.language || doc.language || 'English',
                        sentiment: doc.sentiment || 'neutral',
                        readingLevel: doc.readingLevel || 'intermediate',
                        processingTime: completedData.processing_time || doc.processingTime,
                        statusMessage: completedData.message || 'File processing completed with AI analysis'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'priority_file_completed':
              console.log('Priority file completed:', message.data)
              if (message.data?.file_index !== undefined) {
                setDocuments((prev) => 
                  prev.map((doc, index) => {
                    if (doc.batchId === batchId && index === message.data.file_index) {
                      return { 
                        ...doc, 
                        status: "completed", 
                        progress: 100,
                        statusMessage: message.data?.message || 'Priority processing completed - RAG is ready'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'background_file_completed':
              console.log('Background file completed:', message.data)
              if (message.data?.file_index !== undefined) {
                setDocuments((prev) => 
                  prev.map((doc, index) => {
                    if (doc.batchId === batchId && index === (message.data.file_index - 1)) {
                      return { 
                        ...doc, 
                        status: "completed", 
                        progress: 100,
                        statusMessage: message.data?.message || 'Background processing completed'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'rag_available':
              console.log('RAG available:', message.data)
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId) {
                    return { 
                      ...doc, 
                      statusMessage: message.data?.message || 'RAG is now available! First document processed and searchable'
                    }
                  }
                  return doc
                })
              )
              return

            case 'priority_processing_started':
              console.log('Priority processing started:', message.data)
              if (message.data?.file_index !== undefined) {
                setDocuments((prev) => 
                  prev.map((doc, index) => {
                    if (doc.batchId === batchId && index === message.data.file_index) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: 30,
                        statusMessage: message.data?.message || 'Priority processing initiated'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'background_processing_started':
              console.log('Background processing started:', message.data)
              if (message.data?.file_index !== undefined) {
                setDocuments((prev) => 
                  prev.map((doc, index) => {
                    if (doc.batchId === batchId && index === (message.data.file_index - 1)) {
                      return { 
                        ...doc, 
                        status: "processing", 
                        progress: 25,
                        statusMessage: message.data?.message || 'Background processing started'
                      }
                    }
                    return doc
                  })
                )
              }
              return

            case 'tier1_initiated':
              console.log('Tier 1 processing initiated:', message.data)
              setDocuments((prev) => 
                prev.map((doc, index) => {
                  if (doc.batchId === batchId && index === 0) {
                    return { 
                      ...doc, 
                      status: "processing", 
                      progress: 20,
                      statusMessage: message.data?.message || 'Priority processing starting'
                    }
                  }
                  return doc
                })
              )
              return

            case 'tier2_initiated':
              console.log('Tier 2 processing initiated:', message.data)
              setDocuments((prev) => 
                prev.map((doc, index) => {
                  if (doc.batchId === batchId && index > 0) {
                    return { 
                      ...doc, 
                      status: "processing", 
                      progress: 15,
                      statusMessage: message.data?.message || 'Background processing queued'
                    }
                  }
                  return doc
                })
              )
              return

            case 'background_batch_started':
              console.log('Background batch started:', message.data)
              setDocuments((prev) => 
                prev.map((doc, index) => {
                  if (doc.batchId === batchId && index > 0 && doc.status === "processing") {
                    return { 
                      ...doc, 
                      progress: Math.max(doc.progress, 20),
                      statusMessage: message.data?.message || 'Background batch processing started'
                    }
                  }
                  return doc
                })
              )
              return

            case 'error':
              // For error messages, match by filename or apply to all
              if (message.data?.file) {
                const messageFileName = extractFilename(message.data.file)
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && (
                      doc.name === messageFileName || 
                      doc.name.includes(messageFileName) || 
                      messageFileName.includes(doc.name)
                    )) {
                      return { 
                        ...doc, 
                        status: "error", 
                        error: message.data.error || 'Processing failed',
                        statusMessage: message.data.message || 'Processing failed'
                      }
                    }
                    return doc
                  })
                )
              } else {
                // If no specific file, mark all processing documents as error
                setDocuments((prev) => 
                  prev.map((doc) => {
                    if (doc.batchId === batchId && doc.status === "processing") {
                      return {
                        ...doc,
                        status: "error",
                        error: message.data?.error || 'Processing failed',
                        statusMessage: message.data?.message || 'Processing failed'
                      }
                    }
                    return doc
                  })
                )
              }
              stopProgressFallback()
              return

            case 'connection_failed':
              console.error('Connection failed after multiple attempts:', message.data)
              // Show user-friendly error message
              setDocuments((prev) => 
                prev.map((doc) => {
                  if (doc.batchId === batchId && doc.status === "processing") {
                    return {
                      ...doc,
                      statusMessage: 'Connection issues detected - processing may continue in background'
                    }
                  }
                  return doc
                })
              )
              return

                case 'batch_completed':
                  console.log('Batch processing completed:', message.data)
                  setDocuments((prev) => 
                    prev.map((doc) => {
                      if (doc.batchId === batchId) {
                        return { 
                          ...doc, 
                          status: "completed", 
                          progress: 100,
                          statusMessage: message.data?.message || 'All processing completed successfully'
                        }
                      }
                      return doc
                    })
                  )
                  
                  // Remove from active batches when batch is fully completed
                  try {
                    const currentBatches = JSON.parse(localStorage.getItem('sixthvault_processing_batches') || '[]')
                    const updatedBatches = currentBatches.filter((b: any) => b.batchId !== batchId)
                    localStorage.setItem('sixthvault_processing_batches', JSON.stringify(updatedBatches))
                    console.log('Background Processing: Removed completed batch from localStorage:', batchId)
                  } catch (error) {
                    console.error('Background Processing: Failed to remove completed batch:', error)
                  }
                  return

                case 'websocket_closing':
                  console.log('WebSocket closing notification:', message.data)
                  // Show user-friendly message that processing is complete
                  setDocuments((prev) => 
                    prev.map((doc) => {
                      if (doc.batchId === batchId) {
                        return { 
                          ...doc, 
                          statusMessage: 'Processing completed - Connection closing'
                        }
                      }
                      return doc
                    })
                  )
                  return

                case 'heartbeat':
                case 'pong':
                  // Handle heartbeat messages silently
                  return

                default:
                  console.log('Unhandled message type:', message.type, message.data)
                  return
          }
        } catch (error) {
          console.error('Error handling enhanced WebSocket message:', error)
        }
      })

      // Start the reliable connection and fallback progress
        reliableWS.connect().then(() => {
          console.log('Reliable WebSocket connection established for batch:', batchId)
          connectionStatus = 'connected'
          batchProgressTracker.isSequentialProcessing = true
          startSequentialProgressUpdater()
        }).catch((error) => {
          console.error('Failed to establish reliable WebSocket connection:', error)
          connectionStatus = 'failed'
          // Still start fallback progress even if WebSocket fails
          batchProgressTracker.isSequentialProcessing = true
          startSequentialProgressUpdater()
        })

      // Cleanup function for WebSocket and intervals
      const cleanup = () => {
        isCleanedUp = true
        reliableWS.disconnect()
        stopProgressFallback()
      }

      // Store cleanup function for potential later use
      ;(window as any)[`cleanup_${batchId}`] = cleanup

      // Auto cleanup after 15 minutes (increased for longer processing)
      setTimeout(() => {
        if (!isCleanedUp) {
          cleanup()
        }
      }, 900000)

    } catch (error) {
      console.error('Upload failed:', error)
      // Update temp document to error state
      setDocuments((prev) => 
        prev.map((doc) => 
          doc.id === tempDocument.id
            ? { ...doc, status: "error", error: `Upload failed: ${error instanceof Error ? error.message : 'Unknown error'}` }
            : doc
        )
      )
    }
  }, [])

  const handleSummarize = useCallback(async (docId: string) => {
    const doc = documents.find((d) => d.id === docId)
    if (!doc || !doc.content) return

    setIsProcessing(true)
    try {
      const summary = await aiService.generateSummary(doc.name, doc.content)

      setDocuments((prev) => prev.map((d) => (d.id === docId ? { ...d, summary } : d)))
      // Note: Backend will handle persistence automatically
    } catch (error) {
      console.error("Error generating summary:", error)
      alert("AI summarization failed. The document is still available for manual review.")
    } finally {
      setIsProcessing(false)
    }
  }, [documents])

  const handleAddTag = (docId: string, tag: string) => {
    const tagToAdd = tag || newTags[docId]
    if (!tagToAdd?.trim()) return

    setDocuments((prev) =>
      prev.map((d) => (d.id === docId ? { ...d, themes: [...(d.themes || []), tagToAdd.trim()] } : d)),
    )

    // Note: Backend will handle persistence automatically

    setNewTags((prev) => ({ ...prev, [docId]: "" }))
  }

  const handleRemoveTag = (docId: string, tagToRemove: string) => {
    setDocuments((prev) =>
      prev.map((d) => (d.id === docId ? { ...d, themes: d.themes?.filter((tag) => tag !== tagToRemove) || [] } : d)),
    )

    // Note: Backend will handle persistence automatically
  }

  const handleAddKeyword = (docId: string, keyword: string) => {
    const keywordToAdd = keyword || newKeywords[docId]
    if (!keywordToAdd?.trim()) return

    setDocuments((prev) =>
      prev.map((d) => (d.id === docId ? { ...d, keywords: [...(d.keywords || []), keywordToAdd.trim()] } : d)),
    )

    // Note: Backend will handle persistence automatically

    setNewKeywords((prev) => ({ ...prev, [docId]: "" }))
  }

  const handleRemoveKeyword = (docId: string, keywordToRemove: string) => {
    setDocuments((prev) =>
      prev.map((d) =>
        d.id === docId ? { ...d, keywords: d.keywords?.filter((keyword) => keyword !== keywordToRemove) || [] } : d,
      ),
    )

    // Note: Backend will handle persistence automatically
  }

  const handleAddDemographic = (docId: string, demographic: string) => {
    const demographicToAdd = demographic || newDemographics[docId]
    if (!demographicToAdd?.trim()) return

    setDocuments((prev) =>
      prev.map((d) =>
        d.id === docId
          ? {
              ...d,
              demographics: [...(d.demographics || []), demographicToAdd.trim()],
            }
          : d,
      ),
    )

    // Note: Backend will handle persistence automatically

    setNewDemographics((prev) => ({ ...prev, [docId]: "" }))
  }

  const handleRemoveDemographic = (docId: string, demographicToRemove: string) => {
    setDocuments((prev) =>
      prev.map((d) =>
        d.id === docId
          ? {
              ...d,
              demographics: d.demographics?.filter((demo) => demo !== demographicToRemove) || [],
            }
          : d,
      ),
    )

    // Note: Backend will handle persistence automatically
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  // Professional delete handler that opens the dialog
  const handleDeleteDocument = useCallback((docId: string) => {
    const doc = documents.find(d => d.id === docId)
    if (!doc) return

    setDeleteDialog({
      isOpen: true,
      document: doc,
      isDeleting: false
    })
  }, [documents])

  // Actual delete function called from the dialog
  const confirmDeleteDocument = useCallback(async () => {
    if (!deleteDialog.document) return

    const documentToDelete = deleteDialog.document
    
    // Set loading state but keep dialog open
    setDeleteDialog(prev => ({ ...prev, isDeleting: true }))

    try {
      console.log('🔄 Documents: Starting backend deletion for:', documentToDelete.name)
      
      // Call backend delete API and wait for response
      const success = await ragApiClient.deleteDocument(documentToDelete.id)
      
      if (!success) {
        throw new Error('Backend delete operation failed')
      }

      console.log('✅ Documents: Backend deletion successful for:', documentToDelete.name)
      
      // FRONTEND REMOVAL: Remove document from UI after successful backend deletion
      console.log('🚀 Documents: Removing document from frontend after backend success:', documentToDelete.name)
      
      // Remove from local state
      setDocuments((prev) => prev.filter((d) => d.id !== documentToDelete.id))
      
      // Clear selected document if it's the one being deleted
      if (selectedDocument?.id === documentToDelete.id) {
        setSelectedDocument(null)
      }

      // CROSS-TAB NOTIFICATION: Signal all other tabs to refresh
      console.log('📡 Documents: Signaling all tabs for document removal')
      localStorage.setItem('sixthvault_document_delete_event', JSON.stringify({
        documentId: documentToDelete.id,
        documentName: documentToDelete.name,
        timestamp: Date.now()
      }))
      localStorage.removeItem('sixthvault_document_delete_event') // Trigger storage event
      
      // CACHE INVALIDATION: Clear all document-related caches after successful backend deletion
      console.log('🗑️ Documents: Clearing document caches after backend deletion')
      
      // Clear document store cache
      if (typeof window !== 'undefined') {
        // Use cache manager to clear document caches
        const { cacheManager } = await import('@/lib/cache-manager')
        cacheManager.clearDocumentCaches()
        
        // Clear localStorage caches
        Object.keys(localStorage).forEach(key => {
          if (key.includes('documents') || key.includes('vault') || key.includes('sixthvault')) {
            localStorage.removeItem(key)
          }
        })
        
        // Clear sessionStorage caches
        Object.keys(sessionStorage).forEach(key => {
          if (key.includes('documents') || key.includes('vault') || key.includes('sixthvault')) {
            sessionStorage.removeItem(key)
          }
        })
      }
      
      // Trigger cache invalidation signal for other components
      localStorage.setItem('sixthvault_cache_invalidate_documents', Date.now().toString())
      localStorage.removeItem('sixthvault_cache_invalidate_documents')
      
      // VAULT STATE REFRESH: Call the vault state provider to refresh documents
      try {
        await refreshDocuments()
        console.log('✅ Documents: Vault state refreshed after backend deletion')
      } catch (refreshError) {
        console.error('❌ Documents: Failed to refresh vault state:', refreshError)
      }
      
      // Force refresh of document store
      try {
        await documentStore.refreshDocuments()
        console.log('✅ Documents: Document store refreshed after backend deletion')
      } catch (refreshError) {
        console.error('❌ Documents: Failed to refresh document store:', refreshError)
      }

      // Close dialog after successful deletion
      setDeleteDialog({
        isOpen: false,
        document: null,
        isDeleting: false
      })
      
    } catch (error) {
      console.error('❌ Documents: Backend deletion failed for:', documentToDelete.name, error)
      
      // Show error message to user
      alert(`Failed to delete "${documentToDelete.name}": ${error instanceof Error ? error.message : 'Unknown error'}. Please try again.`)
      
      // Reset dialog state to allow retry
      setDeleteDialog(prev => ({ ...prev, isDeleting: false }))
    }
  }, [deleteDialog.document, selectedDocument, refreshDocuments])

  // Cancel delete dialog
  const cancelDeleteDocument = useCallback(() => {
    setDeleteDialog({
      isOpen: false,
      document: null,
      isDeleting: false
    })
  }, [])

  const getSentimentColor = (sentiment: string) => {
    switch (sentiment?.toLowerCase()) {
      case "positive":
        return "bg-green-100 text-green-800"
      case "negative":
        return "bg-red-100 text-red-800"
      case "neutral":
        return "bg-blue-100 text-blue-800"
      case "mixed":
        return "bg-purple-100 text-purple-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  const getReadingLevelColor = (level: string) => {
    switch (level?.toLowerCase()) {
      case "basic":
        return "bg-green-100 text-green-800"
      case "intermediate":
        return "bg-blue-100 text-blue-800"
      case "advanced":
        return "bg-purple-100 text-purple-800"
      case "technical":
        return "bg-orange-100 text-orange-800"
      default:
        return "bg-gray-100 text-gray-800"
    }
  }

  // Filter documents based on search and status - memoized for performance
  const filteredDocuments = useMemo(() => {
    return documents.filter(doc => {
      const matchesSearch = doc.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
                           doc.themes?.some(theme => theme.toLowerCase().includes(searchQuery.toLowerCase())) ||
                           doc.keywords?.some(keyword => keyword.toLowerCase().includes(searchQuery.toLowerCase()))
      
      const matchesStatus = filterStatus === "all" || doc.status === filterStatus
      
      return matchesSearch && matchesStatus
    })
  }, [documents, searchQuery, filterStatus])

  // Calculate statistics - memoized for performance
  const stats = useMemo(() => {
    return {
      total: documents.length,
      completed: documents.filter(d => d.status === "completed").length,
      processing: documents.filter(d => d.status === "processing").length,
      errors: documents.filter(d => d.status === "error").length,
      totalSize: documents.reduce((sum, doc) => sum + doc.size, 0),
      languages: Array.from(new Set(documents.map(d => d.language).filter(Boolean))),
      avgProcessingTime: documents.filter(d => d.processingTime).reduce((sum, d) => sum + (d.processingTime || 0), 0) / Math.max(documents.filter(d => d.processingTime).length, 1)
    }
  }, [documents])

  // Show loading state while documents are being initialized
  if (isInitialLoading) {
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
                    <p className="text-sm text-gray-500">{loadingMessage}</p>
                  </div>
                </div>
              </div>
              
              <div className="flex items-center space-x-4">
                <div className="flex items-center space-x-2 bg-blue-50 px-3 py-2 rounded-lg">
                  <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                  <span className="text-sm font-medium text-blue-700">Loading Documents</span>
                </div>
                <Button 
                  onClick={logout}
                  className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200"
                >
                  <LogOut className="w-4 h-4 mr-2" />
                  Logout
                </Button>
              </div>
            </div>
          </div>
        </div>

        {/* Loading Content */}
        <div className="flex-1 p-6">
          <div className="text-center py-16">
            <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-6 animate-pulse">
              <Database className="w-10 h-10 text-blue-600" />
            </div>
            <h2 className="text-2xl font-bold text-gray-900 mb-3">Loading Your Documents</h2>
            <p className="text-gray-600 mb-6 max-w-md mx-auto">
              {loadingMessage}
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
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-white relative overflow-hidden">
      {/* Beautiful flowing wave background - Hidden on mobile for clean look */}
      <div className="absolute inset-0 z-0 hidden md:block">
        <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
          <defs>
            <linearGradient id="waveGradient1-page" x1="0%" y1="0%" x2="100%" y2="100%">
              <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.6"/>
              <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.4"/>
              <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.3"/>
              <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.2"/>
              <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
            </linearGradient>
            <linearGradient id="waveGradient2-page" x1="100%" y1="0%" x2="0%" y2="100%">
              <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.5"/>
              <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.3"/>
              <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.2"/>
              <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.15"/>
              <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
            </linearGradient>
            <linearGradient id="waveGradient3-page" x1="50%" y1="0%" x2="50%" y2="100%">
              <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.4"/>
              <stop offset="25%" stopColor="#ffe0b2" stopOpacity="0.3"/>
              <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.2"/>
              <stop offset="75%" stopColor="#ffb74d" stopOpacity="0.15"/>
              <stop offset="100%" stopColor="#ffa726" stopOpacity="0.1"/>
            </linearGradient>
          </defs>
          
          {/* Main flowing wave patterns */}
          <g stroke="url(#waveGradient1-page)" strokeWidth="1.5" fill="none" opacity="0.8">
            <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
            <path d="M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200"/>
            <path d="M0,240 Q180,160 360,220 T720,200 Q900,180 1080,220 T1440,200"/>
            <path d="M0,260 Q300,180 600,240 T1200,220 Q1320,200 1440,240"/>
          </g>
          
          <g stroke="url(#waveGradient2-page)" strokeWidth="1.2" fill="none" opacity="0.7">
            <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
            <path d="M0,320 Q360,240 720,300 T1440,280"/>
            <path d="M0,340 Q240,260 480,320 T960,300 Q1200,280 1440,320"/>
            <path d="M0,360 Q420,280 840,340 T1440,320"/>
          </g>
          
          <g stroke="url(#waveGradient3-page)" strokeWidth="1.0" fill="none" opacity="0.6">
            <path d="M0,380 Q180,300 360,360 T720,340 Q900,320 1080,360 T1440,340"/>
            <path d="M0,400 Q270,320 540,380 T1080,360 Q1260,340 1440,380"/>
            <path d="M0,420 Q210,340 420,400 T840,380 Q1020,360 1200,400 T1440,380"/>
          </g>
          
          {/* Filled wave areas for depth */}
          <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#waveGradient1-page)" opacity="0.08"/>
          <path d="M0,350 Q300,270 600,330 T1200,310 Q1320,290 1440,330 L1440,800 L0,800 Z" fill="url(#waveGradient2-page)" opacity="0.06"/>
          <path d="M0,450 Q360,370 720,430 T1440,410 L1440,800 L0,800 Z" fill="url(#waveGradient3-page)" opacity="0.04"/>
          
          {/* Additional flowing lines for complexity */}
          <g stroke="url(#waveGradient1-page)" strokeWidth="0.8" fill="none" opacity="0.5">
            <path d="M0,180 Q120,100 240,160 T480,140 Q600,120 720,160 T960,140 Q1080,120 1200,160 T1440,140"/>
            <path d="M0,500 Q150,420 300,480 T600,460 Q750,440 900,480 T1200,460 Q1320,440 1440,480"/>
          </g>
          
          <g stroke="url(#waveGradient2-page)" strokeWidth="0.6" fill="none" opacity="0.4">
            <path d="M0,160 Q90,80 180,140 T360,120 Q450,100 540,140 T720,120 Q810,100 900,140 T1080,120 Q1170,100 1260,140 T1440,120"/>
            <path d="M0,520 Q135,440 270,500 T540,480 Q675,460 810,500 T1080,480 Q1215,460 1350,500 T1440,480"/>
          </g>
        </svg>
      </div>
      {/* Modern Header */}
      <div className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
                className="lg:hidden"
              >
                <Menu className="w-5 h-5" />
              </Button>
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center">
                  <Database className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h1 className="text-2xl font-bold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
                    Document Intelligence Hub
                  </h1>
                  <p className="text-sm text-gray-500">Enterprise Document Management & AI Analysis</p>
                </div>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <div className="hidden md:flex items-center space-x-2 bg-green-50 px-3 py-2 rounded-lg">
                <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                <span className="text-sm font-medium text-green-700">AI Analysis Active</span>
              </div>
              <Button variant="ghost" size="sm">
                <Bell className="w-5 h-5" />
              </Button>
              <Button variant="ghost" size="sm">
                <HelpCircle className="w-5 h-5" />
              </Button>
              <Button 
                onClick={logout}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile: Gallery-First Layout */}
      <div className="md:hidden">
        {/* Mobile Header */}
        <div className="flex items-center justify-between p-4 bg-white border-b border-gray-200">
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setSidebarCollapsed(!sidebarCollapsed)}
          >
            <Menu className="w-5 h-5" />
          </Button>
          <h1 className="text-lg font-semibold text-gray-800">Documents</h1>
          <Button variant="ghost" size="sm">
            <Search className="w-5 h-5" />
          </Button>
        </div>

        {/* Mobile Document Gallery */}
        <div className="p-4 pb-24">
          {documents.length === 0 ? (
            <div className="text-center py-12">
              <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                <FileText className="w-8 h-8 text-gray-400" />
              </div>
              <h3 className="text-lg font-medium text-gray-900 mb-2">No documents yet</h3>
              <p className="text-gray-500 mb-6">Upload your first document to get started</p>
              <Button
                onClick={() => document.getElementById("mobile-file-upload")?.click()}
                className="bg-blue-500 hover:bg-blue-600 text-white"
              >
                <Plus className="w-4 h-4 mr-2" />
                Upload Document
              </Button>
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.rtf"
                onChange={handleFileSelect}
                className="hidden"
                id="mobile-file-upload"
              />
            </div>
          ) : (
            <div className="space-y-4">
              {filteredDocuments.map((doc) => (
                <div
                  key={doc.id}
                  className="bg-white rounded-xl p-4 border border-gray-200 shadow-sm hover:shadow-md transition-all duration-200"
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                      <FileText className="w-5 h-5 text-blue-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <h4 className="font-medium text-gray-900 truncate">{doc.name}</h4>
                      <p className="text-sm text-gray-500">{formatFileSize(doc.size)}</p>
                    </div>
                    <div className="flex items-center space-x-2">
                      {doc.status === "completed" && <CheckCircle className="w-5 h-5 text-green-500" />}
                      {doc.status === "processing" && <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />}
                      {doc.status === "error" && <AlertCircle className="w-5 h-5 text-red-500" />}
                    </div>
                  </div>
                  
                  {doc.status !== "completed" && (
                    <div className="space-y-2 mb-3">
                      <div className="flex justify-between text-sm">
                        <span className="text-gray-600">{doc.statusMessage || "Processing..."}</span>
                        <span className="font-medium text-gray-700">{Math.round(doc.progress)}%</span>
                      </div>
                      <Progress value={Math.round(doc.progress)} className="h-2" />
                    </div>
                  )}

                  {doc.status === "completed" && doc.themes && (
                    <div className="flex flex-wrap gap-2 mb-3">
                      {doc.themes.slice(0, 2).map((theme, index) => (
                        <Badge key={index} variant="secondary" className="text-xs">
                          {theme}
                        </Badge>
                      ))}
                      {doc.themes.length > 2 && (
                        <Badge variant="outline" className="text-xs">
                          +{doc.themes.length - 2}
                        </Badge>
                      )}
                    </div>
                  )}

                  <div className="flex items-center justify-between">
                    <span className="text-xs text-gray-500">{doc.uploadDate}</span>
                    <div className="flex items-center space-x-2">
                      <Button variant="ghost" size="sm" className="h-8 w-8 p-0">
                        <Eye className="w-4 h-4" />
                      </Button>
                      <Button
                        variant="ghost"
                        size="sm"
                        className="h-8 w-8 p-0 text-red-500 hover:text-red-700 hover:bg-red-50"
                        onClick={() => handleDeleteDocument(doc.id)}
                      >
                        <Trash2 className="w-4 h-4" />
                      </Button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Mobile Floating Action Button */}
        <div className="fixed bottom-6 right-6 z-50">
          <Button
            onClick={() => document.getElementById("mobile-file-upload")?.click()}
            className="w-14 h-14 rounded-full bg-blue-500 hover:bg-blue-600 text-white shadow-lg hover:shadow-xl transition-all duration-300"
          >
            <Plus className="w-6 h-6" />
          </Button>
        </div>

        {/* Mobile Menu Overlay */}
        {!sidebarCollapsed && (
          <div className="fixed inset-0 z-40 bg-black/20 backdrop-blur-sm" onClick={() => setSidebarCollapsed(true)}>
            <div className="absolute left-0 top-0 bottom-0 w-80 bg-white shadow-xl" onClick={(e) => e.stopPropagation()}>
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-gray-800">Menu</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => setSidebarCollapsed(true)}
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </div>
              </div>
              
              <div className="p-4 space-y-4">
                <div>
                  <h3 className="text-sm font-medium text-gray-600 mb-3">Quick Actions</h3>
                  <div className="space-y-2">
                    <Button
                      variant="ghost"
                      className="w-full justify-start text-sm"
                      onClick={() => document.getElementById("mobile-file-upload")?.click()}
                    >
                      <Upload className="w-4 h-4 mr-3" />
                      Upload Documents
                    </Button>
                    <Link href="/vault">
                      <Button variant="ghost" className="w-full justify-start text-sm">
                        <Home className="w-4 h-4 mr-3" />
                        Back to Vault
                      </Button>
                    </Link>
                    <Button 
                      onClick={logout}
                      variant="ghost" 
                      className="w-full justify-start text-sm text-red-600 hover:text-red-700 hover:bg-red-50"
                    >
                      <LogOut className="w-4 h-4 mr-3" />
                      Logout
                    </Button>
                  </div>
                </div>

                <div className="pt-4 border-t border-gray-200">
                  <div className="bg-gray-50 rounded-lg p-3">
                    <h4 className="text-sm font-medium text-gray-700 mb-2">Statistics</h4>
                    <div className="space-y-1 text-xs text-gray-600">
                      <div className="flex justify-between">
                        <span>Total:</span>
                        <span>{stats.total}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Processed:</span>
                        <span>{stats.completed}</span>
                      </div>
                      <div className="flex justify-between">
                        <span>Size:</span>
                        <span>{formatFileSize(stats.totalSize)}</span>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Desktop: Keep original layout */}
      <div className="hidden md:flex h-[calc(100vh-80px)]">
        {/* Enhanced Sidebar */}
        <div className={`${sidebarCollapsed ? 'w-16' : 'w-80'} bg-white/90 backdrop-blur-xl border-r border-gray-200/50 transition-all duration-300 flex flex-col`}>
          {/* Logo Section */}
          <div className="p-3 border-b border-gray-200/50">
            {!sidebarCollapsed && (
              <div className="w-full h-16 flex items-center justify-center">
                <div className="max-w-full max-h-full">
                  <SixthvaultLogo size="large" />
                </div>
              </div>
            )}
            {sidebarCollapsed && (
              <div className="flex justify-center">
                <div className="w-8 h-8 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-lg flex items-center justify-center">
                  <Shield className="w-4 h-4 text-white" />
                </div>
              </div>
            )}
          </div>

          {/* Navigation */}
          <div className="flex-1 p-4 space-y-2">
            {[
              { id: "dashboard", label: "Dashboard", icon: BarChart3, color: "blue" },
              { id: "upload", label: "Upload & Process", icon: Upload, color: "green" },
              { id: "analyze", label: "AI Analysis", icon: Brain, color: "purple" },
              { id: "insights", label: "Insights", icon: Lightbulb, color: "yellow" },
              { id: "manage", label: "Manage Tags", icon: Tag, color: "pink" },
              { id: "demographics", label: "Demographics", icon: Users, color: "indigo" }
            ].map((tab) => {
              const Icon = tab.icon
              return (
                <Button
                  key={tab.id}
                  onClick={() => setActiveTab(tab.id)}
                  className={`w-full justify-start h-12 transition-all duration-200 ${
                    activeTab === tab.id
                      ? `bg-gradient-to-r from-${tab.color}-50 to-${tab.color}-100 text-${tab.color}-700 border-${tab.color}-200 shadow-sm`
                      : "hover:bg-gray-50 text-gray-600"
                  }`}
                  variant={activeTab === tab.id ? "secondary" : "ghost"}
                >
                  <Icon className={`w-5 h-5 ${sidebarCollapsed ? '' : 'mr-3'}`} />
                  {!sidebarCollapsed && <span className="font-medium">{tab.label}</span>}
                  {!sidebarCollapsed && activeTab === tab.id && (
                    <ChevronRight className="w-4 h-4 ml-auto" />
                  )}
                </Button>
              )
            })}
          </div>

          {/* Stats Panel */}
          {!sidebarCollapsed && (
            <div className="p-4 border-t border-gray-200/50">
              <div className="bg-gradient-to-br from-gray-50 to-gray-100 rounded-xl p-4 space-y-3">
                <h3 className="font-semibold text-gray-800 text-sm">Quick Stats</h3>
                <div className="grid grid-cols-2 gap-3">
                  <div className="text-center">
                    <div className="text-2xl font-bold text-blue-600">{stats.completed}</div>
                    <div className="text-xs text-gray-500">Processed</div>
                  </div>
                  <div className="text-center">
                    <div className="text-2xl font-bold text-green-600">{formatFileSize(stats.totalSize)}</div>
                    <div className="text-xs text-gray-500">Total Size</div>
                  </div>
                </div>
                <div className="pt-2 border-t border-gray-200">
                  <div className="flex items-center justify-between text-xs">
                    <span className="text-gray-500">Languages:</span>
                    <span className="font-medium text-gray-700">{stats.languages.length}</span>
                  </div>
                </div>
              </div>
            </div>
          )}

          {/* Back to Vault */}
          <div className="p-4 border-t border-gray-200/50">
            <Link href="/vault">
              <Button className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
                {!sidebarCollapsed && (
                  <>
                    <Home className="w-4 h-4 mr-2" />
                    Back to Vault
                  </>
                )}
                {sidebarCollapsed && <Home className="w-4 h-4" />}
              </Button>
            </Link>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col overflow-hidden">
          {/* Search and Filter Bar */}
          <div className="bg-white/80 backdrop-blur-xl border-b border-gray-200/50 p-6">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between space-y-4 lg:space-y-0">
              <div className="flex-1 max-w-md">
                <div className="relative">
                  <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-5 h-5" />
                  <Input
                    placeholder="Search documents, tags, keywords..."
                    value={searchQuery}
                    onChange={(e) => setSearchQuery(e.target.value)}
                    className="pl-10 bg-white/50 border-gray-200 focus:border-blue-500 focus:ring-blue-500"
                  />
                </div>
              </div>
              
              <div className="flex items-center space-x-3">
                <div className="flex items-center space-x-2">
                  <Filter className="w-4 h-4 text-gray-500" />
                  <select
                    value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}
                    className="bg-white/50 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value="all">All Status</option>
                    <option value="completed">Completed</option>
                    <option value="processing">Processing</option>
                    <option value="error">Error</option>
                  </select>
                </div>

                {/* AI Provider Selection */}
                <div className="flex items-center space-x-2">
                  <Brain className="w-4 h-4 text-gray-500" />
                  <select
                    value={selectedProvider}
                    onChange={(e) => setSelectedProvider(e.target.value)}
                    className="bg-white/50 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    {modelProviders.map((provider) => (
                      <option key={provider.name} value={provider.name}>
                        {provider.displayName}
                        {provider.name === 'ollama' && provider.models.length > 0 && ' ✓'}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Model Selection */}
                <div className="flex items-center space-x-2">
                  <Settings className="w-4 h-4 text-gray-500" />
                  <select
                    value={selectedModel}
                    onChange={(e) => setSelectedModel(e.target.value)}
                    className="bg-white/50 border border-gray-200 rounded-lg px-3 py-2 text-sm focus:border-blue-500 focus:ring-blue-500"
                  >
                    <option value="">Select model...</option>
                    {availableModels.map((model) => (
                      <option key={model.name} value={model.name}>
                        {model.displayName}
                        {model.isLocal && ' (Local)'}
                      </option>
                    ))}
                  </select>
                </div>

                {/* Current Selection Indicator */}
                {selectedProvider && selectedModel && (
                  <div className="flex items-center space-x-2 bg-blue-50 px-3 py-2 rounded-lg border border-blue-200">
                    <div className="w-2 h-2 bg-blue-500 rounded-full animate-pulse"></div>
                    <span className="text-sm font-medium text-blue-700">
                      Using: {modelProviders.find(p => p.name === selectedProvider)?.displayName} / {availableModels.find(m => m.name === selectedModel)?.displayName}
                    </span>
                  </div>
                )}
                
                <Button variant="outline" size="sm">
                  <RefreshCw className="w-4 h-4 mr-2" />
                  Refresh
                </Button>
              </div>
            </div>
          </div>

          {/* Content Area */}
          <div className="flex-1 overflow-y-auto p-6">
            {/* Dashboard Tab */}
            {activeTab === "dashboard" && (
              <div className="space-y-6">
                {/* Stats Cards */}
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card className="bg-gradient-to-br from-blue-50 to-blue-100 border-blue-200 hover:shadow-lg transition-all duration-300">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-blue-600 text-sm font-medium">Total Documents</p>
                          <p className="text-3xl font-bold text-blue-900">{stats.total}</p>
                        </div>
                        <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                          <FileText className="w-6 h-6 text-white" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-green-50 to-green-100 border-green-200 hover:shadow-lg transition-all duration-300">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-green-600 text-sm font-medium">Processed</p>
                          <p className="text-3xl font-bold text-green-900">{stats.completed}</p>
                        </div>
                        <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
                          <CheckCircle className="w-6 h-6 text-white" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-purple-50 to-purple-100 border-purple-200 hover:shadow-lg transition-all duration-300">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-purple-600 text-sm font-medium">Total Size</p>
                          <p className="text-3xl font-bold text-purple-900">{formatFileSize(stats.totalSize)}</p>
                        </div>
                        <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
                          <Database className="w-6 h-6 text-white" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  <Card className="bg-gradient-to-br from-orange-50 to-orange-100 border-orange-200 hover:shadow-lg transition-all duration-300">
                    <CardContent className="p-6">
                      <div className="flex items-center justify-between">
                        <div>
                          <p className="text-orange-600 text-sm font-medium">Languages</p>
                          <p className="text-3xl font-bold text-orange-900">{stats.languages.length}</p>
                        </div>
                        <div className="w-12 h-12 bg-orange-500 rounded-xl flex items-center justify-center">
                          <Globe className="w-6 h-6 text-white" />
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Recent Documents */}
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader>
                    <CardTitle className="flex items-center space-x-2">
                      <Activity className="w-5 h-5 text-blue-600" />
                      <span>Recent Documents</span>
                    </CardTitle>
                    <CardDescription>Latest processed documents with AI analysis</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {filteredDocuments.map((doc) => (
                        <div key={doc.id} className="flex items-center justify-between p-4 bg-gray-50/50 rounded-lg hover:bg-gray-100/50 transition-colors">
                          <div className="flex items-center space-x-4">
                            <div className="w-10 h-10 bg-blue-100 rounded-lg flex items-center justify-center">
                              <FileText className="w-5 h-5 text-blue-600" />
                            </div>
                            <div>
                              <h4 className="font-medium text-gray-900">{doc.name}</h4>
                              <div className="flex items-center space-x-2 text-sm text-gray-500">
                                <span>{formatFileSize(doc.size)}</span>
                                <span>•</span>
                                <span>{doc.uploadDate}</span>
                                {doc.language && (
                                  <>
                                    <span>•</span>
                                    <Badge variant="outline" className="text-xs">{doc.language}</Badge>
                                  </>
                                )}
                              </div>
                            </div>
                          </div>
                          <div className="flex items-center space-x-2">
                            {doc.status === "completed" && <CheckCircle className="w-5 h-5 text-green-500" />}
                            {doc.status === "processing" && <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />}
                            {doc.status === "error" && <AlertCircle className="w-5 h-5 text-red-500" />}
                            <Button variant="ghost" size="sm">
                              <Eye className="w-4 h-4" />
                            </Button>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Upload Tab */}
            {activeTab === "upload" && (
              <div className="space-y-6">
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader className="bg-gradient-to-r from-blue-50 to-indigo-50 border-b border-blue-100">
                    <CardTitle className="flex items-center space-x-2 text-blue-900">
                      <Upload className="w-6 h-6" />
                      <span>Document Upload & Processing</span>
                    </CardTitle>
                    <CardDescription className="text-blue-700">
                      Upload documents for AI-powered analysis and intelligent categorization
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-8">
                    <div
                      className={`border-2 border-dashed rounded-xl p-12 text-center transition-all cursor-pointer ${
                        isDragging
                          ? "border-blue-500 bg-blue-50 scale-105 shadow-lg"
                          : "border-gray-300 hover:border-gray-400 hover:bg-gray-50"
                      }`}
                      onDragOver={handleDragOver}
                      onDragEnter={handleDragEnter}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      onClick={() => document.getElementById("file-upload")?.click()}
                    >
                      <div className="space-y-4">
                        <div className={`mx-auto w-16 h-16 rounded-full flex items-center justify-center ${
                          isDragging ? "bg-blue-100" : "bg-gray-100"
                        }`}>
                          <Upload className={`w-8 h-8 ${isDragging ? "text-blue-600" : "text-gray-400"}`} />
                        </div>
                        <div>
                          <h3 className="text-xl font-semibold text-gray-900 mb-2">
                            {isDragging ? "Drop files here!" : "Upload Documents"}
                          </h3>
                          <p className="text-gray-600 mb-4">
                            Drag and drop files here, or click to browse
                          </p>
                          <p className="text-sm text-gray-500">
                            Supports PDF, DOCX, TXT, RTF • Max 100MB per file
                          </p>
                        </div>
                        <input
                          type="file"
                          multiple
                          accept=".pdf,.docx,.txt,.rtf"
                          onChange={handleFileSelect}
                          className="hidden"
                          id="file-upload"
                        />
                        <Button
                          type="button"
                          className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-8 py-3 rounded-lg font-semibold shadow-lg hover:shadow-xl transition-all duration-300"
                          onClick={(e) => {
                            e.stopPropagation()
                            document.getElementById("file-upload")?.click()
                          }}
                        >
                          <Plus className="w-5 h-5 mr-2" />
                          Select Files
                        </Button>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Processing Queue */}
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader>
                    <CardTitle className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <Activity className="w-5 h-5 text-blue-600" />
                        <span>Processing Queue ({documents.length})</span>
                      </div>
                      {stats.processing > 0 && (
                        <Badge className="bg-blue-100 text-blue-800">
                          {stats.processing} Processing
                        </Badge>
                      )}
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {documents.length === 0 ? (
                      <div className="text-center py-12">
                        <div className="w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mx-auto mb-4">
                          <FileText className="w-8 h-8 text-gray-400" />
                        </div>
                        <h3 className="text-lg font-medium text-gray-900 mb-2">No documents yet</h3>
                        <p className="text-gray-500">Upload your first document to get started</p>
                      </div>
                    ) : (
                      <div className="space-y-4">
                        {documents.map((doc) => (
                          <div key={doc.id} className="bg-gray-50/50 rounded-xl p-6 hover:bg-gray-100/50 transition-colors">
                            <div className="flex items-center justify-between mb-4">
                              <div className="flex items-center space-x-4">
                                <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                                  <FileText className="w-6 h-6 text-blue-600" />
                                </div>
                                <div>
                                  <h4 className="font-semibold text-gray-900">{doc.name}</h4>
                                  <div className="flex items-center space-x-2 text-sm text-gray-500">
                                    <span>{formatFileSize(doc.size)}</span>
                                    <span>•</span>
                                    <span>{doc.uploadDate}</span>
                                    {doc.language && (
                                      <>
                                        <span>•</span>
                                        <Badge variant="outline" className="text-xs">{doc.language}</Badge>
                                      </>
                                    )}
                                  </div>
                                </div>
                              </div>
                              <div className="flex items-center space-x-3">
                                {doc.status === "completed" && (
                                  <div className="flex items-center space-x-2">
                                    <CheckCircle className="w-5 h-5 text-green-500" />
                                    <span className="text-sm font-medium text-green-700">Completed</span>
                                  </div>
                                )}
                                {doc.status === "error" && (
                                  <div className="flex items-center space-x-2">
                                    <AlertCircle className="w-5 h-5 text-red-500" />
                                    <span className="text-sm font-medium text-red-700">Error</span>
                                  </div>
                                )}
                                {(doc.status === "uploading" || doc.status === "processing") && (
                                  <div className="flex items-center space-x-2">
                                    <Loader2 className="w-5 h-5 text-blue-500 animate-spin" />
                                    <span className="text-sm font-medium text-blue-700">
                                      {doc.status === "uploading" ? "Uploading" : "Processing"}
                                    </span>
                                  </div>
                                )}
                                <Button 
                                  variant="ghost" 
                                  size="sm" 
                                  onClick={() => handleDeleteDocument(doc.id)}
                                  className="hover:bg-red-50 hover:text-red-700 transition-colors"
                                  disabled={deletingDocuments.has(doc.id)}
                                >
                                  {deletingDocuments.has(doc.id) ? (
                                    <Loader2 className="w-4 h-4 text-red-500 animate-spin" />
                                  ) : (
                                    <Trash2 className="w-4 h-4 text-red-500" />
                                  )}
                                </Button>
                              </div>
                            </div>

                            {doc.status !== "completed" && (
                                <div className="space-y-2">
                                  <div className="flex justify-between text-sm">
                                    <span className="text-gray-600">
                                      {doc.statusMessage || "Processing..."}
                                    </span>
                                    <span className="font-medium text-gray-700">{Math.round(doc.progress)}%</span>
                                  </div>
                                  <Progress value={Math.round(doc.progress)} className="h-2" />
                                  {doc.processingStage && (
                                    <div className="text-xs text-blue-600 font-medium">
                                      Stage: {doc.processingStage}
                                    </div>
                                  )}
                                </div>
                            )}

                            {doc.status === "error" && (
                              <div className="mt-3 p-3 bg-red-50 rounded-lg border border-red-200">
                                <p className="text-red-700 text-sm font-medium">Error: {doc.error}</p>
                              </div>
                            )}

                            {doc.status === "completed" && (
                              <div className="mt-4 space-y-3">
                                <div className="flex flex-wrap gap-2">
                                  {doc.themes && doc.themes.slice(0, 3).map((theme, index) => (
                                    <Badge key={index} variant="secondary" className="text-xs">
                                      {theme}
                                    </Badge>
                                  ))}
                                  {doc.themes && doc.themes.length > 3 && (
                                    <Badge variant="outline" className="text-xs">
                                      +{doc.themes.length - 3} more
                                    </Badge>
                                  )}
                                </div>
                                
                                {doc.summary && (
                                  <div className="bg-blue-50 rounded-lg p-3">
                                    <p className="text-sm text-blue-800 line-clamp-2">
                                      {doc.summary.substring(0, 150)}...
                                    </p>
                                  </div>
                                )}
                              </div>
                            )}
                          </div>
                        ))}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}

            {/* AI Analysis Tab - Enterprise-Grade Interface */}
            {activeTab === "analyze" && (
              <div className="space-y-8">
                {/* Revolutionary Header Section */}
                <div className="relative overflow-hidden bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 rounded-3xl p-12 border border-purple-500/20 shadow-2xl">
                  {/* Animated Background */}
                  <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 via-purple-600/10 to-indigo-600/10 animate-pulse"></div>
                  <div className="absolute inset-0 opacity-20">
                    <div className="w-full h-full bg-gradient-to-r from-purple-500/5 via-blue-500/5 to-indigo-500/5"></div>
                  </div>
                  
                  <div className="relative z-10">
                    <div className="flex items-center justify-between mb-8">
                      <div className="flex items-center space-x-6">
                        <div className="relative">
                          <div className="w-20 h-20 bg-gradient-to-br from-blue-500 via-purple-500 to-indigo-500 rounded-3xl flex items-center justify-center shadow-2xl transform rotate-3 hover:rotate-0 transition-transform duration-500">
                            <Brain className="w-10 h-10 text-white" />
                          </div>
                          <div className="absolute -top-2 -right-2 w-6 h-6 bg-green-400 rounded-full flex items-center justify-center animate-bounce">
                            <Zap className="w-3 h-3 text-white" />
                          </div>
                        </div>
                        <div>
                          <h1 className="text-5xl font-black bg-gradient-to-r from-white via-blue-100 to-purple-100 bg-clip-text text-transparent mb-3">
                            AI Document Intelligence
                          </h1>
                          <p className="text-xl text-gray-300 font-medium">
                            Enterprise-Grade Document Analysis & Strategic Intelligence Platform
                          </p>
                          <div className="flex items-center space-x-4 mt-3">
                            <div className="flex items-center space-x-2 bg-green-500/20 px-3 py-1 rounded-full border border-green-400/30">
                              <div className="w-2 h-2 bg-green-400 rounded-full animate-pulse"></div>
                              <span className="text-green-300 font-medium text-sm">Neural Engine Active</span>
                            </div>
                            <div className="flex items-center space-x-2 bg-blue-500/20 px-3 py-1 rounded-full border border-blue-400/30">
                              <Shield className="w-3 h-3 text-blue-300" />
                              <span className="text-blue-300 font-medium text-sm">Enterprise Security</span>
                            </div>
                          </div>
                        </div>
                      </div>
                      
                      <div className="text-right">
                        <div className="bg-white/10 backdrop-blur-xl rounded-2xl p-6 border border-white/20">
                          <div className="text-3xl font-bold text-white mb-1">
                            {documents.filter(d => d.status === "completed").length}
                          </div>
                          <div className="text-gray-300 text-sm font-medium">Documents Ready</div>
                          <div className="text-xs text-gray-400 mt-1">for AI Analysis</div>
                        </div>
                      </div>
                    </div>
                    
                    {/* Advanced Capabilities Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
                      {[
                        { 
                          icon: Target, 
                          label: "Strategic Intelligence", 
                          desc: "Business insights & market opportunities",
                          color: "from-blue-500 to-cyan-500",
                          bgColor: "bg-blue-500/10 border-blue-400/30"
                        },
                        { 
                          icon: TrendingUp, 
                          label: "Sentiment Analysis", 
                          desc: "Emotional intelligence & context mapping",
                          color: "from-green-500 to-emerald-500",
                          bgColor: "bg-green-500/10 border-green-400/30"
                        },
                        { 
                          icon: Users, 
                          label: "Audience Profiling", 
                          desc: "Demographics & persona identification",
                          color: "from-purple-500 to-violet-500",
                          bgColor: "bg-purple-500/10 border-purple-400/30"
                        },
                        { 
                          icon: Lightbulb, 
                          label: "Strategic Recommendations", 
                          desc: "AI-powered actionable insights",
                          color: "from-orange-500 to-yellow-500",
                          bgColor: "bg-orange-500/10 border-orange-400/30"
                        }
                      ].map((capability, index) => {
                        const Icon = capability.icon
                        return (
                          <div key={index} className={`${capability.bgColor} backdrop-blur-xl rounded-2xl p-6 border hover:scale-105 transition-all duration-300 cursor-pointer group`}>
                            <div className={`w-12 h-12 bg-gradient-to-br ${capability.color} rounded-xl flex items-center justify-center mb-4 group-hover:scale-110 transition-transform duration-300`}>
                              <Icon className="w-6 h-6 text-white" />
                            </div>
                            <h4 className="font-bold text-white text-sm mb-2">{capability.label}</h4>
                            <p className="text-xs text-gray-300 leading-relaxed">{capability.desc}</p>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                </div>

                {/* Document Analysis Interface */}
                <div className="space-y-8">
                  {documents.filter((d) => d.status === "completed").length === 0 ? (
                    <div className="relative overflow-hidden bg-gradient-to-br from-gray-50 to-white rounded-3xl p-16 border border-gray-200 shadow-xl">
                      <div className="absolute inset-0 bg-gradient-to-r from-blue-50/50 via-purple-50/50 to-indigo-50/50"></div>
                      <div className="relative z-10 text-center">
                        <div className="w-32 h-32 bg-gradient-to-br from-purple-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-8 shadow-lg">
                          <Brain className="w-16 h-16 text-purple-600" />
                        </div>
                        <h3 className="text-3xl font-bold text-gray-900 mb-4">Ready to Unlock Intelligence</h3>
                        <p className="text-xl text-gray-600 mb-8 max-w-2xl mx-auto leading-relaxed">
                          Upload and process documents to access our revolutionary AI-powered analysis engine. 
                          Transform raw content into strategic business intelligence.
                        </p>
                        <Button 
                          onClick={() => setActiveTab("upload")}
                          className="bg-gradient-to-r from-purple-600 via-blue-600 to-indigo-600 hover:from-purple-700 hover:via-blue-700 hover:to-indigo-700 text-white px-12 py-4 rounded-2xl font-bold text-lg shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-105"
                        >
                          <Upload className="w-6 h-6 mr-3" />
                          Begin Document Upload
                        </Button>
                      </div>
                    </div>
                  ) : (
                    documents
                      .filter((d) => d.status === "completed")
                      .map((doc) => (
                        <div key={doc.id} className="relative overflow-hidden bg-white rounded-3xl shadow-2xl border border-gray-200/50 hover:shadow-3xl transition-all duration-700 transform hover:scale-[1.02]">
                          {/* Premium Document Header */}
                          <div className="relative bg-gradient-to-r from-slate-50 via-blue-50 to-indigo-50 border-b border-gray-200/50 p-8">
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-purple-500/5 to-indigo-500/5"></div>
                            <div className="relative z-10 flex items-center justify-between">
                              <div className="flex items-center space-x-8">
                                <div className="relative">
                                  <div className="w-20 h-20 bg-gradient-to-br from-blue-100 via-purple-100 to-indigo-100 rounded-3xl flex items-center justify-center shadow-xl border border-white/50">
                                    <FileText className="w-10 h-10 text-purple-600" />
                                  </div>
                                  <div className="absolute -top-2 -right-2 w-8 h-8 bg-gradient-to-br from-green-400 to-emerald-500 rounded-full flex items-center justify-center shadow-lg">
                                    <CheckCircle className="w-5 h-5 text-white" />
                                  </div>
                                </div>
                                <div>
                                  <h2 className="text-2xl font-bold text-gray-900 mb-3">{doc.name}</h2>
                                  <div className="flex items-center space-x-6 text-sm">
                                    <div className="flex items-center space-x-2 bg-white/70 px-3 py-1 rounded-lg">
                                      <Database className="w-4 h-4 text-gray-500" />
                                      <span className="text-gray-700 font-medium">{formatFileSize(doc.size)}</span>
                                    </div>
                                    <div className="flex items-center space-x-2 bg-white/70 px-3 py-1 rounded-lg">
                                      <Calendar className="w-4 h-4 text-gray-500" />
                                      <span className="text-gray-700 font-medium">{doc.uploadDate}</span>
                                    </div>
                                    <Badge className="bg-gradient-to-r from-blue-500 to-indigo-500 text-white px-3 py-1 shadow-lg">
                                      <Globe className="w-3 h-3 mr-1" />
                                      {doc.language || 'English'}
                                    </Badge>
                                    <Badge className={getSentimentColor(doc.sentiment || "neutral")}>
                                      <TrendingUp className="w-3 h-3 mr-1" />
                                      {doc.sentiment || "Neutral"}
                                    </Badge>
                                  </div>
                                </div>
                              </div>
                              
                              {/* Analysis Actions */}
                              <div className="flex items-center space-x-4">
                                {doc.summary ? (
                                  <div className="flex items-center space-x-2 bg-green-500/20 px-4 py-2 rounded-full border border-green-400/30">
                                    <CheckCircle className="w-4 h-4 text-green-400" />
                                    <span className="text-green-300 font-medium text-sm">Analysis Complete</span>
                                  </div>
                                ) : (
                                  <div className="flex items-center space-x-2 bg-yellow-500/20 px-4 py-2 rounded-full border border-yellow-400/30">
                                    <Clock className="w-4 h-4 text-yellow-400" />
                                    <span className="text-yellow-300 font-medium text-sm">Ready for Analysis</span>
                                  </div>
                                )}
                                <Button
                                  onClick={() => handleSummarize(doc.id)}
                                  disabled={isProcessing}
                                  className="bg-gradient-to-r from-purple-600 to-indigo-600 hover:from-purple-700 hover:to-indigo-700 text-white px-8 py-4 rounded-2xl font-bold shadow-2xl hover:shadow-3xl transition-all duration-300 transform hover:scale-105"
                                >
                                  {isProcessing ? (
                                    <>
                                      <Loader2 className="w-5 h-5 mr-3 animate-spin" />
                                      Analyzing...
                                    </>
                                  ) : doc.summary ? (
                                    <>
                                      <RefreshCw className="w-5 h-5 mr-3" />
                                      Re-analyze
                                    </>
                                  ) : (
                                    <>
                                      <Zap className="w-5 h-5 mr-3" />
                                      Start Analysis
                                    </>
                                  )}
                                </Button>
                              </div>
                            </div>
                          </div>

                          {/* Enterprise Analysis Results */}
                          <div className="p-0">
                            {doc.summary ? (
                              <div className="p-12">
                                {/* Executive Dashboard */}
                                <div className="grid grid-cols-1 md:grid-cols-3 gap-8 mb-12">
                                  <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-2xl p-8 border border-blue-200 shadow-lg hover:shadow-xl transition-all duration-300">
                                    <div className="flex items-center justify-between mb-6">
                                      <h4 className="font-bold text-blue-900 text-lg">Document Intelligence</h4>
                                      <div className="w-12 h-12 bg-blue-500 rounded-xl flex items-center justify-center">
                                        <Tag className="w-6 h-6 text-white" />
                                      </div>
                                    </div>
                                    <div className="space-y-3">
                                      <div className="flex flex-wrap gap-2">
                                        {doc.themes?.slice(0, 3).map((theme, index) => (
                                          <Badge key={index} className="bg-blue-200 text-blue-800 text-xs font-semibold px-3 py-1">
                                            {theme}
                                          </Badge>
                                        ))}
                                        {doc.themes && doc.themes.length > 3 && (
                                          <Badge variant="outline" className="text-xs border-blue-300 text-blue-700 font-semibold">
                                            +{doc.themes.length - 3} more
                                          </Badge>
                                        )}
                                      </div>
                                    </div>
                                  </div>

                                  <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-2xl p-8 border border-green-200 shadow-lg hover:shadow-xl transition-all duration-300">
                                    <div className="flex items-center justify-between mb-6">
                                      <h4 className="font-bold text-green-900 text-lg">Analysis Metrics</h4>
                                      <div className="w-12 h-12 bg-green-500 rounded-xl flex items-center justify-center">
                                        <BarChart3 className="w-6 h-6 text-white" />
                                      </div>
                                    </div>
                                    <div className="space-y-4">
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm text-green-700 font-medium">Sentiment:</span>
                                        <Badge className={getSentimentColor(doc.sentiment || "neutral")}>
                                          {doc.sentiment || "Neutral"}
                                        </Badge>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm text-green-700 font-medium">Complexity:</span>
                                        <Badge className={getReadingLevelColor(doc.readingLevel || "intermediate")}>
                                          {doc.readingLevel || "Intermediate"}
                                        </Badge>
                                      </div>
                                    </div>
                                  </div>

                                  <div className="bg-gradient-to-br from-purple-50 to-purple-100 rounded-2xl p-8 border border-purple-200 shadow-lg hover:shadow-xl transition-all duration-300">
                                    <div className="flex items-center justify-between mb-6">
                                      <h4 className="font-bold text-purple-900 text-lg">AI Insights</h4>
                                      <div className="w-12 h-12 bg-purple-500 rounded-xl flex items-center justify-center">
                                        <Brain className="w-6 h-6 text-white" />
                                      </div>
                                    </div>
                                    <div className="space-y-4">
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm text-purple-700 font-medium">Key Insights:</span>
                                        <Badge className="bg-purple-200 text-purple-800 font-bold">
                                          {doc.keyInsights?.length || 0}
                                        </Badge>
                                      </div>
                                      <div className="flex items-center justify-between">
                                        <span className="text-sm text-purple-700 font-medium">Demographics:</span>
                                        <Badge className="bg-purple-200 text-purple-800 font-bold">
                                          {doc.demographics?.length || 0}
                                        </Badge>
                                      </div>
                                    </div>
                                  </div>
                                </div>

                                {/* Clean Analysis Report */}
                                <div className="bg-white rounded-xl border border-gray-200 shadow-sm">
                                  <div className="flex items-center justify-between p-6 border-b border-gray-200">
                                    <h1 className="text-2xl font-bold text-gray-900 flex items-center">
                                      <FileCheck className="w-6 h-6 mr-3 text-blue-600" />
                                      Analysis Report
                                    </h1>
                                    <div className="flex items-center space-x-2">
                                      <Button variant="outline" size="sm">
                                        <Download className="w-4 h-4 mr-2" />
                                        Export
                                      </Button>
                                      <Button variant="outline" size="sm">
                                        <Eye className="w-4 h-4 mr-2" />
                                        View
                                      </Button>
                                    </div>
                                  </div>
                                  
                                  <div className="p-6">
                                    <div className="space-y-6">
                                      {/* Executive Summary */}
                                      <div>
                                        <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                                          <span className="w-2 h-2 bg-blue-500 rounded-full mr-3"></span>
                                          Executive Summary
                                        </h2>
                                        <div className="bg-gray-50 rounded-lg p-4">
                                          <div className="prose prose-sm max-w-none text-gray-700">
                                            {formatSummaryContent(doc.summary)}
                                          </div>
                                        </div>
                                      </div>

                                      {/* Key Insights */}
                                      {doc.keyInsights && doc.keyInsights.length > 0 && (
                                        <div>
                                          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center">
                                            <span className="w-2 h-2 bg-yellow-500 rounded-full mr-3"></span>
                                            Key Insights
                                          </h2>
                                          <div className="space-y-3">
                                            {doc.keyInsights.map((insight, index) => (
                                              <div key={index} className="bg-yellow-50 border-l-4 border-yellow-400 p-4 rounded-r-lg">
                                                <div className="prose prose-sm max-w-none text-gray-700">
                                                  {formatInsightContent(insight)}
                                                </div>
                                              </div>
                                            ))}
                                          </div>
                                        </div>
                                      )}

                                      {/* Topics and Demographics Grid */}
                                      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                                        <div>
                                          <h3 className="text-md font-semibold text-gray-800 mb-3 flex items-center">
                                            <Target className="w-4 h-4 mr-2 text-blue-600" />
                                            Key Topics
                                          </h3>
                                          <div className="flex flex-wrap gap-2">
                                            {doc.mainTopics?.map((topic, index) => (
                                              <Badge
                                                key={index}
                                                variant="secondary"
                                                className="bg-blue-50 text-blue-700 text-xs"
                                              >
                                                {topic}
                                              </Badge>
                                            ))}
                                          </div>
                                        </div>

                                        <div>
                                          <h3 className="text-md font-semibold text-gray-800 mb-3 flex items-center">
                                            <Users className="w-4 h-4 mr-2 text-green-600" />
                                            Target Audience
                                          </h3>
                                          <div className="flex flex-wrap gap-2">
                                            {doc.demographics && doc.demographics.length > 0 ? (
                                              doc.demographics.map((demo, index) => (
                                                <Badge
                                                  key={index}
                                                  className="bg-green-50 text-green-700 border-green-200 text-xs"
                                                >
                                                  {demo}
                                                </Badge>
                                              ))
                                            ) : (
                                              <span className="text-sm text-gray-500">General audience</span>
                                            )}
                                          </div>
                                        </div>
                                      </div>
                                    </div>
                                  </div>
                                </div>
                              </div>
                            ) : (
                              <div className="p-16 text-center bg-gradient-to-br from-gray-50 to-white">
                                <div className="w-24 h-24 bg-gradient-to-br from-purple-100 to-indigo-100 rounded-full flex items-center justify-center mx-auto mb-8 shadow-lg">
                                  <Brain className="w-12 h-12 text-purple-600" />
                                </div>
                                <h3 className="text-2xl font-bold text-gray-900 mb-4">Ready for AI Analysis</h3>
                                <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto leading-relaxed">
                                  Click "Start Analysis" to generate comprehensive insights, summaries, and strategic recommendations for this document.
                                </p>
                                <div className="flex items-center justify-center space-x-8 text-sm text-gray-500">
                                  <div className="flex items-center space-x-2">
                                    <Zap className="w-5 h-5" />
                                    <span className="font-medium">AI-Powered</span>
                                  </div>
                                  <div className="flex items-center space-x-2">
                                    <Target className="w-5 h-5" />
                                    <span className="font-medium">Strategic Focus</span>
                                  </div>
                                  <div className="flex items-center space-x-2">
                                    <TrendingUp className="w-5 h-5" />
                                    <span className="font-medium">Actionable Insights</span>
                                  </div>
                                </div>
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                  )}
                </div>
              </div>
            )}

            {/* Insights Tab */}
            {activeTab === "insights" && (
              <div className="space-y-6">
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader className="bg-gradient-to-r from-yellow-50 to-orange-50 border-b border-yellow-100">
                    <CardTitle className="flex items-center space-x-2 text-yellow-900">
                      <Lightbulb className="w-6 h-6" />
                      <span>Document Insights & Analytics</span>
                    </CardTitle>
                    <CardDescription className="text-yellow-700">
                      Comprehensive insights and analysis from your document collection
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-6">
                      {documents
                        .filter((d) => d.status === "completed")
                        .map((doc) => (
                          <div key={doc.id} className="bg-gradient-to-r from-white to-yellow-50 rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                            <div className="flex items-center space-x-4 mb-6">
                              <div className="w-14 h-14 bg-gradient-to-br from-yellow-100 to-orange-100 rounded-xl flex items-center justify-center">
                                <FileText className="w-7 h-7 text-yellow-600" />
                              </div>
                              <div>
                                <h4 className="text-lg font-semibold text-gray-900">{doc.name}</h4>
                                <div className="flex items-center space-x-3 text-sm text-gray-500">
                                  <span>{formatFileSize(doc.size)}</span>
                                  <Badge variant="outline" className="text-xs">
                                    {doc.language || 'English'}
                                  </Badge>
                                </div>
                              </div>
                            </div>

                            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                              {/* Main Topics */}
                              <div className="bg-blue-50 rounded-xl p-4">
                                <h5 className="font-semibold mb-3 text-blue-800 flex items-center">
                                  <Target className="w-4 h-4 mr-2" />
                                  Main Topics
                                </h5>
                                <div className="flex flex-wrap gap-2">
                                  {doc.mainTopics?.map((topic, index) => (
                                    <Badge
                                      key={index}
                                      variant="secondary"
                                      className="bg-blue-100 text-blue-800 hover:bg-blue-200"
                                    >
                                      {topic}
                                    </Badge>
                                  ))}
                                </div>
                              </div>

                              {/* Sentiment Analysis */}
                              <div className="bg-green-50 rounded-xl p-4">
                                <h5 className="font-semibold mb-3 text-green-800 flex items-center">
                                  <TrendingUp className="w-4 h-4 mr-2" />
                                  Analysis Metrics
                                </h5>
                                <div className="space-y-2">
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm text-gray-600">Sentiment:</span>
                                    <Badge className={getSentimentColor(doc.sentiment || "neutral")}>
                                      {doc.sentiment || "Neutral"}
                                    </Badge>
                                  </div>
                                  <div className="flex items-center justify-between">
                                    <span className="text-sm text-gray-600">Reading Level:</span>
                                    <Badge className={getReadingLevelColor(doc.readingLevel || "intermediate")}>
                                      {doc.readingLevel || "Intermediate"}
                                    </Badge>
                                  </div>
                                </div>
                              </div>

                              {/* Key Insights */}
                              <div className="bg-purple-50 rounded-xl p-4 md:col-span-2">
                                <h5 className="font-semibold mb-3 text-purple-800 flex items-center">
                                  <Brain className="w-4 h-4 mr-2" />
                                  Strategic Insights
                                </h5>
                                {doc.keyInsights && doc.keyInsights.length > 0 ? (
                                  <div
                                    dangerouslySetInnerHTML={{ __html: parseInsightsContent(doc.keyInsights) }}
                                  />
                                ) : (
                                  <p className="text-sm text-gray-600">No insights available for this document</p>
                                )}
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Manage Tags Tab */}
            {activeTab === "manage" && (
              <div className="space-y-6">
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader className="bg-gradient-to-r from-pink-50 to-rose-50 border-b border-pink-100">
                    <CardTitle className="flex items-center space-x-2 text-pink-900">
                      <Tag className="w-6 h-6" />
                      <span>Tag & Keyword Management</span>
                    </CardTitle>
                    <CardDescription className="text-pink-700">
                      Manage generated tags and keywords, or add custom ones for better organization
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-6">
                      {documents
                        .filter((d) => d.status === "completed")
                        .map((doc) => (
                          <div key={doc.id} className="bg-gradient-to-r from-white to-pink-50 rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                            <div className="flex items-center space-x-4 mb-6">
                              <div className="w-12 h-12 bg-gradient-to-br from-pink-100 to-rose-100 rounded-xl flex items-center justify-center">
                                <FileText className="w-6 h-6 text-pink-600" />
                              </div>
                              <div>
                                <h4 className="text-lg font-semibold text-gray-900">{doc.name}</h4>
                                <p className="text-sm text-gray-500">{formatFileSize(doc.size)}</p>
                              </div>
                            </div>

                            {/* Tags Section */}
                            <div className="mb-6">
                              <Label className="text-sm font-semibold mb-3 block text-gray-700">Generated & Custom Tags:</Label>
                              <div className="flex flex-wrap gap-2 mb-3">
                                {doc.themes?.map((theme, index) => (
                                  <div key={index} className="flex items-center bg-pink-100 rounded-lg">
                                    <Badge variant="secondary" className="bg-pink-100 text-pink-800 border-0">
                                      {theme}
                                    </Badge>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handleRemoveTag(doc.id, theme)}
                                      className="h-6 w-6 p-0 ml-1 hover:bg-pink-200"
                                    >
                                      <X className="h-3 w-3 text-pink-600" />
                                    </Button>
                                  </div>
                                ))}
                              </div>
                              <div className="flex space-x-2">
                                <Input
                                  placeholder="Add custom tag..."
                                  value={newTags[doc.id] || ""}
                                  onChange={(e) => setNewTags((prev) => ({ ...prev, [doc.id]: e.target.value }))}
                                  onKeyPress={(e) => {
                                    if (e.key === "Enter") {
                                      handleAddTag(doc.id, "")
                                    }
                                  }}
                                  className="flex-1 bg-white/50"
                                />
                                <Button
                                  onClick={() => handleAddTag(doc.id, "")}
                                  size="sm"
                                  disabled={!newTags[doc.id]?.trim()}
                                  className="bg-pink-600 hover:bg-pink-700 text-white"
                                >
                                  <Plus className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>

                            {/* Keywords Section */}
                            <div>
                              <Label className="text-sm font-semibold mb-3 block text-gray-700">Generated & Custom Keywords:</Label>
                              <div className="flex flex-wrap gap-2 mb-3">
                                {doc.keywords?.map((keyword, index) => (
                                  <div key={index} className="flex items-center bg-blue-100 rounded-lg">
                                    <Badge variant="outline" className="bg-blue-100 text-blue-800 border-blue-200">
                                      {keyword}
                                    </Badge>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handleRemoveKeyword(doc.id, keyword)}
                                      className="h-6 w-6 p-0 ml-1 hover:bg-blue-200"
                                    >
                                      <X className="h-3 w-3 text-blue-600" />
                                    </Button>
                                  </div>
                                ))}
                              </div>
                              <div className="flex space-x-2">
                                <Input
                                  placeholder="Add custom keyword..."
                                  value={newKeywords[doc.id] || ""}
                                  onChange={(e) => setNewKeywords((prev) => ({ ...prev, [doc.id]: e.target.value }))}
                                  onKeyPress={(e) => {
                                    if (e.key === "Enter") {
                                      handleAddKeyword(doc.id, "")
                                    }
                                  }}
                                  className="flex-1 bg-white/50"
                                />
                                <Button
                                  onClick={() => handleAddKeyword(doc.id, "")}
                                  size="sm"
                                  disabled={!newKeywords[doc.id]?.trim()}
                                  className="bg-blue-600 hover:bg-blue-700 text-white"
                                >
                                  <Plus className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}

            {/* Demographics Tab */}
            {activeTab === "demographics" && (
              <div className="space-y-6">
                <Card className="bg-white/80 backdrop-blur-sm border-gray-200/50">
                  <CardHeader className="bg-gradient-to-r from-indigo-50 to-purple-50 border-b border-indigo-100">
                    <CardTitle className="flex items-center space-x-2 text-indigo-900">
                      <Users className="w-6 h-6" />
                      <span>Demographics Analysis</span>
                    </CardTitle>
                    <CardDescription className="text-indigo-700">
                      View and manage detected demographic information from your documents
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="p-6">
                    <div className="space-y-6">
                      {documents
                        .filter((d) => d.status === "completed")
                        .map((doc) => (
                          <div key={doc.id} className="bg-gradient-to-r from-white to-indigo-50 rounded-xl p-6 border border-gray-200 hover:shadow-lg transition-all duration-300">
                            <div className="flex items-center space-x-4 mb-6">
                              <div className="w-12 h-12 bg-gradient-to-br from-indigo-100 to-purple-100 rounded-xl flex items-center justify-center">
                                <FileText className="w-6 h-6 text-indigo-600" />
                              </div>
                              <div>
                                <h4 className="text-lg font-semibold text-gray-900">{doc.name}</h4>
                                <div className="flex items-center text-sm text-gray-500">
                                  <Languages className="w-3 h-3 mr-1" /> {doc.language || 'English'}
                                </div>
                              </div>
                            </div>

                            {/* Demographics Section */}
                            <div className="mb-6">
                              <Label className="text-sm font-semibold mb-3 block text-gray-700">Detected Demographics:</Label>
                              <div className="flex flex-wrap gap-2 mb-3">
                                {doc.demographics?.map((demo, index) => (
                                  <div key={index} className="flex items-center bg-purple-100 rounded-lg">
                                    <Badge
                                      variant="secondary"
                                      className="bg-purple-100 text-purple-800 hover:bg-purple-200 border-0"
                                    >
                                      <Users className="w-3 h-3 mr-1" /> {demo}
                                    </Badge>
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={() => handleRemoveDemographic(doc.id, demo)}
                                      className="h-6 w-6 p-0 ml-1 hover:bg-purple-200"
                                    >
                                      <X className="h-3 w-3 text-purple-600" />
                                    </Button>
                                  </div>
                                ))}
                              </div>
                              <div className="flex space-x-2">
                                <Input
                                  placeholder="Add custom demographic..."
                                  value={newDemographics[doc.id] || ""}
                                  onChange={(e) =>
                                    setNewDemographics((prev) => ({ ...prev, [doc.id]: e.target.value }))
                                  }
                                  onKeyPress={(e) => {
                                    if (e.key === "Enter") {
                                      handleAddDemographic(doc.id, "")
                                    }
                                  }}
                                  className="flex-1 bg-white/50"
                                />
                                <Button
                                  onClick={() => handleAddDemographic(doc.id, "")}
                                  size="sm"
                                  disabled={!newDemographics[doc.id]?.trim()}
                                  className="bg-indigo-600 hover:bg-indigo-700 text-white"
                                >
                                  <Plus className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>

                            {/* Document Audience Analysis */}
                            <div className="bg-gray-50 rounded-xl p-4">
                              <h5 className="font-semibold mb-3 flex items-center text-gray-800">
                                <BookMarked className="w-4 h-4 mr-2" /> Document Audience Analysis
                              </h5>
                              <p className="text-sm text-gray-700 mb-3">
                                This document appears to be targeted at{" "}
                                <strong>{doc.demographics?.join(", ") || "general audiences"}</strong>. The content is written at a{" "}
                                <Badge className={getReadingLevelColor(doc.readingLevel || "intermediate")}>
                                  {doc.readingLevel || "intermediate"}
                                </Badge>{" "}
                                level with a{" "}
                                <Badge className={getSentimentColor(doc.sentiment || "neutral")}>
                                  {doc.sentiment || "neutral"}
                                </Badge>{" "}
                                tone.
                              </p>
                            </div>
                          </div>
                        ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Professional Delete Confirmation Dialog */}
      <AlertDialog open={deleteDialog.isOpen} onOpenChange={() => {}}>
        <AlertDialogContent className="max-w-md">
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center space-x-3">
              <div className="w-12 h-12 bg-red-100 rounded-full flex items-center justify-center">
                <Trash2 className="w-6 h-6 text-red-600" />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-gray-900">Delete Document</h3>
                <p className="text-sm text-gray-500 mt-1">This action cannot be undone</p>
              </div>
            </AlertDialogTitle>
          </AlertDialogHeader>
          
          <AlertDialogDescription className="text-gray-600 mt-4">
            Are you sure you want to permanently delete "{deleteDialog.document?.name}"? 
            This will remove the document and all its associated data including AI analysis and insights, tags and keywords, processing history, and vector embeddings.
          </AlertDialogDescription>

          <AlertDialogFooter className="mt-6 space-x-3">
            <Button
              onClick={cancelDeleteDocument}
              disabled={deleteDialog.isDeleting}
              variant="outline"
              className="bg-gray-100 hover:bg-gray-200 text-gray-700"
            >
              Cancel
            </Button>
            <Button
              onClick={confirmDeleteDocument}
              disabled={deleteDialog.isDeleting}
              className="bg-red-600 hover:bg-red-700 text-white min-w-[120px]"
            >
              {deleteDialog.isDeleting ? (
                <div className="flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin" />
                  <span>Deleting...</span>
                </div>
              ) : (
                <div className="flex items-center space-x-2">
                  <Trash2 className="w-4 h-4" />
                  <span>Delete Forever</span>
                </div>
              )}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  )
}

export default function DocumentsPage() {
  return (
    <RouteGuard>
      <DocumentsPageContent />
    </RouteGuard>
  )
}
