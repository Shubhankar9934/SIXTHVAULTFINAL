"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Separator } from "@/components/ui/separator"
import {
  Send,
  Loader2,
  Settings,
  Database,
  Brain,
  Sparkles,
  FileText,
  TrendingUp,
  Globe,
  Building2,
  Users,
  BarChart3,
  Menu,
  X,
  Clock,
  Calendar,
  Share2,
  Download,
  ExternalLink,
  Info,
  ChevronDown,
  ChevronUp,
  Eye,
  BarChart,
  Search,
  Filter,
  Zap,
  Target,
  Activity,
  Shield,
  Star,
  Layers,
  MessageSquare,
  PieChart,
  TrendingDown,
  ArrowRight,
  CheckCircle,
  AlertCircle,
  Maximize2,
  Minimize2,
  Check,
  Plus,
  Minus,
  LogOut,
  History,
} from "lucide-react"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandInput,
  CommandItem,
  CommandList,
} from "@/components/ui/command"
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
} from "@/components/ui/dialog"
import { Label } from "@/components/ui/label"
import { Textarea } from "@/components/ui/textarea"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { RouteGuard } from "@/components/route-guard"
import { useAuth } from "@/lib/auth-context"
import { useVaultState } from "@/lib/vault-state-provider"
import { cacheManager } from "@/lib/cache-manager"
import HistorySection from "./history-section"

function VaultPageContent() {
  const { logout, user } = useAuth()
  const {
    state,
    dispatch,
    // UI actions
    toggleSidebar,
    toggleSidebarExpansion,
    setActiveTab,
    toggleExpandedSources,
    toggleExpandedRelevance,
    // Settings actions
    updateProviderSettings,
    updateContextSettings,
    setSearchTag,
    setSelectedFiles,
    // Content generation actions
    generateCurationContent,
    generateSummaryContent,
    createCustomCuration,
    createCustomSummary,
    deleteCuration,
    deleteSummary,
    // Conversation actions
    sendMessage,
    loadConversationHistory,
    startNewConversation,
  } = useVaultState()

  // Local state for modals and input
  const [documentSelectorOpen, setDocumentSelectorOpen] = useState(false)
  const [inputMessage, setInputMessage] = useState("")

  // Custom Curation Modal State
  const [showCustomCurationModal, setShowCustomCurationModal] = useState(false)
  const [customCurationTitle, setCustomCurationTitle] = useState("")
  const [customCurationDescription, setCustomCurationDescription] = useState("")
  const [customCurationKeywords, setCustomCurationKeywords] = useState("")
  const [isCreatingCustomCuration, setIsCreatingCustomCuration] = useState(false)

  // Custom Summary Modal State
  const [showCustomSummaryModal, setShowCustomSummaryModal] = useState(false)
  const [customSummaryTitle, setCustomSummaryTitle] = useState("")
  const [customSummaryDescription, setCustomSummaryDescription] = useState("")
  const [customSummaryKeywords, setCustomSummaryKeywords] = useState("")
  const [customSummaryFocusArea, setCustomSummaryFocusArea] = useState("")
  const [isCreatingCustomSummary, setIsCreatingCustomSummary] = useState(false)

  // Extract state values for easier access
  const {
    documents: { data: documents, availableTags, selectedFiles },
    curations: { customCurations, dynamicCurations, activeCuration, isGenerating: isGeneratingCuration },
    summaries: { dynamicSummaries, activeSummary, isGenerating: isGeneratingSummary },
    conversations: { chatMessages, currentConversationId, selectedConversationId, conversationTitle },
    ui: { activeTab, sidebarOpen, sidebarExpanded, expandedSources, expandedRelevance },
    settings: { selectedProvider, selectedModel, availableModels, modelProviders, keepContext, maxContext, searchTag }
  } = state

  // Cross-tab document deletion listener
  useEffect(() => {
    const handleStorageChange = (event: StorageEvent) => {
      if (event.key === 'sixthvault_document_delete_event') {
        console.log('📡 Vault: Received cross-tab document deletion event')
        
        try {
          const eventData = JSON.parse(event.newValue || '{}')
          console.log('📡 Vault: Deletion event data:', eventData)
          
          // Force refresh documents from backend after a short delay
          setTimeout(() => {
            // Trigger document refresh in the vault state provider
            dispatch({ type: 'REFRESH_DOCUMENTS' })
            console.log('✅ Vault: Triggered document refresh after cross-tab deletion')
          }, 1000)
          
        } catch (error) {
          console.error('❌ Vault: Failed to parse deletion event:', error)
        }
      }
      
      // Handle cache invalidation events
      if (event.key === 'sixthvault_cache_invalidate_documents') {
        console.log('📡 Vault: Received cache invalidation event')
        
        // Clear cache and refresh documents
        setTimeout(() => {
          dispatch({ type: 'REFRESH_DOCUMENTS' })
          console.log('✅ Vault: Triggered document refresh after cache invalidation')
        }, 500)
      }
      
      // Handle document upload events
      if (event.key === 'sixthvault_document_upload_event') {
        console.log('📡 Vault: Received document upload event')
        
        // Refresh documents to show newly uploaded documents
        setTimeout(() => {
          dispatch({ type: 'REFRESH_DOCUMENTS' })
          console.log('✅ Vault: Triggered document refresh after upload')
        }, 2000)
      }
    }

    // Add storage event listener
    window.addEventListener('storage', handleStorageChange)

    return () => {
      window.removeEventListener('storage', handleStorageChange)
    }
  }, [dispatch])

  // The global state provider handles all initialization automatically
  // No need for manual loading effects - everything is cached and preloaded

  // Helper function to get icon based on keywords
  const getIconForCuration = (keywords: string[]) => {
    const keywordsStr = (keywords || []).join(' ').toLowerCase()
    
    if (keywordsStr.includes('trend') || keywordsStr.includes('growth')) {
      return TrendingUp
    } else if (keywordsStr.includes('market') || keywordsStr.includes('business')) {
      return BarChart3
    } else if (keywordsStr.includes('user') || keywordsStr.includes('customer') || keywordsStr.includes('demographic')) {
      return Users
    } else if (keywordsStr.includes('tech') || keywordsStr.includes('digital') || keywordsStr.includes('innovation')) {
      return Brain
    } else if (keywordsStr.includes('industry') || keywordsStr.includes('sector')) {
      return Building2
    } else if (keywordsStr.includes('global') || keywordsStr.includes('world') || keywordsStr.includes('international')) {
      return Globe
    } else {
      return Sparkles // Default icon
    }
  }

  // Helper function to generate conversation titles
  const generateConversationTitle = (firstMessage: string): string => {
    // Extract key words and create a meaningful title
    const words = firstMessage.toLowerCase().split(' ')
    const stopWords = ['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'what', 'how', 'why', 'when', 'where', 'can', 'could', 'would', 'should']
    const keyWords = words.filter(word => !stopWords.includes(word) && word.length > 2).slice(0, 3)
    
    if (keyWords.length > 0) {
      return keyWords.map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
    }
    
    // Fallback titles based on content
    if (firstMessage.toLowerCase().includes('market')) return 'Market Analysis'
    if (firstMessage.toLowerCase().includes('trend')) return 'Trend Analysis'
    if (firstMessage.toLowerCase().includes('customer')) return 'Customer Insights'
    if (firstMessage.toLowerCase().includes('business')) return 'Business Discussion'
    if (firstMessage.toLowerCase().includes('data')) return 'Data Analysis'
    
    return 'New Conversation'
  }

  const handleCurationClick = async (curationTitle: string, isExistingCuration: boolean = true) => {
    try {
      // Check if we have any documents available
      if (documents.length === 0) {
        console.log('⚠️ Vault: No documents available for curation generation')
        
        const errorMessage = {
          id: Date.now(),
          type: "ai" as const,
          content: `Sorry, I can't generate the "${curationTitle}" curation because no documents are available. Please upload some documents first.`,
          timestamp: new Date().toISOString()
        }
        
        dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage })
        return
      }

      // FIXED: Always start a new conversation for each curation click
      console.log('🆕 Vault: Starting new conversation for curation:', curationTitle)
      startNewConversation()

      if (isExistingCuration) {
        // FIXED: Check for cached content at the page level first to prevent any loading states
        const curation = [...customCurations, ...dynamicCurations].find(c => c.title === curationTitle)
        if (curation) {
          const contentCacheKey = `curation_content_${curation.id}`
          const cachedContent = cacheManager.get<string>(contentCacheKey)
          
          if (cachedContent) {
            // Display cached content instantly without any loading state or API calls
            console.log('⚡ Vault: Found cached content, displaying instantly without any loaders:', curationTitle)
            
            const cachedMessage = {
              id: Date.now(),
              type: "curation" as const,
              content: cachedContent,
              title: curationTitle,
              isGenerating: false,
              themes: ["AI Curation", "Cached Content", "Instant Access"],
              documentsUsed: selectedFiles.length > 0 ? selectedFiles.length : documents.length,
              documentNames: selectedFiles.length > 0 
                ? selectedFiles.map(fileId => {
                    const doc = documents.find(d => d.id === fileId)
                    return doc?.name || fileId
                  })
                : documents.map(doc => doc.name),
              timestamp: new Date().toISOString(),
              cached: true
            }
            
            dispatch({ type: 'ADD_CHAT_MESSAGE', payload: cachedMessage })
            // FIXED: Ensure loading state is always cleared
            dispatch({ type: 'SET_CURATION_GENERATING', payload: false })
            return // Exit early - no need to call any generation functions
          }
        }
        
        // FIXED: Only set loading state and let vault state provider handle the loading message
        console.log('🔄 Vault: No cached content found, delegating to vault state provider:', curationTitle)
        
        // Check if there's already a loading message for this curation to prevent duplicates
        const existingLoadingMessage = chatMessages.find(msg => 
          msg.type === "curation" && 
          msg.title === curationTitle && 
          msg.isGenerating === true
        )
        
        if (existingLoadingMessage) {
          console.log('⚠️ Vault: Loading message already exists for this curation, skipping duplicate')
          return
        }
        
        // FIXED: Only set the global loading state - don't create loading message here
        dispatch({ type: 'SET_CURATION_GENERATING', payload: true })
        
        try {
          // Call generation function which will handle creating and updating the loading message
          await generateCurationContent(curationTitle, curationTitle)
        } catch (generationError) {
          console.error('Generation function failed:', generationError)
          // Ensure loading state is cleared even if generation fails
          dispatch({ type: 'SET_CURATION_GENERATING', payload: false })
          throw generationError
        }
      } else {
        // This is a user question that should go through the chat interface in the new conversation
        console.log('Vault: Redirecting question to chat interface in new conversation:', curationTitle)
        await sendMessage(curationTitle)
      }
    } catch (error) {
      console.error('Failed to generate curation content:', error)
      
      // FIXED: Always clear loading state on error with multiple attempts
      dispatch({ type: 'SET_CURATION_GENERATING', payload: false })
      
      // Also clear any loading messages
      const updatedMessages = chatMessages.filter(msg => 
        !(msg.type === "curation" && msg.title === curationTitle && msg.isGenerating)
      )
      dispatch({ type: 'SET_CHAT_MESSAGES', payload: updatedMessages })
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now(),
        type: "ai" as const,
        content: `Sorry, I encountered an error generating the curation "${curationTitle}". Please try again or check your connection.`,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage })
    }
  }

  const handleSummaryClick = async (summaryName: string) => {
    if (!summaryName.trim()) return // Don't process empty cards
    
    try {
      // FIXED: Always start a new conversation for each summary click
      console.log('🆕 Vault: Starting new conversation for summary:', summaryName)
      startNewConversation()
      
      await generateSummaryContent(summaryName)
    } catch (error) {
      console.error('Failed to generate summary content:', error)
    }
  }

  const handleDeleteCuration = async (curationId: string) => {
    try {
      await deleteCuration(curationId)
    } catch (error) {
      console.error('Failed to delete curation:', error)
    }
  }

  // Function to format content with better typography
  const formatContent = (content: string) => {
    // Remove text-based separators (equals signs and dashes)
    let formattedContent = content
      .replace(/^=+$/gm, "")
      .replace(/^-+$/gm, "")
      .replace(/^_+$/gm, "")
      .replace(/^\*+$/gm, "")

    // Replace markdown headings with styled headings
    formattedContent = formattedContent
      .replace(/^# (.*?)$/gm, '<h1 class="text-2xl font-bold mb-6 pb-3 border-b-2 border-gradient-to-r from-blue-500 to-purple-600 text-slate-800">$1</h1>')
      .replace(/^## (.*?)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-4 pb-2 border-b border-slate-200 text-slate-700">$1</h2>')
      .replace(/^### (.*?)$/gm, '<h3 class="text-lg font-medium mt-6 mb-3 text-slate-600">$1</h3>')

      // Format lists
      .replace(/^\* (.*?)$/gm, '<li class="ml-4 list-disc my-2 text-slate-700">$1</li>')
      .replace(/^\d+\. (.*?)$/gm, '<li class="ml-4 list-decimal my-2 text-slate-700">$1</li>')

      // Format bold and italic
      .replace(/\*\*(.*?)\*\*/g, '<strong class="font-semibold text-slate-800">$1</strong>')
      .replace(/\*(.*?)\*/g, '<em class="italic text-slate-600">$1</em>')

      // Format paragraphs
      .replace(/\n\n/g, '</p><p class="my-4 text-slate-700 leading-relaxed">')

      // Format line breaks
      .replace(/\n/g, "<br>")

    // Wrap in paragraph if not already
    if (!formattedContent.startsWith("<h") && !formattedContent.startsWith("<p")) {
      formattedContent = `<p class="my-4 text-slate-700 leading-relaxed">${formattedContent}</p>`
    }

    return formattedContent
  }

  const handleConversationSelect = (conversationId: string) => {
    // Use the global state action
    loadConversationHistory(conversationId)
  }

  const handleHistoryCardClick = async (conversation: any, messages: any[]) => {
    // FIXED: Always start a new conversation for each history click
    console.log('🆕 Vault: Starting new conversation for history item:', conversation.title)
    startNewConversation()
    
    // Use preloaded data directly for instant loading - no API calls needed!
    console.log(`⚡ VaultPage: Loading conversation ${conversation.id} instantly from preloaded data in new conversation`)
    
    // Convert messages to chat format instantly
    const chatMessages = messages.map((msg: any, index: number) => ({
      id: index,
      type: msg.role === 'user' ? 'user' as const : 'ai' as const,
      content: msg.content,
      timestamp: msg.timestamp || new Date().toISOString(),
      relevanceScores: msg.relevanceScores,
      contextMode: msg.contextMode
    }))
    
    // Set chat messages instantly in the new conversation - no loading delays!
    dispatch({ type: 'SET_CHAT_MESSAGES', payload: chatMessages })
    
    console.log(`✅ VaultPage: Conversation ${conversation.id} loaded instantly with ${chatMessages.length} messages in new conversation`)
    
    // Keep the current active tab - don't force switch to curations
    // This maintains the user's current view context
  }

  const handleNewConversation = () => {
    // Use the global state action to start a new conversation
    startNewConversation()
    setActiveTab("curations") // Switch to main chat view
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return
    
    const messageToSend = inputMessage.trim()
    // Clear input immediately for better UX
    setInputMessage("")
    
    try {
      await sendMessage(messageToSend)
    } catch (error) {
      console.error('Failed to send message:', error)
      // Optionally restore the message if sending failed
      // setInputMessage(messageToSend)
    }
  }

  return (
    <div className="h-screen bg-white overflow-hidden relative mobile-safe-top mobile-safe-bottom">
      {/* Beautiful flowing wave background - Hidden on mobile for clean ChatGPT look */}
      <div className="absolute inset-0 hidden md:block">
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

      {/* Mobile: ChatGPT-style layout */}
      <div className="md:hidden flex flex-col h-full bg-white relative z-10">
        {/* Ultra-Minimal Mobile Header */}
        <div className="flex-shrink-0 bg-white/95 backdrop-blur-xl border-b border-gray-100 shadow-sm">
          <div className="flex items-center justify-between px-4 py-3 h-14">
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleSidebar}
              className="p-2 hover:bg-gray-100 rounded-xl transition-colors"
            >
              <Menu className="h-5 w-5 text-gray-600" />
            </Button>
            
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8">
                <SixthvaultLogo size="small" />
              </div>
              <span className="text-lg font-semibold text-gray-800">SixthVault</span>
            </div>
            
            <Button
              variant="ghost"
              size="sm"
              onClick={toggleSidebarExpansion}
              className="p-2 hover:bg-gray-100 rounded-xl transition-colors"
            >
              <Settings className="h-5 w-5 text-gray-600" />
            </Button>
          </div>
        </div>

        {/* Chat Area - 85% of screen */}
        <div className="flex-1 overflow-auto bg-gray-50">
          <div className="p-4 min-h-full">
            {chatMessages.length === 0 && !isGeneratingCuration && !isGeneratingSummary && (
              <div className="flex items-center justify-center min-h-[calc(100vh-200px)]">
                <div className="text-center max-w-sm px-6">
                  <div className="mx-auto w-16 h-16 bg-gray-100 rounded-full flex items-center justify-center mb-6">
                    <MessageSquare className="w-8 h-8 text-gray-400" />
                  </div>
                  <h3 className="text-xl font-semibold mb-3 text-gray-800">
                    How can I help you today?
                  </h3>
                  <p className="text-gray-500 text-sm">
                    Ask questions about your documents or start a conversation
                  </p>
                </div>
              </div>
            )}

            <div className="space-y-4 pb-6">
              {chatMessages.map((message) => (
                <div key={message.id} className={`flex ${message.type === "user" ? "justify-end" : "justify-start"}`}>
                  <div
                    className={`max-w-[85%] rounded-2xl px-4 py-3 ${
                      message.type === "user"
                        ? "bg-blue-500 text-white ml-4"
                        : "bg-white border border-gray-200 shadow-sm mr-4"
                    }`}
                  >
                    {message.isGenerating ? (
                      <div className="flex items-center space-x-2">
                        <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
                        <span className="text-gray-500 text-sm">Thinking...</span>
                      </div>
                    ) : (
                      <div className={`text-sm leading-relaxed ${message.type === "user" ? "text-white" : "text-gray-800"}`}>
                        {message.type === "user" ? (
                          message.content
                        ) : (
                          <div
                            dangerouslySetInnerHTML={{
                              __html: formatContent(message.content),
                            }}
                          />
                        )}
                      </div>
                    )}

                    {/* Enhanced Metadata for AI responses - Mobile optimized */}
                    {(message.type === "ai" || message.type === "curation" || message.type === "summary") &&
                      !message.isGenerating && (
                        <div className="mt-3 pt-3 border-t border-gray-100">
                          <div className="flex flex-wrap gap-1 mb-2">
                            {message.themes &&
                              message.themes.slice(0, 2).map((theme, index) => (
                                <Badge key={index} variant="secondary" className="text-xs bg-gray-100 text-gray-600">
                                  {theme}
                                </Badge>
                              ))}
                          </div>

                          <div className="flex items-center justify-between text-xs text-gray-500">
                            <div className="flex items-center space-x-1">
                              <Brain className="w-3 h-3" />
                              <span>{message.documentsUsed || 0} docs</span>
                            </div>
                            
                            {message.documentNames && message.documentNames.length > 0 && (
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => toggleExpandedSources(message.id)}
                                className="h-6 px-2 text-xs text-blue-600 hover:bg-blue-50"
                              >
                                <Eye className="w-3 h-3 mr-1" />
                                Sources
                                {expandedSources.has(message.id) ? (
                                  <ChevronUp className="w-3 h-3 ml-1" />
                                ) : (
                                  <ChevronDown className="w-3 h-3 ml-1" />
                                )}
                              </Button>
                            )}
                          </div>

                          {/* Mobile-optimized expandable sources */}
                          {message.documentNames && message.documentNames.length > 0 && expandedSources.has(message.id) && (
                            <div className="mt-3 pt-3 border-t border-gray-100">
                              <div className="text-xs font-medium text-gray-600 mb-2">Sources:</div>
                              <div className="space-y-1">
                                {message.documentNames.slice(0, 3).map((name, index) => (
                                  <div key={index} className="text-xs text-gray-500 bg-gray-50 rounded px-2 py-1">
                                    {name}
                                  </div>
                                ))}
                                {message.documentNames.length > 3 && (
                                  <div className="text-xs text-gray-400">
                                    +{message.documentNames.length - 3} more
                                  </div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>
                      )}
                  </div>
                </div>
              ))}

              {(isGeneratingCuration || isGeneratingSummary) && (
                <div className="flex justify-start">
                  <div className="bg-white border border-gray-200 shadow-sm rounded-2xl px-4 py-3 mr-4">
                    <div className="flex items-center space-x-2">
                      <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                      <span className="text-gray-600 text-sm">AI is thinking...</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Floating Input Area - 7% of screen */}
        <div className="flex-shrink-0 bg-white border-t border-gray-100 p-4 pb-safe">
          <div className="relative max-w-4xl mx-auto">
            <div className="flex items-center space-x-3 bg-gray-100 rounded-full p-2">
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                placeholder="Message SixthVault..."
                className="flex-1 bg-transparent border-none shadow-none focus:ring-0 text-sm placeholder:text-gray-500"
                disabled={isGeneratingCuration || isGeneratingSummary}
              />
              <Button
                size="sm"
                onClick={handleSendMessage}
                disabled={isGeneratingCuration || isGeneratingSummary || !inputMessage.trim()}
                className="bg-blue-500 hover:bg-blue-600 text-white h-8 w-8 p-0 rounded-full flex-shrink-0"
              >
                {(isGeneratingCuration || isGeneratingSummary) ? (
                  <Loader2 className="w-4 h-4 animate-spin" />
                ) : (
                  <Send className="w-4 h-4" />
                )}
              </Button>
            </div>
          </div>
        </div>

        {/* Mobile Slide-out Menu Overlay */}
        {sidebarOpen && (
          <div className="fixed inset-0 z-50 md:hidden">
            <div className="absolute inset-0 bg-black/20 backdrop-blur-sm" onClick={toggleSidebar} />
            <div className="absolute left-0 top-0 bottom-0 w-80 bg-white shadow-xl overflow-auto">
              <div className="p-4 border-b border-gray-200">
                <div className="flex items-center justify-between">
                  <h2 className="text-lg font-semibold text-gray-800">Menu</h2>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={toggleSidebar}
                    className="p-1"
                  >
                    <X className="h-5 w-5" />
                  </Button>
                </div>
              </div>
              
              <div className="p-4 space-y-4">
                {/* Quick Actions */}
                <div>
                  <h3 className="text-sm font-medium text-gray-600 mb-3">Quick Actions</h3>
                  <div className="space-y-2">
                    <Link href="/documents">
                      <Button variant="ghost" className="w-full justify-start text-sm">
                        <FileText className="w-4 h-4 mr-3" />
                        Document Management
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

                {/* Curations Preview */}
                <div>
                  <h3 className="text-sm font-medium text-gray-600 mb-3">AI Curations</h3>
                  <div className="space-y-2">
                    {[...customCurations, ...dynamicCurations].slice(0, 3).map((curation, index) => {
                      const IconComponent = curation.icon
                      return (
                        <Button
                          key={index}
                          variant="ghost"
                          onClick={() => {
                            handleCurationClick(curation.title)
                            toggleSidebar()
                          }}
                          className="w-full justify-start text-sm p-3 h-auto"
                          disabled={documents.length === 0}
                        >
                          <IconComponent className="w-4 h-4 mr-3 flex-shrink-0" />
                          <span className="truncate text-left">{curation.title}</span>
                        </Button>
                      )
                    })}
                  </div>
                </div>

                {/* Status */}
                <div className="pt-4 border-t border-gray-200">
                  <div className="flex items-center justify-between text-xs text-gray-500">
                    <span>Documents: {documents.length}</span>
                    <div className="flex items-center space-x-1">
                      <div className="w-2 h-2 bg-green-500 rounded-full"></div>
                      <span>Online</span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Desktop: Keep original layout */}
      <div className="hidden md:flex h-full relative z-10">
        {/* Enhanced Left Sidebar with Tabs */}
        <div
          className={`${
            sidebarOpen ? "translate-x-0" : "-translate-x-full"
          } md:translate-x-0 fixed md:relative z-30 ${
            sidebarExpanded ? "w-80" : "w-16"
          } bg-white/95 backdrop-blur-xl border-r border-slate-200/60 flex flex-col h-full transition-all duration-300 ease-in-out shadow-xl`}
        >
          {/* Sidebar Header */}
          <div className="border-b border-slate-200/60 flex-shrink-0 bg-white relative">
            {sidebarExpanded ? (
              <div className="w-full h-32 flex items-center justify-center p-4">
                <SixthvaultLogo size="section-fill" />
              </div>
            ) : (
              <div className="w-full h-16 flex items-center justify-center">
                <SixthvaultLogo size="small" />
              </div>
            )}
            {/* Control buttons positioned absolutely */}
            <div className="absolute top-2 right-2 flex items-center space-x-2">
              <Button
                variant="ghost"
                size="sm"
                onClick={toggleSidebarExpansion}
                className="text-slate-600 hover:bg-slate-100 h-8 w-8 p-0"
              >
                {sidebarExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
              </Button>
            </div>
          </div>

          {sidebarExpanded && (
            <div className="flex-1 overflow-hidden">
              {/* Professional Dropdown Navigation */}
              <div className="p-4 border-b border-slate-200/60">
                <Select value={activeTab} onValueChange={setActiveTab}>
                  <SelectTrigger className="w-full bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors">
                    <SelectValue>
                      <div className="flex items-center">
                        {activeTab === "curations" && (
                          <>
                            <Sparkles className="w-4 h-4 mr-2 text-purple-600" />
                            AI Curations
                          </>
                        )}
                        {activeTab === "summaries" && (
                          <>
                            <Brain className="w-4 h-4 mr-2 text-emerald-600" />
                            AI Summaries
                          </>
                        )}
                        {activeTab === "history" && (
                          <>
                            <History className="w-4 h-4 mr-2 text-blue-600" />
                            Chat History
                          </>
                        )}
                      </div>
                    </SelectValue>
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="history">
                      <div className="flex items-center">
                        <History className="w-4 h-4 mr-2 text-blue-600" />
                        Chat History
                      </div>
                    </SelectItem>
                    <SelectItem value="curations">
                      <div className="flex items-center">
                        <Sparkles className="w-4 h-4 mr-2 text-purple-600" />
                        AI Curations
                      </div>
                    </SelectItem>
                    <SelectItem value="summaries">
                      <div className="flex items-center">
                        <Brain className="w-4 h-4 mr-2 text-emerald-600" />
                        AI Summaries
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div className="flex-1 overflow-hidden">
                {activeTab === "curations" && (
                  <div className="flex-1 overflow-auto px-4 pb-4">
                    <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-xl relative overflow-hidden">
                    {/* Beautiful flowing wave background for curations */}
                    <div className="absolute inset-0 pointer-events-none">
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid slice">
                        <defs>
                          <linearGradient id="curationWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                            <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                            <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                          </linearGradient>
                          <linearGradient id="curationWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.2"/>
                            <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.15"/>
                            <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
                          </linearGradient>
                        </defs>
                        <g stroke="url(#curationWave1)" strokeWidth="1" fill="none" opacity="0.6">
                          <path d="M0,50 Q100,30 200,45 T400,40"/>
                          <path d="M0,70 Q120,50 240,65 T400,60"/>
                        </g>
                        <g stroke="url(#curationWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
                          <path d="M0,90 Q150,70 300,85 T400,80"/>
                          <path d="M0,110 Q180,90 360,105 T400,100"/>
                        </g>
                        <path d="M0,80 Q100,60 200,75 T400,70 L400,200 L0,200 Z" fill="url(#curationWave1)" opacity="0.05"/>
                        <path d="M0,120 Q150,100 300,115 T400,110 L400,200 L0,200 Z" fill="url(#curationWave2)" opacity="0.03"/>
                      </svg>
                    </div>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-semibold text-slate-800 flex items-center">
                        <Sparkles className="w-4 h-4 mr-2 text-purple-600" />
                        AI CURATIONS
                        <Badge variant="secondary" className="ml-2 text-xs bg-purple-100 text-purple-700">
                          {customCurations.length + dynamicCurations.length}
                        </Badge>
                        <Dialog open={showCustomCurationModal} onOpenChange={setShowCustomCurationModal}>
                          <DialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="ml-auto h-6 w-6 p-0 text-purple-600 hover:bg-purple-100 hover:text-purple-700"
                              disabled={documents.length === 0}
                              title={documents.length === 0 ? "Upload documents to create custom curations" : "Create custom curation"}
                            >
                              <Plus className="h-4 w-4" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="sm:max-w-[500px]">
                            <DialogHeader>
                              <DialogTitle className="flex items-center">
                                <div className="p-2 bg-purple-100 rounded-lg mr-3">
                                  <Sparkles className="w-5 h-5 text-purple-600" />
                                </div>
                                Create Custom AI Curation
                              </DialogTitle>
                              <DialogDescription>
                                Create a custom AI curation based on your specific topic and keywords. The AI will analyze your documents and generate insights focused on your chosen theme.
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid gap-2">
                                <Label htmlFor="title" className="text-sm font-medium">
                                  Curation Title *
                                </Label>
                                <Input
                                  id="title"
                                  placeholder="e.g., Market Trends Analysis, Customer Insights, Technology Innovation..."
                                  value={customCurationTitle}
                                  onChange={(e) => setCustomCurationTitle(e.target.value)}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid gap-2">
                                <Label htmlFor="description" className="text-sm font-medium">
                                  Description (Optional)
                                </Label>
                                <Textarea
                                  id="description"
                                  placeholder="Describe what specific insights or analysis you want to focus on..."
                                  value={customCurationDescription}
                                  onChange={(e) => setCustomCurationDescription(e.target.value)}
                                  className="col-span-3 min-h-[80px]"
                                />
                              </div>
                              <div className="grid gap-2">
                                <Label htmlFor="keywords" className="text-sm font-medium">
                                  Keywords & Topics *
                                </Label>
                                <Input
                                  id="keywords"
                                  placeholder="e.g., market research, consumer behavior, digital transformation (comma-separated)"
                                  value={customCurationKeywords}
                                  onChange={(e) => setCustomCurationKeywords(e.target.value)}
                                  className="col-span-3"
                                />
                                <p className="text-xs text-slate-500">
                                  Enter keywords separated by commas. These will guide the AI analysis.
                                </p>
                              </div>
                              <div className="grid gap-2">
                                <Label className="text-sm font-medium">AI Provider & Model</Label>
                                <div className="flex items-center space-x-2 p-3 bg-slate-50 rounded-lg">
                                  <div className="p-1 bg-blue-100 rounded">
                                    <Brain className="w-4 h-4 text-blue-600" />
                                  </div>
                                  <div className="flex-1">
                                    <p className="text-sm font-medium text-slate-700">
                                      {modelProviders.find(p => p.name === selectedProvider)?.displayName || selectedProvider}
                                    </p>
                                    <p className="text-xs text-slate-500">
                                      {availableModels.find(m => m.name === selectedModel)?.displayName || selectedModel}
                                    </p>
                                  </div>
                                  <Badge variant="outline" className="text-xs">
                                    {availableModels.find(m => m.name === selectedModel)?.isLocal ? 'Local' : 'Cloud'}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                            <DialogFooter>
                              <Button
                                variant="outline"
                                onClick={() => {
                                  setShowCustomCurationModal(false)
                                  setCustomCurationTitle("")
                                  setCustomCurationDescription("")
                                  setCustomCurationKeywords("")
                                }}
                                disabled={isCreatingCustomCuration}
                              >
                                Cancel
                              </Button>
                              <Button
                                onClick={async () => {
                                  if (!customCurationTitle.trim() || !customCurationKeywords.trim()) {
                                    return
                                  }
                                  
                                  setIsCreatingCustomCuration(true)
                                  
                                  try {
                                    const keywords = customCurationKeywords.split(',').map(k => k.trim()).filter(k => k.length > 0)
                                    
                                    await createCustomCuration(
                                      customCurationTitle.trim(),
                                      customCurationDescription.trim(),
                                      keywords
                                    )
                                    
                                    // Close modal and reset form
                                    setShowCustomCurationModal(false)
                                    setCustomCurationTitle("")
                                    setCustomCurationDescription("")
                                    setCustomCurationKeywords("")
                                    
                                    // Auto-trigger content generation for the new curation
                                    handleCurationClick(customCurationTitle.trim())
                                  } catch (error) {
                                    console.error('Error creating custom curation:', error)
                                  } finally {
                                    setIsCreatingCustomCuration(false)
                                  }
                                }}
                                disabled={!customCurationTitle.trim() || !customCurationKeywords.trim() || isCreatingCustomCuration}
                                className="bg-purple-600 hover:bg-purple-700"
                              >
                                {isCreatingCustomCuration ? (
                                  <>
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    Creating...
                                  </>
                                ) : (
                                  <>
                                    <Sparkles className="w-4 h-4 mr-2" />
                                    Create Curation
                                  </>
                                )}
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      {/* Scrollable container for AI Curation cards */}
                      <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100 hover:scrollbar-thumb-slate-400 pr-2">
                        <div className="grid grid-cols-1 gap-3">
                          {/* Combine dynamic curations and custom curations */}
                          {[...customCurations, ...dynamicCurations].length === 0 ? (
                            <div className="text-center py-8">
                              <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                                <Sparkles className="w-6 h-6 text-purple-400" />
                              </div>
                              <p className="text-sm text-slate-600 mb-2">No AI curations available</p>
                              <p className="text-xs text-slate-500">Upload documents to generate intelligent curations</p>
                            </div>
                          ) : (
                            [...customCurations, ...dynamicCurations].map((curation, index) => {
                              const IconComponent = curation.icon
                              const isCustomCuration = customCurations.includes(curation)
                              return (
                                <div
                                  key={`${isCustomCuration ? 'custom' : 'dynamic'}-${index}`}
                                  className={`group relative cursor-pointer border rounded-lg transition-all duration-200 py-2 px-3 transform hover:scale-[1.02] active:scale-[0.98] active:shadow-lg ${
                                    curation.active
                                      ? "border-blue-200 bg-blue-50 shadow-md"
                                      : documents.length === 0
                                      ? "border-gray-200 bg-gray-50 cursor-not-allowed opacity-60"
                                      : "border-gray-200 bg-white hover:border-blue-300 hover:shadow-md hover:bg-blue-50/30"
                                  } ${isGeneratingCuration || documents.length === 0 ? "pointer-events-none" : ""}`}
                                >
                                  <div className="flex items-center justify-between">
                                    <div 
                                      className="flex items-center space-x-3 flex-1 min-w-0"
                                      onClick={() => handleCurationClick(curation.title)}
                                    >
                                      <IconComponent className={`w-4 h-4 flex-shrink-0 ${
                                        curation.active ? "text-blue-600" : "text-gray-500"
                                      }`} />
                                      <span className={`text-sm truncate ${
                                        curation.active ? "text-blue-900 font-medium" : "text-gray-700"
                                      }`}>
                                        {curation.title}
                                      </span>
                                    </div>
                                    
                                    <Popover>
                                      <PopoverTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0 text-gray-400 hover:text-gray-600"
                                          onClick={(e) => e.stopPropagation()}
                                        >
                                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 16 16">
                                            <circle cx="2" cy="8" r="1.5"/>
                                            <circle cx="8" cy="8" r="1.5"/>
                                            <circle cx="14" cy="8" r="1.5"/>
                                          </svg>
                                        </Button>
                                      </PopoverTrigger>
                                      <PopoverContent className="w-40 p-1" align="end">
                                        <div className="space-y-1">
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full justify-start text-xs h-8"
                                            onClick={() => {
                                              // Share functionality
                                              navigator.clipboard.writeText(`AI Curation: ${curation.title}`)
                                            }}
                                          >
                                            <Share2 className="w-3 h-3 mr-2" />
                                            Share
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full justify-start text-xs h-8"
                                            onClick={() => {
                                              // Rename functionality (placeholder)
                                              const newName = prompt("Enter new name:", curation.title)
                                              if (newName && newName.trim()) {
                                                console.log("Rename to:", newName)
                                              }
                                            }}
                                          >
                                            <svg className="w-3 h-3 mr-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z" />
                                            </svg>
                                            Rename
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full justify-start text-xs h-8 text-red-600 hover:text-red-700 hover:bg-red-50"
                                            onClick={() => handleDeleteCuration(curation.id)}
                                          >
                                            <X className="w-3 h-3 mr-2" />
                                            Delete
                                          </Button>
                                        </div>
                                      </PopoverContent>
                                    </Popover>
                                  </div>
                                </div>
                              )
                            })
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  </div>
                )}

                {/* AI Summaries Tab */}
                {activeTab === "summaries" && (
                  <div className="flex-1 overflow-auto px-4 pb-4">
                  <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-xl relative overflow-hidden">
                    {/* Beautiful flowing wave background for summaries */}
                    <div className="absolute inset-0 pointer-events-none">
                      <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid slice">
                        <defs>
                          <linearGradient id="summaryWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                            <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                            <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                            <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
                          </linearGradient>
                          <linearGradient id="summaryWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                            <stop offset="0%" stopColor="#fff3e0" stopOpacity="0.2"/>
                            <stop offset="50%" stopColor="#ffcc80" stopOpacity="0.15"/>
                            <stop offset="100%" stopColor="#ffa726" stopOpacity="0.1"/>
                          </linearGradient>
                        </defs>
                        <g stroke="url(#summaryWave1)" strokeWidth="1" fill="none" opacity="0.6">
                          <path d="M0,50 Q100,30 200,45 T400,40"/>
                          <path d="M0,70 Q120,50 240,65 T400,60"/>
                        </g>
                        <g stroke="url(#summaryWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
                          <path d="M0,90 Q150,70 300,85 T400,80"/>
                          <path d="M0,110 Q180,90 360,105 T400,100"/>
                        </g>
                        <path d="M0,80 Q100,60 200,75 T400,70 L400,200 L0,200 Z" fill="url(#summaryWave1)" opacity="0.05"/>
                        <path d="M0,120 Q150,100 300,115 T400,110 L400,200 L0,200 Z" fill="url(#summaryWave2)" opacity="0.03"/>
                      </svg>
                    </div>
                    <CardHeader className="pb-3">
                      <CardTitle className="text-sm font-semibold text-slate-800 flex items-center">
                        <Brain className="w-4 h-4 mr-2 text-emerald-600" />
                        AI SUMMARIES
                        <Badge variant="secondary" className="ml-2 text-xs bg-emerald-100 text-emerald-700">
                          {dynamicSummaries.length}
                        </Badge>
                        <Dialog open={showCustomSummaryModal} onOpenChange={setShowCustomSummaryModal}>
                          <DialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="ml-auto h-6 w-6 p-0 text-emerald-600 hover:bg-emerald-100 hover:text-emerald-700"
                              disabled={documents.length === 0}
                              title={documents.length === 0 ? "Upload documents to create custom summaries" : "Create custom summary"}
                            >
                              <Plus className="h-4 w-4" />
                            </Button>
                          </DialogTrigger>
                          <DialogContent className="sm:max-w-[500px]">
                            <DialogHeader>
                              <DialogTitle className="flex items-center">
                                <div className="p-2 bg-emerald-100 rounded-lg mr-3">
                                  <Brain className="w-5 h-5 text-emerald-600" />
                                </div>
                                Create Custom AI Summary
                              </DialogTitle>
                              <DialogDescription>
                                Create a custom AI summary with your specific focus and keywords. The AI will analyze your documents and generate a targeted summary based on your requirements.
                              </DialogDescription>
                            </DialogHeader>
                            <div className="grid gap-4 py-4">
                              <div className="grid gap-2">
                                <Label htmlFor="summary-title" className="text-sm font-medium">
                                  Summary Title *
                                </Label>
                                <Input
                                  id="summary-title"
                                  placeholder="e.g., Executive Summary, Technical Analysis, Market Overview..."
                                  value={customSummaryTitle}
                                  onChange={(e) => setCustomSummaryTitle(e.target.value)}
                                  className="col-span-3"
                                />
                              </div>
                              <div className="grid gap-2">
                                <Label htmlFor="summary-description" className="text-sm font-medium">
                                  Description (Optional)
                                </Label>
                                <Textarea
                                  id="summary-description"
                                  placeholder="Describe what specific aspects you want the summary to focus on..."
                                  value={customSummaryDescription}
                                  onChange={(e) => setCustomSummaryDescription(e.target.value)}
                                  className="col-span-3 min-h-[80px]"
                                />
                              </div>
                              <div className="grid gap-2">
                                <Label htmlFor="summary-keywords" className="text-sm font-medium">
                                  Keywords & Topics *
                                </Label>
                                <Input
                                  id="summary-keywords"
                                  placeholder="e.g., key findings, recommendations, trends, insights (comma-separated)"
                                  value={customSummaryKeywords}
                                  onChange={(e) => setCustomSummaryKeywords(e.target.value)}
                                  className="col-span-3"
                                />
                                <p className="text-xs text-slate-500">
                                  Enter keywords separated by commas. These will guide the summary focus.
                                </p>
                              </div>
                              <div className="grid gap-2">
                                <Label htmlFor="summary-focus" className="text-sm font-medium">
                                  Focus Area (Optional)
                                </Label>
                                <Input
                                  id="summary-focus"
                                  placeholder="e.g., business implications, technical details, strategic recommendations..."
                                  value={customSummaryFocusArea}
                                  onChange={(e) => setCustomSummaryFocusArea(e.target.value)}
                                  className="col-span-3"
                                />
                                <p className="text-xs text-slate-500">
                                  Specify the main focus or perspective for the summary.
                                </p>
                              </div>
                              <div className="grid gap-2">
                                <Label className="text-sm font-medium">AI Provider & Model</Label>
                                <div className="flex items-center space-x-2 p-3 bg-slate-50 rounded-lg">
                                  <div className="p-1 bg-emerald-100 rounded">
                                    <Brain className="w-4 h-4 text-emerald-600" />
                                  </div>
                                  <div className="flex-1">
                                    <p className="text-sm font-medium text-slate-700">
                                      {modelProviders.find(p => p.name === selectedProvider)?.displayName || selectedProvider}
                                    </p>
                                    <p className="text-xs text-slate-500">
                                      {availableModels.find(m => m.name === selectedModel)?.displayName || selectedModel}
                                    </p>
                                  </div>
                                  <Badge variant="outline" className="text-xs">
                                    {availableModels.find(m => m.name === selectedModel)?.isLocal ? 'Local' : 'Cloud'}
                                  </Badge>
                                </div>
                              </div>
                            </div>
                            <DialogFooter>
                              <Button
                                variant="outline"
                                onClick={() => {
                                  setShowCustomSummaryModal(false)
                                  setCustomSummaryTitle("")
                                  setCustomSummaryDescription("")
                                  setCustomSummaryKeywords("")
                                  setCustomSummaryFocusArea("")
                                }}
                                disabled={isCreatingCustomSummary}
                              >
                                Cancel
                              </Button>
                              <Button
                                onClick={async () => {
                                  if (!customSummaryTitle.trim() || !customSummaryKeywords.trim()) {
                                    return
                                  }
                                  
                                  setIsCreatingCustomSummary(true)
                                  
                                  try {
                                    const keywords = customSummaryKeywords.split(',').map(k => k.trim()).filter(k => k.length > 0)
                                    
                                    await createCustomSummary(
                                      customSummaryTitle.trim(),
                                      customSummaryDescription.trim(),
                                      keywords
                                    )
                                    
                                    // Close modal and reset form
                                    setShowCustomSummaryModal(false)
                                    setCustomSummaryTitle("")
                                    setCustomSummaryDescription("")
                                    setCustomSummaryKeywords("")
                                    setCustomSummaryFocusArea("")
                                    
                                    // Auto-trigger content generation for the new summary
                                    handleSummaryClick(customSummaryTitle.trim())
                                  } catch (error) {
                                    console.error('Error creating custom summary:', error)
                                  } finally {
                                    setIsCreatingCustomSummary(false)
                                  }
                                }}
                                disabled={!customSummaryTitle.trim() || !customSummaryKeywords.trim() || isCreatingCustomSummary}
                                className="bg-emerald-600 hover:bg-emerald-700"
                              >
                                {isCreatingCustomSummary ? (
                                  <>
                                    <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                                    Creating...
                                  </>
                                ) : (
                                  <>
                                    <Brain className="w-4 h-4 mr-2" />
                                    Create Summary
                                  </>
                                )}
                              </Button>
                            </DialogFooter>
                          </DialogContent>
                        </Dialog>
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="pt-0">
                      {/* Scrollable container for AI Summary cards */}
                      <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100 hover:scrollbar-thumb-slate-400 pr-2">
                        <div className="grid grid-cols-1 gap-3">
                          {dynamicSummaries.length === 0 ? (
                            <div className="text-center py-8">
                              <div className="w-12 h-12 bg-emerald-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                                <Brain className="w-6 h-6 text-emerald-400" />
                              </div>
                              <p className="text-sm text-slate-600 mb-2">No AI summaries available</p>
                              <p className="text-xs text-slate-500">Upload documents to generate intelligent summaries</p>
                            </div>
                          ) : (
                            dynamicSummaries.map((summary, index) => (
                              <div
                                key={index}
                                className={`group relative cursor-pointer border rounded-lg transition-all duration-200 py-2 px-3 transform hover:scale-[1.02] active:scale-[0.98] active:shadow-lg ${
                                  summary.active
                                    ? "border-green-200 bg-green-50 shadow-md"
                                    : !summary.name.trim() || documents.length === 0
                                    ? "border-gray-200 bg-gray-50 cursor-not-allowed opacity-60"
                                    : "border-gray-200 bg-white hover:border-emerald-300 hover:shadow-md hover:bg-emerald-50/30"
                                } ${isGeneratingSummary || !summary.name.trim() || documents.length === 0 ? "pointer-events-none" : ""}`}
                              >
                                <div className="flex items-center justify-between">
                                  <div 
                                    className="flex items-center space-x-3 flex-1 min-w-0"
                                    onClick={() => handleSummaryClick(summary.name)}
                                  >
                                    <FileText className={`w-4 h-4 flex-shrink-0 ${
                                      summary.active 
                                        ? "text-green-600" 
                                        : summary.name.trim()
                                        ? "text-gray-500"
                                        : "text-gray-400"
                                    }`} />
                                    <span className={`text-sm truncate ${
                                      summary.active 
                                        ? "text-green-900 font-medium" 
                                        : summary.name.trim()
                                        ? "text-gray-700"
                                        : "text-gray-400"
                                    }`}>
                                      {summary.name || "Empty Slot"}
                                    </span>
                                  </div>
                                  
                                  {summary.name.trim() && (
                                    <Popover>
                                      <PopoverTrigger asChild>
                                        <Button
                                          variant="ghost"
                                          size="sm"
                                          className="opacity-0 group-hover:opacity-100 h-6 w-6 p-0 text-gray-400 hover:text-gray-600"
                                          onClick={(e) => e.stopPropagation()}
                                        >
                                          <svg className="w-4 h-4" fill="currentColor" viewBox="0 0 16 16">
                                            <circle cx="2" cy="8" r="1.5"/>
                                            <circle cx="8" cy="8" r="1.5"/>
                                            <circle cx="14" cy="8" r="1.5"/>
                                          </svg>
                                        </Button>
                                      </PopoverTrigger>
                                      <PopoverContent className="w-40 p-1" align="end">
                                        <div className="space-y-1">
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full justify-start text-xs h-8"
                                            onClick={() => {
                                              navigator.clipboard.writeText(`AI Summary: ${summary.name}`)
                                            }}
                                          >
                                            <Share2 className="w-3 h-3 mr-2" />
                                            Share
                                          </Button>
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="w-full justify-start text-xs h-8 text-red-600 hover:text-red-700 hover:bg-red-50"
                                            onClick={async () => {
                                              try {
                                                // Check if this is a custom summary with an ID that needs backend deletion
                                                const summaryData = state.summaries.data.find(s => s.title === summary.name)
                                                
                                                if (summaryData && summaryData.id) {
                                                  // This is a custom summary stored in backend - delete it properly
                                                  console.log('🗑️ Deleting custom AI summary from backend:', summary.name)
                                                  await deleteSummary(summaryData.id)
                                                  console.log('✅ Custom AI summary deleted successfully:', summary.name)
                                                } else {
                                                  // This is a dynamic summary - just remove from UI
                                                  console.log('✅ Dynamic summary removed from UI:', summary.name)
                                                  const updatedSummaries = state.summaries.dynamicSummaries.filter(s => s.name !== summary.name)
                                                  dispatch({ type: 'SET_DYNAMIC_SUMMARIES', payload: updatedSummaries })
                                                }
                                              } catch (error) {
                                                console.error('Failed to delete summary:', error)
                                                // Show error message to user
                                                const errorMessage = {
                                                  id: Date.now(),
                                                  type: "ai" as const,
                                                  content: `❌ Error deleting summary "${summary.name}": ${error instanceof Error ? error.message : 'Unknown error'}`,
                                                  timestamp: new Date().toISOString(),
                                                  themes: ["System", "Error", "Deletion"]
                                                }
                                                dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage })
                                              }
                                            }}
                                          >
                                            <X className="w-3 h-3 mr-2" />
                                            Delete
                                          </Button>
                                        </div>
                                      </PopoverContent>
                                    </Popover>
                                  )}
                                </div>
                              </div>
                            ))
                          )}
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                  </div>
                )}

                {/* History Tab */}
                {activeTab === "history" && (
                  <div className="flex-1 overflow-hidden">
                    <HistorySection 
                      onConversationSelect={handleConversationSelect}
                      selectedConversationId={selectedConversationId}
                      onHistoryCardClick={handleHistoryCardClick}
                      onNewConversation={handleNewConversation}
                    />
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Collapsed Sidebar Icons */}
          {!sidebarExpanded && (
            <div className="p-2 flex flex-col items-center space-y-4">
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={toggleSidebarExpansion}
              >
                <Sparkles className="w-5 h-5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={toggleSidebarExpansion}
              >
                <Brain className="w-5 h-5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={toggleSidebarExpansion}
              >
                <History className="w-5 h-5" />
              </Button>
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full">
          {/* Ultra-Compact Mobile Header */}
          <div className="mobile-header lg:p-4 border-b border-slate-200/60 bg-white/95 backdrop-blur-xl flex-shrink-0 shadow-sm">
            <div className="flex justify-between items-center w-full">
              {/* Mobile: Minimal Header */}
              <div className="md:hidden flex items-center justify-between w-full">
                <Button variant="ghost" size="sm" onClick={toggleSidebar} className="mobile-menu-button mr-2">
                  <Menu className="h-4 w-4" />
                </Button>
                <Button 
                  onClick={logout}
                  variant="ghost"
                  size="sm"
                  className="h-8 px-2 text-xs"
                >
                  <LogOut className="w-3 h-3" />
                </Button>
              </div>

              {/* Desktop: Full Header */}
              <div className="hidden md:flex justify-between items-center w-full">
                {/* Center Section - Navigation Buttons */}
                <div className="flex items-center space-x-3">
                  <Link href="/documents">
                    <Button 
                      variant="outline"
                      className="bg-slate-50 hover:bg-slate-100 text-slate-700 hover:text-slate-800 border-slate-200 hover:border-slate-300 font-medium px-2 py-1 h-6 text-xs rounded-md transition-all duration-200"
                    >
                      <FileText className="w-3 h-3 mr-1" />
                      DOCUMENT MANAGEMENT
                    </Button>
                  </Link>

                  {user?.is_admin && (
                    <Button 
                      variant="outline"
                      onClick={(e) => {
                        e.preventDefault();
                        // Admin panel temporarily disabled
                      }}
                      className="bg-slate-50 hover:bg-slate-100 text-slate-700 hover:text-slate-800 border-slate-200 hover:border-slate-300 font-medium px-2 py-1 h-6 text-xs rounded-md transition-all duration-200 cursor-not-allowed"
                    >
                      <Settings className="w-3 h-3 mr-1" />
                      ADMIN PANEL
                    </Button>
                  )}
                </div>

                {/* Right Section - Logout and Status */}
                <div className="flex items-center space-x-3">
                  <Button 
                    onClick={logout}
                    variant="outline"
                    className="bg-slate-50 hover:bg-slate-100 text-slate-700 hover:text-slate-800 border-slate-200 hover:border-slate-300 font-medium px-2 py-1 h-6 text-xs rounded-md transition-all duration-200"
                  >
                    <LogOut className="w-3 h-3 mr-1" />
                    LOGOUT
                  </Button>
                  <Badge variant="outline" className="bg-green-50 text-green-700 border-green-200">
                    <Activity className="w-3 h-3 mr-1" />
                    System Active
                  </Badge>
                </div>
              </div>
            </div>
          </div>

          {/* Compact Controls */}
          <div className="p-2 bg-white/95 backdrop-blur-xl border-b border-slate-200/60 flex-shrink-0 shadow-sm">
            {/* Compact Controls */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-2">
              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Documents</label>
                <Popover open={documentSelectorOpen} onOpenChange={setDocumentSelectorOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={documentSelectorOpen}
                      className="w-full justify-between bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors"
                    >
                      <div className="flex items-center space-x-2 flex-1 min-w-0">
                        <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
                        <span className="truncate">
                          {selectedFiles.length === 0
                            ? `ALL Documents (${documents.length})`
                            : selectedFiles.length === 1
                            ? documents.find(doc => doc.id === selectedFiles[0])?.name || "Unknown Document"
                            : `${selectedFiles.length} Documents Selected`}
                        </span>
                      </div>
                      <div className="flex items-center space-x-2 flex-shrink-0">
                        {selectedFiles.length > 0 && (
                          <Badge variant="secondary" className="bg-blue-100 text-blue-700 text-xs">
                            {selectedFiles.length}
                          </Badge>
                        )}
                        <ChevronDown className="h-4 w-4 shrink-0 opacity-50" />
                      </div>
                    </Button>
                  </PopoverTrigger>
                  <PopoverContent className="w-80 p-0" align="start">
                    <Command>
                      <CommandInput placeholder="Search documents..." className="h-9" />
                      <CommandList>
                        <CommandEmpty>No documents found.</CommandEmpty>
                        <CommandGroup>
                          <CommandItem
                            onSelect={() => {
                              setSelectedFiles([])
                              setDocumentSelectorOpen(false)
                            }}
                            className="flex items-center space-x-2 cursor-pointer"
                          >
                            <div className="flex items-center space-x-2 flex-1">
                              <div className={`w-4 h-4 border-2 rounded flex items-center justify-center ${
                                selectedFiles.length === 0 ? 'bg-blue-600 border-blue-600' : 'border-slate-300'
                              }`}>
                                {selectedFiles.length === 0 && <Check className="w-3 h-3 text-white" />}
                              </div>
                              <Globe className="w-4 h-4 text-blue-600" />
                              <span className="font-medium">ALL Documents</span>
                            </div>
                            <Badge variant="outline" className="bg-blue-50 text-blue-700 border-blue-200">
                              {documents.length}
                            </Badge>
                          </CommandItem>
                          
                          {documents.length > 0 && (
                            <>
                              <div className="px-2 py-1.5">
                                <div className="h-px bg-slate-200"></div>
                              </div>
                              
                              <div className="px-2 py-1">
                                <div className="flex items-center justify-between">
                                  <span className="text-xs font-medium text-slate-500 uppercase tracking-wide">
                                    Select Documents
                                  </span>
                                  {selectedFiles.length > 0 && (
                                    <Button
                                      variant="ghost"
                                      size="sm"
                                      onClick={(e) => {
                                        e.stopPropagation()
                                        setSelectedFiles([])
                                      }}
                                      className="h-6 px-2 text-xs text-slate-500 hover:text-slate-700"
                                    >
                                      Clear All
                                    </Button>
                                  )}
                                </div>
                              </div>
                              
                              {documents.map((doc) => {
                                const isSelected = selectedFiles.includes(doc.id)
                                return (
                                  <CommandItem
                                    key={doc.id}
                                    onSelect={() => {
                                      if (isSelected) {
                                        setSelectedFiles(selectedFiles.filter((id: string) => id !== doc.id))
                                      } else {
                                        setSelectedFiles([...selectedFiles, doc.id])
                                      }
                                    }}
                                    className="flex items-center space-x-2 cursor-pointer"
                                  >
                                    <div className="flex items-center space-x-2 flex-1 min-w-0">
                                      <div className={`w-4 h-4 border-2 rounded flex items-center justify-center ${
                                        isSelected ? 'bg-blue-600 border-blue-600' : 'border-slate-300'
                                      }`}>
                                        {isSelected && <Check className="w-3 h-3 text-white" />}
                                      </div>
                                      <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
                                      <div className="flex-1 min-w-0">
                                        <p className="text-sm font-medium text-slate-800 truncate" title={doc.name}>
                                          {doc.name}
                                        </p>
                                        <p className="text-xs text-slate-500 truncate">
                                          {doc.themes.slice(0, 2).join(", ")}
                                          {doc.themes.length > 2 && "..."}
                                        </p>
                                      </div>
                                    </div>
                                    <div className="flex items-center space-x-1 flex-shrink-0">
                                      <Badge variant="outline" className="text-xs bg-slate-50 text-slate-600 border-slate-200">
                                        {doc.themes.length} tags
                                      </Badge>
                                    </div>
                                  </CommandItem>
                                )
                              })}
                            </>
                          )}
                        </CommandGroup>
                      </CommandList>
                    </Command>
                    
                    {selectedFiles.length > 0 && (
                      <div className="border-t border-slate-200 p-3 bg-slate-50">
                        <div className="flex items-center justify-between">
                          <span className="text-sm font-medium text-slate-700">
                            {selectedFiles.length} document{selectedFiles.length !== 1 ? 's' : ''} selected
                          </span>
                          <Button
                            size="sm"
                            onClick={() => setDocumentSelectorOpen(false)}
                            className="h-7 px-3 text-xs bg-blue-600 hover:bg-blue-700"
                          >
                            Apply Selection
                          </Button>
                        </div>
                      </div>
                    )}
                  </PopoverContent>
                </Popover>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Tags Filter</label>
                <Input
                  value={searchTag}
                  onChange={(e) => setSearchTag(e.target.value)}
                  placeholder="Filter by tags..."
                  className="bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Mode</label>
                <Select value={keepContext} onValueChange={(value) => updateContextSettings(value, maxContext)}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="NO">Hybrid Mode</SelectItem>
                    <SelectItem value="YES">Pure RAG</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Context</label>
                <Select value={maxContext} onValueChange={(value) => updateContextSettings(keepContext, value)}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value="NO">Standard</SelectItem>
                    <SelectItem value="YES">Maximum</SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">AI Provider</label>
                <Select value={selectedProvider} onValueChange={(value) => updateProviderSettings(value, selectedModel)}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {modelProviders.map((provider) => (
                      <SelectItem key={provider.name} value={provider.name}>
                        {provider.displayName}
                        {provider.name === 'ollama' && provider.models.length > 0 && ' ✓'}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Model</label>
                <Select value={selectedModel} onValueChange={(value) => updateProviderSettings(selectedProvider, value)}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-8 text-xs hover:border-slate-300 transition-colors">
                    <SelectValue placeholder="Select model..." />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model, index) => (
                      <SelectItem key={`${model.name}-${index}`} value={model.name}>
                        <div className="flex items-center space-x-2">
                          <span>{model.displayName}</span>
                          {model.isLocal && (
                            <Badge variant="outline" className="text-xs bg-green-50 text-green-700 border-green-200">
                              Local
                            </Badge>
                          )}
                        </div>
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </div>
          </div>

          {/* Enhanced Chat Area with Mobile Responsive Design */}
          <div className="flex-1 bg-gradient-to-br from-slate-50/50 via-blue-50/20 to-indigo-50/10 overflow-auto mobile-chat-container mobile-smooth-scroll pb-32 md:pb-6">
            <div className="p-6 min-h-full">
              {chatMessages.length === 0 && !isGeneratingCuration && !isGeneratingSummary && (
                <div className="flex items-center justify-center min-h-[calc(100vh-300px)]">
                  <div className="text-center max-w-2xl px-6">
                    <div className="mx-auto w-16 h-16 bg-gradient-to-br from-slate-100 to-slate-200 rounded-xl flex items-center justify-center mb-6">
                      <MessageSquare className="w-8 h-8 text-slate-500" />
                    </div>
                    <h3 className="text-xl font-semibold mb-3 text-slate-700">
                      Ready to analyze your documents
                    </h3>
                    <p className="text-slate-500 mb-6">
                      Ask questions about your uploaded documents or start a conversation with AI
                    </p>
                    
                    {documents.length === 0 && (
                      <div className="bg-slate-50 border border-slate-200 rounded-lg p-4">
                        <div className="flex items-center justify-center space-x-2 text-slate-600">
                          <FileText className="w-5 h-5" />
                          <span className="text-sm">Upload documents to get started</span>
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              )}

              <div className="space-y-6 pb-6">
                {chatMessages.map((message) => (
                  <div key={message.id} className={`${message.type === "user" ? "text-right" : "text-left"}`}>
                    <div
                      className={`inline-block max-w-full md:max-w-5xl rounded-2xl ${
                        message.type === "user"
                          ? "bg-gradient-to-r from-blue-600 to-indigo-600 text-white p-6 shadow-lg"
                          : message.type === "curation" || message.type === "summary"
                            ? "bg-gradient-to-br from-slate-50 to-blue-50/30 border-2 border-slate-200 text-slate-800 shadow-xl p-8"
                            : "bg-white border-2 border-slate-100 text-slate-800 shadow-xl p-8"
                      }`}
                    >
                      {/* Enhanced Title for curations and summaries */}
                      {(message.type === "curation" || message.type === "summary") && message.title && (
                        <div className="mb-6 pb-4 border-b-2 border-slate-200/60">
                          <div className="flex justify-between items-start">
                            <h3 className="font-bold text-xl flex items-center">
                              {message.type === "curation" ? (
                                <div className="p-2 bg-slate-100 rounded-lg mr-3">
                                  <Sparkles className="w-6 h-6 text-slate-600" />
                                </div>
                              ) : (
                                <div className="p-2 bg-slate-100 rounded-lg mr-3">
                                  <Brain className="w-6 h-6 text-slate-600" />
                                </div>
                              )}
                              {message.title}
                            </h3>
                            <div className="flex items-center text-xs text-slate-500">
                              <Clock className="w-3 h-3 mr-1" />
                              {message.timestamp || "Just now"}
                            </div>
                          </div>
                          {!message.isGenerating && (
                            <div className="flex flex-wrap gap-2 mt-3">
                              <Badge variant="outline" className="text-xs bg-white/80 border-slate-300">
                                <Calendar className="w-3 h-3 mr-1" /> {new Date().toLocaleDateString()}
                              </Badge>
                              <Badge variant="outline" className="text-xs bg-white/80 border-slate-300">
                                <FileText className="w-3 h-3 mr-1" /> {message.documentsUsed || 0} documents
                              </Badge>
                              <Badge variant="outline" className="text-xs bg-white/80 border-slate-300">
                                <Star className="w-3 h-3 mr-1" /> Enterprise Grade
                              </Badge>
                            </div>
                          )}
                        </div>
                      )}

                      {/* Enhanced Loading state */}
                      {message.isGenerating ? (
                        <div className="flex items-center space-x-4 py-8">
                          <div className="relative">
                            <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                            <div className="absolute inset-0 w-8 h-8 border-2 border-blue-200 rounded-full animate-pulse"></div>
                          </div>
                          <div>
                            <span className="text-lg font-medium text-slate-700">AI generating content...</span>
                            <p className="text-sm text-slate-500 mt-1">Analyzing documents and creating insights</p>
                          </div>
                        </div>
                      ) : (
                        <div className={`prose prose-lg max-w-none ${message.type === "user" ? "text-white prose-invert" : ""}`}>
                          {message.type === "user" ? (
                            <p className="text-lg leading-relaxed">{message.content}</p>
                          ) : (
                            <div
                              className="text-base leading-relaxed enterprise-content"
                              dangerouslySetInnerHTML={{
                                __html: formatContent(message.content),
                              }}
                            />
                          )}
                        </div>
                      )}

                      {/* Enhanced Metadata for AI responses */}
                      {(message.type === "ai" || message.type === "curation" || message.type === "summary") &&
                        !message.isGenerating && (
                          <div className="mt-6 pt-6 border-t-2 border-slate-200/60">
                            <div className="flex flex-wrap gap-2 mb-4">
                              {message.themes &&
                                message.themes.map((theme, index) => (
                                  <Badge key={index} variant="secondary" className="text-sm bg-slate-100 text-slate-700 border border-slate-200">
                                    {theme}
                                  </Badge>
                                ))}
                            </div>

                            <div className="flex flex-wrap justify-between items-center">
                              <div className="flex flex-wrap gap-4 text-sm text-slate-600">
                                <div className="flex items-center bg-slate-50 rounded-lg px-3 py-2">
                                  <Brain className="w-4 h-4 mr-2 text-blue-600" />
                                  <span className="font-medium">
                                    {message.type === "curation"
                                      ? "AI Curation"
                                      : message.type === "summary"
                                        ? "AI Summary"
                                        : message.contextMode === "documents-only"
                                          ? "Pure RAG"
                                          : "Hybrid AI"}
                                  </span>
                                </div>
                                <div className="flex items-center bg-slate-50 rounded-lg px-3 py-2">
                                  <FileText className="w-4 h-4 mr-2 text-emerald-600" />
                                  <span className="font-medium">{message.documentsUsed || 0} docs</span>
                                </div>
                                <div className="flex items-center bg-slate-50 rounded-lg px-3 py-2">
                                  <Globe className="w-4 h-4 mr-2 text-purple-600" />
                                  <span className="font-medium">{message.language || "English"}</span>
                                </div>
                                
                                {/* Enhanced Clickable Sources and Relevance */}
                                {message.documentNames && message.documentNames.length > 0 && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => toggleExpandedSources(message.id)}
                                    className="h-8 px-3 text-sm bg-blue-50 text-blue-700 border-blue-200 hover:bg-blue-100 hover:border-blue-300"
                                  >
                                    <Eye className="w-4 h-4 mr-2" />
                                    Sources ({message.documentNames.length})
                                    {expandedSources.has(message.id) ? (
                                      <ChevronUp className="w-4 h-4 ml-2" />
                                    ) : (
                                      <ChevronDown className="w-4 h-4 ml-2" />
                                    )}
                                  </Button>
                                )}

                                {message.relevanceScores && message.relevanceScores.length > 0 && (
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    onClick={() => toggleExpandedRelevance(message.id)}
                                    className="h-8 px-3 text-sm bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300"
                                  >
                                    <BarChart className="w-4 h-4 mr-2" />
                                    Relevance
                                    {expandedRelevance.has(message.id) ? (
                                      <ChevronUp className="w-4 h-4 ml-2" />
                                    ) : (
                                      <ChevronDown className="w-4 h-4 ml-2" />
                                    )}
                                  </Button>
                                )}
                              </div>
                            </div>

                            {/* Enhanced expandable sources */}
                            {message.documentNames && message.documentNames.length > 0 && expandedSources.has(message.id) && (
                              <div className="mt-6 pt-4 border-t border-slate-200">
                                <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center">
                                  <FileText className="w-4 h-4 mr-2 text-blue-600" />
                                  Source Documents ({message.documentNames.length})
                                </h4>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                                  {message.documentNames.map((name, index) => (
                                    <div key={index} className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                                      <div className="flex items-center space-x-2">
                                        <FileText className="w-4 h-4 text-slate-500 flex-shrink-0" />
                                        <span className="text-sm font-medium text-slate-700 truncate" title={name}>
                                          {name}
                                        </span>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Enhanced expandable relevance scores */}
                            {message.relevanceScores && message.relevanceScores.length > 0 && expandedRelevance.has(message.id) && (
                              <div className="mt-6 pt-4 border-t border-slate-200">
                                <h4 className="text-sm font-semibold text-slate-700 mb-3 flex items-center">
                                  <BarChart className="w-4 h-4 mr-2 text-emerald-600" />
                                  Document Relevance Scores
                                </h4>
                                <div className="space-y-3">
                                  {message.relevanceScores.map((item, index) => (
                                    <div key={index} className="bg-slate-50 border border-slate-200 rounded-lg p-3">
                                      <div className="flex items-center justify-between mb-2">
                                        <span className="text-sm font-medium text-slate-700 truncate" title={item.name}>
                                          {item.name}
                                        </span>
                                        <Badge variant="outline" className="text-xs bg-emerald-50 text-emerald-700 border-emerald-200">
                                          {(item.score * 100).toFixed(1)}%
                                        </Badge>
                                      </div>
                                      <div className="w-full bg-slate-200 rounded-full h-2">
                                        <div
                                          className="bg-gradient-to-r from-emerald-500 to-emerald-600 h-2 rounded-full transition-all duration-300"
                                          style={{ width: `${item.score * 100}%` }}
                                        ></div>
                                      </div>
                                    </div>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                    </div>
                  </div>
                ))}

              </div>
            </div>
          </div>

          {/* Enhanced Input Area */}
          <div className="p-6 bg-white/95 backdrop-blur-xl border-t border-slate-200/60 flex-shrink-0 shadow-lg">
            <div className="relative max-w-4xl mx-auto">
              <div className="flex items-center space-x-4 bg-gradient-to-r from-slate-50 to-blue-50/30 rounded-2xl p-4 border-2 border-slate-200/60 shadow-lg">
                <Input
                  value={inputMessage}
                  onChange={(e) => setInputMessage(e.target.value)}
                  onKeyPress={(e) => e.key === "Enter" && !e.shiftKey && handleSendMessage()}
                  placeholder="Ask questions about your documents or start a conversation..."
                  className="flex-1 bg-white/80 border-slate-200 rounded-xl text-base placeholder:text-slate-500 focus:border-blue-300 focus:ring-2 focus:ring-blue-200 transition-all duration-200"
                  disabled={isGeneratingCuration || isGeneratingSummary}
                />
                <Button
                  size="lg"
                  onClick={handleSendMessage}
                  disabled={isGeneratingCuration || isGeneratingSummary || !inputMessage.trim()}
                  className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white px-6 py-3 rounded-xl shadow-lg transition-all duration-200 flex-shrink-0"
                >
                  {(isGeneratingCuration || isGeneratingSummary) ? (
                    <Loader2 className="w-5 h-5 animate-spin" />
                  ) : (
                    <Send className="w-5 h-5" />
                  )}
                </Button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default function VaultPage() {
  return (
    <RouteGuard>
      <VaultPageContent />
    </RouteGuard>
  )
}
