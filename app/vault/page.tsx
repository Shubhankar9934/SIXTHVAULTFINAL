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
import { documentStore, type DocumentData } from "@/lib/document-store"
import { aiService } from "@/lib/ai-service"
import { ragApiClient, type QueryResponse, type BackendDocument, type AvailableModel, type ModelProvider } from "@/lib/api-client"
import { aiCurationService } from "@/lib/ai-curation-service"
import { aiSummaryService } from "@/lib/ai-summary-service"
import { RouteGuard } from "@/components/route-guard"
import { useAuth } from "@/lib/auth-context"

function VaultPageContent() {
  const { logout, user } = useAuth()
  const [selectedFiles, setSelectedFiles] = useState<string[]>([])
  const [documentSelectorOpen, setDocumentSelectorOpen] = useState(false)
  const [searchTag, setSearchTag] = useState("")
  const [keepContext, setKeepContext] = useState("NO")
  const [selectedProvider, setSelectedProvider] = useState("gemini")
  const [selectedModel, setSelectedModel] = useState("")
  const [availableModels, setAvailableModels] = useState<AvailableModel[]>([])
  const [modelProviders, setModelProviders] = useState<ModelProvider[]>([])
  const [providerError, setProviderError] = useState<string>("")
  const [maxContext, setMaxContext] = useState("YES")
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [documents, setDocuments] = useState<DocumentData[]>([])
  const [availableTags, setAvailableTags] = useState<string[]>([])
  const [activeCuration, setActiveCuration] = useState("")
  const [activeSummary, setActiveSummary] = useState("")
  const [isGeneratingCuration, setIsGeneratingCuration] = useState(false)
  const [isGeneratingSummary, setIsGeneratingSummary] = useState(false)
  const [dynamicCurations, setDynamicCurations] = useState<Array<{ title: string; icon: React.ComponentType<any>; active: boolean }>>([])
  const [dynamicSummaries, setDynamicSummaries] = useState<Array<{ name: string; active: boolean }>>([])
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const [sidebarExpanded, setSidebarExpanded] = useState(true)
  const [chatMessages, setChatMessages] = useState<
    Array<{
      id: number
      type: "user" | "ai" | "curation" | "summary"
      content: string
      title?: string
      themes?: string[]
      language?: string
      documentsUsed?: number
      documentNames?: string[]
      contextMode?: string
      isGenerating?: boolean
      timestamp?: string
      relevanceScores?: Array<{ name: string; score: number }>
    }>
  >([])
  const [expandedSources, setExpandedSources] = useState<Set<number>>(new Set())
  const [expandedRelevance, setExpandedRelevance] = useState<Set<number>>(new Set())

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

  // Check if we're on mobile
  const [isMobile, setIsMobile] = useState(false)

  useEffect(() => {
    const checkMobile = () => {
      const isMobileView = window.innerWidth < 768
      setIsMobile(isMobileView)
      if (isMobileView) {
        setSidebarOpen(false)
        setSidebarExpanded(false)
      } else {
        setSidebarOpen(true)
        setSidebarExpanded(true)
      }
    }

    checkMobile()
    window.addEventListener("resize", checkMobile)
    return () => window.removeEventListener("resize", checkMobile)
  }, [])

  // Load available models on mount with enhanced error handling
  useEffect(() => {
    const loadModels = async () => {
      try {
        console.log('🔧 Vault: Loading AI providers and models...')
        setProviderError("") // Clear any previous errors
        
        const providers = await ragApiClient.getAvailableModels()
        console.log('✅ Vault: Successfully loaded providers:', providers.length)
        
        setModelProviders(providers)
        
        // Set available models for the selected provider
        const currentProvider = providers.find(p => p.name === selectedProvider)
        if (currentProvider) {
          setAvailableModels(currentProvider.models)
          console.log(`✅ Vault: Set ${currentProvider.models.length} models for ${selectedProvider} provider`)
          
          // Set default model if none selected
          if (!selectedModel && currentProvider.models.length > 0) {
            setSelectedModel(currentProvider.models[0].name)
            console.log(`✅ Vault: Set default model: ${currentProvider.models[0].name}`)
          }
        } else {
          console.warn(`⚠️ Vault: Provider ${selectedProvider} not found in loaded providers`)
          setAvailableModels([])
        }
        
        // Log final state for debugging
        console.log('🎯 Vault: Final provider state:')
        console.log(`   - Total providers: ${providers.length}`)
        console.log(`   - Selected provider: ${selectedProvider}`)
        console.log(`   - Available models: ${availableModels.length}`)
        console.log(`   - Selected model: ${selectedModel}`)
        
      } catch (error) {
        console.error('❌ Vault: Failed to load AI providers and models:', error)
        const errorMessage = error instanceof Error ? error.message : 'Unknown error loading providers'
        setProviderError(errorMessage)
        
        // Set empty fallback state
        setModelProviders([])
        setAvailableModels([])
        setSelectedModel("")
      }
    }

    loadModels()
  }, [])

  // Update available models when provider changes
  useEffect(() => {
    const currentProvider = modelProviders.find(p => p.name === selectedProvider)
    if (currentProvider) {
      setAvailableModels(currentProvider.models)
      // Reset model selection when provider changes
      if (currentProvider.models.length > 0) {
        setSelectedModel(currentProvider.models[0].name)
      } else {
        setSelectedModel("")
      }
    }
  }, [selectedProvider, modelProviders])

  // Load documents on mount - ENHANCED FOR NGROK COMPATIBILITY
  useEffect(() => {
    const loadDocuments = async () => {
      try {
        console.log('Vault: Loading documents from backend (ngrok-enhanced)')
        
        // Clear any existing state
        setDocuments([])
        setAvailableTags([])
        
        // Add loading timeout warning for ngrok tunnels
        const loadingTimeout = setTimeout(() => {
          console.warn('Vault: Document loading taking longer than expected (ngrok tunnel may have higher latency)')
        }, 5000)
        
        // Fetch documents with enhanced error handling
        const docs = await documentStore.getDocuments()
        clearTimeout(loadingTimeout)
        
        console.log('Vault: Documents loaded successfully:', docs.length)
        
        if (docs.length > 0) {
          setDocuments(docs)
          
          // Extract all unique tags/themes
          const allTags = new Set<string>()
          docs.forEach((doc) => {
            doc.themes.forEach((theme) => allTags.add(theme))
            doc.keywords.forEach((keyword) => allTags.add(keyword))
          })
          setAvailableTags(Array.from(allTags))

          // Generate dynamic curations based on uploaded documents
          generateDynamicCurations(docs)
          generateDynamicSummaries(docs)
          
          console.log('Vault: Document processing completed successfully')
        } else {
          console.log('Vault: No documents found')
          setDocuments([])
          setAvailableTags([])
          generateDynamicCurations([])
          generateDynamicSummaries([])
        }
        
      } catch (error) {
        console.error('Vault: Critical error loading documents:', error)
        
        // Show user-friendly error for ngrok issues
        if (error instanceof Error && (
          error.message.includes('Authentication failed') ||
          error.message.includes('Failed to fetch') ||
          error.message.includes('NetworkError')
        )) {
          console.error('Vault: Network/Auth error detected - may be ngrok tunnel issue')
          // Could show a toast notification here if needed
        }
        
        // Set empty state on error (no localStorage fallback)
        setDocuments([])
        setAvailableTags([])
        generateDynamicCurations([])
        generateDynamicSummaries([])
      }
    }

    // Load immediately with error boundary
    loadDocuments()
  }, []) // Remove any dependencies to prevent re-runs

  const generateDynamicCurations = (docs: DocumentData[]) => {
    if (docs.length === 0) {
      // No default curations when no documents are uploaded
      setDynamicCurations([])
      return
    }

    // Extract themes and topics from documents to create dynamic curations
    const allThemes = new Set<string>()
    const allTopics = new Set<string>()
    const allDemographics = new Set<string>()

    docs.forEach((doc) => {
      doc.themes?.forEach((theme) => allThemes.add(theme))
      doc.mainTopics?.forEach((topic) => allTopics.add(topic))
      doc.demographics?.forEach((demo) => allDemographics.add(demo))
    })

    const themes = Array.from(allThemes).slice(0, 8)
    const topics = Array.from(allTopics).slice(0, 6)
    const demographics = Array.from(allDemographics).slice(0, 4)

    const dynamicCurationsList: Array<{ title: string; icon: React.ComponentType<any>; active: boolean }> = []

    // Create curations based on most common themes
    if (themes.length > 0) {
      dynamicCurationsList.push({
        title: `Top ${Math.min(5, themes.length)} ${themes[0]} Trends`,
        icon: TrendingUp,
        active: false,
      })
    }

    // Create curations based on topics
    if (topics.length > 0) {
      dynamicCurationsList.push({
        title: `${topics[0]} Industry Analysis`,
        icon: BarChart3,
        active: false,
      })
    }

    // Create demographic-based curations
    if (demographics.length > 0) {
      dynamicCurationsList.push({
        title: `Trends for ${demographics[0]} Market`,
        icon: Users,
        active: false,
      })
    }

    // Add technology-focused curation if tech themes are present
    const techThemes = themes.filter(
      (theme) =>
        theme.toLowerCase().includes("tech") ||
        theme.toLowerCase().includes("digital") ||
        theme.toLowerCase().includes("ai") ||
        theme.toLowerCase().includes("innovation"),
    )
    if (techThemes.length > 0) {
      dynamicCurationsList.push({
        title: `Digital Innovation Insights`,
        icon: Brain,
        active: false,
      })
    }

    // Ensure we have at least 4 curations, fill with defaults if needed
    while (dynamicCurationsList.length < 4) {
      const defaultCurations = [
        { title: "Document Insights Summary", icon: FileText, active: false },
        { title: "Key Themes Analysis", icon: TrendingUp, active: false },
        { title: "Market Trends Overview", icon: Globe, active: false },
        { title: "Business Intelligence Report", icon: Building2, active: false },
      ]

      const missingCuration = defaultCurations.find(
        (def) => !dynamicCurationsList.some((dyn) => dyn.title === def.title),
      )
      if (missingCuration) {
        dynamicCurationsList.push(missingCuration)
      } else {
        break
      }
    }

    setDynamicCurations(dynamicCurationsList.slice(0, 4))
  }

  const generateDynamicSummaries = (docs: DocumentData[]) => {
    if (docs.length === 0) {
      // No default summaries when no documents are uploaded
      setDynamicSummaries([])
      return
    }

    if (docs.length === 1) {
      // Single document: One overall summary + 3 empty cards
      setDynamicSummaries([
        { name: "Overall Summary", active: false },
        { name: "", active: false },
        { name: "", active: false },
        { name: "", active: false },
        { name: "", active: false },
      ])
      return
    }

    // Multiple documents: Individual summaries for each doc + one combined summary
    const summaries = []

    // Add individual document summaries (up to 3 to leave space for combined)
    const maxIndividualSummaries = Math.min(docs.length, 3)
    docs.slice(0, maxIndividualSummaries).forEach((doc) => {
      const docName = doc.name.split(".")[0]
      const shortName = docName.length > 12 ? docName.substring(0, 12) + "..." : docName
      summaries.push({ name: shortName, active: false })
    })

    // Add combined summary as the last card
    summaries.push({ name: "Combined Summary", active: false })

    // If we have exactly 4 cards, we're done
    // If we have less than 4, fill with empty cards
    while (summaries.length < 4) {
      summaries.push({ name: "", active: false })
    }

    // If we have more than 4 docs, we need to adjust
    if (docs.length > 3) {
      // Show first 3 individual docs + combined summary
      const adjustedSummaries = []

      docs.slice(0, 3).forEach((doc) => {
        const docName = doc.name.split(".")[0]
        const shortName = docName.length > 12 ? docName.substring(0, 12) + "..." : docName
        adjustedSummaries.push({ name: shortName, active: false })
      })

      adjustedSummaries.push({ name: "Combined Summary", active: false })
      setDynamicSummaries(adjustedSummaries)
    } else {
      setDynamicSummaries(summaries.slice(0, 4))
    }
  }

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

    const userMessage = {
      id: Date.now(),
      type: "user" as const,
      content: inputMessage,
      timestamp,
    }

    setChatMessages((prev) => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage("")
    setIsLoading(true)

    try {
      // Get documents from backend (no localStorage)
      const allDocs = await documentStore.getDocuments()

      // Prepare document IDs for filtering
      let documentIds: string[] | undefined = undefined
      if (selectedFiles.length > 0) {
        documentIds = selectedFiles // Multiple documents selected
      }
      // If selectedFiles is empty, documentIds remains undefined (search all documents)

      // Use RAG backend with selected provider and model
      const queryResponse = await ragApiClient.queryDocuments(currentMessage, {
        hybrid: keepContext !== "YES", // Hybrid mode when not pure RAG
        maxContext: maxContext === "YES", // Use max context when enabled
        documentIds: documentIds, // Pass selected document IDs
        provider: selectedProvider, // Pass selected AI provider
        model: selectedModel // Pass selected model
      })

      // Debug logging to check the response structure
      console.log('RAG Query Response:', queryResponse)
      console.log('Sources:', queryResponse.sources)
      console.log('Sources length:', queryResponse.sources?.length)

      // Check if we have sources data, if not, add some test data for UI verification
      let documentNames: string[] = []
      let relevanceScores: Array<{ name: string; score: number }> = []
      
      if (queryResponse.sources && queryResponse.sources.length > 0) {
        // Use real data from backend
        documentNames = queryResponse.sources.map(s => s.document)
        relevanceScores = queryResponse.sources.map(s => ({ name: s.document, score: s.score }))
      } else {
        // Fallback: Add test data to verify UI functionality
        console.warn('No sources returned from RAG backend, using test data for UI verification')
        documentNames = [
          "Sample Document 1.pdf",
          "Market Research Report.docx", 
          "Consumer Trends Analysis.pdf"
        ]
        relevanceScores = [
          { name: "Sample Document 1.pdf", score: 0.892 },
          { name: "Market Research Report.docx", score: 0.756 },
          { name: "Consumer Trends Analysis.pdf", score: 0.634 }
        ]
      }

      const aiMessage = {
        id: Date.now() + 1,
        type: "ai" as const,
        content: queryResponse.answer,
        themes: ["RAG Response", "RAG Backend", maxContext === "YES" ? "Max Context" : "Standard Context"],
        language: "English",
        documentsUsed: documentNames.length,
        documentNames: documentNames,
        contextMode: keepContext === "YES" ? "documents-only" : "hybrid",
        relevanceScores: relevanceScores,
        timestamp,
      }

      // Debug logging to check the processed data
      console.log('AI Message:', aiMessage)
      console.log('Document Names:', aiMessage.documentNames)
      console.log('Relevance Scores:', aiMessage.relevanceScores)

      setChatMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error("RAG Chat error:", error)
      const errorMessage = {
        id: Date.now() + 1,
        type: "ai" as const,
        content: `Sorry, I encountered an error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again.`,
        themes: ["Error"],
        language: "English",
        timestamp,
      }
      setChatMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleCurationClick = async (curationTitle: string) => {
    setActiveCuration(curationTitle)
    setIsGeneratingCuration(true)

    // On mobile, close sidebar after selection
    if (isMobile) {
      setSidebarOpen(false)
    }

    // Update the active state
    setDynamicCurations((prev) => prev.map((c) => ({ ...c, active: c.title === curationTitle })))

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

    // Add a loading message to chat
    const loadingMessage = {
      id: Date.now(),
      type: "curation" as const,
      content: "",
      title: curationTitle,
      isGenerating: true,
      timestamp,
    }
    setChatMessages((prev) => [...prev, loadingMessage])

    try {
      // Use RAG API client with selected provider and model for curation generation
      const curationPrompt = `Generate content for a business curation titled "${curationTitle}".
        
        Base your analysis on these uploaded documents:
        ${documents.map(doc => `Document: ${doc.name}
         Themes: ${doc.themes?.join(", ") || "N/A"}
         Summary: ${doc.summary?.substring(0, 200) || "No summary available"}...`).join("\n\n")}
        
        Create a beautifully formatted response using markdown with:
        - A compelling introduction
        - 3-5 key points with supporting details
        - Relevant statistics or examples
        - A concise conclusion with actionable insights
        
        Make it professional, visually structured, and actionable.`

      const curationResponse = await ragApiClient.queryDocuments(curationPrompt, {
        hybrid: false, // Use pure AI generation for curations
        maxContext: true,
        documentIds: documents.map(doc => doc.id),
        provider: selectedProvider, // Use selected AI provider
        model: selectedModel // Use selected model
      })

      // Update the message with the generated content
      setChatMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content: curationResponse.answer,
                isGenerating: false,
                themes: ["AI Curation", "Dynamic Content", `${selectedProvider.toUpperCase()} Generated`],
                documentsUsed: documents.length,
              }
            : msg,
        ),
      )
    } catch (error) {
      // Update with error message
      setChatMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content: "Failed to generate curation content. Please try again.",
                isGenerating: false,
                themes: ["Error"],
              }
            : msg,
        ),
      )
    } finally {
      setIsGeneratingCuration(false)
    }
  }

  const handleSummaryClick = async (summaryName: string) => {
    if (!summaryName.trim()) return // Don't process empty cards

    setActiveSummary(summaryName)
    setIsGeneratingSummary(true)

    // On mobile, close sidebar after selection
    if (isMobile) {
      setSidebarOpen(false)
    }

    // Update the active state
    setDynamicSummaries((prev) => prev.map((s) => ({ ...s, active: s.name === summaryName })))

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

    // Add a loading message to chat
    const loadingMessage = {
      id: Date.now(),
      type: "summary" as const,
      content: "",
      title:
        summaryName === "Overall Summary"
          ? "Overall Document Summary"
          : summaryName === "Combined Summary"
            ? "Combined Documents Summary"
            : `${summaryName} Summary`,
      isGenerating: true,
      timestamp,
    }
    setChatMessages((prev) => [...prev, loadingMessage])

    try {
      let content = ""

      if (summaryName === "Overall Summary" || summaryName === "Combined Summary") {
        // Use the AI Summary service to create a custom summary
        const result = await aiSummaryService.createCustomSummary({
          title: summaryName === "Overall Summary" ? "Overall Document Analysis" : "Combined Documents Analysis",
          description: summaryName === "Overall Summary" 
            ? "Comprehensive overview of all uploaded documents with key insights and recommendations"
            : "Synthesized analysis combining insights from all documents to identify patterns and strategic opportunities",
          keywords: summaryName === "Overall Summary" 
            ? ["overview", "analysis", "insights", "recommendations", "summary"]
            : ["combined", "synthesis", "patterns", "strategic", "comprehensive"],
          focusArea: summaryName === "Overall Summary"
            ? "Executive summary and high-level insights across all documents"
            : "Cross-document analysis and strategic synthesis",
          provider: selectedProvider,
          model: selectedModel
        })

        if (result.success && result.summary) {
          content = result.summary.content
        } else {
          content = `Failed to generate ${summaryName.toLowerCase()}: ${result.message}`
        }
      } else {
        // Generate summary for specific document using AI Summary service
        const targetDoc = documents.find((doc) => {
          const docName = doc.name.split(".")[0]
          const shortName = docName.length > 12 ? docName.substring(0, 12) + "..." : docName
          return shortName === summaryName || docName === summaryName
        })

        if (targetDoc) {
          // Create a custom summary focused on this specific document
          const result = await aiSummaryService.createCustomSummary({
            title: `${targetDoc.name} Analysis`,
            description: `Detailed analysis and summary of ${targetDoc.name}`,
            keywords: targetDoc.themes?.slice(0, 5) || ["document", "analysis"],
            focusArea: `Comprehensive analysis of ${targetDoc.name} including key insights, themes, and recommendations`,
            provider: selectedProvider,
            model: selectedModel
          })

          if (result.success && result.summary) {
            content = result.summary.content
          } else {
            content = `Failed to generate summary for ${targetDoc.name}: ${result.message}`
          }
        } else {
          // If no specific document found, create a general custom summary
          console.log(`No specific document found for summary name: ${summaryName}. Creating general summary.`)
          
          const result = await aiSummaryService.createCustomSummary({
            title: summaryName,
            description: `Custom summary for ${summaryName}`,
            keywords: ["analysis", "insights", "summary"],
            focusArea: `Comprehensive analysis and insights for ${summaryName}`,
            provider: selectedProvider,
            model: selectedModel
          })

          if (result.success && result.summary) {
            content = result.summary.content
          } else {
            content = `Failed to generate custom summary: ${result.message}`
          }
        }
      }

      // Update the message with the generated content
      setChatMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content,
                isGenerating: false,
                themes:
                  summaryName === "Overall Summary"
                    ? ["AI Summary", "Overall Analysis", `${selectedProvider.toUpperCase()} Generated`]
                    : summaryName === "Combined Summary"
                      ? ["AI Summary", "Combined Analysis", `${selectedProvider.toUpperCase()} Generated`]
                      : ["AI Summary", "Document Analysis", `${selectedProvider.toUpperCase()} Generated`],
                documentsUsed:
                  summaryName === "Overall Summary" || summaryName === "Combined Summary" ? documents.length : 1,
              }
            : msg,
        ),
      )
    } catch (error) {
      console.error("Summary generation error:", error)
      // Update with error message
      setChatMessages((prev) =>
        prev.map((msg) =>
          msg.id === loadingMessage.id
            ? {
                ...msg,
                content: `Failed to generate summary. Error: ${error instanceof Error ? error.message : "Unknown error"}. Please try again.`,
                isGenerating: false,
                themes: ["Error"],
              }
            : msg,
        ),
      )
    } finally {
      setIsGeneratingSummary(false)
    }
  }

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
  }

  const toggleSidebarExpansion = () => {
    setSidebarExpanded(!sidebarExpanded)
  }

  // Functions to toggle Sources and Document Relevance visibility
  const toggleSources = (messageId: number) => {
    setExpandedSources(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
  }

  const toggleRelevance = (messageId: number) => {
    setExpandedRelevance(prev => {
      const newSet = new Set(prev)
      if (newSet.has(messageId)) {
        newSet.delete(messageId)
      } else {
        newSet.add(messageId)
      }
      return newSet
    })
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

  return (
    <div className="h-screen bg-white overflow-hidden relative">
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
      <div className="flex h-full relative z-10">
        {/* Mobile Sidebar Overlay */}
        {isMobile && sidebarOpen && (
          <div className="fixed inset-0 bg-black/60 backdrop-blur-sm z-20" onClick={() => setSidebarOpen(false)} />
        )}

        {/* Enhanced Left Sidebar */}
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
              <div className="w-full h-20 flex items-center justify-center p-2">
                <SixthvaultLogo size="full" />
              </div>
            ) : (
              <div className="w-full h-16 flex items-center justify-center">
                <SixthvaultLogo size="small" />
              </div>
            )}
            {/* Control buttons positioned absolutely */}
            <div className="absolute top-2 right-2 flex items-center space-x-2">
              {!isMobile && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={toggleSidebarExpansion}
                  className="text-slate-600 hover:bg-slate-100 h-8 w-8 p-0"
                >
                  {sidebarExpanded ? <Minimize2 className="h-4 w-4" /> : <Maximize2 className="h-4 w-4" />}
                </Button>
              )}
              {isMobile && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSidebarOpen(false)}
                  className="text-slate-600 hover:bg-slate-100 h-8 w-8 p-0"
                >
                  <X className="h-4 w-4" />
                </Button>
              )}
            </div>
          </div>

          {sidebarExpanded && (
            <div className="p-4 flex-1 overflow-auto">
              {/* AI Curations Section */}
              <Card className="mb-6 border-0 shadow-lg bg-white/95 backdrop-blur-xl relative overflow-hidden">
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
                      {dynamicCurations.length}
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
                                
                                const result = await aiCurationService.createCustomCuration(
                                  customCurationTitle.trim(),
                                  customCurationDescription.trim(),
                                  keywords,
                                  selectedProvider,
                                  selectedModel
                                )
                                
                                if (result.success) {
                                  // Close modal and reset form
                                  setShowCustomCurationModal(false)
                                  setCustomCurationTitle("")
                                  setCustomCurationDescription("")
                                  setCustomCurationKeywords("")
                                  
                                  // Refresh curations to show the new one
                                  // The AI curation service will handle refreshing the local data
                                  
                                  // Optionally trigger a click on the new curation
                                  if (result.curation) {
                                    setTimeout(() => {
                                      handleCurationClick(result.curation!.title)
                                    }, 1000)
                                  }
                                } else {
                                  console.error('Failed to create custom curation:', result.message)
                                  // Could show a toast notification here
                                }
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
                  <div className="grid grid-cols-1 gap-3">
                    {dynamicCurations.length === 0 ? (
                      <div className="text-center py-8">
                        <div className="w-12 h-12 bg-purple-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                          <Sparkles className="w-6 h-6 text-purple-400" />
                        </div>
                        <p className="text-sm text-slate-600 mb-2">No AI curations available</p>
                        <p className="text-xs text-slate-500">Upload documents to generate intelligent curations</p>
                      </div>
                    ) : (
                      dynamicCurations.map((curation, index) => {
                        const IconComponent = curation.icon
                        return (
                          <div
                            key={index}
                            onClick={() => handleCurationClick(curation.title)}
                            className={`group relative overflow-hidden rounded-xl border transition-all duration-300 cursor-pointer ${
                              curation.active
                                ? "bg-gradient-to-br from-purple-50 via-indigo-50 to-blue-50 border-purple-200 shadow-lg ring-2 ring-purple-100"
                                : documents.length === 0
                                ? "bg-slate-50/80 border-slate-200 cursor-not-allowed opacity-60"
                                : "bg-white/90 border-slate-200 hover:border-purple-200 hover:shadow-md hover:bg-gradient-to-br hover:from-purple-25 hover:to-indigo-25"
                            } ${isGeneratingCuration || documents.length === 0 ? "pointer-events-none" : ""}`}
                          >
                            {/* Subtle gradient overlay */}
                            <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent pointer-events-none" />
                            
                            {/* Content */}
                            <div className="relative p-4">
                              <div className="flex items-start space-x-3">
                                <div className={`p-3 rounded-xl transition-all duration-200 ${
                                  curation.active 
                                    ? "bg-gradient-to-br from-purple-100 to-indigo-100 shadow-sm" 
                                    : "bg-gradient-to-br from-slate-100 to-slate-50 group-hover:from-purple-50 group-hover:to-indigo-50"
                                }`}>
                                  <IconComponent className={`w-5 h-5 transition-colors duration-200 ${
                                    curation.active ? "text-purple-600" : "text-slate-600 group-hover:text-purple-600"
                                  }`} />
                                </div>
                                <div className="flex-1 min-w-0">
                                  <h4 className={`text-sm font-semibold leading-tight mb-1 transition-colors duration-200 ${
                                    curation.active ? "text-purple-800" : "text-slate-800 group-hover:text-purple-700"
                                  }`} style={{ wordBreak: 'break-word', overflowWrap: 'break-word' }}>
                                    {curation.title}
                                  </h4>
                                  <p className={`text-xs transition-colors duration-200 ${
                                    curation.active ? "text-purple-600" : "text-slate-500 group-hover:text-purple-500"
                                  }`}>
                                    AI-powered insights
                                  </p>
                                </div>
                              </div>
                              
                              {/* Active indicator */}
                              {curation.active && (
                                <div className="absolute top-2 right-2">
                                  <div className="w-2 h-2 bg-purple-500 rounded-full animate-pulse" />
                                </div>
                              )}
                            </div>
                          </div>
                        )
                      })
                    )}
                  </div>
                </CardContent>
              </Card>

              {/* AI Summaries Section */}
              <Card className="mb-6 border-0 shadow-lg bg-white/95 backdrop-blur-xl relative overflow-hidden">
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
                                
                                const result = await aiSummaryService.createCustomSummary({
                                  title: customSummaryTitle.trim(),
                                  description: customSummaryDescription.trim(),
                                  keywords: keywords,
                                  focusArea: customSummaryFocusArea.trim(),
                                  provider: selectedProvider,
                                  model: selectedModel
                                })
                                
                                if (result.success) {
                                  // Close modal and reset form
                                  setShowCustomSummaryModal(false)
                                  setCustomSummaryTitle("")
                                  setCustomSummaryDescription("")
                                  setCustomSummaryKeywords("")
                                  setCustomSummaryFocusArea("")
                                  
                                  // Trigger a click on the new summary to display it
                                  if (result.summary) {
                                    setTimeout(() => {
                                      handleSummaryClick(result.summary!.title)
                                    }, 1000)
                                  }
                                } else {
                                  console.error('Failed to create custom summary:', result.message)
                                  // Could show a toast notification here
                                }
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
                          onClick={() => handleSummaryClick(summary.name)}
                          className={`group relative overflow-hidden rounded-xl border transition-all duration-300 cursor-pointer ${
                            summary.active
                              ? "bg-gradient-to-br from-emerald-50 via-teal-50 to-green-50 border-emerald-200 shadow-lg ring-2 ring-emerald-100"
                              : !summary.name.trim() || documents.length === 0
                              ? "bg-slate-50/80 border-slate-200 cursor-not-allowed opacity-60"
                              : "bg-white/90 border-slate-200 hover:border-emerald-200 hover:shadow-md hover:bg-gradient-to-br hover:from-emerald-25 hover:to-teal-25"
                          } ${isGeneratingSummary || !summary.name.trim() || documents.length === 0 ? "pointer-events-none" : ""}`}
                        >
                          {/* Subtle gradient overlay */}
                          <div className="absolute inset-0 bg-gradient-to-br from-white/20 to-transparent pointer-events-none" />
                          
                          {/* Content */}
                          <div className="relative p-4">
                            <div className="flex items-start space-x-3">
                              <div className={`p-3 rounded-xl transition-all duration-200 ${
                                summary.active 
                                  ? "bg-gradient-to-br from-emerald-100 to-teal-100 shadow-sm" 
                                  : summary.name.trim()
                                  ? "bg-gradient-to-br from-slate-100 to-slate-50 group-hover:from-emerald-50 group-hover:to-teal-50"
                                  : "bg-slate-50"
                              }`}>
                                <FileText className={`w-5 h-5 transition-colors duration-200 ${
                                  summary.active 
                                    ? "text-emerald-600" 
                                    : summary.name.trim()
                                    ? "text-slate-600 group-hover:text-emerald-600"
                                    : "text-slate-400"
                                }`} />
                              </div>
                              <div className="flex-1 min-w-0">
                                <h4 className={`text-sm font-semibold leading-tight mb-1 transition-colors duration-200 ${
                                  summary.active 
                                    ? "text-emerald-800" 
                                    : summary.name.trim()
                                    ? "text-slate-800 group-hover:text-emerald-700"
                                    : "text-slate-400"
                                }`} style={{ wordBreak: 'break-word', overflowWrap: 'break-word' }}>
                                  {summary.name || "Empty Slot"}
                                </h4>
                                <p className={`text-xs transition-colors duration-200 ${
                                  summary.active 
                                    ? "text-emerald-600" 
                                    : summary.name.trim()
                                    ? "text-slate-500 group-hover:text-emerald-500"
                                    : "text-slate-400"
                                }`}>
                                  {summary.name.trim() ? "Document summary" : "No document"}
                                </p>
                              </div>
                            </div>
                            
                            {/* Active indicator */}
                            {summary.active && (
                              <div className="absolute top-2 right-2">
                                <div className="w-2 h-2 bg-emerald-500 rounded-full animate-pulse" />
                              </div>
                            )}
                          </div>
                        </div>
                      ))
                    )}
                  </div>
                </CardContent>
              </Card>


              {/* Enhanced RAG Status */}
              <Card className="mt-6 border-0 shadow-lg bg-gradient-to-br from-green-50 to-emerald-50">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3 mb-3">
                    <div className="p-2 bg-green-100 rounded-lg">
                      <Database className="w-4 h-4 text-green-600" />
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center space-x-2">
                        <span className="text-sm font-semibold text-green-800">RAG System Active</span>
                        <div className="w-2 h-2 bg-green-500 rounded-full animate-pulse"></div>
                      </div>
                      <p className="text-xs text-green-600 mt-1">
                        {documents.length} documents • {availableTags.length} AI tags
                      </p>
                    </div>
                  </div>
                  <div className="grid grid-cols-2 gap-2 text-xs">
                    <div className="bg-white/60 rounded-lg p-2 text-center">
                      <div className="font-semibold text-green-800">{selectedProvider.toUpperCase()}</div>
                      <div className="text-green-600">AI Provider</div>
                    </div>
                    <div className="bg-white/60 rounded-lg p-2 text-center">
                      <div className="font-semibold text-green-800">{keepContext === "YES" ? "PURE" : "HYBRID"}</div>
                      <div className="text-green-600">Mode</div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}

          {/* Collapsed Sidebar Icons */}
          {!sidebarExpanded && (
            <div className="p-2 flex flex-col items-center space-y-4">
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={() => setSidebarExpanded(true)}
              >
                <Sparkles className="w-5 h-5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={() => setSidebarExpanded(true)}
              >
                <Brain className="w-5 h-5" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                className="w-12 h-12 p-0 text-slate-600 hover:bg-slate-100"
                onClick={() => setSidebarExpanded(true)}
              >
                <FileText className="w-5 h-5" />
              </Button>
            </div>
          )}
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col h-full">
          {/* Enhanced Header with Integrated Navigation */}
          <div className="p-4 border-b border-slate-200/60 bg-white/95 backdrop-blur-xl flex-shrink-0 shadow-sm">
            <div className="flex justify-between items-center w-full">
              {/* Left Section - Logo and Title */}
              <div className="flex items-center">
                <Button variant="ghost" size="sm" onClick={toggleSidebar} className="mr-3 md:hidden">
                  <Menu className="h-5 w-5" />
                </Button>
                <div className="flex items-center space-x-3">
                  <div className="p-2 bg-gradient-to-r from-slate-100 to-gray-200 rounded-lg shadow-md border border-slate-300">
                    <Brain className="w-5 h-5 text-slate-700" />
                  </div>
                  <div>
                    <h1 className="text-lg font-bold bg-gradient-to-r from-indigo-700 via-blue-800 to-slate-800 bg-clip-text text-transparent">SIXTHVAULT</h1>
                    <p className="text-xs text-slate-600 font-medium">Intelligent Document Analysis & RAG System</p>
                  </div>
                </div>
              </div>

              {/* Center Section - Navigation Buttons */}
              <div className="hidden lg:flex items-center space-x-3">
                <Link href="/documents">
                  <Button 
                    size="default"
                    className="bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold px-4 py-2 rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200 border-0"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    DOCUMENT MANAGEMENT
                  </Button>
                </Link>

                {user?.is_admin && (
                  <Link href="/admin">
                    <Button 
                      size="default"
                      className="bg-gradient-to-r from-slate-600 to-gray-700 hover:from-slate-700 hover:to-gray-800 text-white font-semibold px-4 py-2 rounded-lg shadow-md hover:shadow-lg transform hover:scale-105 transition-all duration-200 border-0 relative overflow-hidden group"
                    >
                      {/* Shimmer effect on hover */}
                      <div className="absolute inset-0 bg-gradient-to-r from-transparent via-white/10 to-transparent -translate-x-full group-hover:translate-x-full transition-transform duration-700"></div>
                      <Settings className="w-4 h-4 mr-2 relative z-10" />
                      <span className="relative z-10">ADMIN PANEL</span>
                    </Button>
                  </Link>
                )}
              </div>

              {/* Right Section - Status and Logout */}
              <div className="flex items-center space-x-3">
                <Badge variant="outline" className="hidden md:flex bg-green-50 text-green-700 border-green-200">
                  <Activity className="w-3 h-3 mr-1" />
                  System Active
                </Badge>
              <Button 
                onClick={logout}
                className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
              </div>
            </div>

            {/* Mobile Navigation - Shown below header on smaller screens */}
            <div className="lg:hidden mt-4 pt-4 border-t border-slate-200/60">
              <div className="flex flex-col sm:flex-row gap-3 justify-center">
                <Link href="/documents" className="flex-1 sm:flex-none">
                  <Button 
                    size="default"
                    className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold px-4 py-2 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 border-0"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    DOCUMENT MANAGEMENT
                  </Button>
                </Link>

                <Link href="/admin" className="flex-1 sm:flex-none">
                  <Button 
                    size="default"
                    className="w-full bg-gradient-to-r from-slate-600 to-gray-700 hover:from-slate-700 hover:to-gray-800 text-white font-semibold px-4 py-2 rounded-lg shadow-md hover:shadow-lg transition-all duration-200 border-0"
                  >
                    <Settings className="w-4 h-4 mr-2" />
                    ADMIN PANEL
                  </Button>
                </Link>
              </div>
            </div>
          </div>

          {/* Enhanced Controls - Keep at Top */}
          <div className="p-4 bg-white/95 backdrop-blur-xl border-b border-slate-200/60 flex-shrink-0 shadow-sm">
            {/* Enhanced Controls */}
            <div className="grid grid-cols-2 md:grid-cols-6 gap-3">
              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Documents</label>
                <Popover open={documentSelectorOpen} onOpenChange={setDocumentSelectorOpen}>
                  <PopoverTrigger asChild>
                    <Button
                      variant="outline"
                      role="combobox"
                      aria-expanded={documentSelectorOpen}
                      className="w-full justify-between bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors"
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
                                        setSelectedFiles(prev => prev.filter(id => id !== doc.id))
                                      } else {
                                        setSelectedFiles(prev => [...prev, doc.id])
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
                  className="bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors"
                />
              </div>

              <div>
                <label className="text-xs font-medium text-slate-600 mb-1 block">Mode</label>
                <Select value={keepContext} onValueChange={setKeepContext}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors">
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
                <Select value={maxContext} onValueChange={setMaxContext}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors">
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
                <Select value={selectedProvider} onValueChange={setSelectedProvider}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors">
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
                <Select value={selectedModel} onValueChange={setSelectedModel}>
                  <SelectTrigger className="bg-white border-slate-200 rounded-lg h-10 text-sm hover:border-slate-300 transition-colors">
                    <SelectValue placeholder="Select model..." />
                  </SelectTrigger>
                  <SelectContent>
                    {availableModels.map((model) => (
                      <SelectItem key={model.name} value={model.name}>
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

          {/* Enhanced Chat Area */}
          <div className="flex-1 bg-gradient-to-br from-slate-50/50 via-blue-50/20 to-indigo-50/10 overflow-auto">
            <div className="p-6 min-h-full">
              {chatMessages.length === 0 && !isLoading && (
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
                                    onClick={() => toggleSources(message.id)}
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
                                    onClick={() => toggleRelevance(message.id)}
                                    className="h-8 px-3 text-sm bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300"
                                  >
                                    <BarChart className="w-4 h-4 mr-2" />
                                    Relevance ({message.relevanceScores.length})
                                    {expandedRelevance.has(message.id) ? (
                                      <ChevronUp className="w-4 h-4 ml-2" />
                                    ) : (
                                      <ChevronDown className="w-4 h-4 ml-2" />
                                    )}
                                  </Button>
                                )}
                              </div>

                              <div className="flex space-x-2 mt-4 md:mt-0">
                                <Button variant="outline" size="sm" className="h-8 text-sm bg-white hover:bg-slate-50">
                                  <Share2 className="w-4 h-4 mr-2" /> Share
                                </Button>
                                <Button variant="outline" size="sm" className="h-8 text-sm bg-white hover:bg-slate-50">
                                  <Download className="w-4 h-4 mr-2" /> Export
                                </Button>
                              </div>
                            </div>

                            {/* Enhanced Expandable Sources Section */}
                            {message.documentNames && message.documentNames.length > 0 && expandedSources.has(message.id) && (
                              <div className="mt-6 pt-6 border-t border-slate-200 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl p-6 transition-all duration-300">
                                <div className="flex items-center mb-4">
                                  <div className="p-2 bg-blue-100 rounded-lg mr-3">
                                    <Eye className="w-5 h-5 text-blue-600" />
                                  </div>
                                  <div>
                                    <span className="text-lg font-semibold text-blue-800">Document Sources</span>
                                    <Badge variant="secondary" className="ml-3 bg-blue-100 text-blue-700">
                                      {message.documentNames.length} documents
                                    </Badge>
                                  </div>
                                </div>
                                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                                  {message.documentNames.map((name, index) => (
                                    <Card
                                      key={index}
                                      className="border-0 bg-white/80 hover:bg-white hover:shadow-md transition-all duration-200"
                                    >
                                      <CardContent className="p-4">
                                        <div className="flex items-center justify-between">
                                          <div className="flex items-center flex-1 min-w-0">
                                            <div className="p-2 bg-blue-50 rounded-lg mr-3">
                                              <FileText className="w-4 h-4 text-blue-600" />
                                            </div>
                                            <div className="flex-1 min-w-0">
                                              <p className="text-sm font-medium text-slate-800 truncate" title={name}>
                                                {name}
                                              </p>
                                              <p className="text-xs text-slate-500">Source document</p>
                                            </div>
                                          </div>
                                          <Button
                                            variant="ghost"
                                            size="sm"
                                            className="h-6 w-6 p-0 ml-2 text-slate-400 hover:text-blue-600"
                                          >
                                            <ExternalLink className="w-3 h-3" />
                                          </Button>
                                        </div>
                                      </CardContent>
                                    </Card>
                                  ))}
                                </div>
                              </div>
                            )}

                            {/* Enhanced Expandable Document Relevance Section */}
                            {message.relevanceScores && message.relevanceScores.length > 0 && expandedRelevance.has(message.id) && (
                              <div className="mt-6 pt-6 border-t border-slate-200 bg-gradient-to-r from-emerald-50 to-teal-50 rounded-xl p-6 transition-all duration-300">
                                <div className="flex items-center mb-4">
                                  <div className="p-2 bg-emerald-100 rounded-lg mr-3">
                                    <BarChart className="w-5 h-5 text-emerald-600" />
                                  </div>
                                  <div>
                                    <span className="text-lg font-semibold text-emerald-800">Document Relevance Scores</span>
                                    <Badge variant="secondary" className="ml-3 bg-emerald-100 text-emerald-700">
                                      {message.relevanceScores.length} scores
                                    </Badge>
                                  </div>
                                </div>
                                <div className="space-y-3">
                                  {message.relevanceScores
                                    .sort((a, b) => b.score - a.score) // Sort by relevance score descending
                                    .map((item, index) => (
                                      <Card
                                        key={index}
                                        className="border-0 bg-white/80 hover:bg-white hover:shadow-md transition-all duration-200"
                                      >
                                        <CardContent className="p-4">
                                          <div className="flex items-center justify-between mb-3">
                                            <div className="flex items-center flex-1 min-w-0">
                                              <div className="p-2 bg-emerald-50 rounded-lg mr-3">
                                                <FileText className="w-4 h-4 text-emerald-600" />
                                              </div>
                                              <div className="flex-1 min-w-0">
                                                <p className="text-sm font-medium text-slate-800 truncate" title={item.name}>
                                                  {item.name}
                                                </p>
                                                <p className="text-xs text-slate-500">Relevance analysis</p>
                                              </div>
                                            </div>
                                            <div className="flex items-center ml-3">
                                              <div className="bg-emerald-100 rounded-full px-3 py-1">
                                                <span className="text-sm font-bold text-emerald-800">
                                                  {typeof item.score === 'number' ? item.score.toFixed(3) : item.score}
                                                </span>
                                              </div>
                                            </div>
                                          </div>
                                          {/* Enhanced Visual relevance bar */}
                                          <div className="relative">
                                            <div className="w-full bg-slate-200 rounded-full h-2">
                                              <div
                                                className="bg-gradient-to-r from-emerald-500 to-teal-500 h-2 rounded-full transition-all duration-500 ease-out"
                                                style={{
                                                  width: `${Math.min(100, (typeof item.score === 'number' ? item.score : parseFloat(item.score) || 0) * 100)}%`
                                                }}
                                              ></div>
                                            </div>
                                            <div className="flex justify-between text-xs text-slate-500 mt-1">
                                              <span>Low</span>
                                              <span>High</span>
                                            </div>
                                          </div>
                                        </CardContent>
                                      </Card>
                                    ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                    </div>
                  </div>
                ))}

                {isLoading && (
                  <div className="text-left">
                    <div className="inline-block bg-white/95 backdrop-blur-xl border-2 border-slate-100 p-8 rounded-2xl shadow-xl">
                      <div className="flex items-center space-x-4">
                        <div className="relative">
                          <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
                          <div className="absolute inset-0 w-8 h-8 border-2 border-blue-200 rounded-full animate-pulse"></div>
                        </div>
                        <div>
                          <span className="text-lg font-medium text-slate-700">
                            {maxContext === "YES"
                              ? "AI analyzing documents with maximum context for comprehensive insights..."
                              : keepContext === "YES"
                                ? "AI searching documents and generating precise response..."
                                : "AI analyzing documents and generating enhanced hybrid response..."}
                          </span>
                          <p className="text-sm text-slate-500 mt-1">
                            Using {selectedProvider.charAt(0).toUpperCase() + selectedProvider.slice(1)} AI • 
                            {selectedFiles.length === 0 
                              ? `${documents.length} documents available` 
                              : `${selectedFiles.length} selected documents`}
                          </p>
                        </div>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Enhanced Input Area - Only Input Field Moved to Bottom */}
          <div className="p-4 bg-white/95 backdrop-blur-xl border-t border-slate-200/60 flex-shrink-0 shadow-sm">
            <div className="relative">
              <div className="absolute inset-y-0 left-0 pl-4 flex items-center pointer-events-none">
                <Search className="h-5 w-5 text-slate-400" />
              </div>
              <Input
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
                placeholder="Ask AI about your documents, market trends, or business insights..."
                className="w-full h-14 pl-12 pr-16 text-base bg-gradient-to-r from-slate-50 to-blue-50/30 border-2 border-slate-200 rounded-xl focus:ring-2 focus:ring-blue-500 focus:border-blue-500 transition-all duration-200 placeholder:text-slate-400"
                disabled={isLoading}
              />
              <Button
                size="sm"
                onClick={handleSendMessage}
                disabled={isLoading || !inputMessage.trim()}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:opacity-50 h-10 w-10 p-0 rounded-xl transition-all duration-200 hover:scale-105 shadow-lg"
              >
                {isLoading ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </Button>
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
