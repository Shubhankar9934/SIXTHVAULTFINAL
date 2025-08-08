"use client"

import React, { createContext, useContext, useReducer, useEffect, useRef, useCallback } from 'react'
import { useAuth } from './auth-context'
import { aiCurationService } from './ai-curation-service'
import { aiSummaryService } from './ai-summary-service'
import { documentStore, type DocumentData } from './document-store'
import { cacheManager, CacheKeys } from './cache-manager'
import { ragApiClient, type Conversation, type AICuration } from './api-client'

// ============================================================================
// TYPES AND INTERFACES
// ============================================================================

// Define AISummary interface since it's not exported from api-client
export interface AISummary {
  id: string
  title: string
  content: string
  created_at: string
  updated_at: string
  document_count: number
  status: 'active' | 'stale' | 'updating'
}

export interface VaultState {
  // Data State
  curations: {
    data: AICuration[]
    customCurations: Array<{ id: string; title: string; icon: any; active: boolean }>
    dynamicCurations: Array<{ id: string; title: string; icon: any; active: boolean }>
    loading: boolean
    lastUpdated: number
    activeTab: string
    activeCuration: string
    isGenerating: boolean
  }
  summaries: {
    data: AISummary[]
    dynamicSummaries: Array<{ name: string; active: boolean }>
    loading: boolean
    lastUpdated: number
    activeSummary: string
    isGenerating: boolean
  }
  conversations: {
    data: Conversation[]
    loading: boolean
    currentConversationId: string
    selectedConversationId: string
    chatMessages: Array<{
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
      conversationId?: string
    }>
    conversationTitle: string
  }
  documents: {
    data: DocumentData[]
    loading: boolean
    lastUpdated: number
    availableTags: string[]
    selectedFiles: string[]
  }
  ui: {
    activeTab: 'curations' | 'summaries' | 'history'
    sidebarOpen: boolean
    sidebarExpanded: boolean
    expandedSources: Set<number>
    expandedRelevance: Set<number>
  }
  settings: {
    selectedProvider: string
    selectedModel: string
    availableModels: any[]
    modelProviders: any[]
    keepContext: string
    maxContext: string
    searchTag: string
  }
  performance: {
    loadTimes: Record<string, number>
    cacheHits: number
    cacheMisses: number
    backgroundRefreshes: number
  }
}

export type VaultAction =
  | { type: 'SET_LOADING'; payload: { section: keyof VaultState; loading: boolean } }
  | { type: 'SET_CURATIONS'; payload: AICuration[] }
  | { type: 'SET_CUSTOM_CURATIONS'; payload: Array<{ id: string; title: string; icon: any; active: boolean }> }
  | { type: 'SET_DYNAMIC_CURATIONS'; payload: Array<{ id: string; title: string; icon: any; active: boolean }> }
  | { type: 'SET_ACTIVE_CURATION'; payload: string }
  | { type: 'SET_CURATION_GENERATING'; payload: boolean }
  | { type: 'SET_SUMMARIES'; payload: AISummary[] }
  | { type: 'SET_DYNAMIC_SUMMARIES'; payload: Array<{ name: string; active: boolean }> }
  | { type: 'SET_ACTIVE_SUMMARY'; payload: string }
  | { type: 'SET_SUMMARY_GENERATING'; payload: boolean }
  | { type: 'SET_CONVERSATIONS'; payload: Conversation[] }
  | { type: 'SET_CURRENT_CONVERSATION'; payload: string }
  | { type: 'SET_SELECTED_CONVERSATION'; payload: string }
  | { type: 'SET_CHAT_MESSAGES'; payload: VaultState['conversations']['chatMessages'] }
  | { type: 'ADD_CHAT_MESSAGE'; payload: VaultState['conversations']['chatMessages'][0] }
  | { type: 'UPDATE_CHAT_MESSAGE'; payload: { id: number; updates: Partial<VaultState['conversations']['chatMessages'][0]> } }
  | { type: 'SET_CONVERSATION_TITLE'; payload: string }
  | { type: 'SET_DOCUMENTS'; payload: DocumentData[] }
  | { type: 'SET_AVAILABLE_TAGS'; payload: string[] }
  | { type: 'SET_SELECTED_FILES'; payload: string[] }
  | { type: 'SET_ACTIVE_TAB'; payload: 'curations' | 'summaries' | 'history' }
  | { type: 'SET_SIDEBAR_OPEN'; payload: boolean }
  | { type: 'SET_SIDEBAR_EXPANDED'; payload: boolean }
  | { type: 'TOGGLE_EXPANDED_SOURCES'; payload: number }
  | { type: 'TOGGLE_EXPANDED_RELEVANCE'; payload: number }
  | { type: 'SET_PROVIDER_SETTINGS'; payload: { provider: string; model: string; availableModels: any[]; modelProviders: any[] } }
  | { type: 'SET_CONTEXT_SETTINGS'; payload: { keepContext: string; maxContext: string } }
  | { type: 'SET_SEARCH_TAG'; payload: string }
  | { type: 'UPDATE_PERFORMANCE'; payload: Partial<VaultState['performance']> }
  | { type: 'RESET_STATE' }

// ============================================================================
// REDUCER
// ============================================================================

const initialState: VaultState = {
  curations: {
    data: [],
    customCurations: [],
    dynamicCurations: [],
    loading: false,
    lastUpdated: 0,
    activeTab: 'curations',
    activeCuration: '',
    isGenerating: false
  },
  summaries: {
    data: [],
    dynamicSummaries: [],
    loading: false,
    lastUpdated: 0,
    activeSummary: '',
    isGenerating: false
  },
  conversations: {
    data: [],
    loading: false,
    currentConversationId: '',
    selectedConversationId: '',
    chatMessages: [],
    conversationTitle: ''
  },
  documents: {
    data: [],
    loading: false,
    lastUpdated: 0,
    availableTags: [],
    selectedFiles: []
  },
  ui: {
    activeTab: 'curations',
    sidebarOpen: true,
    sidebarExpanded: true,
    expandedSources: new Set(),
    expandedRelevance: new Set()
  },
  settings: {
    selectedProvider: 'gemini',
    selectedModel: 'gemini-1.5-flash',
    availableModels: [],
    modelProviders: [],
    keepContext: 'NO',
    maxContext: 'YES',
    searchTag: ''
  },
  performance: {
    loadTimes: {},
    cacheHits: 0,
    cacheMisses: 0,
    backgroundRefreshes: 0
  }
}

function vaultReducer(state: VaultState, action: VaultAction): VaultState {
  switch (action.type) {
    case 'SET_LOADING':
      return {
        ...state,
        [action.payload.section]: {
          ...state[action.payload.section],
          loading: action.payload.loading
        }
      }

    case 'SET_CURATIONS':
      return {
        ...state,
        curations: {
          ...state.curations,
          data: action.payload,
          lastUpdated: Date.now()
        }
      }

    case 'SET_CUSTOM_CURATIONS':
      return {
        ...state,
        curations: {
          ...state.curations,
          customCurations: action.payload
        }
      }

    case 'SET_DYNAMIC_CURATIONS':
      return {
        ...state,
        curations: {
          ...state.curations,
          dynamicCurations: action.payload
        }
      }

    case 'SET_ACTIVE_CURATION':
      return {
        ...state,
        curations: {
          ...state.curations,
          activeCuration: action.payload,
          customCurations: state.curations.customCurations.map(c => ({ ...c, active: c.title === action.payload })),
          dynamicCurations: state.curations.dynamicCurations.map(c => ({ ...c, active: c.title === action.payload }))
        }
      }

    case 'SET_CURATION_GENERATING':
      return {
        ...state,
        curations: {
          ...state.curations,
          isGenerating: action.payload
        }
      }

    case 'SET_SUMMARIES':
      return {
        ...state,
        summaries: {
          ...state.summaries,
          data: action.payload,
          lastUpdated: Date.now()
        }
      }

    case 'SET_DYNAMIC_SUMMARIES':
      return {
        ...state,
        summaries: {
          ...state.summaries,
          dynamicSummaries: action.payload
        }
      }

    case 'SET_ACTIVE_SUMMARY':
      return {
        ...state,
        summaries: {
          ...state.summaries,
          activeSummary: action.payload,
          dynamicSummaries: state.summaries.dynamicSummaries.map(s => ({ ...s, active: s.name === action.payload }))
        }
      }

    case 'SET_SUMMARY_GENERATING':
      return {
        ...state,
        summaries: {
          ...state.summaries,
          isGenerating: action.payload
        }
      }

    case 'SET_CONVERSATIONS':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          data: action.payload
        }
      }

    case 'SET_CURRENT_CONVERSATION':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          currentConversationId: action.payload
        }
      }

    case 'SET_SELECTED_CONVERSATION':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          selectedConversationId: action.payload
        }
      }

    case 'SET_CHAT_MESSAGES':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          chatMessages: action.payload
        }
      }

    case 'ADD_CHAT_MESSAGE':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          chatMessages: [...state.conversations.chatMessages, action.payload]
        }
      }

    case 'UPDATE_CHAT_MESSAGE':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          chatMessages: state.conversations.chatMessages.map(msg =>
            msg.id === action.payload.id ? { ...msg, ...action.payload.updates } : msg
          )
        }
      }

    case 'SET_CONVERSATION_TITLE':
      return {
        ...state,
        conversations: {
          ...state.conversations,
          conversationTitle: action.payload
        }
      }

    case 'SET_DOCUMENTS':
      return {
        ...state,
        documents: {
          ...state.documents,
          data: action.payload,
          lastUpdated: Date.now()
        }
      }

    case 'SET_AVAILABLE_TAGS':
      return {
        ...state,
        documents: {
          ...state.documents,
          availableTags: action.payload
        }
      }

    case 'SET_SELECTED_FILES':
      return {
        ...state,
        documents: {
          ...state.documents,
          selectedFiles: action.payload
        }
      }

    case 'SET_ACTIVE_TAB':
      return {
        ...state,
        ui: {
          ...state.ui,
          activeTab: action.payload
        }
      }

    case 'SET_SIDEBAR_OPEN':
      return {
        ...state,
        ui: {
          ...state.ui,
          sidebarOpen: action.payload
        }
      }

    case 'SET_SIDEBAR_EXPANDED':
      return {
        ...state,
        ui: {
          ...state.ui,
          sidebarExpanded: action.payload
        }
      }

    case 'TOGGLE_EXPANDED_SOURCES':
      const newExpandedSources = new Set(state.ui.expandedSources)
      if (newExpandedSources.has(action.payload)) {
        newExpandedSources.delete(action.payload)
      } else {
        newExpandedSources.add(action.payload)
      }
      return {
        ...state,
        ui: {
          ...state.ui,
          expandedSources: newExpandedSources
        }
      }

    case 'TOGGLE_EXPANDED_RELEVANCE':
      const newExpandedRelevance = new Set(state.ui.expandedRelevance)
      if (newExpandedRelevance.has(action.payload)) {
        newExpandedRelevance.delete(action.payload)
      } else {
        newExpandedRelevance.add(action.payload)
      }
      return {
        ...state,
        ui: {
          ...state.ui,
          expandedRelevance: newExpandedRelevance
        }
      }

    case 'SET_PROVIDER_SETTINGS':
      return {
        ...state,
        settings: {
          ...state.settings,
          selectedProvider: action.payload.provider,
          selectedModel: action.payload.model,
          availableModels: action.payload.availableModels,
          modelProviders: action.payload.modelProviders
        }
      }

    case 'SET_CONTEXT_SETTINGS':
      return {
        ...state,
        settings: {
          ...state.settings,
          keepContext: action.payload.keepContext,
          maxContext: action.payload.maxContext
        }
      }

    case 'SET_SEARCH_TAG':
      return {
        ...state,
        settings: {
          ...state.settings,
          searchTag: action.payload
        }
      }

    case 'UPDATE_PERFORMANCE':
      return {
        ...state,
        performance: {
          ...state.performance,
          ...action.payload
        }
      }

    case 'RESET_STATE':
      return initialState

    default:
      return state
  }
}

// ============================================================================
// CONTEXT AND PROVIDER
// ============================================================================

interface VaultContextType {
  state: VaultState
  dispatch: React.Dispatch<VaultAction>
  
  // High-level actions
  initializeVault: () => Promise<void>
  refreshAllData: () => Promise<void>
  
  // Curation actions
  loadCurations: () => Promise<void>
  generateCurationContent: (curationId: string, title: string) => Promise<void>
  createCustomCuration: (title: string, description: string, keywords: string[]) => Promise<{ success: boolean; message: string; curation?: AICuration }>
  deleteCuration: (curationId: string) => Promise<void>
  
  // Summary actions
  loadSummaries: () => Promise<void>
  generateSummaryContent: (summaryName: string) => Promise<void>
  createCustomSummary: (title: string, description: string, keywords: string[]) => Promise<void>
  deleteSummary: (summaryId: string) => Promise<void>
  
  // Conversation actions
  loadConversations: () => Promise<void>
  sendMessage: (message: string) => Promise<void>
  loadConversationHistory: (conversationId: string) => Promise<void>
  startNewConversation: () => void
  deleteConversation: (conversationId: string) => Promise<void>
  
  // Document actions
  loadDocuments: () => Promise<void>
  
  // UI actions
  toggleSidebar: () => void
  toggleSidebarExpansion: () => void
  setActiveTab: (tab: 'curations' | 'summaries' | 'history') => void
  toggleExpandedSources: (messageId: number) => void
  toggleExpandedRelevance: (messageId: number) => void
  
  // Settings actions
  updateProviderSettings: (provider: string, model: string) => void
  updateContextSettings: (keepContext: string, maxContext: string) => void
  setSearchTag: (tag: string) => void
  setSelectedFiles: (files: string[]) => void
}

const VaultContext = createContext<VaultContextType | undefined>(undefined)

// ============================================================================
// PROVIDER COMPONENT
// ============================================================================

export function VaultStateProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(vaultReducer, initialState)
  const { isAuthenticated, isLoading: authLoading, user } = useAuth()
  const initializationRef = useRef(false)
  const backgroundRefreshRef = useRef<NodeJS.Timeout | null>(null)
  const lastAuthStateRef = useRef<boolean>(false)

  // ============================================================================
  // INITIALIZATION AND BACKGROUND REFRESH
  // ============================================================================

  const initializeVault = useCallback(async () => {
    // Don't initialize if auth is still loading
    if (authLoading) {
      console.log('🔄 VaultState: Waiting for authentication to complete...')
      return
    }

    // Don't initialize if user is not authenticated
    if (!isAuthenticated) {
      console.log('🚫 VaultState: User not authenticated, skipping vault initialization')
      // Reset state to initial state for unauthenticated users
      dispatch({ type: 'RESET_STATE' })
      initializationRef.current = false
      return
    }

    // Prevent multiple initializations for the same auth state
    if (initializationRef.current) return
    initializationRef.current = true

    console.log('🚀 VaultState: Starting comprehensive initialization with persistent cache for authenticated user')
    const startTime = Date.now()

    try {
      // Phase 1: Load cached data instantly for immediate UI response
      console.log('⚡ VaultState: Phase 1 - Loading cached data for instant UI')
      
      const cachedDocuments = cacheManager.get<DocumentData[]>(CacheKeys.documents())
      const cachedCurations = cacheManager.get<AICuration[]>(CacheKeys.curations())
      const cachedConversations = cacheManager.get<Conversation[]>('conversations_list')

      if (cachedDocuments) {
        dispatch({ type: 'SET_DOCUMENTS', payload: cachedDocuments })
        generateDynamicCurationsAndSummaries(cachedDocuments)
        console.log('✅ VaultState: Loaded cached documents instantly:', cachedDocuments.length)
      }

      if (cachedCurations) {
        dispatch({ type: 'SET_CURATIONS', payload: cachedCurations })
        const customCurations = cachedCurations.map(curation => ({
          id: curation.id,
          title: curation.title,
          icon: getIconForCuration(curation.keywords || curation.topicKeywords || []),
          active: false
        }))
        dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: customCurations })
        console.log('✅ VaultState: Loaded cached curations instantly:', cachedCurations.length)
      }

      if (cachedConversations) {
        dispatch({ type: 'SET_CONVERSATIONS', payload: cachedConversations })
        console.log('✅ VaultState: Loaded cached conversations instantly:', cachedConversations.length)
      } else {
        // If no cached conversations, load them immediately for instant history access
        console.log('🔄 VaultState: No cached conversations, loading immediately for instant access')
        try {
          const conversations = await ragApiClient.getConversations({ limit: 50 })
          dispatch({ type: 'SET_CONVERSATIONS', payload: conversations })
          cacheManager.set('conversations_list', conversations, 15 * 60 * 1000)
          console.log('✅ VaultState: Loaded conversations immediately:', conversations.length)
        } catch (error) {
          console.error('❌ VaultState: Failed to load conversations immediately:', error)
        }
      }

  // Phase 1.5: Deep preloading for instant access - AI Providers, Curation Content, and Chat History
  console.log('⚡ VaultState: Phase 1.5 - Deep preloading for instant access')
  
  // 1. Preload ALL AI providers and their models for instant access
  console.log('🤖 VaultState: Preloading ALL AI providers and models for instant access')
  const cachedAllProviders = localStorage.getItem('vault_all_providers_cache')
  if (cachedAllProviders) {
    try {
      const { providers, timestamp } = JSON.parse(cachedAllProviders)
      const isExpired = Date.now() - timestamp > 6 * 60 * 60 * 1000 // 6 hours cache
      
      if (!isExpired && providers && Array.isArray(providers)) {
        // Set all providers immediately for instant UI access
        dispatch({ 
          type: 'SET_PROVIDER_SETTINGS', 
          payload: { 
            provider: state.settings.selectedProvider,
            model: state.settings.selectedModel,
            availableModels: providers.find((p: any) => p.name === state.settings.selectedProvider)?.models || [],
            modelProviders: providers
          }
        })
        console.log(`✅ VaultState: Preloaded ${providers.length} AI providers with all models instantly`)
        
        // Log provider details for verification
        providers.forEach((provider: any) => {
          console.log(`   - ${provider.displayName}: ${provider.models?.length || 0} models available`)
        })
      } else {
        localStorage.removeItem('vault_all_providers_cache')
        console.log('🔄 VaultState: AI providers cache expired, will refresh in background')
      }
    } catch (error) {
      console.error('Error parsing cached AI providers:', error)
      localStorage.removeItem('vault_all_providers_cache')
    }
  } else {
    console.log('🔄 VaultState: No cached AI providers found, will load in background')
  }

  // 2. Load cached AI provider settings for instant restoration
  console.log('⚙️ VaultState: Loading cached AI provider settings for instant restoration')
  const cachedProviderSettings = localStorage.getItem('vault_provider_settings')
  if (cachedProviderSettings) {
    try {
      const { provider, model, timestamp } = JSON.parse(cachedProviderSettings)
      const isExpired = Date.now() - timestamp > 24 * 60 * 60 * 1000 // 24 hours
      
      if (!isExpired && provider && model) {
        // Get available models for this provider from cached data
        const cachedProviders = localStorage.getItem('vault_all_providers_cache')
        let availableModels: any[] = []
        let modelProviders: any[] = []
        
        if (cachedProviders) {
          try {
            const { providers } = JSON.parse(cachedProviders)
            modelProviders = providers || []
            availableModels = providers.find((p: any) => p.name === provider)?.models || []
          } catch (error) {
            console.error('Error parsing cached providers for settings:', error)
          }
        }
        
        // Set provider settings immediately for instant UI update
        dispatch({ 
          type: 'SET_PROVIDER_SETTINGS', 
          payload: { 
            provider, 
            model, 
            availableModels,
            modelProviders
          }
        })
        console.log(`✅ VaultState: Restored cached AI provider settings instantly: ${provider} - ${model}`)
      } else {
        localStorage.removeItem('vault_provider_settings')
      }
    } catch (error) {
      console.error('Error parsing cached provider settings:', error)
      localStorage.removeItem('vault_provider_settings')
    }
  }

  // 3. Load cached curation content for all curations for instant access
  console.log('📚 VaultState: Loading cached AI curation content for instant access')
  if (cachedCurations && cachedCurations.length > 0) {
    const cachedCurationContent: any[] = []
    
    cachedCurations.forEach(curation => {
      const contentCacheKey = `curation_content_${curation.id}`
      const cachedContent = cacheManager.get<string>(contentCacheKey)
      
      if (cachedContent) {
        cachedCurationContent.push({
          id: curation.id,
          title: curation.title,
          content: cachedContent,
          type: 'curation',
          cached: true
        })
      }
    })
    
    if (cachedCurationContent.length > 0) {
      console.log(`✅ VaultState: Loaded ${cachedCurationContent.length} cached AI curation contents for instant access`)
      
      // Store cached curation content for instant display
      const cachedMessages = cachedCurationContent.map((item, index) => ({
        id: Date.now() + index,
        type: "curation" as const,
        content: item.content,
        title: item.title,
        themes: ["AI Curation", "Cached Content", "Instant Access"],
        documentsUsed: cachedDocuments?.length || 0,
        documentNames: cachedDocuments?.map(doc => doc.name) || [],
        timestamp: new Date().toISOString(),
        cached: true
      }))
      
      sessionStorage.setItem('cached_curation_messages', JSON.stringify(cachedMessages))
    }
  }

  // 4. DEEP PRELOAD: Load cached conversation content for instant chat history access
  console.log('💬 VaultState: Deep preloading conversation content for instant chat history access')
  if (cachedConversations && cachedConversations.length > 0) {
    let preloadedConversationCount = 0
    const conversationContentCache: Record<string, any> = {}
    
    // Preload content for up to 20 most recent conversations for instant access
    const conversationsToPreload = cachedConversations
      .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
      .slice(0, 20)
    
    conversationsToPreload.forEach(conversation => {
      const conversationCacheKey = `conversation_content_${conversation.id}`
      const cachedConversationContent = cacheManager.get<any>(conversationCacheKey)
      
      if (cachedConversationContent) {
        conversationContentCache[conversation.id] = cachedConversationContent
        preloadedConversationCount++
      }
    })
    
    if (preloadedConversationCount > 0) {
      console.log(`✅ VaultState: Deep preloaded ${preloadedConversationCount} conversation contents for instant access`)
      
      // Store preloaded conversation content for instant access
      sessionStorage.setItem('preloaded_conversation_content', JSON.stringify(conversationContentCache))
      
      // Log details for verification
      Object.keys(conversationContentCache).forEach(conversationId => {
        const content = conversationContentCache[conversationId]
        console.log(`   - Conversation ${conversationId}: ${content.messages?.length || 0} messages preloaded`)
      })
    } else {
      console.log('🔄 VaultState: No cached conversation content found, starting immediate preload')
      
      // If no cached content exists, start immediate preloading for instant access
      immediatePreloadConversationContent(conversationsToPreload)
    }
  }

      // Phase 2: Initialize services with their own caching
      console.log('🔄 VaultState: Phase 2 - Initializing services with persistent cache')
      
      await Promise.allSettled([
        aiCurationService.initialize(),
        aiSummaryService.initialize()
      ])

      // Phase 3: Load fresh data in background if cache is stale
      console.log('🔄 VaultState: Phase 3 - Background refresh for stale data')
      
      const cacheAge = cachedDocuments ? Date.now() - (cacheManager.get<{ timestamp: number }>(CacheKeys.documents() + '_meta')?.timestamp || 0) : Infinity
      const maxCacheAge = 5 * 60 * 1000 // 5 minutes

      if (cacheAge > maxCacheAge) {
        console.log('🔄 VaultState: Cache is stale, refreshing in background')
        refreshAllDataInBackground()
      } else {
        console.log('✅ VaultState: Cache is fresh, skipping background refresh')
      }

      // Phase 4: Load and cache ALL AI providers with comprehensive model preloading
      await loadAndCacheAllAIProviders()

      // Phase 5: Background preloading of conversation content for instant access
      if (cachedConversations && cachedConversations.length > 0) {
        console.log('🚀 VaultState: Phase 5 - Background preloading conversation content for instant access')
        backgroundPreloadConversationContent(cachedConversations)
      }

      const initTime = Date.now() - startTime
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { loadTimes: { initialization: initTime } } })
      
      console.log(`🎉 VaultState: Initialization completed in ${initTime}ms with persistent cache`)

    } catch (error) {
      console.error('💥 VaultState: Initialization failed:', error)
      // Continue with empty state rather than blocking
    }
  }, [authLoading, isAuthenticated])

  const refreshAllDataInBackground = useCallback(async () => {
    console.log('🔄 VaultState: Starting background refresh')
    
    try {
      const startTime = Date.now()
      
      // Load fresh data in parallel
      const [documentsResult, curationsResult, conversationsResult] = await Promise.allSettled([
        documentStore.getDocuments(),
        ragApiClient.getAICurations(),
        ragApiClient.getConversations({ limit: 50 })
      ])

      // Update documents
      if (documentsResult.status === 'fulfilled') {
        dispatch({ type: 'SET_DOCUMENTS', payload: documentsResult.value })
        cacheManager.set(CacheKeys.documents(), documentsResult.value, 30 * 60 * 1000)
        cacheManager.set(CacheKeys.documents() + '_meta', { timestamp: Date.now() }, 30 * 60 * 1000)
        generateDynamicCurationsAndSummaries(documentsResult.value)
        console.log('✅ VaultState: Background refresh - documents updated')
      }

      // Update curations
      if (curationsResult.status === 'fulfilled') {
        dispatch({ type: 'SET_CURATIONS', payload: curationsResult.value })
        cacheManager.set(CacheKeys.curations(), curationsResult.value, 30 * 60 * 1000)
        
        const customCurations = curationsResult.value.map(curation => ({
          id: curation.id,
          title: curation.title,
          icon: getIconForCuration(curation.keywords || curation.topicKeywords || []),
          active: false
        }))
        dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: customCurations })
        console.log('✅ VaultState: Background refresh - curations updated')
      }

      // Update conversations
      if (conversationsResult.status === 'fulfilled') {
        dispatch({ type: 'SET_CONVERSATIONS', payload: conversationsResult.value })
        cacheManager.set('conversations_list', conversationsResult.value, 15 * 60 * 1000)
        console.log('✅ VaultState: Background refresh - conversations updated')
      }

      const refreshTime = Date.now() - startTime
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { 
        backgroundRefreshes: state.performance.backgroundRefreshes + 1,
        loadTimes: { ...state.performance.loadTimes, backgroundRefresh: refreshTime }
      }})

      console.log(`✅ VaultState: Background refresh completed in ${refreshTime}ms`)

    } catch (error) {
      console.error('❌ VaultState: Background refresh failed:', error)
    }
  }, [state.performance])

  const refreshAllData = useCallback(async () => {
    console.log('🔄 VaultState: Manual refresh triggered')
    
    dispatch({ type: 'SET_LOADING', payload: { section: 'documents', loading: true } })
    dispatch({ type: 'SET_LOADING', payload: { section: 'curations', loading: true } })
    dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: true } })

    await refreshAllDataInBackground()

    dispatch({ type: 'SET_LOADING', payload: { section: 'documents', loading: false } })
    dispatch({ type: 'SET_LOADING', payload: { section: 'curations', loading: false } })
    dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: false } })
  }, [refreshAllDataInBackground])

  // ============================================================================
  // HELPER FUNCTIONS
  // ============================================================================

  const generateDynamicCurationsAndSummaries = useCallback(async (docs: DocumentData[]) => {
    // Generate dynamic curations
    if (docs.length === 0) {
      dispatch({ type: 'SET_DYNAMIC_CURATIONS', payload: [] })
      dispatch({ type: 'SET_DYNAMIC_SUMMARIES', payload: [] })
      
      // Clear orphaned custom curations when no documents exist
      console.log('🧹 VaultState: No documents available, clearing orphaned curations')
      dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: [] })
      
      // Clear all curations from AI service when no documents exist
      await aiCurationService.handleAllDocumentsDeleted()
      
      return
    }

    // Extract themes and topics
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

    // Generate dynamic curations based purely on document content - no artificial limits
    const dynamicCurationsList: Array<{ id: string; title: string; icon: any; active: boolean }> = []

    // Generate curations for each theme found in documents
    themes.forEach((theme, index) => {
      dynamicCurationsList.push({
        id: `trends-${theme.toLowerCase().replace(/\s+/g, '-')}`,
        title: `Top ${Math.min(5, themes.length)} ${theme} Trends`,
        icon: require('lucide-react').TrendingUp,
        active: false
      })
    })

    // Generate curations for each topic found in documents
    topics.forEach((topic, index) => {
      dynamicCurationsList.push({
        id: `industry-${topic.toLowerCase().replace(/\s+/g, '-')}`,
        title: `${topic} Industry Analysis`,
        icon: require('lucide-react').BarChart3,
        active: false
      })
    })

    // Generate curations for each demographic found in documents
    demographics.forEach((demographic, index) => {
      dynamicCurationsList.push({
        id: `market-${demographic.toLowerCase().replace(/\s+/g, '-')}`,
        title: `Trends for ${demographic} Market`,
        icon: require('lucide-react').Users,
        active: false
      })
    })

    // Add tech-specific curations if tech themes are found
    const techThemes = themes.filter(theme => 
      theme.toLowerCase().includes("tech") ||
      theme.toLowerCase().includes("digital") ||
      theme.toLowerCase().includes("ai") ||
      theme.toLowerCase().includes("innovation")
    )
    
    if (techThemes.length > 0) {
      dynamicCurationsList.push({
        id: 'digital-innovation',
        title: `Digital Innovation Insights`,
        icon: require('lucide-react').Brain,
        active: false
      })
    }

    // Add general document insights curation
    dynamicCurationsList.push({
      id: 'doc-insights',
      title: "Document Insights Summary",
      icon: require('lucide-react').FileText,
      active: false
    })

    // Remove any duplicates by ID
    const uniqueCurations = dynamicCurationsList.filter((curation, index, self) => 
      index === self.findIndex(c => c.id === curation.id)
    )

    console.log(`🎯 VaultState: Generated ${uniqueCurations.length} dynamic curations based on document content`)
    console.log(`   - Themes: ${themes.length}, Topics: ${topics.length}, Demographics: ${demographics.length}`)

    dispatch({ type: 'SET_DYNAMIC_CURATIONS', payload: uniqueCurations })

    // Generate dynamic summaries
    const summaries = []

    if (docs.length === 1) {
      summaries.push({ name: "Overall Summary", active: false })
    } else {
      // Create individual summaries for each document (no artificial limit)
      docs.forEach((doc) => {
        const docName = doc.name.split(".")[0]
        const shortName = docName.length > 12 ? docName.substring(0, 12) + "..." : docName
        summaries.push({ name: shortName, active: false })
      })

      // Add combined summary at the end
      summaries.push({ name: "Combined Summary", active: false })
    }

    // Set all summaries without limiting to 4
    dispatch({ type: 'SET_DYNAMIC_SUMMARIES', payload: summaries })

    // Update available tags
    const allTags = new Set<string>()
    docs.forEach((doc) => {
      doc.themes?.forEach((theme: string) => allTags.add(theme))
      doc.keywords?.forEach((keyword: string) => allTags.add(keyword))
    })
    dispatch({ type: 'SET_AVAILABLE_TAGS', payload: Array.from(allTags) })
  }, [])

  const getIconForCuration = useCallback((keywords: string[]) => {
    const keywordsStr = (keywords || []).join(' ').toLowerCase()
    
    if (keywordsStr.includes('trend') || keywordsStr.includes('growth')) {
      return require('lucide-react').TrendingUp
    } else if (keywordsStr.includes('market') || keywordsStr.includes('business')) {
      return require('lucide-react').BarChart3
    } else if (keywordsStr.includes('user') || keywordsStr.includes('customer') || keywordsStr.includes('demographic')) {
      return require('lucide-react').Users
    } else if (keywordsStr.includes('tech') || keywordsStr.includes('digital') || keywordsStr.includes('innovation')) {
      return require('lucide-react').Brain
    } else if (keywordsStr.includes('industry') || keywordsStr.includes('sector')) {
      return require('lucide-react').Building2
    } else if (keywordsStr.includes('global') || keywordsStr.includes('world') || keywordsStr.includes('international')) {
      return require('lucide-react').Globe
    } else {
      return require('lucide-react').Sparkles
    }
  }, [])

  // Helper function to separate curations by type
  const separateCurationsByType = useCallback((curations: AICuration[]) => {
    const custom = curations.filter(c => !c.autoGenerated).map(curation => ({
      id: curation.id,
      title: curation.title,
      icon: getIconForCuration(curation.keywords || curation.topicKeywords || []),
      active: false,
      status: curation.status
    }))
    
    const dynamic = curations.filter(c => c.autoGenerated).map(curation => ({
      id: curation.id,
      title: curation.title,
      icon: getIconForCuration(curation.keywords || curation.topicKeywords || []),
      active: false,
      status: curation.status
    }))
    
    return { custom, dynamic }
  }, [getIconForCuration])

  const loadAndCacheAllAIProviders = useCallback(async () => {
    console.log('🤖 VaultState: Loading and caching ALL AI providers with comprehensive model preloading')
    
    try {
      // Check if we already have fresh cached providers
      const cachedAllProviders = localStorage.getItem('vault_all_providers_cache')
      if (cachedAllProviders) {
        try {
          const { providers, timestamp } = JSON.parse(cachedAllProviders)
          const isExpired = Date.now() - timestamp > 6 * 60 * 60 * 1000 // 6 hours cache
          
          if (!isExpired && providers && Array.isArray(providers)) {
            console.log(`✅ VaultState: Using fresh cached AI providers (${providers.length} providers)`)
            
            // Update state with cached providers
            const currentProvider = state.settings.selectedProvider
            const availableModels = providers.find((p: any) => p.name === currentProvider)?.models || []
            
            dispatch({ 
              type: 'SET_PROVIDER_SETTINGS', 
              payload: { 
                provider: currentProvider,
                model: state.settings.selectedModel,
                availableModels,
                modelProviders: providers
              }
            })
            return
          }
        } catch (error) {
          console.error('Error parsing cached AI providers:', error)
          localStorage.removeItem('vault_all_providers_cache')
        }
      }

      // Load fresh providers from API
      console.log('🔄 VaultState: Loading fresh AI providers from API')
      const providers = await ragApiClient.getAvailableModels()
      
      if (providers && providers.length > 0) {
        // Cache ALL providers with comprehensive model data for 6 hours
        const providerCacheData = {
          providers,
          timestamp: Date.now()
        }
        localStorage.setItem('vault_all_providers_cache', JSON.stringify(providerCacheData))
        console.log(`✅ VaultState: Cached ${providers.length} AI providers with all models for instant access`)
        
        // Also update session storage for backward compatibility
        sessionStorage.setItem('vault-providers', JSON.stringify(providers))
        
        // Update state with fresh providers
        const currentProvider = state.settings.selectedProvider
        const availableModels = providers.find((p: any) => p.name === currentProvider)?.models || []
        
        // Auto-select default model if none is selected
        let selectedModel = state.settings.selectedModel
        if (!selectedModel && availableModels.length > 0) {
          const defaultModels: Record<string, string> = {
            'gemini': 'gemini-1.5-flash',
            'openai': 'gpt-4o-mini',
            'groq': 'llama3-8b-8192',
            'deepseek': 'deepseek-chat',
            'bedrock': 'anthropic.claude-3-haiku-20240307-v1:0',
            'anthropic': 'claude-3-haiku-20240307'
            
          }
          
          const defaultModel = defaultModels[currentProvider]
          if (defaultModel && availableModels.some((m: any) => m.name === defaultModel)) {
            selectedModel = defaultModel
          } else if (availableModels.length > 0) {
            selectedModel = availableModels[0].name
          }
          
          console.log(`🎯 VaultState: Auto-selected default model for ${currentProvider}: ${selectedModel}`)
        }
        
        dispatch({ 
          type: 'SET_PROVIDER_SETTINGS', 
          payload: { 
            provider: currentProvider,
            model: selectedModel,
            availableModels,
            modelProviders: providers
          }
        })
        
        // Log comprehensive provider details for verification
        console.log('🎉 VaultState: Comprehensive AI provider preloading completed:')
        providers.forEach((provider: any) => {
          console.log(`   - ${provider.displayName || provider.name}: ${provider.models?.length || 0} models available`)
          if (provider.models && provider.models.length > 0) {
            provider.models.slice(0, 3).forEach((model: any) => {
              console.log(`     • ${model.displayName || model.name}`)
            })
            if (provider.models.length > 3) {
              console.log(`     • ... and ${provider.models.length - 3} more models`)
            }
          }
        })
        
      } else {
        console.warn('⚠️ VaultState: No AI providers received from API')
      }
      
    } catch (error) {
      console.error('❌ VaultState: Failed to load and cache AI providers:', error)
      
      // Fallback to existing cached data if available
      const cachedAllProviders = localStorage.getItem('vault_all_providers_cache')
      if (cachedAllProviders) {
        try {
          const { providers } = JSON.parse(cachedAllProviders)
          if (providers && Array.isArray(providers)) {
            console.log('🔄 VaultState: Using stale cached providers as fallback')
            
            const currentProvider = state.settings.selectedProvider
            const availableModels = providers.find((p: any) => p.name === currentProvider)?.models || []
            
            dispatch({ 
              type: 'SET_PROVIDER_SETTINGS', 
              payload: { 
                provider: currentProvider,
                model: state.settings.selectedModel,
                availableModels,
                modelProviders: providers
              }
            })
          }
        } catch (fallbackError) {
          console.error('❌ VaultState: Failed to use cached providers as fallback:', fallbackError)
        }
      }
    }
  }, [state.settings.selectedProvider, state.settings.selectedModel])

  const loadAIProviders = useCallback(async () => {
    // This function is kept for backward compatibility but now calls the comprehensive version
    await loadAndCacheAllAIProviders()
  }, [loadAndCacheAllAIProviders])

  const immediatePreloadConversationContent = useCallback(async (conversations: Conversation[]) => {
    console.log('⚡ VaultState: Starting immediate preloading of conversation content for instant access')
    
    try {
      // Sort conversations by most recent and take top 10 for immediate preloading
      const conversationsToPreload = conversations
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
        .slice(0, 10)
      
      let preloadedCount = 0
      const conversationContentCache: Record<string, any> = {}
      
      // Load first 5 conversations immediately for instant access
      const immediatePromises = conversationsToPreload.slice(0, 5).map(async (conversation) => {
        try {
          const conversationCacheKey = `conversation_content_${conversation.id}`
          
          // Check if already cached
          const existingCache = cacheManager.get<any>(conversationCacheKey)
          if (existingCache) {
            conversationContentCache[conversation.id] = existingCache
            preloadedCount++
            console.log(`⚡ VaultState: Conversation ${conversation.id} already cached for immediate access`)
            return
          }
          
          // Load conversation content from API immediately
          const conversationData = await ragApiClient.getConversation(conversation.id)
          
          if (conversationData) {
            // Cache for 24 hours for instant future access
            cacheManager.set(conversationCacheKey, conversationData, 24 * 60 * 60 * 1000)
            conversationContentCache[conversation.id] = conversationData
            preloadedCount++
            console.log(`✅ VaultState: Immediately preloaded conversation ${conversation.id} (${conversation.title})`)
          }
        } catch (error) {
          console.error(`❌ VaultState: Failed to immediately preload conversation ${conversation.id}:`, error)
        }
      })
      
      // Execute immediate preloading
      await Promise.allSettled(immediatePromises)
      
      if (preloadedCount > 0) {
        console.log(`🎉 VaultState: Immediate preloading completed - ${preloadedCount} conversations preloaded for instant access`)
        
        // Store preloaded conversation content for instant access
        sessionStorage.setItem('preloaded_conversation_content', JSON.stringify(conversationContentCache))
        
        // Log details for verification
        Object.keys(conversationContentCache).forEach(conversationId => {
          const content = conversationContentCache[conversationId]
          console.log(`   - Conversation ${conversationId}: ${content.messages?.length || 0} messages immediately available`)
        })
      }
      
      // Continue with remaining conversations in background
      if (conversationsToPreload.length > 5) {
        console.log('🔄 VaultState: Continuing with background preloading for remaining conversations')
        setTimeout(() => {
          backgroundPreloadConversationContent(conversationsToPreload.slice(5))
        }, 1000) // Start background preloading after 1 second
      }
      
    } catch (error) {
      console.error('❌ VaultState: Immediate conversation preloading failed:', error)
    }
  }, [])

  const backgroundPreloadConversationContent = useCallback(async (conversations: Conversation[]) => {
    console.log('🚀 VaultState: Starting background preloading of conversation content for instant access')
    
    try {
      // Sort conversations by most recent and take top 15 for background preloading
      const conversationsToPreload = conversations
        .sort((a, b) => new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime())
        .slice(0, 15)
      
      let preloadedCount = 0
      const preloadPromises = conversationsToPreload.map(async (conversation) => {
        try {
          const conversationCacheKey = `conversation_content_${conversation.id}`
          
          // Check if already cached
          const existingCache = cacheManager.get<any>(conversationCacheKey)
          if (existingCache) {
            console.log(`⚡ VaultState: Conversation ${conversation.id} already cached, skipping`)
            return
          }
          
          // Load conversation content from API
          const conversationData = await ragApiClient.getConversation(conversation.id)
          
          if (conversationData) {
            // Cache for 24 hours for instant future access
            cacheManager.set(conversationCacheKey, conversationData, 24 * 60 * 60 * 1000)
            preloadedCount++
            console.log(`✅ VaultState: Background preloaded conversation ${conversation.id} (${conversation.title})`)
          }
        } catch (error) {
          console.error(`❌ VaultState: Failed to preload conversation ${conversation.id}:`, error)
        }
      })
      
      // Execute preloading in batches of 3 to avoid overwhelming the API
      const batchSize = 3
      for (let i = 0; i < preloadPromises.length; i += batchSize) {
        const batch = preloadPromises.slice(i, i + batchSize)
        await Promise.allSettled(batch)
        
        // Small delay between batches to be respectful to the API
        if (i + batchSize < preloadPromises.length) {
          await new Promise(resolve => setTimeout(resolve, 500))
        }
      }
      
      console.log(`🎉 VaultState: Background preloading completed - ${preloadedCount} conversations preloaded for instant access`)
      
      // Update session storage with newly preloaded content
      const allPreloadedContent: Record<string, any> = {}
      conversationsToPreload.forEach(conversation => {
        const conversationCacheKey = `conversation_content_${conversation.id}`
        const cachedData = cacheManager.get<any>(conversationCacheKey)
        if (cachedData) {
          allPreloadedContent[conversation.id] = cachedData
        }
      })
      
      if (Object.keys(allPreloadedContent).length > 0) {
        sessionStorage.setItem('preloaded_conversation_content', JSON.stringify(allPreloadedContent))
        console.log(`✅ VaultState: Updated session storage with ${Object.keys(allPreloadedContent).length} preloaded conversations`)
      }
      
    } catch (error) {
      console.error('❌ VaultState: Background conversation preloading failed:', error)
    }
  }, [])

  // ============================================================================
  // REAL-TIME DOCUMENT UPDATES
  // ============================================================================

  // Subscribe to real-time document updates from document store
  useEffect(() => {
    if (!isAuthenticated) return

    console.log('🔄 VaultState: Subscribing to real-time document updates')
    
    const unsubscribe = documentStore.subscribeToUpdates((updatedDocuments) => {
      console.log('✅ VaultState: Received real-time document update:', updatedDocuments.length, 'documents')
      
      // Update documents in state
      dispatch({ type: 'SET_DOCUMENTS', payload: updatedDocuments })
      
      // Regenerate dynamic curations and summaries based on new documents
      generateDynamicCurationsAndSummaries(updatedDocuments)
      
      // Update cache with fresh data
      cacheManager.set(CacheKeys.documents(), updatedDocuments, 30 * 60 * 1000)
      cacheManager.set(CacheKeys.documents() + '_meta', { timestamp: Date.now() }, 30 * 60 * 1000)
      
      console.log('🎉 VaultState: Document state updated in real-time - dropdown will show latest documents')
    })

    // Cleanup subscription on unmount or auth change
    return () => {
      console.log('🧹 VaultState: Unsubscribing from document updates')
      unsubscribe()
    }
  }, [isAuthenticated, generateDynamicCurationsAndSummaries])

  // ============================================================================
  // DATA LOADING ACTIONS
  // ============================================================================

  const loadCurations = useCallback(async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { section: 'curations', loading: true } })
      
      const cached = cacheManager.get<AICuration[]>(CacheKeys.curations())
      if (cached) {
        dispatch({ type: 'SET_CURATIONS', payload: cached })
        
        // Only update custom curations from API, preserve document-based dynamic curations
        const { custom } = separateCurationsByType(cached)
        dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: custom })
        
        dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheHits: state.performance.cacheHits + 1 } })
      }

      const curations = await ragApiClient.getAICurations()
      dispatch({ type: 'SET_CURATIONS', payload: curations })
      cacheManager.set(CacheKeys.curations(), curations, 30 * 60 * 1000)

      // Only update custom curations from API, preserve document-based dynamic curations
      const { custom } = separateCurationsByType(curations)
      dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: custom })
      
      // Don't override dynamic curations here - they are managed by generateDynamicCurationsAndSummaries
      console.log(`✅ VaultState: Loaded ${custom.length} custom curations, preserved document-based dynamic curations`)

    } catch (error) {
      console.error('Failed to load curations:', error)
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheMisses: state.performance.cacheMisses + 1 } })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { section: 'curations', loading: false } })
    }
  }, [state.performance, separateCurationsByType])

  const loadSummaries = useCallback(async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { section: 'summaries', loading: true } })
      
      const cached = cacheManager.get<any[]>(CacheKeys.summaries())
      if (cached) {
        dispatch({ type: 'SET_SUMMARIES', payload: cached })
        dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheHits: state.performance.cacheHits + 1 } })
      }

      // For now, use empty array since getAISummaries doesn't exist in API client
      const summaries: AISummary[] = []
      dispatch({ type: 'SET_SUMMARIES', payload: summaries })
      cacheManager.set(CacheKeys.summaries(), summaries, 30 * 60 * 1000)

    } catch (error) {
      console.error('Failed to load summaries:', error)
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheMisses: state.performance.cacheMisses + 1 } })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { section: 'summaries', loading: false } })
    }
  }, [state.performance])

  const loadConversations = useCallback(async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: true } })
      
      const cached = cacheManager.get<Conversation[]>('conversations_list')
      if (cached) {
        dispatch({ type: 'SET_CONVERSATIONS', payload: cached })
        dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheHits: state.performance.cacheHits + 1 } })
      }

      const conversations = await ragApiClient.getConversations({ limit: 50 })
      dispatch({ type: 'SET_CONVERSATIONS', payload: conversations })
      cacheManager.set('conversations_list', conversations, 15 * 60 * 1000)

    } catch (error) {
      console.error('Failed to load conversations:', error)
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheMisses: state.performance.cacheMisses + 1 } })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: false } })
    }
  }, [state.performance])

  const loadDocuments = useCallback(async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { section: 'documents', loading: true } })
      
      const cached = cacheManager.get<DocumentData[]>(CacheKeys.documents())
      if (cached) {
        dispatch({ type: 'SET_DOCUMENTS', payload: cached })
        generateDynamicCurationsAndSummaries(cached)
        dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheHits: state.performance.cacheHits + 1 } })
      }

      const documents = await documentStore.getDocuments()
      dispatch({ type: 'SET_DOCUMENTS', payload: documents })
      generateDynamicCurationsAndSummaries(documents)
      cacheManager.set(CacheKeys.documents(), documents, 30 * 60 * 1000)

    } catch (error) {
      console.error('Failed to load documents:', error)
      dispatch({ type: 'UPDATE_PERFORMANCE', payload: { cacheMisses: state.performance.cacheMisses + 1 } })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { section: 'documents', loading: false } })
    }
  }, [state.performance, generateDynamicCurationsAndSummaries])

  // ============================================================================
  // CONTENT GENERATION ACTIONS
  // ============================================================================

  const generateCurationContent = useCallback(async (curationIdOrTitle: string, title?: string) => {
    try {
      // Determine if this is an existing curation ID or a new title
      const isExistingCuration = state.curations.data.some(c => c.id === curationIdOrTitle)
      const displayTitle = title || curationIdOrTitle
      
      console.log('VaultState: Processing curation content request for:', {
        input: curationIdOrTitle,
        isExistingCuration,
        displayTitle,
        provider: state.settings.selectedProvider,
        model: state.settings.selectedModel
      })

      // FIXED: For existing curations, check cache first and display instantly if available
      if (isExistingCuration) {
        const curation = state.curations.data.find(c => c.id === curationIdOrTitle || c.title === curationIdOrTitle)
        if (curation) {
          const contentCacheKey = `curation_content_${curation.id}`
          const cachedContent = cacheManager.get<string>(contentCacheKey)
          
          if (cachedContent) {
            // Display cached content instantly without any loading state
            console.log('✅ VaultState: Loading cached AI curation content instantly:', displayTitle)
            
            const cachedMessage = {
              id: Date.now(),
              type: "curation" as const,
              content: cachedContent,
              title: displayTitle,
              isGenerating: false,
              themes: ["AI Curation", "Cached Content", "Instant Access"],
              documentsUsed: state.documents.selectedFiles.length > 0 ? state.documents.selectedFiles.length : state.documents.data.length,
              documentNames: state.documents.selectedFiles.length > 0 
                ? state.documents.selectedFiles.map(fileId => {
                    const doc = state.documents.data.find(d => d.id === fileId)
                    return doc?.name || fileId
                  })
                : state.documents.data.map(doc => doc.name),
              timestamp: new Date().toISOString(),
              cached: true
            }
            
            dispatch({ type: 'ADD_CHAT_MESSAGE', payload: cachedMessage })
            return // Exit early - no need to generate new content
          }
        }
      }

      // FIXED: Only show loading state when we actually need to generate content
      // For existing curations without cache, or for new curations
      console.log('🔄 VaultState: No cached content found, will generate fresh content with loading state')
      dispatch({ type: 'SET_CURATION_GENERATING', payload: true })

      // The AI service now handles cache checking internally and will return instantly for cached content
      // This call will only show loading if content actually needs to be generated
      const content = await aiCurationService.generateCurationContent(
        curationIdOrTitle, // This can now handle both IDs and titles
        state.settings.selectedProvider,
        state.settings.selectedModel
      )

      // Add the final content message to chat
      const contentMessage = {
        id: Date.now(),
        type: "curation" as const,
        content: content,
        title: displayTitle,
        isGenerating: false,
        themes: ["AI Curation", "Document Analysis", "Insights"],
        documentsUsed: state.documents.selectedFiles.length > 0 ? state.documents.selectedFiles.length : state.documents.data.length,
        documentNames: state.documents.selectedFiles.length > 0 
          ? state.documents.selectedFiles.map(fileId => {
              const doc = state.documents.data.find(d => d.id === fileId)
              return doc?.name || fileId
            })
          : state.documents.data.map(doc => doc.name),
        timestamp: new Date().toISOString()
      }

      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: contentMessage })

      // Cache the generated curation content for instant access on future logins
      const curationForCaching = state.curations.data.find(c => c.id === curationIdOrTitle) || 
                                 state.curations.data.find(c => c.title === curationIdOrTitle)
      
      if (curationForCaching) {
        const contentCacheKey = `curation_content_${curationForCaching.id}`
        cacheManager.set(contentCacheKey, content, 24 * 60 * 60 * 1000) // Cache for 24 hours
        console.log(`✅ VaultState: Cached AI curation content for instant access: ${curationForCaching.title}`)
      } else if (!isExistingCuration) {
        // For new curations created from titles, cache with a temporary key
        const tempCacheKey = `curation_content_temp_${displayTitle.toLowerCase().replace(/\s+/g, '_')}`
        cacheManager.set(tempCacheKey, content, 24 * 60 * 60 * 1000)
        console.log(`✅ VaultState: Cached new AI curation content temporarily: ${displayTitle}`)
      }

      // If this was a new curation created from a title, refresh curations list
      if (!isExistingCuration) {
        console.log('VaultState: Refreshing curations after creating new one from title')
        await loadCurations()
        
        // After refresh, try to cache with proper ID
        const newCuration = state.curations.data.find(c => c.title === displayTitle)
        if (newCuration) {
          const properCacheKey = `curation_content_${newCuration.id}`
          cacheManager.set(properCacheKey, content, 24 * 60 * 60 * 1000)
          console.log(`✅ VaultState: Re-cached AI curation content with proper ID: ${newCuration.id}`)
        }
      }

    } catch (error) {
      console.error('Failed to generate curation content:', error)
      
      // Determine display title for error message
      const errorDisplayTitle = title || curationIdOrTitle
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now(),
        type: "ai" as const,
        content: `Sorry, I encountered an error generating the curation "${errorDisplayTitle}". Please try again or check your connection.`,
        title: errorDisplayTitle,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage })
    } finally {
      dispatch({ type: 'SET_CURATION_GENERATING', payload: false })
    }
  }, [state.settings, state.documents.selectedFiles, state.curations.data, state.documents.data, loadCurations])

  const generateSummaryContent = useCallback(async (summaryName: string) => {
    try {
      dispatch({ type: 'SET_SUMMARY_GENERATING', payload: true })
      
      // Add loading message to chat immediately for visual feedback
      const loadingMessage = {
        id: Date.now(),
        type: "summary" as const,
        content: "",
        title: summaryName,
        isGenerating: true,
        language: 'en',
        documentsUsed: state.documents.selectedFiles.length > 0 ? state.documents.selectedFiles.length : state.documents.data.length,
        documentNames: state.documents.selectedFiles.length > 0 
          ? state.documents.selectedFiles.map(fileId => {
              const doc = state.documents.data.find(d => d.id === fileId)
              return doc?.name || fileId
            })
          : state.documents.data.map(doc => doc.name),
        timestamp: new Date().toISOString()
      }

      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: loadingMessage })

      let content: string

      // Handle different summary types
      if (summaryName === "Combined Summary") {
        // For combined summary, create a custom summary request
        const result = await aiSummaryService.createCustomSummary({
          title: "Combined Document Summary",
          description: "A comprehensive summary combining insights from all selected documents",
          keywords: ["summary", "overview", "key insights"],
          provider: state.settings.selectedProvider,
          model: state.settings.selectedModel,
          documentIds: state.documents.selectedFiles.length > 0 ? state.documents.selectedFiles : state.documents.data.map(doc => doc.id)
        })
        
        if (result.success && result.summary) {
          content = result.summary.content
        } else {
          throw new Error(result.message || "Failed to generate combined summary")
        }
      } else if (summaryName === "Overall Summary") {
        // For overall summary (single document case)
        const result = await aiSummaryService.createCustomSummary({
          title: "Document Summary",
          description: "A comprehensive summary of the document",
          keywords: ["summary", "overview", "key points"],
          provider: state.settings.selectedProvider,
          model: state.settings.selectedModel,
          documentIds: state.documents.selectedFiles.length > 0 ? state.documents.selectedFiles : state.documents.data.map(doc => doc.id)
        })
        
        if (result.success && result.summary) {
          content = result.summary.content
        } else {
          throw new Error(result.message || "Failed to generate overall summary")
        }
      } else {
        // For individual document summaries, find the matching document
        // The summaryName might be truncated, so we need to find the actual document
        let targetDocument = null
        
        // First, try to find exact match
        targetDocument = state.documents.data.find(doc => {
          const docNameWithoutExt = doc.name.split('.').slice(0, -1).join('.')
          return docNameWithoutExt === summaryName
        })
        
        // If no exact match, try to find by truncated name
        if (!targetDocument) {
          targetDocument = state.documents.data.find(doc => {
            const docName = doc.name.split(".")[0]
            const shortName = docName.length > 12 ? docName.substring(0, 12) + "..." : docName
            return shortName === summaryName
          })
        }
        
        // If still no match, try partial matching
        if (!targetDocument) {
          const cleanSummaryName = summaryName.replace("...", "")
          targetDocument = state.documents.data.find(doc => {
            const docName = doc.name.split(".")[0]
            return docName.startsWith(cleanSummaryName)
          })
        }
        
        if (!targetDocument) {
          throw new Error(`Document not found for summary: ${summaryName}`)
        }
        
        // Create individual summary for the found document
        const result = await aiSummaryService.createCustomSummary({
          title: `Summary: ${targetDocument.name}`,
          description: `Individual summary for ${targetDocument.name}`,
          keywords: ["summary", "document analysis", "key insights"],
          provider: state.settings.selectedProvider,
          model: state.settings.selectedModel,
          documentIds: [targetDocument.id]
        })
        
        if (result.success && result.summary) {
          content = result.summary.content
        } else {
          throw new Error(result.message || `Failed to generate summary for ${targetDocument.name}`)
        }
      }

      // Update the loading message with actual content
      dispatch({ 
        type: 'UPDATE_CHAT_MESSAGE', 
        payload: { 
          id: loadingMessage.id, 
          updates: { 
            content: content,
            isGenerating: false
          }
        }
      })

    } catch (error) {
      console.error('Failed to generate summary content:', error)
      
      // Add error message to chat
      const errorMessage = {
        id: Date.now(),
        type: "ai" as const,
        content: `Sorry, I encountered an error generating the summary "${summaryName}". Please try again or check your connection.`,
        title: summaryName,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: errorMessage })
    } finally {
      dispatch({ type: 'SET_SUMMARY_GENERATING', payload: false })
    }
  }, [state.settings, state.documents.selectedFiles, state.documents.data])

  const createCustomCuration = useCallback(async (title: string, description: string, keywords: string[]) => {
    try {
      const result = await aiCurationService.createCustomCuration(
        title, 
        description, 
        keywords, 
        state.settings.selectedProvider, 
        state.settings.selectedModel
      )
      
      if (result.success) {
        // Refresh curations to get the latest data including the new curation
        await loadCurations()
        console.log('VaultState: Successfully created and refreshed curations after custom curation creation')
      } else {
        throw new Error(result.message || 'Failed to create custom curation')
      }
      
      return result
    } catch (error) {
      console.error('Failed to create custom curation:', error)
      throw error
    }
  }, [loadCurations, state.settings])

  const createCustomSummary = useCallback(async (title: string, description: string, keywords: string[]) => {
    try {
      await aiSummaryService.createCustomSummary({
        title,
        description,
        keywords
      })
      
      // Refresh summaries
      await loadSummaries()
    } catch (error) {
      console.error('Failed to create custom summary:', error)
      throw error
    }
  }, [loadSummaries])

  const deleteCuration = useCallback(async (curationId: string) => {
    try {
      // Instantly remove the curation from the UI (optimistic update)
      dispatch({ type: 'SET_CURATIONS', payload: state.curations.data.filter(curation => curation.id !== curationId) })
      
      // Also update custom curations list
      dispatch({ type: 'SET_CUSTOM_CURATIONS', payload: state.curations.customCurations.filter(curation => curation.id !== curationId) })
      
      // Then make the API call to delete from backend
      await ragApiClient.deleteCuration(curationId)
      
      // Remove from cache
      cacheManager.clearByPattern(`curation_.*_${curationId}`)
      
    } catch (error) {
      console.error('Failed to delete curation:', error)
      // If deletion fails, refresh curations to restore the correct state
      await loadCurations()
      throw error
    }
  }, [state.curations.data, state.curations.customCurations, loadCurations])

  const deleteSummary = useCallback(async (summaryId: string) => {
    try {
      // Instantly remove the summary from the UI (optimistic update)
      dispatch({ type: 'SET_SUMMARIES', payload: state.summaries.data.filter(summary => summary.id !== summaryId) })
      
      // Then make the API call to delete from backend
      await ragApiClient.deleteSummary(summaryId)
      
      // Remove from cache
      cacheManager.clearByPattern(`summary_.*_${summaryId}`)
      
    } catch (error) {
      console.error('Failed to delete summary:', error)
      // If deletion fails, refresh summaries to restore the correct state
      await loadSummaries()
      throw error
    }
  }, [state.summaries.data, loadSummaries])

  const deleteConversation = useCallback(async (conversationId: string) => {
    try {
      // Instantly remove the conversation from the UI (optimistic update)
      dispatch({ type: 'SET_CONVERSATIONS', payload: state.conversations.data.filter(conversation => conversation.id !== conversationId) })
      
      // Then make the API call to delete from backend
      await ragApiClient.deleteConversation(conversationId)
      
      // Remove from cache
      cacheManager.clearByPattern(`conversation_.*_${conversationId}`)
      
    } catch (error) {
      console.error('Failed to delete conversation:', error)
      // If deletion fails, refresh conversations to restore the correct state
      await loadConversations()
      throw error
    }
  }, [state.conversations.data, loadConversations])

  // ============================================================================
  // CONVERSATION ACTIONS
  // ============================================================================

  const sendMessage = useCallback(async (message: string) => {
    try {
      // Add user message immediately
      const userMessage = {
        id: Date.now(),
        type: "user" as const,
        content: message,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: userMessage })

      // Add generating AI message
      const aiMessage = {
        id: Date.now() + 1,
        type: "ai" as const,
        content: "",
        isGenerating: true,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'ADD_CHAT_MESSAGE', payload: aiMessage })

      // Send to backend using queryDocumentsWithConversation
      const response = await ragApiClient.queryDocumentsWithConversation(message, {
        provider: state.settings.selectedProvider,
        model: state.settings.selectedModel,
        maxContext: state.settings.maxContext === 'YES',
        documentIds: state.documents.selectedFiles,
        conversationId: state.conversations.currentConversationId || undefined,
        saveConversation: true
      })

      // Update AI message with response
      dispatch({ 
        type: 'UPDATE_CHAT_MESSAGE', 
        payload: { 
          id: aiMessage.id, 
          updates: { 
            content: response.answer,
            isGenerating: false,
            conversationId: response.conversation_id
          }
        }
      })

      // Update current conversation ID if new
      if (response.conversation_id && response.conversation_id !== state.conversations.currentConversationId) {
        dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: response.conversation_id })
      }

    } catch (error) {
      console.error('Failed to send message:', error)
      
      // Update AI message with error
      const errorMessage = {
        id: Date.now() + 1,
        type: "ai" as const,
        content: "Sorry, I encountered an error processing your message. Please try again.",
        isGenerating: false,
        timestamp: new Date().toISOString()
      }
      dispatch({ type: 'UPDATE_CHAT_MESSAGE', payload: { id: errorMessage.id, updates: errorMessage } })
    }
  }, [state.conversations.currentConversationId, state.settings, state.documents.selectedFiles])

  const loadConversationHistory = useCallback(async (conversationId: string) => {
    try {
      const conversation = await ragApiClient.getConversation(conversationId)
      
      if (conversation) {
        dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: conversationId })
        dispatch({ type: 'SET_SELECTED_CONVERSATION', payload: conversationId })
        dispatch({ type: 'SET_CONVERSATION_TITLE', payload: conversation.conversation.title })
        
        // Convert messages to chat format
        const chatMessages = conversation.messages.map((msg: any, index: number) => ({
          id: index,
          type: msg.role === 'user' ? 'user' as const : 'ai' as const,
          content: msg.content,
          timestamp: msg.timestamp || new Date().toISOString(),
          relevanceScores: msg.relevanceScores,
          contextMode: msg.contextMode
        }))
        
        dispatch({ type: 'SET_CHAT_MESSAGES', payload: chatMessages })
      }
    } catch (error) {
      console.error('Failed to load conversation history:', error)
    }
  }, [])

  const startNewConversation = useCallback(() => {
    dispatch({ type: 'SET_CURRENT_CONVERSATION', payload: '' })
    dispatch({ type: 'SET_SELECTED_CONVERSATION', payload: '' })
    dispatch({ type: 'SET_CONVERSATION_TITLE', payload: '' })
    dispatch({ type: 'SET_CHAT_MESSAGES', payload: [] })
  }, [])

  // ============================================================================
  // UI ACTIONS
  // ============================================================================

  const toggleSidebar = useCallback(() => {
    dispatch({ type: 'SET_SIDEBAR_OPEN', payload: !state.ui.sidebarOpen })
  }, [state.ui.sidebarOpen])

  const toggleSidebarExpansion = useCallback(() => {
    dispatch({ type: 'SET_SIDEBAR_EXPANDED', payload: !state.ui.sidebarExpanded })
  }, [state.ui.sidebarExpanded])

  const setActiveTab = useCallback((tab: 'curations' | 'summaries' | 'history') => {
    dispatch({ type: 'SET_ACTIVE_TAB', payload: tab })
  }, [])

  const toggleExpandedSources = useCallback((messageId: number) => {
    dispatch({ type: 'TOGGLE_EXPANDED_SOURCES', payload: messageId })
  }, [])

  const toggleExpandedRelevance = useCallback((messageId: number) => {
    dispatch({ type: 'TOGGLE_EXPANDED_RELEVANCE', payload: messageId })
  }, [])

  // ============================================================================
  // SETTINGS ACTIONS
  // ============================================================================

  const updateProviderSettings = useCallback((provider: string, model: string) => {
    const selectedProvider = state.settings.modelProviders.find((p: any) => p.name === provider)
    const availableModels = selectedProvider?.models || []
    
    // Auto-select default model if no model is provided or if switching providers
    let selectedModel = model
    if (!selectedModel || provider !== state.settings.selectedProvider) {
      // Define default models for each provider
      const defaultModels: Record<string, string> = {
        'gemini': 'gemini-1.5-flash',
        'openai': 'gpt-4o-mini',
        'groq': 'llama3-8b-8192',
        'deepseek': 'deepseek-chat',
        'bedrock': 'anthropic.claude-3-haiku-20240307-v1:0',
        'anthropic': 'claude-3-haiku-20240307',
        'ollama': 'llama3.2:3b'
      }
      
      // Get the default model for this provider
      const defaultModel = defaultModels[provider]
      
      // Check if the default model is available in the provider's models
      if (defaultModel && availableModels.some((m: any) => m.name === defaultModel)) {
        selectedModel = defaultModel
      } else if (availableModels.length > 0) {
        // If default model not available, select the first available model
        selectedModel = availableModels[0].name
      }
      
      console.log(`🎯 VaultState: Auto-selected default model for ${provider}: ${selectedModel}`)
    }
    
    // Cache the provider settings for instant restoration on next login
    const providerSettings = {
      provider,
      model: selectedModel,
      timestamp: Date.now()
    }
    localStorage.setItem('vault_provider_settings', JSON.stringify(providerSettings))
    console.log(`✅ VaultState: Cached AI provider settings for instant access: ${provider} - ${selectedModel}`)
    
    dispatch({ 
      type: 'SET_PROVIDER_SETTINGS', 
      payload: { 
        provider, 
        model: selectedModel, 
        availableModels,
        modelProviders: state.settings.modelProviders
      }
    })
  }, [state.settings.modelProviders, state.settings.selectedProvider])

  const updateContextSettings = useCallback((keepContext: string, maxContext: string) => {
    dispatch({ type: 'SET_CONTEXT_SETTINGS', payload: { keepContext, maxContext } })
  }, [])

  const setSearchTag = useCallback((tag: string) => {
    dispatch({ type: 'SET_SEARCH_TAG', payload: tag })
  }, [])

  const setSelectedFiles = useCallback((files: string[]) => {
    dispatch({ type: 'SET_SELECTED_FILES', payload: files })
  }, [])

  // ============================================================================
  // EFFECTS
  // ============================================================================

  // Handle authentication state changes
  useEffect(() => {
    const wasAuthenticated = lastAuthStateRef.current
    const isNowAuthenticated = isAuthenticated

    // If user just logged out, reset initialization
    if (wasAuthenticated && !isNowAuthenticated) {
      console.log('🔄 VaultState: User logged out, resetting state')
      initializationRef.current = false
      dispatch({ type: 'RESET_STATE' })
    }

    // If user just logged in, initialize vault
    if (!wasAuthenticated && isNowAuthenticated && !authLoading) {
      console.log('🔄 VaultState: User logged in, initializing vault')
      initializationRef.current = false
      initializeVault()
    }

    // Update the last auth state
    lastAuthStateRef.current = isAuthenticated
  }, [isAuthenticated, authLoading, initializeVault])

  // Initialize vault on mount
  useEffect(() => {
    initializeVault()
  }, [initializeVault])

  // Setup background refresh interval (only when authenticated)
  useEffect(() => {
    if (!isAuthenticated) return

    const interval = setInterval(() => {
      if (isAuthenticated) {
        refreshAllDataInBackground()
      }
    }, 10 * 60 * 1000) // Every 10 minutes

    return () => clearInterval(interval)
  }, [isAuthenticated, refreshAllDataInBackground])

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (backgroundRefreshRef.current) {
        clearTimeout(backgroundRefreshRef.current)
      }
    }
  }, [])

  // ============================================================================
  // CONTEXT VALUE
  // ============================================================================

  const contextValue: VaultContextType = {
    state,
    dispatch,
    
    // High-level actions
    initializeVault,
    refreshAllData,
    
    // Curation actions
    loadCurations,
    generateCurationContent,
    createCustomCuration,
    deleteCuration,
    
    // Summary actions
    loadSummaries,
    generateSummaryContent,
    createCustomSummary,
    deleteSummary,
    
    // Conversation actions
    loadConversations,
    sendMessage,
    loadConversationHistory,
    startNewConversation,
    deleteConversation,
    
    // Document actions
    loadDocuments,
    
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
    setSelectedFiles
  }

  return (
    <VaultContext.Provider value={contextValue}>
      {children}
    </VaultContext.Provider>
  )
}

// ============================================================================
// HOOK
// ============================================================================

export function useVaultState() {
  const context = useContext(VaultContext)
  if (context === undefined) {
    throw new Error('useVaultState must be used within a VaultStateProvider')
  }
  return context
}

export default VaultStateProvider
