'use client'

import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { Input } from '@/components/ui/input'
import { 
  MessageSquare, 
  Search, 
  Clock, 
  Calendar, 
  Trash2, 
  Archive, 
  Pin, 
  PinOff,
  RefreshCw,
  Plus,
  Filter,
  MoreVertical,
  Loader2,
  Mail
} from 'lucide-react'
import {
  Popover,
  PopoverContent,
  PopoverTrigger,
} from "@/components/ui/popover"
import { ragApiClient, type Conversation, type ConversationWithMessages } from '@/lib/api-client'
import { formatDistanceToNow } from 'date-fns'
import { useVaultState } from '@/lib/vault-state-provider'
import { cacheManager } from '@/lib/cache-manager'
import { EmailShareModal } from '@/components/email-share-modal'
import { useEmailShare, getUserInfoForEmail } from '@/hooks/use-email-share'

interface HistorySectionProps {
  onConversationSelect?: (conversationId: string) => void
  selectedConversationId?: string
  onHistoryCardClick?: (conversation: any, messages: any[]) => void
  onNewConversation?: () => void
}

export default function HistorySection({ 
  onConversationSelect, 
  selectedConversationId, 
  onHistoryCardClick,
  onNewConversation 
}: HistorySectionProps) {
  const { state, loadConversations: vaultLoadConversations, dispatch } = useVaultState()
  const [searchQuery, setSearchQuery] = useState('')
  const [showArchived, setShowArchived] = useState(false)
  const [refreshing, setRefreshing] = useState(false)
  const [loadingConversationId, setLoadingConversationId] = useState<string | null>(null)

  // Email sharing functionality
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  // Use conversations from vault state (preloaded)
  const conversations = state.conversations.data
  const loading = state.conversations.loading

  // Load conversations when showArchived changes
  useEffect(() => {
    if (showArchived !== false) {
      // Only reload if we need archived conversations (vault state loads active by default)
      loadConversations()
    }
  }, [showArchived])

  // Filter conversations based on search query
  const filteredConversations = conversations.filter(conversation =>
    conversation.title.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const loadConversations = async () => {
    try {
      dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: true } })
      const data = await ragApiClient.getConversations({
        includeArchived: showArchived,
        limit: 50
      })
      dispatch({ type: 'SET_CONVERSATIONS', payload: data })
      console.log('✅ HistorySection: Loaded conversations instantly:', data.length)
    } catch (error) {
      console.error('Failed to load conversations:', error)
      dispatch({ type: 'SET_CONVERSATIONS', payload: [] })
    } finally {
      dispatch({ type: 'SET_LOADING', payload: { section: 'conversations', loading: false } })
    }
  }

  const refreshConversations = async () => {
    setRefreshing(true)
    await vaultLoadConversations()
    setRefreshing(false)
  }

  const loadConversation = async (conversationId: string) => {
    try {
      // Set loading state immediately for visual feedback
      setLoadingConversationId(conversationId)
      
      // Immediately update UI to show selection for instant visual feedback
      dispatch({ type: 'SET_SELECTED_CONVERSATION', payload: conversationId })
      console.log(`⚡ HistorySection: Instantly selected conversation ${conversationId} for immediate UI feedback`)
      
      // Check if conversation content is preloaded for instant access
      const preloadedContent = sessionStorage.getItem('preloaded_conversation_content')
      if (preloadedContent) {
        try {
          const conversationContentCache = JSON.parse(preloadedContent)
          const cachedConversation = conversationContentCache[conversationId]
          
          if (cachedConversation && onHistoryCardClick) {
            console.log(`✅ HistorySection: Loading conversation ${conversationId} instantly from preloaded cache`)
            onHistoryCardClick(cachedConversation.conversation, cachedConversation.messages)
            setLoadingConversationId(null) // Clear loading state
            return // Exit early - conversation loaded instantly!
          }
        } catch (error) {
          console.error('Error parsing preloaded conversation content:', error)
        }
      }
      
      // Check individual conversation cache for instant access
      const conversationCacheKey = `conversation_content_${conversationId}`
      const cachedConversationData = cacheManager.get<any>(conversationCacheKey)
      
      if (cachedConversationData && onHistoryCardClick) {
        console.log(`✅ HistorySection: Loading conversation ${conversationId} instantly from individual cache`)
        onHistoryCardClick(cachedConversationData.conversation, cachedConversationData.messages)
        setLoadingConversationId(null) // Clear loading state
        return // Exit early - conversation loaded instantly!
      }
      
      // If not cached, load from API and cache for future instant access
      console.log(`🔄 HistorySection: Loading conversation ${conversationId} from API and caching for future instant access`)
      const data = await ragApiClient.getConversation(conversationId)
      
      if (data) {
        // Cache the conversation content for future instant access (24 hours)
        cacheManager.set(conversationCacheKey, data, 24 * 60 * 60 * 1000)
        console.log(`✅ HistorySection: Cached conversation ${conversationId} for future instant access`)
        
        if (onHistoryCardClick) {
          onHistoryCardClick(data.conversation, data.messages)
        }
      }
    } catch (error) {
      console.error('Failed to load conversation:', error)
    } finally {
      // Always clear loading state
      setLoadingConversationId(null)
    }
  }

  const deleteConversation = async (conversationId: string, event: React.MouseEvent) => {
    event.stopPropagation()

    // Instantly remove from frontend for immediate UI response
    const updatedConversations = conversations.filter(conv => conv.id !== conversationId)
    dispatch({ type: 'SET_CONVERSATIONS', payload: updatedConversations })
    console.log('✅ HistorySection: Conversation removed from UI instantly')

    // Handle backend deletion in background
    try {
      const success = await ragApiClient.deleteConversation(conversationId)
      if (!success) {
        console.error('Backend deletion failed, but UI already updated')
        // Optionally: Could restore the conversation if backend fails
        // But for professional UX, we keep it removed from frontend
      } else {
        console.log('✅ HistorySection: Backend deletion completed successfully')
      }
    } catch (error) {
      console.error('Failed to delete conversation from backend:', error)
      // UI already updated, so user sees instant deletion regardless
    }
  }

  const toggleArchiveConversation = async (conversation: Conversation, event: React.MouseEvent) => {
    event.stopPropagation()
    
    try {
      const updated = await ragApiClient.updateConversation(conversation.id, {
        isArchived: !conversation.is_archived
      })
      
      if (updated) {
        const updatedConversations = conversations.map(conv => 
          conv.id === conversation.id 
            ? { ...conv, is_archived: !conv.is_archived }
            : conv
        )
        dispatch({ type: 'SET_CONVERSATIONS', payload: updatedConversations })
        console.log(`✅ HistorySection: Conversation ${conversation.is_archived ? 'unarchived' : 'archived'} successfully`)
      }
    } catch (error) {
      console.error('Failed to toggle archive status:', error)
    }
  }

  const togglePinConversation = async (conversation: Conversation, event: React.MouseEvent) => {
    event.stopPropagation()
    
    try {
      const updated = await ragApiClient.updateConversation(conversation.id, {
        isPinned: !conversation.is_pinned
      })
      
      if (updated) {
        const updatedConversations = conversations.map(conv => 
          conv.id === conversation.id 
            ? { ...conv, is_pinned: !conv.is_pinned }
            : conv
        )
        dispatch({ type: 'SET_CONVERSATIONS', payload: updatedConversations })
        console.log(`✅ HistorySection: Conversation ${conversation.is_pinned ? 'unpinned' : 'pinned'} successfully`)
      }
    } catch (error) {
      console.error('Failed to toggle pin status:', error)
    }
  }

  const formatMessageTime = (timestamp: string) => {
    return formatDistanceToNow(new Date(timestamp), { addSuffix: true })
  }

  const getConversationIcon = (conversation: Conversation) => {
    if (conversation.is_pinned) return Pin
    if (conversation.is_archived) return Archive
    return MessageSquare
  }

  const shareConversationAsEmail = async (conversation: Conversation, event: React.MouseEvent) => {
    event.stopPropagation()
    
    try {
      // Load conversation messages if not already loaded
      const conversationCacheKey = `conversation_content_${conversation.id}`
      let conversationData = cacheManager.get<any>(conversationCacheKey)
      
      if (!conversationData) {
        // Load from API if not cached
        conversationData = await ragApiClient.getConversation(conversation.id)
        if (conversationData) {
          cacheManager.set(conversationCacheKey, conversationData, 24 * 60 * 60 * 1000)
        }
      }
      
      if (conversationData && conversationData.messages) {
        // Format conversation for email
        const messages = conversationData.messages.map((msg: any) => ({
          role: msg.role,
          content: msg.content,
          timestamp: msg.timestamp
        }))
        
        // Create formatted conversation content
        let conversationContent = `# ${conversation.title}\n\n`
        conversationContent += `**Date:** ${new Date(conversation.created_at).toLocaleDateString()}\n\n`
        
        messages.forEach((message: any, index: number) => {
          const timestamp = message.timestamp 
            ? new Date(message.timestamp).toLocaleString()
            : `Message ${index + 1}`
          
          if (message.role === 'user') {
            conversationContent += `## 👤 User (${timestamp})\n${message.content}\n\n`
          } else {
            conversationContent += `## 🤖 Assistant (${timestamp})\n${message.content}\n\n`
          }
        })
        
        // Share as RAG query with conversation content
        emailShare.shareRagQuery({
          query: `Full Conversation: ${conversation.title}`,
          answer: conversationContent,
          title: `Chat History - ${conversation.title}`,
          metadata: {
            provider: 'SIXTHVAULT',
            model: 'Chat History',
            responseTime: messages.length * 100, // Approximate based on message count
            mode: 'conversation_export'
          }
        })
      }
    } catch (error) {
      console.error('Failed to share conversation as email:', error)
    }
  }

  return (
    <div className="flex-1 overflow-auto px-4 pb-4">
      <Card className="border-0 shadow-lg bg-white/95 backdrop-blur-xl relative overflow-hidden">
        {/* Beautiful flowing wave background for history */}
        <div className="absolute inset-0 pointer-events-none">
          <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 400 200" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="historyWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.3"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.2"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.1"/>
              </linearGradient>
              <linearGradient id="historyWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.2"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.15"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.1"/>
              </linearGradient>
            </defs>
            <g stroke="url(#historyWave1)" strokeWidth="1" fill="none" opacity="0.6">
              <path d="M0,50 Q100,30 200,45 T400,40"/>
              <path d="M0,70 Q120,50 240,65 T400,60"/>
            </g>
            <g stroke="url(#historyWave2)" strokeWidth="0.8" fill="none" opacity="0.5">
              <path d="M0,90 Q150,70 300,85 T400,80"/>
              <path d="M0,110 Q180,90 360,105 T400,100"/>
            </g>
            <path d="M0,80 Q100,60 200,75 T400,70 L400,200 L0,200 Z" fill="url(#historyWave1)" opacity="0.05"/>
            <path d="M0,120 Q150,100 300,115 T400,110 L400,200 L0,200 Z" fill="url(#historyWave2)" opacity="0.03"/>
          </svg>
        </div>
        <CardHeader className="pb-3">
          <CardTitle className="text-sm font-semibold text-slate-800 flex items-center">
            <MessageSquare className="w-4 h-4 mr-2 text-blue-600" />
            CHAT HISTORY
            <Badge variant="secondary" className="ml-2 text-xs bg-blue-100 text-blue-700">
              {conversations.length}
            </Badge>
            <div className="flex items-center space-x-1 ml-auto">
              <Button
                variant="ghost"
                size="sm"
                onClick={onNewConversation}
                className="h-6 w-6 p-0 text-blue-600 hover:bg-blue-100 hover:text-blue-700"
                title="Start new conversation"
              >
                <Plus className="h-4 w-4" />
              </Button>
              <Button
                variant="ghost"
                size="sm"
                onClick={refreshConversations}
                disabled={refreshing}
                className="h-6 w-6 p-0 text-slate-600 hover:bg-slate-100"
                title="Refresh conversations"
              >
                <RefreshCw className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              </Button>
            </div>
          </CardTitle>
        </CardHeader>
        <CardContent className="pt-0">
          {/* Search and Filters */}
          <div className="space-y-3 mb-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-slate-400" />
              <Input
                placeholder="Search conversations..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                className="pl-9 h-8 text-sm bg-white border-slate-200"
              />
            </div>
            
            <div className="flex items-center justify-between">
              <Button
                variant={showArchived ? "default" : "outline"}
                size="sm"
                onClick={() => setShowArchived(!showArchived)}
                className="h-7 px-3 text-xs"
              >
                <Archive className="w-3 h-3 mr-1" />
                {showArchived ? 'Show Active' : 'Show Archived'}
              </Button>
              
              <div className="text-xs text-slate-500">
                {filteredConversations.length} of {conversations.length}
              </div>
            </div>
          </div>

          {/* Conversations List - Scrollable container */}
          <div className="max-h-96 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-300 scrollbar-track-slate-100 hover:scrollbar-thumb-slate-400 pr-2">
            <div className="space-y-3">
              {loading ? (
                <div className="text-center py-8">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                    <MessageSquare className="w-6 h-6 text-blue-400 animate-pulse" />
                  </div>
                  <p className="text-sm text-slate-600 mb-2">Loading conversations...</p>
                  <p className="text-xs text-slate-500">Please wait while we fetch your chat history</p>
                </div>
              ) : filteredConversations.length === 0 ? (
                <div className="text-center py-8">
                  <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center mx-auto mb-4">
                    <MessageSquare className="w-6 h-6 text-blue-400" />
                  </div>
                  <p className="text-sm text-slate-600 mb-2">
                    {searchQuery ? 'No conversations found' : 'No conversations yet'}
                  </p>
                  <p className="text-xs text-slate-500">
                    {searchQuery ? 'Try a different search term' : 'Start chatting to see your conversation history'}
                  </p>
                </div>
              ) : (
                filteredConversations
                  .sort((a, b) => {
                    // Sort by pinned first, then by updated date
                    if (a.is_pinned && !b.is_pinned) return -1
                    if (!a.is_pinned && b.is_pinned) return 1
                    return new Date(b.updated_at).getTime() - new Date(a.updated_at).getTime()
                  })
                  .map((conversation) => {
                    const IconComponent = getConversationIcon(conversation)
                    const isSelected = selectedConversationId === conversation.id
                    const isLoading = loadingConversationId === conversation.id
                    
                    return (
                      <div
                        key={conversation.id}
                        className={`group relative cursor-pointer border rounded-lg transition-all duration-200 py-2 px-3 transform hover:scale-[1.02] active:scale-[0.98] active:shadow-lg ${
                          isSelected
                            ? "border-blue-200 bg-blue-50 shadow-md"
                            : conversation.is_archived
                              ? "border-gray-200 bg-gray-50 cursor-not-allowed opacity-60"
                              : isLoading
                                ? "border-blue-300 bg-blue-50/50 shadow-md"
                                : "border-gray-200 bg-white hover:border-blue-300 hover:shadow-md hover:bg-blue-50/30"
                        }`}
                        onClick={() => loadConversation(conversation.id)}
                      >
                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-3 flex-1 min-w-0">
                            {isLoading ? (
                              <Loader2 className="w-4 h-4 flex-shrink-0 text-blue-600 animate-spin" />
                            ) : (
                              <IconComponent className={`w-4 h-4 flex-shrink-0 ${
                                isSelected ? "text-blue-600" : "text-gray-500"
                              }`} />
                            )}
                            <span className={`text-sm truncate ${
                              isSelected ? "text-blue-900 font-medium" : isLoading ? "text-blue-800 font-medium" : "text-gray-700"
                            }`}>
                              {conversation.title}
                            </span>
                            {isLoading && (
                              <span className="text-xs text-blue-600 font-medium">Loading...</span>
                            )}
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
                                  onClick={(e) => shareConversationAsEmail(conversation, e)}
                                >
                                  <Mail className="w-3 h-3 mr-2" />
                                  Send as Email
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="w-full justify-start text-xs h-8"
                                  onClick={(e) => togglePinConversation(conversation, e)}
                                >
                                  {conversation.is_pinned ? (
                                    <>
                                      <PinOff className="w-3 h-3 mr-2" />
                                      Unpin
                                    </>
                                  ) : (
                                    <>
                                      <Pin className="w-3 h-3 mr-2" />
                                      Pin
                                    </>
                                  )}
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="w-full justify-start text-xs h-8"
                                  onClick={(e) => toggleArchiveConversation(conversation, e)}
                                >
                                  <Archive className="w-3 h-3 mr-2" />
                                  {conversation.is_archived ? 'Unarchive' : 'Archive'}
                                </Button>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  className="w-full justify-start text-xs h-8 text-red-600 hover:text-red-700 hover:bg-red-50"
                                  onClick={(e) => deleteConversation(conversation.id, e)}
                                >
                                  <Trash2 className="w-3 h-3 mr-2" />
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

      {/* Email Share Modal */}
      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </div>
  )
}
