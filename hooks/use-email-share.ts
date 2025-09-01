"use client"

import { useState } from 'react'
import { EmailShareData } from '@/components/email-share-modal'
import { emailShareService, EmailRequest } from '@/lib/email-share-service'
import { useToast } from '@/hooks/use-toast'

export interface UseEmailShareOptions {
  defaultSenderName?: string
  defaultSenderCompany?: string
}

export function useEmailShare(options: UseEmailShareOptions = {}) {
  const { toast } = useToast()
  const [isModalOpen, setIsModalOpen] = useState(false)
  const [currentData, setCurrentData] = useState<EmailShareData | null>(null)
  const [isLoading, setIsLoading] = useState(false)

  /**
   * Open email share modal with RAG query data
   */
  const shareRagQuery = (data: {
    query: string
    answer: string
    sources?: Array<{ name: string; id: string }>
    metadata?: {
      provider?: string
      model?: string
      responseTime?: number
      confidence?: number
      mode?: string
    }
    title?: string
  }) => {
    const emailData: EmailShareData = {
      type: 'rag',
      title: data.title || extractTitleFromQuery(data.query),
      content: data.answer,
      query: data.query,
      sources: data.sources,
      metadata: {
        ...data.metadata,
        generatedAt: new Date().toISOString(),
      },
      senderName: options.defaultSenderName,
      senderCompany: options.defaultSenderCompany,
    }

    setCurrentData(emailData)
    setIsModalOpen(true)
  }

  /**
   * Open email share modal with summary data
   */
  const shareSummary = (data: {
    title: string
    content: string
    sources?: Array<{ name: string; id: string }>
    metadata?: {
      provider?: string
      model?: string
      documentCount?: number
      summaryType?: string
    }
  }) => {
    const emailData: EmailShareData = {
      type: 'summary',
      title: data.title,
      content: data.content,
      sources: data.sources,
      metadata: {
        ...data.metadata,
        generatedAt: new Date().toISOString(),
      },
      senderName: options.defaultSenderName,
      senderCompany: options.defaultSenderCompany,
    }

    setCurrentData(emailData)
    setIsModalOpen(true)
  }

  /**
   * Open email share modal with curation data
   */
  const shareCuration = (data: {
    title: string
    content: string
    sources?: Array<{ name: string; id: string }>
    metadata?: {
      provider?: string
      model?: string
      documentCount?: number
      keywords?: string[]
    }
  }) => {
    const emailData: EmailShareData = {
      type: 'curation',
      title: data.title,
      content: data.content,
      sources: data.sources,
      metadata: {
        ...data.metadata,
        generatedAt: new Date().toISOString(),
      },
      senderName: options.defaultSenderName,
      senderCompany: options.defaultSenderCompany,
    }

    setCurrentData(emailData)
    setIsModalOpen(true)
  }

  /**
   * Handle sending email
   */
  const handleSendEmail = async (emailData: {
    to: string[]
    cc: string[]
    bcc: string[]
    subject: string
    personalMessage: string
    includeMetadata: boolean
    useActualEmail?: boolean
  }) => {
    if (!currentData) {
      throw new Error('No data to send')
    }

    setIsLoading(true)

    try {
      const request: EmailRequest = {
        ...emailData,
        data: currentData,
      }

      const result = await emailShareService.sendEmail(request, emailData.useActualEmail || false)

      if (!result.success) {
        throw new Error(result.message)
      }

      // Show success message
      const emailType = emailData.useActualEmail ? "actual email" : "simulated email"
      toast({
        title: "Email Sent Successfully",
        description: result.simulated 
          ? `${emailType} was processed (${result.simulated ? 'simulated' : 'sent'})`
          : `${emailType} sent to ${emailData.to.length} recipient${emailData.to.length > 1 ? 's' : ''}`,
      })

      // Close modal
      setIsModalOpen(false)
      setCurrentData(null)

      return result
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Failed to send email'
      
      toast({
        title: "Failed to Send Email",
        description: errorMessage,
        variant: "destructive"
      })

      throw error
    } finally {
      setIsLoading(false)
    }
  }

  /**
   * Close modal
   */
  const closeModal = () => {
    setIsModalOpen(false)
    setCurrentData(null)
  }

  /**
   * Extract a meaningful title from a query
   */
  const extractTitleFromQuery = (query: string): string => {
    // Take first 50 characters and add ellipsis if longer
    const title = query.length > 50 ? query.substring(0, 50) + '...' : query
    return title
  }

  return {
    // State
    isModalOpen,
    currentData,
    isLoading,

    // Actions
    shareRagQuery,
    shareSummary,
    shareCuration,
    handleSendEmail,
    closeModal,
    setCurrentData,
    setIsModalOpen,

    // Utilities
    extractTitleFromQuery,
  }
}

/**
 * Hook for sharing conversation messages as email
 */
export function useConversationEmailShare(options: UseEmailShareOptions = {}) {
  const emailShare = useEmailShare(options)

  /**
   * Share a conversation message pair (user query + assistant response)
   */
  const shareConversationMessage = (data: {
    userMessage: string
    assistantMessage: string
    sources?: Array<{ name: string; id: string }>
    metadata?: {
      provider?: string
      model?: string
      responseTime?: number
      confidence?: number
      mode?: string
    }
    conversationTitle?: string
  }) => {
    emailShare.shareRagQuery({
      query: data.userMessage,
      answer: data.assistantMessage,
      sources: data.sources,
      metadata: data.metadata,
      title: data.conversationTitle || emailShare.extractTitleFromQuery(data.userMessage),
    })
  }

  /**
   * Share entire conversation as email
   */
  const shareEntireConversation = (data: {
    title: string
    messages: Array<{
      role: 'user' | 'assistant'
      content: string
      timestamp?: string
    }>
    metadata?: {
      provider?: string
      model?: string
      totalMessages?: number
    }
  }) => {
    // Format conversation as content
    let conversationContent = `# Conversation: ${data.title}\n\n`
    
    data.messages.forEach((message, index) => {
      const timestamp = message.timestamp 
        ? new Date(message.timestamp).toLocaleString()
        : `Message ${index + 1}`
      
      if (message.role === 'user') {
        conversationContent += `## 👤 User (${timestamp})\n${message.content}\n\n`
      } else {
        conversationContent += `## 🤖 Assistant (${timestamp})\n${message.content}\n\n`
      }
    })

    const emailData: EmailShareData = {
      type: 'rag',
      title: `Conversation: ${data.title}`,
      content: conversationContent,
      metadata: {
        ...data.metadata,
        generatedAt: new Date().toISOString(),
      },
      senderName: options.defaultSenderName,
      senderCompany: options.defaultSenderCompany,
    }

    emailShare.setCurrentData(emailData)
    emailShare.setIsModalOpen(true)
  }

  return {
    ...emailShare,
    shareConversationMessage,
    shareEntireConversation,
  }
}

/**
 * Utility function to get user info for email sharing
 */
export function getUserInfoForEmail(): { senderName?: string; senderCompany?: string } {
  // This would typically come from your auth context or user store
  // For now, return empty - you can integrate with your auth system
  return {
    senderName: undefined,
    senderCompany: undefined,
  }
}
