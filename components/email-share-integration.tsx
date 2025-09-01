"use client"

import React from 'react'
import { Button } from '@/components/ui/button'
import { EmailShareModal } from '@/components/email-share-modal'
import { useEmailShare, getUserInfoForEmail } from '@/hooks/use-email-share'
import { Mail, Share } from 'lucide-react'

/**
 * Example component showing how to integrate email sharing with RAG queries
 */
export function RagQueryEmailShare({ 
  query, 
  answer, 
  sources, 
  metadata 
}: {
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
}) {
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  const handleShare = () => {
    emailShare.shareRagQuery({
      query,
      answer,
      sources,
      metadata,
    })
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={handleShare}
        className="flex items-center gap-2"
      >
        <Mail className="h-4 w-4" />
        Send as Email
      </Button>

      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </>
  )
}

/**
 * Example component showing how to integrate email sharing with summaries
 */
export function SummaryEmailShare({ 
  title, 
  content, 
  sources, 
  metadata 
}: {
  title: string
  content: string
  sources?: Array<{ name: string; id: string }>
  metadata?: {
    provider?: string
    model?: string
    documentCount?: number
    summaryType?: string
  }
}) {
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  const handleShare = () => {
    emailShare.shareSummary({
      title,
      content,
      sources,
      metadata,
    })
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={handleShare}
        className="flex items-center gap-2"
      >
        <Share className="h-4 w-4" />
        Share Summary
      </Button>

      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </>
  )
}

/**
 * Example component showing how to integrate email sharing with curations
 */
export function CurationEmailShare({ 
  title, 
  content, 
  sources, 
  metadata 
}: {
  title: string
  content: string
  sources?: Array<{ name: string; id: string }>
  metadata?: {
    provider?: string
    model?: string
    documentCount?: number
    keywords?: string[]
  }
}) {
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  const handleShare = () => {
    emailShare.shareCuration({
      title,
      content,
      sources,
      metadata,
    })
  }

  return (
    <>
      <Button
        variant="outline"
        size="sm"
        onClick={handleShare}
        className="flex items-center gap-2"
      >
        <Mail className="h-4 w-4" />
        Email Curation
      </Button>

      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </>
  )
}

/**
 * Generic email share button that can be used anywhere
 */
export function GenericEmailShareButton({ 
  data,
  buttonText = "Send as Email",
  variant = "outline",
  size = "sm"
}: {
  data: {
    type: 'rag' | 'summary' | 'curation'
    title: string
    content: string
    query?: string
    sources?: Array<{ name: string; id: string }>
    metadata?: any
  }
  buttonText?: string
  variant?: "default" | "destructive" | "outline" | "secondary" | "ghost" | "link"
  size?: "default" | "sm" | "lg" | "icon"
}) {
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  const handleShare = () => {
    switch (data.type) {
      case 'rag':
        emailShare.shareRagQuery({
          query: data.query || '',
          answer: data.content,
          sources: data.sources,
          metadata: data.metadata,
          title: data.title,
        })
        break
      case 'summary':
        emailShare.shareSummary({
          title: data.title,
          content: data.content,
          sources: data.sources,
          metadata: data.metadata,
        })
        break
      case 'curation':
        emailShare.shareCuration({
          title: data.title,
          content: data.content,
          sources: data.sources,
          metadata: data.metadata,
        })
        break
    }
  }

  return (
    <>
      <Button
        variant={variant}
        size={size}
        onClick={handleShare}
        className="flex items-center gap-2"
      >
        <Mail className="h-4 w-4" />
        {buttonText}
      </Button>

      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </>
  )
}

/**
 * Example of how to add email sharing to the vault chat interface
 */
export function VaultChatEmailShare({ 
  messages 
}: {
  messages: Array<{
    role: 'user' | 'assistant'
    content: string
    sources?: Array<{ name: string; id: string }>
    metadata?: any
    timestamp?: string
  }>
}) {
  const userInfo = getUserInfoForEmail()
  const emailShare = useEmailShare({
    defaultSenderName: userInfo.senderName,
    defaultSenderCompany: userInfo.senderCompany
  })

  const handleShareLastQuery = () => {
    // Find the last user message and corresponding assistant response
    const userMessages = messages.filter(m => m.role === 'user')
    const assistantMessages = messages.filter(m => m.role === 'assistant')
    
    if (userMessages.length === 0 || assistantMessages.length === 0) return

    const lastUserMessage = userMessages[userMessages.length - 1]
    const lastAssistantMessage = assistantMessages[assistantMessages.length - 1]

    emailShare.shareRagQuery({
      query: lastUserMessage.content,
      answer: lastAssistantMessage.content,
      sources: lastAssistantMessage.sources,
      metadata: lastAssistantMessage.metadata,
    })
  }

  const handleShareConversation = () => {
    const conversationTitle = `Conversation - ${new Date().toLocaleDateString()}`
    
    emailShare.shareRagQuery({
      query: 'Full Conversation',
      answer: formatConversationForEmail(messages),
      title: conversationTitle,
    })
  }

  return (
    <div className="flex gap-2">
      <Button
        variant="outline"
        size="sm"
        onClick={handleShareLastQuery}
        disabled={messages.length < 2}
        className="flex items-center gap-2"
      >
        <Mail className="h-4 w-4" />
        Email Last Query
      </Button>

      <Button
        variant="outline"
        size="sm"
        onClick={handleShareConversation}
        disabled={messages.length === 0}
        className="flex items-center gap-2"
      >
        <Share className="h-4 w-4" />
        Email Conversation
      </Button>

      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </div>
  )
}

/**
 * Helper function to format conversation for email
 */
function formatConversationForEmail(messages: Array<{
  role: 'user' | 'assistant'
  content: string
  timestamp?: string
}>): string {
  let formatted = '# Conversation Summary\n\n'
  
  messages.forEach((message, index) => {
    const timestamp = message.timestamp 
      ? new Date(message.timestamp).toLocaleString()
      : `Message ${index + 1}`
    
    if (message.role === 'user') {
      formatted += `## 👤 User (${timestamp})\n${message.content}\n\n`
    } else {
      formatted += `## 🤖 Assistant (${timestamp})\n${message.content}\n\n`
    }
  })
  
  return formatted
}
