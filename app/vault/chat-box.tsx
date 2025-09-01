"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Badge } from "@/components/ui/badge"
import { Send, Loader2, Clock, Calendar, Share2, Download, MessageSquare } from "lucide-react"
import { ragApiClient, type Message, type Conversation } from "@/lib/api-client"
import { formatContentWithMath } from "@/components/ui/math-renderer"

interface ChatMessage {
  id: number
  type: "user" | "ai"
  content: string
  themes?: string[]
  language?: string
  timestamp?: string
  conversationId?: string
}

interface ChatBoxProps {
  conversationId?: string
  onConversationCreated?: (conversationId: string) => void
  initialMessages?: ChatMessage[]
}

export default function ChatBox({ 
  conversationId, 
  onConversationCreated, 
  initialMessages = [] 
}: ChatBoxProps) {
  const [inputMessage, setInputMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>(initialMessages)
  const [currentConversationId, setCurrentConversationId] = useState<string | undefined>(conversationId)
  const [error, setError] = useState("")

  // Update messages when initialMessages prop changes
  useEffect(() => {
    setChatMessages(initialMessages)
  }, [initialMessages])

  // Update conversation ID when prop changes
  useEffect(() => {
    setCurrentConversationId(conversationId)
  }, [conversationId])

  const handleSendMessage = async () => {
    if (!inputMessage.trim()) return
    setError("")

    const timestamp = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })

    const userMessage = {
      id: Date.now(),
      type: "user" as const,
      content: inputMessage,
      timestamp,
      conversationId: currentConversationId,
    }

    setChatMessages((prev) => [...prev, userMessage])
    const currentMessage = inputMessage
    setInputMessage("")
    setIsLoading(true)

    try {
      // Use RAG backend with conversation support
      const queryResponse = await ragApiClient.queryDocumentsWithConversation(currentMessage, {
        hybrid: true, // Default to hybrid mode
        maxContext: false, // Default to standard context
        conversationId: currentConversationId,
        saveConversation: true // Always save conversations
      })

      // Update current conversation ID if a new one was created
      if (queryResponse.conversation_id && !currentConversationId) {
        setCurrentConversationId(queryResponse.conversation_id)
        onConversationCreated?.(queryResponse.conversation_id)
      }

      const aiMessage = {
        id: Date.now() + 1,
        type: "ai" as const,
        content: queryResponse.answer,
        themes: ["AI Response", "RAG Backend", "Document Analysis"],
        language: "English",
        timestamp,
        conversationId: queryResponse.conversation_id || currentConversationId,
      }

      setChatMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      console.error("Chat error:", error)
      setError(error instanceof Error ? error.message : "Unknown error")
      
      // Add error message to chat
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

  const startNewConversation = () => {
    setChatMessages([])
    setCurrentConversationId(undefined)
    setError("")
  }

  // Function to format content with better typography and math support
  const formatContent = (content: string) => {
    // Use the enhanced math-enabled formatter
    return formatContentWithMath(content)
  }

  return (
    <div className="flex flex-col h-full">
      <div className="p-6 bg-white border-b border-gray-200 shadow-sm">
        <div className="relative">
          <Input
            value={inputMessage}
            onChange={(e) => setInputMessage(e.target.value)}
            onKeyPress={(e) => e.key === "Enter" && handleSendMessage()}
            placeholder="Ask questions about consumer trends..."
            className="w-full h-12 pr-12 text-lg bg-gray-100 border-0 rounded-lg focus:ring-2 focus:ring-blue-500 transition-all"
            disabled={isLoading}
          />
          <Button
            size="sm"
            onClick={handleSendMessage}
            disabled={isLoading || !inputMessage.trim()}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-blue-600 hover:bg-blue-700 disabled:opacity-50 rounded-full transition-all duration-200 hover:scale-105"
          >
            {isLoading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
          </Button>
        </div>
      </div>

      {error && (
        <div className="p-4 bg-red-50 border-b border-red-200">
          <p className="text-red-600 text-sm">Error: {error}</p>
        </div>
      )}

      <div className="flex-1 bg-gray-50 p-6 overflow-y-auto">
        {chatMessages.length === 0 && !isLoading && (
          <div className="flex items-center justify-center h-full text-gray-500">
            <div className="text-center">
              <h3 className="text-lg font-medium mb-2">Welcome to SIXTHVAULT RAG System</h3>
              <p className="mb-4">Ask questions about consumer trends or documents.</p>
              <p className="text-sm">Powered by Groq AI</p>
            </div>
          </div>
        )}

        <div className="space-y-6">
          {chatMessages.map((message) => (
            <div key={message.id} className={`${message.type === "user" ? "text-right" : "text-left"}`}>
              <div
                className={`inline-block max-w-3xl rounded-lg ${
                  message.type === "user"
                    ? "bg-blue-600 text-white p-4 shadow-md"
                    : "bg-white border border-gray-200 text-gray-800 shadow-md p-5"
                }`}
              >
                <div className="flex justify-between items-center mb-2">
                  <div className="flex items-center">
                    {message.type === "ai" && (
                      <Badge variant="outline" className="mr-2 bg-blue-50">
                        AI
                      </Badge>
                    )}
                  </div>
                  <div className="flex items-center text-xs text-gray-500">
                    <Clock className="w-3 h-3 mr-1" />
                    {message.timestamp || "Just now"}
                  </div>
                </div>

                {message.type === "user" ? (
                  <p className="text-base md:text-lg">{message.content}</p>
                ) : (
                  <div
                    className="prose prose-sm md:prose-base max-w-none"
                    dangerouslySetInnerHTML={{
                      __html: formatContent(message.content),
                    }}
                  />
                )}

                {message.type === "ai" && (message.themes || message.language) && (
                  <div className="mt-4 pt-4 border-t border-gray-200">
                    <div className="flex flex-wrap gap-2 mb-3">
                      {message.themes &&
                        message.themes.map((theme, index) => (
                          <Badge key={index} variant="secondary" className="text-xs">
                            {theme}
                          </Badge>
                        ))}
                    </div>

                    <div className="flex flex-wrap justify-between items-center">
                      <div className="flex items-center text-xs text-gray-600">
                        <Calendar className="w-3 h-3 mr-1" />
                        <span>{new Date().toLocaleDateString()}</span>
                      </div>

                      <div className="flex space-x-2">
                        <Button variant="outline" size="sm" className="h-7 text-xs">
                          <Share2 className="w-3 h-3 mr-1" /> Share
                        </Button>
                        <Button variant="outline" size="sm" className="h-7 text-xs">
                          <Download className="w-3 h-3 mr-1" /> Export
                        </Button>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {isLoading && (
            <div className="text-left mb-4">
              <div className="inline-block bg-white border border-gray-200 p-5 rounded-lg shadow-md">
                <div className="flex items-center space-x-3">
                  <Loader2 className="w-5 h-5 animate-spin text-blue-600" />
                  <span className="text-gray-600 text-lg">Generating response with Groq AI...</span>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
