"use client"

import React, { useState, useEffect } from 'react'
import { Dialog, DialogContent, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Textarea } from '@/components/ui/textarea'
import { Label } from '@/components/ui/label'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Badge } from '@/components/ui/badge'
import { Separator } from '@/components/ui/separator'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { ScrollArea } from '@/components/ui/scroll-area'
import { 
  Mail, 
  Send, 
  Eye, 
  Copy, 
  Plus, 
  X, 
  User, 
  Building, 
  Calendar,
  Clock,
  FileText,
  Brain,
  Sparkles,
  AlertCircle
} from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

export interface EmailShareData {
  type: 'rag' | 'summary' | 'curation'
  title: string
  content: string
  query?: string
  sources?: Array<{ name: string; id: string }>
  metadata?: {
    provider?: string
    model?: string
    responseTime?: number
    confidence?: number
    documentCount?: number
    generatedAt?: string
    mode?: string
  }
  senderName?: string
  senderCompany?: string
}

interface EmailShareModalProps {
  isOpen: boolean
  onClose: () => void
  data: EmailShareData
  onSend: (emailData: {
    to: string[]
    cc: string[]
    bcc: string[]
    subject: string
    personalMessage: string
    includeMetadata: boolean
    useActualEmail?: boolean
  }) => Promise<any>
}

export function EmailShareModal({ isOpen, onClose, data, onSend }: EmailShareModalProps) {
  const { toast } = useToast()
  const [activeTab, setActiveTab] = useState('compose')
  const [isLoading, setIsLoading] = useState(false)
  
  // Form state
  const [toEmails, setToEmails] = useState<string[]>([])
  const [ccEmails, setCcEmails] = useState<string[]>([])
  const [bccEmails, setBccEmails] = useState<string[]>([])
  const [currentToEmail, setCurrentToEmail] = useState('')
  const [currentCcEmail, setCurrentCcEmail] = useState('')
  const [currentBccEmail, setCurrentBccEmail] = useState('')
  const [subject, setSubject] = useState('')
  const [personalMessage, setPersonalMessage] = useState('')
  const [includeMetadata, setIncludeMetadata] = useState(true)
  const [useActualEmail, setUseActualEmail] = useState(false)

  // Auto-generate subject line based on data type
  useEffect(() => {
    if (data) {
      const date = new Date().toLocaleDateString()
      let autoSubject = ''
      
      switch (data.type) {
        case 'rag':
          autoSubject = `SIXTHVAULT Analysis: ${data.title} - ${date}`
          break
        case 'summary':
          autoSubject = `Document Summary: ${data.title} - ${date}`
          break
        case 'curation':
          autoSubject = `AI Curation Report: ${data.title} - ${date}`
          break
      }
      
      setSubject(autoSubject)
    }
  }, [data])

  // Email validation
  const isValidEmail = (email: string) => {
    return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email)
  }

  // Add email to list
  const addEmail = (email: string, type: 'to' | 'cc' | 'bcc') => {
    if (!email.trim()) return
    
    if (!isValidEmail(email)) {
      toast({
        title: "Invalid Email",
        description: "Please enter a valid email address",
        variant: "destructive"
      })
      return
    }

    switch (type) {
      case 'to':
        if (!toEmails.includes(email)) {
          setToEmails([...toEmails, email])
          setCurrentToEmail('')
        }
        break
      case 'cc':
        if (!ccEmails.includes(email)) {
          setCcEmails([...ccEmails, email])
          setCurrentCcEmail('')
        }
        break
      case 'bcc':
        if (!bccEmails.includes(email)) {
          setBccEmails([...bccEmails, email])
          setCurrentBccEmail('')
        }
        break
    }
  }

  // Remove email from list
  const removeEmail = (email: string, type: 'to' | 'cc' | 'bcc') => {
    switch (type) {
      case 'to':
        setToEmails(toEmails.filter(e => e !== email))
        break
      case 'cc':
        setCcEmails(ccEmails.filter(e => e !== email))
        break
      case 'bcc':
        setBccEmails(bccEmails.filter(e => e !== email))
        break
    }
  }

  // Handle key press for email input
  const handleEmailKeyPress = (e: React.KeyboardEvent, email: string, type: 'to' | 'cc' | 'bcc') => {
    if (e.key === 'Enter' || e.key === ',') {
      e.preventDefault()
      addEmail(email, type)
    }
  }

  // Send email
  const handleSend = async () => {
    if (toEmails.length === 0) {
      toast({
        title: "Recipients Required",
        description: "Please add at least one recipient",
        variant: "destructive"
      })
      return
    }

    if (!subject.trim()) {
      toast({
        title: "Subject Required",
        description: "Please enter an email subject",
        variant: "destructive"
      })
      return
    }

    setIsLoading(true)
    
    try {
      await onSend({
        to: toEmails,
        cc: ccEmails,
        bcc: bccEmails,
        subject: subject.trim(),
        personalMessage: personalMessage.trim(),
        includeMetadata,
        useActualEmail
      })
      
      const emailType = useActualEmail ? "actual email" : "simulated email"
      toast({
        title: "Email Sent Successfully",
        description: `${emailType} sent to ${toEmails.length} recipient${toEmails.length > 1 ? 's' : ''}`,
      })
      
      onClose()
    } catch (error) {
      toast({
        title: "Failed to Send Email",
        description: error instanceof Error ? error.message : "Unknown error occurred",
        variant: "destructive"
      })
    } finally {
      setIsLoading(false)
    }
  }

  // Get icon for content type
  const getTypeIcon = () => {
    switch (data?.type) {
      case 'rag':
        return <Brain className="h-5 w-5 text-blue-500" />
      case 'summary':
        return <FileText className="h-5 w-5 text-green-500" />
      case 'curation':
        return <Sparkles className="h-5 w-5 text-purple-500" />
      default:
        return <Mail className="h-5 w-5" />
    }
  }

  // Clean chat history content for professional display
  const cleanChatHistoryContent = (content: string): string => {
    let cleaned = content
    
    // Remove user/assistant message headers with emojis
    cleaned = cleaned.replace(/👤 User \(Message \d+\)/g, '')
    cleaned = cleaned.replace(/🤖 Assistant \(Message \d+\)/g, '')
    
    // Remove markdown headers with the query title (# Explain Chai Badaldi in detail...)
    cleaned = cleaned.replace(/^#\s+.*\.\.\.\s*\n\n/gm, '')
    
    // Remove date sections (**Date:** DD/MM/YYYY)
    cleaned = cleaned.replace(/^\*\*Date:\*\*\s*\d{2}\/\d{2}\/\d{4}\s*\n\n/gm, '')
    
    // Remove empty markdown headers (## followed by empty line)
    cleaned = cleaned.replace(/^##\s*\n/gm, '')
    
    // Remove query repetitions and dates at the beginning
    // Pattern: "Query text...\nDate: DD/MM/YYYY\n\nQuery text ?\n\n"
    cleaned = cleaned.replace(/^.*\.\.\.\s*\nDate:\s*\d{2}\/\d{2}\/\d{4}\s*\n\n/, '')
    
    // Remove standalone dates at the beginning
    cleaned = cleaned.replace(/^Date:\s*\d{2}\/\d{2}\/\d{4}\s*\n\n/gm, '')
    
    // Remove query repetitions (lines ending with ... followed by same line with ?)
    cleaned = cleaned.replace(/^(.*)\.\.\.\s*\n\1\s*\?\s*\n/gm, '')
    
    // Remove empty lines created by removing headers
    cleaned = cleaned.replace(/\n\n\n+/g, '\n\n')
    
    // Clean up any remaining message formatting
    cleaned = cleaned.replace(/Message \d+:/g, '')
    
    // Remove leading/trailing whitespace
    cleaned = cleaned.trim()
    
    return cleaned
  }

  // Check if content is from chat history
  const isChatHistoryContent = (content: string): boolean => {
    return content.includes('👤 User') || content.includes('🤖 Assistant') || content.includes('Message ')
  }

  // Format content for preview
  const formatContentPreview = () => {
    if (!data) return ''
    
    let content = data.content
    
    // Clean up chat history formatting for professional appearance
    if (isChatHistoryContent(content)) {
      content = cleanChatHistoryContent(content)
    }
    
    return content
  }

  if (!data) return null

  return (
    <Dialog open={isOpen} onOpenChange={onClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            {getTypeIcon()}
            Share via Email
          </DialogTitle>
        </DialogHeader>

        <Tabs value={activeTab} onValueChange={setActiveTab} className="flex-1">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="compose" className="flex items-center gap-2">
              <Mail className="h-4 w-4" />
              Compose
            </TabsTrigger>
            <TabsTrigger value="preview" className="flex items-center gap-2">
              <Eye className="h-4 w-4" />
              Preview
            </TabsTrigger>
          </TabsList>

          <TabsContent value="compose" className="space-y-4 mt-4">
            <ScrollArea className="h-[500px] pr-4">
              {/* Recipients Section */}
              <div className="space-y-4">
                {/* To Field */}
                <div className="space-y-2">
                  <Label htmlFor="to-email" className="flex items-center gap-2">
                    <Mail className="h-4 w-4" />
                    To <span className="text-red-500">*</span>
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      id="to-email"
                      placeholder="Enter recipient email"
                      value={currentToEmail}
                      onChange={(e) => setCurrentToEmail(e.target.value)}
                      onKeyDown={(e) => handleEmailKeyPress(e, currentToEmail, 'to')}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => addEmail(currentToEmail, 'to')}
                      disabled={!currentToEmail.trim()}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  {toEmails.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {toEmails.map((email) => (
                        <Badge key={email} variant="secondary" className="flex items-center gap-1">
                          {email}
                          <X
                            className="h-3 w-3 cursor-pointer"
                            onClick={() => removeEmail(email, 'to')}
                          />
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>

                {/* CC Field */}
                <div className="space-y-2">
                  <Label htmlFor="cc-email" className="flex items-center gap-2">
                    <Copy className="h-4 w-4" />
                    CC (Optional)
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      id="cc-email"
                      placeholder="Enter CC email"
                      value={currentCcEmail}
                      onChange={(e) => setCurrentCcEmail(e.target.value)}
                      onKeyDown={(e) => handleEmailKeyPress(e, currentCcEmail, 'cc')}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => addEmail(currentCcEmail, 'cc')}
                      disabled={!currentCcEmail.trim()}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  {ccEmails.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {ccEmails.map((email) => (
                        <Badge key={email} variant="outline" className="flex items-center gap-1">
                          {email}
                          <X
                            className="h-3 w-3 cursor-pointer"
                            onClick={() => removeEmail(email, 'cc')}
                          />
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>

                {/* BCC Field */}
                <div className="space-y-2">
                  <Label htmlFor="bcc-email" className="flex items-center gap-2">
                    <Copy className="h-4 w-4" />
                    BCC (Optional)
                  </Label>
                  <div className="flex gap-2">
                    <Input
                      id="bcc-email"
                      placeholder="Enter BCC email"
                      value={currentBccEmail}
                      onChange={(e) => setCurrentBccEmail(e.target.value)}
                      onKeyDown={(e) => handleEmailKeyPress(e, currentBccEmail, 'bcc')}
                    />
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      onClick={() => addEmail(currentBccEmail, 'bcc')}
                      disabled={!currentBccEmail.trim()}
                    >
                      <Plus className="h-4 w-4" />
                    </Button>
                  </div>
                  {bccEmails.length > 0 && (
                    <div className="flex flex-wrap gap-2">
                      {bccEmails.map((email) => (
                        <Badge key={email} variant="outline" className="flex items-center gap-1">
                          {email}
                          <X
                            className="h-3 w-3 cursor-pointer"
                            onClick={() => removeEmail(email, 'bcc')}
                          />
                        </Badge>
                      ))}
                    </div>
                  )}
                </div>

                <Separator />

                {/* Subject Field */}
                <div className="space-y-2">
                  <Label htmlFor="subject">
                    Subject <span className="text-red-500">*</span>
                  </Label>
                  <Input
                    id="subject"
                    placeholder="Enter email subject"
                    value={subject}
                    onChange={(e) => setSubject(e.target.value)}
                  />
                </div>

                {/* Personal Message */}
                <div className="space-y-2">
                  <Label htmlFor="message">Personal Message (Optional)</Label>
                  <Textarea
                    id="message"
                    placeholder="Add a personal message to include before the content..."
                    value={personalMessage}
                    onChange={(e) => setPersonalMessage(e.target.value)}
                    rows={3}
                  />
                </div>

                {/* Options - Only show for non-chat history content */}
                {!data.title.toLowerCase().includes('chat history') && (
                  <div className="space-y-3">
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="include-metadata"
                        checked={includeMetadata}
                        onChange={(e) => setIncludeMetadata(e.target.checked)}
                        className="rounded"
                      />
                      <Label htmlFor="include-metadata" className="text-sm">
                        Include technical metadata (processing time, confidence, etc.)
                      </Label>
                    </div>
                    
                    <div className="flex items-center space-x-2">
                      <input
                        type="checkbox"
                        id="use-actual-email"
                        checked={useActualEmail}
                        onChange={(e) => setUseActualEmail(e.target.checked)}
                        className="rounded"
                      />
                      <Label htmlFor="use-actual-email" className="text-sm">
                        Send actual email (via Resend service)
                      </Label>
                    </div>
                    
                    {!useActualEmail && (
                      <div className="flex items-start space-x-2 p-3 bg-amber-50 border border-amber-200 rounded-lg">
                        <AlertCircle className="h-4 w-4 text-amber-600 mt-0.5 flex-shrink-0" />
                        <div className="text-sm text-amber-800">
                          <strong>Simulation Mode:</strong> Email will be logged to console only. 
                          Check the option above to send actual emails.
                        </div>
                      </div>
                    )}
                    
                    {useActualEmail && (
                      <div className="flex items-start space-x-2 p-3 bg-green-50 border border-green-200 rounded-lg">
                        <Mail className="h-4 w-4 text-green-600 mt-0.5 flex-shrink-0" />
                        <div className="text-sm text-green-800">
                          <strong>Actual Email:</strong> Email will be sent via Resend service to the specified recipients.
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            </ScrollArea>
          </TabsContent>

          <TabsContent value="preview" className="mt-4">
            <Card>
              <CardHeader>
                <CardTitle className="text-lg">Email Preview</CardTitle>
              </CardHeader>
              <CardContent>
                <ScrollArea className="h-[500px]">
                  <div className="space-y-4">
                    {/* Email Headers */}
                    <div className="space-y-2 text-sm">
                      <div><strong>To:</strong> {toEmails.join(', ') || 'No recipients'}</div>
                      {ccEmails.length > 0 && (
                        <div><strong>CC:</strong> {ccEmails.join(', ')}</div>
                      )}
                      <div><strong>Subject:</strong> {subject || 'No subject'}</div>
                    </div>

                    <Separator />

                    {/* Email Content Preview */}
                    <div className="prose prose-sm max-w-none">
                      {personalMessage && (
                        <div className="bg-blue-50 p-4 rounded-lg mb-4">
                          <p className="text-blue-800 italic">{personalMessage}</p>
                        </div>
                      )}

                      <div className="bg-white border rounded-lg p-6">
                        {/* Header */}
                        <div className="text-center mb-6">
                          <h1 className="text-2xl font-bold text-gray-800">SIXTHVAULT</h1>
                          <p className="text-gray-600">AI-Powered Document Intelligence</p>
                        </div>

                        {/* Content */}
                        <div className="space-y-4">
                          {/* Only show title and query for non-chat history content */}
                          {!data.title.toLowerCase().includes('chat history') && (
                            <>
                              <div className="flex items-center gap-2 mb-4">
                                {getTypeIcon()}
                                <h2 className="text-xl font-semibold">{data.title}</h2>
                              </div>

                              {data.type === 'rag' && data.query && (
                                <div className="bg-gray-50 p-4 rounded-lg">
                                  <h3 className="font-semibold text-gray-700 mb-2">Query:</h3>
                                  <p className="text-gray-800">{data.query}</p>
                                </div>
                              )}
                            </>
                          )}

                          <div className="whitespace-pre-wrap">{formatContentPreview()}</div>

                          {data.sources && data.sources.length > 0 && (
                            <div className="mt-6">
                              <h3 className="font-semibold text-gray-700 mb-2">Source Documents:</h3>
                              <ul className="list-disc list-inside space-y-1">
                                {data.sources.map((source, index) => (
                                  <li key={index} className="text-gray-600">{source.name}</li>
                                ))}
                              </ul>
                            </div>
                          )}

                          {includeMetadata && data.metadata && !data.title.toLowerCase().includes('chat history') && (
                            <div className="mt-6 bg-gray-50 p-4 rounded-lg">
                              <h3 className="font-semibold text-gray-700 mb-2">Technical Details:</h3>
                              <div className="grid grid-cols-2 gap-2 text-sm text-gray-600">
                                {data.metadata.provider && (
                                  <div>Provider: {data.metadata.provider}</div>
                                )}
                                {data.metadata.model && (
                                  <div>Model: {data.metadata.model}</div>
                                )}
                                {data.metadata.responseTime && (
                                  <div>Response Time: {data.metadata.responseTime}ms</div>
                                )}
                                {data.metadata.confidence && (
                                  <div>Confidence: {(data.metadata.confidence * 100).toFixed(1)}%</div>
                                )}
                                {data.metadata.generatedAt && (
                                  <div>Generated: {new Date(data.metadata.generatedAt).toLocaleString()}</div>
                                )}
                              </div>
                            </div>
                          )}
                        </div>

                        {/* Footer */}
                        <div className="mt-8 pt-6 border-t text-center text-sm text-gray-500">
                          <div className="space-y-2">
                            {data.senderName && (
                              <div className="flex items-center justify-center gap-2">
                                <User className="h-4 w-4" />
                                <span>Generated by: {data.senderName}</span>
                              </div>
                            )}
                            {data.senderCompany && (
                              <div className="flex items-center justify-center gap-2">
                                <Building className="h-4 w-4" />
                                <span>{data.senderCompany}</span>
                              </div>
                            )}
                            <div className="flex items-center justify-center gap-2">
                              <Calendar className="h-4 w-4" />
                              <span>Generated on: {new Date().toLocaleDateString()}</span>
                            </div>
                            <div className="text-xs text-gray-400 mt-4">
                              Powered by SIXTHVAULT - AI-Powered Document Intelligence
                            </div>
                          </div>
                        </div>
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>

        {/* Action Buttons */}
        <div className="flex justify-between pt-4 border-t">
          <Button variant="outline" onClick={onClose} disabled={isLoading}>
            Cancel
          </Button>
          <div className="flex gap-2">
            <Button
              variant="outline"
              onClick={() => setActiveTab('preview')}
              disabled={isLoading}
            >
              <Eye className="h-4 w-4 mr-2" />
              Preview
            </Button>
            <Button onClick={handleSend} disabled={isLoading || toEmails.length === 0}>
              {isLoading ? (
                <>
                  <Clock className="h-4 w-4 mr-2 animate-spin" />
                  Sending...
                </>
              ) : (
                <>
                  <Send className="h-4 w-4 mr-2" />
                  Send Email
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}
