import { EmailShareData } from '@/components/email-share-modal'

export interface EmailRequest {
  to: string[]
  cc: string[]
  bcc: string[]
  subject: string
  personalMessage: string
  includeMetadata: boolean
  data: EmailShareData
}

export interface EmailResponse {
  success: boolean
  messageId?: string
  message: string
  simulated?: boolean
}

class EmailShareService {
  private baseUrl = '/api/send-email'

  /**
   * Send email with formatted content
   */
  async sendEmail(request: EmailRequest, useActualEmail: boolean = true): Promise<EmailResponse> {
    try {
      const htmlContent = this.generateHtmlContent(request)
      const textContent = this.generateTextContent(request)

      // Get current user info for sender
      let senderEmail = null
      let senderName = null
      
      if (useActualEmail && typeof window !== 'undefined') {
        try {
          // Get auth token from cookie
          const authToken = document.cookie
            .split('; ')
            .find(row => row.startsWith('auth-token='))
            ?.split('=')[1]
          
          if (authToken) {
            // Get user info from backend
            const userResponse = await fetch(`${process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'}/auth/me`, {
              headers: {
                'Authorization': `Bearer ${decodeURIComponent(authToken)}`,
              },
            })
            
            if (userResponse.ok) {
              const userData = await userResponse.json()
              senderEmail = userData.email
              senderName = `${userData.first_name} ${userData.last_name}`.trim()
            }
          }
        } catch (error) {
          console.log('Could not get user info for sender, using default:', error)
        }
      }

      const response = await fetch(this.baseUrl, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          to: request.to,
          cc: request.cc.length > 0 ? request.cc : undefined,
          bcc: request.bcc.length > 0 ? request.bcc : undefined,
          subject: request.subject,
          html: htmlContent,
          text: textContent,
          useBackendService: useActualEmail, // This will trigger actual email sending
          senderEmail,
          senderName,
        }),
      })

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`)
      }

      const result = await response.json()
      return {
        success: result.success || false,
        messageId: result.messageId,
        message: result.message || 'Email sent successfully',
        simulated: result.simulated || false,
      }
    } catch (error) {
      console.error('Failed to send email:', error)
      throw new Error(error instanceof Error ? error.message : 'Failed to send email')
    }
  }

  /**
   * Send actual email (not simulated) - convenience method
   */
  async sendActualEmail(request: EmailRequest): Promise<EmailResponse> {
    return this.sendEmail(request, true)
  }

  /**
   * Generate HTML content for email
   */
  private generateHtmlContent(request: EmailRequest): string {
    const { data, personalMessage, includeMetadata } = request
    const currentDate = new Date().toLocaleDateString()
    const currentTime = new Date().toLocaleTimeString()

    return `
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>${request.subject}</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 800px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8fafc;
        }
        .email-container {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            margin: 0;
            font-size: 32px;
            font-weight: bold;
            letter-spacing: 1px;
        }
        .header p {
            margin: 10px 0 0 0;
            font-size: 16px;
            opacity: 0.9;
        }
        .content {
            padding: 30px;
        }
        .personal-message {
            background-color: #eff6ff;
            border-left: 4px solid #3b82f6;
            padding: 20px;
            margin-bottom: 30px;
            border-radius: 0 8px 8px 0;
        }
        .personal-message p {
            margin: 0;
            font-style: italic;
            color: #1e40af;
        }
        .content-type {
            margin-bottom: 30px;
            padding: 20px 0;
            border-bottom: 2px solid #e5e7eb;
        }
        .content-title {
            font-size: 28px;
            font-weight: 600;
            color: #1f2937;
            margin: 0 0 8px 0;
            line-height: 1.2;
        }
        .content-type-label {
            font-size: 14px;
            font-weight: 500;
            color: #6b7280;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .query-section {
            background-color: #f9fafb;
            border: 1px solid #e5e7eb;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
        }
        .query-label {
            font-weight: bold;
            color: #374151;
            margin-bottom: 10px;
            font-size: 16px;
        }
        .query-text {
            color: #1f2937;
            font-size: 16px;
            line-height: 1.5;
        }
        .main-content {
            margin: 30px 0;
            line-height: 1.7;
            color: #374151;
        }
        .main-content h1, .main-content h2, .main-content h3 {
            color: #1f2937;
            margin-top: 30px;
            margin-bottom: 15px;
        }
        .main-content h1 { font-size: 28px; }
        .main-content h2 { font-size: 24px; }
        .main-content h3 { font-size: 20px; }
        .main-content ul, .main-content ol {
            padding-left: 25px;
            margin: 15px 0;
        }
        .main-content li {
            margin-bottom: 8px;
        }
        .main-content p {
            margin: 15px 0;
        }
        .sources-section {
            background-color: #f0f9ff;
            border: 1px solid #bae6fd;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }
        .sources-title {
            font-weight: bold;
            color: #0c4a6e;
            margin-bottom: 15px;
            font-size: 18px;
        }
        .sources-list {
            list-style: none;
            padding: 0;
            margin: 0;
        }
        .sources-list li {
            padding: 8px 0;
            border-bottom: 1px solid #e0f2fe;
            color: #0369a1;
        }
        .sources-list li:last-child {
            border-bottom: none;
        }
        .sources-list li:before {
            content: "•";
            margin-right: 10px;
            font-weight: bold;
            color: #0369a1;
        }
        .metadata-section {
            background-color: #fafafa;
            border: 1px solid #e5e5e5;
            border-radius: 8px;
            padding: 20px;
            margin: 30px 0;
        }
        .metadata-title {
            font-weight: bold;
            color: #525252;
            margin-bottom: 15px;
            font-size: 16px;
        }
        .metadata-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 15px;
        }
        .metadata-item {
            display: flex;
            justify-content: space-between;
            padding: 8px 0;
            border-bottom: 1px solid #e5e5e5;
        }
        .metadata-item:last-child {
            border-bottom: none;
        }
        .metadata-label {
            font-weight: 500;
            color: #6b7280;
        }
        .metadata-value {
            color: #374151;
            font-weight: 500;
        }
        .footer {
            background-color: #f8fafc;
            padding: 30px;
            text-align: center;
            border-top: 1px solid #e5e7eb;
        }
        .footer-content {
            max-width: 500px;
            margin: 0 auto;
        }
        .sender-info {
            margin-bottom: 20px;
        }
        .sender-item {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 8px;
            margin: 8px 0;
            color: #6b7280;
            font-size: 14px;
        }
        .sender-icon {
            width: 16px;
            height: 16px;
        }
        .company-branding {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid #d1d5db;
        }
        .company-logo {
            font-size: 12px;
            color: #9ca3af;
            margin-bottom: 10px;
        }
        .company-tagline {
            font-size: 11px;
            color: #6b7280;
        }
        @media (max-width: 600px) {
            body { padding: 10px; }
            .content { padding: 20px; }
            .header { padding: 20px; }
            .header h1 { font-size: 24px; }
            .content-title { font-size: 20px; }
            .metadata-grid { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="email-container">
        <!-- Header -->
        <div class="header">
            <h1>SIXTHVAULT</h1>
            <p>AI-Powered Document Intelligence</p>
        </div>

        <!-- Content -->
        <div class="content">
            ${personalMessage ? `
            <div class="personal-message">
                <p>${this.escapeHtml(personalMessage)}</p>
            </div>
            ` : ''}

            <!-- Only show content type header for non-chat history content -->
            ${!data.title.toLowerCase().includes('chat history') ? `
            <div class="content-type">
                <h2 class="content-title">${this.escapeHtml(data.title)}</h2>
                <div class="content-type-label">${this.getTypeLabel(data.type)}</div>
            </div>

            ${data.type === 'rag' && data.query ? `
            <div class="query-section">
                <div class="query-label">Query:</div>
                <div class="query-text">${this.escapeHtml(data.query)}</div>
            </div>
            ` : ''}
            ` : ''}

            <!-- Main Content -->
            <div class="main-content">
                ${this.formatContentForHtml(data.content)}
            </div>

            ${data.sources && data.sources.length > 0 ? `
            <div class="sources-section">
                <div class="sources-title">Source Documents</div>
                <ul class="sources-list">
                    ${data.sources.map(source => `
                        <li>${this.escapeHtml(source.name)}</li>
                    `).join('')}
                </ul>
            </div>
            ` : ''}

            ${includeMetadata && data.metadata && !data.title.toLowerCase().includes('chat history') ? `
            <div class="metadata-section">
                <div class="metadata-title">Technical Details</div>
                <div class="metadata-grid">
                    ${data.metadata.provider ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Provider:</span>
                        <span class="metadata-value">${this.escapeHtml(data.metadata.provider)}</span>
                    </div>
                    ` : ''}
                    ${data.metadata.model ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Model:</span>
                        <span class="metadata-value">${this.escapeHtml(data.metadata.model)}</span>
                    </div>
                    ` : ''}
                    ${data.metadata.responseTime ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Response Time:</span>
                        <span class="metadata-value">${data.metadata.responseTime}ms</span>
                    </div>
                    ` : ''}
                    ${data.metadata.confidence ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Confidence:</span>
                        <span class="metadata-value">${(data.metadata.confidence * 100).toFixed(1)}%</span>
                    </div>
                    ` : ''}
                    ${data.metadata.documentCount ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Documents:</span>
                        <span class="metadata-value">${data.metadata.documentCount}</span>
                    </div>
                    ` : ''}
                    ${data.metadata.mode ? `
                    <div class="metadata-item">
                        <span class="metadata-label">Mode:</span>
                        <span class="metadata-value">${this.escapeHtml(data.metadata.mode)}</span>
                    </div>
                    ` : ''}
                    <div class="metadata-item">
                        <span class="metadata-label">Generated:</span>
                        <span class="metadata-value">${currentDate} at ${currentTime}</span>
                    </div>
                </div>
            </div>
            ` : ''}
        </div>

        <!-- Footer -->
        <div class="footer">
            <div class="footer-content">
                <div class="sender-info">
                    ${data.senderName ? `
                    <div class="sender-item">
                        <span>Generated by: ${this.escapeHtml(data.senderName)}</span>
                    </div>
                    ` : ''}
                    ${data.senderCompany ? `
                    <div class="sender-item">
                        <span>${this.escapeHtml(data.senderCompany)}</span>
                    </div>
                    ` : ''}
                    ${!data.title.toLowerCase().includes('chat history') ? `
                    <div class="sender-item">
                        <span>Generated on: ${currentDate}</span>
                    </div>
                    ` : ''}
                </div>

                <div class="company-branding">
                    <div class="company-logo">
                        Powered by SIXTHVAULT
                    </div>
                    <div class="company-tagline">
                        AI-Powered Document Intelligence Platform
                    </div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>
    `
  }

  /**
   * Generate plain text content for email
   */
  private generateTextContent(request: EmailRequest): string {
    const { data, personalMessage, includeMetadata } = request
    const currentDate = new Date().toLocaleDateString()
    const currentTime = new Date().toLocaleTimeString()

    let content = `SIXTHVAULT - AI-Powered Document Intelligence\n`
    content += `${'='.repeat(50)}\n\n`

    if (personalMessage) {
      content += `Personal Message:\n${personalMessage}\n\n`
    }

    // Only show title and query for non-chat history content
    if (!data.title.toLowerCase().includes('chat history')) {
      content += `${this.getTypeLabel(data.type)}: ${data.title}\n`
      content += `${'-'.repeat(data.title.length + this.getTypeLabel(data.type).length + 2)}\n\n`

      if (data.type === 'rag' && data.query) {
        content += `Query: ${data.query}\n\n`
      }
    }

    content += `${this.stripHtmlTags(data.content)}\n\n`

    if (data.sources && data.sources.length > 0) {
      content += `Source Documents:\n`
      data.sources.forEach(source => {
        content += `• ${source.name}\n`
      })
      content += `\n`
    }

    if (includeMetadata && data.metadata && !data.title.toLowerCase().includes('chat history')) {
      content += `Technical Details:\n`
      if (data.metadata.provider) content += `Provider: ${data.metadata.provider}\n`
      if (data.metadata.model) content += `Model: ${data.metadata.model}\n`
      if (data.metadata.responseTime) content += `Response Time: ${data.metadata.responseTime}ms\n`
      if (data.metadata.confidence) content += `Confidence: ${(data.metadata.confidence * 100).toFixed(1)}%\n`
      if (data.metadata.documentCount) content += `Documents: ${data.metadata.documentCount}\n`
      if (data.metadata.mode) content += `Mode: ${data.metadata.mode}\n`
      content += `Generated: ${currentDate} at ${currentTime}\n\n`
    }

    content += `${'='.repeat(50)}\n`
    if (data.senderName) content += `Generated by: ${data.senderName}\n`
    if (data.senderCompany) content += `Company: ${data.senderCompany}\n`
    if (!data.title.toLowerCase().includes('chat history')) {
      content += `Generated on: ${currentDate}\n`
    }
    content += `\nPowered by SIXTHVAULT - AI-Powered Document Intelligence Platform`

    return content
  }

  /**
   * Get type symbol for HTML display
   */
  private getTypeSymbol(type: string): string {
    switch (type) {
      case 'rag': return '🧠'
      case 'summary': return '📄'
      case 'curation': return '✨'
      default: return '📧'
    }
  }

  /**
   * Get type label for text display
   */
  private getTypeLabel(type: string): string {
    switch (type) {
      case 'rag': return 'RAG Analysis'
      case 'summary': return 'Document Summary'
      case 'curation': return 'AI Curation'
      default: return 'Content'
    }
  }

  /**
   * Escape HTML characters
   */
  private escapeHtml(text: string): string {
    const div = document.createElement('div')
    div.textContent = text
    return div.innerHTML
  }

  /**
   * Strip HTML tags from text
   */
  private stripHtmlTags(html: string): string {
    const div = document.createElement('div')
    div.innerHTML = html
    return div.textContent || div.innerText || ''
  }

  /**
   * Format content for HTML display (convert markdown-like formatting)
   */
  private formatContentForHtml(content: string): string {
    let formatted = this.escapeHtml(content)
    
    // Clean up chat history formatting for professional appearance
    if (this.isChatHistoryContent(formatted)) {
      formatted = this.cleanChatHistoryContent(formatted)
    }
    
    // Convert markdown-style formatting to HTML
    formatted = formatted
      // Headers
      .replace(/^### (.*$)/gm, '<h3>$1</h3>')
      .replace(/^## (.*$)/gm, '<h2>$1</h2>')
      .replace(/^# (.*$)/gm, '<h1>$1</h1>')
      // Bold
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      // Italic
      .replace(/\*(.*?)\*/g, '<em>$1</em>')
      // Line breaks
      .replace(/\n\n/g, '</p><p>')
      .replace(/\n/g, '<br>')
      // Bullet points
      .replace(/^- (.*$)/gm, '<li>$1</li>')
      // Wrap in paragraphs
      .replace(/^(?!<[h|l])/gm, '<p>')
      .replace(/(?<!>)$/gm, '</p>')
      // Clean up empty paragraphs
      .replace(/<p><\/p>/g, '')
      // Wrap lists
      .replace(/(<li>.*<\/li>)/g, '<ul>$1</ul>')

    return formatted
  }

  /**
   * Check if content is from chat history
   */
  private isChatHistoryContent(content: string): boolean {
    return content.includes('👤 User') || content.includes('🤖 Assistant') || content.includes('Message ')
  }

  /**
   * Clean chat history content for professional display
   */
  private cleanChatHistoryContent(content: string): string {
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
}

export const emailShareService = new EmailShareService()
