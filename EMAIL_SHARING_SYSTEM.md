# Email Sharing System for SIXTHVAULT

This document describes the comprehensive email sharing system that allows users to send RAG queries, summaries, and AI curations via email with professional formatting.

## Overview

The email sharing system consists of several components that work together to provide a seamless email sharing experience:

- **Email Composition Modal**: Professional email interface with To/CC/BCC fields
- **Email Templates**: Beautiful HTML and text email templates
- **Content Formatting**: Automatic formatting of AI-generated content
- **Integration Hooks**: Easy-to-use React hooks for integration
- **Professional Branding**: SIXTHVAULT branding and sender information

## Components

### 1. EmailShareModal (`components/email-share-modal.tsx`)

The main modal component that provides the email composition interface.

**Features:**
- To/CC/BCC recipient management
- Auto-generated subject lines
- Personal message support
- Live email preview
- Technical metadata toggle
- Professional email formatting

**Usage:**
```tsx
import { EmailShareModal } from '@/components/email-share-modal'

<EmailShareModal
  isOpen={isModalOpen}
  onClose={closeModal}
  data={emailData}
  onSend={handleSendEmail}
/>
```

### 2. Email Service (`lib/email-share-service.ts`)

Handles email formatting and delivery.

**Features:**
- Professional HTML email templates
- Plain text fallback
- Responsive design
- SIXTHVAULT branding
- Content type-specific formatting

### 3. Email Hooks (`hooks/use-email-share.ts`)

React hooks for easy integration with existing components.

**Main Hook: `useEmailShare`**
```tsx
const emailShare = useEmailShare({
  defaultSenderName: "John Doe",
  defaultSenderCompany: "Acme Corp"
})

// Share RAG query
emailShare.shareRagQuery({
  query: "What are the key insights?",
  answer: "Based on the analysis...",
  sources: [{ name: "document.pdf", id: "123" }],
  metadata: { provider: "openai", model: "gpt-4" }
})
```

**Conversation Hook: `useConversationEmailShare`**
```tsx
const conversationShare = useConversationEmailShare()

// Share single message
conversationShare.shareConversationMessage({
  userMessage: "What is AI?",
  assistantMessage: "AI is...",
  sources: [...],
  metadata: {...}
})

// Share entire conversation
conversationShare.shareEntireConversation({
  title: "AI Discussion",
  messages: [...],
  metadata: {...}
})
```

### 4. Integration Components (`components/email-share-integration.tsx`)

Pre-built components for common use cases.

**Available Components:**
- `RagQueryEmailShare`: For RAG query results
- `SummaryEmailShare`: For document summaries
- `CurationEmailShare`: For AI curations
- `GenericEmailShareButton`: Universal email share button
- `VaultChatEmailShare`: For chat conversations

## Integration Guide

### Step 1: Add Email Share Button

```tsx
import { RagQueryEmailShare } from '@/components/email-share-integration'

function QueryResult({ query, answer, sources, metadata }) {
  return (
    <div>
      <div>{answer}</div>
      <RagQueryEmailShare
        query={query}
        answer={answer}
        sources={sources}
        metadata={metadata}
      />
    </div>
  )
}
```

### Step 2: Use Generic Button for Custom Cases

```tsx
import { GenericEmailShareButton } from '@/components/email-share-integration'

<GenericEmailShareButton
  data={{
    type: 'rag',
    title: 'Market Analysis',
    content: 'Analysis results...',
    query: 'What are the market trends?',
    sources: [...],
    metadata: {...}
  }}
  buttonText="Email Analysis"
  variant="default"
  size="sm"
/>
```

### Step 3: Custom Integration with Hooks

```tsx
import { useEmailShare } from '@/hooks/use-email-share'
import { EmailShareModal } from '@/components/email-share-modal'

function CustomComponent() {
  const emailShare = useEmailShare({
    defaultSenderName: "Jane Smith",
    defaultSenderCompany: "Tech Corp"
  })

  const handleShare = () => {
    emailShare.shareSummary({
      title: "Q4 Report Summary",
      content: "Key findings...",
      sources: [...],
      metadata: { documentCount: 5 }
    })
  }

  return (
    <>
      <button onClick={handleShare}>Share Summary</button>
      
      <EmailShareModal
        isOpen={emailShare.isModalOpen}
        onClose={emailShare.closeModal}
        data={emailShare.currentData!}
        onSend={emailShare.handleSendEmail}
      />
    </>
  )
}
```

## Email Templates

### HTML Template Features

- **Responsive Design**: Works on desktop and mobile
- **Professional Styling**: Clean, modern design
- **SIXTHVAULT Branding**: Consistent brand identity
- **Content Sections**:
  - Header with logo and tagline
  - Personal message (if provided)
  - Content type indicator
  - Query section (for RAG queries)
  - Main content with formatting
  - Source documents list
  - Technical metadata (optional)
  - Footer with sender info and branding

### Text Template Features

- **Plain Text Fallback**: For email clients that don't support HTML
- **Structured Format**: Clear sections and formatting
- **All Content Included**: Same information as HTML version

## Content Types

### RAG Queries
- Query text
- AI-generated answer
- Source documents
- Technical metadata (provider, model, response time, confidence)

### Summaries
- Summary title and content
- Source documents analyzed
- Summary type and metadata

### AI Curations
- Curation title and content
- Keywords and topics
- Generation metadata

## Email Features

### Recipients Management
- **To**: Primary recipients (required)
- **CC**: Carbon copy recipients (optional)
- **BCC**: Blind carbon copy recipients (optional)
- Email validation and duplicate prevention
- Easy add/remove with badges

### Subject Lines
Auto-generated based on content type:
- RAG: "SIXTHVAULT Analysis: [Title] - [Date]"
- Summary: "Document Summary: [Title] - [Date]"
- Curation: "AI Curation Report: [Title] - [Date]"

### Personal Messages
- Optional personal message from sender
- Appears prominently at the top of the email
- Supports multi-line text

### Technical Metadata
Optional inclusion of:
- AI provider and model used
- Processing time and confidence scores
- Document count and generation details
- Timestamps and other technical info

## Sender Information

### Default Behavior
- Uses sender name and company from hook options
- Falls back to generic "Generated by SIXTHVAULT" if not provided

### Integration with Auth System
Update `getUserInfoForEmail()` in `hooks/use-email-share.ts`:

```tsx
export function getUserInfoForEmail() {
  // Get from your auth context
  const { user } = useAuth()
  
  return {
    senderName: user?.fullName,
    senderCompany: user?.company
  }
}
```

## Email Delivery

### Current Implementation
- Uses existing `/api/send-email` endpoint
- Supports both HTML and text content
- Handles email simulation in development
- Integrates with Resend API for production

### Email Status
- Success/failure notifications via toast
- Simulation mode detection
- Error handling and user feedback

## Styling and Branding

### Email Styling
- Professional gradient header
- Consistent color scheme
- Responsive grid layouts
- Clear typography hierarchy
- Mobile-optimized design

### Customization
To customize the email templates, modify:
- `generateHtmlContent()` in `lib/email-share-service.ts`
- CSS styles in the HTML template
- Color scheme and branding elements

## Best Practices

### Performance
- Email generation is client-side for speed
- Templates are optimized for email clients
- Minimal external dependencies

### User Experience
- Live preview before sending
- Clear validation messages
- Loading states and feedback
- Intuitive recipient management

### Email Deliverability
- Both HTML and text versions
- Proper email headers
- Mobile-responsive design
- Professional formatting

## Troubleshooting

### Common Issues

1. **TypeScript Errors**: Ensure all interfaces match between components
2. **Email Not Sending**: Check email service configuration
3. **Styling Issues**: Verify CSS compatibility with email clients
4. **Missing Data**: Ensure all required props are passed to components

### Debug Mode
Enable console logging in email service for debugging:
```tsx
console.log('Email content:', htmlContent)
console.log('Recipients:', emailData.to)
```

## Future Enhancements

### Planned Features
- Email templates customization UI
- Attachment support for PDF exports
- Email scheduling
- Recipient groups/contacts management
- Email analytics and tracking
- Bulk email sending

### Integration Opportunities
- CRM system integration
- Email campaign management
- User preference settings
- Advanced email templates

## API Reference

### EmailShareData Interface
```tsx
interface EmailShareData {
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
```

### Hook Return Values
```tsx
const {
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

  // Utilities
  extractTitleFromQuery
} = useEmailShare(options)
```

This email sharing system provides a comprehensive solution for sharing AI-generated content via email with professional formatting and user-friendly interface.
