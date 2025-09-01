# SIXTHVAULT Email Functionality Guide

## Overview
The email system has been enhanced to support both simulated and actual email sending while preserving the existing verification system.

## Key Features

### 1. Dual Email Modes
- **Simulation Mode** (Default): Emails are logged to console only - preserves existing verification flow
- **Actual Email Mode**: Emails are sent via Resend service using professional sender addresses

### 2. Professional Sender Addresses
- When sending actual emails, the system uses the current user's email and name as the sender
- Format: `John Doe <john.doe@example.com>` instead of `SIXTHVAULT <verify@sixth-vault.com>`

### 3. Preserved Verification System
- Account verification emails continue to work as before (simulated)
- No disruption to the signup/login flow
- Verification codes are still extracted and logged for testing

## How to Use

### For Email Sharing in Vault
1. Generate any analysis, summary, or curation in the vault
2. Click the email share button
3. Fill in recipient details
4. **Check "Send actual email (via Resend service)"** to send real emails
5. Leave unchecked for simulation mode

### For Developers

#### Frontend API Usage
```javascript
// Simulated email (default)
await fetch('/api/send-email', {
  method: 'POST',
  body: JSON.stringify({
    to: 'recipient@example.com',
    subject: 'Test Email',
    html: '<p>HTML content</p>',
    text: 'Text content',
    useBackendService: false // or omit
  })
})

// Actual email with user as sender
await fetch('/api/send-email', {
  method: 'POST',
  body: JSON.stringify({
    to: 'recipient@example.com',
    subject: 'Test Email',
    html: '<p>HTML content</p>',
    text: 'Text content',
    useBackendService: true, // This triggers actual email sending
    senderEmail: 'user@example.com', // Optional - auto-detected from auth
    senderName: 'User Name' // Optional - auto-detected from auth
  })
})
```

#### Backend API Usage
```python
# Direct backend email sending
POST /email/send
{
  "to": "recipient@example.com",
  "subject": "Test Email",
  "html_content": "<p>HTML content</p>",
  "text_content": "Text content",
  "from_email": "sender@example.com",  // Optional
  "from_name": "Sender Name"           // Optional
}
```

## Configuration

### Environment Variables Required
- `RESEND_API_KEY`: Your Resend API key (already configured)
- `EMAIL_DOMAIN`: Domain for default emails (already set to sixth-vault.com)

### Current Configuration Status
✅ Resend API Key: Configured  
✅ Email Domain: sixth-vault.com  
✅ Backend Service: Running on port 8000  
✅ Frontend Integration: Complete  

## Testing

### Test Results
- ✅ Simulated emails work (preserves verification system)
- ✅ Actual emails work via Resend service
- ✅ Professional sender addresses implemented
- ✅ User authentication integration working
- ✅ Email sharing modal updated with toggle option

### How to Test
1. Go to `/vault` and generate some analysis
2. Click the email share button
3. Add your email as recipient
4. Check "Send actual email (via Resend service)"
5. Send the email
6. Check your inbox for the professionally formatted email

## Troubleshooting

### Common Issues
1. **422 Unprocessable Entity**: Usually means email validation failed - check email format
2. **Resend Testing Limitations**: In testing mode, you can only send to verified email addresses
3. **Missing Sender Info**: If user info can't be retrieved, falls back to default SIXTHVAULT sender

### Debug Information
- All email attempts are logged to console with detailed information
- Check browser console and backend logs for debugging
- Verification codes are always extracted and logged for testing

## Security Notes
- Verification emails still use the default SIXTHVAULT sender for security
- User emails are only used for analysis sharing, not system emails
- All email sending requires user authentication
- Sensitive information is not exposed in error messages

## Future Enhancements
- Add email templates for different content types
- Implement email scheduling
- Add email delivery status tracking
- Support for email attachments
