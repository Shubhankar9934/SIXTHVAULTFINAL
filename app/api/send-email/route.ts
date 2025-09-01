import { type NextRequest, NextResponse } from "next/server"

// Force Node.js runtime (though this may not work in all environments)
export const runtime = "nodejs"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    const { to, subject, html, text, useBackendService = false, senderEmail, senderName } = body

    // Validate input
    if (!to || !subject || (!html && !text)) {
      return NextResponse.json({ error: "Missing required fields" }, { status: 400 })
    }

    console.log("=== EMAIL SENDING ATTEMPT ===")
    console.log(`Environment: ${process.env.NODE_ENV}`)
    console.log(`To: ${to}`)
    console.log(`Subject: ${subject}`)
    console.log(`Use Backend Service: ${useBackendService}`)

    // Extract verification code from email content for easy access
    const codeMatch = text?.match(/verification code is: (\w+)/) || html?.match(/(\w{6})/)
    const verificationCode = codeMatch ? codeMatch[1] : "Not found"

    console.log("=== EMAIL CONTENT PREVIEW ===")
    console.log(`Verification Code: ${verificationCode}`)
    console.log(`Text Content: ${text?.substring(0, 200)}...`)
    console.log("=== END EMAIL PREVIEW ===")

    // Always use backend service in development, or when explicitly requested
    const shouldUseBackendService = process.env.NODE_ENV === 'development' ? true : (useBackendService ?? false);
    
    if (shouldUseBackendService) {
      try {
        const backendUrl = process.env.NEXT_PUBLIC_RAG_API_URL || "http://localhost:8000"
        
        // Call backend email service
        const response = await fetch(`${backendUrl}/email/send`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            to: Array.isArray(to) ? to[0] : to,
            subject,
            html_content: html,
            text_content: text,
            from_email: senderEmail,
            from_name: senderName,
          }),
        })

        if (response.ok) {
          const result = await response.json()
          console.log("Email sent successfully via backend service")
          return NextResponse.json({
            success: true,
            messageId: result.messageId || `backend-${Date.now()}`,
            message: "Email sent successfully via backend service",
            simulated: result.simulated || false,
            verificationCode: verificationCode,
            recipient: to,
            subject: subject,
          })
        } else {
          console.log("Backend email service failed, falling back to simulation")
        }
      } catch (backendError) {
        console.log("Backend email service error, falling back to simulation:", backendError)
      }
    }

    // Fallback to simulation (preserves existing verification system behavior)
    return NextResponse.json({
      success: true,
      messageId: `simulated-${Date.now()}`,
      message: "Email simulated - Use backend service for actual delivery",
      simulated: true,
      verificationCode: verificationCode,
      recipient: to,
      subject: subject,
      note: "To send actual emails, set useBackendService: true in request body",
    })
  } catch (error: any) {
    console.error("Email API error:", error)

    // Always return success with simulation to prevent signup flow from breaking
    return NextResponse.json({
      success: true,
      messageId: `error-sim-${Date.now()}`,
      message: "Email simulated due to environment limitations",
      simulated: true,
      error: error.message,
    })
  }
}
