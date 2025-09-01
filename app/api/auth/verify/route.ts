import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    const { email, verificationCode } = await req.json()
    console.log('Verify API called with:', { email, verificationCode })

    if (!email || !verificationCode) {
      console.log('Missing required fields:', { email, verificationCode })
      return NextResponse.json(
        { error: "Email and verification code are required" },
        { status: 400 }
      )
    }

    const response = await fetch(`${RAG_API_URL}/auth/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, verification_code: verificationCode }),
    })

    const data = await response.json()

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Verification failed"
      if (data.detail) {
        // Handle both string and object error formats
        errorMessage = typeof data.detail === 'string' 
          ? data.detail 
          : data.detail.msg || data.detail.message || JSON.stringify(data.detail)
      }
      return NextResponse.json(
        { error: errorMessage },
        { status: response.status }
      )
    }

    return NextResponse.json(data)
  } catch (error) {
    console.error("Verification error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}

// Handle verification links
export async function GET(req: NextRequest) {
  try {
    const { searchParams } = new URL(req.url)
    const email = searchParams.get('email')
    const code = searchParams.get('code')

    // ENHANCED: Handle diagnostic requests gracefully
    if (!email && !code) {
      console.log('🔍 Auth Verify: Diagnostic request detected - returning status info')
      return NextResponse.json({
        status: 'ready',
        message: 'Auth verification endpoint is operational',
        methods: ['GET (with email & code params)', 'POST (with email & verificationCode body)'],
        timestamp: new Date().toISOString()
      }, { status: 200 })
    }

    if (!email || !code) {
      console.warn('🔍 Auth Verify: Missing required parameters:', { email: !!email, code: !!code })
      return NextResponse.redirect('/verify?error=Invalid verification link')
    }

    console.log('🔍 Auth Verify: Processing verification link for:', email)

    const response = await fetch(`${RAG_API_URL}/auth/verify`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ email, verification_code: code }),
    })

    if (!response.ok) {
      const data = await response.json().catch(() => ({ detail: 'Unknown error' }))
      let errorMessage = "Verification failed"
      if (data.detail) {
        // Handle both string and object error formats
        errorMessage = typeof data.detail === 'string'
          ? data.detail
          : data.detail.msg || data.detail.message || JSON.stringify(data.detail)
      }
      console.error('🔍 Auth Verify: Backend verification failed:', errorMessage)
      return NextResponse.redirect(`/verify?error=${encodeURIComponent(errorMessage)}`)
    }

    console.log('✅ Auth Verify: Verification successful for:', email)
    return NextResponse.redirect('/verify?success=true')
  } catch (error) {
    console.error("🔍 Auth Verify: Verification link error:", error)
    return NextResponse.redirect('/verify?error=Verification failed')
  }
}
