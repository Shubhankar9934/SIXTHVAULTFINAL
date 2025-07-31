import { NextRequest, NextResponse } from 'next/server'

const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()

    if (!email) {
      return NextResponse.json(
        { message: 'Email is required' },
        { status: 400 }
      )
    }

    // Validate email format
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/
    if (!emailRegex.test(email)) {
      return NextResponse.json(
        { message: 'Invalid email format' },
        { status: 400 }
      )
    }

    // Call the backend API to send password reset verification code
    const response = await fetch(`${RAG_API_URL}/auth/forgot-password`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: email.toLowerCase().trim()
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Failed to send verification code"
      if (data.detail) {
        errorMessage = typeof data.detail === 'string' 
          ? data.detail 
          : data.detail.msg || data.detail.message || JSON.stringify(data.detail)
      }
      return NextResponse.json(
        { message: errorMessage },
        { status: response.status }
      )
    }

    // Always return success to prevent email enumeration attacks
    return NextResponse.json({
      message: 'If an account with that email exists, we have sent a verification code.',
      success: true
    })

  } catch (error) {
    console.error('Forgot password error:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}
