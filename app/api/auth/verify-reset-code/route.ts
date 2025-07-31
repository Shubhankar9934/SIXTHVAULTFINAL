import { NextRequest, NextResponse } from 'next/server'

const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const { email, verificationCode } = await request.json()

    if (!email || !verificationCode) {
      return NextResponse.json(
        { message: 'Email and verification code are required' },
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

    // Validate verification code format
    if (verificationCode.length !== 6) {
      return NextResponse.json(
        { message: 'Verification code must be 6 characters' },
        { status: 400 }
      )
    }

    // Call the backend API to verify the reset code
    const response = await fetch(`${RAG_API_URL}/auth/verify-reset-code`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        email: email.toLowerCase().trim(),
        verification_code: verificationCode.toUpperCase()
      }),
    })

    const data = await response.json()

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Invalid verification code"
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

    // Return the reset token for the next step
    return NextResponse.json({
      message: 'Verification code verified successfully',
      resetToken: data.reset_token
    })

  } catch (error) {
    console.error('Verify reset code error:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}
