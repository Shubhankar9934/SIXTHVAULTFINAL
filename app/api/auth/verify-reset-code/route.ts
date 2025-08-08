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
        verificationCode: verificationCode.toUpperCase()
      }),
    })

    const data = await response.json()
    console.log('Backend response status:', response.status)
    console.log('Backend response data:', JSON.stringify(data, null, 2))

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Invalid verification code"
      if (data.detail) {
        errorMessage = typeof data.detail === 'string' 
          ? data.detail 
          : data.detail.msg || data.detail.message || JSON.stringify(data.detail)
      }
      console.error('Backend error response:', errorMessage)
      return NextResponse.json(
        { message: errorMessage },
        { status: response.status }
      )
    }

    // Return the reset token for the next step
    const resetToken = data.reset_token || data.resetToken
    console.log('Available keys in response:', Object.keys(data))
    console.log('Extracted reset token:', resetToken)
    console.log('data.reset_token:', data.reset_token)
    console.log('data.resetToken:', data.resetToken)
    
    if (!resetToken) {
      console.error('No reset token received from backend. Full response:', JSON.stringify(data, null, 2))
      return NextResponse.json(
        { message: 'Failed to generate reset token' },
        { status: 500 }
      )
    }
    
    console.log('Returning successful response with resetToken:', resetToken)
    return NextResponse.json({
      message: 'Verification code verified successfully',
      resetToken: resetToken
    })

  } catch (error) {
    console.error('Verify reset code error:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}
