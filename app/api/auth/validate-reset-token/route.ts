import { NextRequest, NextResponse } from 'next/server'

const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()
    console.log('Validate reset token - request body:', JSON.stringify(body, null, 2))
    
    const { token } = body

    console.log('Validate reset token - extracted token:', token)
    console.log('Validate reset token - token type:', typeof token)
    console.log('Validate reset token - token length:', token ? token.length : 'N/A')

    if (!token) {
      console.log('Validate reset token - no token provided')
      return NextResponse.json(
        { message: 'Token is required' },
        { status: 400 }
      )
    }

    // Call the backend API to validate the reset token
    const response = await fetch(`${RAG_API_URL}/auth/validate-reset-token`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({ token }),
    })

    const data = await response.json()

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Invalid reset token"
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

    return NextResponse.json({
      message: 'Token is valid',
      email: data.email
    })

  } catch (error) {
    console.error('Validate reset token error:', error)
    return NextResponse.json(
      { message: 'Internal server error' },
      { status: 500 }
    )
  }
}
