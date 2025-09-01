import { NextRequest, NextResponse } from "next/server"

// This endpoint is for preview/development only
// It allows retrieving verification codes for testing
// In production, this should be removed or secured

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const email = searchParams.get("email")

    // ENHANCED: Handle diagnostic requests gracefully
    if (!email) {
      return NextResponse.json({
        message: "Preview codes endpoint for development",
        usage: "GET /api/preview-codes?email=user@example.com",
        description: "Retrieves verification codes for testing purposes",
        timestamp: new Date().toISOString()
      }, { status: 200 })
    }

    // Mock verification code for development
    const mockCode = "123456"

    return NextResponse.json({ 
      email, 
      code: mockCode,
      message: "Mock verification code for development",
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error("Error retrieving preview code:", error)
    return NextResponse.json({ error: "Failed to retrieve verification code" }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const { email } = await request.json()
    
    if (!email) {
      return NextResponse.json(
        { error: "Email is required" },
        { status: 400 }
      )
    }

    // Mock preview code generation for development
    const previewCode = "123456"
    
    return NextResponse.json({
      success: true,
      previewCode,
      email,
      message: "Mock preview code generated for development",
      timestamp: new Date().toISOString()
    })
  } catch (error) {
    console.error("Preview code generation error:", error)
    return NextResponse.json(
      { error: "Failed to generate preview code" },
      { status: 500 }
    )
  }
}

// Handle other HTTP methods gracefully
export async function OPTIONS(request: NextRequest) {
  return NextResponse.json({
    methods: ['GET', 'POST'],
    message: 'Preview codes endpoint for development',
    timestamp: new Date().toISOString()
  }, {
    headers: {
      'Allow': 'GET, POST, OPTIONS'
    }
  })
}

export async function HEAD(request: NextRequest) {
  return new NextResponse(null, {
    status: 200,
    headers: {
      'Allow': 'GET, POST, OPTIONS'
    }
  })
}
