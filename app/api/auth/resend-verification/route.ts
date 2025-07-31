import { NextRequest, NextResponse } from "next/server"

const RAG_API_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'http://localhost:8000'

export async function POST(req: NextRequest) {
  try {
    const { email } = await req.json()

    if (!email) {
      return NextResponse.json(
        { error: "Email is required" },
        { status: 400 }
      )
    }

    // Send email as query parameter
    const response = await fetch(`${RAG_API_URL}/auth/resend-verification?email=${encodeURIComponent(email)}`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      }
    })

    const data = await response.json()

    if (!response.ok) {
      // Extract error message from FastAPI response
      let errorMessage = "Failed to resend verification code"
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
    console.error("Resend verification error:", error)
    return NextResponse.json(
      { error: "Internal server error" },
      { status: 500 }
    )
  }
}
