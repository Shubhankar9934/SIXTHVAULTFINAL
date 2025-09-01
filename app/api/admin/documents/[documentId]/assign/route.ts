import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function POST(
  request: NextRequest,
  { params }: { params: Promise<{ documentId: string }> }
) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Authorization header required' }, { status: 401 })
    }

    const assignmentData = await request.json()
    const { documentId } = await params

    // Forward request to FastAPI backend
    const response = await fetch(`${BACKEND_URL}/admin/documents/${documentId}/assign`, {
      method: 'POST',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(assignmentData),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to assign document' }))
      return NextResponse.json(errorData, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('Error assigning document:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
