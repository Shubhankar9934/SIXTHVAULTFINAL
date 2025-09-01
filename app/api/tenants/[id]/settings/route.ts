import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

function getAuthToken(request: NextRequest) {
  const authHeader = request.headers.get('authorization')
  if (authHeader?.startsWith('Bearer ')) {
    return authHeader.substring(7)
  }
  
  // Fallback to cookie
  const cookieHeader = request.headers.get('cookie')
  if (cookieHeader) {
    const authCookie = cookieHeader.split('; ').find(row => row.startsWith('auth-token='))
    if (authCookie) {
      return authCookie.split('=')[1]
    }
  }
  
  return null
}

// GET /api/tenants/[id]/settings - Get tenant settings
export async function GET(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const token = getAuthToken(request)
    if (!token) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 })
    }

    const response = await fetch(`${BACKEND_URL}/tenants/${params.id}/settings`, {
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch tenant settings' }))
      return NextResponse.json({ error: errorData.detail || 'Failed to fetch tenant settings' }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error fetching tenant settings:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

// PUT /api/tenants/[id]/settings - Update tenant settings
export async function PUT(request: NextRequest, { params }: { params: { id: string } }) {
  try {
    const token = getAuthToken(request)
    if (!token) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 })
    }

    const body = await request.json()
    
    const response = await fetch(`${BACKEND_URL}/tenants/${params.id}/settings`, {
      method: 'PUT',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to update tenant settings' }))
      return NextResponse.json({ error: errorData.detail || 'Failed to update tenant settings' }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error updating tenant settings:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
