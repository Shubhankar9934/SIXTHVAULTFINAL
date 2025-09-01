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

// POST /api/tenants/initialize - Initialize default tenants
export async function POST(request: NextRequest) {
  try {
    const token = getAuthToken(request)
    if (!token) {
      return NextResponse.json({ error: 'Authentication required' }, { status: 401 })
    }

    const response = await fetch(`${BACKEND_URL}/tenants/initialize`, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ detail: 'Failed to initialize tenants' }))
      return NextResponse.json({ error: errorData.detail || 'Failed to initialize tenants' }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error('Error initializing tenants:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

// Handle other HTTP methods gracefully (for diagnostic purposes)
export async function GET(request: NextRequest) {
  return NextResponse.json({
    message: 'Tenant initialization endpoint',
    methods: ['POST'],
    description: 'Use POST to initialize default tenants',
    timestamp: new Date().toISOString()
  })
}

export async function OPTIONS(request: NextRequest) {
  return NextResponse.json({
    methods: ['GET', 'POST'],
    message: 'Tenant initialization endpoint',
    timestamp: new Date().toISOString()
  }, {
    headers: {
      'Allow': 'GET, POST, OPTIONS'
    }
  })
}
