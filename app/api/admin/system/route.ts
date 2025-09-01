import { NextRequest, NextResponse } from 'next/server'

const BACKEND_URL = process.env.NEXT_PUBLIC_RAG_API_URL || process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

export async function GET(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Authorization header required' }, { status: 401 })
    }

    // Forward request to FastAPI backend admin stats endpoint
    const response = await fetch(`${BACKEND_URL}/admin/system/stats`, {
      method: 'GET',
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: 'Failed to fetch system stats' }))
      return NextResponse.json(errorData, { status: response.status })
    }

    const systemStats = await response.json()
    return NextResponse.json(systemStats)
  } catch (error) {
    console.error('Error fetching system stats:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}

export async function POST(request: NextRequest) {
  try {
    const authHeader = request.headers.get('authorization')
    if (!authHeader) {
      return NextResponse.json({ error: 'Authorization header required' }, { status: 401 })
    }

    const { action, ...data } = await request.json()

    let endpoint = ''
    let method = 'POST'

    switch (action) {
      case 'clear_cache':
        endpoint = '/admin/system/clear-cache'
        break
      case 'reindex_documents':
        endpoint = '/admin/system/reindex'
        break
      case 'export_data':
        endpoint = '/admin/system/export'
        method = 'GET'
        break
      case 'update_config':
        endpoint = '/admin/system/config'
        method = 'PUT'
        break
      default:
        return NextResponse.json({ error: 'Invalid action' }, { status: 400 })
    }

    // Forward request to FastAPI backend
    const response = await fetch(`${BACKEND_URL}${endpoint}`, {
      method,
      headers: {
        'Authorization': authHeader,
        'Content-Type': 'application/json',
      },
      body: method !== 'GET' ? JSON.stringify(data) : undefined,
    })

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({ error: `Failed to ${action}` }))
      return NextResponse.json(errorData, { status: response.status })
    }

    const result = await response.json()
    return NextResponse.json(result)
  } catch (error) {
    console.error('Error performing system action:', error)
    return NextResponse.json({ error: 'Internal server error' }, { status: 500 })
  }
}
