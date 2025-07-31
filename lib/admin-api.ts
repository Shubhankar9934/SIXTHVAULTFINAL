const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000'

class AdminAPI {
  private getAuthHeaders() {
    // Get token from cookie (consistent with auth context)
    const getAuthCookie = () => {
      if (typeof window === 'undefined') return null
      return document.cookie.split('; ').find(row => row.startsWith('auth-token='))?.split('=')[1]
    }
    
    const token = getAuthCookie()
    return {
      'Content-Type': 'application/json',
      'Authorization': token ? `Bearer ${token}` : '',
    }
  }

  async getUsers() {
    try {
      console.log('AdminAPI: Fetching users...')
      const headers = this.getAuthHeaders()
      console.log('AdminAPI: Auth headers prepared, token present:', !!headers.Authorization)
      
      const response = await fetch('/api/admin/users', {
        headers,
      })
      
      console.log('AdminAPI: Response status:', response.status, response.statusText)
      
      if (!response.ok) {
        let errorMessage = 'Failed to fetch users'
        try {
          const errorData = await response.json()
          errorMessage = errorData.error || errorData.detail || errorMessage
          console.log('AdminAPI: Error response:', errorData)
        } catch (parseError) {
          console.log('AdminAPI: Could not parse error response')
        }
        
        if (response.status === 401) {
          errorMessage = 'Authentication required. Please log in again.'
        } else if (response.status === 403) {
          errorMessage = 'Access denied. Admin permissions required.'
        }
        
        throw new Error(errorMessage)
      }
      
      const data = await response.json()
      console.log('AdminAPI: Users fetched successfully:', data.length, 'users')
      return data
    } catch (error) {
      console.error('AdminAPI: Error fetching users:', error)
      throw error
    }
  }

  async createUser(userData: {
    name: string
    email: string
    role: 'admin' | 'user'
    password?: string
  }) {
    try {
      // Split name into first and last name
      const nameParts = userData.name.trim().split(' ')
      const firstName = nameParts[0] || ''
      const lastName = nameParts.slice(1).join(' ') || ''

      const response = await fetch('/api/admin/users', {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          first_name: firstName,
          last_name: lastName,
          email: userData.email,
          role: userData.role,
          password: userData.password || 'TempPassword123!', // Default password
          company: 'Admin Created'
        }),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to create user')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error creating user:', error)
      throw error
    }
  }

  async updateUser(userId: string, userData: any) {
    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'PUT',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(userData),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to update user')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error updating user:', error)
      throw error
    }
  }

  async deleteUser(userId: string) {
    try {
      const response = await fetch(`/api/admin/users/${userId}`, {
        method: 'DELETE',
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to delete user')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error deleting user:', error)
      throw error
    }
  }

  async getDocuments(userId?: string) {
    try {
      const url = userId ? `/api/admin/documents?userId=${userId}` : '/api/admin/documents'
      const response = await fetch(url, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        throw new Error('Failed to fetch documents')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching documents:', error)
      throw error
    }
  }

  async deleteDocument(documentId: string) {
    try {
      const response = await fetch('/api/admin/documents', {
        method: 'DELETE',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ documentId }),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to delete document')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error deleting document:', error)
      throw error
    }
  }

  async getSystemStatus() {
    try {
      const response = await fetch('/api/admin/system', {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        throw new Error('Failed to fetch system status')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching system status:', error)
      throw error
    }
  }

  async performSystemAction(action: string, data?: any) {
    try {
      const response = await fetch('/api/admin/system', {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({ action, ...data }),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || `Failed to ${action}`)
      }
      
      return await response.json()
    } catch (error) {
      console.error(`Error performing ${action}:`, error)
      throw error
    }
  }

  async getSystemStats() {
    try {
      const response = await fetch('/api/admin/system', {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        throw new Error('Failed to fetch system stats')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching system stats:', error)
      // Return fallback data if backend is not available
      return {
        users: { total: 0, active: 0, inactive: 0 },
        documents: { total: 0, processing: 0, completed: 0 },
        storage: { total_bytes: 0, total_gb: 0 },
        ai: { curations: 0, summaries: 0 }
      }
    }
  }
}

export const adminAPI = new AdminAPI()
