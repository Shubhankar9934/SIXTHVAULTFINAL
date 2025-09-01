const API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_URL || 'https://sixth-vault.com/api'

class AdminAPI {
  private getAuthHeaders() {
    // Get token from cookie with enhanced error handling and debugging
    let token: string | undefined
    
    try {
      // Check if we're in a browser environment
      if (typeof window === 'undefined') {
        console.warn('🚨 AdminAPI: Not in browser environment, no cookies available')
        return {
          'Content-Type': 'application/json',
        }
      }

      // Enhanced cookie parsing logic with multiple fallback strategies
      const rawCookies = document.cookie
      console.log('🔍 AdminAPI: Raw cookies string:', rawCookies)
      
      if (!rawCookies || rawCookies.trim() === '') {
        console.warn('🚨 AdminAPI: No cookies found in document.cookie')
        return {
          'Content-Type': 'application/json',
        }
      }

      // Strategy 1: Standard cookie parsing
      const cookies = rawCookies.split(';').map(cookie => cookie.trim())
      let authCookie = cookies.find(cookie => cookie.startsWith('auth-token='))
      
      // Strategy 2: If not found, try without trimming (in case of edge cases)
      if (!authCookie) {
        const rawCookieArray = rawCookies.split(';')
        authCookie = rawCookieArray.find(cookie => cookie.includes('auth-token='))
      }
      
      // Strategy 3: Direct string search as last resort
      if (!authCookie && rawCookies.includes('auth-token=')) {
        const startIndex = rawCookies.indexOf('auth-token=')
        const endIndex = rawCookies.indexOf(';', startIndex)
        authCookie = endIndex === -1 
          ? rawCookies.substring(startIndex)
          : rawCookies.substring(startIndex, endIndex)
      }
      
      console.log('🔍 AdminAPI: Cookie parsing strategies:')
      console.log('  - Total cookies found:', cookies.length)
      console.log('  - Auth cookie found:', !!authCookie)
      console.log('  - Auth cookie value:', authCookie)
      
      if (authCookie) {
        // Extract token value after the '=' sign with enhanced validation
        const equalIndex = authCookie.indexOf('=')
        if (equalIndex !== -1 && equalIndex < authCookie.length - 1) {
          const tokenValue = authCookie.substring(equalIndex + 1).trim()
          
          if (tokenValue && 
              tokenValue !== 'undefined' && 
              tokenValue !== 'null' && 
              tokenValue !== '' &&
              tokenValue.length > 10) {
            
            // Try to decode URI component in case the token was encoded
            try {
              const decodedToken = decodeURIComponent(tokenValue)
              // Validate decoded token
              if (decodedToken && 
                  decodedToken !== 'undefined' && 
                  decodedToken !== 'null' && 
                  decodedToken.trim() !== '' &&
                  decodedToken.length > 10) {
                token = decodedToken
                console.log('✅ AdminAPI: Successfully decoded token')
              } else {
                token = tokenValue // Use raw token if decoded is invalid
                console.log('🔧 AdminAPI: Using raw token (decode validation failed)')
              }
            } catch (decodeError) {
              console.warn('🔧 AdminAPI: Token decode failed, using raw token:', decodeError)
              token = tokenValue
            }
          } else {
            console.warn('🚨 AdminAPI: Invalid token value found:', {
              exists: !!tokenValue,
              notUndefined: tokenValue !== 'undefined',
              notNull: tokenValue !== 'null',
              notEmpty: tokenValue !== '',
              lengthValid: tokenValue.length > 10,
              actualLength: tokenValue.length
            })
          }
        } else {
          console.warn('🚨 AdminAPI: Malformed auth cookie (no = found or empty value):', authCookie)
        }
      } else {
        console.warn('🚨 AdminAPI: No auth-token cookie found in:', rawCookies)
      }
      
      console.log('🔍 AdminAPI: Final token extraction result:')
      console.log('  - Token extracted:', !!token)
      console.log('  - Token length:', token?.length || 0)
      console.log('  - Token preview:', token ? token.substring(0, 20) + '...' : 'none')
      
    } catch (error) {
      console.error('🚨 AdminAPI: Error reading auth token from cookies:', error)
      console.log('🍪 AdminAPI: Raw document.cookie:', document.cookie)
    }

    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    }

    // Only add Authorization header if we have a valid token
    if (token && token.trim() && token !== 'undefined' && token !== 'null' && token.length > 10) {
      headers['Authorization'] = `Bearer ${token}`
      console.log('✅ AdminAPI: Authorization header added successfully')
      console.log('🔑 AdminAPI: Using auth token:', token.substring(0, 20) + '...')
    } else {
      console.warn('🚨 AdminAPI: No valid authentication token found - request will be sent without Authorization header')
      console.log('🍪 AdminAPI: Available cookies:', document.cookie)
      console.log('🔍 AdminAPI: Token validation failed:')
      console.log('  - Token exists:', !!token)
      console.log('  - Token trimmed:', !!token?.trim())
      console.log('  - Not undefined string:', token !== 'undefined')
      console.log('  - Not null string:', token !== 'null')
      console.log('  - Length > 10:', (token?.length || 0) > 10)
    }

    return headers
  }

  async getUsers() {
    try {
      console.log('AdminAPI: Fetching users...')
      const headers = this.getAuthHeaders()
      console.log('AdminAPI: Auth headers prepared, token present:', !!headers.Authorization)
      
      // If no auth token, return empty array instead of throwing error
      if (!headers.Authorization) {
        console.log('AdminAPI: No auth token, returning empty users array')
        return []
      }
      
      const response = await fetch('/api/admin/users', {
        headers,
      })
      
      console.log('AdminAPI: Response status:', response.status, response.statusText)
      
      if (!response.ok) {
        // For auth errors, return empty array instead of throwing
        if (response.status === 401 || response.status === 403) {
          console.log('AdminAPI: Auth error, returning empty users array')
          return []
        }
        
        let errorMessage = 'Failed to fetch users'
        try {
          const errorData = await response.json()
          errorMessage = errorData.error || errorData.detail || errorMessage
          console.log('AdminAPI: Error response:', errorData)
        } catch (parseError) {
          console.log('AdminAPI: Could not parse error response')
        }
        
        throw new Error(errorMessage)
      }
      
      const data = await response.json()
      console.log('AdminAPI: Users fetched successfully:', data.length, 'users')
      return data
    } catch (error) {
      console.log('AdminAPI: Error fetching users (returning empty array):', error instanceof Error ? error.message : 'Unknown error')
      return []
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
      const headers = this.getAuthHeaders()
      
      // If no auth token, return empty array
      if (!headers.Authorization) {
        console.log('AdminAPI: No auth token, returning empty documents array')
        return []
      }
      
      const url = userId ? `/api/admin/documents?userId=${userId}` : '/api/admin/documents'
      const response = await fetch(url, {
        headers,
      })
      
      if (!response.ok) {
        // For auth errors, return empty array instead of throwing
        if (response.status === 401 || response.status === 403) {
          console.log('AdminAPI: Auth error, returning empty documents array')
          return []
        }
        throw new Error('Failed to fetch documents')
      }
      
      return await response.json()
    } catch (error) {
      console.log('AdminAPI: Error fetching documents (returning empty array):', error instanceof Error ? error.message : 'Unknown error')
      return []
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
      const headers = this.getAuthHeaders()
      
      // If no auth token, return fallback data
      if (!headers.Authorization) {
        console.log('AdminAPI: No auth token, returning fallback system stats')
        return {
          users: { total: 0, active: 0, inactive: 0 },
          documents: { total: 0, processing: 0, completed: 0 },
          storage: { total_bytes: 0, total_gb: 0 },
          ai: { curations: 0, summaries: 0 }
        }
      }
      
      const response = await fetch('/api/admin/system', {
        headers,
      })
      
      if (!response.ok) {
        // For auth errors, return fallback data instead of throwing
        if (response.status === 401 || response.status === 403) {
          console.log('AdminAPI: Auth error, returning fallback system stats')
          return {
            users: { total: 0, active: 0, inactive: 0 },
            documents: { total: 0, processing: 0, completed: 0 },
            storage: { total_bytes: 0, total_gb: 0 },
            ai: { curations: 0, summaries: 0 }
          }
        }
        throw new Error('Failed to fetch system stats')
      }
      
      return await response.json()
    } catch (error) {
      console.log('AdminAPI: Error fetching system stats (returning fallback):', error instanceof Error ? error.message : 'Unknown error')
      // Return fallback data if backend is not available
      return {
        users: { total: 0, active: 0, inactive: 0 },
        documents: { total: 0, processing: 0, completed: 0 },
        storage: { total_bytes: 0, total_gb: 0 },
        ai: { curations: 0, summaries: 0 }
      }
    }
  }

  // Tenant Management Methods
  async getTenants() {
    try {
      const headers = this.getAuthHeaders()
      
      // If no auth token, return empty array
      if (!headers.Authorization) {
        console.log('AdminAPI: No auth token, returning empty tenants array')
        return []
      }
      
      const response = await fetch('/api/tenants/', {
        headers,
      })
      
      if (!response.ok) {
        // For auth errors, return empty array instead of throwing
        if (response.status === 401 || response.status === 403) {
          console.log('AdminAPI: Auth error, returning empty tenants array')
          return []
        }
        
        const errorData = await response.json().catch(() => ({ detail: 'Failed to fetch tenants' }))
        throw new Error(errorData.detail || 'Failed to fetch tenants')
      }
      
      return await response.json()
    } catch (error) {
      console.log('AdminAPI: Error fetching tenants (returning empty array):', error instanceof Error ? error.message : 'Unknown error')
      return []
    }
  }

  async createTenant(tenantData: {
    slug: string
    name: string
    tenant_type: string
    primary_color?: string
    secondary_color?: string
    logo_url?: string
    custom_domain?: string
  }) {
    try {
      const response = await fetch('/api/tenants/', {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(tenantData),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to create tenant')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error creating tenant:', error)
      throw error
    }
  }

  async updateTenant(tenantId: string, tenantData: any) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}`, {
        method: 'PUT',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(tenantData),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to update tenant')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error updating tenant:', error)
      throw error
    }
  }

  async getTenant(tenantId: string) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch tenant')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching tenant:', error)
      throw error
    }
  }

  async getTenantUsers(tenantId: string) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/users`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch tenant users')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching tenant users:', error)
      throw error
    }
  }

  async addUserToTenant(tenantId: string, userData: {
    user_id: string
    tenant_role: string
    permissions?: string[]
  }) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/users`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(userData),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to add user to tenant')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error adding user to tenant:', error)
      throw error
    }
  }

  async removeUserFromTenant(tenantId: string, userId: string) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/users/${userId}`, {
        method: 'DELETE',
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to remove user from tenant')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error removing user from tenant:', error)
      throw error
    }
  }

  async getTenantSettings(tenantId: string) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/settings`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch tenant settings')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching tenant settings:', error)
      throw error
    }
  }

  async updateTenantSettings(tenantId: string, settings: any) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/settings`, {
        method: 'PUT',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(settings),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to update tenant settings')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error updating tenant settings:', error)
      throw error
    }
  }

  async getTenantAnalytics(tenantId: string, days: number = 30) {
    try {
      const response = await fetch(`/api/tenants/${tenantId}/analytics?days=${days}`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch tenant analytics')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching tenant analytics:', error)
      throw error
    }
  }

  async initializeDefaultTenants() {
    try {
      const response = await fetch('/api/tenants/initialize', {
        method: 'POST',
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to initialize default tenants')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error initializing default tenants:', error)
      throw error
    }
  }

  async getUserTenants(userId: string) {
    try {
      const response = await fetch(`/api/tenants/user/${userId}/tenants`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch user tenants')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching user tenants:', error)
      throw error
    }
  }

  // Document Assignment Methods
  async assignDocumentToUsers(documentId: string, userIds: string[], permissions: string[] = ['read']) {
    try {
      const response = await fetch(`/api/admin/documents/${documentId}/assign`, {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify({
          user_ids: userIds,
          permissions: permissions
        }),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to assign document')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error assigning document:', error)
      throw error
    }
  }

  async removeDocumentAssignment(documentId: string, userId: string) {
    try {
      const response = await fetch(`/api/admin/documents/${documentId}/unassign/${userId}`, {
        method: 'DELETE',
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to unassign document')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error unassigning document:', error)
      throw error
    }
  }

  async getDocumentAssignments(documentId: string) {
    try {
      const response = await fetch(`/api/admin/documents/${documentId}/assignments`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch document assignments')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching document assignments:', error)
      throw error
    }
  }

  async createUserWithDocuments(userData: {
    first_name: string
    last_name: string
    email: string
    username?: string
    password: string
    role: string
    company?: string
    assigned_document_ids?: string[]
    document_permissions?: string[]
  }) {
    try {
      const response = await fetch('/api/admin/users/create-with-documents', {
        method: 'POST',
        headers: this.getAuthHeaders(),
        body: JSON.stringify(userData),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to create user with documents')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error creating user with documents:', error)
      throw error
    }
  }

  async getUserDocuments(userId: string) {
    try {
      const response = await fetch(`/api/admin/users/${userId}/documents`, {
        headers: this.getAuthHeaders(),
      })
      
      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch user documents')
      }
      
      return await response.json()
    } catch (error) {
      console.error('Error fetching user documents:', error)
      throw error
    }
  }
}

export const adminAPI = new AdminAPI()
