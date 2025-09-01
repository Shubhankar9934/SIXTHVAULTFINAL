"use client"

import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Badge } from "@/components/ui/badge"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table"
import { AlertDialog, AlertDialogAction, AlertDialogCancel, AlertDialogContent, AlertDialogDescription, AlertDialogFooter, AlertDialogHeader, AlertDialogTitle, AlertDialogTrigger } from "@/components/ui/alert-dialog"
import { Progress } from "@/components/ui/progress"
import { Checkbox } from "@/components/ui/checkbox"
import { 
  Users, FileText, BarChart3, Shield, Trash2, Edit, Brain, AlertCircle, LogOut, Home,
  Settings, Database, Activity, Search, Plus, Eye, RefreshCw, Download, UserPlus, HardDrive,
  Share, Lock, Unlock, CheckCircle, XCircle, Building, UserCheck, FileCheck, ArrowRight,
  Crown, User
} from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { adminAPI } from "@/lib/admin-api"
import { useToast } from "@/hooks/use-toast"
import { useAuth } from "@/lib/auth-context"

interface User {
  id: string
  name: string
  email: string
  role: "admin" | "user"
  status: "active" | "inactive"
  lastLogin: string
  documentsCount: number
  company?: string
  verified: boolean
  created_at: string
  first_name: string
  last_name: string
  assignedDocuments?: UserDocument[]
}

interface UserDocument {
  id: string
  name: string
  size: number
  assignment_type: "owner" | "assigned"
  permissions: string[]
  assigned_at: string
  assigned_by: string
}

interface Document {
  id: string
  name: string
  size: number
  type: string
  uploadDate: string
  owner?: {
    id: string
    name: string
    email: string
  }
  language: string
  themes: string[]
  keywords: string[]
  demographics: string[]
  summary?: string
  keyInsights: string[]
  path: string
}

interface Tenant {
  id: string
  slug: string
  name: string
  tenant_type: string
  is_active: boolean
  current_users: number
  current_storage_mb: number
  current_documents: number
  max_users?: number
  max_storage_gb?: number
  max_documents?: number
  features: Record<string, any>
  primary_color?: string
  secondary_color?: string
  logo_url?: string
  created_at: string
  updated_at: string
}

interface TenantUser {
  id: string
  email: string
  first_name: string
  last_name: string
  tenant_role: string
  status: string
  permissions: string[]
  joined_at: string
  last_active?: string
  storage_used_mb: number
  storage_limit_mb: number
  document_count: number
  document_limit: number
}

export default function AdminPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const [users, setUsers] = useState<User[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [tenants, setTenants] = useState<Tenant[]>([])
  const [selectedTenant, setSelectedTenant] = useState<Tenant | null>(null)
  const [tenantUsers, setTenantUsers] = useState<TenantUser[]>([])
  const [systemStats, setSystemStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [accessDenied, setAccessDenied] = useState(false)
  const [activeTab, setActiveTab] = useState("users")
  const [searchTerm, setSearchTerm] = useState("")
  const [isAddingUser, setIsAddingUser] = useState(false)
  const [isAddingTenant, setIsAddingTenant] = useState(false)
  const [deletingUserId, setDeletingUserId] = useState<string | null>(null)
  const [selectedDocument, setSelectedDocument] = useState<Document | null>(null)
  const [selectedUsersForAssignment, setSelectedUsersForAssignment] = useState<string[]>([])
  const [assignmentPermissions, setAssignmentPermissions] = useState<string[]>(["read"])
  const [isAssigningDocument, setIsAssigningDocument] = useState(false)
  const [loadingUserDocuments, setLoadingUserDocuments] = useState<string | null>(null)
  const [userDocuments, setUserDocuments] = useState<Record<string, UserDocument[]>>({})
  const [unassigningDocument, setUnassigningDocument] = useState<string | null>(null)
  const [expandedUserDocuments, setExpandedUserDocuments] = useState<Set<string>>(new Set())
  const [deletingDocumentId, setDeletingDocumentId] = useState<string | null>(null)
  const [deleteDialogOpen, setDeleteDialogOpen] = useState<string | null>(null)
  
  const [newUser, setNewUser] = useState({
    first_name: "",
    last_name: "",
    email: "",
    username: "",
    password: "",
    role: "user" as "admin" | "user",
    company: "",
    assigned_document_ids: [] as string[],
    document_permissions: ["read"] as string[]
  })

  const [newTenant, setNewTenant] = useState({
    slug: "",
    name: "",
    tenant_type: "enterprise" as "enterprise" | "partner" | "public",
    primary_color: "",
    secondary_color: "",
    logo_url: "",
    custom_domain: ""
  })

  const { toast } = useToast()

  // Check admin access on mount
  useEffect(() => {
    if (!isAuthenticated) {
      setLoading(false)
      return
    }

    if (!user || !user.is_admin || user.role !== 'admin') {
      console.log('Admin page: Access denied - user is not admin:', user)
      setAccessDenied(true)
      setLoading(false)
      return
    }

    console.log('Admin page: Admin access confirmed for user:', user.email)
    setAccessDenied(false)
  }, [isAuthenticated, user])

  // Load data on mount
  useEffect(() => {
    if (!isAuthenticated || !user || accessDenied) {
      setLoading(false)
      return
    }

    const loadData = async () => {
      try {
        setLoading(true)
        console.log('Admin page: Starting to load data...')
        
        const getAuthCookie = () => {
          if (typeof window === 'undefined') return null
          return document.cookie.split('; ').find(row => row.startsWith('auth-token='))?.split('=')[1]
        }
        
        const token = getAuthCookie()
        console.log('Admin page: Auth token present:', !!token)
        
        if (!token) {
          console.log('Admin page: No auth token, redirecting to login')
          window.location.href = '/login'
          return
        }
        
        // Load data with better error handling
        const [usersData, documentsData, statsData, tenantsData] = await Promise.all([
          adminAPI.getUsers().catch(e => { 
            console.log('Failed to load users (non-blocking):', e.message); 
            return [] 
          }),
          adminAPI.getDocuments().catch(e => { 
            console.log('Failed to load documents (non-blocking):', e.message); 
            return [] 
          }),
          adminAPI.getSystemStats().catch(e => { 
            console.log('Failed to load stats (non-blocking):', e.message); 
            return {
              users: { total: 0, active: 0, inactive: 0 },
              documents: { total: 0, processing: 0, completed: 0 },
              storage: { total_bytes: 0, total_gb: 0 },
              ai: { curations: 0, summaries: 0 }
            }
          }),
          adminAPI.getTenants().catch(e => { 
            console.log('Failed to load tenants (non-blocking):', e.message); 
            return [] 
          })
        ])
        
        console.log('Admin page: Data loaded successfully')
        setUsers(usersData)
        setDocuments(documentsData)
        setSystemStats(statsData)
        setTenants(tenantsData)
        
      } catch (error) {
        console.error('Admin page: Critical error loading admin data:', error)
        
        // Only show toast for critical errors, not for missing data
        if (error instanceof Error && 
            (error.message.includes('401') || error.message.includes('403'))) {
          console.log('Admin page: Authentication error, redirecting to login')
          window.location.href = '/login'
          return
        }
        
        // Set fallback data without showing error toast
        setUsers([])
        setDocuments([])
        setSystemStats({
          users: { total: 0, active: 0, inactive: 0 },
          documents: { total: 0, processing: 0, completed: 0 },
          storage: { total_bytes: 0, total_gb: 0 },
          ai: { curations: 0, summaries: 0 }
        })
        setTenants([])
      } finally {
        setLoading(false)
      }
    }
    
    loadData()
  }, [isAuthenticated, user, accessDenied])

  const handleAddUser = async () => {
    if (newUser.first_name && newUser.last_name && newUser.email && newUser.password) {
      try {
        setIsAddingUser(true)
        
        // Use the new API if user has document assignments, otherwise use the regular API
        if (newUser.role === "user" && newUser.assigned_document_ids.length > 0) {
          const result = await adminAPI.createUserWithDocuments({
            first_name: newUser.first_name,
            last_name: newUser.last_name,
            email: newUser.email,
            username: newUser.username,
            password: newUser.password,
            role: newUser.role,
            company: newUser.company,
            assigned_document_ids: newUser.assigned_document_ids,
            document_permissions: newUser.document_permissions
          })
          
          toast({
            title: "Success",
            description: `User created successfully with access to ${result.assigned_documents.length} document(s)`,
          })
        } else {
          // Use regular user creation for admins or users without document assignments
          const userData = {
            name: `${newUser.first_name} ${newUser.last_name}`,
            email: newUser.email,
            role: newUser.role,
            password: newUser.password,
            first_name: newUser.first_name,
            last_name: newUser.last_name,
            username: newUser.username,
            company: newUser.company
          }
          
          await adminAPI.createUser(userData)
          
          toast({
            title: "Success",
            description: `${newUser.role === 'admin' ? 'Admin' : 'User'} created successfully`,
          })
        }
        
        // Reload users
        const usersData = await adminAPI.getUsers()
        setUsers(usersData)
        
        setNewUser({ 
          first_name: "", 
          last_name: "", 
          email: "", 
          username: "", 
          password: "", 
          role: "user",
          company: "",
          assigned_document_ids: [],
          document_permissions: ["read"]
        })
        
      } catch (error: any) {
        toast({
          title: "Error",
          description: error.message || "Failed to create user",
          variant: "destructive",
        })
      } finally {
        setIsAddingUser(false)
      }
    }
  }

  const handleDeleteUser = async (userId: string) => {
    try {
      setDeletingUserId(userId)
      const result = await adminAPI.deleteUser(userId)
      
      setUsers(users.filter((user) => user.id !== userId))
      
      const deletedItems = []
      if (result.deleted_documents > 0) deletedItems.push(`${result.deleted_documents} documents`)
      if (result.deleted_files > 0) deletedItems.push(`${result.deleted_files} files`)
      if (result.deleted_curations > 0) deletedItems.push(`${result.deleted_curations} AI curations`)
      if (result.deleted_summaries > 0) deletedItems.push(`${result.deleted_summaries} AI summaries`)
      if (result.deleted_tokens > 0) deletedItems.push(`${result.deleted_tokens} active sessions`)
      
      const detailMessage = deletedItems.length > 0 
        ? `User account and all associated data deleted: ${deletedItems.join(', ')}`
        : "User account deleted successfully"
      
      toast({
        title: "User Deleted Successfully",
        description: detailMessage,
      })
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete user",
        variant: "destructive",
      })
    } finally {
      setDeletingUserId(null)
    }
  }

  // Helper function to check if user can be deleted
  const canDeleteUser = (targetUser: User) => {
    // Cannot delete your own account
    if (user && targetUser.id === String(user.id)) {
      return false
    }
    return true
  }

  // Helper function to get delete restriction message
  const getDeleteRestrictionMessage = (targetUser: User) => {
    if (user && targetUser.id === String(user.id)) {
      return "You cannot delete your own admin account. Please ask another admin to delete your account if needed."
    }
    return null
  }

  const handleDeleteDocument = async (docId: string) => {
    try {
      setDeletingDocumentId(docId)
      const result = await adminAPI.deleteDocument(docId)
      
      setDocuments((prev) => prev.filter((doc) => doc.id !== docId))
      
      const deletedItems = []
      if (result.file_deleted) deletedItems.push("physical file")
      if (result.deleted_access_records > 0) deletedItems.push(`${result.deleted_access_records} access records`)
      if (result.deleted_access_logs > 0) deletedItems.push(`${result.deleted_access_logs} access logs`)
      if (result.deleted_curation_mappings > 0) deletedItems.push(`${result.deleted_curation_mappings} curation mappings`)
      if (result.deleted_summary_mappings > 0) deletedItems.push(`${result.deleted_summary_mappings} summary mappings`)
      if (result.deleted_processing_records > 0) deletedItems.push(`${result.deleted_processing_records} processing records`)
      
      const detailMessage = deletedItems.length > 0 
        ? `Document "${result.document_name}" and associated data deleted: ${deletedItems.join(', ')}`
        : `Document "${result.document_name}" deleted successfully`
      
      toast({
        title: "Document Deleted Successfully",
        description: detailMessage,
      })
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to delete document",
        variant: "destructive",
      })
    } finally {
      setDeletingDocumentId(null)
    }
  }

  const handleAddTenant = async () => {
    if (newTenant.slug && newTenant.name && newTenant.tenant_type) {
      try {
        setIsAddingTenant(true)
        
        await adminAPI.createTenant(newTenant)
        
        // Reload tenants
        const tenantsData = await adminAPI.getTenants()
        setTenants(tenantsData)
        
        setNewTenant({
          slug: "",
          name: "",
          tenant_type: "enterprise",
          primary_color: "",
          secondary_color: "",
          logo_url: "",
          custom_domain: ""
        })
        
        toast({
          title: "Success",
          description: "Tenant created successfully",
        })
      } catch (error: any) {
        toast({
          title: "Error",
          description: error.message || "Failed to create tenant",
          variant: "destructive",
        })
      } finally {
        setIsAddingTenant(false)
      }
    }
  }

  const handleSelectTenant = async (tenant: Tenant) => {
    try {
      setSelectedTenant(tenant)
      const users = await adminAPI.getTenantUsers(tenant.id)
      setTenantUsers(users)
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to load tenant users",
        variant: "destructive",
      })
    }
  }

  const handleInitializeDefaultTenants = async () => {
    try {
      await adminAPI.initializeDefaultTenants()
      
      // Reload tenants
      const tenantsData = await adminAPI.getTenants()
      setTenants(tenantsData)
      
      toast({
        title: "Success",
        description: "Default tenants initialized successfully",
      })
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to initialize default tenants",
        variant: "destructive",
      })
    }
  }

  const handleAssignDocument = async () => {
    if (!selectedDocument || selectedUsersForAssignment.length === 0) return
    
    try {
      setIsAssigningDocument(true)
      
      const result = await adminAPI.assignDocumentToUsers(
        selectedDocument.id,
        selectedUsersForAssignment,
        assignmentPermissions
      )
      
      toast({
        title: "Success",
        description: `Document "${selectedDocument.name}" assigned to ${result.assigned_users.length} user(s)`,
      })
      
      if (result.failed_assignments && result.failed_assignments.length > 0) {
        toast({
          title: "Warning",
          description: `${result.failed_assignments.length} assignment(s) failed`,
          variant: "destructive",
        })
      }
      
      // Refresh users data to show updated document assignments
      try {
        const updatedUsers = await adminAPI.getUsers()
        setUsers(updatedUsers)
        
        // Clear cached user documents for affected users to force reload
        const updatedUserDocuments = { ...userDocuments }
        selectedUsersForAssignment.forEach(userId => {
          delete updatedUserDocuments[userId]
        })
        setUserDocuments(updatedUserDocuments)
        
        console.log('Admin page: Users data refreshed after document assignment')
      } catch (refreshError) {
        console.error('Admin page: Failed to refresh users data after assignment:', refreshError)
        // Don't show error toast for refresh failure as the assignment was successful
      }
      
      setSelectedDocument(null)
      setSelectedUsersForAssignment([])
      setAssignmentPermissions(["read"])
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to assign document",
        variant: "destructive",
      })
    } finally {
      setIsAssigningDocument(false)
    }
  }

  // Function to load user documents
  const loadUserDocuments = async (userId: string) => {
    if (userDocuments[userId] || loadingUserDocuments === userId) return
    
    try {
      setLoadingUserDocuments(userId)
      const docs = await adminAPI.getUserDocuments(userId)
      setUserDocuments(prev => ({ ...prev, [userId]: docs }))
    } catch (error: any) {
      console.error('Error loading user documents:', error)
      toast({
        title: "Error",
        description: "Failed to load user documents",
        variant: "destructive",
      })
    } finally {
      setLoadingUserDocuments(null)
    }
  }

  // Function to unassign document from user
  const handleUnassignDocument = async (documentId: string, userId: string, documentName: string) => {
    try {
      setUnassigningDocument(`${documentId}-${userId}`)
      
      const result = await adminAPI.removeDocumentAssignment(documentId, userId)
      
      // Refresh user documents for the specific user
      const docs = await adminAPI.getUserDocuments(userId)
      setUserDocuments(prev => ({ ...prev, [userId]: docs }))
      
      // Refresh users data to update document counts
      try {
        const updatedUsers = await adminAPI.getUsers()
        setUsers(updatedUsers)
        console.log('Admin page: Users data refreshed after document unassignment')
      } catch (refreshError) {
        console.error('Admin page: Failed to refresh users data after unassignment:', refreshError)
        // Don't show error toast for refresh failure as the unassignment was successful
      }
      
      toast({
        title: "Success",
        description: `Document "${documentName}" unassigned successfully`,
      })
    } catch (error: any) {
      toast({
        title: "Error",
        description: error.message || "Failed to unassign document",
        variant: "destructive",
      })
    } finally {
      setUnassigningDocument(null)
    }
  }

  // Filter functions
  const filteredUsers = users.filter(user => 
    user.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    user.email.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const filteredDocuments = documents.filter(doc => 
    doc.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
    (doc.owner?.name.toLowerCase().includes(searchTerm.toLowerCase()))
  )

  // Calculate stats
  const totalUsers = users.length
  const activeUsers = users.filter((u) => u.status === "active").length
  const adminUsers = users.filter((u) => u.role === "admin").length
  const regularUsers = users.filter((u) => u.role === "user").length
  const totalDocuments = documents.length
  const processedDocuments = documents.length
  const totalTenants = tenants.length
  const activeTenants = tenants.filter((t) => t.is_active).length

  const totalStorage = documents.reduce((sum, doc) => sum + doc.size, 0)
  const storageInGB = (totalStorage / (1024 * 1024 * 1024)).toFixed(2)

  // Show access denied screen for non-admin users
  if (accessDenied) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-indigo-50/10 flex items-center justify-center">
        <Card className="w-full max-w-md border-0 shadow-2xl bg-white/95 backdrop-blur-xl">
          <CardHeader className="text-center">
            <div className="mx-auto w-16 h-16 bg-gradient-to-br from-red-100 to-red-200 rounded-2xl flex items-center justify-center mb-6 shadow-lg">
              <AlertCircle className="w-8 h-8 text-red-600" />
            </div>
            <CardTitle className="text-2xl font-bold bg-gradient-to-r from-red-700 to-red-800 bg-clip-text text-transparent">
              Access Denied
            </CardTitle>
            <CardDescription className="text-gray-600 text-lg">
              Administrative privileges required
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-6">
            <div className="p-4 bg-gradient-to-r from-red-50 to-red-100 rounded-xl border border-red-200">
              <p className="text-sm text-red-700 font-medium">
                Only users with admin role can access the admin panel.
              </p>
              <p className="text-xs text-red-600 mt-2">
                If you believe this is an error, please contact your system administrator.
              </p>
            </div>
            <div className="flex flex-col space-y-3">
              <Link href="/vault">
                <Button className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
                  <Home className="w-4 h-4 mr-2" />
                  Go to Vault
                </Button>
              </Link>
              <Button variant="outline" onClick={logout} className="w-full border-2 hover:bg-gray-50">
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Loading screen
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-indigo-50/10 flex items-center justify-center">
        <Card className="w-full max-w-lg border-0 shadow-2xl bg-white/95 backdrop-blur-xl">
          <CardContent className="text-center p-12">
            <div className="mb-8">
              <div className="w-24 h-24 bg-gradient-to-br from-blue-100 via-indigo-100 to-purple-100 rounded-3xl flex items-center justify-center mx-auto mb-6 shadow-xl">
                <Shield className="w-12 h-12 text-blue-600" />
              </div>
              <SixthvaultLogo size="full" />
            </div>

            <div className="mb-8">
              <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-200 border-t-blue-600 mx-auto mb-6"></div>
            </div>

            <div className="space-y-4">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 bg-clip-text text-transparent">
                Loading Admin Panel
              </h2>
              <p className="text-xl font-semibold text-slate-700">
                Preparing administrative interface...
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-indigo-50/10">
      <div className="flex h-screen">
        {/* Enhanced Left Sidebar */}
        <div className="w-80 bg-white/95 backdrop-blur-xl border-r border-gray-200/50 flex flex-col shadow-xl">
          <div className="p-6 border-b border-gray-200/50">
            <SixthvaultLogo size="full" />
          </div>

          <div className="p-6 flex-1">
            <div className="flex items-center space-x-3 mb-6">
              <div className="w-10 h-10 bg-gradient-to-br from-blue-600 to-indigo-600 rounded-xl flex items-center justify-center shadow-lg">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <div>
                <h3 className="font-bold text-gray-900 text-lg">Admin Panel</h3>
              </div>
            </div>

            {/* Current User Info */}
            <div className="mb-6 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
              <div className="flex items-center space-x-3">
                <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
                  <UserCheck className="w-4 h-4 text-white" />
                </div>
                <div>
                  <p className="font-medium text-blue-800">{user?.first_name} {user?.last_name}</p>
                  <p className="text-sm text-blue-600">{user?.email}</p>
                </div>
              </div>
            </div>




          </div>

          <div className="p-4 border-t border-gray-200/50">
            <Link href="/vault">
              <Button className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 text-white font-semibold shadow-lg hover:shadow-xl transition-all duration-300">
                <Home className="w-4 h-4 mr-2" />
                Back to Vault
              </Button>
            </Link>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Enhanced Header */}
          <div className="p-6 border-b border-gray-200/50 bg-white/95 backdrop-blur-xl flex justify-between items-center shadow-sm">
            <div>
              <h1 className="text-3xl font-bold bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 bg-clip-text text-transparent flex items-center">
                <Brain className="w-8 h-8 mr-3 text-blue-600" />
                Admin Panel
              </h1>
            </div>
            <div className="flex items-center space-x-4">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 w-4 h-4" />
                <Input
                  placeholder="Search..."
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                  className="pl-10 w-60"
                />
              </div>
              <Button 
                onClick={logout}
                variant="outline"
                className="border-red-200 text-red-600 hover:bg-red-50 hover:border-red-300"
              >
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </div>

          {/* Admin Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <Tabs value={activeTab} onValueChange={setActiveTab} className="space-y-6">
              <TabsList className="grid w-full grid-cols-2 bg-white/80 backdrop-blur-sm border border-gray-200/50 shadow-sm">
                <TabsTrigger value="users" className="data-[state=active]:bg-green-100 data-[state=active]:text-green-700">
                  <Users className="w-4 h-4 mr-2" />
                  Users
                </TabsTrigger>
                <TabsTrigger value="documents" className="data-[state=active]:bg-purple-100 data-[state=active]:text-purple-700">
                  <FileText className="w-4 h-4 mr-2" />
                  Documents
                </TabsTrigger>
                {/* Tenant Management Tab - Hidden for now */}
                {/* <TabsTrigger value="tenants" className="data-[state=active]:bg-indigo-100 data-[state=active]:text-indigo-700">
                  <Building className="w-4 h-4 mr-2" />
                  Tenants
                </TabsTrigger> */}
              </TabsList>


              {/* User Management Tab */}
              <TabsContent value="users" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Create New User */}
                  <Card className="shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <UserPlus className="w-5 h-5 mr-2 text-green-600" />
                        Add New User
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <div>
                          <Label htmlFor="first_name">First Name</Label>
                          <Input
                            id="first_name"
                            value={newUser.first_name}
                            onChange={(e) => setNewUser({ ...newUser, first_name: e.target.value })}
                            placeholder="Enter first name"
                          />
                        </div>
                        <div>
                          <Label htmlFor="last_name">Last Name</Label>
                          <Input
                            id="last_name"
                            value={newUser.last_name}
                            onChange={(e) => setNewUser({ ...newUser, last_name: e.target.value })}
                            placeholder="Enter last name"
                          />
                        </div>
                        <div>
                          <Label htmlFor="email">Email</Label>
                          <Input
                            id="email"
                            type="email"
                            value={newUser.email}
                            onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                            placeholder="Enter email"
                          />
                        </div>
                        <div>
                          <Label htmlFor="company">Company</Label>
                          <Input
                            id="company"
                            value={newUser.company}
                            onChange={(e) => setNewUser({ ...newUser, company: e.target.value })}
                            placeholder="Enter company"
                          />
                        </div>
                        <div>
                          <Label htmlFor="password">Password</Label>
                          <Input
                            id="password"
                            type="password"
                            value={newUser.password}
                            onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                            placeholder="Enter password"
                          />
                        </div>
                        <div>
                          <Label htmlFor="role">Role</Label>
                          <Select
                            value={newUser.role}
                            onValueChange={(value: "admin" | "user") => setNewUser({ ...newUser, role: value })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="user">
                                <div className="flex items-center">
                                  <User className="w-4 h-4 mr-2" />
                                  Regular User
                                </div>
                              </SelectItem>
                              <SelectItem value="admin">
                                <div className="flex items-center">
                                  <Crown className="w-4 h-4 mr-2" />
                                  Admin User
                                </div>
                              </SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                      </div>
                      
                      {/* Document Assignment Section - Only show for regular users */}
                      {newUser.role === "user" && (
                        <div className="space-y-4 p-4 bg-gray-50 rounded-lg border border-gray-200">
                          <div className="flex items-center space-x-2">
                            <FileCheck className="w-5 h-5 text-purple-600" />
                            <Label className="text-base font-medium">Document Access</Label>
                          </div>
                          
                          <div>
                            <Label className="text-sm text-gray-600 mb-2 block">
                              Select documents this user can access (optional)
                            </Label>
                            <div className="max-h-40 overflow-y-auto border rounded-lg p-3 bg-white">
                              {documents.length > 0 ? (
                                documents.map((doc) => (
                                  <div key={doc.id} className="flex items-center space-x-2 py-1">
                                    <Checkbox
                                      id={`doc-${doc.id}`}
                                      checked={newUser.assigned_document_ids.includes(doc.id)}
                                      onCheckedChange={(checked) => {
                                        if (checked) {
                                          setNewUser({
                                            ...newUser,
                                            assigned_document_ids: [...newUser.assigned_document_ids, doc.id]
                                          })
                                        } else {
                                          setNewUser({
                                            ...newUser,
                                            assigned_document_ids: newUser.assigned_document_ids.filter(id => id !== doc.id)
                                          })
                                        }
                                      }}
                                    />
                                    <Label htmlFor={`doc-${doc.id}`} className="flex items-center space-x-2 cursor-pointer text-sm">
                                      <FileText className="w-4 h-4 text-gray-500" />
                                      <span>{doc.name}</span>
                                      <span className="text-xs text-gray-400">
                                        ({(doc.size / 1024 / 1024).toFixed(1)} MB)
                                      </span>
                                    </Label>
                                  </div>
                                ))
                              ) : (
                                <p className="text-sm text-gray-500 py-2">No documents available</p>
                              )}
                            </div>
                          </div>
                          
                          {newUser.assigned_document_ids.length > 0 && (
                            <div>
                              <Label className="text-sm text-gray-600 mb-2 block">Permissions</Label>
                              <div className="flex flex-wrap gap-2">
                                {["read", "download", "search", "query"].map((permission) => (
                                  <div key={permission} className="flex items-center space-x-1">
                                    <Checkbox
                                      id={`new-user-perm-${permission}`}
                                      checked={newUser.document_permissions.includes(permission)}
                                      onCheckedChange={(checked) => {
                                        if (checked) {
                                          setNewUser({
                                            ...newUser,
                                            document_permissions: [...newUser.document_permissions, permission]
                                          })
                                        } else {
                                          setNewUser({
                                            ...newUser,
                                            document_permissions: newUser.document_permissions.filter(p => p !== permission)
                                          })
                                        }
                                      }}
                                    />
                                    <Label htmlFor={`new-user-perm-${permission}`} className="text-sm capitalize cursor-pointer">
                                      {permission}
                                    </Label>
                                  </div>
                                ))}
                              </div>
                            </div>
                          )}
                          
                          {newUser.assigned_document_ids.length > 0 && (
                            <div className="text-sm text-blue-600 bg-blue-50 p-2 rounded border border-blue-200">
                              <strong>{newUser.assigned_document_ids.length}</strong> document(s) selected with{" "}
                              <strong>{newUser.document_permissions.join(", ")}</strong> permissions
                            </div>
                          )}
                        </div>
                      )}

                      <Button 
                        onClick={handleAddUser}
                        disabled={!newUser.first_name || !newUser.last_name || !newUser.email || !newUser.password || isAddingUser}
                        className="w-full bg-gradient-to-r from-green-600 to-green-700 hover:from-green-700 hover:to-green-800 disabled:opacity-50"
                      >
                        {isAddingUser ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                            Creating User...
                          </>
                        ) : (
                          <>
                            <Plus className="w-4 h-4 mr-2" />
                            Create {newUser.role === 'admin' ? 'Admin' : 'User'}
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>

                  {/* User Statistics */}
                  <Card className="shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <BarChart3 className="w-5 h-5 mr-2 text-blue-600" />
                        User Statistics
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {/* System Overview - Compact Grid */}
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                        <div className="text-center p-3 bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg border border-blue-200">
                          <Users className="h-5 w-5 text-blue-600 mx-auto mb-1" />
                          <div className="text-xl font-bold text-blue-900">{totalUsers}</div>
                          <div className="text-xs text-blue-600">{activeUsers} active</div>
                        </div>
                        <div className="text-center p-3 bg-gradient-to-br from-green-50 to-green-100 rounded-lg border border-green-200">
                          <FileText className="h-5 w-5 text-green-600 mx-auto mb-1" />
                          <div className="text-xl font-bold text-green-900">{totalDocuments}</div>
                          <div className="text-xs text-green-600">{processedDocuments} processed</div>
                        </div>
                        <div className="text-center p-3 bg-gradient-to-br from-orange-50 to-orange-100 rounded-lg border border-orange-200">
                          <HardDrive className="h-5 w-5 text-orange-600 mx-auto mb-1" />
                          <div className="text-xl font-bold text-orange-900">{storageInGB} GB</div>
                          <div className="text-xs text-orange-600">Total storage</div>
                        </div>
                        <div className="text-center p-3 bg-gradient-to-br from-indigo-50 to-indigo-100 rounded-lg border border-indigo-200">
                          <Building className="h-5 w-5 text-indigo-600 mx-auto mb-1" />
                          <div className="text-xl font-bold text-indigo-900">{totalTenants}</div>
                          <div className="text-xs text-indigo-600">{activeTenants} active</div>
                        </div>
                      </div>

                      {/* User Role Breakdown */}
                      <div className="grid grid-cols-2 gap-3">
                        <div className="text-center p-3 bg-blue-50 rounded-lg border border-blue-200">
                          <div className="text-lg font-bold text-blue-900">{adminUsers}</div>
                          <div className="text-xs text-blue-600 flex items-center justify-center">
                            <Crown className="w-3 h-3 mr-1" />
                            Admin Users
                          </div>
                        </div>
                        <div className="text-center p-3 bg-green-50 rounded-lg border border-green-200">
                          <div className="text-lg font-bold text-green-900">{regularUsers}</div>
                          <div className="text-xs text-green-600 flex items-center justify-center">
                            <User className="w-3 h-3 mr-1" />
                            Regular Users
                          </div>
                        </div>
                      </div>

                      {/* System Status - Compact */}
                      <div className="space-y-2">
                        <h4 className="text-sm font-medium text-gray-800 flex items-center">
                          <Shield className="w-4 h-4 mr-2 text-blue-600" />
                          System Status
                        </h4>
                        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
                          <div className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <span className="text-xs font-medium text-green-800">API</span>
                            <Badge className="bg-green-100 text-green-800 border-green-300 text-xs px-1 py-0">Running</Badge>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <span className="text-xs font-medium text-green-800">Database</span>
                            <Badge className="bg-green-100 text-green-800 border-green-300 text-xs px-1 py-0">Connected</Badge>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <span className="text-xs font-medium text-green-800">AI Models</span>
                            <Badge className="bg-green-100 text-green-800 border-green-300 text-xs px-1 py-0">Ready</Badge>
                          </div>
                          <div className="flex items-center justify-between p-2 bg-green-50 rounded border border-green-200">
                            <span className="text-xs font-medium text-green-800">Security</span>
                            <Badge className="bg-green-100 text-green-800 border-green-300 text-xs px-1 py-0">Active</Badge>
                          </div>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>

                {/* Users List */}
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Users className="w-5 h-5 mr-2 text-green-600" />
                      Users
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Company</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Documents</TableHead>
                          <TableHead>Last Login</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredUsers.map((user) => (
                          <TableRow key={user.id}>
                            <TableCell className="font-medium">{user.name}</TableCell>
                            <TableCell>{user.email}</TableCell>
                            <TableCell>{user.company || "N/A"}</TableCell>
                            <TableCell>
                              <Badge variant={user.role === "admin" ? "default" : "secondary"} className="flex items-center w-fit">
                                {user.role === "admin" ? (
                                  <>
                                    <Crown className="w-3 h-3 mr-1" />
                                    Admin
                                  </>
                                ) : (
                                  <>
                                    <User className="w-3 h-3 mr-1" />
                                    User
                                  </>
                                )}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <Badge variant={user.status === "active" ? "default" : "secondary"}>
                                {user.status}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex items-center space-x-2">
                                <span>{user.documentsCount}</span>
                                {user.role !== "admin" && (
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => loadUserDocuments(user.id)}
                                    disabled={loadingUserDocuments === user.id}
                                    className="text-blue-600 hover:text-blue-700 p-1 h-6"
                                    title="View assigned documents"
                                  >
                                    {loadingUserDocuments === user.id ? (
                                      <div className="animate-spin rounded-full h-3 w-3 border-2 border-blue-600 border-t-transparent"></div>
                                    ) : (
                                      <FileText className="h-3 w-3" />
                                    )}
                                  </Button>
                                )}
                              </div>
                              {userDocuments[user.id] && userDocuments[user.id].length > 0 && (
                                <div className="mt-2 space-y-1">
                                  {userDocuments[user.id].slice(0, 3).map((doc) => (
                                    <div key={doc.id} className="flex items-center justify-between text-xs bg-gray-50 p-2 rounded border">
                                      <div className="flex items-center space-x-1 flex-1">
                                        <Badge 
                                          variant={doc.assignment_type === "owner" ? "default" : "secondary"}
                                          className="text-xs px-1 py-0"
                                        >
                                          {doc.assignment_type === "owner" ? "Own" : "Assigned"}
                                        </Badge>
                                        <span className="text-gray-600 truncate max-w-[100px]" title={doc.name}>
                                          {doc.name}
                                        </span>
                                        <span className="text-gray-400">
                                          ({(doc.size / 1024 / 1024).toFixed(1)}MB)
                                        </span>
                                      </div>
                                      {doc.assignment_type === "assigned" && (
                                        <AlertDialog>
                                          <AlertDialogTrigger asChild>
                                            <Button
                                              variant="ghost"
                                              size="sm"
                                              className="text-red-600 hover:text-red-700 hover:bg-red-50 p-1 h-6 w-6"
                                              title="Unassign document"
                                              disabled={unassigningDocument === `${doc.id}-${user.id}`}
                                            >
                                              {unassigningDocument === `${doc.id}-${user.id}` ? (
                                                <div className="animate-spin rounded-full h-3 w-3 border-2 border-red-600 border-t-transparent"></div>
                                              ) : (
                                                <XCircle className="h-3 w-3" />
                                              )}
                                            </Button>
                                          </AlertDialogTrigger>
                                          <AlertDialogContent>
                                            <AlertDialogHeader>
                                              <AlertDialogTitle>Unassign Document</AlertDialogTitle>
                                              <AlertDialogDescription>
                                                Are you sure you want to unassign <strong>"{doc.name}"</strong> from <strong>{user.name}</strong>?
                                                <br /><br />
                                                This will remove the user's access to this document.
                                              </AlertDialogDescription>
                                            </AlertDialogHeader>
                                            <AlertDialogFooter>
                                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                                              <AlertDialogAction
                                                onClick={() => handleUnassignDocument(doc.id, user.id, doc.name)}
                                                className="bg-red-600 hover:bg-red-700 text-white"
                                                disabled={unassigningDocument === `${doc.id}-${user.id}`}
                                              >
                                                {unassigningDocument === `${doc.id}-${user.id}` ? (
                                                  <>
                                                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                                                    Unassigning...
                                                  </>
                                                ) : (
                                                  "Unassign Document"
                                                )}
                                              </AlertDialogAction>
                                            </AlertDialogFooter>
                                          </AlertDialogContent>
                                        </AlertDialog>
                                      )}
                                    </div>
                                  ))}
                                  {userDocuments[user.id].length > 3 && (
                                    <div className="text-xs text-gray-500 p-2">
                                      +{userDocuments[user.id].length - 3} more documents
                                      <Button
                                        variant="ghost"
                                        size="sm"
                                        className="text-blue-600 hover:text-blue-700 ml-2 p-0 h-auto text-xs"
                                        onClick={() => {
                                          const newExpanded = new Set(expandedUserDocuments)
                                          if (expandedUserDocuments.has(user.id)) {
                                            newExpanded.delete(user.id)
                                          } else {
                                            newExpanded.add(user.id)
                                          }
                                          setExpandedUserDocuments(newExpanded)
                                        }}
                                      >
                                        {expandedUserDocuments.has(user.id) ? "Show Less" : "View All"}
                                      </Button>
                                    </div>
                                  )}
                                  {expandedUserDocuments.has(user.id) && userDocuments[user.id].length > 3 && (
                                    <div className="mt-1 space-y-1">
                                      {userDocuments[user.id].slice(3).map((doc) => (
                                        <div key={doc.id} className="flex items-center justify-between text-xs bg-gray-50 p-2 rounded border">
                                          <div className="flex items-center space-x-1 flex-1">
                                            <Badge 
                                              variant={doc.assignment_type === "owner" ? "default" : "secondary"}
                                              className="text-xs px-1 py-0"
                                            >
                                              {doc.assignment_type === "owner" ? "Own" : "Assigned"}
                                            </Badge>
                                            <span className="text-gray-600 truncate max-w-[100px]" title={doc.name}>
                                              {doc.name}
                                            </span>
                                            <span className="text-gray-400">
                                              ({(doc.size / 1024 / 1024).toFixed(1)}MB)
                                            </span>
                                          </div>
                                          {doc.assignment_type === "assigned" && (
                                            <AlertDialog>
                                              <AlertDialogTrigger asChild>
                                                <Button
                                                  variant="ghost"
                                                  size="sm"
                                                  className="text-red-600 hover:text-red-700 hover:bg-red-50 p-1 h-6 w-6"
                                                  title="Unassign document"
                                                  disabled={unassigningDocument === `${doc.id}-${user.id}`}
                                                >
                                                  {unassigningDocument === `${doc.id}-${user.id}` ? (
                                                    <div className="animate-spin rounded-full h-3 w-3 border-2 border-red-600 border-t-transparent"></div>
                                                  ) : (
                                                    <XCircle className="h-3 w-3" />
                                                  )}
                                                </Button>
                                              </AlertDialogTrigger>
                                              <AlertDialogContent>
                                                <AlertDialogHeader>
                                                  <AlertDialogTitle>Unassign Document</AlertDialogTitle>
                                                  <AlertDialogDescription>
                                                    Are you sure you want to unassign <strong>"{doc.name}"</strong> from <strong>{user.name}</strong>?
                                                    <br /><br />
                                                    This will remove the user's access to this document.
                                                  </AlertDialogDescription>
                                                </AlertDialogHeader>
                                                <AlertDialogFooter>
                                                  <AlertDialogCancel>Cancel</AlertDialogCancel>
                                                  <AlertDialogAction
                                                    onClick={() => handleUnassignDocument(doc.id, user.id, doc.name)}
                                                    className="bg-red-600 hover:bg-red-700 text-white"
                                                    disabled={unassigningDocument === `${doc.id}-${user.id}`}
                                                  >
                                                    {unassigningDocument === `${doc.id}-${user.id}` ? (
                                                      <>
                                                        <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                                                        Unassigning...
                                                      </>
                                                    ) : (
                                                      "Unassign Document"
                                                    )}
                                                  </AlertDialogAction>
                                                </AlertDialogFooter>
                                              </AlertDialogContent>
                                            </AlertDialog>
                                          )}
                                        </div>
                                      ))}
                                    </div>
                                  )}
                                </div>
                              )}
                            </TableCell>
                            <TableCell>{user.lastLogin}</TableCell>
                            <TableCell>
                              <div className="flex space-x-2">
                                <Button variant="ghost" size="sm" title="View Details">
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button variant="ghost" size="sm" title="Edit User">
                                  <Edit className="h-4 w-4" />
                                </Button>
                                {canDeleteUser(user) ? (
                                  <AlertDialog>
                                    <AlertDialogTrigger asChild>
                                      <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700 hover:bg-red-50" title="Delete User">
                                        <Trash2 className="h-4 w-4" />
                                      </Button>
                                    </AlertDialogTrigger>
                                    <AlertDialogContent>
                                      <AlertDialogHeader>
                                        <AlertDialogTitle>Delete User Account</AlertDialogTitle>
                                        <AlertDialogDescription asChild>
                                          <div className="space-y-4">
                                            <p>
                                              Are you sure you want to delete <strong>{user.name}</strong> ({user.email})?
                                            </p>
                                            <div>
                                              <p className="text-red-600 font-medium mb-2">
                                                This action cannot be undone. This will permanently delete:
                                              </p>
                                              <ul className="text-sm text-gray-600 list-disc list-inside space-y-1">
                                                <li>User account and profile information</li>
                                                <li>All uploaded documents ({user.documentsCount} documents)</li>
                                                <li>All AI curations and summaries</li>
                                                <li>All processing history and settings</li>
                                                <li>All active sessions and tokens</li>
                                              </ul>
                                            </div>
                                          </div>
                                        </AlertDialogDescription>
                                      </AlertDialogHeader>
                                      <AlertDialogFooter>
                                        <AlertDialogCancel>Cancel</AlertDialogCancel>
                                        <AlertDialogAction
                                          onClick={() => handleDeleteUser(user.id)}
                                          className="bg-red-600 hover:bg-red-700 text-white"
                                          disabled={deletingUserId === user.id}
                                        >
                                          {deletingUserId === user.id ? (
                                            <>
                                              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                                              Deleting...
                                            </>
                                          ) : (
                                            "Delete Account"
                                          )}
                                        </AlertDialogAction>
                                      </AlertDialogFooter>
                                    </AlertDialogContent>
                                  </AlertDialog>
                                ) : (
                                  <Button 
                                    variant="ghost" 
                                    size="sm" 
                                    className="text-gray-400 cursor-not-allowed" 
                                    title={getDeleteRestrictionMessage(user) || "Cannot delete this user"}
                                    disabled
                                  >
                                    <Trash2 className="h-4 w-4" />
                                  </Button>
                                )}
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Documents & Access Tab */}
              <TabsContent value="documents" className="space-y-6">
                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <FileText className="w-5 h-5 mr-2 text-purple-600" />
                      Documents
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Document Name</TableHead>
                          <TableHead>Owner</TableHead>
                          <TableHead>Size</TableHead>
                          <TableHead>Upload Date</TableHead>
                          <TableHead>Access Control</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredDocuments.map((doc) => (
                          <TableRow key={doc.id}>
                            <TableCell className="font-medium">
                              <div className="flex items-center space-x-2">
                                <FileText className="w-4 h-4 text-gray-500" />
                                <span>{doc.name}</span>
                              </div>
                            </TableCell>
                            <TableCell>
                              {doc.owner ? (
                                <div>
                                  <div className="font-medium">{doc.owner.name}</div>
                                  <div className="text-sm text-gray-500">{doc.owner.email}</div>
                                </div>
                              ) : (
                                "Unknown"
                              )}
                            </TableCell>
                            <TableCell>{(doc.size / 1024 / 1024).toFixed(2)} MB</TableCell>
                            <TableCell>{doc.uploadDate}</TableCell>
                            <TableCell>
                              <div className="flex items-center space-x-2">
                                <Badge variant="outline" className="text-xs">
                                  <Lock className="w-3 h-3 mr-1" />
                                  Admin Only
                                </Badge>
                                <Button 
                                  variant="ghost" 
                                  size="sm"
                                  onClick={() => setSelectedDocument(doc)}
                                  className="text-blue-600 hover:text-blue-700"
                                >
                                  <Share className="w-3 h-3 mr-1" />
                                  Assign
                                </Button>
                              </div>
                            </TableCell>
                            <TableCell>
                              <div className="flex space-x-2">
                                <Button variant="ghost" size="sm" title="View Document">
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <AlertDialog open={deleteDialogOpen === doc.id} onOpenChange={(open) => !open && deletingDocumentId !== doc.id && setDeleteDialogOpen(null)}>
                                  <AlertDialogTrigger asChild>
                                    <Button 
                                      variant="ghost" 
                                      size="sm" 
                                      className="text-red-600 hover:text-red-700 hover:bg-red-50" 
                                      title="Delete Document"
                                      onClick={() => setDeleteDialogOpen(doc.id)}
                                    >
                                      <Trash2 className="h-4 w-4" />
                                    </Button>
                                  </AlertDialogTrigger>
                                  <AlertDialogContent>
                                    <AlertDialogHeader>
                                      <AlertDialogTitle>Delete Document</AlertDialogTitle>
                                      <AlertDialogDescription>
                                        Are you sure you want to delete <strong>"{doc.name}"</strong>?
                                        <br /><br />
                                        <span className="text-red-600 font-medium">
                                          This action cannot be undone. This will permanently delete the document and all associated data.
                                        </span>
                                      </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                      <AlertDialogCancel 
                                        disabled={deletingDocumentId === doc.id}
                                        onClick={() => setDeleteDialogOpen(null)}
                                      >
                                        Cancel
                                      </AlertDialogCancel>
                                      <AlertDialogAction
                                        onClick={async () => {
                                          await handleDeleteDocument(doc.id)
                                          setDeleteDialogOpen(null)
                                        }}
                                        className="bg-red-600 hover:bg-red-700 text-white"
                                        disabled={deletingDocumentId === doc.id}
                                      >
                                        {deletingDocumentId === doc.id ? (
                                          <>
                                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                                            Deleting...
                                          </>
                                        ) : (
                                          "Delete Document"
                                        )}
                                      </AlertDialogAction>
                                    </AlertDialogFooter>
                                  </AlertDialogContent>
                                </AlertDialog>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>

                {/* Document Assignment Dialog */}
                {selectedDocument && (
                  <Card className="shadow-lg border-2 border-blue-200">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Share className="w-5 h-5 mr-2 text-blue-600" />
                        Assign Access
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <h4 className="font-medium text-blue-800 mb-2">Document Details</h4>
                        <div className="text-sm text-blue-700">
                          <p><strong>Name:</strong> {selectedDocument.name}</p>
                          <p><strong>Size:</strong> {(selectedDocument.size / 1024 / 1024).toFixed(2)} MB</p>
                          <p><strong>Owner:</strong> {selectedDocument.owner?.name}</p>
                        </div>
                      </div>
                      
                      <div>
                        <Label>Select Users to Grant Access</Label>
                        <div className="mt-2 space-y-2 max-h-40 overflow-y-auto border rounded-lg p-3">
                          {filteredUsers.filter(u => u.role !== 'admin').map((user) => (
                            <div key={user.id} className="flex items-center space-x-2">
                              <Checkbox
                                id={`user-${user.id}`}
                                checked={selectedUsersForAssignment.includes(user.id)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setSelectedUsersForAssignment([...selectedUsersForAssignment, user.id])
                                  } else {
                                    setSelectedUsersForAssignment(selectedUsersForAssignment.filter(id => id !== user.id))
                                  }
                                }}
                              />
                              <Label htmlFor={`user-${user.id}`} className="flex items-center space-x-2 cursor-pointer">
                                <User className="w-4 h-4 text-gray-500" />
                                <span>{user.name} ({user.email})</span>
                              </Label>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div>
                        <Label>Permissions</Label>
                        <div className="mt-2 space-y-2">
                          {["read", "download", "search"].map((permission) => (
                            <div key={permission} className="flex items-center space-x-2">
                              <Checkbox
                                id={`perm-${permission}`}
                                checked={assignmentPermissions.includes(permission)}
                                onCheckedChange={(checked) => {
                                  if (checked) {
                                    setAssignmentPermissions([...assignmentPermissions, permission])
                                  } else {
                                    setAssignmentPermissions(assignmentPermissions.filter(p => p !== permission))
                                  }
                                }}
                              />
                              <Label htmlFor={`perm-${permission}`} className="capitalize cursor-pointer">
                                {permission}
                              </Label>
                            </div>
                          ))}
                        </div>
                      </div>
                      
                      <div className="flex space-x-3">
                        <Button 
                          onClick={handleAssignDocument}
                          disabled={selectedUsersForAssignment.length === 0 || assignmentPermissions.length === 0 || isAssigningDocument}
                          className="bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:opacity-50"
                        >
                          {isAssigningDocument ? (
                            <>
                              <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                              Assigning...
                            </>
                          ) : (
                            <>
                              <Share className="w-4 h-4 mr-2" />
                              Assign Access
                            </>
                          )}
                        </Button>
                        <Button 
                          variant="outline"
                          onClick={() => {
                            setSelectedDocument(null)
                            setSelectedUsersForAssignment([])
                            setAssignmentPermissions(["read"])
                          }}
                        >
                          Cancel
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>

              {/* Tenant Management Tab - Hidden for now */}
              {/* 
              <TabsContent value="tenants" className="space-y-6">
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  <Card className="shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Plus className="w-5 h-5 mr-2 text-indigo-600" />
                        Create Tenant
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="grid grid-cols-1 gap-4">
                        <div>
                          <Label htmlFor="tenant_slug">Tenant Slug</Label>
                          <Input
                            id="tenant_slug"
                            value={newTenant.slug}
                            onChange={(e) => setNewTenant({ ...newTenant, slug: e.target.value })}
                            placeholder="e.g., pepsi, kfc, airtel"
                          />
                        </div>
                        <div>
                          <Label htmlFor="tenant_name">Tenant Name</Label>
                          <Input
                            id="tenant_name"
                            value={newTenant.name}
                            onChange={(e) => setNewTenant({ ...newTenant, name: e.target.value })}
                            placeholder="e.g., PepsiCo, KFC Corporation"
                          />
                        </div>
                        <div>
                          <Label htmlFor="tenant_type">Tenant Type</Label>
                          <Select
                            value={newTenant.tenant_type}
                            onValueChange={(value: "enterprise" | "partner" | "public") => 
                              setNewTenant({ ...newTenant, tenant_type: value })}
                          >
                            <SelectTrigger>
                              <SelectValue />
                            </SelectTrigger>
                            <SelectContent>
                              <SelectItem value="enterprise">Enterprise</SelectItem>
                              <SelectItem value="partner">Partner</SelectItem>
                              <SelectItem value="public">Public</SelectItem>
                            </SelectContent>
                          </Select>
                        </div>
                        <div className="grid grid-cols-2 gap-4">
                          <div>
                            <Label htmlFor="primary_color">Primary Color</Label>
                            <Input
                              id="primary_color"
                              value={newTenant.primary_color}
                              onChange={(e) => setNewTenant({ ...newTenant, primary_color: e.target.value })}
                              placeholder="#004B93"
                            />
                          </div>
                          <div>
                            <Label htmlFor="secondary_color">Secondary Color</Label>
                            <Input
                              id="secondary_color"
                              value={newTenant.secondary_color}
                              onChange={(e) => setNewTenant({ ...newTenant, secondary_color: e.target.value })}
                              placeholder="#E32934"
                            />
                          </div>
                        </div>
                      </div>
                      <Button 
                        onClick={handleAddTenant}
                        disabled={!newTenant.slug || !newTenant.name || isAddingTenant}
                        className="w-full bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-700 hover:to-indigo-800 disabled:opacity-50"
                      >
                        {isAddingTenant ? (
                          <>
                            <div className="animate-spin rounded-full h-4 w-4 border-2 border-white border-t-transparent mr-2"></div>
                            Creating Tenant...
                          </>
                        ) : (
                          <>
                            <Plus className="w-4 h-4 mr-2" />
                            Create Tenant
                          </>
                        )}
                      </Button>
                    </CardContent>
                  </Card>

                  <Card className="shadow-lg">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Database className="w-5 h-5 mr-2 text-blue-600" />
                        Quick Setup
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="p-4 bg-blue-50 rounded-lg border border-blue-200">
                        <h4 className="font-medium text-blue-800 mb-2">Default Tenants</h4>
                        <ul className="text-sm text-blue-700 space-y-1">
                          <li>• PepsiCo (Enterprise)</li>
                          <li>• KFC Corporation (Enterprise)</li>
                          <li>• Bharti Airtel (Enterprise)</li>
                          <li>• SapienPlus.ai (Partner)</li>
                          <li>• Public Access (Public)</li>
                        </ul>
                      </div>
                      <Button 
                        onClick={handleInitializeDefaultTenants}
                        variant="outline"
                        className="w-full border-blue-200 text-blue-600 hover:bg-blue-50"
                      >
                        <Database className="w-4 h-4 mr-2" />
                        Initialize Default Tenants
                      </Button>
                    </CardContent>
                  </Card>
                </div>

                <Card className="shadow-lg">
                  <CardHeader>
                    <CardTitle className="flex items-center">
                      <Building className="w-5 h-5 mr-2 text-indigo-600" />
                      Tenants
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Name</TableHead>
                          <TableHead>Slug</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead>Users</TableHead>
                          <TableHead>Documents</TableHead>
                          <TableHead>Storage</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {tenants.map((tenant) => (
                          <TableRow key={tenant.id}>
                            <TableCell className="font-medium">
                              <div className="flex items-center space-x-3">
                                {tenant.primary_color && (
                                  <div 
                                    className="w-4 h-4 rounded-full border border-gray-300"
                                    style={{ backgroundColor: tenant.primary_color }}
                                  />
                                )}
                                {tenant.name}
                              </div>
                            </TableCell>
                            <TableCell className="font-mono text-sm">{tenant.slug}</TableCell>
                            <TableCell>
                              <Badge variant={
                                tenant.tenant_type === "enterprise" ? "default" : 
                                tenant.tenant_type === "partner" ? "secondary" : "outline"
                              }>
                                {tenant.tenant_type}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              {tenant.current_users}
                              {tenant.max_users && ` / ${tenant.max_users}`}
                            </TableCell>
                            <TableCell>
                              {tenant.current_documents}
                              {tenant.max_documents && ` / ${tenant.max_documents}`}
                            </TableCell>
                            <TableCell>
                              {(tenant.current_storage_mb / 1024).toFixed(2)} GB
                              {tenant.max_storage_gb && ` / ${tenant.max_storage_gb} GB`}
                            </TableCell>
                            <TableCell>
                              <Badge variant={tenant.is_active ? "default" : "secondary"}>
                                {tenant.is_active ? "Active" : "Inactive"}
                              </Badge>
                            </TableCell>
                            <TableCell>
                              <div className="flex space-x-2">
                                <Button 
                                  variant="ghost" 
                                  size="sm"
                                  onClick={() => handleSelectTenant(tenant)}
                                  title="View Tenant Users"
                                >
                                  <Eye className="h-4 w-4" />
                                </Button>
                                <Button variant="ghost" size="sm" title="Edit Tenant">
                                  <Edit className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>

                {selectedTenant && (
                  <Card className="shadow-lg border-2 border-indigo-200">
                    <CardHeader>
                      <CardTitle className="flex items-center">
                        <Users className="w-5 h-5 mr-2 text-green-600" />
                        {selectedTenant.name} - Users
                      </CardTitle>
                    </CardHeader>
                    <CardContent>
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Name</TableHead>
                            <TableHead>Email</TableHead>
                            <TableHead>Role</TableHead>
                            <TableHead>Status</TableHead>
                            <TableHead>Documents</TableHead>
                            <TableHead>Storage Used</TableHead>
                            <TableHead>Joined</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {tenantUsers.map((tenantUser) => (
                            <TableRow key={tenantUser.id}>
                              <TableCell className="font-medium">
                                {tenantUser.first_name} {tenantUser.last_name}
                              </TableCell>
                              <TableCell>{tenantUser.email}</TableCell>
                              <TableCell>
                                <Badge variant={tenantUser.tenant_role === "admin" ? "default" : "secondary"}>
                                  {tenantUser.tenant_role}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                <Badge variant={tenantUser.status === "active" ? "default" : "secondary"}>
                                  {tenantUser.status}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                {tenantUser.document_count}
                                {tenantUser.document_limit && ` / ${tenantUser.document_limit}`}
                              </TableCell>
                              <TableCell>
                                {(tenantUser.storage_used_mb / 1024).toFixed(2)} GB
                                {tenantUser.storage_limit_mb && ` / ${(tenantUser.storage_limit_mb / 1024).toFixed(2)} GB`}
                              </TableCell>
                              <TableCell>
                                {new Date(tenantUser.joined_at).toLocaleDateString()}
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    </CardContent>
                  </Card>
                )}
              </TabsContent>
              */}
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  )
}
