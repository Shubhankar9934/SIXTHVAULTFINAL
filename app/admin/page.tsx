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
import { Users, FileText, BarChart3, Shield, Trash2, Edit, Brain, AlertCircle, LogOut, Home } from "lucide-react"
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

export default function AdminPage() {
  const { logout, user, isAuthenticated } = useAuth()
  const [users, setUsers] = useState<User[]>([])
  const [documents, setDocuments] = useState<Document[]>([])
  const [systemStats, setSystemStats] = useState<any>(null)
  const [loading, setLoading] = useState(true)
  const [accessDenied, setAccessDenied] = useState(false)
  const [newUser, setNewUser] = useState({
    first_name: "",
    last_name: "",
    email: "",
    username: "",
    password: "",
    role: "user" as "admin" | "user",
  })
  const { toast } = useToast()

  // Check admin access on mount
  useEffect(() => {
    if (!isAuthenticated) {
      setLoading(false)
      return
    }

    // Check if user has admin privileges
    if (!user || !user.is_admin || user.role !== 'admin') {
      console.log('Admin page: Access denied - user is not admin:', user)
      setAccessDenied(true)
      setLoading(false)
      return
    }

    console.log('Admin page: Admin access confirmed for user:', user.email)
    setAccessDenied(false)
  }, [isAuthenticated, user])

  // Load real data on mount
  useEffect(() => {
    if (!isAuthenticated || !user || accessDenied) {
      return
    }

    const loadData = async () => {
      try {
        setLoading(true)
        console.log('Admin page: Starting to load data...')
        
        // Check if user is authenticated and has admin role
        const getAuthCookie = () => {
          if (typeof window === 'undefined') return null
          return document.cookie.split('; ').find(row => row.startsWith('auth-token='))?.split('=')[1]
        }
        
        const token = getAuthCookie()
        console.log('Admin page: Auth token present:', !!token)
        
        if (!token) {
          throw new Error('No authentication token found. Please log in again.')
        }
        
        // Load users from backend
        console.log('Admin page: Loading users...')
        const usersData = await adminAPI.getUsers()
        console.log('Admin page: Users loaded successfully:', usersData.length, 'users')
        setUsers(usersData)
        
        // Load documents from backend
        console.log('Admin page: Loading documents...')
        const documentsData = await adminAPI.getDocuments()
        console.log('Admin page: Documents loaded successfully:', documentsData.length, 'documents')
        setDocuments(documentsData)
        
        // Load system stats
        console.log('Admin page: Loading system stats...')
        const statsData = await adminAPI.getSystemStats()
        console.log('Admin page: System stats loaded successfully:', statsData)
        setSystemStats(statsData)
        
      } catch (error) {
        console.error('Admin page: Failed to load admin data:', error)
        
        // Show more specific error messages
        let errorMessage = "Failed to load admin data. Some features may not work properly."
        if (error instanceof Error) {
          if (error.message.includes('No authentication token')) {
            errorMessage = "Authentication required. Please log in again."
          } else if (error.message.includes('Failed to fetch users')) {
            errorMessage = "Failed to load users. Please check your admin permissions."
          } else if (error.message.includes('403') || error.message.includes('Forbidden')) {
            errorMessage = "Access denied. Admin permissions required."
          } else if (error.message.includes('401') || error.message.includes('Unauthorized')) {
            errorMessage = "Authentication expired. Please log in again."
          }
        }
        
        toast({
          title: "Error",
          description: errorMessage,
          variant: "destructive",
        })
        
        // Set fallback data
        setUsers([])
        setDocuments([])
        setSystemStats({
          users: { total: 0, active: 0, inactive: 0 },
          documents: { total: 0, processing: 0, completed: 0 },
          storage: { total_bytes: 0, total_gb: 0 },
          ai: { curations: 0, summaries: 0 }
        })
      } finally {
        setLoading(false)
      }
    }
    
    loadData()
  }, [isAuthenticated, user, accessDenied, toast])

  const handleAddUser = async () => {
    if (newUser.first_name && newUser.last_name && newUser.email && newUser.password) {
      try {
        const userData = {
          name: `${newUser.first_name} ${newUser.last_name}`,
          email: newUser.email,
          role: newUser.role,
          password: newUser.password,
          first_name: newUser.first_name,
          last_name: newUser.last_name,
          username: newUser.username
        }
        
        await adminAPI.createUser(userData)
        
        // Reload users
        const usersData = await adminAPI.getUsers()
        setUsers(usersData)
        
        setNewUser({ 
          first_name: "", 
          last_name: "", 
          email: "", 
          username: "", 
          password: "", 
          role: "user" 
        })
        
        toast({
          title: "Success",
          description: "User created successfully",
        })
      } catch (error: any) {
        toast({
          title: "Error",
          description: error.message || "Failed to create user",
          variant: "destructive",
        })
      }
    }
  }

  const handleDeleteUser = async (userId: string) => {
    try {
      const result = await adminAPI.deleteUser(userId)
      
      // Remove user from local state
      setUsers(users.filter((user) => user.id !== userId))
      
      // Show detailed success message
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
    }
  }

  const handleDeleteDocument = async (docId: string) => {
    try {
      const result = await adminAPI.deleteDocument(docId)
      
      // Remove document from local state
      setDocuments((prev) => prev.filter((doc) => doc.id !== docId))
      
      // Show detailed success message
      const deletedItems = []
      if (result.file_deleted) deletedItems.push("physical file")
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
    }
  }

  const totalUsers = users.length
  const activeUsers = users.filter((u) => u.status === "active").length
  const totalDocuments = documents.length
  const processedDocuments = documents.length // All documents in store are processed

  // Calculate total storage used
  const totalStorage = documents.reduce((sum, doc) => sum + doc.size, 0)
  const storageInGB = (totalStorage / (1024 * 1024 * 1024)).toFixed(2)

  // Show access denied screen for non-admin users
  if (accessDenied) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <Card className="w-full max-w-md">
          <CardHeader className="text-center">
            <div className="mx-auto w-12 h-12 bg-red-100 rounded-full flex items-center justify-center mb-4">
              <AlertCircle className="w-6 h-6 text-red-600" />
            </div>
            <CardTitle className="text-xl font-bold text-gray-900">Access Denied</CardTitle>
            <CardDescription className="text-gray-600">
              You don't have admin privileges to access this page.
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <p className="text-sm text-gray-500">
              Only users with admin role can access the admin panel. If you believe this is an error, please contact your system administrator.
            </p>
            <div className="flex flex-col space-y-2">
              <Link href="/vault">
                <Button className="w-full">
                  <Home className="w-4 h-4 mr-2" />
                  Go to Vault
                </Button>
              </Link>
              <Button variant="outline" onClick={logout} className="w-full">
                <LogOut className="w-4 h-4 mr-2" />
                Logout
              </Button>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  // Show enhanced loading screen with better UX
  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50/20 to-indigo-50/10 flex items-center justify-center relative overflow-hidden">
        {/* Beautiful flowing wave background */}
        <div className="absolute inset-0">
          <svg className="absolute inset-0 w-full h-full" viewBox="0 0 1440 800" preserveAspectRatio="xMidYMid slice">
            <defs>
              <linearGradient id="loadingWave1" x1="0%" y1="0%" x2="100%" y2="100%">
                <stop offset="0%" stopColor="#e0f2fe" stopOpacity="0.8"/>
                <stop offset="25%" stopColor="#b3e5fc" stopOpacity="0.6"/>
                <stop offset="50%" stopColor="#81d4fa" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#4fc3f7" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#29b6f6" stopOpacity="0.2"/>
              </linearGradient>
              <linearGradient id="loadingWave2" x1="100%" y1="0%" x2="0%" y2="100%">
                <stop offset="0%" stopColor="#f3e5f5" stopOpacity="0.7"/>
                <stop offset="25%" stopColor="#e1bee7" stopOpacity="0.5"/>
                <stop offset="50%" stopColor="#ce93d8" stopOpacity="0.4"/>
                <stop offset="75%" stopColor="#ba68c8" stopOpacity="0.3"/>
                <stop offset="100%" stopColor="#ab47bc" stopOpacity="0.2"/>
              </linearGradient>
            </defs>
            
            <g stroke="url(#loadingWave1)" strokeWidth="1.5" fill="none" opacity="0.8">
              <path d="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160">
                <animate attributeName="d" dur="8s" repeatCount="indefinite"
                  values="M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160;
                          M0,220 Q240,140 480,200 T960,180 Q1200,160 1440,200;
                          M0,200 Q200,120 400,180 T800,160 Q1000,140 1200,180 T1440,160"/>
              </path>
            </g>
            
            <g stroke="url(#loadingWave2)" strokeWidth="1.2" fill="none" opacity="0.7">
              <path d="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280">
                <animate attributeName="d" dur="6s" repeatCount="indefinite"
                  values="M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280;
                          M0,320 Q360,240 720,300 T1440,280;
                          M0,300 Q300,220 600,280 T1200,260 Q1320,240 1440,280"/>
              </path>
            </g>
            
            <path d="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z" fill="url(#loadingWave1)" opacity="0.1">
              <animate attributeName="d" dur="10s" repeatCount="indefinite"
                values="M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z;
                        M0,270 Q240,190 480,250 T960,230 Q1200,210 1440,250 L1440,800 L0,800 Z;
                        M0,250 Q200,170 400,230 T800,210 Q1000,190 1200,230 T1440,210 L1440,800 L0,800 Z"/>
            </path>
          </svg>
        </div>

        <Card className="w-full max-w-lg relative z-10 border-0 shadow-2xl bg-white/95 backdrop-blur-xl">
          <CardContent className="text-center p-12">
            {/* Enhanced Logo Section */}
            <div className="mb-8">
              <div className="w-20 h-20 bg-gradient-to-br from-blue-100 to-indigo-100 rounded-2xl flex items-center justify-center mx-auto mb-6 shadow-lg">
                <Shield className="w-10 h-10 text-blue-600" />
              </div>
              <SixthvaultLogo size="full" />
            </div>

            {/* Enhanced Loading Animation */}
            <div className="mb-8">
              <div className="relative">
                {/* Main spinner */}
                <div className="animate-spin rounded-full h-16 w-16 border-4 border-blue-200 border-t-blue-600 mx-auto mb-4"></div>
                
                {/* Pulsing rings */}
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-20 h-20 border-2 border-blue-300 rounded-full animate-ping opacity-20"></div>
                </div>
                <div className="absolute inset-0 flex items-center justify-center">
                  <div className="w-24 h-24 border-2 border-indigo-300 rounded-full animate-ping opacity-10" style={{animationDelay: '0.5s'}}></div>
                </div>
              </div>
            </div>

            {/* Enhanced Loading Text */}
            <div className="space-y-4">
              <h2 className="text-2xl font-bold bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 bg-clip-text text-transparent">
                Initializing Admin Panel
              </h2>
              
              <div className="space-y-2">
                <p className="text-lg font-semibold text-slate-700">
                  Setting up administrative interface...
                </p>
                <p className="text-sm text-slate-500">
                  Loading user management, system analytics, and AI configurations
                </p>
              </div>

              {/* Progress Steps */}
              <div className="mt-8 space-y-3">
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse"></div>
                    <span className="text-slate-600">Authenticating admin access</span>
                  </div>
                  <div className="text-blue-600 font-medium">✓</div>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-blue-600 rounded-full animate-pulse" style={{animationDelay: '0.2s'}}></div>
                    <span className="text-slate-600">Loading system data</span>
                  </div>
                  <div className="w-4 h-4 border-2 border-blue-600 border-t-transparent rounded-full animate-spin"></div>
                </div>
                
                <div className="flex items-center justify-between text-sm">
                  <div className="flex items-center space-x-2">
                    <div className="w-2 h-2 bg-slate-300 rounded-full"></div>
                    <span className="text-slate-400">Preparing dashboard</span>
                  </div>
                  <div className="text-slate-300">○</div>
                </div>
              </div>

              {/* Loading Progress Bar */}
              <div className="mt-6">
                <div className="w-full bg-slate-200 rounded-full h-2 overflow-hidden">
                  <div className="bg-gradient-to-r from-blue-600 to-indigo-600 h-2 rounded-full animate-pulse" 
                       style={{
                         width: '65%',
                         animation: 'loading-progress 3s ease-in-out infinite'
                       }}>
                  </div>
                </div>
                <p className="text-xs text-slate-500 mt-2">Please wait while we prepare your admin dashboard...</p>
              </div>

              {/* Security Notice */}
              <div className="mt-8 p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-xl border border-blue-200">
                <div className="flex items-center justify-center space-x-2 text-blue-700">
                  <Shield className="w-4 h-4" />
                  <span className="text-sm font-medium">Secure Admin Environment</span>
                </div>
                <p className="text-xs text-blue-600 mt-1">
                  Verifying administrative privileges and loading encrypted data
                </p>
              </div>
            </div>
          </CardContent>
        </Card>

        {/* Add custom CSS for loading animation */}
        <style jsx>{`
          @keyframes loading-progress {
            0% { width: 20%; }
            50% { width: 65%; }
            100% { width: 85%; }
          }
        `}</style>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex h-screen">
        {/* Left Sidebar */}
        <div className="w-80 bg-white border-r-4 border-blue-500 flex flex-col">
          <div className="p-6 border-b border-gray-200">
            <SixthvaultLogo size="full" />
          </div>

          <div className="p-6 flex-1">
            <h3 className="font-bold text-black mb-6 text-lg flex items-center">
              <Shield className="w-5 h-5 mr-2 text-blue-600" />
              AI ADMIN PANEL
            </h3>

            <div className="space-y-4 mb-8">
              <Card className="bg-blue-50 border-blue-200">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <Users className="h-8 w-8 text-blue-600" />
                    <div>
                      <p className="text-sm font-medium text-blue-800">Active Users</p>
                      <p className="text-2xl font-bold text-blue-900">{activeUsers}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-green-50 border-green-200">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <FileText className="h-8 w-8 text-green-600" />
                    <div>
                      <p className="text-sm font-medium text-green-800">AI Documents</p>
                      <p className="text-2xl font-bold text-green-900">{totalDocuments}</p>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card className="bg-purple-50 border-purple-200">
                <CardContent className="p-4">
                  <div className="flex items-center space-x-3">
                    <Brain className="h-8 w-8 text-purple-600" />
                    <div>
                      <p className="text-sm font-medium text-purple-800">AI Storage</p>
                      <p className="text-2xl font-bold text-purple-900">{storageInGB} GB</p>
                    </div>
                  </div>
                </CardContent>
              </Card>
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
          {/* Header */}
          <div className="p-6 border-b border-gray-200 bg-white flex justify-between items-center">
            <h1 className="text-2xl font-bold text-black flex items-center">
              <Brain className="w-6 h-6 mr-2 text-blue-600" />
              AI System Administration
            </h1>
            <Button 
              onClick={logout}
              className="bg-red-600 hover:bg-red-700 text-white px-4 py-2 rounded-lg font-medium transition-colors duration-200"
            >
              <LogOut className="w-4 h-4 mr-2" />
              Logout
            </Button>
          </div>

          {/* Admin Content */}
          <div className="flex-1 p-6 overflow-y-auto">
            <Tabs defaultValue="overview" className="space-y-6">
              <TabsList className="grid w-full grid-cols-4">
                <TabsTrigger value="overview">System Overview</TabsTrigger>
                <TabsTrigger value="users">User Management</TabsTrigger>
                <TabsTrigger value="documents">AI Documents</TabsTrigger>
                <TabsTrigger value="settings">AI Settings</TabsTrigger>
              </TabsList>

              <TabsContent value="overview" className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Total Users</CardTitle>
                      <Users className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{totalUsers}</div>
                      <p className="text-xs text-muted-foreground">{activeUsers} active users</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">AI Documents</CardTitle>
                      <FileText className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{totalDocuments}</div>
                      <p className="text-xs text-muted-foreground">{processedDocuments} AI-processed</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">Storage Used</CardTitle>
                      <BarChart3 className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold">{storageInGB} GB</div>
                      <p className="text-xs text-muted-foreground">AI-analyzed content</p>
                    </CardContent>
                  </Card>

                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                      <CardTitle className="text-sm font-medium">AI System</CardTitle>
                      <Brain className="h-4 w-4 text-muted-foreground" />
                    </CardHeader>
                    <CardContent>
                      <div className="text-2xl font-bold text-green-600">Active</div>
                      <p className="text-xs text-muted-foreground">Groq AI operational</p>
                    </CardContent>
                  </Card>
                </div>

                <Card>
                  <CardHeader>
                    <CardTitle>AI System Status</CardTitle>
                    <CardDescription>Real-time status of AI components</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <Brain className="h-5 w-5 text-green-600" />
                          <span className="font-medium">Groq AI Model</span>
                        </div>
                        <Badge className="bg-green-100 text-green-800">Operational</Badge>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <FileText className="h-5 w-5 text-green-600" />
                          <span className="font-medium">Document Processing</span>
                        </div>
                        <Badge className="bg-green-100 text-green-800">Active</Badge>
                      </div>
                      <div className="flex items-center justify-between p-3 bg-green-50 rounded-lg">
                        <div className="flex items-center space-x-3">
                          <BarChart3 className="h-5 w-5 text-green-600" />
                          <span className="font-medium">RAG System</span>
                        </div>
                        <Badge className="bg-green-100 text-green-800">Running</Badge>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="users" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>Add New User</CardTitle>
                    <CardDescription>Create a new user account for the AI system</CardDescription>
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
                        <Label htmlFor="username">Username (Optional)</Label>
                        <Input
                          id="username"
                          value={newUser.username}
                          onChange={(e) => setNewUser({ ...newUser, username: e.target.value })}
                          placeholder="Enter username"
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
                            <SelectItem value="user">User</SelectItem>
                            <SelectItem value="admin">Admin</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </div>
                    <Button 
                      onClick={handleAddUser}
                      disabled={!newUser.first_name || !newUser.last_name || !newUser.email || !newUser.password}
                    >
                      Add User
                    </Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>User Management</CardTitle>
                    <CardDescription>Manage system users and their access</CardDescription>
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
                          <TableHead>Last Login</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {users.map((user) => (
                          <TableRow key={user.id}>
                            <TableCell className="font-medium">{user.name}</TableCell>
                            <TableCell>{user.email}</TableCell>
                            <TableCell>
                              <Badge variant={user.role === "admin" ? "default" : "secondary"}>{user.role}</Badge>
                            </TableCell>
                            <TableCell>
                              <Badge variant={user.status === "active" ? "default" : "secondary"}>{user.status}</Badge>
                            </TableCell>
                            <TableCell>{user.documentsCount}</TableCell>
                            <TableCell>{user.lastLogin}</TableCell>
                            <TableCell>
                              <div className="flex space-x-2">
                                <Button variant="ghost" size="sm">
                                  <Edit className="h-4 w-4" />
                                </Button>
                                <AlertDialog>
                                  <AlertDialogTrigger asChild>
                                    <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700 hover:bg-red-50">
                                      <Trash2 className="h-4 w-4" />
                                    </Button>
                                  </AlertDialogTrigger>
                                  <AlertDialogContent>
                                    <AlertDialogHeader>
                                      <AlertDialogTitle>Delete User Account</AlertDialogTitle>
                                      <AlertDialogDescription>
                                        Are you sure you want to delete <strong>{user.name}</strong> ({user.email})?
                                        <br /><br />
                                        <span className="text-red-600 font-medium">
                                          This action cannot be undone. This will permanently delete:
                                        </span>
                                        <ul className="mt-2 text-sm text-gray-600 list-disc list-inside">
                                          <li>User account and profile information</li>
                                          <li>All uploaded documents ({user.documentsCount} documents)</li>
                                          <li>All AI curations and summaries</li>
                                          <li>All processing history and settings</li>
                                          <li>All active sessions and tokens</li>
                                        </ul>
                                      </AlertDialogDescription>
                                    </AlertDialogHeader>
                                    <AlertDialogFooter>
                                      <AlertDialogCancel>Cancel</AlertDialogCancel>
                                      <AlertDialogAction
                                        onClick={() => handleDeleteUser(user.id)}
                                        className="bg-red-600 hover:bg-red-700 text-white"
                                      >
                                        Delete Account
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
              </TabsContent>

              <TabsContent value="documents" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>AI Document Management</CardTitle>
                    <CardDescription>View and manage all AI-processed documents</CardDescription>
                  </CardHeader>
                  <CardContent>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Document Name</TableHead>
                          <TableHead>Size</TableHead>
                          <TableHead>Language</TableHead>
                          <TableHead>AI Themes</TableHead>
                          <TableHead>Upload Date</TableHead>
                          <TableHead>Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {documents.map((doc) => (
                          <TableRow key={doc.id}>
                            <TableCell className="font-medium">{doc.name}</TableCell>
                            <TableCell>{(doc.size / 1024 / 1024).toFixed(2)} MB</TableCell>
                            <TableCell>{doc.language}</TableCell>
                            <TableCell>
                              <div className="flex flex-wrap gap-1">
                                {doc.themes.slice(0, 2).map((theme, index) => (
                                  <Badge key={index} variant="outline" className="text-xs">
                                    {theme}
                                  </Badge>
                                ))}
                                {doc.themes.length > 2 && (
                                  <Badge variant="outline" className="text-xs">
                                    +{doc.themes.length - 2}
                                  </Badge>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>{doc.uploadDate}</TableCell>
                            <TableCell>
                              <AlertDialog>
                                <AlertDialogTrigger asChild>
                                  <Button variant="ghost" size="sm" className="text-red-600 hover:text-red-700 hover:bg-red-50">
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
                                        This action cannot be undone. This will permanently delete:
                                      </span>
                                      <ul className="mt-2 text-sm text-gray-600 list-disc list-inside">
                                        <li>The document file ({(doc.size / 1024 / 1024).toFixed(2)} MB)</li>
                                        <li>All AI-generated content and analysis</li>
                                        <li>Document curation and summary mappings</li>
                                        <li>Processing history and metadata</li>
                                      </ul>
                                      {doc.owner && (
                                        <p className="mt-2 text-sm text-gray-500">
                                          Owner: {doc.owner.name} ({doc.owner.email})
                                        </p>
                                      )}
                                    </AlertDialogDescription>
                                  </AlertDialogHeader>
                                  <AlertDialogFooter>
                                    <AlertDialogCancel>Cancel</AlertDialogCancel>
                                    <AlertDialogAction
                                      onClick={() => handleDeleteDocument(doc.id)}
                                      className="bg-red-600 hover:bg-red-700 text-white"
                                    >
                                      Delete Document
                                    </AlertDialogAction>
                                  </AlertDialogFooter>
                                </AlertDialogContent>
                              </AlertDialog>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </CardContent>
                </Card>
              </TabsContent>

              <TabsContent value="settings" className="space-y-6">
                <Card>
                  <CardHeader>
                    <CardTitle>AI System Configuration</CardTitle>
                    <CardDescription>Configure AI models and system settings</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div>
                      <Label htmlFor="ai-model">AI Model</Label>
                      <Select defaultValue="llama-3.1-8b-instant">
                        <SelectTrigger>
                          <SelectValue />
                        </SelectTrigger>
                        <SelectContent>
                          <SelectItem value="llama-3.1-8b-instant">Llama 3.1 8B (Fast)</SelectItem>
                          <SelectItem value="llama-3.1-70b-versatile">Llama 3.1 70B (Powerful)</SelectItem>
                          <SelectItem value="mixtral-8x7b-32768">Mixtral 8x7B</SelectItem>
                        </SelectContent>
                      </Select>
                    </div>
                    <div>
                      <Label htmlFor="max-file-size">Maximum File Size (MB)</Label>
                      <Input id="max-file-size" type="number" defaultValue="100" />
                    </div>
                    <div>
                      <Label htmlFor="ai-temperature">AI Temperature (Creativity)</Label>
                      <Input id="ai-temperature" type="number" step="0.1" min="0" max="2" defaultValue="0.7" />
                    </div>
                    <div>
                      <Label htmlFor="max-tokens">Max Response Tokens</Label>
                      <Input id="max-tokens" type="number" defaultValue="1000" />
                    </div>
                    <Button>Save AI Configuration</Button>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle>System Maintenance</CardTitle>
                    <CardDescription>AI system maintenance and optimization</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">Clear AI Cache</h4>
                        <p className="text-sm text-gray-600">Clear cached AI responses and embeddings</p>
                      </div>
                      <Button 
                        variant="outline" 
                        onClick={async () => {
                          try {
                            await adminAPI.performSystemAction('clear_cache')
                            toast({
                              title: "Success",
                              description: "AI cache cleared successfully",
                            })
                          } catch (error: any) {
                            toast({
                              title: "Error",
                              description: error.message || "Failed to clear cache",
                              variant: "destructive",
                            })
                          }
                        }}
                      >
                        Clear Cache
                      </Button>
                    </div>
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">Reindex Documents</h4>
                        <p className="text-sm text-gray-600">Rebuild AI document index for better search</p>
                      </div>
                      <Button 
                        variant="outline"
                        onClick={async () => {
                          try {
                            await adminAPI.performSystemAction('reindex_documents')
                            toast({
                              title: "Success",
                              description: "Document reindexing completed",
                            })
                          } catch (error: any) {
                            toast({
                              title: "Error",
                              description: error.message || "Failed to reindex documents",
                              variant: "destructive",
                            })
                          }
                        }}
                      >
                        Reindex
                      </Button>
                    </div>
                    <div className="flex items-center justify-between p-3 border rounded-lg">
                      <div>
                        <h4 className="font-medium">Export AI Data</h4>
                        <p className="text-sm text-gray-600">Export all AI-processed data and analytics</p>
                      </div>
                      <Button 
                        variant="outline"
                        onClick={async () => {
                          try {
                            const result = await adminAPI.performSystemAction('export_data')
                            toast({
                              title: "Success",
                              description: "Data export prepared successfully",
                            })
                            console.log('Export data:', result)
                          } catch (error: any) {
                            toast({
                              title: "Error",
                              description: error.message || "Failed to export data",
                              variant: "destructive",
                            })
                          }
                        }}
                      >
                        Export
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </TabsContent>
            </Tabs>
          </div>
        </div>
      </div>
    </div>
  )
}
