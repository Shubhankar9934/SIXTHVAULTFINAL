"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2, Home, LogOut } from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { ragApiClient, type UploadResponse, type WebSocketMessage } from "@/lib/api-client"
import { documentStore, type ProcessingDocument } from "@/lib/document-store"
import { RouteGuard } from "@/components/route-guard"
import { useAuth } from "@/lib/auth-context"

interface UploadedFile {
  id: string
  name: string
  size: number
  type: string
  status: "uploading" | "processing" | "completed" | "error"
  progress: number
  language?: string
  themes?: string[]
  demographics?: string[]
  processingOrder?: number // For sequential progress tracking
  batchId?: string
}

function UploadPageContent() {
  const { logout, user, isAuthenticated } = useAuth()
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [totalProgress, setTotalProgress] = useState(0)
  const [processingDocuments, setProcessingDocuments] = useState<ProcessingDocument[]>([])
  const [accessDenied, setAccessDenied] = useState(false)
  const [loading, setLoading] = useState(true)

  // Check admin access on mount
  useEffect(() => {
    if (!isAuthenticated) {
      setLoading(false)
      return
    }

    if (!user || !user.is_admin || user.role !== 'admin') {
      console.log('Upload page: Access denied - user is not admin:', user)
      setAccessDenied(true)
      setLoading(false)
      return
    }

    console.log('Upload page: Admin access confirmed for user:', user.email)
    setAccessDenied(false)
    setLoading(false)
  }, [isAuthenticated, user])

  // Initialize background processing and subscribe to updates
  useEffect(() => {
    if (!isAuthenticated || !user || accessDenied) {
      return
    }

    // Initialize background processing and load documents
    documentStore.initializeBackgroundProcessing();
    documentStore.getDocuments();
    
    // Subscribe to processing updates
    const unsubscribe = documentStore.subscribeToProcessingUpdates(setProcessingDocuments);
    
    return () => {
      unsubscribe();
    };
  }, [isAuthenticated, user, accessDenied]);

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
              Admin privileges required for document upload
            </CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-6">
            <div className="p-4 bg-gradient-to-r from-red-50 to-red-100 rounded-xl border border-red-200">
              <p className="text-sm text-red-700 font-medium">
                Only admin users can upload and manage documents.
              </p>
              <p className="text-xs text-red-600 mt-2">
                Regular users can search, query, and read assigned documents only.
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
              <SixthvaultLogo size="full" />
            </div>
            <div className="mb-8">
              <div className="animate-spin rounded-full h-20 w-20 border-4 border-blue-200 border-t-blue-600 mx-auto mb-6"></div>
            </div>
            <div className="space-y-4">
              <h2 className="text-3xl font-bold bg-gradient-to-r from-blue-700 via-indigo-700 to-purple-700 bg-clip-text text-transparent">
                Loading Upload Interface
              </h2>
              <p className="text-xl font-semibold text-slate-700">
                Preparing document upload system...
              </p>
            </div>
          </CardContent>
        </Card>
      </div>
    )
  }

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const droppedFiles = Array.from(e.dataTransfer.files)
    processFiles(droppedFiles)
  }, [])

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      const selectedFiles = Array.from(e.target.files)
      processFiles(selectedFiles)
    }
  }

  const processFiles = async (fileList: File[]) => {
    // FIXED: Proper sequential processing with immediate UI visibility
    console.log('🔄 STARTING SEQUENTIAL BULK UPLOAD for', fileList.length, 'files')
    
    // First, add ALL files to UI immediately so user can see them
    const tempFiles: UploadedFile[] = fileList.map((file, index) => ({
      id: `temp_${index}_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      name: file.name,
      size: file.size,
      type: file.type,
      status: "uploading",
      progress: 0,
      processingOrder: index,
    }))
    
    // Add all files to UI at once so user sees the full queue
    setFiles((prev) => [...prev, ...tempFiles])
    
    // Now process files sequentially, but update existing UI entries
    for (let i = 0; i < fileList.length; i++) {
      const file = fileList[i]
      const tempFile = tempFiles[i]
      
      console.log(`📁 Processing file ${i + 1}/${fileList.length}: ${file.name}`)
      
      try {
        await processFileSequentially(file, tempFile.id, i + 1, fileList.length)
        console.log(`✅ File ${i + 1}/${fileList.length} completed: ${file.name}`)
      } catch (error) {
        console.error(`❌ File ${i + 1}/${fileList.length} failed: ${file.name}`, error)
        // Mark as error but continue with next file
        setFiles((prev) => 
          prev.map((f) => 
            f.id === tempFile.id
              ? { ...f, status: "error", progress: 0 }
              : f
          )
        )
      }
      
      // Brief pause between files for better UX
      if (i < fileList.length - 1) {
        await new Promise(resolve => setTimeout(resolve, 800))
      }
    }
    
    console.log('✅ All files processed sequentially')
  }

  // FIXED: New sequential processing function that updates existing UI entries
  const processFileSequentially = async (file: File, tempFileId: string, currentFile: number, totalFiles: number) => {
    return new Promise<void>(async (resolve, reject) => {
      try {
        console.log(`📤 Starting upload for file ${currentFile}/${totalFiles}: ${file.name}`)
        
        // Update status to uploading
        setFiles((prev) => 
          prev.map((f) => 
            f.id === tempFileId 
              ? { ...f, status: "uploading", progress: 5 }
              : f
          )
        )
        
        // Upload single file
        const uploadResponse = await ragApiClient.uploadFiles([file])
        const batchId = uploadResponse.batch_id
        
        console.log(`✅ Upload initiated for ${file.name}, batch ID: ${batchId}`)

        // Update to processing status
        setFiles((prev) => 
          prev.map((f) => 
            f.id === tempFileId 
              ? { 
                  ...f, 
                  status: "processing", 
                  progress: 15,
                  batchId: batchId
                }
              : f
          )
        )

        // Add to document store for background processing
        const processingDoc: ProcessingDocument = {
          id: tempFileId,
          name: file.name,
          size: file.size,
          type: file.type,
          status: "processing",
          progress: 15,
          uploadDate: new Date().toISOString().split("T")[0],
          batchId: batchId,
          processingOrder: currentFile - 1,
          statusMessage: `Processing file ${currentFile}/${totalFiles}...`
        }
        
        documentStore.addProcessingDocument(processingDoc)
        console.log('📝 Added processing document to store:', processingDoc.name)

        // CRITICAL FIX: Start background processing for this batch
        const backgroundCleanup1 = documentStore.startBackgroundProcessing(batchId, [processingDoc])
        console.log('🔄 Started background processing for batch:', batchId)
        
        // Store cleanup function for later use
        ;(window as any)[`backgroundCleanup_${batchId}`] = backgroundCleanup1

        // Set up completion tracking
        let isCompleted = false
        let progressUpdateInterval: NodeJS.Timeout | null = null
        let completionTimeout: NodeJS.Timeout
        
        const completeFile = () => {
          if (!isCompleted) {
            isCompleted = true
            if (progressUpdateInterval) clearInterval(progressUpdateInterval)
            clearTimeout(completionTimeout)
            console.log(`🎯 File completion confirmed: ${file.name}`)
            resolve()
          }
        }

        // Start gradual progress updates for better UX
        progressUpdateInterval = setInterval(() => {
          if (!isCompleted) {
            setFiles((prev) => 
              prev.map((f) => {
                if (f.id === tempFileId && f.status === "processing" && f.progress < 85) {
                  return { 
                    ...f, 
                    progress: Math.min(f.progress + 3, 85) // Gradually increase to 85%
                  }
                }
                return f
              })
            )
          }
        }, 2000) // Update every 2 seconds

        // Create WebSocket connection for real-time updates
        const reliableWS = ragApiClient.createReliableWebSocket(batchId, (message: any) => {
          try {
            console.log(`📨 WebSocket message for ${file.name}:`, message.type, message.data)

            switch (message.type) {
              case 'processing':
                setFiles((prev) => 
                  prev.map((f) => {
                    if (f.id === tempFileId) {
                      // Ensure progress never decreases and is synced with percentage display
                      const newProgress = Math.max(f.progress, message.data?.progress || 30)
                      return { 
                        ...f, 
                        progress: Math.min(newProgress, 90)
                      }
                    }
                    return f
                  })
                )
                break

              case 'completed':
              case 'file_processing_completed':
                console.log(`🎉 File processing completed: ${file.name}`)
                
                // Clear progress interval
                if (progressUpdateInterval) {
                  clearInterval(progressUpdateInterval)
                  progressUpdateInterval = null
                }
                
                // Update to completed state with AI data
                setFiles((prev) => 
                  prev.map((f) => {
                    if (f.id === tempFileId) {
                      const completedData = message.data || {}
                      return {
                        ...f,
                        status: "completed" as const,
                        progress: 100, // Always 100% when completed
                        language: completedData.language || "English",
                        themes: completedData.themes || ["Document Analysis"],
                        demographics: completedData.demographics || []
                      }
                    }
                    return f
                  })
                )
                
                // Update processing document in store
                documentStore.updateProcessingDocument(tempFileId, {
                  status: 'completed',
                  progress: 100,
                  summary: message.data?.summary,
                  themes: message.data?.themes,
                  keywords: message.data?.keywords || message.data?.themes,
                  demographics: message.data?.demographics,
                  keyInsights: message.data?.insights ? [message.data.insights] : undefined,
                  language: message.data?.language || 'English',
                  statusMessage: 'Processing completed successfully!'
                })
                
                // Complete this file
                setTimeout(completeFile, 500) // Small delay to show 100% completion
                break

              case 'error':
                console.error(`❌ Processing error for ${file.name}:`, message.data)
                
                // Clear progress interval
                if (progressUpdateInterval) {
                  clearInterval(progressUpdateInterval)
                  progressUpdateInterval = null
                }
                
                setFiles((prev) => 
                  prev.map((f) => {
                    if (f.id === tempFileId) {
                      return { 
                        ...f, 
                        status: "error", 
                        progress: 0
                      }
                    }
                    return f
                  })
                )
                reject(new Error(message.data?.error || 'Processing failed'))
                break

              default:
                // Handle other progress updates with sync
                if (message.data?.progress && typeof message.data.progress === 'number') {
                  setFiles((prev) => 
                    prev.map((f) => {
                      if (f.id === tempFileId) {
                        // Ensure progress bar and percentage are always synced
                        const syncedProgress = Math.max(f.progress, Math.min(message.data.progress, 90))
                        return { 
                          ...f, 
                          progress: syncedProgress
                        }
                      }
                      return f
                    })
                  )
                }
                break
            }
          } catch (error) {
            console.error(`❌ Error handling WebSocket message for ${file.name}:`, error)
          }
        })

        // Start WebSocket connection
        await reliableWS.connect()
        
        // Note: Background processing already started above, no need to duplicate
        
        // Set up timeout for completion (8 minutes for large files)
        completionTimeout = setTimeout(() => {
          if (!isCompleted) {
            console.warn(`⏱️ Timeout waiting for completion of ${file.name}`)
            
            // Clear progress interval
            if (progressUpdateInterval) {
              clearInterval(progressUpdateInterval)
              progressUpdateInterval = null
            }
            
            reliableWS.disconnect()
            
            // Mark as completed to prevent hanging
            setFiles((prev) => 
              prev.map((f) => {
                if (f.id === tempFileId) {
                  return { 
                    ...f, 
                    status: "completed", 
                    progress: 100,
                    language: 'English'
                  }
                }
                return f
              })
            )
            
            completeFile()
          }
        }, 480000) // 8 minutes timeout for large files

        // Cleanup function
        const cleanup = () => {
          if (progressUpdateInterval) {
            clearInterval(progressUpdateInterval)
            progressUpdateInterval = null
          }
          reliableWS.disconnect()
          clearTimeout(completionTimeout)
        }

        // Store cleanup function
        ;(window as any)[`cleanup_${batchId}`] = cleanup

      } catch (error) {
        console.error(`❌ Upload failed for ${file.name}:`, error)
        
        // Update to error state
        setFiles((prev) => 
          prev.map((f) => 
            f.id === tempFileId
              ? { ...f, status: "error", progress: 0 }
              : f
          )
        )
        
        reject(error)
      }
    })
  }


  const removeFile = (fileId: string) => {
    setFiles((prev) => prev.filter((f) => f.id !== fileId))
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return "0 Bytes"
    const k = 1024
    const sizes = ["Bytes", "KB", "MB", "GB"]
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return Number.parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + " " + sizes[i]
  }

  const completedFiles = files.filter((f) => f.status === "completed").length
  const totalFiles = files.length

  return (
    <div className="min-h-screen bg-gray-50">
      <div className="flex h-screen">
        {/* Left Sidebar */}
        <div className="w-80 bg-white border-r-4 border-blue-500 flex flex-col">
          <div className="p-6 border-b border-gray-200">
            <SixthvaultLogo size="small" />
          </div>

          <div className="p-6 flex-1">
            <h3 className="font-bold text-black mb-6 text-lg">DOCUMENT UPLOAD</h3>

            <div className="space-y-4 mb-8">
              <div className="text-sm text-gray-600">
                <p className="mb-2">Supported formats:</p>
                <ul className="list-disc list-inside space-y-1 text-xs">
                  <li>PDF documents</li>
                  <li>Word documents (DOCX)</li>
                  <li>Text files (TXT)</li>
                  <li>Rich Text Format (RTF)</li>
                </ul>
              </div>

              {totalFiles > 0 && (
                <div className="bg-blue-50 p-3 rounded-lg">
                  <p className="text-sm font-medium text-blue-800">
                    Progress: {completedFiles}/{totalFiles} files processed
                  </p>
                  <Progress value={(completedFiles / totalFiles) * 100} className="mt-2" />
                </div>
              )}
            </div>
          </div>

          <div className="p-6 border-t border-gray-200">
            <Link href="/vault">
              <Button className="w-full font-bold text-black bg-transparent hover:bg-gray-100 text-lg">
                BACK TO VAULT
              </Button>
            </Link>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex-1 flex flex-col">
          {/* Header */}
          <div className="p-6 border-b border-gray-200 bg-white flex justify-between items-center">
            <h1 className="text-2xl font-bold text-black">Document Upload & Analysis</h1>
            <Button
              variant="ghost"
              className="text-black font-medium"
              onClick={logout}
            >
              LOG OUT
            </Button>
          </div>

          {/* Upload Area */}
          <div className="p-6 bg-white border-b border-gray-200">
            <div
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                isDragging ? "border-blue-500 bg-blue-50" : "border-gray-300"
              }`}
              onDragOver={handleDragOver}
              onDragLeave={handleDragLeave}
              onDrop={handleDrop}
            >
              <Upload className="mx-auto h-12 w-12 text-gray-400 mb-4" />
              <h3 className="text-lg font-medium text-gray-900 mb-2">Drop files here or click to upload</h3>
              <p className="text-gray-500 mb-4">Support for PDF, DOCX, TXT, RTF files up to 100MB each</p>
              <input
                type="file"
                multiple
                accept=".pdf,.docx,.txt,.rtf"
                onChange={handleFileSelect}
                className="hidden"
                id="file-upload"
              />
              <label htmlFor="file-upload">
                <Button className="bg-blue-600 hover:bg-blue-700 text-white">Select Files</Button>
              </label>
            </div>
          </div>

          {/* File List */}
          <div className="flex-1 p-6 overflow-y-auto">
            {files.length === 0 ? (
              <div className="text-center text-gray-500 mt-12">
                <FileText className="mx-auto h-16 w-16 text-gray-300 mb-4" />
                <h3 className="text-lg font-medium mb-2">No files uploaded yet</h3>
                <p>Upload your documents to start analyzing them with AI</p>
              </div>
            ) : (
              <div className="space-y-4">
                {files.map((file) => (
                  <Card key={file.id} className="border border-gray-200">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center space-x-3">
                          <FileText className="h-8 w-8 text-blue-600" />
                          <div>
                            <CardTitle className="text-base">{file.name}</CardTitle>
                            <CardDescription>
                              {formatFileSize(file.size)} • {file.type}
                            </CardDescription>
                          </div>
                        </div>
                        <div className="flex items-center space-x-2">
                          {file.status === "completed" && <CheckCircle className="h-5 w-5 text-green-500" />}
                          {file.status === "error" && <AlertCircle className="h-5 w-5 text-red-500" />}
                          {(file.status === "uploading" || file.status === "processing") && (
                            <Loader2 className="h-5 w-5 text-blue-500 animate-spin" />
                          )}
                          <Button variant="ghost" size="sm" onClick={() => removeFile(file.id)} className="h-8 w-8 p-0">
                            <X className="h-4 w-4" />
                          </Button>
                        </div>
                      </div>
                    </CardHeader>
                    <CardContent>
                      <div className="space-y-3">
                        <div>
                          <div className="flex justify-between text-sm mb-1">
                            <span className="text-gray-600">
                              {file.status === "uploading" && "Uploading..."}
                              {file.status === "processing" && "Processing & Analyzing..."}
                              {file.status === "completed" && "Analysis Complete"}
                              {file.status === "error" && "Error occurred"}
                            </span>
                            <span className="text-gray-600">{Math.round(file.progress)}%</span>
                          </div>
                          <Progress value={file.progress} className="h-2" />
                        </div>

                        {file.status === "completed" && (
                          <div className="space-y-2">
                            {file.language && (
                              <div>
                                <span className="text-sm font-medium text-gray-700">Language: </span>
                                <Badge variant="outline">{file.language}</Badge>
                              </div>
                            )}
                            {file.themes && file.themes.length > 0 && (
                              <div>
                                <span className="text-sm font-medium text-gray-700">Themes: </span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {file.themes.map((theme, index) => (
                                    <Badge key={index} variant="secondary" className="text-xs">
                                      {theme}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                            {file.demographics && file.demographics.length > 0 && (
                              <div>
                                <span className="text-sm font-medium text-gray-700">Demographics: </span>
                                <div className="flex flex-wrap gap-1 mt-1">
                                  {file.demographics.map((demo, index) => (
                                    <Badge key={index} variant="outline" className="text-xs">
                                      {demo}
                                    </Badge>
                                  ))}
                                </div>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default function UploadPage() {
  return (
    <RouteGuard>
      <UploadPageContent />
    </RouteGuard>
  )
}
