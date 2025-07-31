"use client"

import type React from "react"

import { useState, useCallback, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { Badge } from "@/components/ui/badge"
import { Upload, FileText, X, CheckCircle, AlertCircle, Loader2 } from "lucide-react"
import Link from "next/link"
import SixthvaultLogo from "@/components/SixthvaultLogo"
import { ragApiClient, type UploadResponse, type WebSocketMessage } from "@/lib/api-client"
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
}

function UploadPageContent() {
  const { logout } = useAuth()
  const [files, setFiles] = useState<UploadedFile[]>([])
  const [isDragging, setIsDragging] = useState(false)
  const [totalProgress, setTotalProgress] = useState(0)

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
    const newFiles: UploadedFile[] = fileList.map((file) => ({
      id: Math.random().toString(36).substr(2, 9),
      name: file.name,
      size: file.size,
      type: file.type,
      status: "uploading",
      progress: 0,
    }))

    setFiles((prev) => [...prev, ...newFiles])

    try {
      // Upload files to RAG backend - Ollama Only
      const uploadResponse = await ragApiClient.uploadFiles(fileList)
      console.log('Upload response:', uploadResponse)

      // Update files to processing status
      setFiles((prev) => prev.map((f) => ({ ...f, status: "processing", progress: 0 })))

      // Connect to WebSocket for progress updates
      const ws = ragApiClient.connectWebSocket(uploadResponse.batch_id)
      
      ws.onmessage = (event) => {
        try {
          const message: WebSocketMessage = JSON.parse(event.data)
          console.log('WebSocket message:', message)

          switch (message.type) {
            case 'queued':
              setFiles((prev) => prev.map((f) => ({ ...f, status: "processing", progress: 10 })))
              break
            case 'processing':
              setFiles((prev) => prev.map((f) => ({ ...f, progress: 50 })))
              break
              
            // Enhanced bulk processing events
            case 'bulk_upload_started':
              setFiles((prev) => prev.map((f) => ({ 
                ...f, 
                status: "processing", 
                progress: 5 
              })))
              break
              
            case 'tier1_initiated':
              // Priority file processing started
              setFiles((prev) => prev.map((f, index) => 
                index === 0 ? { ...f, status: "processing", progress: 15 } : f
              ))
              break
              
            case 'priority_processing_started':
              setFiles((prev) => prev.map((f, index) => 
                index === 0 ? { ...f, status: "processing", progress: 25 } : f
              ))
              break
              
            case 'rag_available':
              // RAG is now available - first file completed
              setFiles((prev) => prev.map((f, index) => 
                index === 0 ? { 
                  ...f, 
                  status: "completed", 
                  progress: 100,
                  language: "English",
                  themes: ["First Document - RAG Ready"],
                  demographics: []
                } : f
              ))
              break
              
            case 'tier2_initiated':
              // Background processing started
              setFiles((prev) => prev.map((f, index) => 
                index > 0 ? { ...f, status: "processing", progress: 10 } : f
              ))
              break
              
            case 'background_processing_started':
              // Individual background file started
              if (message.data?.file_index) {
                setFiles((prev) => prev.map((f, index) => 
                  index === (message.data.file_index - 1) ? { ...f, progress: 30 } : f
                ))
              }
              break
              
            case 'background_file_completed':
              // Individual background file completed
              if (message.data?.file_index) {
                setFiles((prev) => prev.map((f, index) => 
                  index === (message.data.file_index - 1) ? { 
                    ...f, 
                    status: "completed", 
                    progress: 100,
                    language: "English",
                    themes: ["Document Analysis"],
                    demographics: []
                  } : f
                ))
              }
              break
              
            case 'parallel_progress':
              // Real-time parallel processing updates
              setFiles((prev) => prev.map((f) => ({ 
                ...f, 
                progress: Math.max(f.progress, message.data?.overall_progress || 50)
              })))
              break
              
            case 'task_completed':
              // Individual task completion (tagging, summary, etc.)
              setFiles((prev) => prev.map((f) => ({ 
                ...f, 
                progress: Math.max(f.progress, 70)
              })))
              break
              
            case 'completed':
              setFiles((prev) => prev.map((f) => ({ 
                ...f, 
                status: "completed", 
                progress: 100,
                language: message.data?.language || "English",
                themes: message.data?.themes || ["Document Analysis"],
                demographics: message.data?.demographics || []
              })))
              ws.close()
              break
              
            case 'error':
            case 'priority_processing_error':
            case 'background_processing_error':
              setFiles((prev) => prev.map((f) => ({ ...f, status: "error", progress: 0 })))
              ws.close()
              break
          }
        } catch (error) {
          console.error('Error parsing WebSocket message:', error)
        }
      }

      // Fallback: Close WebSocket after 5 minutes
      setTimeout(() => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.close()
        }
      }, 300000)

    } catch (error) {
      console.error('Upload error:', error)
      // Mark all files as error
      setFiles((prev) => prev.map((f) => ({ ...f, status: "error", progress: 0 })))
    }
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
                            <span className="text-gray-600">{file.progress}%</span>
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
