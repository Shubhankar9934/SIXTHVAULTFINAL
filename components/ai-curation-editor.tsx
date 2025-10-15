"use client"

import React, { useState, useEffect, useRef } from 'react'
import { Button } from '@/components/ui/button'
import { Textarea } from '@/components/ui/textarea'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Separator } from '@/components/ui/separator'
import { Badge } from '@/components/ui/badge'
import { useToast } from '@/hooks/use-toast'
import { aiCurationService } from '@/lib/ai-curation-service'
import { 
  Save, 
  X, 
  Edit3, 
  FileText, 
  Clock, 
  AlertCircle, 
  CheckCircle2,
  Loader2,
  Eye,
  EyeOff,
  RotateCcw,
  Copy
} from 'lucide-react'
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog'
import { Alert, AlertDescription } from '@/components/ui/alert'

interface AICurationEditorProps {
  curationId: string
  initialContent?: string
  initialTitle?: string
  initialDescription?: string
  isOpen: boolean
  onClose: () => void
  onSave?: (content: string, title?: string, description?: string) => void
}

export function AICurationEditor({
  curationId,
  initialContent = '',
  initialTitle = '',
  initialDescription = '',
  isOpen,
  onClose,
  onSave
}: AICurationEditorProps) {
  const [content, setContent] = useState(initialContent)
  const [title, setTitle] = useState(initialTitle)
  const [description, setDescription] = useState(initialDescription)
  const [originalContent, setOriginalContent] = useState(initialContent)
  const [originalTitle, setOriginalTitle] = useState(initialTitle)
  const [originalDescription, setOriginalDescription] = useState(initialDescription)
  const [isSaving, setIsSaving] = useState(false)
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
  const [isPreviewMode, setIsPreviewMode] = useState(false)
  const [lastSaved, setLastSaved] = useState<Date | null>(null)
  const [saveError, setSaveError] = useState<string | null>(null)
  const [characterCount, setCharacterCount] = useState(0)
  const [wordCount, setWordCount] = useState(0)
  
  const contentRef = useRef<HTMLTextAreaElement>(null)
  const { toast } = useToast()

  // Update character and word count
  useEffect(() => {
    setCharacterCount(content.length)
    setWordCount(content.trim() ? content.trim().split(/\s+/).length : 0)
  }, [content])

  // Track unsaved changes
  useEffect(() => {
    const hasChanges = 
      content !== originalContent || 
      title !== originalTitle || 
      description !== originalDescription
    setHasUnsavedChanges(hasChanges)
  }, [content, title, description, originalContent, originalTitle, originalDescription])

  // Reset state when modal opens/closes
  useEffect(() => {
    if (isOpen) {
      setContent(initialContent)
      setTitle(initialTitle)
      setDescription(initialDescription)
      setOriginalContent(initialContent)
      setOriginalTitle(initialTitle)
      setOriginalDescription(initialDescription)
      setHasUnsavedChanges(false)
      setIsPreviewMode(false)
      setSaveError(null)
      setLastSaved(null)
    }
  }, [isOpen, initialContent, initialTitle, initialDescription])

  // Auto-save functionality (optional - commented out for now)
  // useEffect(() => {
  //   if (!hasUnsavedChanges || !isOpen) return
  //   
  //   const autoSaveTimer = setTimeout(() => {
  //     handleSave(true) // true for auto-save
  //   }, 30000) // Auto-save after 30 seconds of inactivity
  //   
  //   return () => clearTimeout(autoSaveTimer)
  // }, [content, title, description, hasUnsavedChanges, isOpen])

  const handleSave = async (isAutoSave = false) => {
    if (!hasUnsavedChanges && !isAutoSave) {
      toast({
        title: "No Changes",
        description: "No changes to save.",
        variant: "default"
      })
      return
    }

    setIsSaving(true)
    setSaveError(null)

    try {
      const result = await aiCurationService.updateCurationContent(
        curationId,
        content,
        title !== originalTitle ? title : undefined,
        description !== originalDescription ? description : undefined
      )

      if (result.success) {
        // Update original values to reflect saved state
        setOriginalContent(content)
        setOriginalTitle(title)
        setOriginalDescription(description)
        setHasUnsavedChanges(false)
        setLastSaved(new Date())

        if (!isAutoSave) {
          toast({
            title: "Saved Successfully",
            description: "Your curation has been updated and saved to the database.",
            variant: "default"
          })
        }

        // Call parent callback if provided
        if (onSave) {
          onSave(content, title, description)
        }
      } else {
        setSaveError(result.message)
        toast({
          title: "Save Failed",
          description: result.message || "Failed to save curation. Please try again.",
          variant: "destructive"
        })
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred'
      setSaveError(errorMessage)
      toast({
        title: "Save Error",
        description: errorMessage,
        variant: "destructive"
      })
    } finally {
      setIsSaving(false)
    }
  }

  const handleReset = () => {
    setContent(originalContent)
    setTitle(originalTitle)
    setDescription(originalDescription)
    setHasUnsavedChanges(false)
    setSaveError(null)
    toast({
      title: "Changes Discarded",
      description: "All unsaved changes have been discarded.",
      variant: "default"
    })
  }

  const handleCopyContent = () => {
    navigator.clipboard.writeText(content)
    toast({
      title: "Copied",
      description: "Content copied to clipboard.",
      variant: "default"
    })
  }

  const handleClose = () => {
    if (hasUnsavedChanges) {
      const confirmClose = window.confirm(
        "You have unsaved changes. Are you sure you want to close without saving?"
      )
      if (!confirmClose) return
    }
    onClose()
  }

  const renderPreview = () => {
    // Simple markdown-like rendering for preview
    const lines = content.split('\n')
    return (
      <div className="prose prose-sm max-w-none dark:prose-invert">
        {lines.map((line, index) => {
          if (line.startsWith('# ')) {
            return <h1 key={index} className="text-2xl font-bold mb-4">{line.slice(2)}</h1>
          } else if (line.startsWith('## ')) {
            return <h2 key={index} className="text-xl font-semibold mb-3">{line.slice(3)}</h2>
          } else if (line.startsWith('### ')) {
            return <h3 key={index} className="text-lg font-medium mb-2">{line.slice(4)}</h3>
          } else if (line.startsWith('- ')) {
            return <li key={index} className="ml-4">{line.slice(2)}</li>
          } else if (line.trim() === '') {
            return <br key={index} />
          } else {
            return <p key={index} className="mb-2">{line}</p>
          }
        })}
      </div>
    )
  }

  return (
    <Dialog open={isOpen} onOpenChange={handleClose}>
      <DialogContent className="max-w-4xl max-h-[90vh] flex flex-col">
        <DialogHeader>
          <DialogTitle className="flex items-center gap-2">
            <Edit3 className="w-5 h-5" />
            AI Curation Editor
            {hasUnsavedChanges && (
              <Badge variant="outline" className="ml-2">
                <AlertCircle className="w-3 h-3 mr-1" />
                Unsaved
              </Badge>
            )}
            {lastSaved && !hasUnsavedChanges && (
              <Badge variant="secondary" className="ml-2">
                <CheckCircle2 className="w-3 h-3 mr-1" />
                Saved
              </Badge>
            )}
          </DialogTitle>
          <DialogDescription>
            Edit your AI curation content, title, and description. Changes are saved to the database.
          </DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-hidden">
          <div className="space-y-4 h-full">
            {/* Title and Description */}
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="space-y-2">
                <Label htmlFor="title">Title</Label>
                <Input
                  id="title"
                  value={title}
                  onChange={(e) => setTitle(e.target.value)}
                  placeholder="Enter curation title..."
                  className="w-full"
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="description">Description</Label>
                <Input
                  id="description"
                  value={description}
                  onChange={(e) => setDescription(e.target.value)}
                  placeholder="Enter curation description..."
                  className="w-full"
                />
              </div>
            </div>

            <Separator />

            {/* Content Editor */}
            <div className="flex-1 flex flex-col min-h-0">
              <div className="flex items-center justify-between mb-3">
                <Label htmlFor="content" className="text-base font-medium">
                  Content
                </Label>
                <div className="flex items-center gap-2">
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={() => setIsPreviewMode(!isPreviewMode)}
                  >
                    {isPreviewMode ? (
                      <>
                        <Edit3 className="w-4 h-4 mr-1" />
                        Edit
                      </>
                    ) : (
                      <>
                        <Eye className="w-4 h-4 mr-1" />
                        Preview
                      </>
                    )}
                  </Button>
                  <Button
                    type="button"
                    variant="outline"
                    size="sm"
                    onClick={handleCopyContent}
                  >
                    <Copy className="w-4 h-4 mr-1" />
                    Copy
                  </Button>
                </div>
              </div>

              {/* Save Error Alert */}
              {saveError && (
                <Alert variant="destructive" className="mb-3">
                  <AlertCircle className="h-4 w-4" />
                  <AlertDescription>{saveError}</AlertDescription>
                </Alert>
              )}

              <Card className="flex-1 min-h-0">
                <CardContent className="p-4 h-full">
                  {isPreviewMode ? (
                    <div className="h-full overflow-auto">
                      {renderPreview()}
                    </div>
                  ) : (
                    <Textarea
                      ref={contentRef}
                      id="content"
                      value={content}
                      onChange={(e) => setContent(e.target.value)}
                      placeholder="Enter your curation content here... You can use Markdown formatting."
                      className="h-full min-h-[400px] resize-none border-0 focus-visible:ring-0"
                      style={{ fontFamily: 'monospace' }}
                    />
                  )}
                </CardContent>
              </Card>

              {/* Stats and Info */}
              <div className="flex items-center justify-between mt-2 text-sm text-muted-foreground">
                <div className="flex items-center gap-4">
                  <span>{characterCount} characters</span>
                  <span>{wordCount} words</span>
                  {lastSaved && (
                    <span className="flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      Last saved: {lastSaved.toLocaleTimeString()}
                    </span>
                  )}
                </div>
                {hasUnsavedChanges && (
                  <span className="text-amber-600 dark:text-amber-400">
                    Unsaved changes
                  </span>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Footer Actions */}
        <div className="flex items-center justify-between pt-4 border-t">
          <div className="flex items-center gap-2">
            {hasUnsavedChanges && (
              <Button
                type="button"
                variant="ghost"
                size="sm"
                onClick={handleReset}
                className="text-muted-foreground"
              >
                <RotateCcw className="w-4 h-4 mr-1" />
                Reset
              </Button>
            )}
          </div>
          <div className="flex items-center gap-2">
            <Button
              type="button"
              variant="outline"
              onClick={handleClose}
            >
              <X className="w-4 h-4 mr-1" />
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => handleSave()}
              disabled={isSaving || !hasUnsavedChanges}
              className="min-w-[100px]"
            >
              {isSaving ? (
                <>
                  <Loader2 className="w-4 h-4 mr-1 animate-spin" />
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-1" />
                  Save
                </>
              )}
            </Button>
          </div>
        </div>
      </DialogContent>
    </Dialog>
  )
}

export default AICurationEditor
