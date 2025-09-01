"use client"

import { useEffect, useRef } from 'react'
import katex from 'katex'
import 'katex/dist/katex.min.css'

interface MathRendererProps {
  math: string
  displayMode?: boolean
  className?: string
}

export function MathRenderer({ math, displayMode = false, className = '' }: MathRendererProps) {
  const containerRef = useRef<HTMLSpanElement>(null)

  useEffect(() => {
    if (containerRef.current) {
      try {
        katex.render(math, containerRef.current, {
          displayMode,
          throwOnError: false,
          errorColor: '#cc0000',
          strict: false,
          trust: true,
          macros: {
            "\\RR": "\\mathbb{R}",
            "\\NN": "\\mathbb{N}",
            "\\ZZ": "\\mathbb{Z}",
            "\\QQ": "\\mathbb{Q}",
            "\\CC": "\\mathbb{C}",
          }
        })
      } catch (error) {
        console.error('KaTeX rendering error:', error)
        if (containerRef.current) {
          containerRef.current.textContent = math
        }
      }
    }
  }, [math, displayMode])

  return (
    <span 
      ref={containerRef} 
      className={`math-renderer ${displayMode ? 'block my-4 text-center' : 'inline'} ${className}`}
    />
  )
}

interface MathContentProps {
  content: string
  className?: string
}

export function MathContent({ content, className = '' }: MathContentProps) {
  const processedContent = processMathContent(content)
  
  return (
    <div 
      className={`math-content ${className}`}
      dangerouslySetInnerHTML={{ __html: processedContent }}
    />
  )
}

// Function to process content and replace math expressions with rendered math
function processMathContent(content: string): string {
  // Handle display math ($$...$$)
  content = content.replace(/\$\$([^$]+)\$\$/g, (match, math) => {
    const mathId = `math-display-${Math.random().toString(36).substr(2, 9)}`
    setTimeout(() => {
      const element = document.getElementById(mathId)
      if (element) {
        try {
          katex.render(math.trim(), element, {
            displayMode: true,
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true
          })
        } catch (error) {
          console.error('KaTeX display math error:', error)
          element.textContent = `$$${math}$$`
        }
      }
    }, 0)
    return `<div id="${mathId}" class="math-display block my-4 text-center bg-slate-50 p-4 rounded-lg border border-slate-200"></div>`
  })

  // Handle inline math ($...$)
  content = content.replace(/\$([^$\n]+)\$/g, (match, math) => {
    const mathId = `math-inline-${Math.random().toString(36).substr(2, 9)}`
    setTimeout(() => {
      const element = document.getElementById(mathId)
      if (element) {
        try {
          katex.render(math.trim(), element, {
            displayMode: false,
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true
          })
        } catch (error) {
          console.error('KaTeX inline math error:', error)
          element.textContent = `$${math}$`
        }
      }
    }, 0)
    return `<span id="${mathId}" class="math-inline bg-blue-50 px-1 py-0.5 rounded border border-blue-200"></span>`
  })

  // Handle LaTeX-style equations with \[ \] or \( \)
  content = content.replace(/\\\[([^\]]+)\\\]/g, (match, math) => {
    const mathId = `math-bracket-${Math.random().toString(36).substr(2, 9)}`
    setTimeout(() => {
      const element = document.getElementById(mathId)
      if (element) {
        try {
          katex.render(math.trim(), element, {
            displayMode: true,
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true
          })
        } catch (error) {
          console.error('KaTeX bracket math error:', error)
          element.textContent = `\\[${math}\\]`
        }
      }
    }, 0)
    return `<div id="${mathId}" class="math-display block my-4 text-center bg-slate-50 p-4 rounded-lg border border-slate-200"></div>`
  })

  content = content.replace(/\\\(([^)]+)\\\)/g, (match, math) => {
    const mathId = `math-paren-${Math.random().toString(36).substr(2, 9)}`
    setTimeout(() => {
      const element = document.getElementById(mathId)
      if (element) {
        try {
          katex.render(math.trim(), element, {
            displayMode: false,
            throwOnError: false,
            errorColor: '#cc0000',
            strict: false,
            trust: true
          })
        } catch (error) {
          console.error('KaTeX paren math error:', error)
          element.textContent = `\\(${math}\\)`
        }
      }
    }, 0)
    return `<span id="${mathId}" class="math-inline bg-blue-50 px-1 py-0.5 rounded border border-blue-200"></span>`
  })

  return content
}

// Utility function to detect if content contains mathematical expressions
export function containsMath(content: string): boolean {
  return /\$\$[^$]+\$\$|\$[^$\n]+\$|\\\[[^\]]+\\\]|\\\([^)]+\\\)/.test(content)
}

// Enhanced content formatter that handles both regular formatting and math
export function formatContentWithMath(content: string): string {
  // First process math expressions
  let formattedContent = processMathContent(content)
  
  // Remove text-based separators (equals signs and dashes)
  formattedContent = formattedContent
    .replace(/^=+$/gm, "")
    .replace(/^-+$/gm, "")
    .replace(/^_+$/gm, "")
    .replace(/^\*+$/gm, "")

  // Replace markdown headings with styled headings
  formattedContent = formattedContent
    .replace(/^# (.*?)$/gm, '<h1 class="text-2xl font-bold mb-6 pb-3 border-b-2 border-gradient-to-r from-blue-500 to-purple-600 text-slate-800">$1</h1>')
    .replace(/^## (.*?)$/gm, '<h2 class="text-xl font-semibold mt-8 mb-4 pb-2 border-b border-slate-200 text-slate-700">$1</h2>')
    .replace(/^### (.*?)$/gm, '<h3 class="text-lg font-medium mt-6 mb-3 text-slate-600">$1</h3>')

    // Format lists
    .replace(/^\* (.*?)$/gm, '<li class="ml-4 list-disc my-2 text-slate-700">$1</li>')
    .replace(/^\d+\. (.*?)$/gm, '<li class="ml-4 list-decimal my-2 text-slate-700">$1</li>')

    // Format bold and italic (but avoid math expressions)
    .replace(/\*\*((?!\$)[^*]+(?<!\$))\*\*/g, '<strong class="font-semibold text-slate-800">$1</strong>')
    .replace(/\*((?!\$)[^*]+(?<!\$))\*/g, '<em class="italic text-slate-600">$1</em>')

    // Format paragraphs
    .replace(/\n\n/g, '</p><p class="my-4 text-slate-700 leading-relaxed">')

    // Format line breaks
    .replace(/\n/g, "<br>")

  // Wrap in paragraph if not already
  if (!formattedContent.startsWith("<h") && !formattedContent.startsWith("<p") && !formattedContent.startsWith("<div")) {
    formattedContent = `<p class="my-4 text-slate-700 leading-relaxed">${formattedContent}</p>`
  }

  return formattedContent
}
