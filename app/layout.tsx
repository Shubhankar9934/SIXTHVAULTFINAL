import type React from "react"
import type { Metadata, Viewport } from "next"
import { Inter } from "next/font/google"
import "./globals.css"
import { AuthProvider } from "@/lib/auth-context"
import { VaultStateProvider } from "@/lib/vault-state-provider"
import ClientOnly from "@/components/client-only"

const inter = Inter({ 
  subsets: ["latin"],
  display: 'swap',
  variable: '--font-inter'
})

export const metadata: Metadata = {
  title: "SixthVault - AI-Powered Document Intelligence",
  description: "AI-powered document analysis and retrieval system with RAG backend",
  generator: 'v0.dev'
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 5
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={inter.className} suppressHydrationWarning={true}>
        <ClientOnly fallback={<div className="min-h-screen flex items-center justify-center">Loading...</div>}>
          <AuthProvider>
            <VaultStateProvider>
              {children}
            </VaultStateProvider>
          </AuthProvider>
        </ClientOnly>
      </body>
    </html>
  )
}
