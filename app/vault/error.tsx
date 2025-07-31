"use client"

import { useEffect } from "react"
import Link from "next/link"

export default function VaultError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  useEffect(() => {
    console.error("Vault page error:", error)
  }, [error])

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="text-center p-8 bg-white rounded-lg shadow-md max-w-md">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Vault Error</h2>
        <p className="text-gray-600 mb-4">There was an error loading the vault page.</p>
        <details className="mb-4 text-left">
          <summary className="cursor-pointer text-gray-600 mb-2">Technical Details</summary>
          <pre className="text-xs bg-gray-100 p-3 rounded overflow-auto">
            {error.message}
            {error.digest && `\nDigest: ${error.digest}`}
          </pre>
        </details>
        <div className="space-x-2">
          <button onClick={() => reset()} className="bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
            Try again
          </button>
          <Link href="/" className="bg-gray-600 text-white px-4 py-2 rounded hover:bg-gray-700 inline-block">
            Go Home
          </Link>
        </div>
      </div>
    </div>
  )
}
