"use client"

export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <html>
      <body>
        <div className="min-h-screen flex items-center justify-center bg-red-50">
          <div className="text-center p-8">
            <h2 className="text-2xl font-bold text-red-600 mb-4">Global Error!</h2>
            <p className="mb-2">Message: {error.message}</p>
            <p className="mb-2">Name: {error.name}</p>
            {error.digest && <p className="mb-4">Digest: {error.digest}</p>}
            <button onClick={() => reset()} className="bg-blue-600 text-white px-4 py-2 rounded">
              Try again
            </button>
          </div>
        </div>
      </body>
    </html>
  )
}
