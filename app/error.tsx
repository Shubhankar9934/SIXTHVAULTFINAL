"use client"

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string }
  reset: () => void
}) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-red-50">
      <div className="text-center p-8 bg-white rounded-lg shadow-md max-w-lg">
        <h2 className="text-2xl font-bold text-red-600 mb-4">Error Found!</h2>

        <div className="text-left space-y-4">
          <div>
            <h3 className="font-semibold">Error Message:</h3>
            <p className="text-sm bg-red-100 p-2 rounded">{error.message}</p>
          </div>

          {error.digest && (
            <div>
              <h3 className="font-semibold">Error Digest:</h3>
              <p className="text-sm bg-gray-100 p-2 rounded">{error.digest}</p>
            </div>
          )}

          <div>
            <h3 className="font-semibold">Error Name:</h3>
            <p className="text-sm bg-gray-100 p-2 rounded">{error.name}</p>
          </div>

          {error.stack && (
            <div>
              <h3 className="font-semibold">Stack Trace:</h3>
              <pre className="text-xs bg-gray-100 p-2 rounded overflow-auto max-h-32">{error.stack}</pre>
            </div>
          )}
        </div>

        <button onClick={() => reset()} className="mt-4 bg-blue-600 text-white px-4 py-2 rounded hover:bg-blue-700">
          Try Again
        </button>
      </div>
    </div>
  )
}
