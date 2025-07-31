"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card"
import { Label } from "@/components/ui/label"
import { useRouter } from "next/navigation"

export default function PreviewCodes() {
  const [email, setEmail] = useState("")
  const [code, setCode] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const router = useRouter()

  const fetchCode = async () => {
    if (!email) {
      setError("Please enter an email address")
      return
    }

    setLoading(true)
    setError(null)

    try {
      const response = await fetch(`/api/preview-codes?email=${encodeURIComponent(email)}`)
      const data = await response.json()

      if (!response.ok) {
        throw new Error(data.error || "Failed to retrieve code")
      }

      setCode(data.code)
    } catch (err: any) {
      setError(err.message || "Failed to retrieve verification code")
      setCode(null)
    } finally {
      setLoading(false)
    }
  }

  const goToVerify = () => {
    if (code) {
      router.push(`/verify?email=${encodeURIComponent(email)}`)
    }
  }

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-50 p-4">
      <Card className="w-full max-w-md">
        <CardHeader>
          <CardTitle>Preview Verification Codes</CardTitle>
          <CardDescription>
            For development and testing only. This page allows you to retrieve verification codes for accounts you've
            created.
          </CardDescription>
        </CardHeader>
        <CardContent className="space-y-4">
          <div className="space-y-2">
            <Label htmlFor="email">Email Address</Label>
            <Input
              id="email"
              type="email"
              placeholder="Enter your email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>

          {error && <div className="rounded-md bg-red-50 p-3 text-sm text-red-800">{error}</div>}

          {code && (
            <div className="rounded-md bg-green-50 p-4">
              <div className="text-sm text-green-800 mb-2">Verification code for {email}:</div>
              <div className="text-2xl font-mono tracking-wider text-center font-bold text-green-700">{code}</div>
            </div>
          )}
        </CardContent>
        <CardFooter className="flex flex-col space-y-3">
          <Button className="w-full" onClick={fetchCode} disabled={loading}>
            {loading ? "Retrieving..." : "Get Verification Code"}
          </Button>

          {code && (
            <Button className="w-full" variant="outline" onClick={goToVerify}>
              Go to Verification Page
            </Button>
          )}
        </CardFooter>
      </Card>
    </div>
  )
}
