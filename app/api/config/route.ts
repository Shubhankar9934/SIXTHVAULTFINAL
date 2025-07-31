import { NextResponse } from "next/server"

export async function GET() {
  // AI is available for all users with the provided GROQ API key
  const groqApiKey = "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"

  return NextResponse.json({
    groqApiKey,
    hasSupabase: !!process.env.NEXT_PUBLIC_SUPABASE_URL && !!process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    demoMode: !process.env.NEXT_PUBLIC_SUPABASE_URL || !process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY,
    apiKeySource: "provided",
    aiAvailable: true, // Always true for all users
    universalAccess: true, // Indicates AI is available to everyone
  })
}
