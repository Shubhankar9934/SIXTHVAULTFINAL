// import { type NextRequest, NextResponse } from "next/server"
// import { EmailService } from "@/lib/email-service"

// // This endpoint is for preview/development only
// // It allows retrieving verification codes for testing
// // In production, this should be removed or secured

// export async function GET(request: NextRequest) {
//   try {
//     const searchParams = request.nextUrl.searchParams
//     const email = searchParams.get("email")

//     if (!email) {
//       return NextResponse.json({ error: "Email parameter is required" }, { status: 400 })
//     }

//     const code = EmailService.getVerificationCode(email)

//     if (!code) {
//       return NextResponse.json({ error: "No verification code found for this email" }, { status: 404 })
//     }

//     return NextResponse.json({ email, code })
//   } catch (error) {
//     console.error("Error retrieving preview code:", error)
//     return NextResponse.json({ error: "Failed to retrieve verification code" }, { status: 500 })
//   }
// }
