import Image from "next/image"
import { useState } from "react"

interface SixthvaultLogoProps {
  size?: "small" | "medium" | "large" | "full"
}

export default function SixthvaultLogo({ size = "medium" }: SixthvaultLogoProps) {
  const [imageError, setImageError] = useState(false)
  
  const sizeClasses = {
    small: { width: 120, height: 40, text: "text-xl" },
    medium: { width: 180, height: 60, text: "text-2xl" },
    large: { width: 240, height: 80, text: "text-3xl" },
    full: { width: "100%", height: "100%", text: "text-4xl" },
  }

  const dimensions = sizeClasses[size]

  if (imageError) {
    return (
      <div 
        className={`flex items-center justify-center bg-gradient-to-r from-indigo-700 via-blue-800 to-slate-800 text-white font-bold rounded-lg ${
          size === "full" ? "w-full h-full" : ""
        }`}
        style={size !== "full" ? { width: dimensions.width, height: dimensions.height } : {}}
      >
        <span className={dimensions.text}>SixthVault</span>
      </div>
    )
  }

  return (
    <div className={`flex items-center ${size === "full" ? "w-full h-full" : ""}`}>
      <Image
        src="/logo.jpg"
        alt="SixthVault Logo"
        width={size === "full" ? 400 : dimensions.width as number}
        height={size === "full" ? 120 : dimensions.height as number}
        className={`${size === "full" ? "w-full h-full object-cover" : "object-contain"}`}
        priority
        onError={() => setImageError(true)}
      />
    </div>
  )
}
