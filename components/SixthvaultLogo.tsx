import Image from "next/image"
import { useState } from "react"

interface SixthvaultLogoProps {
  size?: "small" | "medium" | "large" | "full" | "section-fill"
}

export default function SixthvaultLogo({ size = "medium" }: SixthvaultLogoProps) {
  const [imageError, setImageError] = useState(false)
  
  const sizeClasses = {
    small: { width: 120, height: 40, text: "text-xl", containerClass: "w-[120px] h-[40px]" },
    medium: { width: 180, height: 60, text: "text-2xl", containerClass: "w-[180px] h-[60px]" },
    large: { width: 240, height: 80, text: "text-3xl", containerClass: "w-[240px] h-[80px]" },
    full: { width: 400, height: 120, text: "text-4xl", containerClass: "w-full h-full" },
    "section-fill": { width: 800, height: 300, text: "text-5xl", containerClass: "w-full h-full min-h-[200px]" },
  }

  const dimensions = sizeClasses[size]

  if (imageError) {
    return (
      <div 
        className={`flex items-center justify-center bg-gradient-to-r from-indigo-700 via-blue-800 to-slate-800 text-white font-bold rounded-lg ${dimensions.containerClass}`}
      >
        <span className={dimensions.text}>SixthVault</span>
      </div>
    )
  }

  return (
    <div className={`flex items-center justify-center ${dimensions.containerClass}`}>
      <Image
        src="/logo.jpg"
        alt="SixthVault Logo"
        width={dimensions.width}
        height={dimensions.height}
        className="object-contain w-full h-full"
        priority
        onError={() => setImageError(true)}
        unoptimized={true}
      />
    </div>
  )
}
