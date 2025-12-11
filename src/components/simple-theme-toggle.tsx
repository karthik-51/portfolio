"use client"

import * as React from "react"
import { Moon, Sun } from "lucide-react"
import { useTheme } from "next-themes"
import { Button } from "@/components/ui/button"

export function SimpleThemeToggle() {
  const { theme, setTheme } = useTheme()
  const [mounted, setMounted] = React.useState(false)

  React.useEffect(() => {
    setMounted(true)
  }, [])

  if (!mounted) {
    return (
      <Button variant="outline" size="lg" className="rounded-full px-4 py-2 h-12 min-w-[120px]" disabled>
        <span className="text-sm font-medium">Loading...</span>
      </Button>
    )
  }

  return (
    <Button
      variant="outline"
      size="lg"
      onClick={() => setTheme(theme === "light" ? "dark" : "light")}
      className="rounded-full px-2 py-2 h-12 min-w-[60px] hover:scale-105 transition-all duration-200 shadow-lg hover:shadow-xl dark:shadow-cyan-300 dark:shadow-sm"
    >
      <div className="flex items-center gap-2">
        {theme === "light" ? (
          <>
            <Moon className="h-5 w-5" />
          </>
        ) : (
          <>
            <Sun className="h-5 w-5" />
          </>
        )}
      </div>
    </Button>
  )
}
