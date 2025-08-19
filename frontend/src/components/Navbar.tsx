import { Link, useLocation } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { ThemeToggle } from "./ThemeToggle"
import { Bot, Home, Zap } from "lucide-react"
import { motion } from "framer-motion"
import { cn } from "@/lib/utils"

export function Navbar() {
  const location = useLocation()

  const isActive = (path: string) => location.pathname === path

  return (
    <motion.nav 
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md border-b border-border/50 neon-glow"
    >
      <div className="container mx-auto px-4 h-16 flex items-center justify-between">
        {/* Logo */}
        <Link to="/" className="flex items-center space-x-2 group">
          <motion.div 
            whileHover={{ rotate: 360 }}
            transition={{ duration: 0.5 }}
            className="p-2 rounded-lg bg-gradient-primary neon-glow"
          >
            <Zap className="h-6 w-6 text-white" />
          </motion.div>
          <span className="font-bold text-xl bg-gradient-primary bg-clip-text text-transparent">
            EconoBot
          </span>
        </Link>

        {/* Navigation Links */}
        <div className="flex items-center space-x-1">
          <Button
            asChild
            variant="ghost"
            className={cn(
              "relative transition-all duration-200",
              isActive("/") 
                ? "text-neon-blue neon-glow" 
                : "hover:text-neon-blue hover:neon-glow"
            )}
          >
            <Link to="/" className="flex items-center space-x-2">
              <Home className="h-4 w-4" />
              <span>Início</span>
              {isActive("/") && (
                <motion.div
                  layoutId="activeIndicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-neon-blue"
                  initial={false}
                />
              )}
            </Link>
          </Button>

          <Button
            asChild
            variant="ghost"
            className={cn(
              "relative transition-all duration-200",
              isActive("/chatbot") 
                ? "text-neon-purple neon-glow" 
                : "hover:text-neon-purple hover:neon-glow"
            )}
          >
            <Link to="/chatbot" className="flex items-center space-x-2">
              <Bot className="h-4 w-4" />
              <span>Chatbot</span>
              {isActive("/chatbot") && (
                <motion.div
                  layoutId="activeIndicator"
                  className="absolute bottom-0 left-0 right-0 h-0.5 bg-neon-purple"
                  initial={false}
                />
              )}
            </Link>
          </Button>

          <ThemeToggle />
        </div>
      </div>
    </motion.nav>
  )
}