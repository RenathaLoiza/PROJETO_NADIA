import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { motion, AnimatePresence } from "framer-motion"
import { 
  Send, 
  Bot, 
  User, 
  FileText, 
  Calculator,
  Clock,
  Loader2
} from "lucide-react"
import { cn } from "@/lib/utils"

interface Message {
  id: string
  content: string
  type: 'user' | 'bot'
  timestamp: Date
  metadata?: {
    source?: string
    calculations?: string[]
    confidence?: number
  }
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Olá! Sou o EconoBot, seu assistente especializado em economia. Posso ajudá-lo com análises econômicas, cálculos financeiros e consultas a documentos especializados. Como posso ajudá-lo hoje?',
      type: 'bot',
      timestamp: new Date(),
      metadata: {
        confidence: 100
      }
    }
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isLoading) return

    const userMessage: Message = {
      id: Date.now().toString(),
      content: inputValue,
      type: 'user',
      timestamp: new Date()
    }

    setMessages(prev => [...prev, userMessage])
    setInputValue("")
    setIsLoading(true)

    // Simulate bot response
    setTimeout(() => {
      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        content: `Obrigado por sua pergunta sobre "${inputValue}". Como um agente RAG especializado em economia, estou processando sua consulta através da minha base de conhecimento. Em breve implementaremos a integração completa com documentos econômicos e cálculos em tempo real.`,
        type: 'bot',
        timestamp: new Date(),
        metadata: {
          source: "Base de Conhecimento Econômico",
          calculations: inputValue.includes('cálculo') ? ['PIB = C + I + G + (X - M)', 'Taxa de Juros Compostos'] : undefined,
          confidence: 85
        }
      }
      setMessages(prev => [...prev, botMessage])
      setIsLoading(false)
    }, 2000)
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  return (
    <div className="min-h-screen pt-16 flex flex-col bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-card/50 backdrop-blur-sm border-b border-border/50"
      >
        <div className="container mx-auto px-4 py-6">
          <div className="flex items-center space-x-3">
            <div className="p-2 rounded-lg bg-gradient-primary neon-glow">
              <Bot className="h-6 w-6 text-white" />
            </div>
            <div>
              <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">
                EconoBot
              </h1>
              <p className="text-muted-foreground text-sm">
                Agente RAG para Análises Econômicas
              </p>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Messages Container */}
      <div className="flex-1 overflow-hidden">
        <div className="h-full flex flex-col">
          <div className="flex-1 overflow-y-auto p-4 space-y-4">
            <AnimatePresence>
              {messages.map((message, index) => (
                <motion.div
                  key={message.id}
                  initial={{ opacity: 0, y: 20, scale: 0.95 }}
                  animate={{ opacity: 1, y: 0, scale: 1 }}
                  exit={{ opacity: 0, y: -20, scale: 0.95 }}
                  transition={{ duration: 0.3, delay: index * 0.1 }}
                  className={cn(
                    "flex w-full",
                    message.type === 'user' ? "justify-end" : "justify-start"
                  )}
                >
                  <div className={cn(
                    "flex items-start space-x-3 max-w-3xl",
                    message.type === 'user' ? "flex-row-reverse space-x-reverse" : ""
                  )}>
                    {/* Avatar */}
                    <div className={cn(
                      "w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0",
                      message.type === 'user' 
                        ? "bg-gradient-accent neon-glow" 
                        : "bg-gradient-primary neon-glow"
                    )}>
                      {message.type === 'user' ? (
                        <User className="h-5 w-5 text-white" />
                      ) : (
                        <Bot className="h-5 w-5 text-white" />
                      )}
                    </div>

                    {/* Message Content */}
                    <Card className={cn(
                      "flex-1 border-border/50",
                      message.type === 'user'
                        ? "bg-gradient-accent/10 neon-glow border-neon-yellow/30"
                        : "bg-gradient-primary/10 neon-glow border-neon-blue/30"
                    )}>
                      <CardContent className="p-4">
                        <div className="space-y-3">
                          <p className="leading-relaxed">{message.content}</p>
                          
                          {/* Metadata */}
                          {message.metadata && (
                            <div className="space-y-2 pt-2 border-t border-border/30">
                              <div className="flex flex-wrap gap-2 items-center text-xs">
                                <div className="flex items-center space-x-1 text-muted-foreground">
                                  <Clock className="h-3 w-3" />
                                  <span>{message.timestamp.toLocaleTimeString()}</span>
                                </div>
                                
                                {message.metadata.confidence && (
                                  <Badge variant="outline" className="text-xs">
                                    Confiança: {message.metadata.confidence}%
                                  </Badge>
                                )}
                              </div>
                              
                              {message.metadata.source && (
                                <div className="flex items-center space-x-2 text-xs text-muted-foreground">
                                  <FileText className="h-3 w-3" />
                                  <span>Fonte: {message.metadata.source}</span>
                                </div>
                              )}
                              
                              {message.metadata.calculations && (
                                <div className="space-y-1">
                                  <div className="flex items-center space-x-1 text-xs text-muted-foreground">
                                    <Calculator className="h-3 w-3" />
                                    <span>Cálculos utilizados:</span>
                                  </div>
                                  <div className="space-y-1">
                                    {message.metadata.calculations.map((calc, idx) => (
                                      <code 
                                        key={idx}
                                        className="block text-xs bg-muted/50 rounded px-2 py-1 text-neon-yellow"
                                      >
                                        {calc}
                                      </code>
                                    ))}
                                  </div>
                                </div>
                              )}
                            </div>
                          )}
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </motion.div>
              ))}
              
              {/* Loading Message */}
              {isLoading && (
                <motion.div
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -20 }}
                  className="flex justify-start"
                >
                  <div className="flex items-start space-x-3 max-w-3xl">
                    <div className="w-10 h-10 rounded-full flex items-center justify-center bg-gradient-primary neon-glow">
                      <Bot className="h-5 w-5 text-white" />
                    </div>
                    <Card className="bg-gradient-primary/10 neon-glow border-neon-blue/30">
                      <CardContent className="p-4">
                        <div className="flex items-center space-x-2">
                          <Loader2 className="h-4 w-4 animate-spin text-neon-blue" />
                          <span className="text-muted-foreground">
                            Processando sua consulta...
                          </span>
                        </div>
                      </CardContent>
                    </Card>
                  </div>
                </motion.div>
              )}
            </AnimatePresence>
          </div>

          {/* Input Area */}
          <motion.div 
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="sticky bottom-0 bg-background/80 backdrop-blur-md border-t border-border/50 p-4"
          >
            <div className="container mx-auto">
              <div className="flex items-center space-x-3 max-w-4xl mx-auto">
                <div className="flex-1 relative">
                  <Input
                    value={inputValue}
                    onChange={(e) => setInputValue(e.target.value)}
                    onKeyPress={handleKeyPress}
                    placeholder="Digite sua pergunta sobre economia..."
                    disabled={isLoading}
                    className="pr-12 bg-card/50 border-border/50 focus:border-neon-blue/50 focus:neon-glow"
                  />
                </div>
                <Button 
                  onClick={handleSendMessage}
                  disabled={!inputValue.trim() || isLoading}
                  size="icon"
                  className="bg-gradient-primary hover:opacity-90 neon-glow text-white"
                >
                  {isLoading ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : (
                    <Send className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>
          </motion.div>
        </div>
      </div>
    </div>
  )
}