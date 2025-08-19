import { useState, useEffect, useRef } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card, CardContent } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { motion, AnimatePresence } from "framer-motion"
import { Send, Bot, User } from "lucide-react"
import ParticleSwarmLoader from "@/components/ParticleSwarmLoader"
import Modal from "@/components/ui/modal"
import ReactMarkdown from "react-markdown"
import remarkGfm from "remark-gfm"
import { cn } from "@/lib/utils"

const API_URL = import.meta.env.VITE_API_URL;

interface Message {
  id: string
  content: string
  type: 'user' | 'bot'
  timestamp: Date
  metadata?: {
    source?: string[]
    calculations?: string[]
    confidence?: number
  }
  feedback?: boolean
}

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: '1',
      content: 'Olá! Sou eu a NADIA, sua assistente especializado em economia. Posso ajudá-lo com análises econômicas, Realizar cálculos financeiros e consultas a documentos PDF. Como posso ajudá-lo hoje?',
      type: 'bot',
      timestamp: new Date(),
      metadata: { confidence: 100 }
    }
  ])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [progress, setProgress] = useState(0)
  const [stepIdx, setStepIdx] = useState(0)
  const [modalOpen, setModalOpen] = useState(false)
  const [currentFeedbackMessage, setCurrentFeedbackMessage] = useState<Message | null>(null)
  const [feedbackComment, setFeedbackComment] = useState("")
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const loadingSteps = [
    "🔍 Buscando documentos...",
    "📑 Carregando PDFs...",
    "🤖 Analisando dados...",
    "✍️ Preparando resposta..."
  ]

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, isLoading])

  useEffect(() => {
    if (!isLoading) return setStepIdx(0), setProgress(0)

    const totalDuration = 8000
    const stepDuration = totalDuration / loadingSteps.length
    let currentTime = 0

    const interval = setInterval(() => {
      currentTime += 100
      setProgress(Math.min((currentTime / totalDuration) * 100, 100))
      setStepIdx(Math.min(Math.floor(currentTime / stepDuration), loadingSteps.length - 1))
      if (currentTime >= totalDuration) clearInterval(interval)
    }, 100)

    return () => clearInterval(interval)
  }, [isLoading])

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

    try {
      const response = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content })
      })
      const data = await response.json()

      const botMessage: Message = {
        id: (Date.now() + 1).toString(),
        type: 'bot',
        content: data.response || "Não consegui gerar resposta.",
        timestamp: new Date(),
        metadata: { source: data.source, calculations: data.snippets, confidence: 85 }
      }
      setMessages(prev => [...prev, botMessage])
    } catch (e) {
      setMessages(prev => [...prev, { id: (Date.now()+1).toString(), type: 'bot', content: 'Erro ao se comunicar com servidor.', timestamp: new Date() }])
    } finally {
      setIsLoading(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const openFeedbackModal = (message: Message) => {
    setCurrentFeedbackMessage(message)
    setFeedbackComment("")
    setModalOpen(true)
  }

  const submitFeedback = async (isPositive: boolean) => {
    if (!currentFeedbackMessage) return
    await fetch(`${API_URL}/api/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        messageId: currentFeedbackMessage.id,
        isPositive,
        userComment: feedbackComment,
        question: currentFeedbackMessage.content,
        answer: currentFeedbackMessage.content
      })
    })
    setMessages(prev => prev.map(msg => msg.id === currentFeedbackMessage.id ? { ...msg, feedback: isPositive } : msg))
    setModalOpen(false)
    setCurrentFeedbackMessage(null)
    setFeedbackComment("")
  }

  const CircularProgress = ({ progress }: { progress: number }) => {
    const radius = 24
    const stroke = 3
    const normalizedRadius = radius - stroke * 2
    const circumference = normalizedRadius * 2 * Math.PI
    const strokeDashoffset = circumference - (progress / 100) * circumference

    return (
      <svg height={radius*2} width={radius*2}>
        <circle
          stroke="#0ff"
          fill="transparent"
          strokeWidth={stroke}
          strokeLinecap="round"
          r={normalizedRadius}
          cx={radius}
          cy={radius}
          style={{ strokeDasharray: circumference, strokeDashoffset, transition: "stroke-dashoffset 0.1s linear" }}
        />
      </svg>
    )
  }

  return (
    <div className="min-h-screen flex flex-col bg-gradient-to-br from-background via-background to-muted/20">
      {/* Header */}
      <motion.div className="bg-card/50 backdrop-blur-sm border-b border-border/50" initial={{ opacity:0, y:-20 }} animate={{ opacity:1, y:0 }}>
        <div className="container mx-auto px-4 py-6 flex items-center space-x-3">
          <div className="p-2 rounded-lg bg-gradient-primary neon-glow"><Bot className="h-6 w-6 text-white"/></div>
          <div>
            <h1 className="text-2xl font-bold bg-gradient-primary bg-clip-text text-transparent">NADIA</h1>
            <p className="text-muted-foreground text-sm">Agente RAG para PDFs Econômicos</p>
          </div>
        </div>
      </motion.div>

      {/* Chat Area */}
      <div className="flex-1 overflow-hidden flex flex-col px-4">
        <div className="flex-1 overflow-y-auto py-4 space-y-4">
          <AnimatePresence>
            {messages.map((msg) => (
              <motion.div key={msg.id} initial={{ opacity:0, y:20 }} animate={{ opacity:1, y:0 }} exit={{ opacity:0, y:-20 }} className={`flex ${msg.type==='user'?'justify-end':'justify-start'}`}>
                <div className={`flex max-w-3xl ${msg.type==='user'?'flex-row-reverse':'flex-row'}`}>
                  <div className={cn("w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0", msg.type==='user'?'bg-gradient-accent neon-glow':'bg-gradient-primary neon-glow')}>
                    {msg.type==='user'?<User className="h-5 w-5 text-white"/>:<Bot className="h-5 w-5 text-white"/>}
                  </div>
                  <Card className={cn("flex-1 border-border/50", msg.type==='user'?'bg-gradient-accent/10 neon-glow border-neon-yellow/30':'bg-gradient-primary/10 neon-glow border-neon-blue/30')}>
                    <CardContent className="p-4">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} className="prose prose-invert">
                        {msg.content}
                      </ReactMarkdown>
                      {msg.metadata?.confidence && (
                        <Badge className="mt-2 bg-blue-500/50 text-white">Confiança: {msg.metadata.confidence}%</Badge>
                      )}
                      {msg.metadata?.source && (
                        <div className="text-xs text-muted-foreground mt-2">
                          <span className="font-semibold">🔗 Fontes:</span>
                          {msg.metadata.source.map((src, i) => <div key={i}><a href={src} target="_blank" className="text-blue-600 underline">{src}</a></div>)}
                        </div>
                      )}
                      {msg.metadata?.calculations && (
                        <div className="text-xs text-muted-foreground mt-2">
                          <span className="font-semibold">📊 Cálculos:</span>
                          {msg.metadata.calculations.map((c,i)=><div key={i}><code>{c}</code></div>)}
                        </div>
                      )}
                      <div className="flex items-center space-x-2 mt-2">
                        <Button variant="ghost" size="sm" onClick={()=>submitFeedback(true)}>👍</Button>
                        <Button variant="ghost" size="sm" onClick={()=>openFeedbackModal(msg)}>👎</Button>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              </motion.div>
            ))}

            {isLoading && (
              <motion.div initial={{opacity:0,y:20}} animate={{opacity:1,y:0}} className="flex justify-center">
                <div className="flex flex-col items-center space-y-2">
                  <CircularProgress progress={progress} />
                  <p className="text-xs mt-1">{loadingSteps[stepIdx]}</p>
                  <ParticleSwarmLoader />
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <div className="sticky bottom-0 bg-background/80 backdrop-blur-md border-t border-border/50 p-4 flex items-center space-x-3">
          <Input value={inputValue} onChange={e=>setInputValue(e.target.value)} onKeyPress={handleKeyPress} placeholder="Digite sua pergunta sobre PDFs..." disabled={isLoading} className="flex-1"/>
          <Button onClick={handleSendMessage} disabled={!inputValue.trim() || isLoading}><Send className="w-4 h-4"/></Button>
        </div>
      </div>

      {/* Modal */}
      {modalOpen && currentFeedbackMessage && (
        <Modal onClose={()=>setModalOpen(false)} title="Feedback Negativo">
          <textarea value={feedbackComment} onChange={e=>setFeedbackComment(e.target.value)} className="w-full p-2 border rounded" placeholder="Explique o problema..."/>
          <div className="flex justify-end mt-2 space-x-2">
            <Button variant="secondary" onClick={()=>setModalOpen(false)}>Cancelar</Button>
            <Button onClick={()=>submitFeedback(false)}>Enviar Feedback</Button>
          </div>
        </Modal>
      )}
    </div>
  )
}
