import { Link } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Card, CardContent } from "@/components/ui/card"
import { motion } from "framer-motion"
import { 
  ArrowRight, 
  Brain, 
  Calculator, 
  FileText, 
  TrendingUp,
  Zap,
  Sparkles
} from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen pt-16">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-neon-blue/5 via-background to-neon-purple/5" />
        <div className="container mx-auto px-4 py-20 relative">
          <div className="text-center space-y-8 max-w-4xl mx-auto">
            {/* Logo */}
            <motion.div
              initial={{ scale: 0, rotate: -180 }}
              animate={{ scale: 1, rotate: 0 }}
              transition={{ duration: 0.8, type: "spring" }}
              className="flex justify-center"
            >
              <div className="p-6 rounded-2xl bg-gradient-primary neon-glow-strong">
                <Zap className="h-16 w-16 text-white" />
              </div>
            </motion.div>

            {/* Title */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.2 }}
              className="space-y-4"
            >
              <h1 className="text-4xl md:text-6xl font-bold">
                <span className="bg-gradient-primary bg-clip-text text-transparent">
                  EconoBot
                </span>
              </h1>
              <p className="text-xl md:text-2xl text-muted-foreground max-w-2xl mx-auto">
                Seu agente inteligente para análises econômicas e cálculos financeiros
              </p>
            </motion.div>

            {/* CTA Button */}
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 0.4 }}
            >
              <Button 
                asChild 
                size="lg" 
                className="bg-gradient-primary hover:opacity-90 neon-glow text-white font-semibold px-8 py-4 text-lg group"
              >
                <Link to="/chatbot" className="flex items-center space-x-2">
                  <span>Acessar Chatbot</span>
                  <motion.div
                    whileHover={{ x: 5 }}
                    transition={{ type: "spring", stiffness: 400 }}
                  >
                    <ArrowRight className="h-5 w-5" />
                  </motion.div>
                </Link>
              </Button>
            </motion.div>
          </div>
        </div>
      </section>

      {/* Features Section */}
      <section className="py-20 bg-muted/20">
        <div className="container mx-auto px-4">
          <motion.div
            initial={{ opacity: 0, y: 30 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.6 }}
            className="text-center mb-16"
          >
            <h2 className="text-3xl md:text-4xl font-bold mb-4">
              Recursos do <span className="text-neon-purple">EconoBot</span>
            </h2>
            <p className="text-muted-foreground text-lg max-w-2xl mx-auto">
              Um agente RAG especializado em economia que combina análise de documentos 
              com cálculos financeiros avançados
            </p>
          </motion.div>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
            {[
              {
                icon: Brain,
                title: "IA Avançada",
                description: "Processamento inteligente de perguntas econômicas com RAG",
                color: "neon-blue",
                delay: 0.8
              },
              {
                icon: FileText,
                title: "Consulta Documentos",
                description: "Acesso instantâneo a base de conhecimento econômico",
                color: "neon-purple",
                delay: 1.0
              },
              {
                icon: Calculator,
                title: "Cálculos Precisos",
                description: "Execução de cálculos financeiros complexos em tempo real",
                color: "neon-yellow",
                delay: 1.2
              },
              {
                icon: TrendingUp,
                title: "Análises Detalhadas",
                description: "Interpretação e análise de dados econômicos",
                color: "neon-blue",
                delay: 1.4
              }
            ].map((feature, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 30 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.6, delay: feature.delay }}
                whileHover={{ y: -5 }}
              >
                <Card className="h-full bg-card/50 backdrop-blur-sm border-border/50 hover:neon-glow transition-all duration-300">
                  <CardContent className="p-6 text-center space-y-4">
                    <div className={`inline-flex p-3 rounded-xl bg-gradient-to-br from-${feature.color}/20 to-${feature.color}/10`}>
                      <feature.icon className={`h-8 w-8 text-${feature.color}`} />
                    </div>
                    <h3 className="text-xl font-semibold">{feature.title}</h3>
                    <p className="text-muted-foreground">{feature.description}</p>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </div>
        </div>
      </section>

      {/* About Section */}
      <section className="py-20">
        <div className="container mx-auto px-4">
          <div className="max-w-4xl mx-auto">
            <motion.div
              initial={{ opacity: 0, y: 30 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.6, delay: 1.6 }}
              className="text-center space-y-8"
            >
              <div className="space-y-4">
                <h2 className="text-3xl md:text-4xl font-bold">
                  Sobre o <span className="text-neon-yellow">Projeto</span>
                </h2>
                <div className="flex justify-center">
                  <Sparkles className="h-8 w-8 text-neon-yellow" />
                </div>
              </div>
              
              <Card className="bg-card/30 backdrop-blur-sm border-border/50 neon-glow">
                <CardContent className="p-8">
                  <div className="space-y-6 text-left">
                    <p className="text-lg leading-relaxed">
                      O <strong className="text-neon-blue">EconoBot</strong> é um agente RAG (Retrieval-Augmented Generation) 
                      especializado em economia que combina o poder da inteligência artificial com uma vasta 
                      base de conhecimento econômico.
                    </p>
                    
                    <p className="text-lg leading-relaxed">
                      Através de processamento de linguagem natural avançado, o sistema é capaz de:
                    </p>
                    
                    <ul className="space-y-3 text-muted-foreground">
                      <li className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-neon-blue mt-3 flex-shrink-0" />
                        <span>Consultar documentos econômicos e extrair informações relevantes</span>
                      </li>
                      <li className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-neon-purple mt-3 flex-shrink-0" />
                        <span>Realizar cálculos financeiros complexos com precisão</span>
                      </li>
                      <li className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-neon-yellow mt-3 flex-shrink-0" />
                        <span>Fornecer análises detalhadas e insights econômicos</span>
                      </li>
                      <li className="flex items-start space-x-3">
                        <div className="w-2 h-2 rounded-full bg-neon-blue mt-3 flex-shrink-0" />
                        <span>Apresentar respostas contextualizadas com fontes confiáveis</span>
                      </li>
                    </ul>
                  </div>
                </CardContent>
              </Card>
            </motion.div>
          </div>
        </div>
      </section>
    </div>
  )
}