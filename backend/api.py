from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from agent import RAGAgentReact

app = FastAPI()

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8001",
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://192.168.8.55:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo para a requisição
class QueryRequest(BaseModel):
    question: str

# Inicializar o agente RAG
rag_agent = RAGAgentReact()

# Endpoint raiz para testes rápidos
@app.get("/")
async def root():
    return {"message": "API RAG ativa!"}

@app.post("/query")
async def query_rag(request: QueryRequest):
    """
    Endpoint para consultar o agente RAG.
    Recebe uma pergunta e retorna a resposta do agente.
    """
    try:
        response = rag_agent.consultar(request.question)
        return {"answer": response}
    except Exception as e:
        return {"error": f"Erro ao processar a pergunta: {str(e)}"}

@app.get("/health")
async def health_check():
    """
    Endpoint para verificar o status do sistema.
    """
    system_info = rag_agent.get_system_info()
    return {
        "status": "healthy" if system_info.get("agent_ready") else "unhealthy",
        "details": system_info
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
