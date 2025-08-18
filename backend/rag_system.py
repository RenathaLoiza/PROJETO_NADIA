import chromadb
from openai import OpenAI
from dotenv import load_dotenv
import os
from typing import List, Dict, Any, Optional, Tuple
import logging
import csv
from datetime import datetime
import numpy as np
from chromadb.utils import embedding_functions

# Configuração de logging (mantido para depuração interna, mas o logging de queries está desabilitado)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importação condicional do reranker
try:
    from sentence_transformers import CrossEncoder
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    logger.warning("sentence_transformers não disponível. Reranqueamento desabilitado.")

class RagSystem:
    """Sistema RAG aprimorado com reranking, fallback e logging avançado."""
    
    def __init__(self, 
                 chroma_path: str = "chroma_db", 
                 collection_name: str = "seade_gecon",
                 reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
                 enable_reranking: bool = True,
                 enable_logging: bool = True,
                 **kwargs):
        """
        Inicializa o sistema RAG aprimorado.
        """
        load_dotenv()
        
        if not os.getenv('OPENAI_API_KEY'):
            raise ValueError("OPENAI_API_KEY não encontrada nas variáveis de ambiente")
        
        self.chroma_path = chroma_path
        self.collection_name = collection_name
        self.enable_reranking = enable_reranking and RERANKER_AVAILABLE
        self.enable_logging = False  # Desabilitado conforme solicitado
        self.log_file = f"rag_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        self.chroma_client = chromadb.PersistentClient(path=self.chroma_path)
        
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=os.environ.get("OPENAI_API_KEY"),
            model_name="text-embedding-3-small"
        )
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name, 
            embedding_function=self.embedding_function
        )
        
        self.reranker = None
        if self.enable_reranking:
            logger.info("Carregando modelo reranker...")
            try:
                self.reranker = CrossEncoder(reranker_model)
                logger.info("✅ Modelo reranker carregado.")
            except Exception as e:
                logger.error(f"Erro ao carregar reranker. Desabilitando. Erro: {e}")
                self.enable_reranking = False

        self.openai_client = OpenAI()
        
        # Prompt do sistema atualizado para conteúdo multimodal e formato conciso para cálculos
        self.system_prompt_template = """
        Você é um assistente especializado na economia do setor automotivo de São Paulo.
        
        Use **apenas** os dados fornecidos abaixo para responder à pergunta do usuário. 
        **Nunca invente informações. Se não houver dados suficientes, diga isso com clareza.**
        
        Os documentos fornecidos podem conter:
        1. **Texto puro** do documento.
        2. **DESCRIÇÃO VISUAL:** Uma descrição textual detalhada de imagens, gráficos, ou tabelas extraída por um modelo de IA. Use essas descrições para responder perguntas sobre o conteúdo visual do documento.
        
        Sua resposta deve:
        - Ser clara, direta e bem estruturada.
        - Incluir fatos, números e fontes sempre que possível.
        - Usar estruturas como listas, seções ou tópicos quando apropriado.
        - Evitar repetições e redundâncias.
        - Estar em português formal e técnico.
        - Indicar claramente quando as informações são limitadas.
        
        Se a pergunta envolver cálculos matemáticos (ex.: percentuais, somas, taxas), responda **APENAS** com os passos matemáticos em formato LaTeX, sem texto explicativo adicional. Use \\[ para equações display e inclua o resultado final.
        
        Se os dados fornecidos forem insuficientes ou irrelevantes para a pergunta, responda:
        "Não tenho informações suficientes para responder essa pergunta com base nos dados disponíveis. 
        Você poderia reformular ou especificar melhor a pergunta?"
        
        📚 Documentos relevantes encontrados:
        {documents}
        
        💡 Confiança dos documentos: {confidence_scores}
        """

    def _query_vector_db(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """
        Consulta o banco de dados vetorial.
        """
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=['metadatas', 'documents', 'distances']
            )
            
            formatted_results = []
            for i in range(len(results['documents'][0])):
                formatted_results.append({
                    'document': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
                
            return formatted_results
        except Exception as e:
            logger.error(f"Erro ao consultar ChromaDB: {e}")
            return []

    def _rerank_documents(self, query: str, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranqueia os documentos usando CrossEncoder.
        """
        if not self.enable_reranking or not self.reranker:
            return sorted(docs, key=lambda x: x['distance'])
        
        try:
            logger.info(f"  -> Reranqueando {len(docs)} documentos...")
            pairs = [[query, doc['document']] for doc in docs]
            scores = self.reranker.predict(pairs)
            
            for doc, score in zip(docs, scores):
                doc['rerank_score'] = float(score)
                
            return sorted(docs, key=lambda x: x['rerank_score'], reverse=True)
        except Exception as e:
            logger.warning(f"Erro no reranking, usando ordenação original: {e}")
            return sorted(docs, key=lambda x: x['distance'])

    def _format_docs(self, docs: List[Dict[str, Any]], top_k_reranked: int = 5) -> Tuple[str, str]:
        """
        Formata os documentos recuperados para inclusão no prompt.
        """
        top_docs = docs[:top_k_reranked]
        docs_str = []
        confidence_scores = []
        
        for idx, doc in enumerate(top_docs, 1):
            metadata = doc['metadata']
            source_info = f"Documento: {metadata.get('file_name', 'N/A')} | Página: {metadata.get('page', 'N/A')} | Chunk: {metadata.get('chunk_index', 'N/A')}/{metadata.get('total_chunks', 'N/A')}"
            content = doc['document']
            
            docs_str.append(f"[{idx}] {source_info}\nConteúdo: {content}")
            
            score = doc.get('rerank_score', 1 - doc.get('distance', 1))
            confidence_scores.append(f"{score:.4f}")
            
        return "\n\n".join(docs_str), ", ".join(confidence_scores)

    def _generate_response_with_openai(self, query: str, formatted_docs: str, confidence_scores: str) -> str:
        """
        Gera a resposta final usando a API da OpenAI.
        """
        try:
            system_prompt = self.system_prompt_template.format(
                documents=formatted_docs,
                confidence_scores=confidence_scores
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            response = self.openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.2,
                max_tokens=2048
            )
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Erro ao gerar resposta com a OpenAI: {e}")
            return "Ocorreu um erro ao gerar a resposta. Por favor, tente novamente."

    def log_query(self, query: str, result: Dict[str, Any]):
        """
        Registra a query e o resultado em um arquivo CSV.
        """
        if not self.enable_logging:
            return
        
        file_exists = os.path.isfile(self.log_file)
        with open(self.log_file, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    'timestamp', 'query', 'response', 'retrieved_docs_count', 
                    'reranked_docs_count', 'reranking_enabled', 'confidence_scores'
                ])
            
            retrieved_count = len(result['retrieved_documents']) if result.get('retrieved_documents') else 0
            reranked_count = len(result['reranked_documents']) if result.get('reranked_documents') else 0
            
            writer.writerow([
                datetime.now().isoformat(),
                query,
                result.get('response', 'N/A'),
                retrieved_count,
                reranked_count,
                result.get('reranking_enabled', False),
                result.get('confidence_scores', 'N/A')
            ])

    def query_rag_system(self, query: str, top_k_retrieval: int = 10, top_k_reranked: int = 5) -> Dict[str, Any]:
        """
        Executa a pipeline completa de RAG e retorna o resultado.
        """
        logger.info(f"Pergunta do usuário: '{query}'")
        
        retrieved_docs = self._query_vector_db(query, top_k=top_k_retrieval)
        
        if not retrieved_docs:
            logger.warning("Nenhum documento relevante encontrado.")
            return {
                "response": "Nenhum documento relevante encontrado para essa pergunta.",
                "retrieved_documents": [],
                "reranked_documents": [],
                "reranking_enabled": self.enable_reranking,
                "confidence_scores": "N/A",
                "error": "No documents found."
            }
            
        reranked_docs = self._rerank_documents(query, retrieved_docs)

        formatted_docs, confidence_scores = self._format_docs(reranked_docs, top_k_reranked=top_k_reranked)

        final_response = self._generate_response_with_openai(query, formatted_docs, confidence_scores)
        
        result = {
            "response": final_response,
            "retrieved_documents": retrieved_docs,
            "reranked_documents": reranked_docs,
            "reranking_enabled": self.enable_reranking,
            "confidence_scores": confidence_scores
        }
        
        self.log_query(query, result)
        
        logger.info("✅ Resposta gerada com sucesso.")
        return result

    def get_system_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o status do sistema RAG."""
        try:
            num_docs = self.collection.count()
            rag_available = num_docs > 0
            return {
                "rag_available": rag_available,
                "rag_status": f"{num_docs} documentos carregados." if rag_available else "Base de dados vazia.",
                "reranking_enabled": self.enable_reranking,
                "llm_model": "gpt-4o"
            }
        except Exception as e:
            return {
                "rag_available": False,
                "rag_status": f"Erro ao acessar ChromaDB: {e}",
                "reranking_enabled": self.enable_reranking,
                "llm_model": "gpt-4o"
            }