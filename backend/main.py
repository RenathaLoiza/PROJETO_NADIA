import os
import sys
import uvicorn
from embedding import process_documents_to_chromadb

# Caminho do arquivo da API FastAPI
api_file = "api.py"

def run_api():
    """Função para iniciar a API FastAPI."""
    print(f"Iniciando API FastAPI com '{api_file}'...")
    try:
        uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
    except Exception as e:
        print(f"Erro ao executar FastAPI: {e}")

def main():
    """Função principal que gerencia o fluxo do programa."""
    
    # Verifica se a base de dados precisa ser populada
    if "--populate" in sys.argv:
        print("\n🚀 Iniciando a criação da base de dados vetorial...")
        try:
            process_documents_to_chromadb(
                data_path="data", 
                chroma_path="chroma_db", 
                collection_name="seade_gecon"
            )
            print("✅ Base de dados criada/atualizada com sucesso!")
        except Exception as e:
            print(f"❌ Erro durante o processamento de embeddings: {e}")
            return

    # Verifica se o arquivo da API existe
    if not os.path.exists(api_file):
        print(f"Erro: O arquivo '{api_file}' não foi encontrado.")
    else:
        run_api()

if __name__ == "__main__":
    main()
