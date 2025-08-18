import os
import logging
from typing import Dict, Any, List
import warnings

# Carregar variáveis do arquivo .env
from dotenv import load_dotenv
load_dotenv()

# Desabilitar LangSmith (opcional - remove warnings)
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""

# Imports corretos para a nova API
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.prompts import PromptTemplate
from langchain.schema import HumanMessage, AIMessage

# LangGraph imports para memória
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict

# Import correto considerando que estamos na pasta rag
try:
    from rag_system import RagSystem
    RAG_AVAILABLE = True
except ImportError as e:
    RAG_AVAILABLE = False
    print(f"⚠️ Aviso: RagSystem não disponível: {e}")

# Verificar se ChromaDB está disponível
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    print("⚠️ Aviso: ChromaDB não disponível")

# Configurar logging
warnings.filterwarnings("ignore", category=UserWarning, module="langsmith")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Estado para o LangGraph
class ConversationState(TypedDict):
    messages: List[Dict[str, str]]
    last_user_message: str
    last_ai_message: str

# Adicione estes imports novos
from sympy import sympify, diff, integrate, solve, symbols, Matrix, latex, SympifyError, log as sym_log
import numpy as np
from scipy import stats
import statistics
import matplotlib.pyplot as plt  # Para descrições de gráficos (não gera imagem, só dados)
from io import StringIO  # Para capturar output de plots textuais

class RAGAgentReact:
    """
    Agente RAG aprimorado com tratamento robusto de erros e fallback,
    incluindo ferramenta math_calculator para cálculos matemáticos independentes e extract_table_data para extração de tabelas.
    """
    
    def __init__(self, openai_api_key: str = None):
        """Inicializa o agente RAG com configurações aprimoradas e tratamento de erro."""
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        else:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY não encontrada. Verifique o .env"
                )
            print(f"✅ API Key carregada do .env: {api_key[:10]}...")

        # Inicialização segura do sistema RAG
        self.rag_available = False
        self.rag_status = "not_initialized"
        
        if RAG_AVAILABLE and CHROMADB_AVAILABLE:
            try:
                print("🔄 Inicializando sistema RAG...")
                self.rag = RagSystem()
                system_info = self.rag.get_system_info()
                if system_info.get('rag_available', False):
                    self.rag_available = True
                    self.rag_status = "active"
                    print(f"✅ Sistema RAG inicializado: {system_info.get('rag_status', 'Status desconhecido')}")
                else:
                    self.rag_status = f"initialization_failed: {system_info.get('rag_status', 'Falha desconhecida')}"
                    print(f"⚠️ Sistema RAG com problemas: {system_info.get('rag_status', 'Falha desconhecida')}")
            except Exception as e:
                logger.error(f"Erro ao inicializar RAG: {e}")
                self.rag_status = f"error: {str(e)}"
                print(f"❌ Erro na inicialização do RAG: {e}")
        elif not CHROMADB_AVAILABLE:
            self.rag_status = "chromadb_not_available"
            print("❌ ChromaDB não disponível - instale com: pip install chromadb")
        else:
            self.rag_status = "rag_system_not_available"
            print("❌ RagSystem não disponível")

        # Configuração do LLM
        self.llm = ChatOpenAI(
            temperature=0,
            model="gpt-4o",
            max_tokens=8000,
            top_p=0.9,
        )

        # Memória LangGraph
        self.memory_saver = MemorySaver()
        self.thread_id = "main_conversation"
        self.conversation_state: ConversationState = {
            "messages": [],
            "last_user_message": "",
            "last_ai_message": ""
        }

        # Ferramentas e prompt
        self.tools = self._create_simplified_tools()
        self.prompt = self._create_simplified_prompt()

        # Agente
        self.agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=self.prompt
        )

        self.agent_executor = AgentExecutor(
            agent=self.agent,
            tools=self.tools,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=10,  # Aumentado para evitar limite
            max_execution_time=120  # Aumentado para evitar timeout
        )

    def _create_simplified_tools(self) -> List[Tool]:
        """Cria ferramentas simplificadas para o agente."""
        tools = []
        
        if self.rag_available:
            # Ferramenta principal para consulta geral
            tools.append(
                Tool(
                    name="consultar_base_conhecimento",
                    func=self._consultar_rag_direto,
                    description="""FERRAMENTA PRINCIPAL: Consulta a base de conhecimento sobre economia de São Paulo.
                    Use esta ferramenta para responder perguntas sobre:
                    - Indústria (automotiva, têxtil, farmacêutica, metalúrgica, etc.)
                    - Economia do Estado de São Paulo
                    - Dados estatísticos e indicadores
                    - Mapa da Indústria Paulista
                    - Balança Comercial
                    - Agropecuária e outros setores
                    
                    Input: A pergunta exata do usuário
                    Output: Resposta completa baseada na base de conhecimento"""
                )
            )
            # Nova ferramenta para extração de tabelas
            tools.append(
                Tool(
                    name="extract_table_data",
                    func=self._extract_table_data,
                    description="""Extrai dados de tabelas da base de conhecimento sobre economia de São Paulo.
                    Use esta ferramenta para perguntas específicas sobre tabelas, como valores numéricos, indicadores ou estatísticas.
                    
                    Input: Pergunta ou instrução (ex.: 'Extraia os dados de exportação da tabela de 2023' ou 'Quais são os valores de PIB por setor?')
                    Output: Dados extraídos em formato textual ou tabular (ex.: 'Tabela: Exportações 2023: [valor1, valor2]')"""
                )
            )
        else:
            tools.append(
                Tool(
                    name="resposta_geral",
                    func=self._resposta_conhecimento_geral,
                    description="""Use esta ferramenta quando o sistema RAG não estiver disponível.
                    Fornece informações gerais sobre economia de São Paulo.
                    
                    Input: Pergunta do usuário
                    Output: Resposta baseada em conhecimento geral"""
                )
            )
        
        # Adicionar calculadora matemática
        tools.append(
            Tool(
                name="math_calculator",
                func=self._math_calculator,
                description="""Realiza cálculos matemáticos avançados e retorna resultados em formato LaTeX com duas casas decimais onde aplicável.
                Use esta ferramenta para resolver expressões matemáticas, estatísticas ou financeiras relacionadas à economia.
                
                Tipos de operações suportadas:
                - 'eval': Avalia expressões aritméticas (ex.: 'eval: 2 + 3')
                - 'diff': Calcula derivadas (ex.: 'diff: x**2, x')
                - 'integrate': Calcula integrais (ex.: 'integrate: x**2, x')
                - 'solve': Resolve equações (ex.: 'solve: x**2 - 4, x')
                - 'matrix_mult': Multiplica matrizes (ex.: 'matrix_mult: [[1,2],[3,4]] * [[5,6],[7,8]]')
                - 'matrix_inv': Calcula inversa de matriz (ex.: 'matrix_inv: [[4,7],[2,6]]')
                - 'mean': Calcula média (ex.: 'mean: [1,2,3]')
                - 'std': Calcula desvio padrão (ex.: 'std: [1,2,3]')
                - 'correlation': Calcula correlação (ex.: 'correlation: [1,2,3], [4,5,6]')
                - 't_test': Realiza teste t (ex.: 't_test: [1,2,3], [4,5,6]')
                - 'regression': Realiza regressão linear (ex.: 'regression: [1,2,3], [2,4,6]')
                - 'simple_interest': Calcula juros simples (ex.: 'simple_interest: 1000, 0.05, 2')
                - 'compound_interest': Calcula juros compostos (ex.: 'compound_interest: 1000, 0.05, 2, 12')
                - 'npv': Calcula Valor Presente Líquido (ex.: 'npv: 0.05, [-100, 50, 60]')
                - 'irr': Calcula Taxa Interna de Retorno (ex.: 'irr: [-100, 50, 60]')
                - 'log': Calcula logaritmo (ex.: 'log: 100, 10')
                - 'probability_normal': Calcula probabilidade normal (ex.: 'probability_normal: 0, 1, 1.96')
                - 'plot': Descreve gráfico (ex.: 'plot: [1,2,3], [4,5,6]')
                
                Input: Formato 'tipo: argumentos' (ex.: 'mean: [1,2,3]' ou 'diff: x**2, x')
                Output: Resultado em LaTeX ou texto descritivo"""
            )
        )
        
        return tools
    
    def _extract_table_data(self, query: str) -> str:
        """Extrai dados de tabelas da base de conhecimento."""
        try:
            if not self.rag_available:
                return f"❌ Sistema RAG não disponível. Status: {self.rag_status}"

            logger.info(f"Extração de tabela para: {query}")
            resultado = self.rag.query_rag_system(query)

            if 'error' in resultado:
                return f"⚠️ Erro no sistema: {resultado['error']}"

            documents = resultado.get('retrieved_documents', [])
            if not documents:
                return "⚠️ Nenhum documento relevante encontrado."

            table_data = []
            for doc in documents[:3]:  # Limita a 3 documentos para brevidade
                if isinstance(doc, dict) and 'content' in doc:
                    content = doc['content']
                    if 'table' in content.lower() or 'tabela' in content.lower():
                        # Extração básica: procurar linhas que pareçam tabelas
                        lines = content.split('\n')
                        table_lines = [line for line in lines if '|' in line or '-' in line]
                        if table_lines:
                            table_data.append("\n".join(table_lines))
                        elif 'DESCRIÇÃO VISUAL' in content:
                            # Extrair descrições de tabelas da análise visual
                            visual_section = content.split('DESCRIÇÃO VISUAL')[-1]
                            table_data.append(visual_section.strip())

            if not table_data:
                return "⚠️ Não foi possível extrair dados de tabela dos documentos."

            return f"📊 Dados de tabela extraídos:\n{'\n---\n'.join(table_data)}"

        except Exception as e:
            logger.error(f"Erro na extração de tabela: {e}")
            return f"❌ Erro na extração de dados de tabela: {str(e)}"
    
    def _create_simplified_prompt(self) -> PromptTemplate:
        """Cria um prompt simplificado que evita loops infinitos."""
        
        base_template = """Você é um ESPECIALISTA em economia do Estado de São Paulo.

IMPORTANTE: Para saudações simples (olá, oi, bom dia, etc.) responda diretamente SEM usar ferramentas.

Para outras perguntas sobre economia paulista, use as ferramentas disponíveis, incluindo 'extract_table_data' para perguntas sobre tabelas.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada para a ferramenta
Observation: resultado da ferramenta
Thought: análise final
Final Answer: resposta completa e estruturada

{agent_scratchpad}"""
        
        if self.rag_available:
            template = """Você é um ESPECIALISTA em economia do Estado de São Paulo, com foco específico em:
- Indústria Automotiva
- Indústria Têxtil e de Confecções  
- Indústria Farmacêutica
- Máquinas e Equipamentos
- Mapa da Indústria Paulista
- Indústria Metalúrgica
- Agropecuária e Transição Energética
- Balança Comercial Paulista
- Biocombustíveis

INSTRUÇÕES PARA RESPOSTAS DETALHADAS:

1. Use a ferramenta disponível para coletar informações abrangentes
2. Estruture suas respostas com numeração, subtópicos e formatação clara
3. Inclua dados específicos, estatísticas e exemplos sempre que disponível
4. Desenvolva cada ponto com explicações detalhadas
5. Use linguagem técnica apropriada mas acessível

FORMATO OBRIGATÓRIO para Final Answer:
- Use numeração (1., 2., 3., etc.) para pontos principais
- Use subtópicos com **negrito** para destacar aspectos importantes
- Inclua dados quantitativos quando disponível
- Desenvolva cada ponto com pelo menos 2-3 frases explicativas

EXCEÇÕES para respostas diretas (SEM usar ferramentas):
- **Saudações**: "Olá", "Oi", "Bom dia", "Boa tarde", "Boa noite", "Tudo bem?", etc.
- **Confirmações**: "Ok", "Entendi", "Certo", "Sim", "Não"
- **Perguntas sobre funcionamento**: "Como você funciona?", "O que você pode fazer?"
- **Despedidas**: "Tchau", "Até logo", "Obrigado"

Para essas exceções, responda diretamente de forma amigável.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta e estratégia
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada específica para a ferramenta
Observation: resultado da ferramenta
Thought: análise final de todas as informações
Final Answer: resposta DETALHADA, ESTRUTURADA e COMPLETA

Exemplo de uso para extract_table_data:
Pergunta: "Quais são os valores de exportação em 2023?"
Thought: Preciso extrair dados de uma tabela sobre exportações.
Action: extract_table_data
Action Input: Extraia os dados de exportação da tabela de 2023
Observation: 📊 Dados de tabela extraídos:
Exportações 2023: | Produto | Valor |
|--------|-------|
| Soja   | 500M  |
| Açúcar | 300M  |
Thought: Os dados mostram valores de exportação para 2023.
Final Answer: 1. **Valores de Exportação 2023**:
   - **Soja**: 500 milhões. Este valor reflete a força do setor agrícola.
   - **Açúcar**: 300 milhões. A produção de açúcar é um componente chave da balança comercial.

Exemplo de uso expandido para math_calculator:
Pergunta: "Qual é a derivada de x^2?"
Thought: Preciso calcular a derivada.
Action: math_calculator
Action Input: diff: x^{{2}}, x
Observation: \\frac{{d}}{{d{{x}}}} x^{{2}} = 2x
Final Answer: 2x

Pergunta: "Calcule o VPL com taxa 0.05 e fluxos [ -100, 50, 60 ]"
Thought: Usar npv.
Action: math_calculator
Action Input: npv: 0.05, [-100, 50, 60]
Observation: VPL = 2.38
Final Answer: 2.38

Pergunta: "Qual a média de [1,2,3]?"
Thought: Calcular média.
Action: math_calculator
Action Input: mean: [1,2,3]
Observation: Média = 2.0
Final Answer: 2.0

Pergunta: "Regressão linear com x=[1,2,3], y=[2,4,6]"
Thought: Usar regressão.
Action: math_calculator
Action Input: regression: [1,2,3], [2,4,6]
Observation: Equação: y = 2.0x + 0.0\nR² = 1.0
Final Answer: y = 2.0x + 0.0, R² = 1.0

{agent_scratchpad}"""
        else:
            template = """Você é um assistente especializado em economia do Estado de São Paulo.

⚠️ AVISO: Sistema de base de conhecimento não disponível. Respostas baseadas em conhecimento geral.

EXCEÇÕES para respostas diretas (SEM usar ferramentas):
- **Saudações**: "Olá", "Oi", "Bom dia", etc.
- **Confirmações**: "Ok", "Entendi", "Certo"
- **Despedidas**: "Tchau", "Até logo"

Para essas exceções, responda diretamente.

HISTÓRICO DA CONVERSA:
{chat_history}

Ferramentas disponíveis:
{tools}

Use o seguinte formato:

Question: {input}
Thought: análise da pergunta
Action: escolha uma ferramenta de [{tool_names}]
Action Input: entrada para a ferramenta
Observation: resultado da ferramenta
Thought: análise final
Final Answer: resposta com base no conhecimento geral disponível

{agent_scratchpad}"""
        
        return PromptTemplate.from_template(template)
    
    def _consultar_rag_direto(self, query: str) -> str:
        """
        CORREÇÃO: Consulta direta e simplificada do RAG.
        """
        try:
            if not self.rag_available:
                return f"❌ Sistema RAG não disponível. Status: {self.rag_status}"
            
            logger.info(f"Consulta RAG: {query}")
            
            resultado = self.rag.query_rag_system(query)
            
            if 'error' in resultado:
                logger.error(f"Erro no RAG: {resultado['error']}")
                return f"⚠️ Erro no sistema: {resultado['error']}"
            
            response = resultado.get("response", "")
            
            if not response or len(response.strip()) < 10:
                return "⚠️ Resposta muito curta ou vazia. Verifique se há documentos na base de dados."
            
            retrieved_docs = len(resultado.get('retrieved_documents', []))
            reranked_docs = len(resultado.get('reranked_documents', []))
            confidence = resultado.get('confidence_scores', 'N/A')
            
            metadata_info = f"\n\n📊 _Consulta baseada em {retrieved_docs} documento(s) recuperado(s)"
            if reranked_docs > 0:
                metadata_info += f", {reranked_docs} reranqueado(s)"
            if confidence != 'N/A':
                metadata_info += f" (confiança: {confidence})"
            metadata_info += "._"
            
            return response + metadata_info
            
        except AttributeError as e:
            logger.error(f"Método não encontrado no RAG: {e}")
            return f"❌ Erro: Método de consulta não encontrado no sistema RAG: {str(e)}"
        except Exception as e:
            logger.error(f"Erro na consulta RAG: {e}")
            return f"❌ Erro na consulta: {str(e)}"
    
    def _resposta_conhecimento_geral(self, query: str) -> str:
        """Resposta quando RAG não está disponível."""
        return f"""⚠️ **Sistema de base de conhecimento indisponível**

Pergunta: "{query}"

**Resposta baseada em conhecimento geral:**

São Paulo é o principal centro econômico do Brasil, responsável por cerca de 1/3 do PIB nacional. O estado se destaca em diversos setores:

**Principais Setores:**
- **Indústria Automotiva**: Concentrada no ABC paulista e região de Campinas
- **Indústria Farmacêutica**: Forte presença na região metropolitana
- **Têxtil e Confecções**: Setor tradicional do estado
- **Máquinas e Equipamentos**: Distribuído por várias regiões
- **Agropecuária**: Interior do estado, forte em cana-de-açúcar, café, laranja

**⚠️ IMPORTANTE**: Resposta baseada em conhecimento geral. Para informações precisas, consulte:
- FIESP (Federação das Indústrias do Estado de São Paulo)
- Fundação SEADE
- IBGE

Status do sistema RAG: {self.rag_status}"""
    
    def _math_calculator(self, input_str: str) -> str:
        """Resolve operações matemáticas avançadas e retorna em formato LaTeX com duas casas decimais onde aplicável."""
        try:
            if ':' in input_str:
                tipo, args = input_str.split(':', 1)
                tipo = tipo.strip().lower()
                args = args.strip()
            else:
                tipo = 'eval'
                args = input_str.strip()

            x, y, t = symbols('x y t')

            if tipo == 'eval':
                result = sympify(args)
                rounded = round(float(result), 2)
                return latex(result) + f" = {rounded}"
            elif tipo == 'diff':
                expr, var = args.split(',')
                expr_sym = sympify(expr.strip())
                var_sym = symbols(var.strip())
                deriv = diff(expr_sym, var_sym)
                return f"\\frac{{d}}{{d{var.strip()}}} \\left( {latex(expr_sym)} \\right) = {latex(deriv)}"
            elif tipo == 'integrate':
                expr, var = args.split(',')
                expr_sym = sympify(expr.strip())
                var_sym = symbols(var.strip())
                integ = integrate(expr_sym, var_sym)
                return f"\\int {latex(expr_sym)} \\, d{var.strip()} = {latex(integ)} + C"
            elif tipo == 'solve':
                eq, var = args.split(',')
                eq_sym = sympify(eq.strip())
                var_sym = symbols(var.strip())
                solutions = solve(eq_sym, var_sym)
                return f"Soluções: {latex(solutions)}"
            elif tipo == 'matrix_mult':
                mat1_str, mat2_str = args.split('*')
                mat1 = Matrix(eval(mat1_str.strip()))
                mat2 = Matrix(eval(mat2_str.strip()))
                result = mat1 * mat2
                return latex(result)
            elif tipo == 'matrix_inv':
                mat_str = args
                mat = Matrix(eval(mat_str.strip()))
                inv = mat.inv()
                return latex(inv)
            elif tipo == 'mean':
                data = eval(args.strip())
                mean_val = statistics.mean(data)
                return f"Média = {round(mean_val, 2)}"
            elif tipo == 'std':
                data = eval(args.strip())
                std_val = statistics.stdev(data)
                return f"Desvio padrão = {round(std_val, 2)}"
            elif tipo == 'correlation':
                data1, data2 = eval(args.strip())
                corr = stats.pearsonr(data1, data2)[0]
                return f"Correlação = {round(corr, 2)}"
            elif tipo == 't_test':
                sample1, sample2 = eval(args.strip())
                t_stat, p_val = stats.ttest_ind(sample1, sample2)
                return f"Estatística t = {round(t_stat, 2)}, p-valor = {round(p_val, 4)}"
            elif tipo == 'regression':
                x_data, y_data = eval(args.strip())
                slope, intercept, r_value, p_value, std_err = stats.linregress(x_data, y_data)
                return f"Equação: y = {round(slope, 2)}x + {round(intercept, 2)}\nR² = {round(r_value**2, 2)}"
            elif tipo == 'simple_interest':
                p, r, t = map(float, args.split(','))
                interest = p * r * t
                return f"Juros simples = {round(interest, 2)}"
            elif tipo == 'compound_interest':
                p, r, t, n = map(float, args.split(','))
                amount = p * (1 + r/n)**(n*t)
                return f"Montante = {round(amount, 2)}"
            elif tipo == 'npv':
                rate, cashflows_str = args.split(',', 1)
                rate = float(rate.strip())
                cashflows = eval(cashflows_str.strip())
                npv_val = np.npv(rate, cashflows)
                return f"VPL = {round(npv_val, 2)}"
            elif tipo == 'irr':
                cashflows = eval(args.strip())
                irr_val = np.irr(cashflows)
                return f"TIR = {round(irr_val * 100, 2)}%"
            elif tipo == 'log':
                value, base = map(float, args.split(',')) if ',' in args else (float(args), np.e)
                log_val = np.log(value) / np.log(base) if base != np.e else sym_log(value)
                return f"\\log_{{{base}}}({value}) = {round(float(log_val), 2)}"
            elif tipo == 'probability_normal':
                mean, std, value = map(float, args.split(','))
                cdf = stats.norm.cdf(value, mean, std)
                return f"Probabilidade cumulativa = {round(cdf, 4)}"
            elif tipo == 'plot':
                x_data, y_data = eval(args.strip())
                fig, ax = plt.subplots()
                ax.plot(x_data, y_data)
                buf = StringIO()
                fig.savefig(buf, format='svg')
                plt.close()
                return f"Gráfico de linha: x={x_data}, y={y_data}. Pico em {max(y_data)}."
            else:
                return "❌ Tipo de operação não suportado. Use: eval, diff, integrate, solve, etc."

        except Exception as e:
            return f"❌ Erro ao calcular '{input_str}': {str(e)}. Verifique a sintaxe."

    def _is_simple_greeting(self, text: str) -> bool:
        """Verifica se é uma saudação simples que não precisa de ferramentas."""
        greetings = [
            "olá", "oi", "oiê", "ola", "bom dia", "boa tarde", "boa noite",
            "como vai", "tudo bem", "e aí", "salve", "alô", "hello", "hi"
        ]
        text_lower = text.lower().strip()
        return any(greeting in text_lower for greeting in greetings) and len(text_lower) < 20
    
    def consultar(self, pergunta: str) -> str:
        """
        CORREÇÃO PRINCIPAL: Consulta simplificada que evita loops e retorna resposta formatada para React.
        """
        if not pergunta.strip():
            return "Por favor, forneça uma pergunta válida."

        try:
            logger.info(f"Processando pergunta: {pergunta}")

            if self._is_simple_greeting(pergunta):
                resposta = """👋 **Olá! Seja bem-vindo!**

Sou um assistente especializado em economia do Estado de São Paulo. Posso ajudá-lo com informações sobre:

🏭 **Setores Industriais:**
- Indústria Automotiva
- Indústria Têxtil e Confecções
- Indústria Farmacêutica
- Máquinas e Equipamentos
- Indústria Metalúrgica

📊 **Dados Econômicos:**
- Balança Comercial Paulista
- Mapa da Indústria Paulista
- Agropecuária e Transição Energética
- Biocombustíveis

💬 **Como posso ajudar?**
Faça sua pergunta sobre qualquer aspecto da economia paulista!"""
                self._add_to_memory(pergunta, resposta)
                return resposta

            chat_history = self._format_chat_history_for_prompt()
            input_with_history = {
                "input": pergunta,
                "chat_history": chat_history
            }

            resultado = self.agent_executor.invoke(
                input_with_history,
                config={"max_execution_time": 45}
            )

            resposta = resultado.get("output", "Não foi possível obter uma resposta.")

            if "Agent stopped due to iteration limit" in resposta:
                if self.rag_available:
                    logger.warning("Fallback: usando consulta RAG direta")
                    resposta = self._consultar_rag_direto(pergunta)
                else:
                    logger.warning("Fallback: usando conhecimento geral")
                    resposta = self._resposta_conhecimento_geral(pergunta)

            self._add_to_memory(pergunta, resposta)
            return resposta

        except Exception as e:
            logger.error(f"Erro ao consultar agente: {e}")
            if self.rag_available:
                try:
                    logger.info("Tentando fallback com RAG direto")
                    resposta = self._consultar_rag_direto(pergunta)
                    self._add_to_memory(pergunta, resposta)
                    return resposta
                except:
                    pass

            resposta_erro = f"""❌ **Erro no processamento**

Ocorreu um erro ao processar sua pergunta: {str(e)}

**Possíveis soluções:**
1. Tente reformular a pergunta
2. Verifique se é uma pergunta sobre economia de São Paulo
3. Se o problema persistir, reinicie o sistema

Status do RAG: {self.rag_status}"""
            self._add_to_memory(pergunta, resposta_erro)
            return resposta_erro
    
    def get_system_info(self) -> Dict[str, Any]:
        """Retorna informações sobre o status do sistema."""
        info = {
            "rag_available": self.rag_available,
            "rag_status": self.rag_status,
            "tools_count": len(self.tools),
            "agent_ready": hasattr(self, 'agent_executor'),
            "max_iterations": 3,
            "max_execution_time": 60,
            "memory_system": "LangGraph MemorySaver",
            "messages_count": len(self._get_chat_history()),
            "chromadb_available": CHROMADB_AVAILABLE
        }
        
        if self.rag_available and hasattr(self, 'rag'):
            try:
                rag_status = self.rag.get_system_info()
                info.update({
                    "rag_detailed_status": rag_status,
                    "reranking_enabled": rag_status.get('reranking_enabled', False),
                    "llm_model": rag_status.get('llm_model', 'unknown')
                })
            except Exception as e:
                info["rag_error"] = str(e)
        
        return info
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método simplificado para compatibilidade genérica, mas não essencial para React.
        Pode ser usado para testes ou integração futura.
        """
        question = inputs.get("question", "")
        if not question:
            return {"response": "Por favor, forneça uma pergunta válida."}
        response = self.consultar(question)
        return {"response": response}
    
    def _format_chat_history_for_prompt(self) -> str:
        """Formata o histórico de conversa para o prompt."""
        history = self._get_chat_history()
        formatted_history = []
        for msg in history:
            if msg["role"] == "user":
                formatted_history.append(f"Usuário: {msg['content']}")
            else:
                formatted_history.append(f"Assistente: {msg['content']}")
        return "\n".join(formatted_history)

    def _get_chat_history(self) -> List[Dict[str, str]]:
        """Obtém o histórico de conversa."""
        return self.conversation_state.get("messages", [])

    def _add_to_memory(self, user_input: str, response: str):
        """Adiciona uma interação ao histórico de memória."""
        self.conversation_state["messages"].append({"role": "user", "content": user_input})
        self.conversation_state["messages"].append({"role": "assistant", "content": response})
        self.conversation_state["last_user_message"] = user_input
        self.conversation_state["last_ai_message"] = response

    def clear_memory(self):
        """Limpa a memória da conversação."""
        try:
            self.conversation_state = {
                "messages": [],
                "last_user_message": "",
                "last_ai_message": ""
            }
            logger.info("Memória limpa com sucesso")
        except Exception as e:
            logger.error(f"Erro ao limpar memória: {e}")
    
    def run_interactive(self):
        """Executa o loop interativo."""
        print("=== Agente RAG Corrigido - Sistema de Consulta ===")
        print("Especialista em economia do Estado de São Paulo")
        print("Agora com LangGraph Memory System")
        
        system_info = self.get_system_info()
        print(f"\n📊 **Status do Sistema:**")
        print(f"RAG disponível: {'✅ Sim' if system_info['rag_available'] else '❌ Não'}")
        print(f"Status: {system_info['rag_status']}")
        print(f"ChromaDB disponível: {'✅ Sim' if system_info['chromadb_available'] else '❌ Não'}")
        print(f"Máx iterações: {system_info['max_iterations']}")
        print(f"Timeout: {system_info['max_execution_time']}s")
        print(f"Sistema de memória: {system_info['memory_system']}")
        print(f"Mensagens na memória: {system_info['messages_count']}")
        
        if system_info.get('rag_detailed_status'):
            rag_details = system_info['rag_detailed_status']
            print(f"Reranking habilitado: {'✅ Sim' if rag_details.get('reranking_enabled') else '❌ Não'}")
            print(f"Modelo LLM: {rag_details.get('llm_model', 'N/A')}")
        
        print(f"\nDigite 'sair' para encerrar, 'limpar' para limpar histórico, 'status' para ver informações\n")
        
        while True:
            try:
                user_input = input("> ").strip()
                
                if user_input.lower() in ["sair", "exit", "quit"]:
                    print("Encerrando. Até logo!")
                    break
                
                if user_input.lower() in ["limpar", "clear"]:
                    self.clear_memory()
                    print("🧹 Histórico limpo!")
                    continue
                
                if user_input.lower() in ["status", "info"]:
                    info = self.get_system_info()
                    print("\n📊 **Status Atual:**")
                    for key, value in info.items():
                        if key != 'rag_detailed_status':
                            print(f"{key}: {value}")
                    print()
                    continue
                
                if not user_input:
                    continue
                
                print(f"\n🔍 Processando...")
                resposta = self.consultar(user_input)
                
                print(f"\n{'='*60}")
                print("📊 RESPOSTA:")
                print(f"{'='*60}")
                print(f"{resposta}")
                print(f"{'='*60}\n")
                
            except KeyboardInterrupt:
                print("\nEncerrando. Até logo!")
                break
            except Exception as e:
                logger.error(f"Erro no loop: {e}")
                print(f"Erro: {e}\n")

def create_rag_agent():
    """
    CORREÇÃO: Função para criar o agente RAG corrigido.
    """
    try:
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        
        print("Inicializando agente RAG com LangGraph...")
        agent = RAGAgentReact()
        
        system_info = agent.get_system_info()
        if system_info['rag_available']:
            print("✅ Agente RAG completo inicializado!")
        else:
            print(f"⚠️ Agente em modo limitado - Status: {system_info['rag_status']}")
        
        return agent
        
    except Exception as e:
        print(f"❌ Erro ao inicializar: {e}")
        raise

if __name__ == "__main__":
    try:
        os.environ["ANONYMIZED_TELEMETRY"] = "False"
        agent = RAGAgentReact()
        agent.run_interactive()
        
    except ValueError as e:
        print(f"Erro de configuração: {e}")
    except Exception as e:
        print(f"Erro: {e}")