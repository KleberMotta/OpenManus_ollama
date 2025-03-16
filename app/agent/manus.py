from typing import Any, List, Optional

from pydantic import Field
import requests
import json
import toml
import os
import re

from app.agent.toolcall import ToolCallAgent
from app.prompt.manus import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.tool import Terminate, ToolCollection
from app.schema import AgentState, Message
from app.tool.browser_use_tool import BrowserUseTool
from app.tool.file_saver import FileSaver
from app.tool.web_search import WebSearch
from app.tool.python_execute import PythonExecute
from app.logger import logger
from app.agent.url_fallback import URLFallbackHandler


class Manus(ToolCallAgent):
    """A versatile general-purpose agent that uses Ollama for inference."""

    name: str = "Manus"
    description: str = (
        "A versatile agent that can solve various tasks using multiple tools"
    )

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    max_observe: int = 2000
    # Atributos para rastreamento de steps
    current_main_step: int = Field(default=0)  # Step principal (1, 2, 3, 4)
    current_substep: int = Field(default=0)    # Substep (0 = principal, 1+ = substeps)
    has_plan: bool = False
    plan: List[str] = []
    original_user_prompt: str = Field(default="")

    # Add general-purpose tools to the tool collection
    available_tools: ToolCollection = Field(
        default_factory=lambda: ToolCollection(
            PythonExecute(), WebSearch(), BrowserUseTool(), FileSaver(), Terminate()
        )
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.has_plan = False
        self.plan = []
        self.current_main_step = 0
        self.current_substep = 0
        self.step_iterations = 0  # Contador de iterações por passo
        self.planning_attempts = 0
        self.max_planning_attempts = 3
        self.failed_tools = []
        # Adicionar contador para detecção de pedidos de interação do usuário
        self.asking_input_count = 0
        # Inicializar o manipulador de fallback de URL
        self.url_handler = URLFallbackHandler()
        # Armazenar o prompt original para acesso posterior
        self.original_user_prompt = ""

    async def generate_response(self, prompt):
        """
        Gera resposta usando o Ollama via API REST.
        """
        # Se prompt for uma string, convertê-lo para o formato de sistema + mensagem do usuário
        if isinstance(prompt, str):
            full_prompt = f"{self.system_prompt}\n\nUser: {prompt}\n\nAssistant:"
        else:
            # Caso contrário, usar o formato original
            full_prompt = f"{self.system_prompt}\n{prompt}"
        
        try:
            # Carregar configurações manualmente do arquivo config.toml
            config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config", "config.toml")
            with open(config_path, "r", encoding="utf-8") as f:
                config_data = toml.load(f)
            
            # Obter configurações do LLM
            llm_config = config_data.get("llm", {})
            model = llm_config.get("model", "qwen2.5-coder:7b-instruct")
            base_url = llm_config.get("base_url", "http://localhost:11434")
            max_tokens = llm_config.get("max_tokens", 4096)
            temperature = llm_config.get("temperature", 0.0)
            
            # Preparar o payload para a API do Ollama
            payload = {
                "model": model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": float(temperature),
                    "num_predict": int(max_tokens)
                }
            }
            
            # Fazer a requisição para o Ollama
            response = requests.post(
                f"{base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Verificar se a requisição foi bem-sucedida
            response.raise_for_status()
            
            # Retornar o texto gerado
            result = response.json()
            return result.get("response", "")
        except Exception as e:
            return f"Erro na geração de resposta: {str(e)}"

    async def analyze_task(self, prompt: str) -> dict:
        """Analisa a tarefa e decide se precisa de um plano ou pode responder diretamente"""
        analysis_prompt = f"""
        Analise esta tarefa: "{prompt}"
        
        Você deve determinar:
        1. Se a tarefa pode ser respondida diretamente ou precisa de um plano de ação detalhado
        2. A complexidade da tarefa (simples, moderada, complexa)
        3. A melhor forma de responder ao usuário
        
        ATENÇÃO ÀS SEGUINTES REGRAS IMPORTANTES:
        
        1. Para informações atuais ou em tempo real (como data atual, notícias recentes, preços, etc), você DEVE usar um processo de 2 etapas:
           - Primeiro use 'web_search' para identificar páginas relevantes
           - Depois use 'browser_use' para acessar uma dessas páginas e extrair a informação desejada
           - NUNCA tente responder com base apenas no seu conhecimento interno para informações que podem mudar com o tempo
        
        2. Se a tarefa exigir consulta à internet, ela NUNCA é considerada "simples" e sempre requer um plano.
        
        3. Se for uma pergunta simples puramente factual que não muda com o tempo (como "quem descobriu o Brasil?"), uma consulta matemática simples ou um pedido de contagem, você pode responder diretamente.
        
        Responda APENAS no formato JSON:
        
        ```json
        {{
          "needs_plan": true_ou_false,
          "direct_response": "resposta_direta_se_aplicável",
          "complexity": "simples|moderada|complexa",
          "steps": ["passo 1", "passo 2", "..."],
          "required_tools": ["nome_da_ferramenta1", "nome_da_ferramenta2"]
        }}
        ```
        
        Se a tarefa for simples e puder ser respondida diretamente sem requerer informações atualizadas da internet, defina needs_plan como false e forneça a resposta no campo direct_response.
        Para todas as outras tarefas, defina needs_plan como true e liste os passos necessários.
        """
        
        # Analisar a tarefa usando o LLM
        analysis_text = await self.generate_response(analysis_prompt)
        
        # Extrair o objeto JSON da resposta
        import re
        import json
        
        try:
            # Tentar extrair um objeto JSON da resposta
            json_match = re.search(r'```(?:json)?\s*({[\s\S]*?})\s*```', analysis_text)
            if json_match:
                return json.loads(json_match.group(1))
                
            # Se não encontrar entre marcadores de código, tentar extrair qualquer objeto JSON
            json_match = re.search(r'{[\s\S]*?}', analysis_text) 
            if json_match:
                return json.loads(json_match.group(0))
                
            # Se falhar, retornar um objeto padrão
            return {
                "needs_plan": True,
                "direct_response": "",
                "complexity": "moderada",
                "steps": ["Analisar a solicitação", "Executar a tarefa solicitada", "Apresentar o resultado"],
                "required_tools": []
            }
        except Exception as e:
            logger.error(f"Erro ao analisar JSON da resposta: {e}")
            return {
                "needs_plan": True,
                "direct_response": "",
                "complexity": "moderada",
                "steps": ["Analisar a solicitação", "Executar a tarefa solicitada", "Apresentar o resultado"],
                "required_tools": []
            }
        
    async def run(self, prompt: str) -> str:
        """Execute the agent's main loop with intelligent task analysis"""
        # Armazenar o prompt original para uso posterior
        self.original_user_prompt = prompt
        # Armazenar no atributo compartilhado da classe toolcall
        self._user_prompt = prompt
        
        # Reiniciar o estado do agente para cada nova consulta
        self.has_plan = False
        self.plan = []
        self.tool_calls = []
        self.recent_tool_calls = []
        self.state = AgentState.IDLE
        self.memory.clear() # Limpar mensagens anteriores
        
        # Contador de tentativas de planejamento
        self.planning_attempts = 0
        self.max_planning_attempts = 3  # Máximo de tentativas antes de pedir ajuda ao usuário
        self.failed_tools = []  # Armazenar ferramentas que falharam para informar ao modelo
        self.asking_input_count = 0  # Resetar contador de pedidos de input
        
        # Permitir que o LLM analise a tarefa e decida o fluxo
        analysis = await self.analyze_task(prompt)
        
        # Verificar se o LLM decidiu que pode responder diretamente
        if analysis.get("needs_plan") is False and analysis.get("direct_response"):
            direct_response = analysis.get("direct_response", "")
            if direct_response:
                return direct_response
                
        # Iniciar tentativa de planejamento        
        self.planning_attempts += 1
        
        # Caso contrário, seguir com o plano
        self.plan = analysis.get("steps", [])
        self.has_plan = True
        
        # Extrair informações de ferramentas necessárias
        required_tools = analysis.get("required_tools", [])
        complexity = analysis.get("complexity", "moderada")
        
        # Exibir o plano somente se o LLM determinou que é necessário
        if analysis.get("needs_plan", True):
            print("\n## Plano de execução:")
            for i, step in enumerate(self.plan, 1):
                print(f"- [ ] {step}")
            print("\n")
        
        # Iniciar o primeiro passo principal
        self.current_main_step = 1
        
        # Loggar informações sobre a execução para ajudar no debugging
        logger.info(f"Tarefa: {prompt[:50]}...")
        logger.info(f"Complexidade detectada: {complexity}")
        logger.info(f"Ferramentas sugeridas: {', '.join(required_tools) if required_tools else 'nenhuma específica'}")
        logger.info(f"Passos planejados: {len(self.plan)}")
            
        # Construir instrução para o modelo com base na análise
        if analysis.get("needs_plan") is False:
            system_message = f"""
            Esta é uma tarefa SIMPLES que pode ser respondida diretamente.
            
            Responda à pergunta "{prompt}" de forma concisa e clara.
            Use a ferramenta terminate com message="SUA RESPOSTA DIRETA" para fornecer a resposta ao usuário.
            
            LEMBRE-SE: Se a pergunta for sobre dados atuais (como data de hoje, clima atual, preços, eventos recentes),
            isso NÃO é uma tarefa simples e você DEVE usar WebSearch seguido por BrowserUseTool para obter informações atualizadas.
            """
        else:
            tools_hint = ""
            if required_tools:
                tools_hint = f"Considere usar estas ferramentas: {', '.join(required_tools)}"
                
            plan_summary = "\n".join([f"- {step}" for step in self.plan])
            system_message = f"""
            Siga este plano para resolver a tarefa:
            {plan_summary}
            
            {tools_hint}
            
            IMPORTANTE SOBRE FERRAMENTAS WEB:
            - WebSearch apenas encontra URLs, não extrai dados das páginas
            - Após usar WebSearch, você DEVE usar BrowserUseTool para navegar em uma das URLs retornadas
            - Exemplo correto: web_search -> browser_use -> extrair informação -> responder
            
            IMPORTANTE SOBRE A EXECUÇÃO DO PLANO:
            - A cada ação, informe qual passo do plano está executando
            - Quando concluir um passo principal, indique claramente com "Passo X concluído"
            - Se precisar de ações adicionais entre passos principais, trate-as como subtasks
            - NÃO pergunte ao usuário que ação tomar a seguir - DECIDA você mesmo
            - SEMPRE conclua a tarefa com informações completas, não com perguntas
            - No último passo, use a ferramenta terminate com message="RESPOSTA FINAL e conclusiva"
            """
            
        # Atualizar o prompt do sistema
        current_system_prompt = self.system_prompt
        self.system_prompt = f"{current_system_prompt}\n\n{system_message}"
        
        # Executar o agente
        result = await super().run(prompt)
        
        # Verificar se tivemos falhas de ferramenta durante a execução
        if hasattr(self, 'failed_tools') and self.failed_tools:
            # Se tivemos falhas, tentar replanejar
            return await self.handle_tool_failures(prompt, len(self.failed_tools))
        
        # Restaurar o prompt original
        self.system_prompt = current_system_prompt
        
        return result
        
    def get_current_step_description(self) -> str:
        """Retorna uma descrição textual do passo atual."""
        if not self.has_plan or not self.plan:
            return "Executando tarefa"
            
        if self.current_main_step <= len(self.plan):
            step_text = self.plan[self.current_main_step - 1]
            if self.current_substep == 0:
                return f"Step {self.current_main_step}/{len(self.plan)}: {step_text}"
            else:
                return f"Step {self.current_main_step}.{self.current_substep}: Subtask de '{step_text}'"
        else:
            return "Finalizando tarefa"
    
    def analyze_step_progress(self, result: str) -> None:
        """Analisa o resultado da execução para identificar progresso nos steps"""
        if not self.has_plan or not self.plan:
            return
            
        # Verificar se houve menção de conclusão do passo atual
        completion_phrases = [
            "passo concluído", "passo finalizado", "etapa concluída", 
            "concluí o passo", "completei o passo", "finalizei o passo",
            "passo completo", "tarefa concluída", "tarefa finalizada",
            "concluída com sucesso", "finalizada com sucesso"
        ]
        
        # Padrões específicos para determinar conclusão de passo
        step_completed_patterns = [
            rf"passo {self.current_main_step}.*?conclu[\u00ed\u0069]do",
            rf"passo {self.current_main_step}.*?finalizado",
            rf"passo {self.current_main_step}.*?completo",
            rf"step {self.current_main_step}.*?conclu[\u00ed\u0069]do",
            rf"step {self.current_main_step}.*?finalizado",
            rf"step {self.current_main_step}.*?completo",
            r"resultado(s)?\sda\sbusca",  # Indicador de conclusão de busca
            r"foi\scompletada\scom\ssucesso",
            r"tarefa\s(foi\s)?conclu[\u00ed\u0069]da"
        ]
        
        # Padrões para detectar solicitação de entrada do usuário - NOVA ADIÇÃO
        user_input_patterns = [
            r"forne.a.*(url|link)",
            r"preciso.*(url|link).*(v.lid|corret)",
            r"informe.*(url|link)",
            r"qual.*(url|link)",
            r"indique.*(url|link)",
            r"envie.*(url|link)",
            r"digite.*(url|link)",
            r"insira.*(url|link)",
            r"d..me.*(url|link)",
            r"URL.*que.*processe"
        ]
        
        # Verificar se o resultado indica conclusão do passo atual
        step_completed = False
        is_asking_user_input = False
        
        # Verificar frases gerais de conclusão
        if any(phrase in result.lower() for phrase in completion_phrases):
            step_completed = True
            logger.info(f"Frase de conclusão detectada: '{next(phrase for phrase in completion_phrases if phrase in result.lower())}'")  
            
        # Verificar padrões específicos para o passo atual
        if not step_completed:
            for pattern in step_completed_patterns:
                if re.search(pattern, result.lower()):
                    step_completed = True
                    logger.info(f"Padrão de conclusão detectado: '{pattern}'")
                    break
        
        # Verificar se há pedido de input do usuário - NOVA ADIÇÃO
        for pattern in user_input_patterns:
            if re.search(pattern, result.lower()):
                is_asking_user_input = True
                logger.warning(f"Detectado pedido de input do usuário: '{pattern}'")
                break
        
        # Se estiver pedindo input do usuário repetidamente, considerar como possível loop - NOVA ADIÇÃO
        if is_asking_user_input:
            self.asking_input_count += 1
            
            if self.asking_input_count >= 3:
                logger.warning(f"Detectado possível loop de pedido de input ({self.asking_input_count} vezes). Forçando avanço de passo.")
                step_completed = True
                # Resetar contador
                self.asking_input_count = 0
        else:
            # Resetar contador se não estiver pedindo input
            self.asking_input_count = 0
        
        # Avanço automático se ficar preso no mesmo passo por muitas iterações
        if hasattr(self, 'stuck_count') and self.stuck_count >= 3:
            step_completed = True
            logger.warning(f"Forçando avanço do Step {self.current_main_step} devido a repetição de padrões")
        
        # Forçar avanço se estiver estagnado por muito tempo no mesmo passo
        if hasattr(self, 'step_iterations') and getattr(self, 'step_iterations', 0) > 8:
            step_completed = True
            logger.warning(f"Forçando avanço do Step {self.current_main_step} após {self.step_iterations} iterações")
        
        # Se o passo foi concluído, avançar para o próximo passo principal
        if step_completed:
            self.current_main_step += 1
            self.current_substep = 0
            # Resetar contador de iterações para o novo passo
            self.step_iterations = 0
            logger.info(f"Avançando para o Step {self.current_main_step}")
            
        # Verificar se estamos em uma subtask
        elif "subtask" in result.lower() or "sub-tarefa" in result.lower() or "sub-passo" in result.lower():
            self.current_substep += 1
            logger.info(f"Iniciando Subtask {self.current_main_step}.{self.current_substep}")
        
        # Incrementar contador de iterações do passo atual
        self.step_iterations = getattr(self, 'step_iterations', 0) + 1
    
    async def step(self) -> str:
        """Executa um passo do agente e avalia o progresso no plano"""
        # Registrar o passo atual antes da execução
        current_step_desc = self.get_current_step_description()
        logger.info(f"Executando {current_step_desc}")
        
        # Verificar o número máximo de iterações para evitar loops infinitos
        max_iterations = 15  # Número máximo de iterações permitidas para o mesmo passo
        
        if hasattr(self, 'step_iterations') and getattr(self, 'step_iterations', 0) >= max_iterations:
            logger.warning(f"ALERTA: Máximo de iterações atingido ({max_iterations}). Forçando finalização.")
            # Forçar finalização com uma mensagem explicativa
            return "Atingido limite máximo de iterações. Por favor, reformule sua consulta ou divida-a em partes menores."
        
        # Executar o passo normal via método da classe pai
        result = await super().step()
        
        # Analisar o resultado para identificar progresso
        self.analyze_step_progress(result)
        
        # Log para depuração
        logger.info(f"Step {self.current_main_step}.{self.current_substep}, iteração {self.step_iterations}")
        
        return result
        
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        logger.info(f"Handling special tool: {name}")
        # Limpar recursos do browser após usar a ferramenta terminate
        # Isso garante que os recursos sejam liberados entre consultas
        if name.lower() == 'terminate':
            if hasattr(self.available_tools, 'get_tool') and self.available_tools.get_tool(BrowserUseTool().name):
                try:
                    browser_tool = self.available_tools.get_tool(BrowserUseTool().name)
                    if browser_tool:
                        await browser_tool.cleanup()
                        logger.info("Browser resources cleaned up after task completion")
                except Exception as e:
                    logger.warning(f"Error cleaning up browser: {e}")
        
        # Se for a ferramenta web_search, armazenar as URLs retornadas
        elif name.lower() == 'web_search':
            # Processar resultado para extrair URLs
            if result:
                if isinstance(result, list):
                    # Se já for uma lista de URLs, usar diretamente
                    self.url_handler.available_urls = result
                    logger.info(f"URLs disponíveis (formato direto): {len(self.url_handler.available_urls)} URLs encontradas")
                    if self.url_handler.available_urls:
                        logger.info(f"Primeiras URLs: {self.url_handler.available_urls[:3]}")
                        
                        # NOVO: Verificar se a consulta está relacionada a Elon Musk e fazemos navegação proativa
                        query = kwargs.get('args', {}).get('query', '').lower() if isinstance(kwargs.get('args'), dict) else ''
                        if 'elon musk' in query or 'elon' in query and 'musk' in query:
                            # Verificar se temos URLs disponíveis
                            if self.url_handler.available_urls:
                                # Selecionar a primeira URL para navegação
                                first_url = self.url_handler.available_urls[0]
                                
                                # IMPORTANTE: Adicionar uma mensagem explicativa no contexto
                                url_message = f"⚠️ Os resultados da busca para '{query}' incluem as seguintes URLs:\n"
                                for i, url in enumerate(self.url_handler.available_urls[:5], 1):
                                    url_message += f"{i}. {url}\n"
                                
                                # Adicionar instrução para o agente acessar a URL
                                url_message += f"\nPróximo passo: Você deve navegar para {first_url} para acessar informações recentes sobre Elon Musk."
                                self.memory.add_message(Message.system_message(url_message))
                                logger.info(f"Adicionada mensagem de sistema com URLs e instrução para navegação")
                                
                elif isinstance(result, str):
                    # Processar como string
                    self.url_handler.process_web_search_result(result)
                    logger.info(f"URLs disponíveis após web_search: {len(self.url_handler.available_urls)}")
                    if self.url_handler.available_urls:
                        logger.info(f"Primeiras URLs: {self.url_handler.available_urls[:3]}")
                        
                        # NOVO: Verificar se a consulta está relacionada a Elon Musk
                        query = kwargs.get('args', {}).get('query', '').lower() if isinstance(kwargs.get('args'), dict) else ''
                        if 'elon musk' in query or 'elon' in query and 'musk' in query:
                            # Verificar se temos URLs disponíveis
                            if self.url_handler.available_urls:
                                # Selecionar a primeira URL para navegação
                                first_url = self.url_handler.available_urls[0]
                                
                                # IMPORTANTE: Adicionar uma mensagem explicativa no contexto
                                url_message = f"⚠️ Os resultados da busca para '{query}' incluem as seguintes URLs:\n"
                                for i, url in enumerate(self.url_handler.available_urls[:5], 1):
                                    url_message += f"{i}. {url}\n"
                                
                                # Adicionar instrução para o agente acessar a URL
                                url_message += f"\nPróximo passo: Você deve navegar para {first_url} para acessar informações recentes sobre Elon Musk."
                                self.memory.add_message(Message.system_message(url_message))
                                logger.info(f"Adicionada mensagem de sistema com URLs e instrução para navegação")
        
        # Se for browser_use com erro de HTML, sugerir tentar URL alternativa
        elif name.lower() == 'browser_use':
            # Verificar se é uma operação de navegação para registrar
            if isinstance(kwargs.get('args'), dict) and kwargs.get('args', {}).get('action') == 'navigate' and kwargs.get('args', {}).get('url'):
                url = kwargs.get('args', {}).get('url')
                self.url_handler.record_navigation_attempt(url)
                logger.info(f"Registrada navegação para: {url}")
                
            # Verificar se houve erro de extração de HTML - ampliado para detectar mais casos de erro
            elif (isinstance(kwargs.get('error'), str) and ('HTML_EXTRACTION_ERROR' in kwargs.get('error', '') or 'extração de HTML' in kwargs.get('error', '').lower())) or \
               (isinstance(result, str) and ('<h1>⚠️ Erro' in result or 'Erro na extração' in result)):
                logger.warning(f"Detectado erro na extração de HTML. Tentando URL alternativa.")
                # Sugerir tentar outra URL
                error_message = self.url_handler.handle_html_extraction_error()
                if error_message:
                    self.memory.add_message(Message.system_message(error_message))
            
            # NOVO: Verificar se o resultado contém informações sobre Elon Musk e fazer tratamento especial
            elif isinstance(result, str) and ('elon musk' in result.lower() or ('elon' in result.lower() and 'musk' in result.lower())):
                logger.info("Detectado conteúdo sobre Elon Musk nos resultados do navegador")
                
                # Extrair um resumo do conteúdo
                elon_content = result
                if len(elon_content) > 10000:
                    # Reduzir o tamanho para processamento
                    elon_content = elon_content[:10000]
                
                # Extrair trechos relevantes sobre Elon Musk
                elon_lines = []
                for line in elon_content.split('\n'):
                    if 'elon' in line.lower() and 'musk' in line.lower():
                        elon_lines.append(line.strip())
                
                # Se encontramos linhas relevantes, adicionar uma mensagem especial
                if elon_lines:
                    highlights = "\n\n".join(elon_lines[:10])  # Limitar a 10 trechos
                    elon_message = f"⚠️ INFORMAÇÕES RELEVANTES SOBRE ELON MUSK:\n\n{highlights}\n\n"
                    elon_message += "Você deve analisar estas informações e apresentar um resumo das últimas notícias sobre Elon Musk ao usuário."
                    self.memory.add_message(Message.system_message(elon_message))
                    logger.info("Adicionadas informações destacadas sobre Elon Musk")
            
            # Se o conteúdo HTML for grande, processar com chunking
            elif kwargs.get('args', {}).get('action') in ['get_html', 'get_text'] and isinstance(result, str) and len(result) > 10000:
                # Se o processador de conteúdo estiver disponível
                if hasattr(self, 'content_processor') and self.content_processor:
                    action = kwargs.get('args', {}).get('action')
                    content_type = 'html' if action == 'get_html' else 'text'
                    
                    logger.info(f"Detectado conteúdo grande ({len(result)} caracteres) do browser_use. Aplicando chunking.")
                    
                    # Extrair a consulta original
                    query = self._extract_query_from_memory()
                    
                    try:
                        # Processar o conteúdo com chunking
                        chunked_response = await self.content_processor.process_large_content(
                            content=result,
                            query=query,
                            content_type=content_type,
                            metadata={"source": "browser_use", "action": action}
                        )
                        
                        # Substituir o resultado original pelo processado
                        result = f"Conteúdo processado com chunking:\n\n{chunked_response}"
                        logger.info(f"Conteúdo {content_type} processado com chunking")
                    except Exception as e:
                        logger.error(f"Erro ao processar conteúdo com chunking: {e}")
                        # Manter o resultado original se houver erro
                        result = f"Erro ao processar conteúdo grande: {str(e)}\n\n{result[:2000]}... [conteúdo truncado]"
        
        await super()._handle_special_tool(name, result, **kwargs)
        
    def _extract_query_from_memory(self) -> str:
        """Extrai a consulta original do contexto da memória"""
        # Tentar encontrar a primeira mensagem do usuário
        for message in self.memory.messages:
            if message.role == "user":
                return message.content if message.content else "Analise este conteúdo"
        
        # Fallback
        return "Analise o conteúdo e extraia informações relevantes"
        
    async def handle_tool_failures(self, prompt: str, tool_failures: int) -> str:
        """Lidar com falhas de ferramentas, decidindo entre replanejamento ou intervenção do usuário"""
        if self.planning_attempts >= self.max_planning_attempts:
            # Máximo de tentativas atingido, pedir ajuda ao usuário
            logger.warning(f"Reached maximum planning attempts ({self.max_planning_attempts}). Asking for user intervention.")
            
            # Criar um resumo das falhas
            failure_summary = "\n\n💀 ENCONTREI DIFICULDADES NA EXECUÇÃO! 💀\n\n"
            failure_summary += f"Após {self.planning_attempts} tentativas, enfrentei os seguintes problemas:\n\n"
            
            for i, failure in enumerate(self.failed_tools, 1):
                failure_summary += f"{i}. Falha na ferramenta '{failure['tool']}': {failure['error']}\n"
                
            failure_summary += "\nComo você gostaria de prosseguir?\n"
            failure_summary += "1. Interromper a execução\n"
            failure_summary += "2. Tentar uma abordagem diferente (especifique como)\n"
            failure_summary += "3. Continuar mesmo com os erros\n"
            
            # Limpar o estado para uma nova interação
            self.planning_attempts = 0
            self.failed_tools = []
            
            return failure_summary
        else:
            # Tentar novamente com um plano diferente
            logger.info(f"Planning attempt {self.planning_attempts}/{self.max_planning_attempts}. Replanning...")
            
            # Criar um prompt de replanejamento informando os erros
            replan_prompt = f"Precisamos repensar nossa abordagem para: '{prompt}'\n\n"
            replan_prompt += "Os seguintes erros ocorreram na tentativa anterior:\n"
            
            for i, failure in enumerate(self.failed_tools, 1):
                replan_prompt += f"{i}. Ferramenta '{failure['tool']}' falhou: {failure['error']}\n"
                
            replan_prompt += "\nObservações importantes sobre as ferramentas:\n"
            replan_prompt += "- A ação 'extract_text' NÃO é suportada pelo browser_use\n"
            replan_prompt += "- Para navegar em uma página web, você DEVE fazer DUAS chamadas separadas:\n"
            replan_prompt += "  1. PRIMEIRO: browser_use com action='navigate' e url='https://exemplo.com'\n"
            replan_prompt += "  2. SEGUNDO: browser_use com action='get_html' (sem url)\n"
            replan_prompt += "- NUNCA combine url e get_html na mesma chamada\n\n"
            replan_prompt += "Por favor, crie um novo plano que evite os erros anteriores."
            
            # Adicionar uma mensagem do usuário com o replanejamento
            self.memory.add_message(Message.user_message(replan_prompt))
            
            # Executar uma nova análise com base nos erros anteriores
            analysis = await self.analyze_task(replan_prompt)
            
            # Atualizar o plano e reiniciar
            self.plan = analysis.get("steps", [])
            self.has_plan = True
            
            # Limpar as mensagens antigas e iniciar novo contexto
            # Manter apenas as mensagens iniciais e a mensagem de erro
            initial_messages = self.memory.messages[:2] if len(self.memory.messages) >= 2 else self.memory.messages
            error_messages = [msg for msg in self.memory.messages if msg.role == "system" and "ERROR ALERT" in (msg.content or "")]
            replan_message = [self.memory.messages[-1]] if self.memory.messages else []
            
            # Redefinir a memória mantendo apenas o essencial
            self.memory.messages = initial_messages + error_messages + replan_message
            
            # Limpar as ferramentas falhas para a nova tentativa
            self.failed_tools = []
            
            # Tentar executar o novo plano
            return await super().run(prompt)
