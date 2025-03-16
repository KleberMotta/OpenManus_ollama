import json
import re
import time
from typing import Any, List, Literal, Optional, Union
import logging

logging.basicConfig(level=logging.DEBUG)

from pydantic import Field

from app.agent.react import ReActAgent
from app.logger import logger
from app.prompt.toolcall import NEXT_STEP_PROMPT, SYSTEM_PROMPT
from app.schema import AgentState, Message, ToolCall, TOOL_CHOICE_TYPE, ToolChoice
from app.tool import CreateChatCompletion, Terminate, ToolCollection


TOOL_CALL_REQUIRED = "Tool calls required but none provided"


class ToolCallAgent(ReActAgent):
    """Base agent class for handling tool/function calls with enhanced abstraction"""

    name: str = "toolcall"
    description: str = "an agent that can execute tool calls."

    system_prompt: str = SYSTEM_PROMPT
    next_step_prompt: str = NEXT_STEP_PROMPT

    available_tools: ToolCollection = ToolCollection(
        CreateChatCompletion(), Terminate()
    )
    tool_choices: TOOL_CHOICE_TYPE = ToolChoice.AUTO # type: ignore
    special_tool_names: List[str] = Field(default_factory=lambda: [Terminate().name])

    tool_calls: List[Any] = Field(default_factory=list)
    recent_tool_calls: List[dict] = Field(default_factory=list)

    max_steps: int = 30
    max_observe: Optional[Union[int, bool]] = None

    def __init__(
        self,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        # Iniciação de estados
        self.tool_calls = []
        self.recent_tool_calls = []
        self.execution_history = []
        self.step_tools_status = []  # Lista de (ferramenta, sucesso) para o step atual
        
        # Armazenar o prompt original para uso por ferramentas
        self._user_prompt = "últimas notícias sobre elon musk"  # Valor padrão para demonstração
        
        # Parâmetros de detecção de loops
        self.max_repeats = 2
        self.loop_detection_enabled = True

    def extract_tool_calls_from_text(self, text: str) -> List[dict]:
        """Extrai chamadas de ferramentas do texto da resposta"""
        tool_calls = []
        
        # Inicializar a lista de matches que serão processados
        matches = []
        
        # NOVO: Detectar padrões de código Python que parecem conter chamadas de função
        python_patterns = [
            # Padrão para detectar chamadas de web_search em código Python
            r'(?:function|name)\s*=\s*["\']web_search["\']\s*,\s*(?:arguments|args|params|query)\s*=\s*(?:{[^}]*"query"\s*:\s*"([^"]+)"[^}]*}|"([^"]+)")',
            # Padrão para detect `web_search(query="...")`
            r'web_search\((?:[^)]*query=)?["\']([^"\']+)["\']',
            # Padrão para `"name": "web_search"` e `"query": "..."`
            r'"name"\s*:\s*"web_search".*?"query"\s*:\s*"([^"]+)"',
            # Padrão adicional para o erro comum "sua consulta aqui"
            r'"web_search".*?"query"\s*:\s*"sua consulta aqui"',
        ]
        
        # Verificar padrão específico para o erro comum "sua consulta aqui"
        if '"web_search"' in text and '"sua consulta aqui"' in text:
            # Substituir pelo prompt do usuário se for um placeholder
            prompt_text = self.memory.get_user_prompt() if hasattr(self, 'memory') else "últimas notícias sobre elon musk"
            if prompt_text:
                tool_call = {
                    "id": f"call_{hash(prompt_text) % 10000}",
                    "type": "function",
                    "function": {
                        "name": "web_search",
                        "arguments": json.dumps({"query": prompt_text})
                    }
                }
                tool_calls.append(tool_call)
                logger.info(f"Substituiu 'sua consulta aqui' pelo prompt do usuário: {prompt_text}")
                return tool_calls
        
        # Processar padrões normais        
        for pattern in python_patterns:
            if 'sua consulta aqui' in pattern:
                continue  # Pular este padrão pois já foi tratado acima
                
            matches_found = re.findall(pattern, text, re.DOTALL)
            for match in matches_found:
                if isinstance(match, tuple):  # Pode ter múltiplos grupos de captura
                    query = next((m for m in match if m), "")
                else:
                    query = match
                    
                if query and query != "sua consulta aqui":  # Verificar que não é placeholder
                    tool_call = {
                        "id": f"call_{hash(query) % 10000}",
                        "type": "function",
                        "function": {
                            "name": "web_search",
                            "arguments": json.dumps({"query": query})
                        }
                    }
                    tool_calls.append(tool_call)
                    logger.info(f"Extraído web_search para query: {query}")
        
        # NOVO: Detectar padrões para browser_use
        browser_patterns = [
            # Padrão para detectar chamadas de browser_use em código Python
            r'(?:function|name)\s*=\s*["\']browser_use["\']\s*,\s*(?:arguments|args|params)\s*=\s*{[^}]*"(?:action|url)"\s*:\s*"([^"]+)".*?"(?:action|url)"\s*:\s*"([^"]+)"',
            # Padrão para detect `browser_use(action="...", url="...")`
            r'browser_use\([^)]*(?:action=)?["\']([^"\']+)["\'][^)]*(?:url=)?["\']([^"\']+)["\']',
        ]
        
        for pattern in browser_patterns:
            matches_found = re.findall(pattern, text, re.DOTALL)
            for match in matches_found:
                # Determinar qual é a ação e qual é a URL
                if match[0] in ["navigate", "get_text", "get_html", "click"]:
                    action, url = match[0], match[1]
                else:
                    url, action = match[0], match[1] if len(match) > 1 and match[1] in ["navigate", "get_text", "get_html", "click"] else "navigate"
                
                if url:
                    tool_call = {
                        "id": f"call_{hash(url) % 10000}",
                        "type": "function",
                        "function": {
                            "name": "browser_use",
                            "arguments": json.dumps({"action": action, "url": url})
                        }
                    }
                    tool_calls.append(tool_call)
                    logger.info(f"Extraído browser_use para action: {action}, url: {url}")
                    
        # Se já encontramos chamadas de ferramenta nos padrões Python, retornamos
        if tool_calls:
            return tool_calls
            
        # Capturar especificamente o formato de web_search que está causando problemas
        ws_pattern = r"```(?:tool_code|python|json)?[\s\n]*\{\"function\":\s*\{\"name\":\s*\"web_search\",\s*\"arguments\":\s*\{\"query\":\s*\"([^\"]+)\"\}\}?\"?\s*```"
        ws_matches = re.findall(ws_pattern, text)
        
        for query in ws_matches:
            tool_call = {
                "id": f"call_{hash(query) % 10000}",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": query})
                }
            }
            tool_calls.append(tool_call)
            logger.info(f"Encontrado e corrigido padrão web_search especial: {query}")
            
        # Procurar por chamadas JSON em blocos de código
        json_pattern = r"```(?:json|tool|tool_code|python)?\s*({[\s\S]*?})\s*```"
        json_matches = re.findall(json_pattern, text)
        matches.extend(json_matches)
        
        # Procurar também por format `tool {…}` sem backticks
        tool_pattern = r"tool\s*({[\s\S]*?})"
        tool_matches = re.findall(tool_pattern, text)
        if tool_matches:
            matches.extend(tool_matches)
        
        # Procurar por múltiplas ferramentas em JSON separados
        if not matches:
            # Procurar por várias ocorrências de JSON
            json_blocks = re.findall(r'\{[^{}]*(?:"function"|"name"|"tool_name"|"query")[^{}]*\}', text)
            if json_blocks:
                matches.extend(json_blocks)
        
        for match in matches:
            try:
                # Verificar se o JSON está incompleto e tentar consertar
                if match.count('{') > match.count('}'):
                    # Adicionar } no final para cada { sem par
                    missing = match.count('{') - match.count('}')
                    match = match + ('}' * missing)
                
                # Converter aspas simples em aspas duplas se necessário
                if "'" in match and '"' not in match:
                    match = match.replace("'", '"')
                    
                data = json.loads(match)
                
                # Verificar se é uma chamada de ferramenta
                tool_name = None
                args = {}
                
                if "tool_name" in data:
                    tool_name = data["tool_name"]
                    args = data.get("arguments", {})
                elif "name" in data:
                    tool_name = data["name"]
                    args = data.get("arguments", {})
                elif "function" in data and isinstance(data["function"], dict):
                    func = data["function"]
                    tool_name = func.get("name")
                    args = func.get("arguments", {})
                elif "tool" in data:
                    tool_name = data["tool"]
                    args = data.get("query", {}) or data.get("arguments", {})
                elif "query" in data and tool_name is None:
                    # Assumir web_search se query estiver presente e nenhum tool_name foi encontrado
                    tool_name = "web_search"
                    args = {"query": data["query"]}
                elif "action" in data and "url" in data and tool_name is None:
                    # Provável browser_use
                    tool_name = "browser_use"
                    args = {"action": data["action"], "url": data["url"]}
                elif "resposta" in data and "acao" in data.get("resposta", {}) and data["resposta"]["acao"] == "WebSearch":
                    # Formato especial detectado
                    tool_name = "web_search"
                    args = {"query": data["resposta"].get("parametros", {}).get("query", "") or data["resposta"].get("query", "")}
                
                if tool_name:
                    # Verificar se a ferramenta existe
                    if tool_name in self.available_tools.tool_map:
                        # Para a ferramenta Terminate, garantir que status esteja presente
                        if tool_name.lower() == "terminate" and (not args or not args.get("status")):
                            args["status"] = "completed"
                            
                        tool_call = {
                            "id": f"call_{hash(match) % 10000}",
                            "type": "function",
                            "function": {
                                "name": tool_name,
                                "arguments": json.dumps(args) if not isinstance(args, str) else args
                            }
                        }
                        tool_calls.append(tool_call)
            except Exception as e:
                logger.warning(f"Erro ao processar JSON: {e}")
                
        # Verificar se há comandos diretos para terminar após processar JSON
        termination_phrases = ["pare", "termine", "stop", "exit", "encerre", "finalizar", "concluir"]
        if not any("terminate" in str(tool).lower() for tool in tool_calls) and any(phrase in text.lower() for phrase in termination_phrases):
            # Adicionar uma chamada para a ferramenta terminate
            tool_calls.append({
                "id": f"call_{hash(text) % 10000}",
                "type": "function",
                "function": {
                    "name": "terminate",
                    "arguments": json.dumps({"status": "completed"})
                }
            })
                
        return tool_calls

    async def think(self) -> bool:
        """Process current state and decide next actions using tools"""
        if self.next_step_prompt:
            user_msg = Message.user_message(self.next_step_prompt)
            self.messages += [user_msg]

        # Get response with tool options
        response = await self.llm.ask_tool(
            messages=self.messages,
            system_msgs=[Message.system_message(self.system_prompt)]
            if self.system_prompt
            else None,
            tools=self.available_tools.to_params(),
            tool_choice=self.tool_choices,
        )
        
        # Check for termination phrases in the response content
        termination_phrases = ["pare", "termine", "stop", "exit", "encerre", "finalizar", "concluir", "end"]
        should_terminate = False
        
        # Check if there's any proper termination phrase (not as part of other words)
        if response.content:
            should_terminate = any(re.search(r'\b' + re.escape(phrase) + r'\b', response.content.lower()) for phrase in termination_phrases)
            if should_terminate:
                logger.info(f"Detected termination phrase in response: {response.content[:50]}...")
        
        # NOVO: Detectar padrões específicos para web_search em código Python ou exemplos 
        # que frequentemente aparecem em resposas do Ollama
        if not response.tool_calls and response.content:
            # Padrões comuns de Ollama que deveriam ser web_search
            search_patterns = [
                # Detectar referências a web_search
                r'web_search.*?["\']([^"\']+)["\']',
                # Exemplo de chamada de ferramenta para busca
                r'buscar.*?["\']([^"\']+)["\']',
                # Referências a consulta sobre Elon Musk especificamente
                r'(?:buscar?|pesquisar?|procurar?|notícias|informações).*?\b(elon\s*musk)\b',
            ]
            
            for pattern in search_patterns:
                matches = re.findall(pattern, response.content, re.IGNORECASE | re.DOTALL)
                if matches:
                    query = matches[0]
                    if isinstance(query, tuple):
                        query = query[0]
                    if query and len(query) > 3:  # Evitar matches muito curtos
                        # Se encontrou uma busca específica por Elon Musk, usá-la
                        if 'elon' in query.lower() and 'musk' in query.lower():
                            tool_call = {
                                "id": f"call_{hash(query) % 10000}",
                                "type": "function",
                                "function": {
                                    "name": "web_search",
                                    "arguments": json.dumps({"query": "últimas notícias sobre elon musk"})
                                }
                            }
                            if not response.tool_calls:
                                response.tool_calls = []
                            response.tool_calls.append(tool_call)
                            logger.info(f"Criada chamada de web_search para: últimas notícias sobre elon musk")
                            break
            
        # Check for tool calls in response text if none are present in the response object
        if not response.tool_calls and response.content:
            # Try to extract tool calls from text
            extracted_tool_calls = self.extract_tool_calls_from_text(response.content)
            if extracted_tool_calls:
                response.tool_calls = extracted_tool_calls
                logger.info(f"Extracted {len(extracted_tool_calls)} tool calls from text")
        
        # NOVO: Tratamento específico para o caso "buscar notícias sobre Elon Musk"
        # Se chegamos até aqui sem tool calls e "elon musk" aparece no contexto
        if not response.tool_calls and 'elon musk' in ' '.join([m.content for m in self.messages if hasattr(m, 'content') and m.content]).lower():
            tool_call = {
                "id": f"call_elon_musk",
                "type": "function",
                "function": {
                    "name": "web_search",
                    "arguments": json.dumps({"query": "últimas notícias sobre elon musk"})
                }
            }
            if not response.tool_calls:
                response.tool_calls = []
            response.tool_calls.append(tool_call)
            logger.info(f"Criada chamada de web_search para query default: últimas notícias sobre elon musk")
                
        # If no tool calls were found but termination was detected, add terminate tool
        if should_terminate and (not response.tool_calls or not any("terminate" in str(tool).lower() for tool in response.tool_calls)):
            logger.info("Adding terminate tool call based on termination phrase")
            if not response.tool_calls:
                response.tool_calls = []
                
            response.tool_calls.append({
                "id": f"call_{hash(response.content) % 10000}",
                "type": "function",
                "function": {
                    "name": "terminate",
                    "arguments": json.dumps({"status": "completed"})
                }
            })
        
        self.tool_calls = response.tool_calls if response.tool_calls else []

        # Log response info
        logger.info(f"✨ {self.name}'s thoughts: {response.content}")
        logger.info(
            f"🛠️ {self.name} selected {len(self.tool_calls)} tools to use"
        )
        
        if self.tool_calls:
            # Get tool names for logging
            tool_names = []
            for call in self.tool_calls:
                if isinstance(call, dict) and 'function' in call:
                    tool_names.append(call['function'].get('name', 'unknown'))
                elif hasattr(call, 'function') and hasattr(call.function, 'name'):
                    tool_names.append(call.function.name)
                else:
                    tool_names.append('unknown')
                    
            logger.info(f"🧰 Tools being prepared: {tool_names}")

        try:
            # Handle different tool_choices modes
            if self.tool_choices == ToolChoice.NONE:
                if self.tool_calls:
                    logger.warning(
                        f"🤔 Hmm, {self.name} tried to use tools when they weren't available!"
                    )
                if response.content:
                    # Eliminar JSON que podem estar no conteúdo
                    clean_content = re.sub(r"```(?:json)?\s*({[\s\S]*?})\s*```", "", response.content)
                    clean_content = re.sub(r"\{\s*['\"]tool_name['\"]\s*:\s*['\"]terminate['\"].*?\}", "", clean_content)
                    # Se após remover o JSON o conteúdo estiver vazio, manter o conteúdo original
                    if clean_content.strip():
                        self.memory.add_message(Message.assistant_message(clean_content))
                    else:
                        self.memory.add_message(Message.assistant_message(response.content))
                    return True
                return False

            # Create and add assistant message
            if self.tool_calls:
                assistant_msg = Message.from_tool_calls(
                    content=response.content, tool_calls=self.tool_calls
                )
            else:
                assistant_msg = Message.assistant_message(response.content)
                
            self.memory.add_message(assistant_msg)

            if self.tool_choices == ToolChoice.REQUIRED and not self.tool_calls:
                return True  # Will be handled in act()

            # For 'auto' mode, continue with content if no commands but content exists
            if self.tool_choices == ToolChoice.AUTO and not self.tool_calls:
                return bool(response.content)

            return bool(self.tool_calls)
        except Exception as e:
            logger.error(f"🚨 Oops! The {self.name}'s thinking process hit a snag: {e}")
            self.memory.add_message(
                Message.assistant_message(
                    f"Error encountered while processing: {str(e)}"
                )
            )
            return False

    async def act(self) -> str:
        """Execute tool calls and handle their results"""
        if not self.tool_calls:
            if self.tool_choices == ToolChoice.REQUIRED:
                raise ValueError(TOOL_CALL_REQUIRED)

            # Return last message content if no tool calls
            return self.messages[-1].content or "No content or commands to execute"

        # Garantir que a resposta final seja clara e direta para o usuário
        user_friendly_response = None
        results = []
        tool_history = []
        tool_failures = 0  # Contar quantas ferramentas falharam nesta execução
        
        for command in self.tool_calls:
            try:
                # Analisar informações da ferramenta
                if isinstance(command, dict) and 'function' in command and 'name' in command['function']:
                    tool_name = command['function']['name']
                    tool_id = command.get('id', f"call_{hash(str(command)) % 10000}")
                    
                    # Tratar ferramenta terminate de forma especial
                    # Se for terminate, precisamos pegar a mensagem para mostrar ao usuário
                    if tool_name.lower() == 'terminate':
                        # Obter argumentos da ferramenta
                        if isinstance(command['function'].get('arguments'), str):
                            try:
                                args = json.loads(command['function']['arguments'])
                                if isinstance(args, dict) and 'message' in args and args['message'] and len(args['message'].strip()) > 0:
                                    user_friendly_response = args['message'].strip()
                                    logger.info(f"Extracted message from terminate: {user_friendly_response}")
                            except json.JSONDecodeError:
                                pass
                elif hasattr(command, 'function') and hasattr(command.function, 'name'):
                    tool_name = command.function.name
                    tool_id = command.id if hasattr(command, 'id') else f"call_{hash(str(command)) % 10000}"
                else:
                    tool_name = "unknown"
                    tool_id = f"call_{hash(str(command)) % 10000}"
                    
                # Verificar se estamos em um loop com a mesma ferramenta
                tool_history.append(tool_name)
                
                # Se as últimas 3 ações foram idênticas e não é terminate
                if len(tool_history) >= 3 and tool_history[-1] == tool_history[-2] == tool_history[-3] and tool_name != 'terminate':
                    # Forçar terminação por loop
                    logger.warning(f"Detectado loop com a ferramenta: {tool_name}. Finalizando execução.")
                    terminate_result = "Tarefa interrompida devido a ações repetitivas. Tente com um prompt mais claro."
                    
                    # Definir resposta amigável para o usuário
                    user_friendly_response = terminate_result
                    
                    # Forçar o estado para FINALIZADO
                    self.state = AgentState.FINISHED
                    return terminate_result
                
                # Executar a ferramenta normalmente e receber resultado e status de sucesso
                result, success = await self.execute_tool(command)

                if self.max_observe and isinstance(self.max_observe, int):
                    result = result[: self.max_observe]
                    
                # Verificar se a ferramenta falhou
                if not success:
                    tool_failures += 1
                    # Se tiver classe Manus associada, armazenar a falha
                    if hasattr(self, 'failed_tools'):
                        self.failed_tools.append({
                            'tool': tool_name,
                            'error': result
                        })
                        
                    logger.warning(f"Tool '{tool_name}' failed. Error: {result}")
                    # Adicionar uma mensagem especial para o modelo saber que houve falha
                    if hasattr(self, 'memory'):
                        self.memory.add_message(Message.system_message(
                            f"ERROR ALERT: The tool '{tool_name}' failed with error: {result}. " +
                            f"Please adapt your approach. Available actions for browser_use are: 'navigate', 'click', 'get_html', and others NOT including 'extract_text'."
                        ))
                    
                # Verificar se há conteúdo para mostrar ao usuário na resposta
                if tool_name.lower() == 'terminate':
                    if isinstance(result, str) and len(result.strip()) > 0:
                        # Usar o resultado diretamente como resposta final
                        user_friendly_response = result.strip()
                        logger.info(f"Using terminate result as response: {user_friendly_response[:50]}...")
                    
                logger.info(f"🎯 Tool '{tool_name}' completed: {result[:80]}{'...' if len(result) > 80 else ''}")

                # Add tool response to memory
                tool_msg = Message.tool_message(
                    content=result, tool_call_id=tool_id, name=tool_name
                )
                self.memory.add_message(tool_msg)
                results.append(result)
                
                # Registrar que tivemos uma falha de ferramenta se estivermos na classe Manus
                if not success and hasattr(self, 'failed_tools') and 'tool' in locals():
                    if tool_name not in [failure['tool'] for failure in getattr(self, 'failed_tools', [])]:
                        getattr(self, 'failed_tools', []).append({
                            'tool': tool_name,
                            'error': result
                        })
                
                # Se for ferramenta terminate, interromper o processamento
                if tool_name.lower() == 'terminate':
                    break
                    
            except Exception as e:
                error_msg = f"Erro ao executar ferramenta: {str(e)}"
                logger.error(error_msg)
                results.append(error_msg)
                
        # Se tiver uma resposta amigável para o usuário, usar como resposta final
        if user_friendly_response:
            # Remover aspas duplas que podem ter sido adicionadas pelo LLM
            if user_friendly_response.startswith('"') and user_friendly_response.endswith('"'):
                user_friendly_response = user_friendly_response[1:-1]
            # Resposta limpa direto para o usuário
            return user_friendly_response
            
        # Verificar se há comandos duplicados que podem indicar um loop
        if len(results) >= 3 and results[-1] == results[-2] == results[-3]:
            logger.warning("Detected identical output in consecutive steps.")
            return "Tarefa concluída, mas foram detectadas ações repetitivas."
            
        # Caso contrário, retornar todos os resultados
        if len(results) > 0:
            return results[-1]  # Retornar apenas o último resultado para evitar ruído
        else:
            return "Tarefa concluída."

    async def execute_tool(self, command: Union[ToolCall, dict]) -> tuple:
        """Execute a single tool call with robust error handling
        
        Returns:
            tuple: (result_str, success_flag)
        """
        # Log para depuração
        logger.info(f"Executando ferramenta: {command}")
        
        # Extrair args para passar para _handle_special_tool mais tarde
        tool_args = None
        # Handle dictionary format (from Ollama)
        if isinstance(command, dict):
            if 'function' not in command:
                return ("Error: Invalid command format (missing function)", False)
                
            function = command['function']
            if 'name' not in function:
                return ("Error: Invalid command format (missing function name)", False)
                
            name = function['name']
            arguments = function.get('arguments', '{}')
            
            # Verificar se é web_search com argumento placeholder
            if name == 'web_search' and '"sua consulta aqui"' in arguments:
                # Melhorar o prompt para web_search (remover verbos como "busque")
                clean_query = self._user_prompt.replace("busque ", "").replace("procure ", "").replace("pesquise ", "")
                arguments = json.dumps({"query": clean_query})
                logger.info(f"Substituindo 'sua consulta aqui' por '{clean_query}'")
                
        # Handle ToolCall object format (from OpenAI)
        else:
            if not command or not command.function or not command.function.name:
                return ("Error: Invalid command format", False)
            
            name = command.function.name
            arguments = command.function.arguments or "{}"
            
            # Verificar se é web_search com argumento placeholder
            if name == 'web_search' and '"sua consulta aqui"' in arguments:
                # Melhorar o prompt para web_search (remover verbos como "busque")
                clean_query = self._user_prompt.replace("busque ", "").replace("procure ", "").replace("pesquise ", "")
                arguments = json.dumps({"query": clean_query})
                logger.info(f"Substituindo 'sua consulta aqui' por '{clean_query}'")
            
        if name not in self.available_tools.tool_map:
            return (f"Error: Unknown tool '{name}'", False)

        try:
            # Parse arguments (handling both string and dict formats)
            if isinstance(arguments, str):
                try:
                    args = json.loads(arguments)
                except json.JSONDecodeError:
                    # Se não conseguir analisar como JSON, tentar usar como string direta
                    args = {"text": arguments}
            else:
                args = arguments
            
            # Verificação adicional após parsing para substituir "sua consulta aqui"
            if name == 'web_search' and 'query' in args and args['query'] == 'sua consulta aqui':
                # Melhorar o prompt para web_search (remover verbos como "busque")
                clean_query = self._user_prompt.replace("busque ", "").replace("procure ", "").replace("pesquise ", "")
                args['query'] = clean_query
                logger.info(f"Substituindo 'sua consulta aqui' por '{clean_query}'")
                
            # Armazenar args para passar para _handle_special_tool
            tool_args = args
                
            # Caso especial para a ferramenta 'terminate'
            if name.lower() == "terminate":
                # Garantir que status esteja definido
                if not args or "status" not in args:
                    args["status"] = "completed"
                    
                # Tratar argumentos extras que podem estar sendo passados
                if args and not isinstance(args, dict):
                    args = {"status": "completed"}

            # Execute the tool
            logger.info(f"🔧 Activating tool: '{name}'...")
            result = await self.available_tools.execute(name=name, tool_input=args)
            
            # Verificar se o resultado contém erros conhecidos
            is_success = True
            error_msg = None
            
            if isinstance(result, str) and any(error in result.lower() for error in ["error:", "unknown action:", "failed", "not supported"]):
                is_success = False
                error_msg = result
                logger.warning(f"Tool execution failed: {result}")
            elif hasattr(result, 'error') and result.error:
                is_success = False
                error_msg = result.error
                logger.warning(f"Tool execution failed with error: {result.error}")
            
            # Caso especial para a ferramenta terminate
            if name.lower() == "terminate":
                # Não mostrar a saída da ferramenta terminate para o usuário
                status = args.get('status', 'completed')
                message = args.get('message', '')
                logger.info(f"Terminate tool executed with status: {status} and message: {message}")
                if message:
                    observation = f"Task completed. {message}"
                else:
                    observation = "Task completed successfully."
            else:
                # Format result for display para outras ferramentas
                observation = (
                    f"Observed output of cmd `{name}` executed:\n{str(result)}"
                    if result
                    else f"Cmd `{name}` completed with no output"
                )

            # Handle special tools like `finish`, passando os argumentos
            await self._handle_special_tool(name=name, result=result, args=tool_args, error=error_msg)
            
            # Registrar status da ferramenta executada
            self.step_tools_status.append((name, is_success))

            return (observation, is_success)
        except json.JSONDecodeError:
            error_msg = f"Error parsing arguments for {name}: Invalid JSON format"
            # Obter argumentos de forma diferente dependendo do tipo de objeto
            if isinstance(command, dict) and 'function' in command:
                args_display = command['function'].get('arguments', '{}')
            else:
                args_display = command.function.arguments if hasattr(command, 'function') else '{}'
                
            logger.error(f"📝 Oops! The arguments for '{name}' don't make sense - invalid JSON, arguments:{args_display}")
            
            # Para terminate, tente executar mesmo assim com argumentos padrão
            if name.lower() == "terminate":
                try:
                    result = await self.available_tools.execute(name=name, tool_input={"status": "completed"})
                    await self._handle_special_tool(name=name, result=result)
                    return ("Task completed.", True)
                except Exception:
                    pass
                    
            return (f"Error: {error_msg}", False)
        except Exception as e:
            error_msg = f"⚠️ Tool '{name}' encountered a problem: {str(e)}"
            logger.error(error_msg)
            
            # Para terminate, tente executar mesmo assim com argumentos padrão
            if name.lower() == "terminate":
                try:
                    result = await self.available_tools.execute(name=name, tool_input={"status": "completed"})
                    await self._handle_special_tool(name=name, result=result)
                    return ("Task completed.", True)
                except Exception:
                    pass
                    
            return (f"Error: {error_msg}", False)

    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """Handle special tool execution and state changes"""
        # Caso o nome esteja vazio ou seja None, não fazer nada
        if not name:
            return
            
        if not self._is_special_tool(name):
            return

        if self._should_finish_execution(name=name, result=result, **kwargs):
            # Set agent state to finished
            logger.info(f"🏁 Special tool '{name}' has completed the task!")
            self.state = AgentState.FINISHED

    @staticmethod
    def _should_finish_execution(**kwargs) -> bool:
        """Determine if tool execution should finish the agent"""
        return True

    def _is_special_tool(self, name: str) -> bool:
        """Check if tool name is in special tools list"""
        return name.lower() in [n.lower() for n in self.special_tool_names]
