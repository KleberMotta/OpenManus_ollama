from typing import Dict, List, Optional, Union
import json
import asyncio
import aiohttp
import toml
import os
import re
import uuid
from tenacity import retry, stop_after_attempt, wait_random_exponential, RetryError

from app.config import LLMSettings, config
from app.logger import logger
from app.schema import Message, TOOL_CHOICE_TYPE, ROLE_VALUES, TOOL_CHOICE_VALUES, ToolChoice


class LLM:
    _instances: Dict[str, "LLM"] = {}

    def __new__(
        cls, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if config_name not in cls._instances:
            instance = super().__new__(cls)
            instance.__init__(config_name, llm_config)
            cls._instances[config_name] = instance
        return cls._instances[config_name]

    def __init__(
        self, config_name: str = "default", llm_config: Optional[LLMSettings] = None
    ):
        if not hasattr(self, "client"):  # Only initialize if not already initialized
            if llm_config:
                self.model = llm_config.model
                self.max_tokens = llm_config.max_tokens
                self.temperature = llm_config.temperature
                self.api_key = llm_config.api_key
                self.base_url = llm_config.base_url or "http://localhost:11434"
            else:
                # Manually load config from toml
                config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config", "config.toml")
                with open(config_path, "r", encoding="utf-8") as f:
                    config_data = toml.load(f)
                
                llm_config = config_data.get("llm", {})
                self.model = llm_config.get("model", "qwen2.5-coder:7b-instruct")
                self.base_url = llm_config.get("base_url", "http://localhost:11434")
                self.max_tokens = int(llm_config.get("max_tokens", 4096))
                self.temperature = float(llm_config.get("temperature", 0.0))
                self.api_key = llm_config.get("api_key", "")
            
            self.session = None  # Will be initialized when needed

    async def _ensure_session(self):
        """Ensure aiohttp session is initialized"""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession()
        return self.session
        
    async def close(self):
        """Fecha a sessão HTTP de forma assíncrona"""
        if hasattr(self, 'session') and self.session is not None and not self.session.closed:
            await self.session.close()
            self.session = None
            
    @retry(
        wait=wait_random_exponential(min=1, max=10),
        stop=stop_after_attempt(3),
    )
    async def check_ollama_status(self):
        """Verifica se o servidor Ollama está respondendo"""
        try:
            session = await self._ensure_session()
            async with session.get(
                f"{self.base_url}/api/tags",
                timeout=5  # timeout curto para verificar disponibilidade
            ) as response:
                if response.status == 200:
                    return True
                return False
        except Exception as e:
            logger.warning(f"Erro ao verificar status do Ollama: {e}")
            return False

    @staticmethod
    def format_messages(messages: List[Union[dict, Message, str]]) -> List[dict]:
        """
        Format messages for LLM by converting them to message format.
        Enhanced version that handles string messages.
        """
        formatted_messages = []

        for message in messages:
            if isinstance(message, dict):
                # If message is already a dict, ensure it has required fields
                if "role" not in message:
                    raise ValueError("Message dict must contain 'role' field")
                formatted_messages.append(message)
            elif isinstance(message, Message):
                # If message is a Message object, convert it to dict
                formatted_messages.append(message.to_dict())
            elif isinstance(message, str):
                # Se for uma string, assumir como mensagem do usuário
                logger.info(f"Convertendo mensagem string para formato de mensagem: {message[:50]}...")
                formatted_messages.append({
                    "role": "user",
                    "content": message
                })
            else:
                # Tipo não suportado, log de aviso e conversão para string
                logger.warning(f"Tipo de mensagem não reconhecido: {type(message)}, tentando converter para string")
                try:
                    # Tenta converter para string como fallback
                    string_content = str(message)
                    formatted_messages.append({
                        "role": "user", 
                        "content": string_content
                    })
                except Exception as e:
                    raise TypeError(f"Não foi possível converter o tipo {type(message)} para string: {e}")

        # Validate all messages have required fields
        for msg in formatted_messages:
            if msg["role"] not in ROLE_VALUES:
                raise ValueError(f"Invalid role: {msg['role']}")
            if "content" not in msg and "tool_calls" not in msg:
                raise ValueError(
                    "Message must contain either 'content' or 'tool_calls'"
                )

        return formatted_messages

    @retry(
        wait=wait_random_exponential(min=1, max=20),
        stop=stop_after_attempt(3),
    )
    async def ask(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        stream: bool = True,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Send a prompt to the Ollama LLM and get the response.
        """
        # Verificar primeiro se o Ollama está disponível
        try:
            ollama_status = await self.check_ollama_status()
            if not ollama_status:
                raise ValueError("Ollama não está disponível. Verifique se o servidor está rodando.")
        except Exception as e:
            logger.error(f"Erro ao verificar status do Ollama: {e}")
            return "Não foi possível conectar ao Ollama. Verifique se o servidor está rodando."
            
        try:
            # Format system and user messages
            if system_msgs:
                system_msgs = self.format_messages(system_msgs)
                messages = system_msgs + self.format_messages(messages)
            else:
                messages = self.format_messages(messages)

            # Convert messages to a prompt format Ollama can understand
            prompt = ""
            for msg in messages:
                role = msg["role"]
                content = msg.get("content", "")
                
                if role == "system":
                    prompt += f"<s>{content}</s>\n\n"
                elif role == "user":
                    prompt += f"{content}\n\n"
                elif role == "assistant":
                    prompt += f"{content}\n\n"
                elif role == "tool":
                    prompt += f"<tool>{content}</tool>\n\n"

            # Prepare the request payload
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature or self.temperature,
                    "num_predict": self.max_tokens
                }
            }

            session = await self._ensure_session()
            
            if not stream:
                # Non-streaming request
                async with session.post(
                    f"{self.base_url}/api/generate",
                    headers={"Content-Type": "application/json"},
                    json=payload,
                    timeout=90  # aumentar timeout para evitar erros
                ) as response:
                    if response.status != 200:
                        error_text = await response.text()
                        raise ValueError(f"Error from Ollama API: {error_text}")
                    
                    result = await response.json()
                    return result.get("response", "")

            # Streaming request
            collected_messages = []
            
            async with session.post(
                f"{self.base_url}/api/generate",
                headers={"Content-Type": "application/json"},
                json=payload,
                timeout=90  # aumentar timeout para evitar erros
            ) as response:
                if response.status != 200:
                    error_text = await response.text()
                    raise ValueError(f"Error from Ollama API: {error_text}")
                
                # Process the streaming response
                buffer = ""
                async for chunk in response.content:
                    buffer += chunk.decode('utf-8')
                    if buffer.endswith('\n'):
                        try:
                            for line in buffer.strip().split('\n'):
                                if line:
                                    chunk_data = json.loads(line)
                                    chunk_message = chunk_data.get("response", "")
                                    collected_messages.append(chunk_message)
                                    print(chunk_message, end="", flush=True)
                        except json.JSONDecodeError:
                            pass
                        buffer = ""

            print()  # Newline after streaming
            full_response = "".join(collected_messages).strip()
            if not full_response:
                raise ValueError("Empty response from streaming LLM")
            return full_response

        except RetryError:
            logger.error("Máximo de tentativas esgotado ao chamar o Ollama")
            return "O servidor Ollama não está respondendo após várias tentativas. Pode estar sem memória ou sobrecarregado."
        except Exception as e:
            logger.error(f"Error in ask: {e}")
            # Se houver erro de memória, fornecer uma resposta mais útil
            if "out of memory" in str(e):
                return "O Ollama está com problemas de memória. Tente reiniciar o servidor Ollama ou usar um modelo menor."
            if "NULL" in str(e) or "mem_buffer" in str(e):
                return "O Ollama está enfrentando problemas de memória. Tente reiniciar o servidor."
            # Resposta genérica para outros erros
            return f"Ocorreu um erro ao processar sua solicitação: {str(e)}"

    def extract_tool_call(self, response_text, tools):
        """Extrai chamada de ferramenta de texto não estruturado de forma mais robusta"""
        
        # NOVO: Detectar chamadas de função em padrões de código Python
        # Para web_search
        web_search_patterns = [
            r'(?:function|name)\s*=\s*["\']web_search["\']\s*,\s*(?:arguments|args|params|query)\s*=\s*(?:{[^}]*"query"\s*:\s*"([^"]+)"[^}]*}|"([^"]+)")',
            r'web_search\((?:[^)]*query=)?["\']([^"\']+)["\']',
            r'"name"\s*:\s*"web_search".*?"query"\s*:\s*"([^"]+)"',
            r'sua consulta aqui|elon musk',  # Valores de exemplo/placeholder comuns
        ]
        
        for pattern in web_search_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                if isinstance(match, tuple):  # Pode ter múltiplos grupos de captura
                    query = next((m for m in match if m), "")
                else:
                    query = match
                    
                if query and query != "sua consulta aqui":
                    # Se for uma consulta real (não um placeholder)
                    return "web_search", {"query": query}
        
        # NOVO: Para browser_use
        browser_use_patterns = [
            r'(?:function|name)\s*=\s*["\']browser_use["\']\s*,\s*(?:arguments|args|params)\s*=\s*{[^}]*"(?:action|url)"\s*:\s*"([^"]+)".*?"(?:action|url)"\s*:\s*"([^"]+)"',
            r'browser_use\([^)]*(?:action=)?["\']([^"\']+)["\'][^)]*(?:url=)?["\']([^"\']+)["\']',
        ]
        
        for pattern in browser_use_patterns:
            matches = re.findall(pattern, response_text, re.DOTALL)
            for match in matches:
                # Determinar qual é a ação e qual é a URL
                if match[0] in ["navigate", "get_text", "get_html", "click"]:
                    action, url = match[0], match[1]
                else:
                    url, action = match[0], match[1] if len(match) > 1 and match[1] in ["navigate", "get_text", "get_html", "click"] else "navigate"
                    
                if url and "exemplo.com" not in url:  # Não usar URLs de exemplo
                    return "browser_use", {"action": action, "url": url}
        
        # Tentar extrair JSON entre marcadores de código
        json_pattern = r"```(?:json|tool|tool_code|python)?\s*({[\s\S]*?})\s*```"
        matches = re.findall(json_pattern, response_text)
        
        if matches:
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
                    # Verificar se tem os campos esperados para uma ferramenta
                    if "tool_name" in data or "name" in data:
                        tool_name = data.get("tool_name") or data.get("name")
                        args = data.get("arguments") or data.get("args") or {}
                        return tool_name, args
                    elif "function" in data and isinstance(data["function"], dict):
                        func = data["function"]
                        tool_name = func.get("name")
                        args = func.get("arguments", {})
                        if tool_name:
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    pass
                            return tool_name, args
                    elif "query" in data:
                        # Assumir web_search se query estiver presente
                        return "web_search", {"query": data["query"]}
                    elif "action" in data and "url" in data:
                        # Provável browser_use
                        return "browser_use", {"action": data["action"], "url": data["url"]}
                except Exception as e:
                    logger.warning(f"Erro ao analisar JSON em marcadores de código: {str(e)}")
                    continue
        
        # Tentar extrair JSON de qualquer lugar no texto
        try:
            # Procurar por JSON que contenha chaves relevantes
            json_blocks = re.findall(r'\{[^{}]*(?:"function"|"name"|"tool_name"|"query"|"action"|"url")[^{}]*\}', response_text)
            
            for json_content in json_blocks:
                # Converter aspas simples em aspas duplas se necessário
                if "'" in json_content and '"' not in json_content:
                    json_content = json_content.replace("'", '"')
                    
                try:
                    data = json.loads(json_content)
                    if "tool_name" in data or "name" in data:
                        tool_name = data.get("tool_name") or data.get("name")
                        args = data.get("arguments") or data.get("args") or {}
                        return tool_name, args
                    elif "function" in data and isinstance(data["function"], dict):
                        func = data["function"]
                        tool_name = func.get("name")
                        args = func.get("arguments", {})
                        if tool_name:
                            if isinstance(args, str):
                                try:
                                    args = json.loads(args)
                                except:
                                    pass
                            return tool_name, args
                    elif "query" in data:
                        # Assumir web_search se query estiver presente
                        return "web_search", {"query": data["query"]}
                    elif "action" in data and "url" in data:
                        # Provável browser_use
                        return "browser_use", {"action": data["action"], "url": data["url"]}
                except:
                    continue
        except Exception as e:
            logger.warning(f"Erro ao extrair JSON do texto completo: {str(e)}")
        
        # Tentar extrair nome de ferramenta e argumentos via heurística
        for tool in tools:
            try:
                if isinstance(tool, dict) and "function" in tool:
                    func_info = tool["function"]
                    tool_name = func_info.get("name")
                    if tool_name and tool_name in response_text:
                        # Tentar extrair JSON próximo ao nome da ferramenta
                        parts = response_text.split(tool_name)
                        for part in parts[1:]:  # Checar após o nome da ferramenta
                            json_start = part.find("{")
                            if json_start >= 0:
                                json_end = part.rfind("}")
                                if json_end > json_start:
                                    try:
                                        json_content = part[json_start:json_end+1]
                                        args = json.loads(json_content)
                                        return tool_name, args
                                    except:
                                        pass
            except Exception as e:
                logger.warning(f"Erro ao processar ferramenta {tool}: {str(e)}")
        
        # Se nenhuma ferramenta for encontrada usando os métodos acima, 
        # e existe apenas uma ferramenta disponível, assume que é essa
        if len(tools) == 1 and "function" in tools[0]:
            tool_name = tools[0]["function"].get("name")
            # Tenta encontrar qualquer objeto JSON na resposta
            try:
                start_idx = response_text.find("{")
                if start_idx >= 0:
                    end_idx = response_text.rfind("}")
                    if end_idx > start_idx:
                        json_content = response_text[start_idx:end_idx+1]
                        args = json.loads(json_content)
                        # Verifica se há pelos menos um argumento válido
                        if isinstance(args, dict) and len(args) > 0:
                            return tool_name, args
            except:
                pass
            
            # Se não encontrou JSON, mas é uma ferramenta simples como get_weather,
            # tenta extrair o argumento de location do texto
            if tool_name == "get_weather":
                location_match = re.search(r'em\s+([^?\.!]+)', response_text)
                if location_match:
                    location = location_match.group(1).strip()
                    return tool_name, {"location": location}
        
        return None, None

    def transform_to_openai_format(self, response_text, tools=None):
        """Transforma resposta do Ollama para formato OpenAI"""
        
        if not tools:
            # Resposta simples de texto
            return {
                "role": "assistant",
                "content": response_text
            }
        
        # Tentar extrair tool call
        tool_name, args = self.extract_tool_call(response_text, tools)
        
        if tool_name and args:
            # Criar formato de resposta com tool call
            tool_call = {
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": tool_name,
                    "arguments": json.dumps(args)
                }
            }
            
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [tool_call]
            }
        else:
            # Nenhuma ferramenta encontrada, retornar como resposta de texto
            return {
                "role": "assistant",
                "content": response_text
            }

    async def ask_tool(
        self,
        messages: List[Union[dict, Message]],
        system_msgs: Optional[List[Union[dict, Message]]] = None,
        timeout: int = 300,
        tools: Optional[List[dict]] = None,
        tool_choice: TOOL_CHOICE_TYPE = ToolChoice.AUTO,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        """
        Ask LLM using functions/tools and return the response.
        Uses improved formatting and parsing to work with Ollama.
        """
        try:
            # Preparar mensagens formatadas
            formatted_messages = []
            
            # Processar mensagens de sistema
            if system_msgs:
                system_msgs_formatted = self.format_messages(system_msgs)
                formatted_messages.extend(system_msgs_formatted)
            
            # Adicionar instruções específicas para formatação de ferramenta
            if tools:
                tool_instructions = """
                Para utilizar uma ferramenta, responda neste formato:
                ```json
                {
                  "tool_name": "NOME_DA_FERRAMENTA",
                  "arguments": {
                    "arg1": "valor1",
                    "arg2": "valor2"
                  }
                }
                ```
                Se não for usar uma ferramenta, responda normalmente em texto.
                """
                
                tools_desc = "Você tem acesso às seguintes ferramentas:\n"
                for tool in tools:
                    if tool.get("type") == "function" and "function" in tool:
                        func = tool["function"]
                        tools_desc += f"- {func.get('name', 'unknown')}: {func.get('description', 'No description')}\n"
                
                # Adicionar descrição das ferramentas como mensagem do sistema
                formatted_messages.append({
                    "role": "system",
                    "content": f"{tools_desc}\n\n{tool_instructions}"
                })
            
            # Adicionar mensagens do usuário
            formatted_messages.extend(self.format_messages(messages))
            
            # Obter resposta como texto
            response_text = await self.ask(
                messages=formatted_messages,
                stream=False,
                temperature=temperature,
            )
            
            # Se a resposta contiver uma mensagem de erro, retorná-la diretamente
            if response_text.startswith("O Ollama está com problemas") or response_text.startswith("Não foi possível conectar"):
                return type('ErrorMessage', (), {'content': response_text, 'tool_calls': None, 'role': 'assistant'})
            
            # Transformar resposta para formato OpenAI
            response = self.transform_to_openai_format(response_text, tools)
            
            # Logging para depuração apenas em modo verbose
            if os.environ.get("OPENMANUS_DEBUG", "0") == "1":
                logger.info(f"Resposta antes da transformação: {response_text[:100]}...")
                if "tool_calls" in response:
                    logger.info(f"Tool calls após transformação: {response['tool_calls']}")
            
            # Criar um objeto Message-like que corresponda ao esperado
            class MockMessage:
                def __init__(self, data):
                    for key, value in data.items():
                        setattr(self, key, value)

                def __getattr__(self, name):
                    return None
                    
                def model_dump(self):
                    return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

            return MockMessage(response)
                
        except Exception as e:
            logger.error(f"Error in ask_tool: {e}")
            # Retornar um objeto de erro que pode ser manipulado pelo sistema
            error_message = f"Erro ao consultar o modelo: {str(e)}"
            if "out of memory" in str(e):
                error_message = "O Ollama está com problemas de memória. Tente reiniciar o servidor."
            
            return type('ErrorMessage', (), {'content': error_message, 'tool_calls': None, 'role': 'assistant'})
