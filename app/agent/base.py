from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from typing import List, Optional

from pydantic import BaseModel, Field, model_validator

from app.llm import LLM
from app.logger import logger
from app.schema import AgentState, Memory, Message, ROLE_TYPE


class BaseAgent(BaseModel, ABC):
    """Abstract base class for managing agent state and execution.

    Provides foundational functionality for state transitions, memory management,
    and a step-based execution loop. Subclasses must implement the `step` method.
    """

    # Core attributes
    name: str = Field(..., description="Unique name of the agent")
    description: Optional[str] = Field(None, description="Optional agent description")

    # Prompts
    system_prompt: Optional[str] = Field(
        None, description="System-level instruction prompt"
    )
    next_step_prompt: Optional[str] = Field(
        None, description="Prompt for determining next action"
    )

    # Dependencies
    llm: LLM = Field(default_factory=LLM, description="Language model instance")
    memory: Memory = Field(default_factory=Memory, description="Agent's memory store")
    state: AgentState = Field(
        default=AgentState.IDLE, description="Current agent state"
    )

    # Execução e controle de loop
    max_consecutive_errors: int = Field(default=3, description="Máximo de erros consecutivos antes de terminar")
    error_count: int = Field(default=0, description="Contador de erros consecutivos")
    current_step: int = Field(default=0, description="Current step in execution")
    duplicate_threshold: int = 2
    stuck_count: int = Field(default=0, description="Number of times agent got stuck")

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"  # Allow extra fields for flexibility in subclasses

    @model_validator(mode="after")
    def initialize_agent(self) -> "BaseAgent":
        """Initialize agent with default settings if not provided."""
        if self.llm is None or not isinstance(self.llm, LLM):
            self.llm = LLM(config_name=self.name.lower())
        if not isinstance(self.memory, Memory):
            self.memory = Memory()
        return self

    @asynccontextmanager
    async def state_context(self, new_state: AgentState):
        """Context manager for safe agent state transitions.

        Args:
            new_state: The state to transition to during the context.

        Yields:
            None: Allows execution within the new state.

        Raises:
            ValueError: If the new_state is invalid.
        """
        if not isinstance(new_state, AgentState):
            raise ValueError(f"Invalid state: {new_state}")

        previous_state = self.state
        self.state = new_state
        try:
            yield
        except Exception as e:
            self.state = AgentState.ERROR  # Transition to ERROR on failure
            raise e
        finally:
            self.state = previous_state  # Revert to previous state

    def update_memory(
        self,
        role: ROLE_TYPE, # type: ignore
        content: str,
        **kwargs,
    ) -> None:
        """Add a message to the agent's memory.

        Args:
            role: The role of the message sender (user, system, assistant, tool).
            content: The message content.
            **kwargs: Additional arguments (e.g., tool_call_id for tool messages).

        Raises:
            ValueError: If the role is unsupported.
        """
        message_map = {
            "user": Message.user_message,
            "system": Message.system_message,
            "assistant": Message.assistant_message,
            "tool": lambda content, **kw: Message.tool_message(content, **kw),
        }

        if role not in message_map:
            raise ValueError(f"Unsupported message role: {role}")

        msg_factory = message_map[role]
        msg = msg_factory(content, **kwargs) if role == "tool" else msg_factory(content)
        self.memory.add_message(msg)

    async def run(self, request: Optional[str] = None) -> str:
        """Execute the agent's main loop asynchronously.

        Args:
            request: Optional initial user request to process.

        Returns:
            A string summarizing the execution results.

        Raises:
            RuntimeError: If the agent is not in IDLE state at start.
        """
        if self.state != AgentState.IDLE:
            raise RuntimeError(f"Cannot run agent from state: {self.state}")

        if request:
            self.update_memory("user", request)

        results: List[str] = []
        self.current_step = 0
        self.error_count = 0
        
        async with self.state_context(AgentState.RUNNING):
            # Executar até que o agente finalize explicitamente
            # ou seja detectado um problema grave
            while self.state != AgentState.FINISHED:
                self.current_step += 1
                
                # Registrar o passo para fins de log
                if hasattr(self, 'get_current_step_description'):
                    step_description = self.get_current_step_description()
                    logger.info(f"Executando {step_description} (passo {self.current_step})")
                else:
                    logger.info(f"Executando passo {self.current_step}")
                
                try:
                    # Executar o passo 
                    step_result = await self.step()
                    
                    # Verificar sinais de que o agente está preso em um loop
                    if self.is_stuck():
                        self.handle_stuck_state()
                        
                        # Verificação mais severa para reset de contexto após várias repetições
                        if self.stuck_count >= 5:
                            logger.warning("Agente severamente preso em loop. Redefinindo contexto.")
                            self.reset_context()
                            
                            # Verificar se devemos encerrar após muitas tentativas
                            if self.stuck_count >= 7:
                                logger.error("Agente preso em loop severo após múltiplas tentativas. Encerrando execução.")
                                
                                # Extrair resultados úteis de ferramentas já executadas
                                tool_results = [msg.content for msg in self.memory.messages 
                                              if msg.role == "tool" and msg.content and len(msg.content) > 50]
                                
                                if tool_results:
                                    # Se obtivemos resultados de ferramentas, usar o mais relevante como resposta
                                    web_results = [res for res in tool_results if "web_search" in res.lower() 
                                                   or "Observed output of cmd `web_search`" in res]
                                    if web_results:
                                        results.append(web_results[-1])
                                        logger.info("Usando resultado de pesquisa web como resposta final")
                                    else:
                                        results.append(tool_results[-1])
                                else:
                                    results.append("Execução encerrada: detectado loop de execução severo.")
                                
                                self.state = AgentState.FINISHED
                                break
                    else:
                        # Resetar contador de erros se não estivermos presos
                        self.error_count = 0
                
                    # Adicionar o resultado ao histórico se não for relacionado ao terminate
                    if not ("terminate" in step_result and "{" in step_result and "}" in step_result):
                        # Remove ou substitui JSON relacionado ao terminate se existir
                        import re
                        clean_result = re.sub(r'\{\s*["\']tool_name["\']\s*:\s*["\']terminate["\'].*?\}', "Task completed.", step_result)
                        clean_result = re.sub(r'\{\s*["\']name["\']\s*:\s*["\']terminate["\'].*?\}', "Task completed.", clean_result)
                        
                        # Só adiciona se não for JSON puro de terminate
                        if not (clean_result.strip().startswith('{') and clean_result.strip().endswith('}') and "terminate" in clean_result):
                            results.append(clean_result)
                            
                except Exception as e:
                    error_msg = f"Erro durante a execução: {str(e)}"
                    logger.error(error_msg)
                    results.append(f"Step {self.current_step} Error: {error_msg}")
                    
                    # Incrementar contador de erros consecutivos
                    self.error_count += 1
                    
                    # Verificar se atingimos o limite de erros
                    if self.error_count >= self.max_consecutive_errors:
                        logger.error(f"Limite de erros consecutivos atingido ({self.max_consecutive_errors}). Encerrando execução.")
                        results.append("Execução encerrada devido a múltiplos erros consecutivos.")
                        self.state = AgentState.FINISHED
                        break
                
                # Verificar se há um limite de tempo de execução (implementação opcional)
                if hasattr(self, 'check_execution_timeout') and self.check_execution_timeout():
                    logger.warning("Tempo limite de execução atingido. Encerrando.")
                    results.append("Execução encerrada: tempo limite atingido.")
                    self.state = AgentState.FINISHED
                    break
                    
                # Verificação de segurança - limitar número total de passos para evitar loops infinitos
                # mas com um limite muito alto para permitir execuções extensas legítimas
                if self.current_step > 100:  # Limite bem alto para casos extremos
                    logger.warning("Número máximo de passos atingido (100). Verificando se há progresso real.")
                    
                    # Verificar se há progresso real nas últimas 10 ações
                    if self.is_making_progress():
                        logger.info("Há progresso real detectado, continuando execução.")
                    else:
                        logger.error("Nenhum progresso real detectado após 100 passos. Encerrando execução.")
                        results.append("Execução encerrada: muitos passos sem progresso detectável.")
                        self.state = AgentState.FINISHED
                        break

            # Limpar recursos ao finalizar
            if hasattr(self, 'cleanup_resources'):
                await self.cleanup_resources()

        # Filtrar resultados para remover menções a terminate e JSON
        filtered_results = []
        for result in results:
            if "Observed output of cmd `terminate`" not in result:
                filtered_results.append(result)
                
        # Se o último resultado não for relacionado ao término, adicione uma mensagem de conclusão
        if filtered_results and not any("Task completed" in r for r in filtered_results[-3:]):
            filtered_results.append("Task completed successfully.")
            
        return "\n".join(filtered_results) if filtered_results else "Task completed."

    def reset_context(self):
        """Reseta o contexto mantendo apenas mensagens essenciais"""
        # Manter apenas a primeira mensagem de sistema, última mensagem do usuário
        # e os resultados mais relevantes de ferramentas
        preserved_messages = []
        last_user_msg = None
        important_tool_results = []
        
        # Primeiro, identificar mensagens importantes
        for msg in self.memory.messages:
            if msg.role == "system" and not preserved_messages:
                preserved_messages.append(msg)
            elif msg.role == "user":
                last_user_msg = msg
            elif msg.role == "tool" and msg.content and len(msg.content) > 100:
                # Manter resultados de ferramentas significativos
                # Especialmente HTML ou resultados de busca
                if any(term in msg.name.lower() for term in ["web_search", "browser_use"]):
                    important_tool_results.append(msg)
        
        # Construir a nova memória com mensagens essenciais
        reset_messages = preserved_messages.copy()
        
        # Adicionar até 2 resultados importantes de ferramentas (os mais recentes)
        if important_tool_results:
            reset_messages.extend(important_tool_results[-2:])
        
        # Adicionar a última mensagem do usuário
        if last_user_msg:
            reset_messages.append(last_user_msg)
            
        # Redefinir a memória com as mensagens preservadas
        self.memory.messages = reset_messages
        
        # Adicionar nova mensagem de sistema para orientar
        self.update_memory(
            "system", 
            "O contexto foi redefinido devido a dificuldades de processamento. " +
            "Analise as informações existentes e fornecer um resumo claro e direto ao usuário " +
            "baseado nos dados já coletados. Não peça mais instruções."
        )
    
    def is_making_progress(self) -> bool:
        """Verifica se há progresso real nas últimas ações do agente"""
        # Implementação básica - verificar se há variedade nas ações recentes
        if hasattr(self, 'tool_calls') and len(getattr(self, 'tool_calls', [])) > 10:
            # Obter as últimas 10 chamadas de ferramentas
            recent_tools = [call.get('function', {}).get('name', '') if isinstance(call, dict) else 
                           call.function.name if hasattr(call, 'function') else '' 
                           for call in getattr(self, 'tool_calls', [])[-10:]]
            
            # Contar quantas ferramentas diferentes foram usadas
            unique_tools = len(set(recent_tools))
            
            # Se usou pelo menos 3 ferramentas diferentes, provavelmente está progredindo
            return unique_tools >= 3
        
        # Se não puder verificar, assumir que está progredindo
        return True

    @abstractmethod
    async def step(self) -> str:
        """Execute a single step in the agent's workflow.

        Must be implemented by subclasses to define specific behavior.
        """

    def handle_stuck_state(self):
        """Handle stuck state by adding a prompt to change strategy"""
        self.stuck_count += 1
        
        # Diferentes mensagens baseadas na quantidade de loops detectados
        if self.stuck_count >= 4:
            stuck_prompt = "You are stuck in a repetitive loop. STOP asking for instructions. Provide a DIRECT ANSWER based on the information you've already collected."
            
            # Quando estiver severamente preso, extrair informações úteis já obtidas
            tool_outputs = [msg.content for msg in self.memory.messages 
                          if msg.role == "tool" and msg.content and len(msg.content) > 50]
            
            # Se tiver resultados de ferramentas, adicionar mensagem de sistema explícita
            if tool_outputs:
                last_output = tool_outputs[-1]
                
                # Adicionar mensagem de sistema para forçar conclusão com dados disponíveis
                self.memory.add_message(Message.system_message(
                    "The system has detected a loop. Ignore any confusion and provide a direct answer " +
                    "based on this information you've already collected:\n\n" + last_output + "\n\n" +
                    "Summarize this information as your final answer now. DO NOT ask for more instructions."
                ))
        elif self.stuck_count >= 3:
            stuck_prompt = "You are repeating yourself. Change your approach completely. Do not restate the problem - provide a direct solution."
        else:
            stuck_prompt = "Observed duplicate responses. Consider new strategies and avoid repeating ineffective paths already attempted."
            
        self.next_step_prompt = f"{stuck_prompt}\n{self.next_step_prompt}"
        logger.warning(f"Agent detected stuck state. Added prompt: {stuck_prompt}")

    def is_stuck(self) -> bool:
        """Check if the agent is stuck in a loop by detecting duplicate content or patterns"""
        if len(self.memory.messages) < 3:
            return False

        # Verificar as últimas mensagens do assistente
        assistant_messages = [msg for msg in self.memory.messages if msg.role == "assistant" and msg.content]
        if len(assistant_messages) < 2:
            return False
            
        last_msgs = assistant_messages[-3:]
        
        # Detectar padrão de 'aguardando instruções'
        waiting_phrases = [
            "i'm ready", "estou pronto", "aguardando", "waiting for", "provide the task", "what would you like",
            "how can i help", "como posso ajudar", "i can help", "i'll help", "ready for your",
            "let's begin", "let me know", "please provide", "please tell me", "what do you want"
        ]
        
        # Detectar se as últimas mensagens são do tipo "aguardando instruções"
        waiting_count = 0
        for msg in last_msgs:
            if msg.content and any(phrase in msg.content.lower() for phrase in waiting_phrases):
                waiting_count += 1
                
        # Se 2 ou mais das últimas 3 mensagens são do tipo "aguardando instruções",
        # o agente está claramente preso esperando input, quando deveria continuar o processamento
        if waiting_count >= 2:
            return True
            
        # Verificar duplicação exata ou alta similaridade
        last_message = assistant_messages[-1]
        duplicate_count = 0
        similar_count = 0
        
        # Só comparar com as últimas 5 mensagens para eficiência
        for msg in reversed(assistant_messages[:-1][-5:]):
            # Verificação de duplicação exata
            if msg.content == last_message.content:
                duplicate_count += 1
                continue
                
            # Verificação de alta similaridade
            if len(msg.content) > 20 and len(last_message.content) > 20:
                # Similaridade de início e fim
                if (msg.content[:30] == last_message.content[:30] or 
                    msg.content[-30:] == last_message.content[-30:]):
                    similar_count += 1
                    continue
                    
                # Similaridade de frases-chave
                key_phrases = [phrase for phrase in waiting_phrases if 
                              phrase in msg.content.lower() and phrase in last_message.content.lower()]
                if len(key_phrases) >= 2:
                    similar_count += 1
                    
        return duplicate_count >= 1 or similar_count >= 1 or waiting_count >= 2

    @property
    def messages(self) -> List[Message]:
        """Retrieve a list of messages from the agent's memory."""
        return self.memory.messages

    @messages.setter
    def messages(self, value: List[Message]):
        """Set the list of messages in the agent's memory."""
        self.memory.messages = value