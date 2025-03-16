"""
Módulo de integração do sistema de chunking no agente Manus.

Este módulo estende as capacidades do agente Manus para processar
conteúdos grandes via chunking.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import re
import json

from app.agent.manus import Manus
from app.agent.content_processor import ContentProcessor
from app.logger import logger
from app.schema import Message


class ChunkingManus(Manus):
    """
    Versão do agente Manus com capacidade de chunking para processar
    conteúdos grandes além do limite de contexto do LLM.
    """
    
    def __init__(self, *args, **kwargs):
        """
        Inicializa o agente ChunkingManus, estendendo o Manus.
        """
        super().__init__(*args, **kwargs)
        
        # Inicializar o processador de conteúdo
        max_token_limit = 6000  # Ajustar conforme necessário
        self.content_processor = ContentProcessor(
            max_token_limit=max_token_limit,
            max_chunk_size=5000,
            overlap_size=500,
            max_total_chunks=5
        )
        
        # Estados de chunking
        self.chunking_active = False
        self.chunking_results = []
    
    async def _handle_special_tool(self, name: str, result: Any, **kwargs):
        """
        Handler estendido para ferramentas especiais, com suporte a chunking.
        
        Args:
            name: Nome da ferramenta.
            result: Resultado da ferramenta.
            kwargs: Argumentos adicionais.
        """
        # Processar resultado grande de browser_use (HTML)
        if name.lower() == 'browser_use' and isinstance(result, str) and len(result) > 10000:
            action = kwargs.get('args', {}).get('action', '')
            
            # Verificar se é um resultado HTML grande (get_html)
            if action == 'get_html':
                logger.info(f"Detectado resultado HTML grande do browser_use ({len(result)} caracteres)")
                
                # Extrair a consulta original do contexto da memória
                query = self._extract_query_from_memory()
                
                # Processar o conteúdo HTML com chunking
                chunked_response = await self.content_processor.process_large_content(
                    content=result,
                    query=query,
                    content_type="html",
                    metadata={"source": "browser_use", "action": "get_html"}
                )
                
                # Substituir o resultado original pelo resultado processado
                result = f"Conteúdo HTML processado com chunking:\n\n{chunked_response}"
                
                logger.info("Conteúdo HTML processado com chunking")
                
            # Verificar se é um resultado de texto grande (get_text)
            elif action == 'get_text' and len(result) > 10000:
                logger.info(f"Detectado resultado de texto grande do browser_use ({len(result)} caracteres)")
                
                # Extrair a consulta original do contexto da memória
                query = self._extract_query_from_memory()
                
                # Processar o conteúdo de texto com chunking
                chunked_response = await self.content_processor.process_large_content(
                    content=result,
                    query=query,
                    content_type="text",
                    metadata={"source": "browser_use", "action": "get_text"}
                )
                
                # Substituir o resultado original pelo resultado processado
                result = f"Conteúdo de texto processado com chunking:\n\n{chunked_response}"
                
                logger.info("Conteúdo de texto processado com chunking")
        
        # Processar resultado grande de python_execute (código)
        elif name.lower() == 'python_execute' and isinstance(result, str) and len(result) > 10000:
            logger.info(f"Detectado resultado grande do python_execute ({len(result)} caracteres)")
            
            # Extrair a consulta original do contexto da memória
            query = self._extract_query_from_memory()
            
            # Processar o conteúdo de código com chunking
            chunked_response = await self.content_processor.process_large_content(
                content=result,
                query=query,
                content_type="code",
                metadata={"source": "python_execute", "language": "python"}
            )
            
            # Substituir o resultado original pelo resultado processado
            result = f"Resultado do código processado com chunking:\n\n{chunked_response}"
            
            logger.info("Resultado de código processado com chunking")
        
        # Continuar o processamento normal da ferramenta
        await super()._handle_special_tool(name, result, **kwargs)
    
    def _extract_query_from_memory(self) -> str:
        """
        Extrai a consulta original do usuário da memória do agente.
        
        Returns:
            String contendo a consulta original ou uma consulta genérica.
        """
        # Tentar encontrar a primeira mensagem do usuário
        for message in self.memory.messages:
            if message.role == "user":
                return message.content if message.content else "Analise este conteúdo"
        
        # Fallback para consulta genérica
        return "Analise o conteúdo e extraia informações relevantes"
    
    async def run(self, prompt: str) -> str:
        """
        Executa o agente com capacidade de chunking, estendendo o método base.
        
        Args:
            prompt: Consulta do usuário.
            
        Returns:
            Resposta do agente.
        """
        # Verificar se o prompt contém um conteúdo grande que deve ser processado diretamente
        if len(prompt) > 10000 and self._is_content_dump(prompt):
            logger.info(f"Detectado conteúdo grande no prompt ({len(prompt)} caracteres)")
            
            # Determinar o tipo de conteúdo
            content_type = self._detect_content_type(prompt)
            
            # Processar o conteúdo com chunking
            chunked_response = self.content_processor.process_large_content(
                content=prompt,
                query="Analise este conteúdo e extraia informações relevantes",
                content_type=content_type
            )
            
            # Adicionar o resultado processado à memória
            self.memory.add_message(Message.user_message(prompt[:1000] + "... [conteúdo truncado]"))
            self.memory.add_message(Message.assistant_message(chunked_response))
            
            return chunked_response
        
        # Executar o agente normalmente para outros casos
        return await super().run(prompt)
    
    def _is_content_dump(self, text: str) -> bool:
        """
        Verifica se o texto parece ser um dump direto de conteúdo.
        
        Args:
            text: Texto a ser verificado.
            
        Returns:
            True se parece ser um dump de conteúdo, False caso contrário.
        """
        # Verificar se é HTML
        if text.strip().startswith('<') and ('</html>' in text or '</body>' in text):
            return True
        
        # Verificar se é JSON
        if (text.strip().startswith('{') and text.strip().endswith('}')) or \
           (text.strip().startswith('[') and text.strip().endswith(']')):
            try:
                json.loads(text.strip())
                return True
            except:
                pass
        
        # Verificar se é código (grande bloco de código)
        code_indicators = [
            'def ', 'class ', 'import ', 'function ', 'public class',
            'const ', 'var ', 'let ', '#include'
        ]
        if any(indicator in text[:1000] for indicator in code_indicators) and '\n' in text[:1000]:
            code_lines = sum(1 for line in text.split('\n') if line.strip())
            if code_lines > 50:  # Se tiver mais de 50 linhas de código
                return True
        
        # Verificar se é um grande bloco de texto sem estrutura de pergunta
        if len(text.split('\n')) > 30 and '?' not in text[:500]:
            return True
        
        return False
    
    def _detect_content_type(self, content: str) -> str:
        """
        Detecta o tipo de conteúdo para processamento adequado.
        
        Args:
            content: Conteúdo a ser analisado.
            
        Returns:
            String indicando o tipo de conteúdo.
        """
        content_trimmed = content.strip()
        
        # Detectar HTML
        if content_trimmed.startswith('<') and ('</html>' in content or '</body>' in content):
            return 'html'
        
        # Detectar JSON
        if (content_trimmed.startswith('{') and content_trimmed.endswith('}')) or \
           (content_trimmed.startswith('[') and content_trimmed.endswith(']')):
            try:
                json.loads(content_trimmed)
                return 'json'
            except:
                pass
        
        # Detectar código
        code_indicators = [
            'def ', 'class ', 'import ', 'function ', 'public class',
            'const ', 'var ', 'let ', 'func ', 'fn ', '#include'
        ]
        
        if any(indicator in content_trimmed for indicator in code_indicators):
            return 'code'
        
        # Padrão para texto
        return 'text'
