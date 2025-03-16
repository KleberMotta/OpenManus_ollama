"""
Módulo de processamento de conteúdo para o OpenManus.

Este módulo integra o sistema de chunking para processamento 
de conteúdos grandes em interações com o LLM.
"""

from typing import List, Dict, Any, Optional, Union
import re
import json

from app.utils.chunking import ChunkProcessor
from app.schema import Message
from app.logger import logger
from app.llm import LLM


class ContentProcessor:
    """
    Processa conteúdos grandes para interações com o LLM,
    aplicando chunking quando necessário.
    """
    
    def __init__(
        self,
        max_token_limit: int = 7000,
        max_chunk_size: int = 6000,
        overlap_size: int = 500,
        max_total_chunks: int = 5
    ):
        """
        Inicializa o processador de conteúdo.
        
        Args:
            max_token_limit: Limite máximo de tokens para o modelo.
            max_chunk_size: Tamanho máximo de cada chunk.
            overlap_size: Tamanho da sobreposição entre chunks.
            max_total_chunks: Número máximo de chunks.
        """
        self.max_token_limit = max_token_limit
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_total_chunks = max_total_chunks
        self.chunk_processor = ChunkProcessor(max_chunk_size, overlap_size, max_total_chunks)
        self.llm = LLM()
    
    async def process_large_content(
        self,
        content: str,
        query: str,
        content_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Processa conteúdo grande dividindo-o em chunks e
        processando cada um sequencialmente.
        
        Args:
            content: Conteúdo a ser processado.
            query: Consulta original do usuário.
            content_type: Tipo de conteúdo ('html', 'code', 'text', 'auto').
            metadata: Metadados associados ao conteúdo.
            system_prompt: Prompt de sistema específico para processamento.
            
        Returns:
            Resposta processada do LLM.
        """
        # Estimar o tamanho em tokens (aproximadamente caracteres/4)
        estimated_tokens = len(content) // 4
        
        # Se o conteúdo for pequeno, processá-lo diretamente
        if estimated_tokens <= self.max_token_limit:
            logger.info(f"Conteúdo pequeno (aprox. {estimated_tokens} tokens), processando diretamente")
            return self._process_single_content(content, query, system_prompt)
        
        # Dividir o conteúdo em chunks
        logger.info(f"Conteúdo grande (aprox. {estimated_tokens} tokens), aplicando chunking")
        chunks = self.chunk_processor.process_content(content, content_type, metadata, query)
        
        # Processar os chunks sequencialmente, mantendo contexto
        return await self._process_content_chunks(chunks, query, system_prompt)
    
    async def _process_single_content(
        self,
        content: str,
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Processa conteúdo pequeno diretamente.
        
        Args:
            content: Conteúdo a ser processado.
            query: Consulta original do usuário.
            system_prompt: Prompt de sistema específico.
            
        Returns:
            Resposta do LLM.
        """
        # Criar prompt para o LLM
        prompt = f"""
        CONTEÚDO:
        {content}
        
        CONSULTA: {query}
        
        Analise o conteúdo acima e responda à consulta de forma completa e precisa.
        """
        
        # Mensagens para o LLM
        messages = []
        
        # Adicionar sistema prompt, se fornecido
        if system_prompt:
            messages.append(Message.system_message(system_prompt))
        
        messages.append(Message.user_message(prompt))
        
        # Obter resposta do LLM
        try:
            response = await self.llm.ask(messages=messages, stream=False)
            return response
        except Exception as e:
            logger.error(f"Erro ao processar conteúdo: {e}")
            return f"Erro ao processar o conteúdo: {str(e)}"
    
    async def _process_content_chunks(
        self,
        chunks: List[Dict[str, Any]],
        query: str,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        Processa os chunks de conteúdo sequencialmente.
        
        Args:
            chunks: Lista de chunks formatados.
            query: Consulta original do usuário.
            system_prompt: Prompt de sistema específico.
            
        Returns:
            Resposta final processada.
        """
        # Resultados intermediários
        intermediate_results = []
        final_result = ""
        
        # Base de contexto para o processamento
        base_context = """
        Você está processando informações em múltiplos chunks. 
        Mantenha o contexto de processamento entre os chunks.
        """
        
        if system_prompt:
            base_context += "\n" + system_prompt
        
        # Processar cada chunk sequencialmente
        for i, chunk in enumerate(chunks):
            # Criar prompt para o chunk atual
            chunk_prompt = f"""
            {chunk['context']}
            
            CONTEÚDO DO CHUNK {i+1}/{chunk['total_chunks']}:
            {chunk['content']}
            
            INSTRUÇÃO: Analise este chunk e extraia informações relevantes para a consulta: '{query}'
            """
            
            if i > 0:
                # Incluir resultados intermediários para continuidade de contexto
                context_summary = self._summarize_intermediate_results(intermediate_results)
                chunk_prompt += f"\n\nCONTEXTO ANTERIOR: {context_summary}"
            
            # Adicionar instruções especiais para o último chunk
            if chunk['is_last']:
                chunk_prompt += f"""
                \n\nEste é o último chunk. 
                Com base em todos os chunks analisados, responda de forma completa e direta à consulta original: '{query}'
                """
            
            # Processar o chunk atual
            try:
                # Criar mensagens para o LLM
                messages = [Message.system_message(base_context)]
                
                # Adicionar resultados intermediários como contexto
                if i > 0:
                    for result in intermediate_results:
                        messages.append(Message.system_message(f"Análise anterior: {result}"))
                
                messages.append(Message.user_message(chunk_prompt))
                
                # Obter resposta do LLM
                response = await self.llm.ask(messages=messages, stream=False)
                
                # Armazenar resultado intermediário
                if not chunk['is_last']:
                    intermediate_results.append(response)
                else:
                    final_result = response
                
                logger.info(f"Chunk {i+1}/{chunk['total_chunks']} processado com sucesso")
            
            except Exception as e:
                logger.error(f"Erro ao processar chunk {i+1}: {e}")
                return f"Erro ao processar o conteúdo (chunk {i+1}): {str(e)}"
        
        # Se não há resultado final (normalmente devido a erro), usar o último resultado intermediário
        if not final_result and intermediate_results:
            final_result = await self._generate_final_response(intermediate_results, query)
        
        return final_result
    
    def _summarize_intermediate_results(self, results: List[str]) -> str:
        """
        Sumariza os resultados intermediários para manter contexto entre chunks.
        
        Args:
            results: Lista de resultados intermediários.
            
        Returns:
            Sumário dos resultados.
        """
        if not results:
            return ""
        
        # Para poucos resultados, concatená-los diretamente
        if len(results) <= 2:
            return " ".join(results)
        
        # Para muitos resultados, criar um resumo mais compacto
        return f"Informações de {len(results)} chunks anteriores: " + " ".join(results[-2:])
    
    async def _generate_final_response(self, results: List[str], query: str) -> str:
        """
        Gera uma resposta final a partir dos resultados intermediários.
        
        Args:
            results: Lista de resultados intermediários.
            query: Consulta original do usuário.
            
        Returns:
            Resposta final gerada.
        """
        # Criar prompt para síntese final
        synthesis_prompt = f"""
        Com base nas seguintes análises parciais:
        
        {" ".join(results)}
        
        Responda de forma completa e direta à consulta original: '{query}'
        """
        
        # Obter síntese final do LLM
        try:
            messages = [Message.user_message(synthesis_prompt)]
            response = await self.llm.ask(messages=messages, stream=False)
            return response
        except Exception as e:
            logger.error(f"Erro ao gerar resposta final: {e}")
            # Fallback para o último resultado intermediário
            return results[-1] if results else f"Erro ao gerar resposta final: {str(e)}"


# Função auxiliar para estimar tokens (heurística simples)
def estimate_tokens(text: str) -> int:
    """
    Estima o número de tokens em um texto (heurística simples).
    
    Args:
        text: Texto para estimar.
        
    Returns:
        Número aproximado de tokens.
    """
    # Regras heurísticas para estimar tokens:
    # 1. Palavras comuns: aproximadamente 1 token por palavra
    # 2. Números, pontuação: aproximadamente 1 token por 2-3 caracteres
    # 3. Código, palavras técnicas: aproximadamente 1.5 tokens por palavra
    
    # Heurística simples: aproximadamente 4 caracteres por token em média
    return len(text) // 4
