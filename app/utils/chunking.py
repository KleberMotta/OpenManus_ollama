"""
Módulo de chunking para o OpenManus.

Este módulo implementa estratégias de chunking para processar conteúdos grandes
para LLMs com janelas de contexto limitadas.
"""

import re
import html
from typing import List, Dict, Tuple, Optional, Union, Any
from bs4 import BeautifulSoup
import json
import uuid
from app.logger import logger


class ChunkStrategy:
    """Estratégia base para chunking de conteúdo"""
    
    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 500):
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
    
    def split(self, content: str) -> List[str]:
        """Divide o conteúdo em chunks"""
        raise NotImplementedError("Método deve ser implementado por subclasses")


class FixedSizeChunkStrategy(ChunkStrategy):
    """Estratégia de chunking por tamanho fixo com sobreposição"""
    
    def split(self, content: str) -> List[str]:
        """
        Divide o conteúdo em chunks de tamanho fixo com sobreposição.
        
        Args:
            content: Conteúdo a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        chunks = []
        for i in range(0, len(content), self.max_chunk_size - self.overlap_size):
            end = min(i + self.max_chunk_size, len(content))
            chunk = content[i:end]
            chunks.append(chunk)
        
        return chunks


class RecursiveChunkStrategy(ChunkStrategy):
    """Estratégia de chunking recursivo baseado em separadores"""
    
    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 500):
        super().__init__(max_chunk_size, overlap_size)
        # Lista de separadores em ordem de preferência
        self.separators = ["\n\n", "\n", ". ", ", ", " ", ""]
    
    def split(self, content: str) -> List[str]:
        """
        Divide o conteúdo recursivamente usando separadores hierárquicos.
        
        Args:
            content: Conteúdo a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        # Se o conteúdo for pequeno o suficiente, retorna-o inteiro
        if len(content) <= self.max_chunk_size:
            return [content]
        
        # Tenta dividir o conteúdo usando os separadores em ordem
        chunks = []
        for separator in self.separators:
            if not separator:  # Último recurso: dividir por caractere
                return FixedSizeChunkStrategy(self.max_chunk_size, self.overlap_size).split(content)
            
            # Dividir o conteúdo pelo separador atual
            splits = content.split(separator)
            
            # Se o separador não dividiu o conteúdo, tente o próximo
            if len(splits) == 1:
                continue
            
            # Recombinar splits para respeitar o tamanho máximo e sobreposição
            good_splits = []
            current_chunk = []
            current_length = 0
            
            for split in splits:
                split_with_sep = split + (separator if separator else "")
                split_length = len(split_with_sep)
                
                # Se o split for maior que o tamanho máximo, recursivamente divida-o
                if split_length > self.max_chunk_size:
                    # Adicionar o chunk atual, se existir
                    if current_chunk:
                        good_splits.append(separator.join(current_chunk))
                        current_chunk = []
                        current_length = 0
                    
                    # Recursivamente dividir o split grande
                    good_splits.extend(self.split(split_with_sep))
                    continue
                
                # Se adicionar este split exceder o tamanho máximo, iniciar novo chunk
                if current_length + split_length > self.max_chunk_size:
                    good_splits.append(separator.join(current_chunk))
                    current_chunk = [split]
                    current_length = split_length
                else:
                    current_chunk.append(split)
                    current_length += split_length
            
            # Não esquecer do último chunk
            if current_chunk:
                good_splits.append(separator.join(current_chunk))
            
            # Se conseguimos dividir bem, retornamos os chunks
            if good_splits:
                # Adicionar sobreposição entre chunks
                chunks = []
                for i, chunk in enumerate(good_splits):
                    # Adicionar pedaço do próximo chunk (se não for o último)
                    if i < len(good_splits) - 1 and self.overlap_size > 0:
                        next_chunk = good_splits[i + 1]
                        overlap = min(self.overlap_size, len(next_chunk))
                        chunk_with_overlap = chunk + next_chunk[:overlap]
                        chunks.append(chunk_with_overlap)
                    else:
                        chunks.append(chunk)
                
                return chunks
        
        # Se nenhum separador funcionou, fallback para chunking de tamanho fixo
        return FixedSizeChunkStrategy(self.max_chunk_size, self.overlap_size).split(content)


class SemanticChunkStrategy(ChunkStrategy):
    """Estratégia de chunking que preserva unidades semânticas"""
    
    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 500):
        super().__init__(max_chunk_size, overlap_size)
        # Define pontos de divisão semântica com prioridade
        self.semantic_boundaries = [
            # Fronteiras de documento
            r"\n#{1,6}\s+",  # Cabeçalhos markdown/documento
            r"\n\s*<h[1-6]>",  # Cabeçalhos HTML
            
            # Fronteiras de parágrafo
            r"\n\s*\n",  # Parágrafos
            
            # Fronteiras de sentença
            r"(?<=[.!?])\s+",  # Fim de sentença
            
            # Fronteiras lógicas em código
            r"\n\s*(?:def|class)\s+",  # Definições Python
            r"\n\s*(?:function|class|const|let|var)\s+",  # Definições JavaScript
            r"\n\s*(?:public|private|protected|class|void)\s+",  # Definições Java/C#
        ]
    
    def split(self, content: str) -> List[str]:
        """
        Divide o conteúdo preservando unidades semânticas.
        
        Args:
            content: Conteúdo a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        # Se o conteúdo for pequeno o suficiente, retorna-o inteiro
        if len(content) <= self.max_chunk_size:
            return [content]
        
        # Tentativa de divisão semântica baseada nas fronteiras
        for boundary_pattern in self.semantic_boundaries:
            # Encontrar todos os pontos de divisão
            splits = []
            last_end = 0
            
            for match in re.finditer(boundary_pattern, content):
                start = match.start()
                # Adicionar o texto até o ponto de divisão
                if start > last_end:
                    splits.append(content[last_end:start])
                last_end = match.end()
            
            # Adicionar o último pedaço
            if last_end < len(content):
                splits.append(content[last_end:])
            
            # Se o padrão não dividiu o conteúdo, tente o próximo
            if len(splits) <= 1:
                continue
            
            # Processar os splits para criar chunks
            chunks = []
            current_chunk = ""
            
            for split in splits:
                # Se o split for maior que o tamanho máximo, dividi-lo recursivamente
                if len(split) > self.max_chunk_size:
                    # Salvar o chunk atual se não estiver vazio
                    if current_chunk:
                        chunks.append(current_chunk)
                        current_chunk = ""
                    
                    # Recursivamente dividir o split grande
                    sub_chunks = self.split(split)
                    chunks.extend(sub_chunks)
                    continue
                
                # Se adicionar este split exceder o tamanho máximo, iniciar novo chunk
                if len(current_chunk) + len(split) > self.max_chunk_size:
                    chunks.append(current_chunk)
                    current_chunk = split
                else:
                    current_chunk += split
            
            # Não esquecer do último chunk
            if current_chunk:
                chunks.append(current_chunk)
            
            # Se conseguimos dividir o conteúdo adequadamente, adicionar sobreposição e retornar
            if chunks:
                # Adicionar sobreposição entre chunks
                chunks_with_overlap = []
                for i, chunk in enumerate(chunks):
                    # Se não for o último chunk, adicionar parte do próximo como sobreposição
                    if i < len(chunks) - 1 and self.overlap_size > 0:
                        next_chunk = chunks[i + 1]
                        overlap_size = min(self.overlap_size, len(next_chunk))
                        chunk_with_overlap = chunk + next_chunk[:overlap_size]
                        chunks_with_overlap.append(chunk_with_overlap)
                    else:
                        chunks_with_overlap.append(chunk)
                
                return chunks_with_overlap
        
        # Se nenhuma divisão semântica funcionou, cair para recursiva
        return RecursiveChunkStrategy(self.max_chunk_size, self.overlap_size).split(content)


class HtmlChunkStrategy(ChunkStrategy):
    """Estratégia específica para chunking de conteúdo HTML"""
    
    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 500):
        super().__init__(max_chunk_size, overlap_size)
    
    def split(self, content: str) -> List[str]:
        """
        Divide o conteúdo HTML respeitando a estrutura DOM.
        
        Args:
            content: Conteúdo HTML a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        try:
            # Tentar analisar o HTML
            soup = BeautifulSoup(content, 'html.parser')
            
            # Identificar blocos principais
            content_blocks = []
            
            # Priorizar elementos que normalmente contêm conteúdo principal
            for selector in ['main', 'article', 'section', 'div.content', 'div.main', 'div.body']:
                if '.' in selector:
                    tag_name, class_name = selector.split('.')
                    blocks = soup.find_all(tag_name, class_=class_name)
                else:
                    blocks = soup.find_all(selector)
                
                if blocks:
                    content_blocks.extend(blocks)
                    break
            
            # Se não encontrar blocos específicos, usar o corpo inteiro
            if not content_blocks:
                if soup.body:
                    content_blocks = [soup.body]
                else:
                    content_blocks = [soup]
            
            # Dividir os blocos em chunks semânticos
            chunks = []
            
            for block in content_blocks:
                # Dividir por cabeçalhos ou outros elementos semânticos
                sections = self._split_by_headers(block)
                
                current_chunk = ""
                for section in sections:
                    section_html = str(section)
                    
                    # Se a seção for maior que o tamanho máximo, dividi-la
                    if len(section_html) > self.max_chunk_size:
                        if current_chunk:
                            chunks.append(current_chunk)
                            current_chunk = ""
                        
                        # Usar estratégia recursiva para dividir seções grandes
                        sub_chunks = RecursiveChunkStrategy(self.max_chunk_size, self.overlap_size).split(section_html)
                        chunks.extend(sub_chunks)
                    else:
                        # Se adicionar esta seção exceder o tamanho máximo, iniciar novo chunk
                        if len(current_chunk) + len(section_html) > self.max_chunk_size:
                            chunks.append(current_chunk)
                            current_chunk = section_html
                        else:
                            current_chunk += section_html
                
                # Não esquecer do último chunk
                if current_chunk:
                    chunks.append(current_chunk)
            
            # Se conseguimos dividir o HTML, retornar os chunks
            if chunks:
                return chunks
        except Exception as e:
            logger.error(f"Erro ao processar HTML: {e}")
        
        # Fallback para chunking recursivo se algo der errado
        return RecursiveChunkStrategy(self.max_chunk_size, self.overlap_size).split(content)
    
    def _split_by_headers(self, element) -> List:
        """
        Divide o conteúdo HTML em seções baseadas em cabeçalhos.
        
        Args:
            element: Elemento HTML a ser dividido.
            
        Returns:
            Lista de elementos HTML, cada um representando uma seção.
        """
        sections = []
        current_section = BeautifulSoup("", "html.parser")
        
        # Iterar pelos filhos do elemento
        for child in element.children:
            # Se for um cabeçalho, iniciar nova seção
            if hasattr(child, 'name') and child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                # Salvar seção atual se não estiver vazia
                if str(current_section).strip():
                    sections.append(current_section)
                
                # Iniciar nova seção com o cabeçalho
                current_section = BeautifulSoup("", "html.parser")
                current_section.append(child)
            else:
                # Adicionar à seção atual
                if hasattr(child, 'name') or str(child).strip():
                    current_section.append(child)
        
        # Não esquecer da última seção
        if str(current_section).strip():
            sections.append(current_section)
        
        # Se não conseguiu encontrar seções, retornar o elemento inteiro
        if not sections:
            sections = [element]
        
        return sections


class CodeChunkStrategy(ChunkStrategy):
    """Estratégia específica para chunking de código-fonte"""
    
    def __init__(self, max_chunk_size: int = 8000, overlap_size: int = 500):
        super().__init__(max_chunk_size, overlap_size)
        # Padrões para diferentes linguagens
        self.language_patterns = {
            'python': r'(\n|^)(?:def\s+\w+|class\s+\w+|@\w+|if\s+__name__\s*==)',
            'javascript': r'(\n|^)(?:function\s+\w+|class\s+\w+|const\s+\w+\s*=\s*(?:function|\(.*?\)\s*=>)|let\s+\w+\s*=)',
            'java': r'(\n|^)(?:public\s+class|private\s+class|protected\s+class|class\s+\w+|public\s+\w+\s+\w+\s*\()',
            'generic': r'(\n|^)(?:function\s+\w+|class\s+\w+|\w+\s*\(.*?\)\s*\{|\w+\s*=\s*function|\w+\s*=>\s*\{)'
        }
    
    def split(self, content: str) -> List[str]:
        """
        Divide o código respeitando estruturas como funções e classes.
        
        Args:
            content: Código a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        # Detectar linguagem para usar o padrão apropriado
        language = self._detect_language(content)
        pattern = self.language_patterns.get(language, self.language_patterns['generic'])
        
        # Encontrar pontos de definição na estrutura do código
        matches = list(re.finditer(pattern, content))
        
        # Se não encontrou pontos de definição, usar chunking por linhas
        if not matches:
            return self._split_by_lines(content)
        
        chunks = []
        
        # Dividir o código pelas definições encontradas
        for i, match in enumerate(matches):
            start = match.start()
            # Ajustar o início para incluir a linha inteira
            if start > 0 and content[start-1] != '\n':
                start = content.rfind('\n', 0, start) + 1
            
            # Para o último match, o fim é o final do código
            if i == len(matches) - 1:
                end = len(content)
            else:
                # O fim é o início do próximo match
                end = matches[i + 1].start()
                if end > 0 and content[end-1] != '\n':
                    end = content.rfind('\n', 0, end) + 1
            
            # Se o chunk for muito grande, dividi-lo por linhas
            if end - start > self.max_chunk_size:
                sub_chunks = self._split_by_lines(content[start:end])
                chunks.extend(sub_chunks)
            else:
                chunks.append(content[start:end])
        
        return chunks
    
    def _split_by_lines(self, content: str) -> List[str]:
        """
        Divide o código por linhas, mantendo a estrutura sintática.
        
        Args:
            content: Código a ser dividido.
            
        Returns:
            Lista de strings, cada uma representando um chunk.
        """
        lines = content.split('\n')
        chunks = []
        current_chunk = []
        current_length = 0
        
        for line in lines:
            line_length = len(line) + 1  # +1 para o '\n'
            
            # Se a linha for maior que o tamanho máximo, dividir caractere por caractere
            if line_length > self.max_chunk_size:
                # Adicionar o chunk atual se não estiver vazio
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_length = 0
                
                # Dividir a linha grande em pedaços
                for i in range(0, len(line), self.max_chunk_size - 1):
                    end = min(i + self.max_chunk_size - 1, len(line))
                    chunks.append(line[i:end])
            else:
                # Se adicionar esta linha exceder o tamanho máximo, iniciar novo chunk
                if current_length + line_length > self.max_chunk_size:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = [line]
                    current_length = line_length
                else:
                    current_chunk.append(line)
                    current_length += line_length
        
        # Não esquecer do último chunk
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return chunks
    
    def _detect_language(self, code: str) -> str:
        """
        Detecta a linguagem de programação do código.
        
        Args:
            code: Código a ser analisado.
            
        Returns:
            String representando a linguagem detectada.
        """
        indicators = {
            'python': ['def ', 'class ', 'import ', 'from ', 'if __name__'],
            'javascript': ['function ', 'const ', 'let ', 'var ', '=>'],
            'java': ['public class', 'private ', 'protected ', 'package ', 'import java'],
        }
        
        code_lower = code.lower()
        matches = {lang: 0 for lang in indicators}
        
        for lang, patterns in indicators.items():
            for pattern in patterns:
                if pattern.lower() in code_lower:
                    matches[lang] += 1
        
        # Retornar a linguagem com mais correspondências
        best_match = max(matches.items(), key=lambda x: x[1])
        return best_match[0] if best_match[1] > 0 else 'generic'


class ChunkProcessor:
    """
    Gerenciador de chunking que seleciona e aplica a estratégia adequada
    para diferentes tipos de conteúdo.
    """
    
    def __init__(
        self, 
        max_chunk_size: int = 8000, 
        overlap_size: int = 500,
        max_total_chunks: int = 10
    ):
        """
        Inicializa o processador de chunks.
        
        Args:
            max_chunk_size: Tamanho máximo de cada chunk em caracteres.
            overlap_size: Tamanho da sobreposição entre chunks adjacentes.
            max_total_chunks: Número máximo de chunks a serem gerados.
        """
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.max_total_chunks = max_total_chunks
        
        # Inicializar estratégias de chunking
        self.strategies = {
            'fixed': FixedSizeChunkStrategy(max_chunk_size, overlap_size),
            'recursive': RecursiveChunkStrategy(max_chunk_size, overlap_size),
            'semantic': SemanticChunkStrategy(max_chunk_size, overlap_size),
            'html': HtmlChunkStrategy(max_chunk_size, overlap_size),
            'code': CodeChunkStrategy(max_chunk_size, overlap_size),
        }
    
    def process_content(
        self,
        content: str,
        content_type: str = "auto",
        metadata: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Processa o conteúdo, dividindo-o em chunks usando a estratégia adequada.
        
        Args:
            content: O conteúdo a ser processado.
            content_type: Tipo de conteúdo ('html', 'code', 'text', 'auto').
            metadata: Metadados associados ao conteúdo.
            query: Consulta original do usuário.
            
        Returns:
            Lista de dicionários, cada um contendo um chunk com metadados.
        """
        if not content:
            logger.warning("Tentativa de processar conteúdo vazio")
            return [{"content": "", "metadata": metadata or {}, "is_last": True}]
        
        # Se o conteúdo for menor que o tamanho máximo, não dividir
        if len(content) <= self.max_chunk_size:
            logger.info(f"Conteúdo não requer chunking (tamanho: {len(content)} caracteres)")
            return [{
                "content": content,
                "metadata": metadata or {},
                "is_last": True,
                "chunk_id": str(uuid.uuid4())[:8],
                "total_chunks": 1,
                "chunk_index": 0
            }]
        
        # Detectar tipo de conteúdo se for 'auto'
        if content_type == "auto":
            content_type = self._detect_content_type(content)
            logger.info(f"Tipo de conteúdo detectado: {content_type}")
        
        # Selecionar a estratégia adequada
        strategy_name = {
            'html': 'html',
            'code': 'code',
            'json': 'recursive',
            'text': 'semantic'
        }.get(content_type, 'semantic')
        
        # Obter a estratégia selecionada
        strategy = self.strategies[strategy_name]
        
        # Dividir o conteúdo em chunks
        chunks = strategy.split(content)
        
        # Limitar o número máximo de chunks
        if len(chunks) > self.max_total_chunks:
            logger.warning(f"Número de chunks ({len(chunks)}) excede o máximo ({self.max_total_chunks}). Truncando.")
            chunks = chunks[:self.max_total_chunks]
        
        # Formatar os chunks com metadados
        return self._format_chunks_with_metadata(chunks, metadata, query)
    
    def _detect_content_type(self, content: str) -> str:
        """
        Detecta automaticamente o tipo de conteúdo.
        
        Args:
            content: Conteúdo a ser analisado.
            
        Returns:
            String representando o tipo de conteúdo.
        """
        content_trimmed = content.strip()
        
        # Detectar HTML
        if content_trimmed.startswith('<!DOCTYPE html>') or content_trimmed.startswith('<html') or \
           ('<head>' in content_trimmed and '</head>' in content_trimmed) or \
           ('<body>' in content_trimmed and '</body>' in content_trimmed):
            return 'html'
        
        # Detectar JSON
        if (content_trimmed.startswith('{') and content_trimmed.endswith('}')) or \
           (content_trimmed.startswith('[') and content_trimmed.endswith(']')):
            try:
                json.loads(content_trimmed)
                return 'json'
            except json.JSONDecodeError:
                pass
        
        # Detectar código
        code_indicators = [
            'def ', 'class ', 'import ', 'function ', 'public class',
            'const ', 'var ', 'let ', 'func ', 'fn ', '#include',
            'package ', 'using namespace'
        ]
        
        if any(indicator in content_trimmed for indicator in code_indicators):
            return 'code'
        
        # Padrão para texto
        return 'text'
    
    def _format_chunks_with_metadata(
        self,
        chunks: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        query: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Formata os chunks com metadados, identificadores e instruções.
        
        Args:
            chunks: Lista de strings, cada uma representando um chunk.
            metadata: Metadados para incluir em cada chunk.
            query: Consulta original do usuário.
            
        Returns:
            Lista de dicionários formatados com metadados e identificadores.
        """
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            # Criar cabeçalho de contexto para cada chunk
            chunk_header = self._create_chunk_header(
                chunk_index=i,
                total_chunks=len(chunks),
                query=query
            )
            
            # Criar instruções especiais para o último chunk, se houver consulta
            final_instructions = ""
            if i == len(chunks) - 1 and query:
                final_instructions = f"\nVocê processou todos os {len(chunks)} chunks. Agora, responda à consulta original: '{query}'"
            
            # Formatar o chunk com metadados
            formatted_chunk = {
                "content": chunk,
                "context": chunk_header,
                "instructions": final_instructions,
                "metadata": metadata or {},
                "is_last": i == len(chunks) - 1,
                "chunk_id": str(uuid.uuid4())[:8],
                "total_chunks": len(chunks),
                "chunk_index": i
            }
            
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks
    
    def _create_chunk_header(
        self,
        chunk_index: int,
        total_chunks: int,
        query: Optional[str] = None
    ) -> str:
        """
        Cria um cabeçalho para o chunk, explicando seu contexto.
        
        Args:
            chunk_index: Índice do chunk na sequência.
            total_chunks: Número total de chunks.
            query: Consulta original do usuário.
            
        Returns:
            String contendo o cabeçalho do chunk.
        """
        header = f"CHUNK {chunk_index + 1} DE {total_chunks}\n"
        
        if query:
            header += f"Consulta original: '{query}'\n"
        
        header += "\nINSTRUÇÕES:\n"
        header += f"- Este é o chunk {chunk_index + 1} de {total_chunks}.\n"
        
        if chunk_index == 0:
            header += "- Este é o primeiro chunk. Comece a analisá-lo.\n"
        else:
            header += "- Continue a análise a partir do chunk anterior.\n"
        
        if chunk_index == total_chunks - 1:
            header += "- Este é o último chunk. Após processá-lo, formule sua resposta final.\n"
        else:
            header += "- Após processar este chunk, aguarde o próximo para continuar.\n"
        
        return header
