"""
Módulo para tratamento de fallback de URLs quando uma URL falha durante a navegação 
ou extração de HTML no projeto OpenManus.
"""

from typing import List, Optional
import ast
import re
from app.logger import logger

class URLFallbackHandler:
    """
    Classe que implementa o mecanismo de fallback para URLs alternativas
    quando uma URL falha durante navegação ou extração de HTML.
    """
    
    def __init__(self):
        # Lista de URLs já tentadas
        self.tried_urls: List[str] = []
        # Lista de URLs disponíveis dos resultados de busca
        self.available_urls: List[str] = []
    
    def process_web_search_result(self, result: str) -> None:
        """
        Processa o resultado de uma busca web para extrair URLs disponíveis.
        
        Args:
            result: Texto contendo o resultado da busca web
        """
        if not result:
            return
            
        # Limpar a lista apenas se encontrarmos novas URLs
        urls_found = False
            
        # Verificar se a resposta parece uma lista de URLs
        if result.startswith('[') and ']' in result and 'http' in result:
            try:
                # Tentar avaliar como lista literal Python
                url_list = ast.literal_eval(result)
                if isinstance(url_list, list):
                    # Filtrar apenas itens que parecem URLs
                    new_urls = [url for url in url_list if isinstance(url, str) and url.startswith('http')]
                    if new_urls:
                        self.available_urls = new_urls
                        urls_found = True
                        logger.info(f"Armazenadas {len(self.available_urls)} URLs dos resultados de busca")
            except Exception as e:
                logger.warning(f"Erro ao processar lista de URLs via ast: {e}")
                
        # Se não encontrou via ast.literal_eval, tentar regex
        if not urls_found:
            try:
                # Padrão para encontrar URLs em texto
                url_pattern = r'https?://[^\s\'"]+\.[^\s\'"]+(?=/|$)'
                found_urls = re.findall(url_pattern, result)
                
                if found_urls:
                    # Filtrar URLs únicas e válidas
                    new_urls = []
                    for url in found_urls:
                        # Limpar a URL removendo possíveis caracteres no final
                        clean_url = re.sub(r'[,.;:]$', '', url)
                        if clean_url not in new_urls:
                            new_urls.append(clean_url)
                    
                    if new_urls:
                        self.available_urls = new_urls
                        urls_found = True
                        logger.info(f"Armazenadas {len(self.available_urls)} URLs dos resultados de busca via regex")
            except Exception as e:
                logger.warning(f"Erro ao processar lista de URLs via regex: {e}")
    
    def record_navigation_attempt(self, url: str) -> None:
        """
        Registra tentativa de navegação para uma URL.
        
        Args:
            url: URL que foi tentada
        """
        if url and url not in self.tried_urls and isinstance(url, str):
            self.tried_urls.append(url)
            logger.info(f"Registrada tentativa de navegação para: {url}")
    
    def get_next_url(self) -> Optional[str]:
        """
        Retorna a próxima URL não tentada da lista de URLs disponíveis.
        
        Returns:
            URL não tentada ou None se todas já foram tentadas
        """
        # Filtrar URLs que ainda não foram tentadas
        untried_urls = [url for url in self.available_urls if url not in self.tried_urls]
        
        # Se houver URLs não tentadas, retornar a primeira
        if untried_urls:
            next_url = untried_urls[0]
            # Adicionar à lista de URLs tentadas
            self.tried_urls.append(next_url)
            logger.info(f"URL alternativa sugerida: {next_url}")
            return next_url
        
        logger.warning("Sem URLs alternativas disponíveis.")
        return None
    
    def handle_html_extraction_error(self) -> Optional[str]:
        """
        Manipulador de erro para falha na extração de HTML.
        Sugere URL alternativa e cria mensagem de erro.
        
        Returns:
            Mensagem de erro com sugestão de URL ou None se não houver URLs disponíveis
        """
        next_url = self.get_next_url()
        if next_url:
            logger.info(f"Sugerindo URL alternativa após erro de HTML: {next_url}")
            # Criar mensagem de erro com sugestão
            error_message = f"""
            ERRO: Falha ao extrair conteúdo da página atual.
            
            SUGESTÃO: Tente navegar para esta URL alternativa: {next_url}
            
            Exemplo de código para executar:
            ```json
            {{
              "function": {{
                "name": "browser_use",
                "arguments": {{
                  "action": "navigate",
                  "url": "{next_url}"
                }}
              }}
            }}
            ```
            """
            return error_message
        else:
            return """
            ERRO: Todas as URLs disponíveis já foram tentadas sem sucesso.
            Recomende ao usuário pesquisar por termos mais específicos ou use outra abordagem.
            """
    
    def detect_navigation_loop(self, url: str) -> bool:
        """
        Detecta se a URL atual está em um loop de navegação.
        
        Args:
            url: URL para verificar
            
        Returns:
            True se for detectado um loop, False caso contrário
        """
        # Verifica se a URL já foi tentada mais de uma vez
        if url in self.tried_urls and self.tried_urls.count(url) > 1:
            logger.warning(f"Detectado loop de navegação para URL: {url}")
            return True
        return False
