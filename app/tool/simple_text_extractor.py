
"""
Implementação de um extrator de texto simples e robusto para
ser usado quando o método browser_use com get_html ou get_text falhar.
"""

import asyncio
import logging
from app.tool.base import BaseTool, ToolResult
from playwright.async_api import async_playwright

logger = logging.getLogger(__name__)


class SimpleTextExtractor(BaseTool):
    """
    Ferramenta para extrair texto de páginas web de forma simples e robusta.
    Usa diretamente o Playwright sem depender da implementação complexa do browser_use.
    """

    name: str = "simple_text_extract"
    description: str = """
    Extrai texto de uma página web de forma simples e robusta.
    Use esta ferramenta quando a extração de HTML ou texto com browser_use falhar.
    É uma ferramenta especializada para obtençao de conteúdo textual.
    """
    parameters: dict = {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "URL da página web para extrair texto",
            },
            "timeout": {
                "type": "integer",
                "description": "Tempo máximo de espera em segundos",
                "default": 30,
            },
        },
        "required": ["url"],
    }

    async def execute(self, url: str, timeout: int = 30) -> ToolResult:
        """
        Executa a extração de texto da URL especificada.
        
        Args:
            url: URL da página web para extrair texto
            timeout: Tempo máximo de espera em segundos
            
        Returns:
            ToolResult com o texto extraído ou mensagem de erro
        """
        try:
            logger.info(f"Inicializando extração de texto de {url}")
            
            # Criar uma task para extrair texto com timeout
            task = asyncio.create_task(self._extract_text(url))
            text = await asyncio.wait_for(task, timeout=timeout)
            
            if not text:
                return ToolResult(error=f"Não foi possível extrair texto de {url}")
                
            return ToolResult(output=text)
        except asyncio.TimeoutError:
            return ToolResult(error=f"Timeout ao extrair texto de {url} após {timeout}s")
        except Exception as e:
            logger.error(f"Erro ao extrair texto: {str(e)}")
            return ToolResult(error=f"Erro ao extrair texto: {str(e)}")

    async def _extract_text(self, url: str) -> str:
        """
        Implementação interna da extração de texto usando Playwright diretamente.
        
        Args:
            url: URL da página web
            
        Returns:
            Texto extraído da página
        """
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            
            try:
                context = await browser.new_context()
                page = await context.new_page()
                
                # Navegar para a URL
                try:
                    await page.goto(url, wait_until="domcontentloaded", timeout=20000)
                    # Aguardar um pouco mais para conteúdo dinâmico
                    await asyncio.sleep(3)
                except Exception as e:
                    logger.error(f"Erro ao navegar para {url}: {str(e)}")
                    return f"Erro de navegação: {str(e)}"
                
                # Tentar várias estratégias de extração de texto
                text = ""
                
                # Estratégia 1: Texto dos elementos de texto comuns
                try:
                    text_elements = await page.evaluate("""
                        Array.from(document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td, th'))
                            .map(el => el.textContent.trim())
                            .filter(text => text.length > 0)
                            .join('\\n')
                    """)
                    
                    if text_elements and len(text_elements) > 50:
                        text = text_elements
                except Exception as e:
                    logger.warning(f"Falha na estratégia 1: {str(e)}")
                
                # Se a estratégia 1 falhar, tente a estratégia 2
                if not text:
                    try:
                        body_text = await page.evaluate("document.body.textContent")
                        if body_text:
                            text = body_text
                    except Exception as e:
                        logger.warning(f"Falha na estratégia 2: {str(e)}")
                
                # Se ambas falharem, tente uma abordagem muito básica
                if not text:
                    try:
                        # Tentar obter texto através de innerText
                        simple_text = await page.evaluate("document.body.innerText")
                        text = simple_text
                    except Exception as e:
                        logger.warning(f"Falha na estratégia 3: {str(e)}")
                
                # Limpar o texto
                if text:
                    # Eliminar espaços em branco extras e quebras de linha múltiplas
                    text = ' '.join(text.split())
                    text = text.replace('\t', ' ').replace('\r', '')
                    
                    # Limitar o tamanho para evitar problemas
                    max_length = 2000
                    if len(text) > max_length:
                        text = text[:max_length] + "... (texto truncado)"
                
                return text or "Não foi possível extrair texto da página"
                
            finally:
                await browser.close()
