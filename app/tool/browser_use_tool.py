import asyncio
import json
from typing import Optional

from browser_use import Browser as BrowserUseBrowser
from browser_use import BrowserConfig
from browser_use.browser.context import BrowserContext, BrowserContextConfig
from browser_use.dom.service import DomService
from pydantic import Field, field_validator
from pydantic_core.core_schema import ValidationInfo

from app.config import config
from app.tool.base import BaseTool, ToolResult
from app.logger import logger


MAX_LENGTH = 2000

_BROWSER_DESCRIPTION = """
Interact with a web browser to perform various actions such as navigation, element interaction,
content extraction, and tab management. Supported actions include:
- 'navigate': Go to a specific URL
- 'click': Click an element by index
- 'input_text': Input text into an element
- 'screenshot': Capture a screenshot
- 'get_html': Get page HTML content
- 'get_text': Get text content of the page
- 'read_links': Get all links on the page
- 'execute_js': Execute JavaScript code
- 'scroll': Scroll the page
- 'switch_tab': Switch to a specific tab
- 'new_tab': Open a new tab
- 'close_tab': Close the current tab
- 'refresh': Refresh the current page
"""


class BrowserUseTool(BaseTool):
    name: str = "browser_use"
    description: str = _BROWSER_DESCRIPTION
    parameters: dict = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": [
                    "navigate",
                    "click",
                    "input_text",
                    "screenshot",
                    "get_html",
                    "get_text",
                    "execute_js",
                    "scroll",
                    "switch_tab",
                    "new_tab",
                    "close_tab",
                    "refresh",
                ],
                "description": "The browser action to perform",
            },
            "url": {
                "type": "string",
                "description": "URL for 'navigate' or 'new_tab' actions",
            },
            "index": {
                "type": "integer",
                "description": "Element index for 'click' or 'input_text' actions",
            },
            "text": {"type": "string", "description": "Text for 'input_text' action"},
            "script": {
                "type": "string",
                "description": "JavaScript code for 'execute_js' action",
            },
            "scroll_amount": {
                "type": "integer",
                "description": "Pixels to scroll (positive for down, negative for up) for 'scroll' action",
            },
            "tab_id": {
                "type": "integer",
                "description": "Tab ID for 'switch_tab' action",
            },
        },
        "required": ["action"],
        "dependencies": {
            "navigate": ["url"],
            "click": ["index"],
            "input_text": ["index", "text"],
            "execute_js": ["script"],
            "switch_tab": ["tab_id"],
            "new_tab": ["url"],
            "scroll": ["scroll_amount"],
        },
    }

    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)
    browser: Optional[BrowserUseBrowser] = Field(default=None, exclude=True)
    context: Optional[BrowserContext] = Field(default=None, exclude=True)
    dom_service: Optional[DomService] = Field(default=None, exclude=True)

    @field_validator("parameters", mode="before")
    def validate_parameters(cls, v: dict, info: ValidationInfo) -> dict:
        if not v:
            raise ValueError("Parameters cannot be empty")
        return v

    async def _ensure_browser_initialized(self) -> BrowserContext:
        """Ensure browser and context are initialized."""
        if self.browser is None:
            browser_config_kwargs = {"headless": False}

            if config.browser_config:
                from browser_use.browser.browser import ProxySettings

                # handle proxy settings.
                if config.browser_config.proxy and config.browser_config.proxy.server:
                    browser_config_kwargs["proxy"] = ProxySettings(
                        server=config.browser_config.proxy.server,
                        username=config.browser_config.proxy.username,
                        password=config.browser_config.proxy.password,
                    )

                browser_attrs = [
                    "headless",
                    "disable_security",
                    "extra_chromium_args",
                    "chrome_instance_path",
                    "wss_url",
                    "cdp_url",
                ]

                for attr in browser_attrs:
                    value = getattr(config.browser_config, attr, None)
                    if value is not None:
                        if not isinstance(value, list) or value:
                            browser_config_kwargs[attr] = value

            try:
                self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
            except Exception as e:
                logger.error(f"Erro ao inicializar o navegador: {str(e)}")
                # Tente com argumento de segurança desativado
                if "disable_security" not in browser_config_kwargs:
                    browser_config_kwargs["disable_security"] = True
                    try:
                        self.browser = BrowserUseBrowser(BrowserConfig(**browser_config_kwargs))
                        logger.info("Navegador inicializado com segurança desativada")
                    except Exception as inner_e:
                        raise RuntimeError(f"Não foi possível inicializar o navegador: {str(inner_e)}")

        if self.context is None:
            context_config = BrowserContextConfig()

            # if there is context config in the config, use it.
            if (
                config.browser_config
                and hasattr(config.browser_config, "new_context_config")
                and config.browser_config.new_context_config
            ):
                context_config = config.browser_config.new_context_config

            self.context = await self.browser.new_context(context_config)
            self.dom_service = DomService(await self.context.get_current_page())

        return self.context

    async def execute(
        self,
        action: str,
        url: Optional[str] = None,
        index: Optional[int] = None,
        text: Optional[str] = None,
        script: Optional[str] = None,
        scroll_amount: Optional[int] = None,
        tab_id: Optional[int] = None,
        **kwargs,
    ) -> ToolResult:
        """
        Execute a specified browser action.

        Args:
            action: The browser action to perform
            url: URL for navigation or new tab
            index: Element index for click or input actions
            text: Text for input action
            script: JavaScript code for execution
            scroll_amount: Pixels to scroll for scroll action
            tab_id: Tab ID for switch_tab action
            **kwargs: Additional arguments

        Returns:
            ToolResult with the action's output or error
        """
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()
                
                # Detectar padrao incorreto comum (url em get_html)
                if action == "get_html" and url:
                    # Uso incorreto detectado, fazer navegação primeiro
                    logger.warning(f"Detected incorrect usage pattern! Automatically navigating to {url} before get_html")
                    await context.navigate_to(url)
                    await asyncio.sleep(2)  # Dar tempo para carregar

                if action == "navigate":
                    if not url:
                        return ToolResult(error="URL is required for 'navigate' action")
                    try:
                        await context.navigate_to(url)
                        # Verificar se a navegação retornou 404 ou erro de conexão
                        status_code = await context.execute_javascript(
                            """(function() {
                                // Verificar se existe propriedade performanceEntries
                                if (window.performance && window.performance.getEntries) {
                                    const entries = window.performance.getEntries();
                                    for (const entry of entries) {
                                        if (entry.name === window.location.href) {
                                            return entry.responseStatus || 200;
                                        }
                                    }
                                }
                                // Verificar status com base no conteúdo da página
                                if (document.title.includes('404') || 
                                    document.body.textContent.includes('not found') ||
                                    document.body.textContent.includes('404') ||
                                    document.body.textContent.includes('não encontrada')) {
                                    return 404;
                                }
                                return 200;
                            })();"""
                        )
                        
                        if status_code == 404:
                            error_msg = f"❌ URL INVÁLIDA: {url} retornou erro 404 (página não encontrada). Por favor revise os resultados da busca e escolha outra URL válida."
                            logger.warning(error_msg)
                            return ToolResult(error=error_msg)
                            
                        return ToolResult(output=f"Navigated to {url}")
                    except Exception as e:
                        error_msg = f"Error navigating to {url}: {str(e)}"
                        # Erro específico para URLs inválidas
                        if "404" in str(e) or "ERR_NAME_NOT_RESOLVED" in str(e) or "net::ERR" in str(e):
                            error_msg = f"❌ URL INVÁLIDA: {url} não existe ou está inacessível. Por favor revise os resultados da busca e escolha outra URL válida."
                        logger.warning(error_msg)
                        return ToolResult(error=error_msg)

                elif action == "click":
                    if index is None:
                        return ToolResult(error="Index is required for 'click' action")
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    download_path = await context._click_element_node(element)
                    output = f"Clicked element at index {index}"
                    if download_path:
                        output += f" - Downloaded file to {download_path}"
                    return ToolResult(output=output)

                elif action == "input_text":
                    if index is None or not text:
                        return ToolResult(
                            error="Index and text are required for 'input_text' action"
                        )
                    element = await context.get_dom_element_by_index(index)
                    if not element:
                        return ToolResult(error=f"Element with index {index} not found")
                    await context._input_text_element_node(element, text)
                    return ToolResult(
                        output=f"Input '{text}' into element at index {index}"
                    )

                elif action == "screenshot":
                    screenshot = await context.take_screenshot(full_page=True)
                    return ToolResult(
                        output=f"Screenshot captured (base64 length: {len(screenshot)})",
                        system=screenshot,
                    )

                elif action == "get_html":
                    try:
                        # Proteger com timeout para evitar bloqueio indefinido
                        async def get_html_with_timeout(self, timeout=30):
                            # Dar tempo para a página carregar completamente
                            await asyncio.sleep(5)
                            
                            # Lista para armazenar resultados de diferentes métodos
                            html_results = []
                            error_messages = []
                            
                            # Aumentado o limite máximo de caracteres para evitar truncamento
                            MAX_LENGTH = 250000
                            
                            # Tentativa 1: Método padrão via browser_use
                            try:
                                html = await context.get_page_html()
                                if html and len(html) > 100 and "<body></body>" not in html:
                                    html_results.append(("standard", html))
                            except Exception as e:
                                error_messages.append(f"Standard method failed: {str(e)}")
                            
                            # Tentativa 2: Via JavaScript document.documentElement.outerHTML
                            try:
                                js_html = await context.execute_javascript("document.documentElement.outerHTML")
                                if js_html and len(js_html) > 100:
                                    html_results.append(("javascript_outerhtml", js_html))
                            except Exception as e:
                                error_messages.append(f"JavaScript outerHTML method failed: {str(e)}")
                            
                            # Tentativa 3: Via JavaScript innerHTML
                            try:
                                js_inner_html = await context.execute_javascript("document.body.innerHTML")
                                if js_inner_html and len(js_inner_html) > 100:
                                    wrapped_html = f"<html><body>{js_inner_html}</body></html>"
                                    html_results.append(("javascript_innerhtml", wrapped_html))
                            except Exception as e:
                                error_messages.append(f"JavaScript innerHTML method failed: {str(e)}")
                            
                            # Nova tentativa: Obter HTML serializado via API do Chrome
                            try:
                                serialized_html = await context.execute_javascript("""
                                    (function() {
                                        try {
                                            let parser = new DOMParser();
                                            let serializer = new XMLSerializer();
                                            let doc = parser.parseFromString(document.documentElement.outerHTML, "text/html");
                                            return serializer.serializeToString(doc);
                                        } catch(e) {
                                            return "Error: " + e.message;
                                        }
                                    })()
                                """)
                                
                                if serialized_html and len(serialized_html) > 100 and not serialized_html.startswith("Error:"):
                                    html_results.append(("serialized", serialized_html))
                            except Exception as e:
                                error_messages.append(f"Serialization method failed: {str(e)}")
                            
                            # Tentativa 4: Extrair apenas o texto se outros métodos falharem
                            try:
                                text_content = await context.execute_javascript("""
                                    (function() {
                                        const textNodes = [];
                                        const walk = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                                        let node;
                                        while(node = walk.nextNode()) {
                                            if (node.textContent.trim().length > 0) {
                                                textNodes.push(node.textContent.trim());
                                            }
                                        }
                                        return textNodes.join('\n');
                                    })()
                                """)
                                
                                if text_content and len(text_content) > 50:
                                    formatted_text = f"<html><body><pre>{text_content}</pre></body></html>"
                                    html_results.append(("text_content", formatted_text))
                            except Exception as e:
                                error_messages.append(f"Text extraction method failed: {str(e)}")
                            
                            # Selecionar o melhor resultado baseado no tamanho e qualidade
                            if html_results:
                                # Verificar se algum resultado contém as tags html e body
                                complete_results = [r for r in html_results if r[1] and "<html" in r[1].lower() and "<body" in r[1].lower()]
                                
                                # Priorizar resultados completos, ou usar todos se não houver completos
                                results_to_use = complete_results if complete_results else html_results
                                
                                # Ordenar por tamanho (do maior para o menor)
                                results_to_use.sort(key=lambda x: len(x[1]), reverse=True)
                                method, html = results_to_use[0]
                                
                                # Registrar sucesso e detalhes para diagnóstico
                                logger.info(f"HTML extraction successful using '{method}' method ({len(html)} characters)")
                                logger.info(f"HTML inicia com: {html[:100].replace('\n', ' ')}...")
                                
                                # Verificar a qualidade do HTML
                                has_html_tag = "<html" in html.lower()
                                has_body_tag = "<body" in html.lower()
                                has_content = len(html) > 1000
                                
                                logger.info(f"Qualidade do HTML - Tags HTML: {has_html_tag}, Tags Body: {has_body_tag}, Tamanho adequado: {has_content}")
                                
                                # Truncar se necessário, mas preservar um tamanho muito maior
                                if len(html) > MAX_LENGTH:
                                    truncated = html[:MAX_LENGTH] + f"\n... (content truncated, total: {len(html)} characters)"
                                else:
                                    truncated = html
                                    
                                return ToolResult(output=truncated)
                            else:
                                # Fallback: registrar o erro e recomendar tentar outra URL
                                error_summary = "\n".join(error_messages)
                                logger.warning(f"All HTML extraction attempts failed: {error_summary}")
                                
                                # Criar um HTML informativo em vez de retornar apenas um erro
                                fallback_html = f"""
                                <html>
                                <body>
                                <h1>⚠️ Erro na extração do conteúdo</h1>
                                <p>Não foi possível obter o conteúdo HTML desta página. Isso pode ocorrer por vários motivos:</p>
                                <ul>
                                    <li>O site usa técnicas anti-scraping</li>
                                    <li>O site requer interação do usuário (login ou CAPTCHA)</li>
                                    <li>Problemas de conexão ou timeout</li>
                                    <li>Conteúdo dinâmico que requer JavaScript</li>
                                </ul>
                                <p><strong>Recomendação:</strong> Tente acessar outra URL dos resultados da busca.</p>
                                <p><strong>IMPORTANTE:</strong> Você deve tentar com uma das outras URLs retornadas pelo comando web_search. Use o comando browser_use com action=navigate para acessar outra URL da lista.</p>
                                </body>
                                </html>
                                """
                                
                                # Retornar o HTML de fallback com um código de erro especial para sinalizar ao agente
                                logger.error("HTML_EXTRACTION_ERROR: Tentando próxima URL dos resultados da busca...")
                                return ToolResult(output=fallback_html, error="HTML_EXTRACTION_ERROR: Tente outra URL dos resultados")
                                
                        # Executar a função de extração de HTML com timeout
                        html_task = asyncio.create_task(get_html_with_timeout(self))
                        return await asyncio.wait_for(html_task, timeout=30)  # 30 segundos máximo
                        
                    except asyncio.TimeoutError:
                        # Timeout atingido, retornar mensagem informativa
                        logger.error("TIMEOUT: A extração de HTML excedeu o limite de tempo (30s)")
                        fallback_html = f"""
                        <html>
                        <body>
                        <h1>⚠️ Erro: Timeout na extração</h1>
                        <p>Não foi possível extrair o HTML desta página dentro do tempo limite (30 segundos).</p>
                        <p>Isso geralmente ocorre em páginas muito complexas ou com problemas de carregamento.</p>
                        <p><strong>Solução recomendada:</strong> Tente uma das seguintes opções:</p>
                        <ol>
                            <li>Use o comando browser_use com action=get_text para extrair apenas o texto da página atual</li>
                            <li>Tente acessar outra URL dos resultados da pesquisa</li>
                        </ol>
                        </body>
                        </html>
                        """
                        return ToolResult(output=fallback_html, error="Timeout na extração de HTML após 30 segundos")
                    except Exception as e:
                        # Erro geral, informar o usuário
                        logger.error(f"Erro na extração de HTML: {str(e)}")
                        error_html = f"""
                        <html>
                        <body>
                        <h1>❌ Erro na extração de HTML</h1>
                        <p>Ocorreu um erro ao tentar extrair o HTML desta página:</p>
                        <pre>{str(e)}</pre>
                        <p>Tente usar action=get_text como alternativa, ou acesse outra URL dos resultados.</p>
                        </body>
                        </html>
                        """
                        return ToolResult(output=error_html, error=f"Erro na extração de HTML: {str(e)}")

                elif action == "get_text":
                    await asyncio.sleep(2)  # Dar tempo para carregar
                    try:
                        # Versão corrigida do JavaScript para extrair texto de forma mais robusta
                        fixed_js = """
                        (function() {
                            try {
                                // Função para obter texto visível de forma mais robusta
                                function extractVisibleText() {
                                    const textParts = [];
                                    
                                    // Selecionar elementos comuns de texto
                                    const selectors = ['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'td', 'th', 'div:not(:has(*))', 'span:not(:has(*))'];
                                    
                                    for (let i = 0; i < selectors.length; i++) {
                                        try {
                                            const elements = document.querySelectorAll(selectors[i]);
                                            for (let j = 0; j < elements.length; j++) {
                                                try {
                                                    const el = elements[j];
                                                    // Verificar se o elemento é visível
                                                    const style = window.getComputedStyle(el);
                                                    if (style.display !== 'none' && style.visibility !== 'hidden') {
                                                        const text = el.innerText || el.textContent;
                                                        if (text && text.trim().length > 0) {
                                                            textParts.push(text.trim());
                                                        }
                                                    }
                                                } catch (elementError) {
                                                    // Ignorar erros em elementos específicos
                                                }
                                            }
                                        } catch (selectorError) {
                                            // Ignorar erros em seletores específicos
                                        }
                                    }
                                    
                                    return textParts.join('\n');
                                }
                                
                                // Tentar extrair texto com método mais seguro
                                const extractedText = extractVisibleText();
                                if (extractedText && extractedText.length > 0) {
                                    return extractedText;
                                }
                                
                                // Fallbacks sequenciais
                                if (document.body.innerText && document.body.innerText.length > 0) {
                                    return document.body.innerText;
                                }
                                
                                if (document.body.textContent && document.body.textContent.length > 0) {
                                    return document.body.textContent;
                                }
                                
                                return 'Nenhum texto extraído';
                            } catch (e) {
                                return 'Erro na extração de texto: ' + e.message;
                            }
                        })();
                        """
                        
                        # Executar o JavaScript corrigido
                        text = await context.execute_javascript(fixed_js)
                        
                        # Verificar se foi obtido um resultado válido
                        if not text or text.startswith('Erro na extração'):
                            logger.warning(f"Problema na extração de texto: {text}")
                            # Tentar método alternativo como fallback
                            try:
                                simple_js = "document.body.textContent.replace(/\\s+/g, ' ').trim()"
                                text = await context.execute_javascript(simple_js)
                            except Exception as inner_e:
                                logger.warning(f"Fallback também falhou: {str(inner_e)}")
                                return ToolResult(error=f"Falha na extração de texto: {text}")
                            
                        # Limpar e formatar o texto
                        cleaned_text = text.replace('\t', ' ').replace('\r', '').replace('\n\n\n', '\n\n')
                        
                        return ToolResult(output=cleaned_text[:MAX_LENGTH] if len(cleaned_text) > MAX_LENGTH else cleaned_text)
                    except Exception as e:
                        error_msg = f"Falha ao extrair texto: {str(e)}"
                        logger.error(error_msg)
                        return ToolResult(error=error_msg)

                elif action == "read_links":
                    links = await context.execute_javascript(
                        "document.querySelectorAll('a[href]').forEach((elem) => {if (elem.innerText) {console.log(elem.innerText, elem.href)}})"
                    )
                    return ToolResult(output=links)

                elif action == "execute_js":
                    if not script:
                        return ToolResult(
                            error="Script is required for 'execute_js' action"
                        )
                    result = await context.execute_javascript(script)
                    return ToolResult(output=str(result))

                elif action == "scroll":
                    if scroll_amount is None:
                        return ToolResult(
                            error="Scroll amount is required for 'scroll' action"
                        )
                    await context.execute_javascript(
                        f"window.scrollBy(0, {scroll_amount});"
                    )
                    direction = "down" if scroll_amount > 0 else "up"
                    return ToolResult(
                        output=f"Scrolled {direction} by {abs(scroll_amount)} pixels"
                    )

                elif action == "switch_tab":
                    if tab_id is None:
                        return ToolResult(
                            error="Tab ID is required for 'switch_tab' action"
                        )
                    await context.switch_to_tab(tab_id)
                    return ToolResult(output=f"Switched to tab {tab_id}")

                elif action == "new_tab":
                    if not url:
                        return ToolResult(error="URL is required for 'new_tab' action")
                    await context.create_new_tab(url)
                    return ToolResult(output=f"Opened new tab with URL {url}")

                elif action == "close_tab":
                    await context.close_current_tab()
                    return ToolResult(output="Closed current tab")

                elif action == "refresh":
                    await context.refresh_page()
                    return ToolResult(output="Refreshed current page")

                else:
                    return ToolResult(error=f"Unknown action: {action}")

            except Exception as e:
                return ToolResult(error=f"Browser action '{action}' failed: {str(e)}")

    async def get_current_state(self) -> ToolResult:
        """Get the current browser state as a ToolResult."""
        async with self.lock:
            try:
                context = await self._ensure_browser_initialized()
                state = await context.get_state()
                state_info = {
                    "url": state.url,
                    "title": state.title,
                    "tabs": [tab.model_dump() for tab in state.tabs],
                    "interactive_elements": state.element_tree.clickable_elements_to_string(),
                }
                return ToolResult(output=json.dumps(state_info))
            except Exception as e:
                return ToolResult(error=f"Failed to get browser state: {str(e)}")

    async def cleanup(self):
        """Clean up browser resources."""
        import sys
        from app.logger import logger
        
        logger.info("Iniciando cleanup do browser...")
        try:
            async with self.lock:
                # Fechar o contexto primeiro
                if self.context is not None:
                    try:
                        logger.info("Fechando contexto do browser...")
                        await self.context.close()
                        self.context = None
                        self.dom_service = None
                        logger.info("Contexto do browser fechado com sucesso.")
                    except Exception as e:
                        logger.error(f"Erro ao fechar o contexto do browser: {e}")
                
                # Depois fechar o browser
                if self.browser is not None:
                    try:
                        logger.info("Fechando instância do browser...")
                        await self.browser.close()
                        self.browser = None
                        logger.info("Instância do browser fechada com sucesso.")
                    except Exception as e:
                        logger.error(f"Erro ao fechar o browser: {e}")
                        
                # Forçar a limpeza de referências
                import gc
                gc.collect()
                
                logger.info("Cleanup do browser concluído com sucesso.")
        except Exception as e:
            logger.error(f"Erro geral durante o cleanup do browser: {e}")

    def __del__(self):
        """Ensure cleanup when object is destroyed."""
        if self.browser is not None or self.context is not None:
            try:
                # Tentar obter o loop existente
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        # Se o loop estiver em execução, agendar o cleanup
                        loop.create_task(self.cleanup())
                    else:
                        # Se não estiver em execução, rodar diretamente
                        loop.run_until_complete(self.cleanup())
                except RuntimeError:
                    # Se não houver um loop em execução, criar um novo
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(self.cleanup())
                    loop.close()
            except Exception as e:
                # Em caso de erro, pelo menos limpar as referências
                print(f"Error during browser cleanup: {e}")
                self.browser = None
                self.context = None
                self.dom_service = None
