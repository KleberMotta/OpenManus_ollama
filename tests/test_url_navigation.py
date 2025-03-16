"""
Teste para verificar como a navegação para URLs diferentes afeta o comportamento do sistema.
"""
import asyncio
import sys
import os
import logging
from unittest.mock import MagicMock, patch
from datetime import datetime

# Adicionar o diretório pai ao path para importar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.manus import Manus
from app.schema import Message, AgentState
from app.agent.url_fallback import URLFallbackHandler

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), 
                             logging.FileHandler(f'url_navigation_test_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')])
logger = logging.getLogger(__name__)

async def test_url_navigation():
    """Testa como a navegação para diferentes URLs afeta o comportamento."""
    
    # URLs de teste
    test_urls = [
        "https://www.apple.com/news/",  # URL incorreta (Apple News+)
        "https://www.apple.com/newsroom/",  # URL correta (Apple Newsroom)
        "https://www.apple.com/",  # URL genérica
        "https://invalid-url-that-doesnt-exist.com/",  # URL inválida
    ]
    
    for url in test_urls:
        logger.info(f"Testando navegação para: {url}")
        
        # Criar uma instância do agente Manus
        agent = Manus()
        agent.memory.clear()
        
        # Definir o prompt inicial
        prompt = "Busque as últimas notícias da Apple"
        agent.memory.add_message(Message.user_message(prompt))
        
        # Inicializar URL handler
        agent.url_handler = URLFallbackHandler()
        
        # Simular execução de web_search
        search_result = [
            "https://www.apple.com/apple-news/",
            "https://www.apple.com/newsroom/",
            "https://apps.apple.com/us/app/apple-news/id1066498020"
        ]
        agent.memory.add_message(Message(
            role="tool",
            content=str(search_result),
            tool_name="web_search"
        ))
        
        # Simular navegação para a URL especificada
        browser_args = {"action": "navigate", "url": url}
        navigation_result = f"Navigated to {url}"
        
        # Chamar diretamente o método _handle_special_tool para a navegação
        await agent._handle_special_tool(
            name="browser_use",
            result=navigation_result,
            args=browser_args,
            error=None
        )
        
        # Verificar o que foi adicionado à memória após a navegação
        nav_messages = [msg.content for msg in agent.memory.messages if "URL" in msg.content or "navegação" in msg.content.lower()]
        
        logger.info(f"Mensagens de navegação adicionadas à memória: {len(nav_messages)}")
        for msg in nav_messages:
            logger.info(f"- {msg[:200]}...")
        
        # Testar o que acontece ao obter o HTML após a navegação
        # Simular HTML baseado na URL
        html_size = 100000 if "newsroom" in url else 200000
        test_html = f"<html><body>{'X' * html_size}</body></html>"
        
        # Simular get_html após navegação
        html_args = {"action": "get_html"}
        
        # Chamar diretamente o método _handle_special_tool para o HTML
        await agent._handle_special_tool(
            name="browser_use",
            result=test_html,
            args=html_args,
            error=None
        )
        
        # Verificar as mensagens após obter o HTML
        html_response_messages = [msg for msg in agent.memory.messages if msg.role == "tool" and msg.content.startswith("<html")]
        
        # Verificar mensagens de sistema após o HTML
        system_after_html = [msg.content for msg in agent.memory.messages if msg.role == "system" and agent.memory.messages.index(msg) > agent.memory.messages.index(html_response_messages[-1]) if html_response_messages]
        
        logger.info(f"Mensagens de sistema após obter HTML: {len(system_after_html)}")
        for msg in system_after_html:
            logger.info(f"- {msg[:200]}...")
        
        # Gerar próximo pensamento para verificar se o contexto foi mantido
        next_thought = await agent.generate_response(
            f"Você navegou para {url} e obteve o HTML. "
            "Qual é o seu próximo passo para completar a tarefa "
            "'Busque as últimas notícias da Apple'?"
        )
        
        # Analisar a resposta para determinar se o contexto foi mantido
        context_maintained = any([
            "notícia" in next_thought.lower(),
            "apple" in next_thought.lower(),
            "extrair" in next_thought.lower(),
            "analisar" in next_thought.lower(),
            "newsroom" in next_thought.lower()
        ])
        
        context_lost = any([
            "instru" in next_thought.lower(),
            "o que devo" in next_thought.lower(),
            "posso ajudar" in next_thought.lower(),
            "como posso" in next_thought.lower(),
            "olá" in next_thought.lower()
        ])
        
        logger.info(f"URL testada: {url}")
        logger.info(f"HTML tamanho: {html_size}")
        logger.info(f"Contexto mantido: {context_maintained}")
        logger.info(f"Contexto perdido: {context_lost}")
        logger.info(f"Resposta do modelo: {next_thought[:200]}...")
        logger.info("-" * 80)
    
    logger.info("Teste de navegação URL concluído")

async def main():
    """Função principal"""
    await test_url_navigation()

if __name__ == "__main__":
    asyncio.run(main())
