"""
Teste para avaliar a capacidade do sistema de manter a continuidade do fluxo
após diferentes operações e erros.
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Adicionar o diretório pai ao path para importar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.manus import Manus
from app.schema import Message, AgentState

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), 
                              logging.FileHandler(f'workflow_continuity_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')])
logger = logging.getLogger(__name__)

async def test_workflow_continuity():
    """Testa diferentes cenários para avaliar a continuidade do fluxo de trabalho."""
    
    scenarios = [
        {
            "name": "Fluxo Normal",
            "steps": [
                {"tool": "web_search", "result": ["https://example.com"], "error": None},
                {"tool": "browser_use", "args": {"action": "navigate", "url": "https://example.com"}, "result": "Navigated to https://example.com", "error": None},
                {"tool": "browser_use", "args": {"action": "get_html"}, "result": "<html><body>Conteúdo normal</body></html>", "error": None}
            ]
        },
        {
            "name": "Erro de Navegação",
            "steps": [
                {"tool": "web_search", "result": ["https://example.com"], "error": None},
                {"tool": "browser_use", "args": {"action": "navigate", "url": "https://invalid-url.com"}, "result": None, "error": "Error navigating to URL: Connection failed"}
            ]
        },
        {
            "name": "Erro de Extração HTML",
            "steps": [
                {"tool": "web_search", "result": ["https://example.com"], "error": None},
                {"tool": "browser_use", "args": {"action": "navigate", "url": "https://example.com"}, "result": "Navigated to https://example.com", "error": None},
                {"tool": "browser_use", "args": {"action": "get_html"}, "result": None, "error": "HTML_EXTRACTION_ERROR: Failed to extract HTML content"}
            ]
        },
        {
            "name": "HTML Tamanho Médio",
            "steps": [
                {"tool": "web_search", "result": ["https://example.com"], "error": None},
                {"tool": "browser_use", "args": {"action": "navigate", "url": "https://example.com"}, "result": "Navigated to https://example.com", "error": None},
                {"tool": "browser_use", "args": {"action": "get_html"}, "result": f"<html><body>{'X' * 5000}</body></html>", "error": None}
            ]
        }
    ]
    
    for scenario in scenarios:
        logger.info(f"Testando cenário: {scenario['name']}")
        
        # Criar uma instância do agente Manus
        agent = Manus()
        
        # Definir o prompt inicial
        prompt = "Busque as últimas notícias da Apple"
        agent.memory.add_message(Message.user_message(prompt))
        
        # Executar os passos do cenário
        for i, step in enumerate(scenario["steps"]):
            logger.info(f"Executando passo {i+1}: {step['tool']}")
            
            # Adicionar resultado da ferramenta à memória
            if step["result"] is not None:
                agent.memory.add_message(Message(
                    role="tool",
                    content=str(step["result"])[:200] + "..." if len(str(step["result"])) > 200 else str(step["result"]),
                    tool_name=step["tool"]
                ))
            
            # Processar o passo especial
            await agent._handle_special_tool(
                name=step["tool"],
                result=step["result"],
                args=step.get("args", {}),
                error=step["error"]
            )
            
            # Verificar se o agente ainda está em estado válido
            if agent.state == "TERMINATED":
                logger.warning(f"Agente terminou prematuramente após o passo {i+1}")
        
        # Ver quais foram as últimas 3 mensagens no sistema
        last_messages = agent.memory.messages[-3:] if len(agent.memory.messages) >= 3 else agent.memory.messages
        logger.info("Últimas mensagens na memória:")
        for msg in last_messages:
            content_preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            logger.info(f"- [{msg.role}]: {content_preview}")
        
        # Verificar se o agente pode continuar após o cenário
        next_thought = await agent.generate_response(
            "Após o cenário acima, qual é o seu próximo passo para completar a tarefa "
            "'Busque as últimas notícias da Apple'?"
        )
        
        # Analisar se o contexto foi mantido
        context_maintained = any([
            "notícia" in next_thought.lower(),
            "apple" in next_thought.lower(),
            "extrair" in next_thought.lower(), 
            "analisar" in next_thought.lower(),
            "próximo passo" in next_thought.lower(),
            "continuar" in next_thought.lower()
        ])
        
        context_lost = any([
            "instrução" in next_thought.lower(),
            "como posso ajudar" in next_thought.lower(),
            "olá" in next_thought.lower()
        ])
        
        recovery_suggested = any([
            "tentar outra" in next_thought.lower(),
            "alternativa" in next_thought.lower(),
            "próxima url" in next_thought.lower(),
            "newsroom" in next_thought.lower() and "news" in scenario["name"].lower()
        ])
        
        logger.info(f"Cenário: {scenario['name']}")
        logger.info(f"Contexto mantido: {context_maintained}")
        logger.info(f"Contexto perdido: {context_lost}")
        logger.info(f"Recuperação sugerida: {recovery_suggested}")
        logger.info(f"Resposta do modelo: {next_thought[:200]}...")
        logger.info("-" * 80)
    
    logger.info("Teste de continuidade de fluxo concluído")

async def main():
    """Função principal"""
    await test_workflow_continuity()

if __name__ == "__main__":
    asyncio.run(main())
