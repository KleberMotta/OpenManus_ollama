"""
Teste para verificar se o tamanho do HTML causa perda de contexto no modelo.
"""
import asyncio
import sys
import os
import logging
from datetime import datetime

# Adicionar o diretório pai ao path para importar os módulos do projeto
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.manus import Manus
from app.schema import Message

# Configurar logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler(), 
                              logging.FileHandler(f'html_size_test_{datetime.now().strftime("%Y%m%d%H%M%S")}.log')])
logger = logging.getLogger(__name__)

async def test_html_size_impact():
    """Testa como diferentes tamanhos de HTML afetam o contexto do modelo."""
    
    # Criar uma instância do agente Manus
    agent = Manus()
    
    # Definir o prompt inicial
    prompt = "Busque as últimas notícias da Apple"
    
    # Adicionar o prompt à memória do agente
    agent.memory.add_message(Message.user_message(prompt))
    
    # Gerar diferentes tamanhos de HTML para teste
    sizes = [1000, 10000, 50000, 100000]
    
    for size in sizes:
        logger.info(f"Testando HTML de tamanho {size} caracteres")
        
        # Criar um HTML de teste com o tamanho especificado
        test_html = f"<html><body>{'X' * size}</body></html>"
        
        # Simular a resposta do browser_use com get_html
        browser_response = test_html
        
        # Simular o processamento do HTML pelo agente
        # Primeiro, limpar mensagens anteriores exceto a primeira
        agent.memory.messages = agent.memory.messages[:1]
        
        # Adicionar mensagem do sistema sobre o contexto
        agent.memory.add_message(Message.system_message(
            f"Você está processando um HTML de tamanho {size} para 'Busque as últimas notícias da Apple'"
        ))
        
        # Adicionar o HTML como uma resposta de ferramenta
        agent.memory.add_message(Message(
            role="tool",
            content=browser_response[:100] + "... [HTML truncado]",  # Truncado para o log
            tool_name="browser_use"
        ))
        
        # Verificar o que o agente pensa a seguir
        next_thought = await agent.generate_response(
            f"Você acabou de receber HTML de tamanho {size}. "
            "Qual é o seu próximo passo para completar a tarefa "
            "'Busque as últimas notícias da Apple'?"
        )
        
        # Analisar a resposta para determinar se o contexto foi mantido
        context_maintained = any([
            "notícia" in next_thought.lower(),
            "apple" in next_thought.lower(),
            "extrair" in next_thought.lower(),
            "analisar" in next_thought.lower(),
            "continu" in next_thought.lower()  # Captura "continuar", "continuidade", etc.
        ])
        
        context_lost = any([
            "instru" in next_thought.lower(),  # Captura "instruções", "instrua-me", etc.
            "o que devo" in next_thought.lower(),
            "posso ajudar" in next_thought.lower(),
            "como posso" in next_thought.lower(),
            "olá" in next_thought.lower()
        ])
        
        logger.info(f"Tamanho HTML: {size}")
        logger.info(f"Contexto mantido: {context_maintained}")
        logger.info(f"Contexto perdido: {context_lost}")
        logger.info(f"Resposta do modelo: {next_thought[:200]}...")
        logger.info("-" * 80)
        
    logger.info("Teste de tamanho HTML concluído")

async def main():
    """Função principal"""
    await test_html_size_impact()

if __name__ == "__main__":
    # Criar diretório de testes se não existir
    os.makedirs("tests", exist_ok=True)
    asyncio.run(main())
