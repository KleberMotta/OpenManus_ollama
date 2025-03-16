import asyncio
import os
import sys
import shutil
from datetime import datetime
import argparse

from app.agent.manus import Manus
from app.agent.content_processor import ContentProcessor
from app.utils.chunking import ChunkProcessor
from app.logger import logger
from app.schema import Message


def verify_playwright_installation():
    """Verifica se o Playwright e seus navegadores estão instalados corretamente"""
    try:
        # Verificar se o pacote playwright está instalado
        import playwright
    except ImportError:
        print("\n⚠️ Erro: O pacote Playwright não está instalado!")
        print("Execute: pip install playwright")
        print("Em seguida: python -m playwright install")
        return False

    # Verificar se o executavel do Chrome existe
    import os
    import platform
    from pathlib import Path
    
    user_home = os.path.expanduser("~")
    
    # Determinar o caminho correto para os navegadores do Playwright com base no sistema operacional
    if platform.system() == "Windows":
        playwright_browsers_path = os.path.join(user_home, "AppData", "Local", "ms-playwright")
    elif platform.system() == "Darwin":  # macOS
        playwright_browsers_path = os.path.join(user_home, "Library", "Caches", "ms-playwright")
    elif platform.system() == "Linux":
        playwright_browsers_path = os.path.join(user_home, ".cache", "ms-playwright")
    else:
        playwright_browsers_path = None
    
    if not playwright_browsers_path or not os.path.exists(playwright_browsers_path):
        print("\n⚠️ Erro: Os navegadores do Playwright não estão instalados!")
        print("Execute: python -m playwright install")
        # Instalar automaticamente para não travar em ambiente não interativo
        try:
            import subprocess
            print("\nInstalando navegadores do Playwright automaticamente...")
            subprocess.run([sys.executable, "-m", "playwright", "install"], check=True)
            print("\n✅ Navegadores instalados com sucesso!")
            return True
        except Exception as e:
            print(f"\n❌ Erro ao instalar navegadores: {str(e)}")
            return False
    
    # Verificar se o Chrome está instalado (um dos navegadores padrão)
    chrome_path = None
    system = platform.system()
    
    # Procurar a pasta do Chrome em todas as versões do Playwright instaladas
    try:
        for item in os.listdir(playwright_browsers_path):
            if item.startswith("chromium-"):
                if system == "Windows":
                    chrome_dir = os.path.join(playwright_browsers_path, item, "chrome-win")
                    if os.path.exists(chrome_dir):
                        chrome_path = os.path.join(chrome_dir, "chrome.exe")
                elif system == "Darwin":  # macOS
                    chrome_dir = os.path.join(playwright_browsers_path, item, "chrome-mac")
                    if os.path.exists(chrome_dir):
                        chrome_path = os.path.join(chrome_dir, "Chromium.app", "Contents", "MacOS", "Chromium")
                elif system == "Linux":
                    chrome_dir = os.path.join(playwright_browsers_path, item, "chrome-linux")
                    if os.path.exists(chrome_dir):
                        chrome_path = os.path.join(chrome_dir, "chrome")
                
                if chrome_path and os.path.exists(chrome_path):
                    break
    except Exception as e:
        logger.warning(f"Erro ao verificar navegadores: {e}")
    
    if not chrome_path or not os.path.exists(chrome_path):
        print("\n⚠️ Erro: O navegador Chrome do Playwright não foi encontrado!")
        print("Execute: python -m playwright install chromium")
        # Instalar automaticamente para não travar em ambiente não interativo
        try:
            import subprocess
            print("\nInstalando navegador Chromium automaticamente...")
            subprocess.run([sys.executable, "-m", "playwright", "install", "chromium"], check=True)
            print("\n✅ Chromium instalado com sucesso!")
            return True
        except Exception as e:
            print(f"\n❌ Erro ao instalar Chromium: {str(e)}")
            return False
    
    return True


async def cleanup_resources(agent):
    """Limpa todos os recursos antes de finalizar o programa"""
    try:
        # Limpar recursos do browser com prioridade
        browser_tool = None
        
        if hasattr(agent, 'available_tools') and agent.available_tools:
            # Primeiro identificar e fechar o BrowserUseTool
            if 'browser_use' in agent.available_tools.tool_map:
                browser_tool = agent.available_tools.tool_map['browser_use']
                try:
                    logger.info("Closing browser resources...")
                    await browser_tool.cleanup()
                    logger.info("Browser resources closed successfully")
                except Exception as e:
                    logger.warning(f"Error cleaning up browser tool: {e}")
            
            # Agora limpar as outras ferramentas
            for tool_name, tool in agent.available_tools.tool_map.items():
                if tool_name != 'browser_use' and hasattr(tool, 'cleanup'):
                    try:
                        await tool.cleanup()
                    except Exception as e:
                        logger.warning(f"Error cleaning up tool {tool_name}: {e}")
        
        # Fechar sessão do LLM
        if hasattr(agent, 'llm') and agent.llm:
            try:
                logger.info("Closing LLM session...")
                await agent.llm.close()
                logger.info("LLM session closed successfully")
            except Exception as e:
                logger.warning(f"Error closing LLM session: {e}")

        # Limpar o processador de conteúdo, se existir
        if hasattr(agent, 'content_processor') and agent.content_processor:
            logger.info("Cleaning up content processor...")
            # Nenhuma operação específica necessária para limpar este recurso atualmente
                
    except Exception as e:
        # Erro geral no cleanup - não é fatal
        logger.warning(f"Error during cleanup: {e}")


async def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="OpenManus AI Assistant")
    parser.add_argument("-p", "--prompt", type=str, help="Prompt to run directly without interactive mode")
    parser.add_argument("-m", "--model", type=str, help="Model to use (e.g., qwen2.5-coder:7b-instruct)")
    parser.add_argument("--no-chunking", action="store_true", help="Disable content chunking for large content")
    args = parser.parse_args()
    
    # Verificar se o Playwright está instalado corretamente
    if not verify_playwright_installation():
        print("\n❌ ERRO: O Playwright não está configurado corretamente.")
        print("O OpenManus não pode funcionar sem o Playwright para navegação web.")
        print("Instale usando: python -m playwright install")
        print("\nEncerrando programa...")
        return

    agent = Manus()
    
    # Inicializar o processador de conteúdo
    agent.content_processor = ContentProcessor(
        max_token_limit=7000,
        max_chunk_size=6000,
        overlap_size=500,
        max_total_chunks=5
    )
    try:
        # Exibir mensagem de boas-vindas com informações do sistema
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"\n{'-'*50}")
        print(f"OpenManus v1.0 iniciado em {current_time}")
        print(f"Diretório de trabalho: {os.getcwd()}")
        
        # Se um prompt for fornecido na linha de comando, executar diretamente
        if args.prompt:
            prompt = args.prompt
            logger.info(f"Processing command line prompt: {prompt[:50]}...")
            try:
                # Armazenar o prompt original na memória do agente
                agent.memory.add_message(Message.user_message(prompt))
                
                # Armazenar o prompt original para usos futuros
                agent.original_user_prompt = prompt
                
                # Se o arquivo --no-chunking foi especificado, não usar chunking
                if args.no_chunking:
                    result = await agent.run(prompt)
                else:
                    # Verificar se o prompt é grande o suficiente para justificar chunking
                    if len(prompt) > 10000:  # Limite para considerar conteúdo grande
                        print("Conteúdo grande detectado. Aplicando chunking...")
                    
                    # Usar o agente normalmente (o chunking será aplicado automaticamente)
                    result = await agent.run(prompt)
                    
                print(f"\n{result}")
                logger.info("Request processing completed.")
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\nOcorreu um erro ao processar sua solicitação: {e}")
            finally:
                # Fechar recursos antes de finalizar
                print("\nLimpando recursos e finalizando...")
                await cleanup_resources(agent)
                return
        
        # Modo interativo
        print("Digite /exit para encerrar o programa")
        print(f"{'-'*50}\n")
        
        while True:
            prompt = input("\nEnter your prompt: ")
            
            # Verificar se o usuário quer sair
            if prompt.strip().lower() == "/exit":
                print("\nEncerrando o OpenManus. Até logo!")
                break
                
            if not prompt.strip():
                logger.warning("Empty prompt provided. Try again or type /exit to quit.")
                continue

            logger.info("Processing your request...")
            try:
                # Verificar se o prompt é grande o suficiente para justificar chunking
                if len(prompt) > 10000 and not args.no_chunking:  # Limite para considerar conteúdo grande
                    print("\nConteúdo grande detectado. Aplicando chunking...")
                    
                # Executar o agente (o chunking será aplicado automaticamente quando necessário)
                result = await agent.run(prompt)
                
                # Exibir a resposta sem formatações extras
                print(f"\n{result}")
                
                logger.info("Request processing completed.")
            except Exception as e:
                logger.error(f"Error processing request: {e}")
                print(f"\nOcorreu um erro ao processar sua solicitação: {e}")
    except KeyboardInterrupt:
        logger.warning("Operation interrupted by user.")
        print("\nOperação interrompida pelo usuário. Encerrando...")
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
    finally:
        # Fechar recursos antes de finalizar
        print("\nLimpando recursos e finalizando...")
        await cleanup_resources(agent)


if __name__ == "__main__":
    asyncio.run(main())
