"""
Exemplo de uso do OpenManus com capacidade de chunking.

Este script demonstra como utilizar o OpenManus com a funcionalidade
de chunking para processar conteúdos grandes.
"""

import asyncio
import argparse
import sys
import os

# Adicionar o diretório pai ao path para imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agent.manus_chunking import ChunkingManus
from app.logger import logger


async def main():
    parser = argparse.ArgumentParser(description="OpenManus com chunking para processamento de conteúdos grandes")
    parser.add_argument("--prompt", type=str, help="Consulta do usuário", default=None)
    parser.add_argument("--file", type=str, help="Arquivo com conteúdo a ser processado", default=None)
    args = parser.parse_args()
    
    # Inicializar o agente
    agent = ChunkingManus()
    
    try:
        if args.file:
            # Ler conteúdo do arquivo
            try:
                with open(args.file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Usar o prompt ou um prompt padrão
                prompt = args.prompt or "Analise este conteúdo e extraia as informações mais relevantes"
                
                # Executar o agente com o conteúdo do arquivo
                print(f"\nProcessando arquivo: {args.file}")
                print(f"Tamanho do conteúdo: {len(content)} caracteres")
                print(f"Consulta: {prompt}\n")
                
                # Combinar o prompt e o conteúdo
                full_prompt = f"{prompt}\n\n{content}"
                
                # Executar o agente
                response = await agent.run(full_prompt)
                print(f"\nResposta do OpenManus:\n{response}")
            
            except Exception as e:
                logger.error(f"Erro ao ler ou processar o arquivo: {e}")
                print(f"Erro ao processar o arquivo: {str(e)}")
        
        elif args.prompt:
            # Executar o agente com o prompt direto
            print(f"\nProcessando consulta: {args.prompt}\n")
            response = await agent.run(args.prompt)
            print(f"\nResposta do OpenManus:\n{response}")
        
        else:
            # Modo interativo
            print("\nModo interativo do OpenManus com chunking")
            print("Digite 'exit' ou 'quit' para sair\n")
            
            while True:
                prompt = input("\nConsulta: ")
                
                if prompt.lower() in ['exit', 'quit']:
                    break
                
                if prompt.startswith("file:"):
                    # Processar um arquivo
                    file_path = prompt[5:].strip()
                    
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        process_prompt = input("Consulta para o conteúdo do arquivo: ")
                        full_prompt = f"{process_prompt}\n\n{content}"
                        
                        print(f"\nProcessando arquivo: {file_path}")
                        print(f"Tamanho do conteúdo: {len(content)} caracteres\n")
                        
                        response = await agent.run(full_prompt)
                    
                    except Exception as e:
                        logger.error(f"Erro ao ler ou processar o arquivo: {e}")
                        response = f"Erro ao processar o arquivo: {str(e)}"
                
                else:
                    # Processar prompt direto
                    response = await agent.run(prompt)
                
                print(f"\nResposta do OpenManus:\n{response}")
    
    finally:
        # Limpar recursos
        if hasattr(agent, 'available_tools') and agent.available_tools:
            for tool_name in ['browser_use']:
                tool = agent.available_tools.get_tool(tool_name)
                if tool:
                    try:
                        await tool.cleanup()
                        print(f"Recursos da ferramenta {tool_name} liberados")
                    except Exception as e:
                        logger.error(f"Erro ao limpar recursos da ferramenta {tool_name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
