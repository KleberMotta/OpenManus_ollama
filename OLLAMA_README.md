# OpenManus com Ollama - Guia de Configuração

Este guia explica como configurar e usar o OpenManus com o Ollama localmente.

## 1. Pré-requisitos

### 1.1. Instalar o Ollama

1. Baixe o Ollama em [https://ollama.com/download](https://ollama.com/download)
2. Instale seguindo as instruções para seu sistema operacional
3. Verifique se o Ollama está rodando acessando `http://localhost:11434` no navegador

### 1.2. Baixar um Modelo

```bash
# Recomendado:
ollama pull qwen2.5-coder:7b-instruct

# Ou outros modelos:
ollama pull phi3:mini  # Requer menos recursos
ollama pull llama3:8b  # Boa relação custo/benefício
```

## 2. Configuração do OpenManus

### 2.1. Instalar Dependências

```bash
pip install -r requirements.txt
```

### 2.2. Configurar o arquivo config.toml

O arquivo `config/config.toml` já está configurado para usar o Ollama:

```toml
[llm]
model = "qwen2.5-coder:7b-instruct"
base_url = "http://localhost:11434"
api_key = "não-necessário-para-ollama"
max_tokens = 4096
temperature = 0.0
```

Você pode modificar as seguintes opções:

- `model`: Nome do modelo baixado com o Ollama
- `base_url`: URL da API do Ollama (padrão: http://localhost:11434)
- `max_tokens`: Número máximo de tokens na saída (afeta a extensão das respostas)
- `temperature`: Temperatura para geração (0.0 = mais determinístico, 1.0 = mais criativo)

## 3. Testar a Instalação

Execute o script de teste para verificar se tudo está funcionando:

```bash
python test_ollama.py
```

Se todos os testes passarem, sua configuração está correta.

## 4. Executar o OpenManus

```bash
python run_flow.py
```

Ou use o script principal:

```bash
python main.py
```

## 5. Solução de Problemas

### 5.1. Erro de Conexão

Se você receber erros de conexão:

1. Verifique se o Ollama está rodando: `ollama serve`
2. Confirme a URL no config.toml (deve ser `http://localhost:11434`)
3. Verifique se o firewall não está bloqueando a porta 11434

### 5.2. Erro "Modelo não Encontrado"

Se o modelo não for encontrado:

1. Liste os modelos disponíveis: `ollama list`
2. Baixe o modelo correto: `ollama pull nome-do-modelo`

### 5.3. Uso Alto de Memória

Se estiver tendo problemas com uso de memória:

1. Use modelos menores como `phi3:mini` ou `tinyllama:1.1b`
2. Reduza o `max_tokens` no config.toml

## 6. Diferenças para a Versão Original

Esta versão modificada do OpenManus:

1. **Usa a API REST do Ollama** em vez de carregar modelos diretamente via llama-cpp-python
2. **Requer menos recursos** já que o Ollama gerencia a memória de maneira mais eficiente
3. **É mais fácil trocar de modelo** - basta alterar o nome no config.toml e baixar o modelo com Ollama
4. **Não precisa baixar arquivos GGUF** - o Ollama gerencia o download e armazenamento dos modelos

## 7. Modelos Recomendados

- **qwen2.5-coder:7b-instruct**: Bom para tarefas de programação (requer ~8GB de RAM)
- **phi3:mini**: Leve e bom para máquinas com menos RAM (~4GB)
- **llama3:8b**: Boa relação entre desempenho e requisitos de memória (~8GB)
- **mistral:7b-instruct-v0.2**: Bom para tarefas de instrução gerais (~8GB)

Mais modelos disponíveis em: [https://ollama.com/library](https://ollama.com/library)
