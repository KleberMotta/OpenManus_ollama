# OpenManus Codebase

## Visão Geral
O OpenManus é um agente de IA especializado em tarefas de busca e navegação web baseado no modelo Qwen2.5-coder. O sistema utiliza a API Ollama para inferência e fornece um conjunto de ferramentas para interagir com a web.

## Estrutura do Projeto
```
app/
├── agent/
│   ├── manus.py         # Agente principal
│   ├── toolcall.py      # Sistema de chamada de ferramentas
│   └── url_fallback.py  # Gerenciador de fallback de URLs
├── tool/
│   ├── browser_use_tool.py # Ferramenta para navegação web
│   ├── web_search.py       # Ferramenta para buscas na web
│   ├── python_execute.py   # Ferramenta para execução de código Python
│   └── file_saver.py       # Ferramenta para salvar arquivos
├── prompt/
│   └── manus.py         # Prompts do sistema
├── config.py            # Configurações do sistema
├── exceptions.py        # Definições de exceções
├── flow/                # Fluxos de execução
├── llm.py               # Interface com modelos de linguagem
├── logger.py            # Sistema de logging
└── schema.py            # Definições de esquemas de dados
```

## Componentes Principais

### Manus (app/agent/manus.py)
- Agente principal que coordena o planejamento e execução de tarefas
- Implementa um sistema de passos e subpassos para organizar a execução
- Realiza análise de tarefas para determinar complexidade e plano de ação
- Gerencia tratamento de erros e fallbacks para URLs

### BrowserUseTool (app/tool/browser_use_tool.py)
- Permite navegação e interação com páginas web via browser-use
- Implementa múltiplos métodos de extração de HTML com fallbacks para conteúdo
- Gerencia timeout para evitar bloqueios indefinidos
- Utiliza técnicas de serialização XML para melhorar a extração de HTML

## Ferramentas Disponíveis

### web_search
Realiza buscas na web e retorna URLs relevantes

### browser_use
Controla um navegador para acessar páginas web, com várias ações:
- `navigate`: Navega para uma URL
- `click`: Clica em um elemento por índice
- `input_text`: Insere texto em um elemento
- `get_html`: Obtém o HTML da página atual
- `get_text`: Obtém o texto da página atual
- `execute_js`: Executa código JavaScript
- `scroll`: Realiza rolagem na página
- `switch_tab`, `new_tab`, `close_tab`: Gerencia abas do navegador

### python_execute
Permite a execução de código Python

### file_saver
Permite salvar arquivos no sistema de arquivos

## Configurações e Timeouts
O timeout para operações de navegação e extração está definido no arquivo de configuração (config.toml) na seção `browser` com a chave `timeout`, com valor padrão de 30 segundos caso não seja especificado.

## Tratamento de Erros
O sistema implementa mecanismos robustos para tratamento de erros, incluindo:
- Fallback automático para URLs alternativas quando uma URL falha
- Múltiplos métodos de extração de HTML para lidar com diferentes tipos de sites
- Detecção de páginas não encontradas (404) e erros de conexão
- Timeout configurável para evitar bloqueios indefinidos
