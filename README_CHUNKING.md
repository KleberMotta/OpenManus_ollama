# Processamento de Conteúdos Grandes com Chunking no OpenManus

O OpenManus agora possui capacidade integrada de processamento de conteúdos grandes através de chunking, permitindo trabalhar com textos, HTML e código que excedam o limite de contexto do modelo LLM.

## O Problema do Contexto Limitado

Os modelos de linguagem como o Qwen2.5-coder têm um limite de contexto (número máximo de tokens que podem processar de uma vez). O sistema de chunking divide o conteúdo grande em partes menores que são processadas sequencialmente, mantendo o contexto.

## Funcionamento do Chunking

O chunking é ativado automaticamente quando:

1. O conteúdo de uma página web obtido via `browser_use` com ação `get_html` ou `get_text` excede 10.000 caracteres
2. Uma consulta com texto grande (>10.000 caracteres) é enviada ao OpenManus

## Estratégias Implementadas

O sistema usa diferentes estratégias dependendo do tipo de conteúdo:

- **HTML**: Preserva a estrutura DOM, dividindo por cabeçalhos e seções
- **Texto**: Divide por parágrafos e sentenças para manter a coerência
- **Código**: Preserva funções, classes e blocos para manter a estrutura do código
- **JSON**: Mantém a estrutura de objetos e arrays

## Uso no OpenManus

O chunking é aplicado automaticamente, mas pode ser controlado com:

```bash
# Desativar chunking para uma execução específica
python main.py --no-chunking -p "minha consulta"
```

## Benefícios

- **Processamento de conteúdos maiores**: Supera o limite de contexto do modelo
- **Preservação semântica**: Mantém a estrutura e significado do conteúdo
- **Processamento otimizado**: Usa estratégias específicas para cada tipo de conteúdo

## Como Funciona Internamente

1. Quando um conteúdo grande é detectado, o sistema o divide em chunks menores
2. Cada chunk é processado sequencialmente, mantendo contexto entre eles
3. Os resultados intermediários são armazenados para manter a continuidade
4. O último chunk sintetiza uma resposta final considerando todos os chunks

Este sistema permite que o OpenManus trabalhe com documentos, páginas web e código muito maiores que o limite do modelo, mantendo a qualidade das respostas.
