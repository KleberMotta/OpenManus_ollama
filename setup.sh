#!/bin/bash
# Script de setup para OpenManus

echo "Instalando dependências do OpenManus..."
pip install -r requirements.txt

echo "Instalando navegadores do Playwright..."
python -m playwright install

echo "Configuração concluída! Agora você pode executar o OpenManus com: python main.py"
