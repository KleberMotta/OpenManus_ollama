@echo off
echo Instalando dependencias do OpenManus...
pip install -r requirements.txt

echo Instalando navegadores do Playwright...
python -m playwright install

echo Configuracao concluida! Agora voce pode executar o OpenManus com: python main.py
pause
