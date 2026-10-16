@echo off
REM Автозапуск PSL-бота. Кинь ярлык этого файла в shell:startup —
REM бот будет сам подниматься при включении компа.
cd /d "D:\DEV\Lookernice"
title PSL-bot @lookernice_bot
python bot.py
pause
