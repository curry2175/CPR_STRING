@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Run RUN_WINDOWS.bat once first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install pytest
".venv\Scripts\python.exe" -m pytest -q
pause
