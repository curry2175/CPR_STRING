@echo off
setlocal
cd /d "%~dp0"
if not exist "..\..\.venv\Scripts\python.exe" (
  echo [ERROR] Run setup.bat from the repository root first.
  pause
  exit /b 1
)
"..\..\.venv\Scripts\python.exe" -u analyze_batch.py --corpus corpus\collected.jsonl --module-dir "..\.." --effort low --max-output-tokens 12000 %*
pause
