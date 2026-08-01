@echo off
setlocal
cd /d "%~dp0"
if not exist "..\..\.venv\Scripts\python.exe" (
  echo [ERROR] Run setup.bat from the repository root first.
  pause
  exit /b 1
)
if "%~1"=="" (
  echo Usage: 1_COLLECT.bat "Europe PMC query" [limit]
  echo Example: 1_COLLECT.bat "OPEN_ACCESS:y AND HAS_FT:y AND (COVID-19)" 50
  pause
  exit /b 2
)
set LIMIT=%~2
if "%LIMIT%"=="" set LIMIT=50
"..\..\.venv\Scripts\python.exe" collect_discussions.py --query "%~1" --limit %LIMIT% --out corpus\collected.jsonl
pause
