@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
call .venv\Scripts\activate.bat
python run_ragtruth_case_6agent.py %*
pause
