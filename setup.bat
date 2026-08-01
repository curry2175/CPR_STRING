@echo off
setlocal
cd /d "%~dp0"

echo [1/4] Checking Python...
python --version >nul 2>&1
if errorlevel 1 (
  echo Python was not found. Open this folder in Anaconda Prompt and run setup.bat again.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [2/4] Creating virtual environment...
  python -m venv .venv
) else (
  echo [2/4] Existing virtual environment found.
)

echo [3/4] Installing dependencies...
".venv\Scripts\python.exe" -m pip install --upgrade pip
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

if not exist ".env" (
  echo [4/4] Creating .env from template...
  copy ".env.example" ".env" >nul
  echo.
  echo Paste your OPENAI_API_KEY into the .env file, save it, then run the desired BAT.
  start "" notepad ".env"
) else (
  echo [4/4] Existing .env found.
)

echo.
echo Setup completed.
echo Discussion Lab: run_hub.bat
echo Direct 6-Agent case review: run_case.bat CASE_ID
echo RAGTruth with preset gate: run_ragqa_with_optimized_gate.bat
pause
endlocal
