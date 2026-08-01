@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] Run setup.bat first.
  pause
  exit /b 1
)
if "%~1"=="" (
  ".venv\Scripts\python.exe" -m applications.aime_self_revision.math_self_revision --file applications\aime_self_revision\sample_problem.txt --model gpt-5.4-nano --output outputs\aime_self_revision_result.json
) else (
  ".venv\Scripts\python.exe" -m applications.aime_self_revision.math_self_revision %*
)
pause
