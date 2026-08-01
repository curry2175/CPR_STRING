@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv not found. Run RUN_WINDOWS.bat once first.
  pause
  exit /b 1
)
echo.
echo Running pilot evaluation: 20 strict binary cases per dataset
echo ProofWriter Unknown and PubMedQA Maybe are excluded.
echo LegalBench includes explicit Yes/No rows only.
echo.
".venv\Scripts\python.exe" run_three_dataset_evaluation.py --limit-per-dataset 20 --legal-tasks hearsay --model gpt-5.6 --reasoning-effort low --repair-iterations 0
if errorlevel 1 (
  echo.
  echo Evaluation failed. Review the error above.
  pause
  exit /b 1
)
echo.
echo Evaluation completed.
pause
