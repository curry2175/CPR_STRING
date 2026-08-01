@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
echo.
echo Alignment threshold optimizer
echo - Reads the newest partial cases.jsonl, including an interrupted run
echo - Searches one global threshold and relation-specific thresholds
echo - Uses completed Raw / Response Graph / Alignment outputs only
echo - OpenAI API calls: 0
echo - Fits directly to the completed cases to maximize observed character F1
echo.
call .venv\Scripts\activate.bat
python -u optimize_alignment_thresholds.py --step 0.01 --random-starts 12 --max-passes 5 --top-k 100
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% EQU 0 (
  echo.
  echo Opening latest report...
  start "" "outputs\alignment_threshold_optimizer\latest_report.html"
  echo.
  echo Best gate JSON:
  echo outputs\alignment_threshold_optimizer\latest_best_gate.json
) else (
  echo.
  echo Threshold optimization failed with exit code %EXITCODE%.
)
pause
exit /b %EXITCODE%
