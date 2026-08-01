@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
echo Fast threshold optimizer: 0.02 grid, 4 random starts, API calls 0
call .venv\Scripts\activate.bat
python -u optimize_alignment_thresholds.py --step 0.02 --random-starts 4 --max-passes 3 --top-k 50
set EXITCODE=%ERRORLEVEL%
if %EXITCODE% EQU 0 start "" "outputs\alignment_threshold_optimizer\latest_report.html"
pause
exit /b %EXITCODE%
