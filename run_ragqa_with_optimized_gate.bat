@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
set "BEST_GATE=config\optimized_gate_330.json"
call .venv\Scripts\activate.bat
python -u run_ragtruth_raw_vs_dual_graph.py ^
  --download ^
  --model gpt-5.4-nano ^
  --limit 0 ^
  --task-types QA ^
  --split test ^
  --quality good ^
  --reasoning-effort low ^
  --max-context-chars 60000 ^
  --require-full-evidence ^
  --alignment-prompt-profile auto ^
  --alignment-gate-config "%BEST_GATE%" ^
  --seed 2040
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE% EQU 0 (
  echo Optimized-gate evaluation completed.
) else (
  echo Run stopped with exit code %EXITCODE%.
)
pause
exit /b %EXITCODE%
