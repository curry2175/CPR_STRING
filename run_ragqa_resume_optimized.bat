@echo off
setlocal
cd /d "%~dp0"
if not exist .venv\Scripts\python.exe (
  echo [ERROR] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
set "BEST_GATE=outputs\alignment_threshold_optimizer\latest_best_gate.json"
if not exist "%BEST_GATE%" set "BEST_GATE=config\optimized_gate_330.json"
set "CACHE=outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json"
if not exist "%CACHE%" (
  echo [ERROR] Local generation cache not found:
  echo %CACHE%
  echo Run scripts\IMPORT_LOCAL_RUN_WINDOWS.bat first, or use run_ragqa_with_optimized_gate.bat for a fresh run.
  pause
  exit /b 1
)
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
  --generation-cache "%CACHE%" ^
  --alignment-prompt-profile auto ^
  --alignment-gate-config "%BEST_GATE%" ^
  --seed 2040
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE% EQU 0 (
  echo Optimized resume completed.
) else (
  echo Run stopped with exit code %EXITCODE%. Re-run this BAT to continue from cache.
)
pause
exit /b %EXITCODE%
