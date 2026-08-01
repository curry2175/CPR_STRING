@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run RUN_WINDOWS.bat once or setup.bat first.
  pause
  exit /b 1
)
echo.
echo Full RAGTruth QA evaluation: Raw Direct vs v046 Conservative Balanced Dual-Graph
echo - QA cases only
 echo - All eligible QA cases: --limit 0
 echo - Both conditions use gpt-5.4-nano with low reasoning effort
 echo - No Source/Response six-agent validation is used in this benchmark
 echo - Dual-Graph uses the unchanged v043 Balanced Compilers plus v046 conservative factuality-gated Alignment
 echo - Existing component outputs are cached for safe restart
 echo - Raw-miss / DualGraph-catch candidates are saved without six-agent calls
 echo.
".venv\Scripts\python.exe" -u run_ragtruth_raw_vs_dual_graph.py ^
  --download ^
  --model gpt-5.4-nano ^
  --limit 0 ^
  --task-types QA ^
  --reasoning-effort low ^
  --max-context-chars 60000 ^
  --require-full-evidence ^
  --generation-cache outputs\ragqa_raw_vs_balanced\generation_cache_v040.json ^
  --seed 2040
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE% EQU 0 (
  echo Full RAGTruth QA Raw-vs-Balanced evaluation completed.
) else (
  echo Evaluation stopped with exit code %EXITCODE%. Re-run this BAT to resume from cache.
)
pause
exit /b %EXITCODE%
