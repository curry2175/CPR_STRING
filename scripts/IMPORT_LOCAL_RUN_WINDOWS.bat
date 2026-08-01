@echo off
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0\.."

set "DEFAULT_CASES=C:\Users\chefc\OneDrive\바탕 화면\YAI Challenge\v046\outputs\ragtruth_raw_vs_dual_graph_nano\ragtruth_raw_vs_dual_graph_nano_20260802_021015\cases.jsonl"
set "DEFAULT_CACHE=C:\Users\chefc\OneDrive\바탕 화면\YAI Challenge\v046\outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json"
set "DEFAULT_OPT_ROOT=C:\Users\chefc\OneDrive\바탕 화면\YAI Challenge\v048\outputs\alignment_threshold_optimizer"
set "DEFAULT_OPT_RUN=%DEFAULT_OPT_ROOT%\threshold_opt_20260802_042622"

set "SRC_CASES=%DEFAULT_CASES%"
if not exist "%SRC_CASES%" (
  echo [WARN] Default cases.jsonl not found:
  echo %SRC_CASES%
  set /p "SRC_CASES=Paste the full path to the 330-case cases.jsonl: "
)
if not exist "%SRC_CASES%" (
  echo [ERROR] cases.jsonl not found.
  pause
  exit /b 1
)

if not exist results mkdir results
copy /Y "%SRC_CASES%" "results\ragtruth_qa_330_cases.jsonl" >nul
if errorlevel 1 (
  echo [ERROR] Failed to copy cases.jsonl.
  pause
  exit /b 1
)

echo [OK] Copied authentic 330-case cases.jsonl.

if exist "%DEFAULT_OPT_ROOT%\latest_best_gate.json" (
  copy /Y "%DEFAULT_OPT_ROOT%\latest_best_gate.json" "config\optimized_gate_330_local.json" >nul
  echo [OK] Copied local optimized gate.
)
if exist "%DEFAULT_OPT_ROOT%\latest_summary.json" copy /Y "%DEFAULT_OPT_ROOT%\latest_summary.json" "results\threshold_optimizer_summary_local.json" >nul
if exist "%DEFAULT_OPT_ROOT%\latest_report.html" copy /Y "%DEFAULT_OPT_ROOT%\latest_report.html" "results\threshold_optimizer_report_local.html" >nul
if exist "%DEFAULT_OPT_RUN%\best_reprojected_cases.jsonl" copy /Y "%DEFAULT_OPT_RUN%\best_reprojected_cases.jsonl" "results\best_reprojected_cases_330.jsonl" >nul
if exist "%DEFAULT_OPT_RUN%\global_threshold_curve.csv" copy /Y "%DEFAULT_OPT_RUN%\global_threshold_curve.csv" "results\global_threshold_curve_330.csv" >nul
if exist "%DEFAULT_OPT_RUN%\top_configs.csv" copy /Y "%DEFAULT_OPT_RUN%\top_configs.csv" "results\top_threshold_configs_330.csv" >nul
if exist "%DEFAULT_OPT_RUN%\report.html" copy /Y "%DEFAULT_OPT_RUN%\report.html" "results\threshold_optimizer_report_330.html" >nul

if exist "%DEFAULT_CACHE%" (
  if not exist "outputs\ragtruth_raw_vs_dual_graph_nano" mkdir "outputs\ragtruth_raw_vs_dual_graph_nano"
  copy /Y "%DEFAULT_CACHE%" "outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json" >nul
  echo [OK] Copied local generation cache for resume. It remains git-ignored.
) else (
  echo [WARN] generation_cache_v040.json was not found. Code/results are still importable, but cached resume will not work until the cache is copied.
)

if not exist "outputs\alignment_threshold_optimizer" mkdir "outputs\alignment_threshold_optimizer"
copy /Y "config\optimized_gate_330.json" "outputs\alignment_threshold_optimizer\latest_best_gate.json" >nul

echo.
echo Import completed.
echo - Git-tracked result: results\ragtruth_qa_330_cases.jsonl
echo - Git-ignored cache: outputs\ragtruth_raw_vs_dual_graph_nano\generation_cache_v040.json
echo - Preset gate: config\optimized_gate_330.json
echo.
pause
