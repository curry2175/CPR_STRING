@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run RUN_WINDOWS.bat once first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -u run_ragtruth_localization.py --download-only
set EXITCODE=%ERRORLEVEL%
echo.
if %EXITCODE% EQU 0 (
  echo RAGTruth download completed.
) else (
  echo RAGTruth download failed with exit code %EXITCODE%.
)
pause
exit /b %EXITCODE%
