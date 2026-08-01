@echo off
setlocal
cd /d "%~dp0"
if exist ".venv\Scripts\python.exe" (
  ".venv\Scripts\python.exe" -u report_optimized_rescues.py --root "."
) else (
  python -u report_optimized_rescues.py --root "."
)
set EXITCODE=%ERRORLEVEL%
echo.
if not "%EXITCODE%"=="0" echo Rescue report failed with exit code %EXITCODE%.
pause
exit /b %EXITCODE%
