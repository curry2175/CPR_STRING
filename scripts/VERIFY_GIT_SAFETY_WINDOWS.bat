@echo off
setlocal
cd /d "%~dp0\.."
echo Checking tracked-style project files for common secret/cache names...
for %%P in (.env .venv local_cache) do (
  if exist "%%P" echo [INFO] Local %%P exists but should remain ignored.
)
where git >nul 2>&1
if errorlevel 1 (
  echo Git is not installed or not in PATH.
  pause
  exit /b 1
)
echo.
git status --short
echo.
echo Suspicious tracked files, if any:
git ls-files | findstr /i /c:".env" /c:"generation_cache" /c:".venv" /c:"OPENAI_API_KEY"
echo.
echo No output above is expected after git add.
pause
