@echo off
setlocal
cd /d "%~dp0"
if exist ".env" (
  echo .env already exists.
  echo Open it in Notepad if you need to replace the key.
  notepad .env
  exit /b 0
)
copy /Y ".env.example" ".env" >nul
notepad .env
echo.
echo Save the file, close Notepad, then restart RUN_WINDOWS.bat.
pause
endlocal
