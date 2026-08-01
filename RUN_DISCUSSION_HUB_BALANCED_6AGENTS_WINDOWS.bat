@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Python environment not found. Run RUN_WINDOWS.bat once or setup.bat first.
  pause
  exit /b 1
)
set DISCUSSION_ARCHITECTURE=graph_native_6agents_balanced
start "" http://127.0.0.1:8765/discussion-lab
".venv\Scripts\python.exe" -m uvicorn app:app --host 127.0.0.1 --port 8765
endlocal
