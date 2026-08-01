@echo off
setlocal
cd /d "%~dp0"
echo ==== Verified Reasoning Graph diagnostics ====
echo Current folder: %CD%
where python
python --version
if exist ".venv\Scripts\python.exe" (
  echo.
  echo ==== Virtual environment ====
  ".venv\Scripts\python.exe" --version
  ".venv\Scripts\python.exe" -c "import fastapi, uvicorn; print('fastapi', fastapi.__version__); print('uvicorn', uvicorn.__version__); import z3; print('z3', z3.get_version_string())"
  echo.
  echo ==== Tests ====
  ".venv\Scripts\python.exe" -m pip install pytest
  ".venv\Scripts\python.exe" -m pytest -q
) else (
  echo .venv does not exist. Run RUN_WINDOWS.bat first.
)
echo.
echo Copy all output above and send it if something fails.
pause
endlocal
