@echo off
setlocal
echo ====================================================
echo Starting Cloud Inference API Server (FastAPI)...
echo ====================================================

set "PYTHON_EXE="
if exist "%LOCALAPPDATA%\Programs\Python\Python314\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python314\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PYTHON_EXE if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" set "PYTHON_EXE=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
if not defined PYTHON_EXE if exist "C:\ProgramData\miniconda3\python.exe" set "PYTHON_EXE=C:\ProgramData\miniconda3\python.exe"
if not defined PYTHON_EXE if exist "%USERPROFILE%\miniconda3\python.exe" set "PYTHON_EXE=%USERPROFILE%\miniconda3\python.exe"
if not defined PYTHON_EXE (
    where python >nul 2>&1 && set "PYTHON_EXE=python"
)

if not defined PYTHON_EXE (
    echo [ERROR] No Python interpreter found!
    echo Please install Python or ensure it is on PATH.
    pause
    exit /b 1
)

echo Using Python: %PYTHON_EXE%
"%PYTHON_EXE%" -m uvicorn api_server:app --host 127.0.0.1 --port 8000
pause
