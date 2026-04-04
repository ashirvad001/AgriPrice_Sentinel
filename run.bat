@echo off
setlocal enabledelayedexpansion

REM Ensure we are in the correct directory even if there are spaces in the path
cd /d "%~dp0"

title AgriPrice Sentinel
color 0A
echo.
echo  ========================================
echo    AgriPrice Sentinel
echo  ========================================
echo.

REM Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

REM Check Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found in PATH!
    pause
    exit /b 1
)

echo  [1/3] Starting FastAPI backend on port 8000...
start "Sentinel-API" cmd /k "call venv\Scripts\activate && python app.py"

echo  [2/3] Starting Next.js dashboard on port 3000...
start "Sentinel-Dashboard" cmd /k "cd /d dashboard && npm run dev"

echo  [3/3] Waiting for servers to start...
timeout /t 10 /nobreak >nul

echo.
echo  ========================================
echo    All services started!
echo  ----------------------------------------
echo    Backend API:   http://localhost:8000
echo    Swagger Docs:  http://localhost:8000/docs
echo    Dashboard:     http://localhost:3000
echo  ========================================
echo.
echo  Opening dashboard in browser...
start http://localhost:3000

echo.
echo  Press any key to stop all services...
pause >nul

echo.
echo  Stopping services...
taskkill /FI "WINDOWTITLE eq Sentinel-*" /F >nul 2>&1
echo  Done!
