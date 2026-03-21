@echo off
title AgriPrice Sentinel
color 0A
echo.
echo  ========================================
echo    AgriPrice Sentinel
echo  ========================================
echo.

:: Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found in PATH!
    pause
    exit /b 1
)

:: Check Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Node.js not found in PATH!
    pause
    exit /b 1
)

echo  [1/3] Starting FastAPI backend on port 8000...
start "FastAPI Backend" cmd /k "cd /d %~dp0 && python app.py"

echo  [2/3] Starting Next.js dashboard on port 3000...
start "Next.js Dashboard" cmd /k "cd /d %~dp0dashboard && npm run dev"

echo  [3/3] Waiting for servers to start...
timeout /t 5 /nobreak >nul

echo.
echo  ========================================
echo    All services started!
echo  ----------------------------------------
echo    Backend API:   http://localhost:8000
echo    Swagger Docs:  http://localhost:8000/docs
echo    Dashboard:     http://localhost:3000
echo    Alerts Page:   http://localhost:3000/alerts
echo  ========================================
echo.
echo  Opening dashboard in browser...
start http://localhost:3000

echo.
echo  Press any key to stop all services...
pause >nul

echo.
echo  Stopping services...
taskkill /FI "WINDOWTITLE eq FastAPI Backend*" /F >nul 2>&1
taskkill /FI "WINDOWTITLE eq Next.js Dashboard*" /F >nul 2>&1
echo  Done!
