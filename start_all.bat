@echo off
setlocal enabledelayedexpansion

REM Ensure we are in the correct directory even if there are spaces in the path
cd /d "%~dp0"

REM ============================================================================
REM  AgriPrice Sentinel - All-in-One Launcher
REM ============================================================================

title AgriPrice Sentinel - Launcher
color 0B
echo.
echo  [SYSTEM] Checking prerequisites...

REM 1. Check Python
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Python is not installed or not in PATH!
    pause
    exit /b 1
)

REM 2. Check Node
where node >nul 2>&1
if %errorlevel% neq 0 (
    echo  [ERROR] Node.js is not installed or not in PATH!
    pause
    exit /b 1
)

REM 3. Check Docker
where docker >nul 2>&1
if %errorlevel% neq 0 (
    echo  [WARN] Docker not found! Skipping Infrastructure auto-start.
) else (
    echo  [INFO] Docker detected. Starting Infrastructure [Postgres and Redis]...
    docker-compose up -d postgres redis
    if %errorlevel% neq 0 (
        echo  [WARN] Failed to start Docker services.
    )
)

echo.
echo  [STEP 1/3] Launching Backend Services...

REM --- Backend (FastAPI) ---
echo  [API] Starting FastAPI on port 8000...
start "Sentinel-API" cmd /k "color 0E && call venv\Scripts\activate && python app.py"

REM --- Scheduler (Main) ---
echo  [SCHEDULER] Starting Task Scheduler...
start "Sentinel-Scheduler" cmd /k "color 3F && call venv\Scripts\activate && python main.py"

echo.
echo  [STEP 2/3] Launching Frontend Dashboard...

REM --- Frontend (Next.js) ---
echo  [DASHBOARD] Starting Next.js on port 3000...
start "Sentinel-Dashboard" cmd /k "color 09 && cd /d dashboard && npm run dev"

echo.
echo  [STEP 3/3] Finalizing...
echo  Waiting for servers to initialize [10s]...
timeout /t 10 /nobreak >nul

echo.
echo  ==============================================================
echo   SERVICES SUMMARY:
echo  --------------------------------------------------------------
echo    - Dashboard:     http://localhost:3000
echo    - Backend API:   http://localhost:8000
echo    - API Docs:      http://localhost:8000/docs
echo  ==============================================================
echo.

REM --- Launch Browser ---
echo  Opening dashboard in your default browser...
start http://localhost:3000

echo.
echo  [SUCCESS] All services are launching in separate windows.
echo  To stop them, close the terminal windows.
echo.
pause
