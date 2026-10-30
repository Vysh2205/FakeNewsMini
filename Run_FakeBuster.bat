@echo off
title FakeBuster AI Launcher
cd /d "%~dp0"

echo ===================================================
echo           Starting FakeBuster AI Platform
echo ===================================================
echo.

set KMP_DUPLICATE_LIB_OK=TRUE

echo [1/3] Starting FastAPI Backend Engine on Port 8000...
start "FakeBuster Backend Engine" cmd /k "C:\Users\vyshn\anaconda3\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000"

echo [2/3] Starting Cloudflare Public Tunnel...
start "FakeBuster Cloudflare Tunnel" cmd /k "cloudflared.exe tunnel --url http://127.0.0.1:8000"

echo [3/3] Starting Local Frontend Dev Server...
start "FakeBuster React UI" cmd /k "cd frontend && npx vite --host 127.0.0.1 --port 5173"

echo.
echo Waiting 5 seconds for services to initialize...
timeout /t 5 /nobreak >nul

echo Opening FakeBuster in your browser...
start http://127.0.0.1:5173
start https://fakenewsmini-fakebuster.surge.sh

echo.
echo ===================================================
echo FakeBuster is now running!
echo - Local Web App:     http://127.0.0.1:5173
echo - Live Cloud App:    https://fakenewsmini-fakebuster.surge.sh
echo - Local Backend API: http://127.0.0.1:8000
echo.
echo You can minimize these windows. Close them when you want to stop.
echo ===================================================
pause
