@echo off
echo ===================================================
echo   Starting FakeBuster AI Backend & Frontend Servers
echo ===================================================
echo.

start "FakeBuster FastAPI Backend" cmd /k "cd backend && python -m uvicorn main:app --reload --port 8000"
timeout /t 3 >nul

start "FakeBuster React Frontend" cmd /k "cd frontend && npm run dev"

echo.
echo FakeBuster servers launched!
echo - Backend: http://127.0.0.1:8000
echo - Frontend: http://localhost:5173
echo ===================================================
pause
