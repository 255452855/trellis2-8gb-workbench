@echo off
REM ==========================================================================
REM  Restart the 3D backend so edited code (app.py / pipeline_worker.py) takes
REM  effect. start.bat will NOT kill a running backend - it just opens the
REM  existing page - which is why this separate entry exists.
REM
REM  ASCII only, see the note in start.bat.
REM ==========================================================================
setlocal
title TRELLIS2 backend restart
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_start.ps1" -Restart
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" goto ok
echo   [FAILED] Exit code %RC%.
echo   Step-by-step log: engine\code\output\boot-log.txt
echo   Backend log:      engine\code\output\wsl-backend-error.log
echo.
echo   If the old process refuses to die, stop it manually:
echo     wsl -d Ubuntu-24.04
echo     bash ./_stop_3d_backend.sh
pause
exit /b %RC%

:ok
echo   Backend restarted. UI: http://127.0.0.1:8080
echo.
pause
exit /b 0
