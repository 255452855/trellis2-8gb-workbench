@echo off
REM ==========================================================================
REM  One-click launcher for TRELLIS.2 / Pixal3D (image -> 3D, text -> 3D).
REM
REM  This file is intentionally 100% ASCII: cmd.exe reads batch files using the
REM  system OEM codepage, so UTF-8 Chinese text in a .bat turns into garbage and
REM  can even be executed as commands. All human-readable Chinese output is done
REM  by PowerShell (_start.ps1), which handles UTF-8 correctly.
REM
REM  First run on a fresh machine: this installs WSL + Ubuntu + CUDA toolkit +
REM  Python env + compiled extensions + ~55GB of model weights, then opens the
REM  browser UI. Expect 1-2 hours. Nothing is deleted on re-run; completed steps
REM  are skipped automatically.
REM
REM  Windows may need one reboot in the middle (only if WSL itself has to be
REM  enabled). If so, run this file again after booting.
REM ==========================================================================
setlocal
title TRELLIS2 one-click launcher
cd /d "%~dp0"

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0_start.ps1"
set RC=%ERRORLEVEL%

echo.
if "%RC%"=="0" goto ok
if "%RC%"=="9" goto reboot

echo   [FAILED] Exit code %RC%.
echo   Step-by-step log: engine\code\output\boot-log.txt
echo   Backend log:      engine\code\output\wsl-backend.log
echo                     engine\code\output\wsl-backend-error.log
echo.
echo   Re-run this file after fixing the cause - finished steps are skipped.
echo   To install only one stage, open the Linux shell and run it there:
echo     wsl -d Ubuntu-24.04
echo     cd /mnt/d/IDM/TRELLIS2     (your own path)
echo     bash ./_oneclick.sh --only weights
pause
exit /b %RC%

:reboot
echo.
echo   Windows features were enabled. Please REBOOT, then run this file again.
echo.
pause
exit /b 9

:ok
echo   Backend is running in the background. You can close this window.
echo   UI address: http://127.0.0.1:8080
echo   After editing code, use restart-backend.bat to reload it.
echo.
pause
exit /b 0
