@echo off
setlocal
cd /d "%~dp0"
title WenSheng3D - FLUX.2-klein-4B + TRELLIS.2/Pixal3D

echo ================================================================
echo   WenSheng3D one-click package
echo   prompt -^> FLUX.2-klein-4B -^> image-to-3D -^> GLB
echo   Chinese status messages are printed by the WSL side below.
echo ================================================================
echo.

REM ---------- 0. locate WSL and project path ----------
set "ROOT=%~dp0"
set "ROOT=%ROOT:~0,-1%"
set "WROOT="
for /f "usebackq delims=" %%i in (`wsl.exe -d Ubuntu-24.04 -- wslpath -a "%ROOT%"`) do set "WROOT=%%i"
if "%WROOT%"=="" (
  echo [ERROR] cannot reach the WSL distro Ubuntu-24.04.
  echo         try: wsl -d Ubuntu-24.04 -- echo ok
  pause
  exit /b 1
)
echo Project dir Windows : %ROOT%
echo Project dir WSL     : %WROOT%
echo.

REM ---------- 1. prepare text-to-image env (idempotent; ~24GB first time) ----------
echo [1/5] Preparing text-to-image environment ...
echo       First run installs dependencies and downloads FLUX.2-klein-4B
echo       from ModelScope (about 24GB). Already-downloaded files are skipped.
echo.
wsl.exe -d Ubuntu-24.04 -- bash %WROOT%/_t2i_setup.sh
if errorlevel 1 (
  echo.
  echo [ERROR] environment setup failed, please send the output above.
  pause
  exit /b 1
)

REM ---------- 2. stop the 3D backend, give VRAM/RAM to FLUX ----------
echo.
echo [2/5] Pausing the 3D backend to free VRAM/RAM for text-to-image ...
wsl.exe -d Ubuntu-24.04 -- bash -lc "pkill -f 'ap[p].py' ; sleep 2 ; echo   done"
echo.

REM ---------- 3. text-to-image (prompt typed in this window) ----------
echo [3/5] Text-to-image (512x512 / 4 steps, about 6 minutes) ...
echo.
wsl.exe -d Ubuntu-24.04 -- bash %WROOT%/_t2i_run.sh t2i
if errorlevel 1 (
  echo.
  echo [ERROR] text-to-image failed.
  pause
  exit /b 1
)

REM ---------- 4. start the 3D backend ----------
echo.
echo [4/5] Starting the 3D backend ...
powershell -NoProfile -ExecutionPolicy Bypass -File "%ROOT%\_start_backend.ps1"

REM ---------- 5. image-to-3D and export GLB ----------
echo.
echo [5/5] Image-to-3D (about 8 minutes) ...
echo.
wsl.exe -d Ubuntu-24.04 -- bash %WROOT%/_t2i_run.sh to3d

echo.
echo ================================================================
echo DONE. Image: output\last-t2i.png    GLB: output\*.glb
echo ================================================================
pause
