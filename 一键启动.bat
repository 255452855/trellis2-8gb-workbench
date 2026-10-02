@echo off
chcp 936 >nul
cd /d "%~dp0"

REM 自愈：修正移动后失效的路径和链接（静默，只有出错时才提示）
"%~dp0engine\tools\python\python.exe" "%~dp0_准备环境.py" >nul 2>&1
if errorlevel 1 (
    echo.
    echo 环境自检未通过，正在重新修复...
    "%~dp0engine\tools\python\python.exe" "%~dp0_准备环境.py"
    echo.
    pause
    exit /b 1
)
powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0_修复链接.ps1" >nul 2>&1

set XFORMERS_IGNORE_FLASH_VERSION_CHECK=1
set ATTN_BACKEND=sdpa
set OPENCV_IO_ENABLE_OPENEXR=1
set HF_HOME=%~dp0engine\code\MODELS
set HF_HUB_OFFLINE=1
set PYTHONUNBUFFERED=1

powershell -ExecutionPolicy Bypass -NoProfile -File "%~dp0_启动WSL后端.ps1"
set RC=%errorlevel%

echo.
if "%RC%"=="0" goto ok
echo   [启动失败] 完整过程日志：
echo   engine\code\output\启动日志.txt
echo.
pause
exit /b 1

:ok
echo   后端已在后台运行，本窗口可以直接关掉。
echo   页面地址： http://127.0.0.1:8080
echo   启动过程日志：engine\code\output\启动日志.txt
echo.
pause
