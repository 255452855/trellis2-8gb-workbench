@echo off
cd /d "%~dp0.\..\tools"

:: Set HF_HOME to be inside 'code\models'. 
:: %~dp0 resolves to the current BAT file's folder (run-projectorz).
:: We go up one level (root), then into 'code', then 'models'.
set "HF_HOME=%~dp0..\code\models"

:: Ensure the directory exists to avoid any weird write errors on first run
if not exist "%HF_HOME%" mkdir "%HF_HOME%"

:: Run the internal launcher
:: You can also specify --ip and --port, for example:  call gradio-internal.bat --ip 127.0.0.1 --port 5555
call projectorz-internal.bat