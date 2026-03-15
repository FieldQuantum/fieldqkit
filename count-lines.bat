@echo off
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
powershell -NoProfile -ExecutionPolicy Bypass -File "%SCRIPT_DIR%\scripts\count-lines.ps1" -Root "%SCRIPT_DIR%"
pause