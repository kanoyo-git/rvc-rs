@echo off
chcp 65001 >nul
setlocal
set "SCRIPT_DIR=%~dp0"
set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"
cd /d "%SCRIPT_DIR%"
set "PATH=%SCRIPT_DIR%;%SCRIPT_DIR%\runtime;%PATH%"
if defined LIBTORCH set "PATH=%LIBTORCH%\lib;%PATH%"
set "RVC_EXE=%SCRIPT_DIR%\rvc.exe"
if not exist "%RVC_EXE%" set "RVC_EXE=%SCRIPT_DIR%\target\release\rvc.exe"
if not exist "%RVC_EXE%" (
    echo 请先按 README.md 中的说明构建 rvc.exe。
    pause
    exit /b 1
)
"%RVC_EXE%" serve --port 7898 --open --panel realtime %*
set "RVC_STATUS=%ERRORLEVEL%"
pause
exit /b %RVC_STATUS%
