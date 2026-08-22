@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo    LLM RG Search Launcher (uv)
echo ==================================================
echo.
echo   [1] v1 stack  server.py     agentic=v6a / fast=v2a / hybrid=v1
echo   [2] v2 stack  server_v2.py  agentic=v6b / fast=v2c / hybrid=v2
echo   [0] quit
echo.

choice /c 120 /n /m "Select stack to start: "
if errorlevel 3 goto quit
if errorlevel 2 goto stack2

:stack1
echo.
echo Starting v1 stack via uv ...
uv run server.py
goto end

:stack2
echo.
echo Starting v2 stack via uv ...
uv run server_v2.py
goto end

:end
echo.
echo Server exited.
pause

:quit
endlocal
