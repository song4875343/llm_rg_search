@echo off
chcp 65001 >nul 2>&1
setlocal
cd /d "%~dp0"

echo ==================================================
echo    LLM RG Search Launcher ^(uv^)
echo ==================================================
echo.

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv not found - https://docs.astral.sh/uv/
    echo         install: powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 ^| iex"
    echo         or: pip install uv
    echo.
    pause
    exit /b 1
)

echo [uv] Python .python-version requires-python ^>=3.11
echo [uv] Syncing dependencies ...

if exist uv.lock (
    uv sync --frozen
    if errorlevel 1 (
        echo [WARN] frozen sync failed, updating lock ...
        uv sync
    )
) else (
    uv sync
)

if errorlevel 1 (
    echo [ERROR] uv sync failed - check pyproject.toml or network
    pause
    exit /b 1
)

echo [uv] Environment ready - .venv
echo.

echo   [1] v1 stack  server.py     agentic=v6a / fast=v2a / hybrid=v1
echo   [2] v2 stack  server_v2.py  agentic=v6b / fast=v2c / hybrid=v2
echo   [0] quit
echo.

choice /c 120 /n /m "Select stack to start [1/2/0]: "
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
exit /b 0

:quit
endlocal
exit /b 0
