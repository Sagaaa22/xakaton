@echo off
setlocal EnableDelayedExpansion

set "TARGET=%~1"
set "IN_FILE=tickets.jsonl"
set "OUT_FILE=pred.jsonl"

:parse_args
shift
if "%~1"=="" goto done_args
if /I "%~1"=="IN" (
    set "IN_FILE=%~2"
    shift
    goto parse_args
)
if /I "%~1"=="OUT" (
    set "OUT_FILE=%~2"
    shift
    goto parse_args
)
set "ARG=%~1"
if /I "!ARG:~0,3!"=="IN=" set "IN_FILE=!ARG:~3!"
if /I "!ARG:~0,4!"=="OUT=" set "OUT_FILE=!ARG:~4!"
goto parse_args

:done_args

if /I "%TARGET%"=="predict" (
    python -m app.cli predict --in "%IN_FILE%" --out "%OUT_FILE%"
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="train" (
    python -m app.cli train
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="evaluate" (
    python -m app.cli evaluate --out "%OUT_FILE%"
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="test" (
    python -m unittest discover -s tests -p "test_*.py" -v
    exit /b %ERRORLEVEL%
)
if /I "%TARGET%"=="run" (
    python -m uvicorn app.main:app --host 127.0.0.1 --port 8000
    exit /b %ERRORLEVEL%
)

echo Usage: make [predict IN=tickets.jsonl OUT=pred.jsonl ^| train ^| evaluate ^| test ^| run]
exit /b 1
