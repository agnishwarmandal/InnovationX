@echo off
REM Batch file to run MQTM system on Windows

REM Check if Python is installed
where python >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if virtual environment exists
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
    call venv\Scripts\activate.bat
    echo Installing dependencies...
    pip install -r requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Parse command line arguments
if "%1"=="" (
    echo Usage: run_mqtm.bat COMMAND [OPTIONS]
    echo.
    echo Commands:
    echo   data       Download and process data
    echo   train      Train MQTM models
    echo   generate   Generate synthetic data
    echo   visualize  Visualize MQTM components
    echo   trade      Run trading example
    echo   pipeline   Run complete MQTM pipeline
    echo.
    echo Run 'run_mqtm.bat COMMAND --help' for more information on a command.
    exit /b 0
)

REM Run the appropriate command
python mqtm_cli.py %*

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat
