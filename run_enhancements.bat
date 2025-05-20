@echo off
REM Batch file to run MQTM enhancements on Windows

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
    pip install -r mqtm_requirements.txt
) else (
    call venv\Scripts\activate.bat
)

REM Parse command line arguments
set MODELS_DIR=models
set OUTPUT_DIR=enhanced_models
set SYMBOLS=BTCUSDT ETHUSDT
set BATCH_SIZE=16

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--models_dir" (
    set MODELS_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--output_dir" (
    set OUTPUT_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--symbols" (
    set SYMBOLS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--batch_size" (
    set BATCH_SIZE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_optimization" (
    set SKIP_OPTIMIZATION=--skip_optimization
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_profiling" (
    set SKIP_PROFILING=--skip_profiling
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_dataloader" (
    set SKIP_DATALOADER=--skip_dataloader
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_mg" (
    set SKIP_MG=--skip_mg
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_tqe" (
    set SKIP_TQE=--skip_tqe
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_sp3" (
    set SKIP_SP3=--skip_sp3
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

REM Create output directory
if not exist %OUTPUT_DIR% mkdir %OUTPUT_DIR%

REM Create profiles directory
if not exist profiles mkdir profiles

REM Create visualizations directory
if not exist visualizations mkdir visualizations

REM Run enhancements
echo Running MQTM enhancements...
python run_enhancements.py --models_dir %MODELS_DIR% --output_dir %OUTPUT_DIR% --symbols %SYMBOLS% --batch_size %BATCH_SIZE% %SKIP_OPTIMIZATION% %SKIP_PROFILING% %SKIP_DATALOADER% %SKIP_MG% %SKIP_TQE% %SKIP_SP3%

REM Test enhanced models
echo Testing enhanced models...
python test_enhanced_models.py --original_dir %MODELS_DIR% --enhanced_dir %OUTPUT_DIR% --symbols %SYMBOLS% --batch_size %BATCH_SIZE%

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo Enhancements completed successfully!
echo See test_results directory for comparison visualizations.
