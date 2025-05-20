@echo off
REM Batch file to run MQTM training pipeline on Windows

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
set DATA_DIR=D:\INNOX\Crypto_Data
set MODELS_DIR=models
set SYMBOLS=
set BATCH_SIZE=32
set EPOCHS=50
set LEARNING_RATE=0.0001
set NUM_EPISODES=1000
set NUM_ITERATIONS=1000
set OPTIMIZE_MEMORY=
set PROFILE_PERFORMANCE=
set SKIP_MQTM=
set SKIP_ASP=
set SKIP_MGI=
set SKIP_BOM=

:parse_args
if "%~1"=="" goto :end_parse_args
if /i "%~1"=="--data_dir" (
    set DATA_DIR=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--models_dir" (
    set MODELS_DIR=%~2
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
if /i "%~1"=="--epochs" (
    set EPOCHS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--learning_rate" (
    set LEARNING_RATE=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--num_episodes" (
    set NUM_EPISODES=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--num_iterations" (
    set NUM_ITERATIONS=%~2
    shift
    shift
    goto :parse_args
)
if /i "%~1"=="--optimize_memory" (
    set OPTIMIZE_MEMORY=--optimize_memory
    shift
    goto :parse_args
)
if /i "%~1"=="--profile_performance" (
    set PROFILE_PERFORMANCE=--profile_performance
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_mqtm" (
    set SKIP_MQTM=--skip_mqtm
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_asp" (
    set SKIP_ASP=--skip_asp
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_mgi" (
    set SKIP_MGI=--skip_mgi
    shift
    goto :parse_args
)
if /i "%~1"=="--skip_bom" (
    set SKIP_BOM=--skip_bom
    shift
    goto :parse_args
)
shift
goto :parse_args
:end_parse_args

REM Create output directories
if not exist %MODELS_DIR% mkdir %MODELS_DIR%
if not exist %MODELS_DIR%\asp mkdir %MODELS_DIR%\asp
if not exist %MODELS_DIR%\mgi mkdir %MODELS_DIR%\mgi
if not exist %MODELS_DIR%\bom mkdir %MODELS_DIR%\bom
if not exist profiles mkdir profiles
if not exist visualizations mkdir visualizations

REM Run MQTM training pipeline
echo Running MQTM training pipeline...
python run_mqtm_training.py --data_dir %DATA_DIR% --models_dir %MODELS_DIR% --symbols %SYMBOLS% --batch_size %BATCH_SIZE% --epochs %EPOCHS% --learning_rate %LEARNING_RATE% --num_episodes %NUM_EPISODES% --num_iterations %NUM_ITERATIONS% %OPTIMIZE_MEMORY% %PROFILE_PERFORMANCE% %SKIP_MQTM% %SKIP_ASP% %SKIP_MGI% %SKIP_BOM%

REM Evaluate trained models
echo Evaluating trained models...
python evaluate_mqtm_system.py --data_dir %DATA_DIR% --models_dir %MODELS_DIR% --output_dir evaluation_results --symbols %SYMBOLS% --batch_size %BATCH_SIZE% --use_asp --use_mgi --use_bom %OPTIMIZE_MEMORY%

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat

echo MQTM training pipeline completed!
echo See evaluation_results directory for evaluation results.
