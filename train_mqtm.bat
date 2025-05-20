@echo off
echo MQTM Training System
echo ====================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Error: Python is not installed or not in PATH.
    echo Please install Python 3.8 or higher.
    exit /b 1
)

REM Check if required packages are installed
echo Checking required packages...
pip install -r requirements.txt

REM Create directories
if not exist "models" mkdir models
if not exist "models\robust_training" mkdir models\robust_training
if not exist "monitor_output" mkdir monitor_output
if not exist "test_results" mkdir test_results

REM Start training
echo.
echo Starting MQTM training on all datasets...
echo.
echo This will use all 136 datasets in D:\INNOX\Crypto_Data
echo Training will be robust against numerical instability
echo Progress will be displayed in real-time
echo.
echo Press Ctrl+C to stop training at any time
echo.
echo Starting in 5 seconds...
timeout /t 5 >nul

REM Run training in the current window to see any error messages
echo Running training in the current window...
echo.
python robust_mqtm_training.py --data_dir=D:\INNOX\Crypto_Data --models_dir=models\robust_training --batch_size=16 --epochs=50 --learning_rate=1e-5

echo.
echo Training completed.
echo.
echo To evaluate performance, run test_model.py:
echo python test_model.py --data_dir=D:\INNOX\Crypto_Data --model_path=models\robust_training\best_model.pt --output_dir=test_results
echo.

pause
