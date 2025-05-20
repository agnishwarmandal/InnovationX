@echo off
echo MQTM Testing System
echo ==================
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

REM Check if model exists
if not exist "models\robust_training\best_model.pt" (
    echo Error: Model file not found.
    echo Please run training first or specify a different model path.
    exit /b 1
)

REM Create directories
if not exist "test_results" mkdir test_results

REM Start testing
echo.
echo Starting MQTM testing on all datasets...
echo.
echo This will test the model on all 136 datasets in D:\INNOX\Crypto_Data
echo Results will be saved to test_results directory
echo.
echo Press Ctrl+C to stop testing at any time
echo.
echo Starting in 3 seconds...
timeout /t 3 >nul

REM Run testing
python test_model.py --data_dir=D:\INNOX\Crypto_Data --model_path=models\robust_training\best_model.pt --output_dir=test_results

echo.
echo Testing completed.
echo Results are available in the test_results directory.
echo.

pause
