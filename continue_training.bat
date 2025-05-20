@echo off
echo MQTM Continuous Training System
echo ====================

echo Starting continuous training from epoch 3 to 50...
echo This will run without interruption until all epochs are completed.
echo Press Ctrl+C to stop training at any time.

echo Starting in 3 seconds...
timeout /t 3 > nul

python robust_mqtm_training.py --force_continue --start_epoch 3 --num_epochs 50 --data_dir="D:\INNOX\Crypto_Data" --models_dir="models\robust_training"

echo Training completed.
echo.
echo To evaluate performance, run test_model.py:
echo python test_model.py --data_dir=D:\INNOX\Crypto_Data --model_path=models\robust_training\best_model.pt --output_dir=test_results

pause
