@echo off
echo MQTM Single Epoch Training System
echo ====================

set /p epoch_num="Enter the epoch number to run (e.g. 5): "

echo.
echo Starting training for epoch %epoch_num%...
echo.

python robust_mqtm_training.py --start_epoch %epoch_num% --num_epochs %epoch_num%+1 --data_dir="D:\INNOX\Crypto_Data" --models_dir="models\robust_training" --no_checkpoint

echo.
echo Epoch %epoch_num% completed.
echo.
echo To run the next epoch, run this batch file again and enter %epoch_num%+1
echo.

pause
