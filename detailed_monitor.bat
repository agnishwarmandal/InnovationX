@echo off
echo Starting detailed monitoring of MQTM training...
python detailed_monitor.py --interval=5 --models_dir=models\robust_training --output_dir=monitor_output --log_file=robust_training.log
