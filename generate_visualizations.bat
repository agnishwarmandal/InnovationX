@echo off
echo Generating detailed visualizations of training progress...
python generate_visualizations.py --models_dir=models\robust_training --output_dir=monitor_output
