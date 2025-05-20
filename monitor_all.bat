@echo off
echo Starting comprehensive monitoring of MQTM training...

REM Start detailed monitoring in a separate window
start "Detailed Monitor" cmd /c "detailed_monitor.bat"

REM Start live training display in a separate window
start "Live Training Display" cmd /c "live_display.bat"

REM Generate visualizations every 5 minutes
:loop
echo Generating visualizations...
call generate_visualizations.bat
echo Waiting 5 minutes before next visualization update...
timeout /t 300
goto loop
