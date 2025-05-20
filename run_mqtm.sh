#!/bin/bash
# Shell script to run MQTM system on Unix/Linux

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python is not installed."
    echo "Please install Python 3.8 or higher."
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    echo "Installing dependencies..."
    pip install -r requirements.txt
else
    source venv/bin/activate
fi

# Parse command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: ./run_mqtm.sh COMMAND [OPTIONS]"
    echo ""
    echo "Commands:"
    echo "  data       Download and process data"
    echo "  train      Train MQTM models"
    echo "  generate   Generate synthetic data"
    echo "  visualize  Visualize MQTM components"
    echo "  trade      Run trading example"
    echo "  pipeline   Run complete MQTM pipeline"
    echo ""
    echo "Run './run_mqtm.sh COMMAND --help' for more information on a command."
    exit 0
fi

# Run the appropriate command
python mqtm_cli.py "$@"

# Deactivate virtual environment
deactivate
