#!/bin/bash
# Shell script to run MQTM enhancements on Unix/Linux

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
    pip install -r mqtm_requirements.txt
else
    source venv/bin/activate
fi

# Parse command line arguments
MODELS_DIR="models"
OUTPUT_DIR="enhanced_models"
SYMBOLS=("BTCUSDT" "ETHUSDT")
BATCH_SIZE=16
SKIP_OPTIMIZATION=""
SKIP_PROFILING=""
SKIP_DATALOADER=""
SKIP_MG=""
SKIP_TQE=""
SKIP_SP3=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --symbols)
            SYMBOLS=("$2")
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --skip_optimization)
            SKIP_OPTIMIZATION="--skip_optimization"
            shift
            ;;
        --skip_profiling)
            SKIP_PROFILING="--skip_profiling"
            shift
            ;;
        --skip_dataloader)
            SKIP_DATALOADER="--skip_dataloader"
            shift
            ;;
        --skip_mg)
            SKIP_MG="--skip_mg"
            shift
            ;;
        --skip_tqe)
            SKIP_TQE="--skip_tqe"
            shift
            ;;
        --skip_sp3)
            SKIP_SP3="--skip_sp3"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Create profiles directory
mkdir -p profiles

# Create visualizations directory
mkdir -p visualizations

# Run enhancements
echo "Running MQTM enhancements..."
python run_enhancements.py --models_dir "$MODELS_DIR" --output_dir "$OUTPUT_DIR" --symbols "${SYMBOLS[@]}" --batch_size "$BATCH_SIZE" $SKIP_OPTIMIZATION $SKIP_PROFILING $SKIP_DATALOADER $SKIP_MG $SKIP_TQE $SKIP_SP3

# Test enhanced models
echo "Testing enhanced models..."
python test_enhanced_models.py --original_dir "$MODELS_DIR" --enhanced_dir "$OUTPUT_DIR" --symbols "${SYMBOLS[@]}" --batch_size "$BATCH_SIZE"

# Deactivate virtual environment
deactivate

echo "Enhancements completed successfully!"
echo "See test_results directory for comparison visualizations."
