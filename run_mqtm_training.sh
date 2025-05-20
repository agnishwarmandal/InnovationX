#!/bin/bash
# Shell script to run MQTM training pipeline on Unix/Linux

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
DATA_DIR="D:/INNOX/Crypto_Data"
MODELS_DIR="models"
SYMBOLS=""
BATCH_SIZE=32
EPOCHS=50
LEARNING_RATE=0.0001
NUM_EPISODES=1000
NUM_ITERATIONS=1000
OPTIMIZE_MEMORY=""
PROFILE_PERFORMANCE=""
SKIP_MQTM=""
SKIP_ASP=""
SKIP_MGI=""
SKIP_BOM=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --data_dir)
            DATA_DIR="$2"
            shift 2
            ;;
        --models_dir)
            MODELS_DIR="$2"
            shift 2
            ;;
        --symbols)
            SYMBOLS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --learning_rate)
            LEARNING_RATE="$2"
            shift 2
            ;;
        --num_episodes)
            NUM_EPISODES="$2"
            shift 2
            ;;
        --num_iterations)
            NUM_ITERATIONS="$2"
            shift 2
            ;;
        --optimize_memory)
            OPTIMIZE_MEMORY="--optimize_memory"
            shift
            ;;
        --profile_performance)
            PROFILE_PERFORMANCE="--profile_performance"
            shift
            ;;
        --skip_mqtm)
            SKIP_MQTM="--skip_mqtm"
            shift
            ;;
        --skip_asp)
            SKIP_ASP="--skip_asp"
            shift
            ;;
        --skip_mgi)
            SKIP_MGI="--skip_mgi"
            shift
            ;;
        --skip_bom)
            SKIP_BOM="--skip_bom"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            shift
            ;;
    esac
done

# Create output directories
mkdir -p "$MODELS_DIR"
mkdir -p "$MODELS_DIR/asp"
mkdir -p "$MODELS_DIR/mgi"
mkdir -p "$MODELS_DIR/bom"
mkdir -p profiles
mkdir -p visualizations

# Run MQTM training pipeline
echo "Running MQTM training pipeline..."
python run_mqtm_training.py --data_dir "$DATA_DIR" --models_dir "$MODELS_DIR" --symbols $SYMBOLS --batch_size "$BATCH_SIZE" --epochs "$EPOCHS" --learning_rate "$LEARNING_RATE" --num_episodes "$NUM_EPISODES" --num_iterations "$NUM_ITERATIONS" $OPTIMIZE_MEMORY $PROFILE_PERFORMANCE $SKIP_MQTM $SKIP_ASP $SKIP_MGI $SKIP_BOM

# Evaluate trained models
echo "Evaluating trained models..."
python evaluate_mqtm_system.py --data_dir "$DATA_DIR" --models_dir "$MODELS_DIR" --output_dir evaluation_results --symbols $SYMBOLS --batch_size "$BATCH_SIZE" --use_asp --use_mgi --use_bom $OPTIMIZE_MEMORY

# Deactivate virtual environment
deactivate

echo "MQTM training pipeline completed!"
echo "See evaluation_results directory for evaluation results."
