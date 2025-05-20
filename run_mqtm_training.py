"""
Script to run the complete MQTM system training pipeline.
"""

import os
import argparse
import logging
import time
import json
import subprocess
import sys
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mqtm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MQTM Training Pipeline")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to train on (if None, use all available)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes for ASP training")
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="Number of iterations for MGI and BOM training")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Whether to optimize memory usage")
    parser.add_argument("--profile_performance", action="store_true",
                        help="Whether to profile performance")
    parser.add_argument("--skip_mqtm", action="store_true",
                        help="Skip MQTM model training")
    parser.add_argument("--skip_asp", action="store_true",
                        help="Skip ASP framework training")
    parser.add_argument("--skip_mgi", action="store_true",
                        help="Skip MGI module training")
    parser.add_argument("--skip_bom", action="store_true",
                        help="Skip BOM module training")
    
    return parser.parse_args()

def run_command(command, description):
    """Run a command and log output."""
    logger.info(f"Running {description}...")
    logger.info(f"Command: {' '.join(command)}")
    
    start_time = time.time()
    
    try:
        # Run command
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output
        for line in iter(process.stdout.readline, ""):
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if return_code == 0:
            logger.info(f"{description} completed successfully in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.error(f"{description} failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"{description} failed with exception: {e}")
        return False

def train_mqtm_models(args):
    """Train MQTM models."""
    # Build command
    command = [
        sys.executable, "train_mqtm_system.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--train_all",
    ]
    
    # Add symbols if specified
    if args.symbols:
        command.extend(["--symbols"] + args.symbols)
    
    # Add optimization flags if specified
    if args.optimize_memory:
        command.append("--optimize_memory")
    
    if args.profile_performance:
        command.append("--profile_performance")
    
    # Run command
    return run_command(command, "MQTM Model Training")

def train_asp_framework(args):
    """Train ASP framework."""
    # Build command
    command = [
        sys.executable, "train_asp_framework.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--asp_dir", os.path.join(args.models_dir, "asp"),
        "--batch_size", str(args.batch_size),
        "--learning_rate", str(args.learning_rate),
        "--num_episodes", str(args.num_episodes),
        "--reward_type", "pnl",
        "--risk_aversion", "0.1",
        "--max_position_size", "10",
        "--stop_loss_pct", "0.02",
        "--take_profit_pct", "0.05",
        "--kl_budget", "0.05",
        "--perturbation_strength", "0.1",
        "--use_regime_detection",
        "--use_causal_model",
    ]
    
    # Add symbols if specified
    if args.symbols:
        command.extend(["--symbols"] + args.symbols)
    
    # Add optimization flags if specified
    if args.optimize_memory:
        command.append("--optimize_memory")
    
    if args.profile_performance:
        command.append("--profile_performance")
    
    # Run command
    return run_command(command, "ASP Framework Training")

def train_mgi_module(args):
    """Train MGI module."""
    # Build command
    command = [
        sys.executable, "train_mgi_module.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--asp_dir", os.path.join(args.models_dir, "asp"),
        "--mgi_dir", os.path.join(args.models_dir, "mgi"),
        "--meta_batch_size", str(args.batch_size),
        "--num_iterations", str(args.num_iterations),
        "--meta_lr", "1e-3",
        "--use_adaptive_lr",
        "--optimize_hyperparams", "learning_rate", "risk_aversion", "stop_loss", "take_profit",
    ]
    
    # Add symbols if specified
    if args.symbols:
        command.extend(["--symbols"] + args.symbols)
    
    # Add optimization flags if specified
    if args.optimize_memory:
        command.append("--optimize_memory")
    
    if args.profile_performance:
        command.append("--profile_performance")
    
    # Run command
    return run_command(command, "MGI Module Training")

def train_bom_module(args):
    """Train BOM module."""
    # Build command
    command = [
        sys.executable, "train_bom_module.py",
        "--data_dir", args.data_dir,
        "--models_dir", args.models_dir,
        "--asp_dir", os.path.join(args.models_dir, "asp"),
        "--bom_dir", os.path.join(args.models_dir, "bom"),
        "--batch_size", str(args.batch_size),
        "--num_iterations", str(args.num_iterations),
        "--num_models", "3",
        "--hidden_dims", "128", "128",
        "--prior_std", "0.1",
        "--learning_rate", "1e-3",
    ]
    
    # Add symbols if specified
    if args.symbols:
        command.extend(["--symbols"] + args.symbols)
    
    # Add optimization flags if specified
    if args.optimize_memory:
        command.append("--optimize_memory")
    
    if args.profile_performance:
        command.append("--profile_performance")
    
    # Run command
    return run_command(command, "BOM Module Training")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directories
    os.makedirs(args.models_dir, exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "asp"), exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "mgi"), exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "bom"), exist_ok=True)
    
    # Create profiles directory
    os.makedirs("profiles", exist_ok=True)
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Train MQTM models
    if not args.skip_mqtm:
        if not train_mqtm_models(args):
            logger.warning("MQTM model training failed, continuing with other steps")
    else:
        logger.info("Skipping MQTM model training")
    
    # Train ASP framework
    if not args.skip_asp:
        if not train_asp_framework(args):
            logger.warning("ASP framework training failed, continuing with other steps")
    else:
        logger.info("Skipping ASP framework training")
    
    # Train MGI module
    if not args.skip_mgi:
        if not train_mgi_module(args):
            logger.warning("MGI module training failed, continuing with other steps")
    else:
        logger.info("Skipping MGI module training")
    
    # Train BOM module
    if not args.skip_bom:
        if not train_bom_module(args):
            logger.warning("BOM module training failed")
    else:
        logger.info("Skipping BOM module training")
    
    logger.info("MQTM training pipeline completed")

if __name__ == "__main__":
    main()
