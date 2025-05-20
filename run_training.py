"""
Script to run the robust MQTM training with all available datasets.
"""

import os
import argparse
import subprocess
import psutil
import time
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("run_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_gpu_memory_info():
    """Get GPU memory information."""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
            gpu_memory_allocated = torch.cuda.memory_allocated(0)
            gpu_memory_reserved = torch.cuda.memory_reserved(0)
            gpu_memory_free = gpu_memory_total - gpu_memory_reserved
            
            return {
                "total": gpu_memory_total / (1024 ** 3),  # GB
                "allocated": gpu_memory_allocated / (1024 ** 3),  # GB
                "reserved": gpu_memory_reserved / (1024 ** 3),  # GB
                "free": gpu_memory_free / (1024 ** 3),  # GB
            }
        else:
            return {"error": "CUDA not available"}
    except Exception as e:
        return {"error": str(e)}

def get_system_info():
    """Get system information."""
    cpu_percent = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    
    gpu_info = get_gpu_memory_info()
    
    return {
        "cpu_percent": cpu_percent,
        "memory_percent": memory.percent,
        "memory_available_gb": memory.available / (1024 ** 3),
        "gpu": gpu_info,
    }

def log_system_info():
    """Log system information."""
    info = get_system_info()
    
    logger.info(f"CPU Usage: {info['cpu_percent']}%")
    logger.info(f"Memory Usage: {info['memory_percent']}% "
                f"(Available: {info['memory_available_gb']:.2f} GB)")
    
    if "error" not in info["gpu"]:
        logger.info(f"GPU Memory: Total={info['gpu']['total']:.2f} GB, "
                    f"Allocated={info['gpu']['allocated']:.2f} GB, "
                    f"Reserved={info['gpu']['reserved']:.2f} GB, "
                    f"Free={info['gpu']['free']:.2f} GB")
    else:
        logger.info(f"GPU Info Error: {info['gpu']['error']}")

def run_training(args):
    """Run the training script with the specified arguments."""
    # Construct command
    cmd = [
        "python",
        "robust_mqtm_training.py",
        f"--data_dir={args.data_dir}",
        f"--models_dir={args.models_dir}",
        f"--batch_size={args.batch_size}",
        f"--epochs={args.epochs}",
        f"--learning_rate={args.learning_rate}",
    ]
    
    if args.max_files is not None:
        cmd.append(f"--max_files={args.max_files}")
    
    # Log command
    logger.info(f"Running command: {' '.join(cmd)}")
    
    # Run command
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
    )
    
    # Monitor process
    try:
        # Set up progress monitoring
        start_time = time.time()
        last_info_time = start_time
        
        # Process output
        for line in iter(process.stdout.readline, ""):
            print(line, end="")
            
            # Log system info periodically
            current_time = time.time()
            if current_time - last_info_time > 300:  # Every 5 minutes
                log_system_info()
                last_info_time = current_time
        
        # Wait for process to complete
        process.wait()
        
        # Log final status
        if process.returncode == 0:
            logger.info("Training completed successfully!")
        else:
            logger.error(f"Training failed with return code {process.returncode}")
        
        # Log total runtime
        total_runtime = time.time() - start_time
        hours, remainder = divmod(total_runtime, 3600)
        minutes, seconds = divmod(remainder, 60)
        logger.info(f"Total runtime: {int(hours)}h {int(minutes)}m {int(seconds)}s")
        
        return process.returncode
    
    except KeyboardInterrupt:
        logger.warning("Keyboard interrupt detected. Terminating training...")
        process.terminate()
        process.wait()
        return 1
    
    except Exception as e:
        logger.error(f"Error monitoring training: {e}")
        process.terminate()
        process.wait()
        return 1

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run Robust MQTM Training")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models/robust_training",
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load (None for all)")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Log system info before starting
    logger.info("System information before training:")
    log_system_info()
    
    # Run training
    return_code = run_training(args)
    
    # Log system info after completion
    logger.info("System information after training:")
    log_system_info()
    
    return return_code

if __name__ == "__main__":
    main()
