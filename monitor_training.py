"""
Script to monitor the training process and visualize results.
"""

import os
import json
import time
import argparse
import logging
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import psutil
import glob

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("monitor_training.log"),
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
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }

def monitor_system(interval=60, duration=None):
    """
    Monitor system resources.
    
    Args:
        interval: Monitoring interval in seconds
        duration: Monitoring duration in seconds (None for indefinite)
    
    Returns:
        List of system info dictionaries
    """
    system_info = []
    start_time = time.time()
    
    try:
        while True:
            # Get system info
            info = get_system_info()
            system_info.append(info)
            
            # Log info
            logger.info(f"CPU: {info['cpu_percent']}%, "
                        f"Memory: {info['memory_percent']}% "
                        f"(Available: {info['memory_available_gb']:.2f} GB)")
            
            if "error" not in info["gpu"]:
                logger.info(f"GPU Memory: Total={info['gpu']['total']:.2f} GB, "
                            f"Allocated={info['gpu']['allocated']:.2f} GB, "
                            f"Reserved={info['gpu']['reserved']:.2f} GB, "
                            f"Free={info['gpu']['free']:.2f} GB")
            else:
                logger.info(f"GPU Info Error: {info['gpu']['error']}")
            
            # Check if duration has elapsed
            if duration is not None and time.time() - start_time >= duration:
                break
            
            # Wait for next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    return system_info

def plot_system_info(system_info, output_dir):
    """
    Plot system information.
    
    Args:
        system_info: List of system info dictionaries
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data
    timestamps = [info["timestamp"] for info in system_info]
    cpu_percent = [info["cpu_percent"] for info in system_info]
    memory_percent = [info["memory_percent"] for info in system_info]
    memory_available = [info["memory_available_gb"] for info in system_info]
    
    # GPU data
    gpu_allocated = []
    gpu_reserved = []
    gpu_free = []
    
    for info in system_info:
        if "error" not in info["gpu"]:
            gpu_allocated.append(info["gpu"]["allocated"])
            gpu_reserved.append(info["gpu"]["reserved"])
            gpu_free.append(info["gpu"]["free"])
        else:
            gpu_allocated.append(np.nan)
            gpu_reserved.append(np.nan)
            gpu_free.append(np.nan)
    
    # Plot CPU usage
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, cpu_percent)
    plt.title("CPU Usage")
    plt.xlabel("Time")
    plt.ylabel("CPU Usage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "cpu_usage.png"))
    plt.close()
    
    # Plot memory usage
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_percent)
    plt.title("Memory Usage")
    plt.xlabel("Time")
    plt.ylabel("Memory Usage (%)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_usage.png"))
    plt.close()
    
    # Plot available memory
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, memory_available)
    plt.title("Available Memory")
    plt.xlabel("Time")
    plt.ylabel("Available Memory (GB)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "memory_available.png"))
    plt.close()
    
    # Plot GPU memory
    plt.figure(figsize=(12, 6))
    plt.plot(timestamps, gpu_allocated, label="Allocated")
    plt.plot(timestamps, gpu_reserved, label="Reserved")
    plt.plot(timestamps, gpu_free, label="Free")
    plt.title("GPU Memory")
    plt.xlabel("Time")
    plt.ylabel("Memory (GB)")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gpu_memory.png"))
    plt.close()

def monitor_training_metrics(models_dir, interval=60, duration=None):
    """
    Monitor training metrics.
    
    Args:
        models_dir: Directory containing training metrics
        interval: Monitoring interval in seconds
        duration: Monitoring duration in seconds (None for indefinite)
    
    Returns:
        Dictionary of metrics
    """
    metrics_file = os.path.join(models_dir, "metrics.json")
    start_time = time.time()
    
    try:
        while True:
            # Check if metrics file exists
            if os.path.exists(metrics_file):
                # Load metrics
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                
                # Log metrics
                if metrics["train_loss"] and metrics["val_loss"]:
                    logger.info(f"Latest metrics - "
                                f"Train Loss: {metrics['train_loss'][-1]:.4f}, "
                                f"Train Acc: {metrics['train_acc'][-1]:.4f}, "
                                f"Val Loss: {metrics['val_loss'][-1]:.4f}, "
                                f"Val Acc: {metrics['val_acc'][-1]:.4f}")
                
                # Plot metrics
                plot_metrics(metrics, models_dir)
            else:
                logger.info(f"Metrics file not found: {metrics_file}")
            
            # Check if duration has elapsed
            if duration is not None and time.time() - start_time >= duration:
                break
            
            # Wait for next interval
            time.sleep(interval)
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    # Final metrics
    if os.path.exists(metrics_file):
        with open(metrics_file, "r") as f:
            metrics = json.load(f)
    else:
        metrics = None
    
    return metrics

def plot_metrics(metrics, output_dir):
    """
    Plot training metrics.
    
    Args:
        metrics: Dictionary of metrics
        output_dir: Output directory for plots
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot loss
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["train_loss"], label="Train")
    plt.plot(metrics["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_monitor.png"))
    plt.close()
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["train_acc"], label="Train")
    plt.plot(metrics["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy_monitor.png"))
    plt.close()
    
    # Plot learning rate
    plt.figure(figsize=(12, 6))
    plt.plot(metrics["learning_rates"])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_rate_monitor.png"))
    plt.close()

def analyze_checkpoints(models_dir):
    """
    Analyze training checkpoints.
    
    Args:
        models_dir: Directory containing checkpoints
    """
    # Find checkpoint files
    checkpoint_files = glob.glob(os.path.join(models_dir, "checkpoint_epoch_*.pt"))
    
    if not checkpoint_files:
        logger.info(f"No checkpoint files found in {models_dir}")
        return
    
    # Sort checkpoint files by epoch
    checkpoint_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    # Log checkpoint information
    logger.info(f"Found {len(checkpoint_files)} checkpoint files:")
    for checkpoint_file in checkpoint_files:
        logger.info(f"  {os.path.basename(checkpoint_file)}")
    
    # Check if best model exists
    best_model_file = os.path.join(models_dir, "best_model.pt")
    if os.path.exists(best_model_file):
        logger.info(f"Best model found: {best_model_file}")
    else:
        logger.info(f"Best model not found: {best_model_file}")
    
    # Check if final model exists
    final_model_file = os.path.join(models_dir, "final_model.pt")
    if os.path.exists(final_model_file):
        logger.info(f"Final model found: {final_model_file}")
    else:
        logger.info(f"Final model not found: {final_model_file}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Monitor Training")
    
    parser.add_argument("--models_dir", type=str, default="models/robust_training",
                        help="Directory containing training metrics")
    parser.add_argument("--interval", type=int, default=60,
                        help="Monitoring interval in seconds")
    parser.add_argument("--duration", type=int, default=None,
                        help="Monitoring duration in seconds (None for indefinite)")
    parser.add_argument("--output_dir", type=str, default="monitor_output",
                        help="Output directory for plots")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Start monitoring
    logger.info(f"Starting monitoring with interval {args.interval}s")
    
    # Monitor system resources
    system_info = []
    
    # Monitor training metrics
    metrics = None
    
    try:
        while True:
            # Get system info
            info = get_system_info()
            system_info.append(info)
            
            # Log system info
            logger.info(f"CPU: {info['cpu_percent']}%, "
                        f"Memory: {info['memory_percent']}% "
                        f"(Available: {info['memory_available_gb']:.2f} GB)")
            
            if "error" not in info["gpu"]:
                logger.info(f"GPU Memory: Total={info['gpu']['total']:.2f} GB, "
                            f"Allocated={info['gpu']['allocated']:.2f} GB, "
                            f"Reserved={info['gpu']['reserved']:.2f} GB, "
                            f"Free={info['gpu']['free']:.2f} GB")
            else:
                logger.info(f"GPU Info Error: {info['gpu']['error']}")
            
            # Check if metrics file exists
            metrics_file = os.path.join(args.models_dir, "metrics.json")
            if os.path.exists(metrics_file):
                # Load metrics
                with open(metrics_file, "r") as f:
                    metrics = json.load(f)
                
                # Log metrics
                if metrics["train_loss"] and metrics["val_loss"]:
                    logger.info(f"Latest metrics - "
                                f"Train Loss: {metrics['train_loss'][-1]:.4f}, "
                                f"Train Acc: {metrics['train_acc'][-1]:.4f}, "
                                f"Val Loss: {metrics['val_loss'][-1]:.4f}, "
                                f"Val Acc: {metrics['val_acc'][-1]:.4f}")
                
                # Plot metrics
                plot_metrics(metrics, args.output_dir)
            
            # Plot system info
            plot_system_info(system_info, args.output_dir)
            
            # Wait for next interval
            time.sleep(args.interval)
            
            # Check if duration has elapsed
            if args.duration is not None and len(system_info) * args.interval >= args.duration:
                break
    
    except KeyboardInterrupt:
        logger.info("Monitoring stopped by user")
    
    # Final analysis
    logger.info("Performing final analysis")
    
    # Analyze checkpoints
    analyze_checkpoints(args.models_dir)
    
    # Final plots
    if system_info:
        plot_system_info(system_info, args.output_dir)
    
    if metrics:
        plot_metrics(metrics, args.output_dir)
    
    logger.info("Monitoring completed")

if __name__ == "__main__":
    main()
