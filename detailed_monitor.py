"""
Detailed monitoring script for MQTM training.
This script provides comprehensive real-time monitoring of the training process.
"""

import os
import sys
import time
import json
import psutil
import argparse
import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import re
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("detailed_monitor.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class DetailedMonitor:
    """Detailed monitoring of training process."""
    
    def __init__(self, args):
        """Initialize monitor."""
        self.models_dir = Path(args.models_dir)
        self.output_dir = Path(args.output_dir)
        self.interval = args.interval
        self.log_file = Path(args.log_file)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize metrics
        self.metrics = {
            "timestamp": [],
            "epoch": [],
            "batch": [],
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "learning_rate": [],
            "cpu_percent": [],
            "memory_percent": [],
            "memory_available": [],
            "gpu_memory_allocated": [],
            "gpu_memory_reserved": [],
            "gpu_utilization": [],
        }
        
        # Load existing metrics if available
        self.metrics_file = self.models_dir / "detailed_metrics.json"
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, "r") as f:
                    self.metrics = json.load(f)
            except Exception as e:
                logger.error(f"Error loading metrics: {e}")
    
    def monitor(self):
        """Monitor training process."""
        logger.info(f"Starting detailed monitoring with interval {self.interval}s")
        
        # Initialize last log position
        last_position = 0
        current_epoch = 0
        current_batch = 0
        
        while True:
            try:
                # Get system metrics
                cpu_percent = psutil.cpu_percent()
                memory = psutil.virtual_memory()
                memory_percent = memory.percent
                memory_available = memory.available / (1024 ** 3)  # GB
                
                # Get GPU metrics
                gpu_memory_allocated = 0
                gpu_memory_reserved = 0
                gpu_utilization = 0
                
                if torch.cuda.is_available():
                    gpu_memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                    gpu_memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)  # GB
                    
                    # Try to get GPU utilization using nvidia-smi
                    try:
                        import subprocess
                        result = subprocess.run(
                            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                            stdout=subprocess.PIPE,
                            text=True
                        )
                        gpu_utilization = float(result.stdout.strip())
                    except Exception as e:
                        logger.warning(f"Could not get GPU utilization: {e}")
                
                # Log system metrics
                logger.info(f"CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}% (Available: {memory_available:.2f} GB)")
                logger.info(f"GPU Memory: Allocated={gpu_memory_allocated:.2f} GB, Reserved={gpu_memory_reserved:.2f} GB, Utilization={gpu_utilization:.1f}%")
                
                # Parse training log for new entries
                if self.log_file.exists():
                    with open(self.log_file, "r") as f:
                        f.seek(last_position)
                        new_lines = f.read()
                        last_position = f.tell()
                    
                    # Extract metrics from log
                    self._parse_log(new_lines, current_epoch, current_batch)
                
                # Save metrics
                self._save_metrics()
                
                # Create visualizations
                self._create_visualizations()
                
                # Sleep for interval
                time.sleep(self.interval)
            
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                break
            
            except Exception as e:
                logger.error(f"Error in monitoring: {e}")
                time.sleep(self.interval)
    
    def _parse_log(self, log_text, current_epoch, current_batch):
        """Parse log text for metrics."""
        # Extract epoch information
        epoch_pattern = r"Epoch (\d+)/(\d+)"
        epoch_matches = re.findall(epoch_pattern, log_text)
        if epoch_matches:
            current_epoch = int(epoch_matches[-1][0])
        
        # Extract batch information
        batch_pattern = r"Batch (\d+)/(\d+)"
        batch_matches = re.findall(batch_pattern, log_text)
        if batch_matches:
            current_batch = int(batch_matches[-1][0])
        
        # Extract loss and accuracy
        loss_pattern = r"loss: ([\d\.]+)"
        loss_matches = re.findall(loss_pattern, log_text)
        
        acc_pattern = r"acc: ([\d\.]+)"
        acc_matches = re.findall(acc_pattern, log_text)
        
        # Extract validation metrics
        val_pattern = r"Validation - Loss: ([\d\.]+), Accuracy: ([\d\.]+)"
        val_matches = re.findall(val_pattern, log_text)
        
        # Extract learning rate
        lr_pattern = r"Learning rate: ([\d\.e\-]+)"
        lr_matches = re.findall(lr_pattern, log_text)
        
        # Update metrics if we have new information
        if loss_matches and acc_matches:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            self.metrics["timestamp"].append(timestamp)
            self.metrics["epoch"].append(current_epoch)
            self.metrics["batch"].append(current_batch)
            self.metrics["train_loss"].append(float(loss_matches[-1]))
            self.metrics["train_acc"].append(float(acc_matches[-1]))
            
            # CPU and GPU metrics
            self.metrics["cpu_percent"].append(psutil.cpu_percent())
            memory = psutil.virtual_memory()
            self.metrics["memory_percent"].append(memory.percent)
            self.metrics["memory_available"].append(memory.available / (1024 ** 3))
            
            if torch.cuda.is_available():
                self.metrics["gpu_memory_allocated"].append(torch.cuda.memory_allocated() / (1024 ** 3))
                self.metrics["gpu_memory_reserved"].append(torch.cuda.memory_reserved() / (1024 ** 3))
                
                # Try to get GPU utilization
                try:
                    import subprocess
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                        stdout=subprocess.PIPE,
                        text=True
                    )
                    self.metrics["gpu_utilization"].append(float(result.stdout.strip()))
                except Exception:
                    self.metrics["gpu_utilization"].append(0)
            else:
                self.metrics["gpu_memory_allocated"].append(0)
                self.metrics["gpu_memory_reserved"].append(0)
                self.metrics["gpu_utilization"].append(0)
            
            # Validation metrics (if available)
            if val_matches:
                self.metrics["val_loss"].append(float(val_matches[-1][0]))
                self.metrics["val_acc"].append(float(val_matches[-1][1]))
            else:
                # Use previous values or None
                if self.metrics["val_loss"]:
                    self.metrics["val_loss"].append(self.metrics["val_loss"][-1])
                    self.metrics["val_acc"].append(self.metrics["val_acc"][-1])
                else:
                    self.metrics["val_loss"].append(None)
                    self.metrics["val_acc"].append(None)
            
            # Learning rate (if available)
            if lr_matches:
                self.metrics["learning_rate"].append(float(lr_matches[-1]))
            else:
                # Use previous value or None
                if self.metrics["learning_rate"]:
                    self.metrics["learning_rate"].append(self.metrics["learning_rate"][-1])
                else:
                    self.metrics["learning_rate"].append(None)
            
            logger.info(f"Epoch {current_epoch}, Batch {current_batch} - "
                       f"Loss: {float(loss_matches[-1]):.4f}, "
                       f"Acc: {float(acc_matches[-1]):.4f}")
    
    def _save_metrics(self):
        """Save metrics to file."""
        with open(self.metrics_file, "w") as f:
            json.dump(self.metrics, f, indent=2)
    
    def _create_visualizations(self):
        """Create visualizations of metrics."""
        if not self.metrics["timestamp"]:
            return
        
        # Create DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Create training progress plot
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
        if df["val_loss"].any():
            plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 2)
        plt.plot(df["epoch"], df["train_acc"], label="Train Acc")
        if df["val_acc"].any():
            plt.plot(df["epoch"], df["val_acc"], label="Val Acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.subplot(2, 2, 3)
        plt.plot(df["timestamp"], df["gpu_memory_allocated"], label="Allocated")
        plt.plot(df["timestamp"], df["gpu_memory_reserved"], label="Reserved")
        plt.xlabel("Time")
        plt.ylabel("GPU Memory (GB)")
        plt.title("GPU Memory Usage")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.subplot(2, 2, 4)
        plt.plot(df["timestamp"], df["gpu_utilization"], label="GPU")
        plt.plot(df["timestamp"], df["cpu_percent"], label="CPU")
        plt.xlabel("Time")
        plt.ylabel("Utilization (%)")
        plt.title("GPU and CPU Utilization")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_progress.png")
        plt.close()
        
        # Create learning rate plot
        if df["learning_rate"].any():
            plt.figure(figsize=(10, 6))
            plt.plot(df["epoch"], df["learning_rate"])
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.title("Learning Rate Schedule")
            plt.grid(True)
            plt.savefig(self.output_dir / "learning_rate.png")
            plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Detailed monitoring of MQTM training")
    parser.add_argument("--models_dir", type=str, default="models/robust_training",
                        help="Directory containing model checkpoints and metrics")
    parser.add_argument("--output_dir", type=str, default="monitor_output",
                        help="Directory to save monitoring output")
    parser.add_argument("--interval", type=int, default=10,
                        help="Monitoring interval in seconds")
    parser.add_argument("--log_file", type=str, default="robust_training.log",
                        help="Training log file to monitor")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    monitor = DetailedMonitor(args)
    monitor.monitor()

if __name__ == "__main__":
    main()
