"""
Live training display for MQTM training.
This script provides a real-time display of training progress in the terminal.
"""

import os
import sys
import time
import json
import argparse
import datetime
import re
from pathlib import Path
import logging
import subprocess
import curses

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("live_display.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class LiveTrainingDisplay:
    """Live display of training progress in terminal."""
    
    def __init__(self, args):
        """Initialize display."""
        self.log_file = Path(args.log_file)
        self.refresh_rate = args.refresh_rate
        self.last_position = 0
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = 0
        self.current_batch = 0
        self.total_batches = 0
        self.train_loss = 0.0
        self.train_acc = 0.0
        self.val_loss = 0.0
        self.val_acc = 0.0
        self.learning_rate = 0.0
        self.best_val_acc = 0.0
        
        # GPU state
        self.gpu_memory_allocated = 0.0
        self.gpu_memory_reserved = 0.0
        self.gpu_utilization = 0.0
        
        # System state
        self.cpu_percent = 0.0
        self.memory_percent = 0.0
        self.memory_available = 0.0
        
        # Log buffer
        self.log_buffer = []
        self.max_log_lines = 10
        
        # Extract total epochs from log file if available
        if self.log_file.exists():
            with open(self.log_file, "r") as f:
                content = f.read()
                epoch_pattern = r"Epoch (\d+)/(\d+)"
                epoch_matches = re.findall(epoch_pattern, content)
                if epoch_matches:
                    self.total_epochs = int(epoch_matches[0][1])
    
    def update_gpu_metrics(self):
        """Update GPU metrics."""
        try:
            # Get GPU memory usage
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                stdout=subprocess.PIPE,
                text=True
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(',')
                if len(parts) >= 3:
                    self.gpu_memory_allocated = float(parts[0].strip()) / 1024  # Convert MB to GB
                    self.gpu_memory_total = float(parts[1].strip()) / 1024  # Convert MB to GB
                    self.gpu_utilization = float(parts[2].strip())
        except Exception as e:
            logger.warning(f"Could not get GPU metrics: {e}")
    
    def update_system_metrics(self):
        """Update system metrics."""
        try:
            import psutil
            self.cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            self.memory_percent = memory.percent
            self.memory_available = memory.available / (1024 ** 3)  # GB
        except Exception as e:
            logger.warning(f"Could not get system metrics: {e}")
    
    def update_training_metrics(self):
        """Update training metrics from log file."""
        if not self.log_file.exists():
            return
        
        try:
            with open(self.log_file, "r") as f:
                f.seek(self.last_position)
                new_lines = f.read()
                self.last_position = f.tell()
            
            if not new_lines:
                return
            
            # Add new lines to log buffer
            for line in new_lines.splitlines():
                if line.strip():
                    self.log_buffer.append(line)
            
            # Keep only the last N lines
            self.log_buffer = self.log_buffer[-self.max_log_lines:]
            
            # Extract epoch information
            epoch_pattern = r"Epoch (\d+)/(\d+)"
            epoch_matches = re.findall(epoch_pattern, new_lines)
            if epoch_matches:
                self.current_epoch = int(epoch_matches[-1][0])
                self.total_epochs = int(epoch_matches[-1][1])
            
            # Extract batch information
            batch_pattern = r"Batch (\d+)/(\d+)"
            batch_matches = re.findall(batch_pattern, new_lines)
            if batch_matches:
                self.current_batch = int(batch_matches[-1][0])
                self.total_batches = int(batch_matches[-1][1])
            
            # Extract loss and accuracy
            loss_pattern = r"loss: ([\d\.]+)"
            loss_matches = re.findall(loss_pattern, new_lines)
            if loss_matches:
                self.train_loss = float(loss_matches[-1])
            
            acc_pattern = r"acc: ([\d\.]+)"
            acc_matches = re.findall(acc_pattern, new_lines)
            if acc_matches:
                self.train_acc = float(acc_matches[-1])
            
            # Extract validation metrics
            val_pattern = r"Validation - Loss: ([\d\.]+), Accuracy: ([\d\.]+)"
            val_matches = re.findall(val_pattern, new_lines)
            if val_matches:
                self.val_loss = float(val_matches[-1][0])
                self.val_acc = float(val_matches[-1][1])
                
                # Update best validation accuracy
                if self.val_acc > self.best_val_acc:
                    self.best_val_acc = self.val_acc
            
            # Extract learning rate
            lr_pattern = r"Learning rate: ([\d\.e\-]+)"
            lr_matches = re.findall(lr_pattern, new_lines)
            if lr_matches:
                self.learning_rate = float(lr_matches[-1])
            
            # Extract GPU memory information
            gpu_pattern = r"GPU Memory at epoch start: Allocated: ([\d\.]+) GB, Cached: ([\d\.]+) GB"
            gpu_matches = re.findall(gpu_pattern, new_lines)
            if gpu_matches:
                self.gpu_memory_allocated = float(gpu_matches[-1][0])
                self.gpu_memory_reserved = float(gpu_matches[-1][1])
            
        except Exception as e:
            logger.error(f"Error updating training metrics: {e}")
    
    def display(self, stdscr):
        """Display training progress in terminal."""
        # Clear screen
        stdscr.clear()
        
        # Set up colors
        curses.start_color()
        curses.init_pair(1, curses.COLOR_GREEN, curses.COLOR_BLACK)  # Good
        curses.init_pair(2, curses.COLOR_YELLOW, curses.COLOR_BLACK)  # Warning
        curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)  # Error
        curses.init_pair(4, curses.COLOR_CYAN, curses.COLOR_BLACK)  # Info
        curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)  # Special
        
        # Hide cursor
        curses.curs_set(0)
        
        # Main loop
        while True:
            try:
                # Update metrics
                self.update_training_metrics()
                self.update_gpu_metrics()
                self.update_system_metrics()
                
                # Clear screen
                stdscr.clear()
                
                # Get terminal size
                max_y, max_x = stdscr.getmaxyx()
                
                # Display title
                title = "MQTM TRAINING MONITOR"
                stdscr.addstr(0, (max_x - len(title)) // 2, title, curses.A_BOLD | curses.color_pair(5))
                
                # Display training progress
                stdscr.addstr(2, 2, f"Epoch: {self.current_epoch}/{self.total_epochs} ", curses.color_pair(4))
                
                # Progress bar for epochs
                if self.total_epochs > 0:
                    progress_width = max_x - 30
                    progress = int(progress_width * self.current_epoch / self.total_epochs)
                    stdscr.addstr(2, 25, "[" + "#" * progress + " " * (progress_width - progress) + "]", curses.color_pair(1))
                    stdscr.addstr(2, 25 + progress_width + 2, f"{self.current_epoch / self.total_epochs * 100:.1f}%", curses.color_pair(1))
                
                # Display batch progress
                stdscr.addstr(3, 2, f"Batch: {self.current_batch}/{self.total_batches} ", curses.color_pair(4))
                
                # Progress bar for batches
                if self.total_batches > 0:
                    progress_width = max_x - 30
                    progress = int(progress_width * self.current_batch / self.total_batches)
                    stdscr.addstr(3, 25, "[" + "#" * progress + " " * (progress_width - progress) + "]", curses.color_pair(1))
                    stdscr.addstr(3, 25 + progress_width + 2, f"{self.current_batch / self.total_batches * 100:.1f}%", curses.color_pair(1))
                
                # Display metrics
                stdscr.addstr(5, 2, "Training Metrics:", curses.A_BOLD)
                stdscr.addstr(6, 4, f"Loss: {self.train_loss:.6f}", curses.color_pair(4))
                stdscr.addstr(6, 25, f"Accuracy: {self.train_acc:.6f}", curses.color_pair(4))
                stdscr.addstr(7, 4, f"Val Loss: {self.val_loss:.6f}", curses.color_pair(4))
                stdscr.addstr(7, 25, f"Val Accuracy: {self.val_acc:.6f}", curses.color_pair(4))
                stdscr.addstr(8, 4, f"Best Val Acc: {self.best_val_acc:.6f}", curses.color_pair(1))
                stdscr.addstr(8, 25, f"Learning Rate: {self.learning_rate:.8f}", curses.color_pair(4))
                
                # Display GPU metrics
                stdscr.addstr(10, 2, "GPU Metrics:", curses.A_BOLD)
                stdscr.addstr(11, 4, f"Memory: {self.gpu_memory_allocated:.2f} GB / {getattr(self, 'gpu_memory_total', 6.0):.2f} GB", curses.color_pair(4))
                stdscr.addstr(11, 40, f"Utilization: {self.gpu_utilization:.1f}%", curses.color_pair(4))
                
                # Display system metrics
                stdscr.addstr(13, 2, "System Metrics:", curses.A_BOLD)
                stdscr.addstr(14, 4, f"CPU: {self.cpu_percent:.1f}%", curses.color_pair(4))
                stdscr.addstr(14, 25, f"Memory: {self.memory_percent:.1f}%", curses.color_pair(4))
                stdscr.addstr(14, 50, f"Available: {self.memory_available:.2f} GB", curses.color_pair(4))
                
                # Display log buffer
                stdscr.addstr(16, 2, "Recent Log Messages:", curses.A_BOLD)
                for i, line in enumerate(self.log_buffer[-8:]):
                    if i + 17 < max_y:
                        # Truncate line if it's too long
                        if len(line) > max_x - 4:
                            line = line[:max_x - 7] + "..."
                        stdscr.addstr(17 + i, 4, line, curses.color_pair(4))
                
                # Display help
                stdscr.addstr(max_y - 2, 2, "Press 'q' to quit", curses.color_pair(2))
                
                # Refresh screen
                stdscr.refresh()
                
                # Check for key press
                stdscr.timeout(self.refresh_rate * 1000)
                key = stdscr.getch()
                if key == ord('q'):
                    break
                
            except KeyboardInterrupt:
                break
            
            except Exception as e:
                # Log error and continue
                logger.error(f"Error in display: {e}")
                time.sleep(self.refresh_rate)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Live display of MQTM training progress")
    parser.add_argument("--log_file", type=str, default="robust_training.log",
                        help="Training log file to monitor")
    parser.add_argument("--refresh_rate", type=float, default=1.0,
                        help="Refresh rate in seconds")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    display = LiveTrainingDisplay(args)
    
    # Run the curses application
    curses.wrapper(display.display)

if __name__ == "__main__":
    main()
