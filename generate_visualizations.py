"""
Generate detailed visualizations of MQTM training progress.
"""

import os
import sys
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import seaborn as sns

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("visualizations.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class TrainingVisualizer:
    """Generate visualizations of training progress."""
    
    def __init__(self, args):
        """Initialize visualizer."""
        self.models_dir = Path(args.models_dir)
        self.output_dir = Path(args.output_dir)
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load metrics
        self.metrics_file = self.models_dir / "detailed_metrics.json"
        if not self.metrics_file.exists():
            self.metrics_file = self.models_dir / "metrics.json"
        
        if not self.metrics_file.exists():
            logger.error(f"Metrics file not found: {self.metrics_file}")
            sys.exit(1)
        
        try:
            with open(self.metrics_file, "r") as f:
                self.metrics = json.load(f)
        except Exception as e:
            logger.error(f"Error loading metrics: {e}")
            sys.exit(1)
    
    def generate_visualizations(self):
        """Generate visualizations."""
        logger.info("Generating visualizations...")
        
        # Convert metrics to DataFrame
        df = pd.DataFrame(self.metrics)
        
        # Set style
        sns.set(style="darkgrid")
        
        # Generate visualizations
        self._plot_training_curves(df)
        self._plot_learning_rate(df)
        self._plot_resource_usage(df)
        self._plot_batch_metrics(df)
        self._plot_correlation_matrix(df)
        
        logger.info(f"Visualizations saved to {self.output_dir}")
    
    def _plot_training_curves(self, df):
        """Plot training and validation curves."""
        plt.figure(figsize=(12, 10))
        
        # Loss plot
        plt.subplot(2, 1, 1)
        plt.plot(df["epoch"], df["train_loss"], label="Training Loss", color="blue", linewidth=2)
        if "val_loss" in df.columns and not df["val_loss"].isna().all():
            plt.plot(df["epoch"], df["val_loss"], label="Validation Loss", color="red", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.grid(True)
        
        # Accuracy plot
        plt.subplot(2, 1, 2)
        plt.plot(df["epoch"], df["train_acc"], label="Training Accuracy", color="green", linewidth=2)
        if "val_acc" in df.columns and not df["val_acc"].isna().all():
            plt.plot(df["epoch"], df["val_acc"], label="Validation Accuracy", color="purple", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "training_curves.png", dpi=300)
        plt.close()
    
    def _plot_learning_rate(self, df):
        """Plot learning rate schedule."""
        if "learning_rate" not in df.columns or df["learning_rate"].isna().all():
            return
        
        plt.figure(figsize=(10, 6))
        plt.plot(df["epoch"], df["learning_rate"], color="orange", linewidth=2)
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.title("Learning Rate Schedule")
        plt.grid(True)
        plt.yscale("log")
        plt.tight_layout()
        plt.savefig(self.output_dir / "learning_rate.png", dpi=300)
        plt.close()
    
    def _plot_resource_usage(self, df):
        """Plot resource usage."""
        plt.figure(figsize=(12, 10))
        
        # GPU memory usage
        plt.subplot(2, 1, 1)
        if "gpu_memory_allocated" in df.columns and not df["gpu_memory_allocated"].isna().all():
            plt.plot(df["timestamp"], df["gpu_memory_allocated"], label="Allocated", color="blue", linewidth=2)
        if "gpu_memory_reserved" in df.columns and not df["gpu_memory_reserved"].isna().all():
            plt.plot(df["timestamp"], df["gpu_memory_reserved"], label="Reserved", color="green", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("GPU Memory (GB)")
        plt.title("GPU Memory Usage")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        # CPU and GPU utilization
        plt.subplot(2, 1, 2)
        if "gpu_utilization" in df.columns and not df["gpu_utilization"].isna().all():
            plt.plot(df["timestamp"], df["gpu_utilization"], label="GPU", color="red", linewidth=2)
        if "cpu_percent" in df.columns and not df["cpu_percent"].isna().all():
            plt.plot(df["timestamp"], df["cpu_percent"], label="CPU", color="purple", linewidth=2)
        plt.xlabel("Time")
        plt.ylabel("Utilization (%)")
        plt.title("GPU and CPU Utilization")
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "resource_usage.png", dpi=300)
        plt.close()
    
    def _plot_batch_metrics(self, df):
        """Plot metrics by batch."""
        if "batch" not in df.columns or df["batch"].isna().all():
            return
        
        # Group by epoch and batch
        df_grouped = df.groupby(["epoch", "batch"]).mean().reset_index()
        
        plt.figure(figsize=(12, 10))
        
        # Loss by batch
        plt.subplot(2, 1, 1)
        for epoch in df_grouped["epoch"].unique():
            epoch_data = df_grouped[df_grouped["epoch"] == epoch]
            plt.plot(epoch_data["batch"], epoch_data["train_loss"], label=f"Epoch {epoch}")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss by Batch")
        plt.legend()
        plt.grid(True)
        
        # Accuracy by batch
        plt.subplot(2, 1, 2)
        for epoch in df_grouped["epoch"].unique():
            epoch_data = df_grouped[df_grouped["epoch"] == epoch]
            plt.plot(epoch_data["batch"], epoch_data["train_acc"], label=f"Epoch {epoch}")
        plt.xlabel("Batch")
        plt.ylabel("Accuracy")
        plt.title("Training Accuracy by Batch")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "batch_metrics.png", dpi=300)
        plt.close()
    
    def _plot_correlation_matrix(self, df):
        """Plot correlation matrix of metrics."""
        # Select numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove columns with all NaN values
        numeric_cols = [col for col in numeric_cols if not df[col].isna().all()]
        
        if len(numeric_cols) < 2:
            return
        
        # Calculate correlation matrix
        corr = df[numeric_cols].corr()
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
        plt.title("Correlation Matrix of Training Metrics")
        plt.tight_layout()
        plt.savefig(self.output_dir / "correlation_matrix.png", dpi=300)
        plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Generate visualizations of MQTM training progress")
    parser.add_argument("--models_dir", type=str, default="models/robust_training",
                        help="Directory containing model checkpoints and metrics")
    parser.add_argument("--output_dir", type=str, default="monitor_output",
                        help="Directory to save visualizations")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    visualizer = TrainingVisualizer(args)
    visualizer.generate_visualizations()

if __name__ == "__main__":
    main()
