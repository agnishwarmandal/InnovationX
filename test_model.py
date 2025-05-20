"""
Script to test the trained model on all datasets.
"""

import os
import argparse
import logging
import json
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc,
)
import seaborn as sns
from typing import Dict, List, Tuple

# Import from training script
from robust_mqtm_training import Config, DataLoader, RobustModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_model.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_model(model_path, config):
    """
    Load trained model.
    
    Args:
        model_path: Path to model file
        config: Model configuration
        
    Returns:
        Loaded model
    """
    # Create model
    model = RobustModel(config)
    
    # Load model weights
    model.load_state_dict(torch.load(model_path, map_location=config.device))
    
    # Move model to device
    model.to(config.device)
    
    # Set model to evaluation mode
    model.eval()
    
    return model

def test_model(model, test_loader, config):
    """
    Test model on test set.
    
    Args:
        model: Trained model
        test_loader: Test data loader
        config: Model configuration
        
    Returns:
        Dictionary of test metrics
    """
    # Initialize metrics
    y_true = []
    y_pred = []
    y_prob = []
    
    # Test loop
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            # Move data to device
            data = data.to(config.device)
            target = target.to(config.device)
            
            # Forward pass
            output = model(data)
            
            # Get predictions
            probabilities = torch.softmax(output, dim=1)
            _, predicted = output.max(1)
            
            # Move to CPU and convert to numpy
            y_true.extend(target.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
            y_prob.extend(probabilities[:, 1].cpu().numpy())
    
    # Convert to numpy arrays
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    y_prob = np.array(y_prob)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    # Log metrics
    logger.info(f"Test Accuracy: {accuracy:.4f}")
    logger.info(f"Test Precision: {precision:.4f}")
    logger.info(f"Test Recall: {recall:.4f}")
    logger.info(f"Test F1 Score: {f1:.4f}")
    logger.info(f"Test AUC: {roc_auc:.4f}")
    
    # Log classification report
    logger.info(f"Classification Report:\n{classification_report(y_true, y_pred)}")
    
    # Return metrics
    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "y_true": y_true.tolist(),
        "y_pred": y_pred.tolist(),
        "y_prob": y_prob.tolist(),
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
    }

def test_model_by_symbol(model, data_loader, config):
    """
    Test model on each symbol separately.
    
    Args:
        model: Trained model
        data_loader: Data loader
        config: Model configuration
        
    Returns:
        Dictionary of test metrics by symbol
    """
    # Initialize metrics
    metrics_by_symbol = {}
    
    # Test on each symbol
    for symbol in tqdm(data_loader.data.keys(), desc="Testing by symbol"):
        # Create sequences
        X_test, y_test = data_loader.create_sequences(data_loader.test_data[symbol])
        
        # Create PyTorch dataset
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        
        # Create DataLoader
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=config.num_workers,
            pin_memory=config.pin_memory,
        )
        
        # Test model
        logger.info(f"Testing on {symbol}...")
        metrics = test_model(model, test_loader, config)
        
        # Store metrics
        metrics_by_symbol[symbol] = metrics
    
    return metrics_by_symbol

def plot_confusion_matrix(cm, output_dir, title="Confusion Matrix"):
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        output_dir: Output directory
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(title)
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

def plot_roc_curve(fpr, tpr, roc_auc, output_dir, title="ROC Curve"):
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rate
        tpr: True positive rate
        roc_auc: Area under ROC curve
        output_dir: Output directory
        title: Plot title
    """
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(output_dir, "roc_curve.png"))
    plt.close()

def plot_metrics_by_symbol(metrics_by_symbol, output_dir):
    """
    Plot metrics by symbol.
    
    Args:
        metrics_by_symbol: Dictionary of metrics by symbol
        output_dir: Output directory
    """
    # Extract metrics
    symbols = list(metrics_by_symbol.keys())
    accuracy = [metrics_by_symbol[symbol]["accuracy"] for symbol in symbols]
    precision = [metrics_by_symbol[symbol]["precision"] for symbol in symbols]
    recall = [metrics_by_symbol[symbol]["recall"] for symbol in symbols]
    f1 = [metrics_by_symbol[symbol]["f1"] for symbol in symbols]
    roc_auc = [metrics_by_symbol[symbol]["roc_auc"] for symbol in symbols]
    
    # Sort by accuracy
    sorted_indices = np.argsort(accuracy)[::-1]
    symbols = [symbols[i] for i in sorted_indices]
    accuracy = [accuracy[i] for i in sorted_indices]
    precision = [precision[i] for i in sorted_indices]
    recall = [recall[i] for i in sorted_indices]
    f1 = [f1[i] for i in sorted_indices]
    roc_auc = [roc_auc[i] for i in sorted_indices]
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    plt.bar(symbols, accuracy)
    plt.title("Accuracy by Symbol")
    plt.xlabel("Symbol")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "accuracy_by_symbol.png"))
    plt.close()
    
    # Plot precision, recall, F1
    plt.figure(figsize=(12, 6))
    plt.bar(symbols, precision, label="Precision")
    plt.bar(symbols, recall, label="Recall", alpha=0.7)
    plt.bar(symbols, f1, label="F1", alpha=0.5)
    plt.title("Precision, Recall, F1 by Symbol")
    plt.xlabel("Symbol")
    plt.ylabel("Score")
    plt.legend()
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "prf_by_symbol.png"))
    plt.close()
    
    # Plot ROC AUC
    plt.figure(figsize=(12, 6))
    plt.bar(symbols, roc_auc)
    plt.title("ROC AUC by Symbol")
    plt.xlabel("Symbol")
    plt.ylabel("ROC AUC")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "roc_auc_by_symbol.png"))
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Trained Model")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--model_path", type=str, default="models/robust_training/best_model.pt",
                        help="Path to trained model")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Output directory for test results")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for testing")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load (None for all)")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Update configuration
    Config.batch_size = args.batch_size
    Config.data_dir = args.data_dir
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Log configuration
    logger.info(f"Using device: {Config.device}")
    logger.info(f"Batch size: {Config.batch_size}")
    logger.info(f"Model path: {args.model_path}")
    
    # Create data loader
    data_loader = DataLoader(Config)
    
    # Load data
    data_loader.load_data(max_files=args.max_files)
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # Load model
    model = load_model(args.model_path, Config)
    
    # Test model
    logger.info("Testing model on all test data...")
    metrics = test_model(model, test_loader, Config)
    
    # Save metrics
    with open(os.path.join(args.output_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot confusion matrix
    plot_confusion_matrix(
        np.array(metrics["confusion_matrix"]),
        args.output_dir,
        title="Confusion Matrix (All Data)",
    )
    
    # Plot ROC curve
    plot_roc_curve(
        metrics["fpr"],
        metrics["tpr"],
        metrics["roc_auc"],
        args.output_dir,
        title="ROC Curve (All Data)",
    )
    
    # Test model by symbol
    logger.info("Testing model on each symbol separately...")
    metrics_by_symbol = test_model_by_symbol(model, data_loader, Config)
    
    # Save metrics by symbol
    with open(os.path.join(args.output_dir, "metrics_by_symbol.json"), "w") as f:
        json.dump(metrics_by_symbol, f, indent=2)
    
    # Plot metrics by symbol
    plot_metrics_by_symbol(metrics_by_symbol, args.output_dir)
    
    logger.info("Testing completed successfully!")

if __name__ == "__main__":
    main()
