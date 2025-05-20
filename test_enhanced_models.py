"""
Script to test the enhanced MQTM models.
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from mqtm.config import config
from mqtm.data_engineering.efficient_dataloader import create_efficient_dataloaders
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.superposition_pool.superposition_model import SuperpositionPool
from mqtm.utils.memory_optimization import MemoryOptimizer
from mqtm.utils.performance_profiling import Timer, TorchProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("test_enhanced.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Enhanced MQTM Models")
    
    parser.add_argument("--original_dir", type=str, default="models",
                        help="Directory with original models")
    parser.add_argument("--enhanced_dir", type=str, default="enhanced_models",
                        help="Directory with enhanced models")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate")
    parser.add_argument("--output_dir", type=str, default="test_results",
                        help="Directory to save test results")
    
    return parser.parse_args()

def load_models(model_dir):
    """Load MQTM models."""
    logger.info(f"Loading models from {model_dir}...")
    
    # Create models
    mg = MultiverseGenerator(
        model_dir=os.path.join(model_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load models if available
    mg.load()
    
    tqe_path = os.path.join(model_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(model_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Move to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    return mg, tqe, sp3

def test_multiverse_generator(original_mg, enhanced_mg, num_samples, output_dir):
    """Test Multiverse Generator."""
    logger.info("Testing Multiverse Generator...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "multiverse_generator"), exist_ok=True)
    
    # Generate samples from original model
    with Timer("Original MG sampling"):
        original_samples, original_graph = original_mg.generate_samples(
            num_samples=num_samples,
        )
    
    # Generate samples from enhanced model
    with Timer("Enhanced MG sampling"):
        enhanced_samples, enhanced_graph = enhanced_mg.generate_samples(
            num_samples=num_samples,
        )
    
    # Compare samples
    logger.info(f"Original samples shape: {original_samples.shape}")
    logger.info(f"Enhanced samples shape: {enhanced_samples.shape}")
    
    # Plot samples
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot original OHLC
    ax = axes[0, 0]
    sample_idx = np.random.randint(0, len(original_samples))
    sample = original_samples[sample_idx]
    
    # Extract OHLC
    open_price = sample[0]
    high = sample[1]
    low = sample[2]
    close = sample[3]
    
    # Plot
    ax.plot(open_price, label="Open")
    ax.plot(high, label="High")
    ax.plot(low, label="Low")
    ax.plot(close, label="Close")
    ax.set_title("Original MG - OHLC")
    ax.legend()
    
    # Plot enhanced OHLC
    ax = axes[0, 1]
    sample_idx = np.random.randint(0, len(enhanced_samples))
    sample = enhanced_samples[sample_idx]
    
    # Extract OHLC
    open_price = sample[0]
    high = sample[1]
    low = sample[2]
    close = sample[3]
    
    # Plot
    ax.plot(open_price, label="Open")
    ax.plot(high, label="High")
    ax.plot(low, label="Low")
    ax.plot(close, label="Close")
    ax.set_title("Enhanced MG - OHLC")
    ax.legend()
    
    # Plot original volume
    ax = axes[1, 0]
    sample_idx = np.random.randint(0, len(original_samples))
    sample = original_samples[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Original MG - Volume")
    
    # Plot enhanced volume
    ax = axes[1, 1]
    sample_idx = np.random.randint(0, len(enhanced_samples))
    sample = enhanced_samples[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Enhanced MG - Volume")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multiverse_generator", "sample_comparison.png"))
    
    # Create figure for distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot close price distribution
    ax = axes[0, 0]
    original_close = np.concatenate([sample[3] for sample in original_samples])
    enhanced_close = np.concatenate([sample[3] for sample in enhanced_samples])
    
    ax.hist(original_close, bins=50, alpha=0.5, label="Original")
    ax.hist(enhanced_close, bins=50, alpha=0.5, label="Enhanced")
    ax.set_title("Close Price Distribution")
    ax.legend()
    
    # Plot volume distribution
    ax = axes[0, 1]
    original_volume = np.concatenate([sample[4] for sample in original_samples])
    enhanced_volume = np.concatenate([sample[4] for sample in enhanced_samples])
    
    ax.hist(original_volume, bins=50, alpha=0.5, label="Original")
    ax.hist(enhanced_volume, bins=50, alpha=0.5, label="Enhanced")
    ax.set_title("Volume Distribution")
    ax.legend()
    
    # Plot returns distribution
    ax = axes[1, 0]
    original_returns = np.diff(np.concatenate([sample[3] for sample in original_samples]))
    enhanced_returns = np.diff(np.concatenate([sample[3] for sample in enhanced_samples]))
    
    ax.hist(original_returns, bins=50, alpha=0.5, label="Original")
    ax.hist(enhanced_returns, bins=50, alpha=0.5, label="Enhanced")
    ax.set_title("Returns Distribution")
    ax.legend()
    
    # Plot volatility distribution
    ax = axes[1, 1]
    original_volatility = np.array([np.std(sample[3]) for sample in original_samples])
    enhanced_volatility = np.array([np.std(sample[3]) for sample in enhanced_samples])
    
    ax.hist(original_volatility, bins=50, alpha=0.5, label="Original")
    ax.hist(enhanced_volatility, bins=50, alpha=0.5, label="Enhanced")
    ax.set_title("Volatility Distribution")
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "multiverse_generator", "distribution_comparison.png"))
    
    return original_samples, enhanced_samples

def test_topo_quantum_encoder(original_tqe, enhanced_tqe, batch_size, output_dir):
    """Test Topo-Quantum Encoder."""
    logger.info("Testing Topo-Quantum Encoder...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "topo_quantum_encoder"), exist_ok=True)
    
    # Create random inputs
    X = torch.randn(batch_size, original_tqe.input_channels, original_tqe.history_length, device=config.hardware.device)
    
    # Forward pass with original model
    with Timer("Original TQE forward pass"):
        with torch.no_grad():
            original_features = original_tqe(X)
    
    # Forward pass with enhanced model
    with Timer("Enhanced TQE forward pass"):
        with torch.no_grad():
            enhanced_features = enhanced_tqe(X)
    
    # Compare features
    logger.info(f"Original features shape: {original_features.shape}")
    logger.info(f"Enhanced features shape: {enhanced_features.shape}")
    
    # Convert to numpy
    original_features_np = original_features.detach().cpu().numpy()
    enhanced_features_np = enhanced_features.detach().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original features
    ax = axes[0]
    im = ax.imshow(original_features_np, aspect='auto', cmap='viridis')
    ax.set_title("Original TQE Features")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced features
    ax = axes[1]
    im = ax.imshow(enhanced_features_np, aspect='auto', cmap='viridis')
    ax.set_title("Enhanced TQE Features")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topo_quantum_encoder", "feature_comparison.png"))
    
    # Create figure for feature correlation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compute correlation matrices
    original_corr = np.corrcoef(original_features_np.T)
    enhanced_corr = np.corrcoef(enhanced_features_np.T)
    
    # Plot original correlation
    ax = axes[0]
    im = ax.imshow(original_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title("Original TQE Feature Correlation")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced correlation
    ax = axes[1]
    im = ax.imshow(enhanced_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title("Enhanced TQE Feature Correlation")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "topo_quantum_encoder", "correlation_comparison.png"))
    
    return original_features, enhanced_features

def test_superposition_pool(original_sp3, enhanced_sp3, features, batch_size, output_dir):
    """Test Superposition Pool."""
    logger.info("Testing Superposition Pool...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "superposition_pool"), exist_ok=True)
    
    # Create random regime
    regime = torch.rand(batch_size, 2, device=config.hardware.device)
    
    # Forward pass with original model
    with Timer("Original SP3 forward pass"):
        with torch.no_grad():
            original_outputs = original_sp3(features, regime)
    
    # Forward pass with enhanced model
    with Timer("Enhanced SP3 forward pass"):
        with torch.no_grad():
            enhanced_outputs = enhanced_sp3(features, regime)
    
    # Compare outputs
    logger.info(f"Original outputs shape: {original_outputs.shape}")
    logger.info(f"Enhanced outputs shape: {enhanced_outputs.shape}")
    
    # Convert to numpy
    original_outputs_np = original_outputs.detach().cpu().numpy()
    enhanced_outputs_np = enhanced_outputs.detach().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original outputs
    ax = axes[0]
    im = ax.imshow(original_outputs_np, aspect='auto', cmap='viridis')
    ax.set_title("Original SP3 Outputs")
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced outputs
    ax = axes[1]
    im = ax.imshow(enhanced_outputs_np, aspect='auto', cmap='viridis')
    ax.set_title("Enhanced SP3 Outputs")
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "superposition_pool", "output_comparison.png"))
    
    # Create figure for output distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original distribution
    ax = axes[0]
    ax.hist(original_outputs_np.flatten(), bins=50)
    ax.set_title("Original SP3 Output Distribution")
    ax.set_xlabel("Output Value")
    ax.set_ylabel("Count")
    
    # Plot enhanced distribution
    ax = axes[1]
    ax.hist(enhanced_outputs_np.flatten(), bins=50)
    ax.set_title("Enhanced SP3 Output Distribution")
    ax.set_xlabel("Output Value")
    ax.set_ylabel("Count")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "superposition_pool", "distribution_comparison.png"))
    
    # Create figure for head weights
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original head weights
    ax = axes[0]
    original_weights = torch.nn.functional.softmax(original_sp3.head_weights, dim=0).detach().cpu().numpy()
    ax.bar(range(len(original_weights)), original_weights)
    ax.set_title("Original SP3 Head Weights")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Weight")
    
    # Plot enhanced head weights
    ax = axes[1]
    enhanced_weights = torch.nn.functional.softmax(enhanced_sp3.head_weights, dim=0).detach().cpu().numpy()
    ax.bar(range(len(enhanced_weights)), enhanced_weights)
    ax.set_title("Enhanced SP3 Head Weights")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Weight")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "superposition_pool", "head_weights_comparison.png"))
    
    return original_outputs, enhanced_outputs

def test_end_to_end(original_models, enhanced_models, train_loader, output_dir):
    """Test end-to-end pipeline."""
    logger.info("Testing end-to-end pipeline...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "end_to_end"), exist_ok=True)
    
    # Unpack models
    original_mg, original_tqe, original_sp3 = original_models
    enhanced_mg, enhanced_tqe, enhanced_sp3 = enhanced_models
    
    # Initialize metrics
    original_correct = 0
    original_total = 0
    enhanced_correct = 0
    enhanced_total = 0
    
    # Process batches
    for batch_idx, (X, y, regime) in enumerate(tqdm(train_loader, desc="Testing")):
        if batch_idx >= 10:
            break
        
        # Move data to device
        X = X.to(config.hardware.device)
        y = y.to(config.hardware.device)
        regime = regime.to(config.hardware.device)
        
        # Forward pass with original models
        with torch.no_grad():
            original_features = original_tqe(X)
            original_outputs = original_sp3(original_features, regime)
        
        # Forward pass with enhanced models
        with torch.no_grad():
            enhanced_features = enhanced_tqe(X)
            enhanced_outputs = enhanced_sp3(enhanced_features, regime)
        
        # Compute accuracy
        _, original_predicted = original_outputs.max(1)
        original_total += y.size(0)
        original_correct += original_predicted.eq(y).sum().item()
        
        _, enhanced_predicted = enhanced_outputs.max(1)
        enhanced_total += y.size(0)
        enhanced_correct += enhanced_predicted.eq(y).sum().item()
    
    # Compute accuracy
    original_accuracy = 100.0 * original_correct / original_total
    enhanced_accuracy = 100.0 * enhanced_correct / enhanced_total
    
    logger.info(f"Original accuracy: {original_accuracy:.2f}%")
    logger.info(f"Enhanced accuracy: {enhanced_accuracy:.2f}%")
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot accuracy
    ax.bar(["Original", "Enhanced"], [original_accuracy, enhanced_accuracy])
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Accuracy Comparison")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "end_to_end", "accuracy_comparison.png"))
    
    return original_accuracy, enhanced_accuracy

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load original models
    original_mg, original_tqe, original_sp3 = load_models(args.original_dir)
    
    # Load enhanced models
    enhanced_mg, enhanced_tqe, enhanced_sp3 = load_models(args.enhanced_dir)
    
    # Test Multiverse Generator
    original_samples, enhanced_samples = test_multiverse_generator(
        original_mg, enhanced_mg, args.num_samples, args.output_dir
    )
    
    # Test Topo-Quantum Encoder
    original_features, enhanced_features = test_topo_quantum_encoder(
        original_tqe, enhanced_tqe, args.batch_size, args.output_dir
    )
    
    # Test Superposition Pool
    original_outputs, enhanced_outputs = test_superposition_pool(
        original_sp3, enhanced_sp3, original_features, args.batch_size, args.output_dir
    )
    
    # Create data loaders
    train_loader, val_loader = create_efficient_dataloaders(
        symbols=args.symbols,
        batch_size=args.batch_size,
        prefetch=True,
        balanced_batches=True,
    )
    
    # Test end-to-end pipeline
    original_accuracy, enhanced_accuracy = test_end_to_end(
        (original_mg, original_tqe, original_sp3),
        (enhanced_mg, enhanced_tqe, enhanced_sp3),
        train_loader, args.output_dir
    )
    
    # Print summary
    logger.info("\nTest Results Summary:")
    logger.info("-" * 50)
    logger.info(f"Original accuracy: {original_accuracy:.2f}%")
    logger.info(f"Enhanced accuracy: {enhanced_accuracy:.2f}%")
    logger.info(f"Improvement: {enhanced_accuracy - original_accuracy:.2f}%")
    logger.info("-" * 50)
    
    logger.info("Testing completed successfully!")

if __name__ == "__main__":
    main()
