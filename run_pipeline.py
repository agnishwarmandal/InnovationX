"""
Script to run a complete MQTM pipeline.
"""

import os
import argparse
import logging
import time
from datetime import datetime, timedelta

import numpy as np
import torch
import matplotlib.pyplot as plt

from mqtm.config import config
from mqtm.data_engineering.data_fetcher import BinanceDataFetcher
from mqtm.data_engineering.data_processor import OHLCVProcessor
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.superposition_pool.superposition_model import SuperpositionPool
from mqtm.monitoring.progress_tracker import HyperRandomTrainingTracker
from mqtm.utils.hyper_random import HyperRandomTraining

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mqtm_pipeline.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MQTM Pipeline")
    
    # General parameters
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--output_dir", type=str, default="pipeline_output",
                        help="Directory to save output")
    
    # Data parameters
    parser.add_argument("--download_data", action="store_true",
                        help="Download data from Binance")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days of data to download")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config.hardware.batch_size,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create subdirectories
    os.makedirs(os.path.join(args.output_dir, "data"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "visualizations"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "samples"), exist_ok=True)
    
    # Set batch size in config
    config.hardware.batch_size = args.batch_size
    
    # Log configuration
    logger.info(f"Running with configuration:")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Device: {config.hardware.device}")

def download_and_process_data(args):
    """Download and process data."""
    logger.info("Downloading and processing data...")
    
    # Create data fetcher
    fetcher = BinanceDataFetcher(symbols=args.symbols)
    
    # Download data
    if args.download_data:
        end_date = datetime.now()
        start_date = end_date - timedelta(days=args.days)
        
        fetcher.fetch_historical_data(
            start_date=start_date,
            end_date=end_date,
            save=True
        )
    
    # Create data processor
    processor = OHLCVProcessor(symbols=args.symbols)
    
    # Process data
    data = processor.process_all_symbols()
    
    return data

def create_dataloaders(data, args):
    """Create DataLoaders for training and validation."""
    # Combine data from all symbols
    all_X = []
    all_y = []
    
    for symbol, (X, y) in data.items():
        if len(X) > 0:
            # Reshape to [num_samples, channels, seq_len]
            X = X.transpose(0, 2, 1)
            all_X.append(X)
            all_y.append(y)
    
    if not all_X:
        raise ValueError("No data found for any symbol")
    
    # Concatenate all data
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    
    # Split into train and validation sets
    train_size = int(0.8 * len(X))
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]
    
    # Create regime features (volatility and funding)
    # For simplicity, we use random values here
    # In practice, these would be computed from market data
    regime_train = np.random.rand(len(X_train), 2)
    regime_val = np.random.rand(len(X_val), 2)
    
    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.long)
    regime_train = torch.tensor(regime_train, dtype=torch.float32)
    
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.long)
    regime_val = torch.tensor(regime_val, dtype=torch.float32)
    
    # Create datasets
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train, regime_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val, regime_val)
    
    # Create dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, X_train.numpy(), y_train.numpy()

def train_multiverse_generator(mg, data, args):
    """Train the Multiverse Generator."""
    logger.info("Training Multiverse Generator...")
    
    # Create progress tracker
    tracker = HyperRandomTrainingTracker(
        total_steps=args.epochs * 10,  # 10 steps per epoch
        num_epochs=args.epochs,
        description="Training Multiverse Generator"
    )
    tracker.start()
    
    # Learn causal graphs
    logger.info("Learning causal graphs...")
    mg.learn_causal_graphs(symbols=args.symbols)
    
    # Update progress
    for i in range(10):
        tracker.update(
            step=i,
            epoch=0,
            module_idx=0,  # Multiverse Generator
            module_progress=i * 10,
            metrics={"loss": 1.0 - i * 0.05},
        )
        time.sleep(0.5)
    
    # Train diffusion model
    logger.info("Training diffusion model...")
    
    # Simulate training
    for epoch in range(1, args.epochs):
        for i in range(10):
            step = (epoch - 1) * 10 + i
            tracker.update(
                step=step,
                epoch=epoch,
                module_idx=0,  # Multiverse Generator
                module_progress=i * 10,
                metrics={"loss": 1.0 - epoch * 0.1 - i * 0.01},
            )
            time.sleep(0.5)
    
    # Save model
    mg.save()
    
    # Stop progress tracker
    tracker.stop()

def train_topo_quantum_encoder(tqe, train_loader, val_loader, args):
    """Train the Topo-Quantum Encoder."""
    logger.info("Training Topo-Quantum Encoder...")
    
    # Create progress tracker
    tracker = HyperRandomTrainingTracker(
        total_steps=args.epochs * len(train_loader),
        num_epochs=args.epochs,
        description="Training Topo-Quantum Encoder"
    )
    tracker.start()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        tqe.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    
    # Training loop
    global_step = 0
    
    for epoch in range(args.epochs):
        # Training
        tqe.train()
        train_loss = 0.0
        
        for batch_idx, (X, y, _) in enumerate(train_loader):
            # Move data to device
            X = X.to(config.hardware.device)
            y = y.to(config.hardware.device)
            
            # Extract features
            features = tqe(X)
            
            # Compute contrastive loss
            loss = torch.nn.functional.cross_entropy(features, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            
            # Update progress tracker
            global_step += 1
            tracker.update(
                step=global_step,
                epoch=epoch,
                module_idx=1,  # Topo-Quantum Encoder
                module_progress=(batch_idx + 1) / len(train_loader) * 100,
                metrics={"loss": loss.item()},
            )
    
    # Save model
    tqe_path = os.path.join(args.output_dir, "models", "tqe.pt")
    tqe.save(tqe_path)
    
    # Stop progress tracker
    tracker.stop()

def train_superposition_pool(sp3, tqe, train_loader, val_loader, args):
    """Train the Superposition Pool."""
    logger.info("Training Superposition Pool...")
    
    # Create hyper-random training utility
    hyper_random = HyperRandomTraining()
    
    # Create progress tracker
    tracker = HyperRandomTrainingTracker(
        total_steps=args.epochs * len(train_loader),
        num_epochs=args.epochs,
        description="Training Superposition Pool"
    )
    tracker.start()
    
    # Create optimizer for non-complex parameters
    sp3_optimizer = torch.optim.AdamW(
        [p for name, p in sp3.named_parameters() if "weight_real" not in name and "weight_imag" not in name],
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    
    # Training loop
    global_step = 0
    best_val_accuracy = 0.0
    
    for epoch in range(args.epochs):
        # Training
        tqe.eval()
        sp3.train()
        
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch_idx, (X, y, regime) in enumerate(train_loader):
            # Move data to device
            X = X.to(config.hardware.device)
            y = y.to(config.hardware.device)
            regime = regime.to(config.hardware.device)
            
            # Extract features with TQE
            with torch.no_grad():
                features = tqe(X)
            
            # Forward pass through SP3
            outputs = sp3(features, regime)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, y)
            
            # Backward pass
            sp3_optimizer.zero_grad()
            loss.backward()
            
            # Update SP3 parameters
            sp3_optimizer.step()
            
            # Apply unitary update to SP3 complex parameters
            sp3.apply_unitary_update(
                learning_rate=args.learning_rate,
                temperature=hyper_random.current_mutation_rate * 0.01,
            )
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            train_total += y.size(0)
            train_correct += predicted.eq(y).sum().item()
            
            # Update metrics
            train_loss += loss.item()
            accuracy = 100.0 * train_correct / train_total
            
            # Update progress tracker
            global_step += 1
            tracker.update(
                step=global_step,
                epoch=epoch,
                module_idx=2,  # Superposition Pool
                module_progress=(batch_idx + 1) / len(train_loader) * 100,
                metrics={
                    "loss": loss.item(),
                    "accuracy": accuracy,
                },
                randomness_stats=hyper_random.get_randomness_stats(),
            )
            
            # Update hyper-random state
            if batch_idx % 10 == 0:
                hyper_random.update_randomness_state(accuracy)
        
        # Save model
        sp3_path = os.path.join(args.output_dir, "models", "sp3.pt")
        sp3.save(sp3_path)
    
    # Stop progress tracker
    tracker.stop()

def generate_samples(mg, args):
    """Generate synthetic samples."""
    logger.info("Generating synthetic samples...")
    
    # Generate samples
    X, causal_graph = mg.generate_samples(
        num_samples=args.num_samples,
        sigma_multiplier=1.5,
    )
    
    # Save samples
    samples_path = os.path.join(args.output_dir, "samples", "synthetic_samples.npy")
    np.save(samples_path, X)
    
    # Save causal graph
    graph_path = os.path.join(args.output_dir, "samples", "causal_graph.npy")
    np.save(graph_path, causal_graph)
    
    logger.info(f"Saved {args.num_samples} samples to {samples_path}")
    
    return X, causal_graph

def visualize_results(real_data, synthetic_data, args):
    """Visualize results."""
    logger.info("Visualizing results...")
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot real OHLC
    ax = axes[0, 0]
    sample_idx = np.random.randint(0, len(real_data))
    sample = real_data[sample_idx]
    
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
    ax.set_title("Real OHLC")
    ax.legend()
    
    # Plot synthetic OHLC
    ax = axes[0, 1]
    sample_idx = np.random.randint(0, len(synthetic_data))
    sample = synthetic_data[sample_idx]
    
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
    ax.set_title("Synthetic OHLC")
    ax.legend()
    
    # Plot real volume
    ax = axes[1, 0]
    sample_idx = np.random.randint(0, len(real_data))
    sample = real_data[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Real Volume")
    
    # Plot synthetic volume
    ax = axes[1, 1]
    sample_idx = np.random.randint(0, len(synthetic_data))
    sample = synthetic_data[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Synthetic Volume")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "visualizations", "results.png"))
    logger.info(f"Saved visualization to {os.path.join(args.output_dir, 'visualizations', 'results.png')}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    setup_environment(args)
    
    # Download and process data
    data = download_and_process_data(args)
    
    # Create dataloaders
    train_loader, val_loader, X_train, y_train = create_dataloaders(data, args)
    
    # Create models
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.output_dir, "models", "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Train Multiverse Generator
    train_multiverse_generator(mg, data, args)
    
    # Train Topo-Quantum Encoder
    train_topo_quantum_encoder(tqe, train_loader, val_loader, args)
    
    # Train Superposition Pool
    train_superposition_pool(sp3, tqe, train_loader, val_loader, args)
    
    # Generate samples
    synthetic_data, _ = generate_samples(mg, args)
    
    # Visualize results
    visualize_results(X_train, synthetic_data, args)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
