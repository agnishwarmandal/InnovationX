"""
Main training script for MQTM.
"""

import os
import argparse
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
        logging.FileHandler("mqtm_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MQTM system")
    
    # General parameters
    parser.add_argument("--mode", type=str, default="train", choices=["train", "test", "generate"],
                        help="Mode: train, test, or generate")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--output_dir", type=str, default="models",
                        help="Directory to save models")
    
    # Data parameters
    parser.add_argument("--download_data", action="store_true",
                        help="Download data from Binance")
    parser.add_argument("--days", type=int, default=30,
                        help="Number of days of data to download")
    
    # Training parameters
    parser.add_argument("--batch_size", type=int, default=config.hardware.batch_size,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed")
    
    # Model parameters
    parser.add_argument("--num_heads", type=int, default=config.superposition_pool.max_heads,
                        help="Number of superposition heads")
    parser.add_argument("--hidden_dim", type=int, default=config.superposition_pool.hidden_dim,
                        help="Hidden dimension for superposition heads")
    
    # Generation parameters
    parser.add_argument("--num_samples", type=int, default=1000,
                        help="Number of samples to generate")
    parser.add_argument("--sigma_multiplier", type=float, default=1.0,
                        help="Sigma multiplier for generation")
    
    return parser.parse_args()

def setup_environment(args):
    """Set up the training environment."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set random seed if provided
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    
    # Set batch size in config
    config.hardware.batch_size = args.batch_size
    
    # Set number of heads in config
    config.superposition_pool.max_heads = args.num_heads
    
    # Set hidden dimension in config
    config.superposition_pool.hidden_dim = args.hidden_dim
    
    # Log configuration
    logger.info(f"Running with configuration:")
    logger.info(f"  Mode: {args.mode}")
    logger.info(f"  Symbols: {args.symbols}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Learning rate: {args.learning_rate}")
    logger.info(f"  Number of heads: {args.num_heads}")
    logger.info(f"  Hidden dimension: {args.hidden_dim}")
    logger.info(f"  Device: {config.hardware.device}")

def download_and_process_data(args):
    """Download and process data."""
    logger.info("Downloading and processing data...")
    
    # Create data fetcher
    fetcher = BinanceDataFetcher(symbols=args.symbols)
    
    # Download data
    if args.download_data:
        from datetime import datetime, timedelta
        
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
    train_dataset = TensorDataset(X_train, y_train, regime_train)
    val_dataset = TensorDataset(X_val, y_val, regime_val)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=config.hardware.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def create_models(args):
    """Create MQTM models."""
    # Create Multiverse Generator
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.output_dir, "multiverse_generator"),
    )
    
    # Create Topo-Quantum Encoder
    tqe = TopoQuantumEncoder(
        input_channels=5,  # OHLCV
        seq_len=config.data.history_length,
    )
    
    # Create Superposition Pool
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    
    return mg, tqe, sp3

def train_models(mg, tqe, sp3, train_loader, val_loader, args):
    """Train MQTM models."""
    # Create hyper-random training utility
    hyper_random = HyperRandomTraining()
    
    # Create progress tracker
    total_steps = args.epochs * len(train_loader)
    tracker = HyperRandomTrainingTracker(
        total_steps=total_steps,
        num_epochs=args.epochs,
        description="MQTM Training"
    )
    tracker.start()
    
    # Create optimizer for TQE
    tqe_optimizer = torch.optim.AdamW(
        tqe.parameters(),
        lr=args.learning_rate,
        weight_decay=1e-5,
    )
    
    # Create optimizer for SP3 (only for non-complex parameters)
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
        tqe.train()
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
            features = tqe(X)
            
            # Forward pass through SP3
            outputs = sp3(features, regime)
            
            # Compute loss
            loss = F.cross_entropy(outputs, y)
            
            # Backward pass
            tqe_optimizer.zero_grad()
            sp3_optimizer.zero_grad()
            loss.backward()
            
            # Update TQE parameters
            tqe_optimizer.step()
            
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
        
        # Compute average training metrics
        train_loss /= len(train_loader)
        train_accuracy = 100.0 * train_correct / train_total
        
        # Validation
        tqe.eval()
        sp3.eval()
        
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for X, y, regime in val_loader:
                # Move data to device
                X = X.to(config.hardware.device)
                y = y.to(config.hardware.device)
                regime = regime.to(config.hardware.device)
                
                # Extract features with TQE
                features = tqe(X)
                
                # Forward pass through SP3
                outputs = sp3(features, regime)
                
                # Compute loss
                loss = F.cross_entropy(outputs, y)
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                val_total += y.size(0)
                val_correct += predicted.eq(y).sum().item()
                
                # Update metrics
                val_loss += loss.item()
        
        # Compute average validation metrics
        val_loss /= len(val_loader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Log metrics
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, "
                   f"Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        
        # Save best model
        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            
            # Save TQE
            tqe_path = os.path.join(args.output_dir, "tqe.pt")
            tqe.save(tqe_path)
            
            # Save SP3
            sp3_path = os.path.join(args.output_dir, "sp3.pt")
            sp3.save(sp3_path)
            
            logger.info(f"Saved best model with validation accuracy {val_accuracy:.2f}%")
        
        # Update SP3 head weights based on validation loss
        # Compute loss for each head
        head_losses = []
        
        with torch.no_grad():
            for head_idx, head in enumerate(sp3.heads):
                head_loss = 0.0
                
                for X, y, regime in val_loader:
                    # Move data to device
                    X = X.to(config.hardware.device)
                    y = y.to(config.hardware.device)
                    regime = regime.to(config.hardware.device)
                    
                    # Extract features with TQE
                    features = tqe(X)
                    
                    # Forward pass through head
                    outputs = head(features, regime)
                    
                    # Compute loss
                    loss = F.cross_entropy(outputs, y)
                    
                    # Update metrics
                    head_loss += loss.item()
                
                # Compute average loss
                head_loss /= len(val_loader)
                head_losses.append(head_loss)
        
        # Update head weights
        head_losses_tensor = torch.tensor(head_losses, device=config.hardware.device)
        sp3.update_head_weights(head_losses_tensor)
        
        # Check if we need to prune or spawn heads
        entropy = sp3.compute_entropy()
        logger.info(f"Head weights entropy: {entropy:.4f}")
        
        # Prune heads with low weight
        for i in range(sp3.num_heads - 1, -1, -1):
            weight = sp3.head_weights[i].item()
            if weight < config.bayesian_online_mixture.min_weight_threshold:
                logger.info(f"Pruning head {i} with weight {weight:.4f}")
                sp3.prune_head(i)
        
        # Spawn new head if entropy is low
        if entropy < config.bayesian_online_mixture.min_entropy_threshold and sp3.num_heads < config.superposition_pool.max_heads:
            logger.info(f"Spawning new head (current entropy: {entropy:.4f})")
            sp3.spawn_head()
    
    # Stop progress tracker
    tracker.stop()
    
    # Save final models
    # Save TQE
    tqe_path = os.path.join(args.output_dir, "tqe_final.pt")
    tqe.save(tqe_path)
    
    # Save SP3
    sp3_path = os.path.join(args.output_dir, "sp3_final.pt")
    sp3.save(sp3_path)
    
    logger.info(f"Saved final models")

def test_models(mg, tqe, sp3, val_loader, args):
    """Test MQTM models."""
    # Load best models
    tqe_path = os.path.join(args.output_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(args.output_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Evaluation
    tqe.eval()
    sp3.eval()
    
    test_loss = 0.0
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for X, y, regime in val_loader:
            # Move data to device
            X = X.to(config.hardware.device)
            y = y.to(config.hardware.device)
            regime = regime.to(config.hardware.device)
            
            # Extract features with TQE
            features = tqe(X)
            
            # Forward pass through SP3
            outputs = sp3(features, regime)
            
            # Compute loss
            loss = F.cross_entropy(outputs, y)
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            test_total += y.size(0)
            test_correct += predicted.eq(y).sum().item()
            
            # Update metrics
            test_loss += loss.item()
    
    # Compute average test metrics
    test_loss /= len(val_loader)
    test_accuracy = 100.0 * test_correct / test_total
    
    # Log metrics
    logger.info(f"Test Loss: {test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")

def generate_samples(mg, args):
    """Generate synthetic samples with the Multiverse Generator."""
    # Load causal graphs if available
    mg_dir = os.path.join(args.output_dir, "multiverse_generator")
    causal_graphs_path = os.path.join(mg_dir, "causal_graphs.pt")
    
    if os.path.exists(causal_graphs_path):
        mg.causal_graphs = torch.load(causal_graphs_path, map_location=config.hardware.device)
    else:
        # Learn causal graphs
        logger.info("Learning causal graphs...")
        mg.learn_causal_graphs(symbols=args.symbols)
    
    # Generate samples
    logger.info(f"Generating {args.num_samples} samples...")
    X, causal_graph = mg.generate_samples(
        num_samples=args.num_samples,
        sigma_multiplier=args.sigma_multiplier,
    )
    
    # Save samples
    samples_dir = os.path.join(args.output_dir, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    
    samples_path = os.path.join(samples_dir, f"samples_{datetime.now().strftime('%Y%m%d_%H%M%S')}.npy")
    np.save(samples_path, X)
    
    logger.info(f"Saved {args.num_samples} samples to {samples_path}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set up environment
    setup_environment(args)
    
    # Download and process data
    data = download_and_process_data(args)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(data, args)
    
    # Create models
    mg, tqe, sp3 = create_models(args)
    
    # Run in specified mode
    if args.mode == "train":
        train_models(mg, tqe, sp3, train_loader, val_loader, args)
    elif args.mode == "test":
        test_models(mg, tqe, sp3, val_loader, args)
    elif args.mode == "generate":
        generate_samples(mg, args)

if __name__ == "__main__":
    main()
