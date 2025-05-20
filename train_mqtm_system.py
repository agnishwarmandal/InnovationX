"""
Script to train the MQTM system using cryptocurrency data.
"""

import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any

from mqtm.config import config
from mqtm.data_engineering.crypto_data_loader import CryptoDataLoader
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
        logging.FileHandler("train_mqtm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train MQTM System")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to train on (if None, use all available)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for training")
    parser.add_argument("--history_length", type=int, default=100,
                        help="Length of historical data to use")
    parser.add_argument("--forecast_horizon", type=int, default=20,
                        help="Length of forecast horizon")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--train_mg", action="store_true",
                        help="Whether to train the Multiverse Generator")
    parser.add_argument("--train_tqe", action="store_true",
                        help="Whether to train the Topo-Quantum Encoder")
    parser.add_argument("--train_sp3", action="store_true",
                        help="Whether to train the Superposition Pool")
    parser.add_argument("--train_all", action="store_true",
                        help="Whether to train all components")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Whether to optimize memory usage")
    parser.add_argument("--profile_performance", action="store_true",
                        help="Whether to profile performance")
    
    return parser.parse_args()

def load_data(args):
    """Load cryptocurrency data."""
    logger.info(f"Loading data from {args.data_dir}...")
    
    # Create data loader
    data_loader = CryptoDataLoader(
        data_dir=args.data_dir,
        history_length=args.history_length,
        forecast_horizon=args.forecast_horizon,
        batch_size=args.batch_size,
        device=config.hardware.device,
    )
    
    # Load data
    data_loader.load_data(args.symbols)
    
    # Get available symbols
    available_symbols = data_loader.get_symbols()
    logger.info(f"Available symbols: {available_symbols}")
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders(args.symbols)
    
    return data_loader, train_loader, val_loader, test_loader

def create_models(args):
    """Create MQTM models."""
    logger.info("Creating MQTM models...")
    
    # Create models directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Create Multiverse Generator
    mg_dir = os.path.join(args.models_dir, "multiverse_generator")
    os.makedirs(mg_dir, exist_ok=True)
    
    mg = MultiverseGenerator(
        model_dir=mg_dir,
    )
    
    # Create Topo-Quantum Encoder
    tqe = TopoQuantumEncoder()
    
    # Create Superposition Pool
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Move models to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    # Optimize memory if requested
    if args.optimize_memory:
        logger.info("Optimizing memory usage...")
        MemoryOptimizer.optimize_model_memory(tqe)
        MemoryOptimizer.optimize_model_memory(sp3)
    
    return mg, tqe, sp3

def train_multiverse_generator(args, mg, train_loader, val_loader):
    """Train the Multiverse Generator."""
    logger.info("Training Multiverse Generator...")
    
    # Create progress bar
    progress_bar = tqdm(total=args.epochs, desc="Training MG")
    
    # Training loop
    for epoch in range(args.epochs):
        # Train for one epoch
        train_loss = mg.train_epoch(
            train_loader=train_loader,
            learning_rate=args.learning_rate,
        )
        
        # Validate
        val_loss = mg.validate(val_loader)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        })
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}")
        
        # Save model
        mg.save()
    
    # Close progress bar
    progress_bar.close()
    
    return mg

def train_topo_quantum_encoder(args, tqe, train_loader, val_loader):
    """Train the Topo-Quantum Encoder."""
    logger.info("Training Topo-Quantum Encoder...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(tqe.parameters(), lr=args.learning_rate)
    
    # Create loss function
    criterion = torch.nn.MSELoss()
    
    # Create progress bar
    progress_bar = tqdm(total=args.epochs, desc="Training TQE")
    
    # Training loop
    for epoch in range(args.epochs):
        # Initialize metrics
        train_loss = 0.0
        val_loss = 0.0
        
        # Train
        tqe.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(config.hardware.device)
            target = target.to(config.hardware.device)
            
            # Forward pass
            features = tqe(data)
            
            # Compute loss (reconstruction loss)
            reconstructed = tqe.decode(features)
            loss = criterion(reconstructed, data)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
        
        # Compute average loss
        train_loss /= len(train_loader)
        
        # Validate
        tqe.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # Move data to device
                data = data.to(config.hardware.device)
                target = target.to(config.hardware.device)
                
                # Forward pass
                features = tqe(data)
                
                # Compute loss (reconstruction loss)
                reconstructed = tqe.decode(features)
                loss = criterion(reconstructed, data)
                
                # Update metrics
                val_loss += loss.item()
        
        # Compute average loss
        val_loss /= len(val_loader)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "val_loss": f"{val_loss:.4f}",
        })
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Val Loss: {val_loss:.4f}")
        
        # Save model
        tqe.save(os.path.join(args.models_dir, "tqe.pt"))
    
    # Close progress bar
    progress_bar.close()
    
    return tqe

def train_superposition_pool(args, tqe, sp3, train_loader, val_loader):
    """Train the Superposition Pool."""
    logger.info("Training Superposition Pool...")
    
    # Create optimizer
    optimizer = torch.optim.Adam(sp3.parameters(), lr=args.learning_rate)
    
    # Create loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Create progress bar
    progress_bar = tqdm(total=args.epochs, desc="Training SP3")
    
    # Training loop
    for epoch in range(args.epochs):
        # Initialize metrics
        train_loss = 0.0
        train_acc = 0.0
        val_loss = 0.0
        val_acc = 0.0
        
        # Train
        sp3.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            # Move data to device
            data = data.to(config.hardware.device)
            target = target.to(config.hardware.device)
            
            # Extract features using TQE
            with torch.no_grad():
                features = tqe(data)
            
            # Create regime tensor (placeholder)
            batch_size = data.size(0)
            regime = torch.rand(batch_size, 2, device=config.hardware.device)
            
            # Forward pass
            outputs = sp3(features, regime)
            
            # Compute loss
            loss = criterion(outputs, target)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Apply unitary update
            sp3.apply_unitary_update()
            
            # Update metrics
            train_loss += loss.item()
            
            # Compute accuracy
            _, predicted = outputs.max(1)
            train_acc += predicted.eq(target).sum().item() / target.size(0)
        
        # Compute average metrics
        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        
        # Validate
        sp3.eval()
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                # Move data to device
                data = data.to(config.hardware.device)
                target = target.to(config.hardware.device)
                
                # Extract features using TQE
                features = tqe(data)
                
                # Create regime tensor (placeholder)
                batch_size = data.size(0)
                regime = torch.rand(batch_size, 2, device=config.hardware.device)
                
                # Forward pass
                outputs = sp3(features, regime)
                
                # Compute loss
                loss = criterion(outputs, target)
                
                # Update metrics
                val_loss += loss.item()
                
                # Compute accuracy
                _, predicted = outputs.max(1)
                val_acc += predicted.eq(target).sum().item() / target.size(0)
        
        # Compute average metrics
        val_loss /= len(val_loader)
        val_acc /= len(val_loader)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "train_loss": f"{train_loss:.4f}",
            "train_acc": f"{train_acc:.4f}",
            "val_loss": f"{val_loss:.4f}",
            "val_acc": f"{val_acc:.4f}",
        })
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{args.epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}")
        
        # Save model
        sp3.save(os.path.join(args.models_dir, "sp3.pt"))
    
    # Close progress bar
    progress_bar.close()
    
    return sp3

def test_models(mg, tqe, sp3, test_loader):
    """Test the trained models."""
    logger.info("Testing models...")
    
    # Initialize metrics
    mg_loss = 0.0
    tqe_loss = 0.0
    sp3_loss = 0.0
    sp3_acc = 0.0
    
    # Create loss functions
    mse_criterion = torch.nn.MSELoss()
    ce_criterion = torch.nn.CrossEntropyLoss()
    
    # Test
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            # Move data to device
            data = data.to(config.hardware.device)
            target = target.to(config.hardware.device)
            
            # Test Multiverse Generator
            mg_output, _ = mg.generate_samples(
                num_samples=data.size(0),
                seq_len=data.size(2),
                conditioning=data.cpu().numpy(),
            )
            mg_output = torch.tensor(mg_output, device=config.hardware.device)
            mg_loss += mse_criterion(mg_output, data).item()
            
            # Test Topo-Quantum Encoder
            tqe_features = tqe(data)
            reconstructed = tqe.decode(tqe_features)
            tqe_loss += mse_criterion(reconstructed, data).item()
            
            # Test Superposition Pool
            batch_size = data.size(0)
            regime = torch.rand(batch_size, 2, device=config.hardware.device)
            sp3_output = sp3(tqe_features, regime)
            sp3_loss += ce_criterion(sp3_output, target).item()
            
            # Compute accuracy
            _, predicted = sp3_output.max(1)
            sp3_acc += predicted.eq(target).sum().item() / target.size(0)
    
    # Compute average metrics
    mg_loss /= len(test_loader)
    tqe_loss /= len(test_loader)
    sp3_loss /= len(test_loader)
    sp3_acc /= len(test_loader)
    
    # Log results
    logger.info(f"Test Results:")
    logger.info(f"  Multiverse Generator Loss: {mg_loss:.4f}")
    logger.info(f"  Topo-Quantum Encoder Loss: {tqe_loss:.4f}")
    logger.info(f"  Superposition Pool Loss: {sp3_loss:.4f}")
    logger.info(f"  Superposition Pool Accuracy: {sp3_acc:.4f}")
    
    return {
        "mg_loss": mg_loss,
        "tqe_loss": tqe_loss,
        "sp3_loss": sp3_loss,
        "sp3_acc": sp3_acc,
    }

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Set training flags if train_all is specified
    if args.train_all:
        args.train_mg = True
        args.train_tqe = True
        args.train_sp3 = True
    
    # Create output directory
    os.makedirs(args.models_dir, exist_ok=True)
    
    # Load data
    data_loader, train_loader, val_loader, test_loader = load_data(args)
    
    # Create models
    mg, tqe, sp3 = create_models(args)
    
    # Profile performance if requested
    if args.profile_performance:
        logger.info("Profiling performance...")
        
        # Create profiler
        profiler = TorchProfiler()
        
        # Profile Multiverse Generator
        with profiler.profile_model(
            model=mg.diffusion_model.model,
            inputs=torch.randn(args.batch_size, 5, args.history_length, device=config.hardware.device),
            trace_filename="profiles/mg_profile",
        ):
            pass
        
        # Profile Topo-Quantum Encoder
        with profiler.profile_model(
            model=tqe,
            inputs=torch.randn(args.batch_size, 5, args.history_length, device=config.hardware.device),
            trace_filename="profiles/tqe_profile",
        ):
            pass
        
        # Profile Superposition Pool
        with profiler.profile_model(
            model=lambda x: sp3(x, torch.rand(args.batch_size, 2, device=config.hardware.device)),
            inputs=torch.randn(args.batch_size, tqe.total_features_dim, device=config.hardware.device),
            trace_filename="profiles/sp3_profile",
        ):
            pass
    
    # Train models
    if args.train_mg:
        mg = train_multiverse_generator(args, mg, train_loader, val_loader)
    
    if args.train_tqe:
        tqe = train_topo_quantum_encoder(args, tqe, train_loader, val_loader)
    
    if args.train_sp3:
        sp3 = train_superposition_pool(args, tqe, sp3, train_loader, val_loader)
    
    # Test models
    test_results = test_models(mg, tqe, sp3, test_loader)
    
    # Save test results
    with open(os.path.join(args.models_dir, "test_results.json"), "w") as f:
        json.dump(test_results, f, indent=2)
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
