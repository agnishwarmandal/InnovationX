"""
Script to profile the performance of MQTM models.
"""

import os
import argparse
import logging
import time
import torch

from mqtm.config import config
from mqtm.utils.performance_profiling import (
    Timer, FunctionProfiler, TorchProfiler, DataLoaderProfiler
)
from mqtm.data_engineering.efficient_dataloader import create_efficient_dataloaders
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.superposition_pool.superposition_model import SuperpositionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Profile MQTM Models")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--batch_size", type=int, default=config.hardware.batch_size,
                        help="Batch size for profiling")
    parser.add_argument("--num_batches", type=int, default=10,
                        help="Number of batches to profile")
    parser.add_argument("--profile_dir", type=str, default="profiles",
                        help="Directory to save profiling results")
    
    return parser.parse_args()

def profile_dataloader(symbols, batch_size, num_batches, profile_dir):
    """Profile data loader performance."""
    logger.info("Profiling data loader performance...")
    
    # Create data loaders
    train_loader, val_loader = create_efficient_dataloaders(
        symbols=symbols,
        batch_size=batch_size,
        prefetch=True,
        balanced_batches=True,
    )
    
    # Profile data loaders
    with Timer("Train loader profiling"):
        train_stats = DataLoaderProfiler.profile_dataloader(
            dataloader=train_loader,
            num_batches=num_batches,
        )
    
    with Timer("Validation loader profiling"):
        val_stats = DataLoaderProfiler.profile_dataloader(
            dataloader=val_loader,
            num_batches=num_batches,
        )
    
    # Save profiling results
    os.makedirs(profile_dir, exist_ok=True)
    
    import json
    with open(os.path.join(profile_dir, "dataloader_profile.json"), "w") as f:
        json.dump({
            "train_loader": train_stats,
            "val_loader": val_stats,
        }, f, indent=2)
    
    return train_loader, val_loader

def profile_multiverse_generator(mg, batch_size, profile_dir):
    """Profile Multiverse Generator performance."""
    logger.info("Profiling Multiverse Generator...")
    
    # Create random inputs
    seq_len = config.data.history_length
    x = torch.randn(batch_size, 5, seq_len, device=config.hardware.device)
    
    # Profile causal graph learning
    if mg.causal_graphs is None:
        # Create random data for causal graph learning
        latent_factors = torch.randn(1000, mg.num_latent_factors, device=config.hardware.device)
        
        with Timer("Causal graph learning"):
            # Convert to numpy for causal graph learner
            latent_factors_np = latent_factors.cpu().numpy()
            
            # Learn causal graph
            graph = mg.causal_graph_learner.fit(latent_factors_np)
            
            # Create causal graphs tensor
            mg.causal_graphs = torch.tensor(
                [graph], dtype=torch.float32, device=config.hardware.device
            )
    
    # Profile diffusion model
    if mg.diffusion_model and mg.diffusion_model.model:
        # Create random inputs for diffusion model
        timesteps = torch.randint(
            0, mg.diffusion_model.num_timesteps, (batch_size,), device=config.hardware.device
        )
        causal_graph = mg.causal_graphs[0].unsqueeze(0).expand(batch_size, -1, -1)
        
        # Profile forward pass
        with Timer("Diffusion model forward pass"):
            for _ in range(10):
                with torch.no_grad():
                    mg.diffusion_model.model(x, timesteps, causal_graph)
        
        # Profile with PyTorch profiler
        trace_filename = os.path.join(profile_dir, "mg_forward_profile")
        
        with TorchProfiler.profile_model(
            model=lambda inputs: mg.diffusion_model.model(
                inputs, timesteps, causal_graph
            ),
            inputs=x,
            trace_filename=trace_filename,
        ):
            pass
    
    # Profile sample generation
    with Timer("Sample generation"):
        X, causal_graph = mg.generate_samples(
            num_samples=batch_size,
            seq_len=seq_len,
            sigma_multiplier=1.0,
            batch_size=batch_size,
        )
    
    logger.info(f"Generated samples shape: {X.shape}")
    
    return X

def profile_topo_quantum_encoder(tqe, batch_size, profile_dir):
    """Profile Topo-Quantum Encoder performance."""
    logger.info("Profiling Topo-Quantum Encoder...")
    
    # Create random inputs
    seq_len = config.data.history_length
    x = torch.randn(batch_size, 5, seq_len, device=config.hardware.device)
    
    # Profile forward pass
    with Timer("TQE forward pass"):
        for _ in range(10):
            with torch.no_grad():
                features = tqe(x)
    
    logger.info(f"TQE features shape: {features.shape}")
    
    # Profile with PyTorch profiler
    trace_filename = os.path.join(profile_dir, "tqe_forward_profile")
    
    with TorchProfiler.profile_model(
        model=tqe,
        inputs=x,
        trace_filename=trace_filename,
    ):
        pass
    
    # Profile individual components
    with Timer("Persistence layer"):
        with torch.no_grad():
            topo_features = tqe.persistence_layer(x)
    
    with Timer("Wavelet layer"):
        with torch.no_grad():
            wavelet_features = tqe.wavelet_layer(x)
    
    if tqe.use_classic_indicators:
        with Timer("Indicators layer"):
            with torch.no_grad():
                indicators_features = tqe.indicators_layer(x)
                indicators_features = torch.nn.functional.adaptive_avg_pool1d(
                    indicators_features, 1
                ).squeeze(-1)
    
    return features

def profile_superposition_pool(sp3, features, profile_dir):
    """Profile Superposition Pool performance."""
    logger.info("Profiling Superposition Pool...")
    
    # Create random inputs
    batch_size = features.shape[0]
    regime = torch.rand(batch_size, 2, device=config.hardware.device)
    
    # Profile forward pass
    with Timer("SP3 forward pass"):
        for _ in range(10):
            with torch.no_grad():
                outputs = sp3(features, regime)
    
    logger.info(f"SP3 outputs shape: {outputs.shape}")
    
    # Profile with PyTorch profiler
    trace_filename = os.path.join(profile_dir, "sp3_forward_profile")
    
    with TorchProfiler.profile_model(
        model=lambda inputs: sp3(inputs, regime),
        inputs=features,
        trace_filename=trace_filename,
    ):
        pass
    
    # Profile individual heads
    for i, head in enumerate(sp3.heads):
        with Timer(f"Head {i} forward pass"):
            with torch.no_grad():
                head_output = head(features, regime)
        
        logger.info(f"Head {i} output shape: {head_output.shape}")
    
    # Profile unitary update
    with Timer("Unitary update"):
        # Create random gradients
        for param in sp3.parameters():
            param.grad = torch.randn_like(param)
        
        # Apply unitary update
        sp3.apply_unitary_update(learning_rate=0.01)
    
    return outputs

def profile_end_to_end(train_loader, mg, tqe, sp3, num_batches, profile_dir):
    """Profile end-to-end training pipeline."""
    logger.info("Profiling end-to-end training pipeline...")
    
    # Create optimizers
    tqe_optimizer = torch.optim.AdamW(
        tqe.parameters(),
        lr=1e-4,
        weight_decay=1e-5,
    )
    
    sp3_optimizer = torch.optim.AdamW(
        [p for name, p in sp3.named_parameters() if "weight_real" not in name and "weight_imag" not in name],
        lr=1e-4,
        weight_decay=1e-5,
    )
    
    # Profile training loop
    with Timer("End-to-end training"):
        for batch_idx, (X, y, regime) in enumerate(train_loader):
            if batch_idx >= num_batches:
                break
            
            # Move data to device
            X = X.to(config.hardware.device)
            y = y.to(config.hardware.device)
            regime = regime.to(config.hardware.device)
            
            # Extract features with TQE
            tqe.train()
            tqe_optimizer.zero_grad()
            features = tqe(X)
            
            # Forward pass through SP3
            sp3.train()
            sp3_optimizer.zero_grad()
            outputs = sp3(features, regime)
            
            # Compute loss
            loss = torch.nn.functional.cross_entropy(outputs, y)
            
            # Backward pass
            loss.backward()
            
            # Update parameters
            tqe_optimizer.step()
            sp3_optimizer.step()
            
            # Apply unitary update
            sp3.apply_unitary_update(learning_rate=1e-4)
            
            logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
    
    # Profile with PyTorch profiler
    trace_filename = os.path.join(profile_dir, "end_to_end_profile")
    
    # Get a batch
    for X, y, regime in train_loader:
        # Move data to device
        X = X.to(config.hardware.device)
        y = y.to(config.hardware.device)
        regime = regime.to(config.hardware.device)
        break
    
    # Profile forward and backward pass
    TorchProfiler.profile_forward_backward(
        model=lambda x: sp3(tqe(x), regime),
        loss_fn=lambda outputs, targets: torch.nn.functional.cross_entropy(outputs, targets),
        optimizer=tqe_optimizer,
        inputs=X,
        targets=y,
        trace_filename=trace_filename,
    )

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create profile directory
    os.makedirs(args.profile_dir, exist_ok=True)
    
    # Profile data loader
    train_loader, val_loader = profile_dataloader(
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_batches=args.num_batches,
        profile_dir=args.profile_dir,
    )
    
    # Load models
    logger.info(f"Loading models from {args.models_dir}...")
    
    # Create models
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.models_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load models if available
    mg.load()
    
    tqe_path = os.path.join(args.models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(args.models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Move models to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    # Profile models
    X = profile_multiverse_generator(mg, args.batch_size, args.profile_dir)
    features = profile_topo_quantum_encoder(tqe, args.batch_size, args.profile_dir)
    outputs = profile_superposition_pool(sp3, features, args.profile_dir)
    
    # Profile end-to-end
    profile_end_to_end(train_loader, mg, tqe, sp3, args.num_batches, args.profile_dir)
    
    # Print profiling statistics
    FunctionProfiler.print_stats()
    
    logger.info(f"Profiling complete. Results saved to {args.profile_dir}")

if __name__ == "__main__":
    main()
