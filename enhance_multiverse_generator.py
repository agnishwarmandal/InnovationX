"""
Script to enhance the Multiverse Generator with attention mechanisms.
"""

import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from mqtm.config import config
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.multiverse_generator.diffusion import DiffusionModel
from mqtm.multiverse_generator.attention_diffusion import AttentionDiffusionModel, AttentionDiffusion
from mqtm.utils.memory_optimization import MemoryOptimizer
from mqtm.utils.performance_profiling import Timer, TorchProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Enhance Multiverse Generator")
    
    parser.add_argument("--model_dir", type=str, default="models/multiverse_generator",
                        help="Directory with trained Multiverse Generator")
    parser.add_argument("--output_dir", type=str, default="models/enhanced_multiverse_generator",
                        help="Directory to save enhanced Multiverse Generator")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="Hidden dimension for attention model")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="Number of layers for attention model")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--head_dim", type=int, default=32,
                        help="Dimension of each attention head")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout probability")
    parser.add_argument("--num_timesteps", type=int, default=100,
                        help="Number of diffusion timesteps")
    parser.add_argument("--noise_schedule", type=str, default="cosine",
                        choices=["linear", "cosine", "quadratic"],
                        help="Noise schedule type")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--num_samples", type=int, default=10,
                        help="Number of samples to generate for comparison")
    
    return parser.parse_args()

def load_multiverse_generator(model_dir):
    """Load Multiverse Generator."""
    logger.info(f"Loading Multiverse Generator from {model_dir}...")
    
    # Create Multiverse Generator
    mg = MultiverseGenerator(
        model_dir=model_dir,
    )
    
    # Load model
    mg.load()
    
    return mg

def create_attention_diffusion_model(args, input_channels, num_latent_factors):
    """Create attention-enhanced diffusion model."""
    logger.info("Creating attention-enhanced diffusion model...")
    
    # Create model
    model = AttentionDiffusionModel(
        input_channels=input_channels,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        dropout=args.dropout,
        num_latent_factors=num_latent_factors,
    )
    
    # Create diffusion process
    diffusion = AttentionDiffusion(
        model=model,
        num_timesteps=args.num_timesteps,
        noise_schedule=args.noise_schedule,
        device=config.hardware.device,
    )
    
    return diffusion

def transfer_weights(original_model, attention_model):
    """Transfer weights from original model to attention model."""
    logger.info("Transferring weights from original model to attention model...")
    
    # Get original model state dict
    if hasattr(original_model, "model"):
        original_state_dict = original_model.model.state_dict()
    else:
        original_state_dict = original_model.state_dict()
    
    # Get attention model state dict
    attention_state_dict = attention_model.model.state_dict()
    
    # Find matching parameters
    transferred = 0
    
    for name, param in attention_state_dict.items():
        # Check if parameter exists in original model
        if name in original_state_dict and param.shape == original_state_dict[name].shape:
            # Transfer parameter
            attention_state_dict[name] = original_state_dict[name]
            transferred += 1
    
    # Load state dict
    attention_model.model.load_state_dict(attention_state_dict)
    
    logger.info(f"Transferred {transferred} parameters from original model to attention model")

def compare_models(original_mg, enhanced_mg, num_samples, batch_size):
    """Compare original and enhanced models."""
    logger.info("Comparing original and enhanced models...")
    
    # Create output directory
    os.makedirs("comparison", exist_ok=True)
    
    # Generate samples from original model
    with Timer("Original model sampling"):
        original_samples, original_graph = original_mg.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
        )
    
    # Generate samples from enhanced model
    with Timer("Enhanced model sampling"):
        enhanced_samples, enhanced_graph = enhanced_mg.generate_samples(
            num_samples=num_samples,
            batch_size=batch_size,
        )
    
    # Compare samples
    logger.info(f"Original samples shape: {original_samples.shape}")
    logger.info(f"Enhanced samples shape: {enhanced_samples.shape}")
    
    # Plot samples
    plot_sample_comparison(original_samples, enhanced_samples)
    
    # Profile models
    profile_models(original_mg, enhanced_mg, batch_size)

def plot_sample_comparison(original_samples, enhanced_samples):
    """Plot comparison of original and enhanced samples."""
    # Create figure
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
    ax.set_title("Original Model - OHLC")
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
    ax.set_title("Enhanced Model - OHLC")
    ax.legend()
    
    # Plot original volume
    ax = axes[1, 0]
    sample_idx = np.random.randint(0, len(original_samples))
    sample = original_samples[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Original Model - Volume")
    
    # Plot enhanced volume
    ax = axes[1, 1]
    sample_idx = np.random.randint(0, len(enhanced_samples))
    sample = enhanced_samples[sample_idx]
    
    # Extract volume
    volume = sample[4]
    
    # Plot
    ax.plot(volume)
    ax.set_title("Enhanced Model - Volume")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("comparison/sample_comparison.png")
    logger.info("Saved sample comparison to comparison/sample_comparison.png")
    
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
    plt.savefig("comparison/distribution_comparison.png")
    logger.info("Saved distribution comparison to comparison/distribution_comparison.png")

def profile_models(original_mg, enhanced_mg, batch_size):
    """Profile original and enhanced models."""
    logger.info("Profiling models...")
    
    # Create random inputs
    seq_len = config.data.history_length
    x = torch.randn(batch_size, 5, seq_len, device=config.hardware.device)
    
    # Get causal graph
    if original_mg.causal_graphs is not None:
        causal_graph = original_mg.causal_graphs[0].unsqueeze(0).expand(batch_size, -1, -1)
    else:
        causal_graph = torch.randn(batch_size, original_mg.num_latent_factors, original_mg.num_latent_factors, device=config.hardware.device)
    
    # Create timesteps
    timesteps = torch.randint(0, original_mg.diffusion_model.num_timesteps, (batch_size,), device=config.hardware.device)
    
    # Profile original model
    if original_mg.diffusion_model and original_mg.diffusion_model.model:
        with Timer("Original model forward pass"):
            for _ in range(10):
                with torch.no_grad():
                    original_mg.diffusion_model.model(x, timesteps, causal_graph)
        
        # Profile with PyTorch profiler
        trace_filename = "comparison/original_model_profile"
        
        with TorchProfiler.profile_model(
            model=lambda inputs: original_mg.diffusion_model.model(inputs, timesteps, causal_graph),
            inputs=x,
            trace_filename=trace_filename,
        ):
            pass
    
    # Profile enhanced model
    if enhanced_mg.diffusion_model and enhanced_mg.diffusion_model.model:
        with Timer("Enhanced model forward pass"):
            for _ in range(10):
                with torch.no_grad():
                    enhanced_mg.diffusion_model.model(x, timesteps, causal_graph)
        
        # Profile with PyTorch profiler
        trace_filename = "comparison/enhanced_model_profile"
        
        with TorchProfiler.profile_model(
            model=lambda inputs: enhanced_mg.diffusion_model.model(inputs, timesteps, causal_graph),
            inputs=x,
            trace_filename=trace_filename,
        ):
            pass

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Multiverse Generator
    original_mg = load_multiverse_generator(args.model_dir)
    
    # Create enhanced Multiverse Generator
    enhanced_mg = MultiverseGenerator(
        model_dir=args.output_dir,
        num_latent_factors=original_mg.num_latent_factors,
    )
    
    # Copy causal graphs
    enhanced_mg.causal_graphs = original_mg.causal_graphs
    enhanced_mg.causal_graph_learner = original_mg.causal_graph_learner
    
    # Create attention-enhanced diffusion model
    attention_diffusion = create_attention_diffusion_model(
        args=args,
        input_channels=5,  # OHLCV
        num_latent_factors=original_mg.num_latent_factors,
    )
    
    # Transfer weights if possible
    if original_mg.diffusion_model:
        transfer_weights(original_mg.diffusion_model, attention_diffusion)
    
    # Set diffusion model
    enhanced_mg.diffusion_model = attention_diffusion
    
    # Optimize model memory
    MemoryOptimizer.optimize_model_memory(attention_diffusion.model)
    
    # Save enhanced model
    enhanced_mg.save()
    
    # Compare models
    compare_models(original_mg, enhanced_mg, args.num_samples, args.batch_size)
    
    logger.info(f"Enhanced Multiverse Generator saved to {args.output_dir}")

if __name__ == "__main__":
    main()
