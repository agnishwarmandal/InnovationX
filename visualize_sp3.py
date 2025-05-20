"""
Script to visualize the Superposition Parameter Pool.
"""

import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from datetime import datetime, timedelta

from mqtm.config import config
from mqtm.superposition_pool.superposition_model import SuperpositionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize SP3")
    
    parser.add_argument("--input_dim", type=int, default=112,
                        help="Input dimension for SP3")
    parser.add_argument("--num_heads", type=int, default=6,
                        help="Number of superposition heads")
    parser.add_argument("--hidden_dim", type=int, default=64,
                        help="Hidden dimension for superposition heads")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to load trained SP3 model")
    
    return parser.parse_args()

def visualize_complex_weights(sp3, output_dir):
    """Visualize complex weights."""
    logger.info("Visualizing complex weights...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    
    # Plot weights for each head
    for i, head in enumerate(sp3.heads[:min(6, len(sp3.heads))]):
        # Get complex parameters
        params = head.get_complex_parameters()
        
        # Get first layer weights
        weight_real, weight_imag = params[0]
        
        # Convert to numpy
        weight_real = weight_real.detach().cpu().numpy()
        weight_imag = weight_imag.detach().cpu().numpy()
        
        # Compute magnitude and phase
        magnitude = np.sqrt(weight_real ** 2 + weight_imag ** 2)
        phase = np.arctan2(weight_imag, weight_real)
        
        # Plot magnitude
        ax = axes[i]
        im = ax.imshow(magnitude, cmap=cmap)
        ax.set_title(f"Head {i} - Weight Magnitude")
        plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complex_weights_magnitude.png"))
    logger.info(f"Saved complex weights magnitude to {os.path.join(output_dir, 'complex_weights_magnitude.png')}")
    
    # Create figure for phase
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot phase for each head
    for i, head in enumerate(sp3.heads[:min(6, len(sp3.heads))]):
        # Get complex parameters
        params = head.get_complex_parameters()
        
        # Get first layer weights
        weight_real, weight_imag = params[0]
        
        # Convert to numpy
        weight_real = weight_real.detach().cpu().numpy()
        weight_imag = weight_imag.detach().cpu().numpy()
        
        # Compute phase
        phase = np.arctan2(weight_imag, weight_real)
        
        # Plot phase
        ax = axes[i]
        im = ax.imshow(phase, cmap='hsv')
        ax.set_title(f"Head {i} - Weight Phase")
        plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "complex_weights_phase.png"))
    logger.info(f"Saved complex weights phase to {os.path.join(output_dir, 'complex_weights_phase.png')}")

def visualize_projection_gate(sp3, output_dir):
    """Visualize projection gate."""
    logger.info("Visualizing projection gate...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create random regime features
    num_samples = 1000
    regime = torch.rand(num_samples, 2)
    
    # Compute projection weights for each head
    projections = []
    
    for head in sp3.heads:
        with torch.no_grad():
            projection = head.projection_gate(regime)
            projections.append(projection.cpu().numpy())
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormap
    cmap = plt.cm.get_cmap('viridis', len(sp3.heads))
    
    # Plot projection weights
    for i, projection in enumerate(projections):
        scatter = ax.scatter(regime[:, 0].numpy(), regime[:, 1].numpy(), c=projection, cmap='plasma', alpha=0.5, label=f"Head {i}")
    
    ax.set_xlabel("Volatility")
    ax.set_ylabel("Funding")
    ax.set_title("Projection Gate Weights")
    plt.colorbar(scatter, ax=ax, label="Projection Weight")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "projection_gate.png"))
    logger.info(f"Saved projection gate visualization to {os.path.join(output_dir, 'projection_gate.png')}")
    
    # Create figure for projection distribution
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Plot projection distribution for each head
    for i, projection in enumerate(projections[:min(6, len(projections))]):
        ax = axes[i]
        ax.hist(projection, bins=50, alpha=0.7)
        ax.set_title(f"Head {i} - Projection Distribution")
        ax.set_xlabel("Projection Weight")
        ax.set_ylabel("Count")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "projection_distribution.png"))
    logger.info(f"Saved projection distribution to {os.path.join(output_dir, 'projection_distribution.png')}")

def visualize_head_weights(sp3, output_dir):
    """Visualize head weights."""
    logger.info("Visualizing head weights...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get head weights
    weights = F.softmax(sp3.head_weights, dim=0).detach().cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot head weights
    ax.bar(range(len(weights)), weights)
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Weight")
    ax.set_title("Superposition Head Weights")
    
    # Add entropy value
    entropy = sp3.compute_entropy()
    ax.text(0.02, 0.95, f"Entropy: {entropy:.4f}", transform=ax.transAxes, fontsize=12,
            bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "head_weights.png"))
    logger.info(f"Saved head weights to {os.path.join(output_dir, 'head_weights.png')}")

def visualize_unitary_update(sp3, output_dir):
    """Visualize unitary update."""
    logger.info("Visualizing unitary update...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get complex parameters for first head
    head = sp3.heads[0]
    params = head.get_complex_parameters()
    
    # Get first layer weights
    weight_real, weight_imag = params[0]
    
    # Convert to numpy
    weight_real = weight_real.detach().cpu().numpy()
    weight_imag = weight_imag.detach().cpu().numpy()
    
    # Create random gradients
    grad_real = np.random.randn(*weight_real.shape) * 0.01
    grad_imag = np.random.randn(*weight_imag.shape) * 0.01
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot original weights
    ax = axes[0, 0]
    ax.scatter(weight_real.flatten(), weight_imag.flatten(), alpha=0.5)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Original Weights")
    
    # Add unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'r-')
    ax.set_aspect('equal')
    
    # Plot gradients
    ax = axes[0, 1]
    ax.scatter(grad_real.flatten(), grad_imag.flatten(), alpha=0.5)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Gradients")
    ax.set_aspect('equal')
    
    # Apply unitary update
    from mqtm.superposition_pool.complex_layer import UnitaryUpdate
    
    # Convert to torch tensors
    weight_real_tensor = torch.tensor(weight_real)
    weight_imag_tensor = torch.tensor(weight_imag)
    grad_real_tensor = torch.tensor(grad_real)
    grad_imag_tensor = torch.tensor(grad_imag)
    
    # Apply update
    update_real, update_imag = UnitaryUpdate.apply_update(
        weight_real_tensor, weight_imag_tensor, grad_real_tensor, grad_imag_tensor, learning_rate=0.1
    )
    
    # Convert to numpy
    update_real = update_real.numpy()
    update_imag = update_imag.numpy()
    
    # Plot updated weights
    ax = axes[1, 0]
    ax.scatter(update_real.flatten(), update_imag.flatten(), alpha=0.5)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.set_title("Updated Weights")
    
    # Add unit circle
    ax.plot(np.cos(theta), np.sin(theta), 'r-')
    ax.set_aspect('equal')
    
    # Plot weight changes
    ax = axes[1, 1]
    ax.scatter(
        (update_real - weight_real).flatten(),
        (update_imag - weight_imag).flatten(),
        alpha=0.5
    )
    ax.set_xlabel("Real Part Change")
    ax.set_ylabel("Imaginary Part Change")
    ax.set_title("Weight Changes")
    ax.set_aspect('equal')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "unitary_update.png"))
    logger.info(f"Saved unitary update visualization to {os.path.join(output_dir, 'unitary_update.png')}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create SP3
    sp3 = SuperpositionPool(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
    )
    
    # Load model if provided
    if args.model_path and os.path.exists(args.model_path):
        sp3.load(args.model_path)
        logger.info(f"Loaded SP3 from {args.model_path}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize complex weights
    visualize_complex_weights(sp3, args.output_dir)
    
    # Visualize projection gate
    visualize_projection_gate(sp3, args.output_dir)
    
    # Visualize head weights
    visualize_head_weights(sp3, args.output_dir)
    
    # Visualize unitary update
    visualize_unitary_update(sp3, args.output_dir)
    
    logger.info("Visualization completed successfully!")

if __name__ == "__main__":
    main()
