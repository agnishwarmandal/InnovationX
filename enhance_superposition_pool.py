"""
Script to enhance the Superposition Pool with advanced features.
"""

import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from mqtm.config import config
from mqtm.superposition_pool.superposition_model import SuperpositionPool
from mqtm.superposition_pool.complex_layer import ComplexLinear
from mqtm.superposition_pool.advanced_unitary import AdvancedUnitaryUpdate, ComplexEntanglement
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
    parser = argparse.ArgumentParser(description="Enhance Superposition Pool")
    
    parser.add_argument("--model_path", type=str, default="models/sp3.pt",
                        help="Path to trained Superposition Pool")
    parser.add_argument("--output_path", type=str, default="models/enhanced_sp3.pt",
                        help="Path to save enhanced Superposition Pool")
    parser.add_argument("--use_advanced_unitary", action="store_true",
                        help="Whether to use advanced unitary updates")
    parser.add_argument("--use_entanglement", action="store_true",
                        help="Whether to use complex entanglement")
    parser.add_argument("--unitary_method", type=str, default="cayley",
                        choices=["cayley", "stiefel", "geodesic", "phase"],
                        help="Unitary update method")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--input_dim", type=int, default=112,
                        help="Input dimension")
    
    return parser.parse_args()

def load_superposition_pool(model_path, input_dim):
    """Load Superposition Pool."""
    logger.info(f"Loading Superposition Pool from {model_path}...")
    
    # Create Superposition Pool
    sp3 = SuperpositionPool(
        input_dim=input_dim,
    )
    
    # Load model if exists
    if os.path.exists(model_path):
        sp3.load(model_path)
    
    return sp3

def enhance_complex_layers(sp3, unitary_method):
    """Enhance complex layers with advanced unitary updates."""
    logger.info(f"Enhancing complex layers with {unitary_method} unitary updates...")
    
    # Update unitary method for all heads
    for i, head in enumerate(sp3.heads):
        # Update complex layers
        for name, module in head.named_modules():
            if isinstance(module, ComplexLinear):
                # Set unitary update method
                module.unitary_method = unitary_method
                
                logger.info(f"Updated head {i}, layer {name} to use {unitary_method} unitary updates")
    
    # Update apply_unitary_update method
    original_apply_unitary_update = sp3.apply_unitary_update
    
    def enhanced_apply_unitary_update(learning_rate=0.01, temperature=0.0):
        """Enhanced unitary update method."""
        for head in sp3.heads:
            for name, module in head.named_modules():
                if isinstance(module, ComplexLinear):
                    # Get parameters
                    weight_real = module.weight_real
                    weight_imag = module.weight_imag
                    
                    # Get gradients
                    if weight_real.grad is not None and weight_imag.grad is not None:
                        grad_real = weight_real.grad
                        grad_imag = weight_imag.grad
                        
                        # Apply unitary update
                        updated_weight_real, updated_weight_imag = AdvancedUnitaryUpdate.apply_update(
                            weight_real, weight_imag, grad_real, grad_imag,
                            learning_rate=learning_rate,
                            method=module.unitary_method,
                            temperature=temperature,
                        )
                        
                        # Update parameters
                        with torch.no_grad():
                            weight_real.copy_(updated_weight_real)
                            weight_imag.copy_(updated_weight_imag)
    
    # Replace method
    sp3.apply_unitary_update = enhanced_apply_unitary_update
    
    logger.info("Complex layers enhanced with advanced unitary updates")

def add_entanglement(sp3):
    """Add complex entanglement to Superposition Pool."""
    logger.info("Adding complex entanglement...")
    
    # Add entanglement layer to each head
    for i, head in enumerate(sp3.heads):
        # Create entanglement layer
        entanglement = ComplexEntanglement(
            num_qubits=head.hidden_dim,
            device=config.hardware.device,
        )
        
        # Add entanglement layer
        head.entanglement = entanglement
        
        # Update forward method
        original_forward = head.forward
        
        def enhanced_forward(self, x, regime):
            """Enhanced forward method with entanglement."""
            # Apply original forward pass
            h_real, h_imag = self.complex_layer1(x)
            
            # Apply activation
            h_real = torch.tanh(h_real)
            h_imag = torch.tanh(h_imag)
            
            # Apply entanglement
            h_real, h_imag = self.entanglement(h_real, h_imag)
            
            # Apply projection gate
            gate = self.projection_gate(regime)
            h_real = h_real * gate.unsqueeze(1)
            h_imag = h_imag * gate.unsqueeze(1)
            
            # Apply second layer
            output_real, output_imag = self.complex_layer2(h_real, h_imag)
            
            # Compute magnitude
            output = torch.sqrt(output_real**2 + output_imag**2)
            
            return output
        
        # Replace method
        head.__class__.forward = enhanced_forward
        
        logger.info(f"Added entanglement to head {i}")
    
    logger.info("Complex entanglement added to Superposition Pool")

def test_superposition_pool(sp3, batch_size, input_dim):
    """Test Superposition Pool with random inputs."""
    logger.info(f"Testing Superposition Pool with batch size {batch_size}...")
    
    # Create random inputs
    x = torch.randn(batch_size, input_dim, device=config.hardware.device)
    regime = torch.rand(batch_size, 2, device=config.hardware.device)
    
    # Forward pass
    with Timer("Superposition Pool forward pass"):
        with torch.no_grad():
            outputs = sp3(x, regime)
    
    logger.info(f"Outputs shape: {outputs.shape}")
    
    # Test individual heads
    for i, head in enumerate(sp3.heads):
        with Timer(f"Head {i} forward pass"):
            with torch.no_grad():
                head_output = head(x, regime)
        
        logger.info(f"Head {i} output shape: {head_output.shape}")
    
    # Profile with PyTorch profiler
    trace_filename = "profiles/sp3_forward_profile"
    
    with TorchProfiler.profile_model(
        model=lambda inputs: sp3(inputs, regime),
        inputs=x,
        trace_filename=trace_filename,
    ):
        pass
    
    # Test unitary update
    with Timer("Unitary update"):
        # Create random gradients
        for head in sp3.heads:
            for name, module in head.named_modules():
                if isinstance(module, ComplexLinear):
                    # Get parameters
                    weight_real = module.weight_real
                    weight_imag = module.weight_imag
                    
                    # Create random gradients
                    weight_real.grad = torch.randn_like(weight_real) * 0.01
                    weight_imag.grad = torch.randn_like(weight_imag) * 0.01
        
        # Apply unitary update
        sp3.apply_unitary_update(learning_rate=0.01, temperature=0.1)
    
    return outputs

def visualize_superposition(original_outputs, enhanced_outputs):
    """Visualize original and enhanced superposition."""
    logger.info("Visualizing superposition...")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Convert to numpy
    original_outputs = original_outputs.detach().cpu().numpy()
    enhanced_outputs = enhanced_outputs.detach().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original outputs
    ax = axes[0]
    im = ax.imshow(original_outputs, aspect='auto', cmap='viridis')
    ax.set_title("Original Outputs")
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced outputs
    ax = axes[1]
    im = ax.imshow(enhanced_outputs, aspect='auto', cmap='viridis')
    ax.set_title("Enhanced Outputs")
    ax.set_xlabel("Output Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/superposition_comparison.png")
    logger.info("Saved superposition comparison to visualizations/superposition_comparison.png")
    
    # Create figure for output distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original distribution
    ax = axes[0]
    ax.hist(original_outputs.flatten(), bins=50)
    ax.set_title("Original Output Distribution")
    ax.set_xlabel("Output Value")
    ax.set_ylabel("Count")
    
    # Plot enhanced distribution
    ax = axes[1]
    ax.hist(enhanced_outputs.flatten(), bins=50)
    ax.set_title("Enhanced Output Distribution")
    ax.set_xlabel("Output Value")
    ax.set_ylabel("Count")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/superposition_distribution.png")
    logger.info("Saved superposition distribution to visualizations/superposition_distribution.png")
    
    # Create figure for head weights
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original head weights
    ax = axes[0]
    original_weights = torch.nn.functional.softmax(original_sp3.head_weights, dim=0).detach().cpu().numpy()
    ax.bar(range(len(original_weights)), original_weights)
    ax.set_title("Original Head Weights")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Weight")
    
    # Plot enhanced head weights
    ax = axes[1]
    enhanced_weights = torch.nn.functional.softmax(enhanced_sp3.head_weights, dim=0).detach().cpu().numpy()
    ax.bar(range(len(enhanced_weights)), enhanced_weights)
    ax.set_title("Enhanced Head Weights")
    ax.set_xlabel("Head Index")
    ax.set_ylabel("Weight")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/head_weights_comparison.png")
    logger.info("Saved head weights comparison to visualizations/head_weights_comparison.png")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load Superposition Pool
    sp3 = load_superposition_pool(args.model_path, args.input_dim)
    
    # Create copy of original model
    global original_sp3
    original_sp3 = SuperpositionPool(
        input_dim=args.input_dim,
    )
    if os.path.exists(args.model_path):
        original_sp3.load(args.model_path)
    
    # Move to device
    sp3.to(config.hardware.device)
    original_sp3.to(config.hardware.device)
    
    # Test original model
    logger.info("Testing original model...")
    original_outputs = test_superposition_pool(original_sp3, args.batch_size, args.input_dim)
    
    # Enhance model
    if args.use_advanced_unitary:
        enhance_complex_layers(sp3, args.unitary_method)
    
    if args.use_entanglement:
        add_entanglement(sp3)
    
    # Optimize model memory
    MemoryOptimizer.optimize_model_memory(sp3)
    
    # Test enhanced model
    logger.info("Testing enhanced model...")
    enhanced_outputs = test_superposition_pool(sp3, args.batch_size, args.input_dim)
    
    # Visualize superposition
    visualize_superposition(original_outputs, enhanced_outputs)
    
    # Save enhanced model
    sp3.save(args.output_path)
    
    logger.info(f"Enhanced Superposition Pool saved to {args.output_path}")

if __name__ == "__main__":
    main()
