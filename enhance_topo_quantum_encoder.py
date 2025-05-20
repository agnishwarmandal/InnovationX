"""
Script to enhance the Topo-Quantum Encoder with advanced features.
"""

import os
import argparse
import logging
import torch
import numpy as np
import matplotlib.pyplot as plt

from mqtm.config import config
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.topo_quantum_encoder.advanced_persistence import AdvancedPersistenceLayer
from mqtm.topo_quantum_encoder.advanced_wavelets import AdvancedWaveletLayer, AdaptiveWaveletSelection
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
    parser = argparse.ArgumentParser(description="Enhance Topo-Quantum Encoder")
    
    parser.add_argument("--model_path", type=str, default="models/tqe.pt",
                        help="Path to trained Topo-Quantum Encoder")
    parser.add_argument("--output_path", type=str, default="models/enhanced_tqe.pt",
                        help="Path to save enhanced Topo-Quantum Encoder")
    parser.add_argument("--use_advanced_persistence", action="store_true",
                        help="Whether to use advanced persistence features")
    parser.add_argument("--use_advanced_wavelets", action="store_true",
                        help="Whether to use advanced wavelet features")
    parser.add_argument("--use_adaptive_wavelets", action="store_true",
                        help="Whether to use adaptive wavelet selection")
    parser.add_argument("--max_homology_dim", type=int, default=2,
                        help="Maximum homology dimension for persistence")
    parser.add_argument("--wavelet_families", type=str, nargs="+", 
                        default=["morl", "mexh", "gaus1", "cgau1"],
                        help="Wavelet families to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    
    return parser.parse_args()

def load_topo_quantum_encoder(model_path):
    """Load Topo-Quantum Encoder."""
    logger.info(f"Loading Topo-Quantum Encoder from {model_path}...")
    
    # Create Topo-Quantum Encoder
    tqe = TopoQuantumEncoder()
    
    # Load model if exists
    if os.path.exists(model_path):
        tqe.load(model_path)
    
    return tqe

def enhance_persistence_layer(tqe, max_homology_dim):
    """Enhance persistence layer with advanced features."""
    logger.info("Enhancing persistence layer...")
    
    # Create advanced persistence layer
    advanced_persistence = AdvancedPersistenceLayer(
        input_dim=tqe.input_channels * tqe.history_length,
        output_dim=tqe.persistence_dim,
        max_homology_dim=max_homology_dim,
        use_alpha_complex=True,
        use_vietoris_rips=True,
        use_witness_complex=False,
        device=config.hardware.device,
    )
    
    # Replace persistence layer
    tqe.persistence_layer = advanced_persistence
    
    logger.info("Persistence layer enhanced with advanced features")

def enhance_wavelet_layer(tqe, wavelet_families, use_adaptive):
    """Enhance wavelet layer with advanced features."""
    logger.info("Enhancing wavelet layer...")
    
    if use_adaptive:
        # Create adaptive wavelet selection
        advanced_wavelets = AdaptiveWaveletSelection(
            input_channels=tqe.input_channels,
            output_dim=tqe.wavelet_dim,
            seq_len=tqe.history_length,
            wavelet_families=wavelet_families,
            scales=[2, 4, 8, 16, 32],
            device=config.hardware.device,
        )
    else:
        # Create advanced wavelet layer
        advanced_wavelets = AdvancedWaveletLayer(
            input_channels=tqe.input_channels,
            output_dim=tqe.wavelet_dim,
            seq_len=tqe.history_length,
            wavelet_families=wavelet_families,
            scales=[2, 4, 8, 16, 32],
            use_cwt=True,
            use_dwt=True,
            use_wpt=False,
            device=config.hardware.device,
        )
    
    # Replace wavelet layer
    tqe.wavelet_layer = advanced_wavelets
    
    logger.info("Wavelet layer enhanced with advanced features")

def test_encoder(tqe, batch_size):
    """Test encoder with random inputs."""
    logger.info(f"Testing encoder with batch size {batch_size}...")
    
    # Create random inputs
    X = torch.randn(batch_size, tqe.input_channels, tqe.history_length, device=config.hardware.device)
    
    # Forward pass
    with Timer("Encoder forward pass"):
        with torch.no_grad():
            features = tqe(X)
    
    logger.info(f"Features shape: {features.shape}")
    
    # Test individual components
    with Timer("Persistence layer"):
        with torch.no_grad():
            persistence_features = tqe.persistence_layer(X)
    
    logger.info(f"Persistence features shape: {persistence_features.shape}")
    
    with Timer("Wavelet layer"):
        with torch.no_grad():
            wavelet_features = tqe.wavelet_layer(X)
    
    logger.info(f"Wavelet features shape: {wavelet_features.shape}")
    
    if tqe.use_classic_indicators:
        with Timer("Indicators layer"):
            with torch.no_grad():
                indicators_features = tqe.indicators_layer(X)
                indicators_features = torch.nn.functional.adaptive_avg_pool1d(
                    indicators_features, 1
                ).squeeze(-1)
        
        logger.info(f"Indicators features shape: {indicators_features.shape}")
    
    # Profile with PyTorch profiler
    trace_filename = "profiles/tqe_forward_profile"
    
    with TorchProfiler.profile_model(
        model=tqe,
        inputs=X,
        trace_filename=trace_filename,
    ):
        pass
    
    return features

def visualize_features(original_features, enhanced_features):
    """Visualize original and enhanced features."""
    logger.info("Visualizing features...")
    
    # Create output directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Convert to numpy
    original_features = original_features.detach().cpu().numpy()
    enhanced_features = enhanced_features.detach().cpu().numpy()
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original features
    ax = axes[0]
    im = ax.imshow(original_features, aspect='auto', cmap='viridis')
    ax.set_title("Original Features")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced features
    ax = axes[1]
    im = ax.imshow(enhanced_features, aspect='auto', cmap='viridis')
    ax.set_title("Enhanced Features")
    ax.set_xlabel("Feature Dimension")
    ax.set_ylabel("Sample")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/feature_comparison.png")
    logger.info("Saved feature comparison to visualizations/feature_comparison.png")
    
    # Create figure for feature correlation
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Compute correlation matrices
    original_corr = np.corrcoef(original_features.T)
    enhanced_corr = np.corrcoef(enhanced_features.T)
    
    # Plot original correlation
    ax = axes[0]
    im = ax.imshow(original_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title("Original Feature Correlation")
    plt.colorbar(im, ax=ax)
    
    # Plot enhanced correlation
    ax = axes[1]
    im = ax.imshow(enhanced_corr, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_title("Enhanced Feature Correlation")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/correlation_comparison.png")
    logger.info("Saved correlation comparison to visualizations/correlation_comparison.png")
    
    # Create figure for feature distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot original distribution
    ax = axes[0]
    ax.hist(original_features.flatten(), bins=50)
    ax.set_title("Original Feature Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Count")
    
    # Plot enhanced distribution
    ax = axes[1]
    ax.hist(enhanced_features.flatten(), bins=50)
    ax.set_title("Enhanced Feature Distribution")
    ax.set_xlabel("Feature Value")
    ax.set_ylabel("Count")
    
    # Save figure
    plt.tight_layout()
    plt.savefig("visualizations/distribution_comparison.png")
    logger.info("Saved distribution comparison to visualizations/distribution_comparison.png")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Load Topo-Quantum Encoder
    tqe = load_topo_quantum_encoder(args.model_path)
    
    # Create copy of original encoder
    original_tqe = TopoQuantumEncoder()
    if os.path.exists(args.model_path):
        original_tqe.load(args.model_path)
    
    # Move to device
    tqe.to(config.hardware.device)
    original_tqe.to(config.hardware.device)
    
    # Test original encoder
    logger.info("Testing original encoder...")
    original_features = test_encoder(original_tqe, args.batch_size)
    
    # Enhance encoder
    if args.use_advanced_persistence:
        enhance_persistence_layer(tqe, args.max_homology_dim)
    
    if args.use_advanced_wavelets or args.use_adaptive_wavelets:
        enhance_wavelet_layer(tqe, args.wavelet_families, args.use_adaptive_wavelets)
    
    # Optimize model memory
    MemoryOptimizer.optimize_model_memory(tqe)
    
    # Test enhanced encoder
    logger.info("Testing enhanced encoder...")
    enhanced_features = test_encoder(tqe, args.batch_size)
    
    # Visualize features
    visualize_features(original_features, enhanced_features)
    
    # Save enhanced encoder
    tqe.save(args.output_path)
    
    logger.info(f"Enhanced Topo-Quantum Encoder saved to {args.output_path}")

if __name__ == "__main__":
    main()
