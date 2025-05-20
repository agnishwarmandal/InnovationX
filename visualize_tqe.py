"""
Script to visualize the Topo-Quantum Encoder features.
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
from mqtm.data_engineering.data_fetcher import BinanceDataFetcher
from mqtm.data_engineering.data_processor import OHLCVProcessor
from mqtm.topo_quantum_encoder.persistence_homology import PersistentHomologyExtractor
from mqtm.topo_quantum_encoder.wavelet_transform import ComplexWaveletTransform
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize TQE Features")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol to use")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to download")
    parser.add_argument("--download", action="store_true",
                        help="Download fresh data")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    
    return parser.parse_args()

def download_data(symbol, days):
    """Download historical OHLCV data."""
    logger.info(f"Downloading {days} days of data for {symbol}...")
    
    # Create data fetcher
    fetcher = BinanceDataFetcher(symbols=[symbol])
    
    # Download data
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        save=True
    )
    
    return data

def process_data(symbol):
    """Process OHLCV data."""
    logger.info(f"Processing data for {symbol}...")
    
    # Create data processor
    processor = OHLCVProcessor(symbols=[symbol])
    
    # Process data
    X, y = processor.process_symbol(symbol, save=True)
    
    return X, y

def visualize_persistence_diagrams(X, output_dir):
    """Visualize persistence diagrams."""
    logger.info("Visualizing persistence diagrams...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create persistent homology extractor
    ph_extractor = PersistentHomologyExtractor()
    
    # Extract barcodes for a sample
    sample_idx = np.random.randint(0, len(X))
    sample = X[sample_idx:sample_idx+1]
    
    # Reshape to [batch_size, channels, seq_len]
    sample = sample.transpose(0, 2, 1)
    
    # Extract barcodes
    barcodes = ph_extractor.extract_barcodes(sample)[0]
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Plot H0 barcodes (connected components)
    ax = axes[0]
    h0_barcode = barcodes[0]
    
    for i, (birth, death) in enumerate(h0_barcode):
        ax.plot([birth, death], [i, i], 'b-', linewidth=2)
    
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Barcode Index")
    ax.set_title("H0 Persistence Barcode (Connected Components)")
    
    # Plot H1 barcodes (loops)
    ax = axes[1]
    h1_barcode = barcodes[1]
    
    for i, (birth, death) in enumerate(h1_barcode):
        ax.plot([birth, death], [i, i], 'r-', linewidth=2)
    
    ax.set_xlabel("Filtration Value")
    ax.set_ylabel("Barcode Index")
    ax.set_title("H1 Persistence Barcode (Loops)")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "persistence_barcodes.png"))
    logger.info(f"Saved persistence barcodes to {os.path.join(output_dir, 'persistence_barcodes.png')}")

def visualize_wavelet_transform(X, output_dir):
    """Visualize wavelet transform."""
    logger.info("Visualizing wavelet transform...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create complex wavelet transform
    cwt = ComplexWaveletTransform()
    
    # Extract wavelet transform for a sample
    sample_idx = np.random.randint(0, len(X))
    sample = X[sample_idx:sample_idx+1]
    
    # Reshape to [batch_size, channels, seq_len]
    sample = sample.transpose(0, 2, 1)
    
    # Extract wavelet transform
    amplitude, phase = cwt.transform(sample)
    
    # Create figure for amplitude
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("", ["blue", "white", "red"])
    
    # Plot amplitude for close price and morlet wavelet
    ax = axes[0, 0]
    im = ax.imshow(amplitude[0, 3, 0], aspect='auto', cmap=cmap)
    ax.set_title("Close Price - Morlet Wavelet Amplitude")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    plt.colorbar(im, ax=ax)
    
    # Plot amplitude for volume and morlet wavelet
    ax = axes[0, 1]
    im = ax.imshow(amplitude[0, 4, 0], aspect='auto', cmap=cmap)
    ax.set_title("Volume - Morlet Wavelet Amplitude")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    plt.colorbar(im, ax=ax)
    
    # Plot phase for close price and morlet wavelet
    ax = axes[1, 0]
    im = ax.imshow(phase[0, 3, 0], aspect='auto', cmap='hsv')
    ax.set_title("Close Price - Morlet Wavelet Phase")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    plt.colorbar(im, ax=ax)
    
    # Plot phase for volume and morlet wavelet
    ax = axes[1, 1]
    im = ax.imshow(phase[0, 4, 0], aspect='auto', cmap='hsv')
    ax.set_title("Volume - Morlet Wavelet Phase")
    ax.set_xlabel("Time")
    ax.set_ylabel("Scale")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "wavelet_transform.png"))
    logger.info(f"Saved wavelet transform to {os.path.join(output_dir, 'wavelet_transform.png')}")

def visualize_tqe_features(X, output_dir):
    """Visualize TQE features."""
    logger.info("Visualizing TQE features...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create TQE
    tqe = TopoQuantumEncoder()
    
    # Extract features for samples
    # Reshape to [batch_size, channels, seq_len]
    X_reshaped = X.transpose(0, 2, 1)
    
    # Convert to torch tensor
    X_torch = torch.tensor(X_reshaped, dtype=torch.float32)
    
    # Extract features
    tqe.eval()
    with torch.no_grad():
        features = tqe.extract_features(X_torch)
    
    # Convert to numpy
    features = features.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute PCA for visualization
    from sklearn.decomposition import PCA
    
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    
    # Plot features
    scatter = ax.scatter(features_2d[:, 0], features_2d[:, 1], c=np.arange(len(features_2d)), cmap='viridis')
    ax.set_title("TQE Features (PCA)")
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    plt.colorbar(scatter, ax=ax, label="Sample Index")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tqe_features.png"))
    logger.info(f"Saved TQE features to {os.path.join(output_dir, 'tqe_features.png')}")
    
    # Create figure for feature correlation
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Compute correlation matrix
    corr = np.corrcoef(features.T)
    
    # Plot correlation matrix
    im = ax.imshow(corr, cmap='coolwarm')
    ax.set_title("TQE Feature Correlation Matrix")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "tqe_feature_correlation.png"))
    logger.info(f"Saved TQE feature correlation to {os.path.join(output_dir, 'tqe_feature_correlation.png')}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Download data if requested
    if args.download:
        download_data(args.symbol, args.days)
    
    # Process data
    X, y = process_data(args.symbol)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize persistence diagrams
    visualize_persistence_diagrams(X, args.output_dir)
    
    # Visualize wavelet transform
    visualize_wavelet_transform(X, args.output_dir)
    
    # Visualize TQE features
    visualize_tqe_features(X, args.output_dir)
    
    logger.info("Visualization completed successfully!")

if __name__ == "__main__":
    main()
