"""
Example script demonstrating how to use the MQTM system.
"""

import os
import argparse
import logging
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

from mqtm.config import config
from mqtm.data_engineering.data_fetcher import BinanceDataFetcher
from mqtm.data_engineering.data_processor import OHLCVProcessor
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
    parser = argparse.ArgumentParser(description="MQTM Example")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol to use")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to download")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--download", action="store_true",
                        help="Download fresh data")
    
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

def load_models(models_dir):
    """Load trained MQTM models."""
    logger.info(f"Loading models from {models_dir}...")
    
    # Create models
    mg = MultiverseGenerator(
        model_dir=os.path.join(models_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load models if available
    tqe_path = os.path.join(models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
        logger.info(f"Loaded TQE from {tqe_path}")
    else:
        logger.warning(f"TQE model not found at {tqe_path}")
    
    sp3_path = os.path.join(models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
        logger.info(f"Loaded SP3 from {sp3_path}")
    else:
        logger.warning(f"SP3 model not found at {sp3_path}")
    
    # Load causal graphs if available
    causal_graphs_path = os.path.join(models_dir, "multiverse_generator", "causal_graphs.pt")
    if os.path.exists(causal_graphs_path):
        mg.causal_graphs = torch.load(causal_graphs_path, map_location=config.hardware.device)
        logger.info(f"Loaded causal graphs from {causal_graphs_path}")
    
    return mg, tqe, sp3

def generate_synthetic_data(mg, num_samples=100):
    """Generate synthetic OHLCV data."""
    logger.info(f"Generating {num_samples} synthetic samples...")
    
    # Check if causal graphs are available
    if mg.causal_graphs is None:
        logger.warning("No causal graphs available. Learning from scratch...")
        mg.learn_causal_graphs(symbols=["BTCUSDT", "ETHUSDT"])
    
    # Generate samples
    X, causal_graph = mg.generate_samples(
        num_samples=num_samples,
        sigma_multiplier=1.5,
    )
    
    return X, causal_graph

def visualize_data(real_data, synthetic_data):
    """Visualize real and synthetic data."""
    logger.info("Visualizing data...")
    
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
    
    # Show plot
    plt.tight_layout()
    plt.savefig("data_visualization.png")
    logger.info("Saved visualization to data_visualization.png")

def predict(tqe, sp3, X):
    """Make predictions using the MQTM models."""
    logger.info(f"Making predictions for {len(X)} samples...")
    
    # Convert to torch tensor
    X_torch = torch.tensor(X, dtype=torch.float32, device=config.hardware.device)
    
    # Create random regime features
    # In practice, these would be computed from market data
    regime = torch.rand(len(X), 2, device=config.hardware.device)
    
    # Extract features with TQE
    tqe.eval()
    with torch.no_grad():
        features = tqe(X_torch)
    
    # Make predictions with SP3
    sp3.eval()
    with torch.no_grad():
        outputs = sp3(features, regime)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Convert to numpy
    probabilities = probabilities.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Map predictions to labels
    labels = ["Down", "Flat", "Up"]
    prediction_labels = [labels[p] for p in predictions]
    
    return predictions, probabilities, prediction_labels

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Download data if requested
    if args.download:
        download_data(args.symbol, args.days)
    
    # Process data
    X, y = process_data(args.symbol)
    
    # Load models
    mg, tqe, sp3 = load_models(args.models_dir)
    
    # Generate synthetic data
    synthetic_X, _ = generate_synthetic_data(mg, num_samples=100)
    
    # Visualize data
    visualize_data(X, synthetic_X)
    
    # Make predictions
    predictions, probabilities, prediction_labels = predict(tqe, sp3, X)
    
    # Print sample predictions
    print("\nSample predictions:")
    for i in range(min(10, len(predictions))):
        print(f"Sample {i}: {prediction_labels[i]} (Probabilities: Down={probabilities[i, 0]:.2f}, Flat={probabilities[i, 1]:.2f}, Up={probabilities[i, 2]:.2f})")
    
    # Compute accuracy if labels are available
    if len(y) > 0:
        accuracy = np.mean(predictions == y) * 100
        print(f"\nAccuracy: {accuracy:.2f}%")
    
    print("\nExample completed successfully!")

if __name__ == "__main__":
    main()
