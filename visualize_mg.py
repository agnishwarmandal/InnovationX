"""
Script to visualize the Multiverse Generator.
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
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.multiverse_generator.causal_graph import CausalGraphLearner

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Visualize Multiverse Generator")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol to use")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to download")
    parser.add_argument("--download", action="store_true",
                        help="Download fresh data")
    parser.add_argument("--output_dir", type=str, default="visualizations",
                        help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--model_dir", type=str, default="models/multiverse_generator",
                        help="Directory with trained MG model")
    
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

def visualize_causal_graph(mg, output_dir):
    """Visualize causal graph."""
    logger.info("Visualizing causal graph...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if causal graphs are available
    if mg.causal_graphs is None:
        logger.warning("No causal graphs available. Learning from scratch...")
        mg.learn_causal_graphs(symbols=["BTCUSDT", "ETHUSDT"])
    
    # Get causal graph
    causal_graph = mg.causal_graphs[0].cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("", ["white", "blue", "red"])
    
    # Plot causal graph
    im = ax.imshow(causal_graph, cmap=cmap)
    ax.set_title("Causal Graph Adjacency Matrix")
    ax.set_xlabel("To Node")
    ax.set_ylabel("From Node")
    plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "causal_graph.png"))
    logger.info(f"Saved causal graph to {os.path.join(output_dir, 'causal_graph.png')}")
    
    # Create figure for graph visualization
    try:
        import networkx as nx
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        for i in range(causal_graph.shape[0]):
            G.add_node(i)
        
        # Add edges
        for i in range(causal_graph.shape[0]):
            for j in range(causal_graph.shape[1]):
                if causal_graph[i, j] > 0.1:
                    G.add_edge(i, j, weight=causal_graph[i, j])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Plot graph
        pos = nx.spring_layout(G, seed=42)
        edges = G.edges()
        weights = [G[u][v]['weight'] * 3 for u, v in edges]
        
        nx.draw_networkx_nodes(G, pos, node_size=500, node_color='lightblue', ax=ax)
        nx.draw_networkx_labels(G, pos, ax=ax)
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='gray', ax=ax)
        
        ax.set_title("Causal Graph Visualization")
        ax.axis('off')
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "causal_graph_network.png"))
        logger.info(f"Saved causal graph network to {os.path.join(output_dir, 'causal_graph_network.png')}")
    
    except ImportError:
        logger.warning("NetworkX not found. Skipping graph visualization.")

def visualize_perturbed_graphs(mg, output_dir):
    """Visualize perturbed causal graphs."""
    logger.info("Visualizing perturbed causal graphs...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if causal graphs are available
    if mg.causal_graphs is None:
        logger.warning("No causal graphs available. Learning from scratch...")
        mg.learn_causal_graphs(symbols=["BTCUSDT", "ETHUSDT"])
    
    # Get causal graph
    causal_graph = mg.causal_graphs[0].cpu().numpy()
    
    # Create causal graph learner
    graph_learner = CausalGraphLearner(n_features=causal_graph.shape[0])
    
    # Create perturbed graphs with different strengths
    perturbation_strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
    perturbed_graphs = []
    
    for strength in perturbation_strengths:
        perturbed_graph = graph_learner.perturb_graph(
            causal_graph,
            perturbation_strength=strength,
        )
        perturbed_graphs.append(perturbed_graph)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    # Create colormap
    cmap = LinearSegmentedColormap.from_list("", ["white", "blue", "red"])
    
    # Plot original graph
    ax = axes[0]
    im = ax.imshow(causal_graph, cmap=cmap)
    ax.set_title("Original Causal Graph")
    plt.colorbar(im, ax=ax)
    
    # Plot perturbed graphs
    for i, (strength, graph) in enumerate(zip(perturbation_strengths, perturbed_graphs)):
        ax = axes[i + 1]
        im = ax.imshow(graph, cmap=cmap)
        ax.set_title(f"Perturbed Graph (strength={strength})")
        plt.colorbar(im, ax=ax)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "perturbed_graphs.png"))
    logger.info(f"Saved perturbed graphs to {os.path.join(output_dir, 'perturbed_graphs.png')}")

def visualize_synthetic_data(real_data, synthetic_data, output_dir):
    """Visualize synthetic data."""
    logger.info("Visualizing synthetic data...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
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
    plt.savefig(os.path.join(output_dir, "synthetic_data.png"))
    logger.info(f"Saved synthetic data visualization to {os.path.join(output_dir, 'synthetic_data.png')}")
    
    # Create figure for distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot close price distribution
    ax = axes[0, 0]
    real_close = np.concatenate([sample[3] for sample in real_data])
    synthetic_close = np.concatenate([sample[3] for sample in synthetic_data])
    
    ax.hist(real_close, bins=50, alpha=0.5, label="Real")
    ax.hist(synthetic_close, bins=50, alpha=0.5, label="Synthetic")
    ax.set_title("Close Price Distribution")
    ax.legend()
    
    # Plot volume distribution
    ax = axes[0, 1]
    real_volume = np.concatenate([sample[4] for sample in real_data])
    synthetic_volume = np.concatenate([sample[4] for sample in synthetic_data])
    
    ax.hist(real_volume, bins=50, alpha=0.5, label="Real")
    ax.hist(synthetic_volume, bins=50, alpha=0.5, label="Synthetic")
    ax.set_title("Volume Distribution")
    ax.legend()
    
    # Plot returns distribution
    ax = axes[1, 0]
    real_returns = np.diff(np.concatenate([sample[3] for sample in real_data]))
    synthetic_returns = np.diff(np.concatenate([sample[3] for sample in synthetic_data]))
    
    ax.hist(real_returns, bins=50, alpha=0.5, label="Real")
    ax.hist(synthetic_returns, bins=50, alpha=0.5, label="Synthetic")
    ax.set_title("Returns Distribution")
    ax.legend()
    
    # Plot volatility distribution
    ax = axes[1, 1]
    real_volatility = np.array([np.std(sample[3]) for sample in real_data])
    synthetic_volatility = np.array([np.std(sample[3]) for sample in synthetic_data])
    
    ax.hist(real_volatility, bins=50, alpha=0.5, label="Real")
    ax.hist(synthetic_volatility, bins=50, alpha=0.5, label="Synthetic")
    ax.set_title("Volatility Distribution")
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "distribution_comparison.png"))
    logger.info(f"Saved distribution comparison to {os.path.join(output_dir, 'distribution_comparison.png')}")

def visualize_curriculum(mg, output_dir):
    """Visualize curriculum samples."""
    logger.info("Visualizing curriculum samples...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate samples with different curriculum stages
    curriculum_stages = [0, 3, 6, 9]
    curriculum_samples = []
    
    for stage in curriculum_stages:
        X, _ = mg.generate_with_curriculum(
            num_samples=10,
            curriculum_stage=stage,
        )
        curriculum_samples.append(X)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    # Plot samples for each curriculum stage
    for i, (stage, samples) in enumerate(zip(curriculum_stages, curriculum_samples)):
        ax = axes[i]
        
        # Plot close prices for multiple samples
        for j in range(min(5, len(samples))):
            ax.plot(samples[j, 3], alpha=0.7)
        
        ax.set_title(f"Curriculum Stage {stage}")
        ax.set_xlabel("Time")
        ax.set_ylabel("Close Price")
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_samples.png"))
    logger.info(f"Saved curriculum samples to {os.path.join(output_dir, 'curriculum_samples.png')}")
    
    # Create figure for tail comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Compute returns for each curriculum stage
    all_returns = []
    
    for samples in curriculum_samples:
        # Compute returns for all samples
        returns = []
        for sample in samples:
            sample_returns = np.diff(sample[3])
            returns.extend(sample_returns)
        
        all_returns.append(returns)
    
    # Plot returns distribution for each curriculum stage
    for i, (stage, returns) in enumerate(zip(curriculum_stages, all_returns)):
        ax.hist(returns, bins=50, alpha=0.5, label=f"Stage {stage}")
    
    ax.set_title("Returns Distribution by Curriculum Stage")
    ax.set_xlabel("Returns")
    ax.set_ylabel("Frequency")
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "curriculum_returns.png"))
    logger.info(f"Saved curriculum returns to {os.path.join(output_dir, 'curriculum_returns.png')}")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Download data if requested
    if args.download:
        download_data(args.symbol, args.days)
    
    # Process data
    X, y = process_data(args.symbol)
    
    # Create Multiverse Generator
    mg = MultiverseGenerator(
        model_dir=args.model_dir,
    )
    
    # Load model if available
    mg.load()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Visualize causal graph
    visualize_causal_graph(mg, args.output_dir)
    
    # Visualize perturbed graphs
    visualize_perturbed_graphs(mg, args.output_dir)
    
    # Generate synthetic data
    synthetic_X, _ = mg.generate_samples(
        num_samples=args.num_samples,
    )
    
    # Reshape real data to match synthetic data format
    X_reshaped = X.transpose(0, 2, 1)
    
    # Visualize synthetic data
    visualize_synthetic_data(X_reshaped, synthetic_X, args.output_dir)
    
    # Visualize curriculum
    visualize_curriculum(mg, args.output_dir)
    
    logger.info("Visualization completed successfully!")

if __name__ == "__main__":
    main()
