"""
Script to evaluate the trained MQTM system.
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
from mqtm.adversarial_self_play.framework import ASPFramework, Action, Position, RewardType
from mqtm.adversarial_self_play.trader_agent import AdvancedTraderAgent
from mqtm.adversarial_self_play.adversary_agent import AdvancedAdversaryAgent
from mqtm.meta_gradient.introspection import MetaGradientIntrospection
from mqtm.bayesian_mixture.online_learning import BayesianOnlineMixture
from mqtm.utils.memory_optimization import MemoryOptimizer
from mqtm.utils.performance_profiling import Timer, TorchProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("evaluate_mqtm.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate MQTM System")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--output_dir", type=str, default="evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to evaluate on (if None, use all available)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for evaluation")
    parser.add_argument("--num_samples", type=int, default=100,
                        help="Number of samples to generate")
    parser.add_argument("--use_asp", action="store_true",
                        help="Whether to use ASP models")
    parser.add_argument("--use_mgi", action="store_true",
                        help="Whether to use MGI models")
    parser.add_argument("--use_bom", action="store_true",
                        help="Whether to use BOM models")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Whether to optimize memory usage")
    
    return parser.parse_args()

def load_data(args):
    """Load cryptocurrency data."""
    logger.info(f"Loading data from {args.data_dir}...")
    
    # Create data loader
    data_loader = CryptoDataLoader(
        data_dir=args.data_dir,
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

def load_models(args):
    """Load MQTM models."""
    logger.info(f"Loading models from {args.models_dir}...")
    
    # Create MQTM models
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.models_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load MQTM models if available
    mg.load()
    
    tqe_path = os.path.join(args.models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(args.models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Create ASP agents
    state_dim = tqe.total_features_dim
    
    trader_agent = AdvancedTraderAgent(
        state_dim=state_dim,
        action_dim=3,  # BUY, SELL, HOLD
        hidden_dim=128,
        learning_rate=1e-4,
        gamma=0.99,
        risk_aversion=0.1,
        max_position_size=10,
        stop_loss_pct=0.02,
        take_profit_pct=0.05,
        use_regime_detection=True,
        device=config.hardware.device,
    )
    
    adversary_agent = AdvancedAdversaryAgent(
        state_dim=state_dim,
        action_dim=8,  # Number of market scenarios
        hidden_dim=128,
        learning_rate=1e-4,
        gamma=0.99,
        kl_budget=0.05,
        perturbation_strength=0.1,
        use_causal_model=True,
        device=config.hardware.device,
    )
    
    # Load ASP agents if available and requested
    if args.use_asp:
        trader_agent.load(os.path.join(args.models_dir, "asp", "trader_agent.pt"))
        adversary_agent.load(os.path.join(args.models_dir, "asp", "adversary_agent.pt"))
    
    # Load MGI models if available and requested
    if args.use_mgi:
        trader_agent.load(os.path.join(args.models_dir, "mgi", "trader_agent.pt"))
        adversary_agent.load(os.path.join(args.models_dir, "mgi", "adversary_agent.pt"))
    
    # Create BOM module if requested
    bom = None
    if args.use_bom:
        bom = BayesianOnlineMixture(
            input_dim=state_dim,
            hidden_dims=[128, 128],
            output_dim=3,  # BUY, SELL, HOLD
            num_models=3,
            prior_mean=0.0,
            prior_std=0.1,
            learning_rate=1e-3,
            device=config.hardware.device,
        )
        
        # Load BOM module if available
        bom_path = os.path.join(args.models_dir, "bom", "bom_module.pt")
        if os.path.exists(bom_path):
            bom.load(bom_path)
    
    # Move models to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    # Optimize memory if requested
    if args.optimize_memory:
        logger.info("Optimizing memory usage...")
        MemoryOptimizer.optimize_model_memory(tqe)
        MemoryOptimizer.optimize_model_memory(sp3)
    
    return mg, tqe, sp3, trader_agent, adversary_agent, bom

def evaluate_multiverse_generator(args, mg, output_dir):
    """Evaluate Multiverse Generator."""
    logger.info("Evaluating Multiverse Generator...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "multiverse_generator"), exist_ok=True)
    
    # Generate samples
    with Timer("Generating samples"):
        samples, graph = mg.generate_samples(
            num_samples=args.num_samples,
            seq_len=config.data.history_length,
        )
    
    logger.info(f"Generated {len(samples)} samples with shape {samples.shape}")
    
    # Analyze samples
    sample_stats = {
        "mean": np.mean(samples, axis=(0, 2)).tolist(),
        "std": np.std(samples, axis=(0, 2)).tolist(),
        "min": np.min(samples, axis=(0, 2)).tolist(),
        "max": np.max(samples, axis=(0, 2)).tolist(),
    }
    
    # Save sample statistics
    with open(os.path.join(output_dir, "multiverse_generator", "sample_stats.json"), "w") as f:
        json.dump(sample_stats, f, indent=2)
    
    # Plot samples
    for i in range(min(5, len(samples))):
        plt.figure(figsize=(12, 8))
        
        # Plot OHLC
        plt.subplot(2, 1, 1)
        plt.plot(samples[i, 0], label="Open")
        plt.plot(samples[i, 1], label="High")
        plt.plot(samples[i, 2], label="Low")
        plt.plot(samples[i, 3], label="Close")
        plt.title(f"Sample {i+1} - OHLC")
        plt.legend()
        plt.grid(True)
        
        # Plot volume
        plt.subplot(2, 1, 2)
        plt.plot(samples[i, 4], label="Volume")
        plt.title(f"Sample {i+1} - Volume")
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "multiverse_generator", f"sample_{i+1}.png"))
        plt.close()
    
    # Plot causal graph if available
    if graph is not None:
        plt.figure(figsize=(10, 8))
        plt.imshow(graph, cmap="viridis")
        plt.colorbar()
        plt.title("Causal Graph")
        plt.savefig(os.path.join(output_dir, "multiverse_generator", "causal_graph.png"))
        plt.close()
    
    return samples

def evaluate_topo_quantum_encoder(args, tqe, samples, output_dir):
    """Evaluate Topo-Quantum Encoder."""
    logger.info("Evaluating Topo-Quantum Encoder...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "topo_quantum_encoder"), exist_ok=True)
    
    # Convert samples to tensor
    samples_tensor = torch.tensor(samples, dtype=torch.float32, device=config.hardware.device)
    
    # Extract features
    with Timer("Extracting features"):
        with torch.no_grad():
            features = tqe(samples_tensor)
    
    logger.info(f"Extracted features with shape {features.shape}")
    
    # Convert features to numpy
    features_np = features.cpu().numpy()
    
    # Analyze features
    feature_stats = {
        "mean": np.mean(features_np, axis=0).tolist(),
        "std": np.std(features_np, axis=0).tolist(),
        "min": np.min(features_np, axis=0).tolist(),
        "max": np.max(features_np, axis=0).tolist(),
    }
    
    # Save feature statistics
    with open(os.path.join(output_dir, "topo_quantum_encoder", "feature_stats.json"), "w") as f:
        json.dump(feature_stats, f, indent=2)
    
    # Plot feature heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(features_np, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Feature Heatmap")
    plt.xlabel("Feature Dimension")
    plt.ylabel("Sample")
    plt.savefig(os.path.join(output_dir, "topo_quantum_encoder", "feature_heatmap.png"))
    plt.close()
    
    # Plot feature correlation
    plt.figure(figsize=(12, 8))
    correlation = np.corrcoef(features_np.T)
    plt.imshow(correlation, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar()
    plt.title("Feature Correlation")
    plt.savefig(os.path.join(output_dir, "topo_quantum_encoder", "feature_correlation.png"))
    plt.close()
    
    return features

def evaluate_superposition_pool(args, sp3, features, output_dir):
    """Evaluate Superposition Pool."""
    logger.info("Evaluating Superposition Pool...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "superposition_pool"), exist_ok=True)
    
    # Create regime tensor
    batch_size = features.size(0)
    regime = torch.rand(batch_size, 2, device=config.hardware.device)
    
    # Forward pass
    with Timer("Forward pass"):
        with torch.no_grad():
            outputs = sp3(features, regime)
    
    logger.info(f"Generated outputs with shape {outputs.shape}")
    
    # Convert outputs to numpy
    outputs_np = outputs.cpu().numpy()
    
    # Analyze outputs
    output_stats = {
        "mean": np.mean(outputs_np, axis=0).tolist(),
        "std": np.std(outputs_np, axis=0).tolist(),
        "min": np.min(outputs_np, axis=0).tolist(),
        "max": np.max(outputs_np, axis=0).tolist(),
    }
    
    # Save output statistics
    with open(os.path.join(output_dir, "superposition_pool", "output_stats.json"), "w") as f:
        json.dump(output_stats, f, indent=2)
    
    # Plot output heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(outputs_np, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Output Heatmap")
    plt.xlabel("Output Dimension")
    plt.ylabel("Sample")
    plt.savefig(os.path.join(output_dir, "superposition_pool", "output_heatmap.png"))
    plt.close()
    
    # Plot output distribution
    plt.figure(figsize=(12, 8))
    plt.hist(outputs_np.flatten(), bins=50)
    plt.title("Output Distribution")
    plt.xlabel("Output Value")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "superposition_pool", "output_distribution.png"))
    plt.close()
    
    # Plot head weights
    plt.figure(figsize=(10, 6))
    weights = torch.nn.functional.softmax(sp3.head_weights, dim=0).cpu().numpy()
    plt.bar(range(len(weights)), weights)
    plt.title("Head Weights")
    plt.xlabel("Head Index")
    plt.ylabel("Weight")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "superposition_pool", "head_weights.png"))
    plt.close()
    
    return outputs

def evaluate_trader_agent(args, trader_agent, features, output_dir):
    """Evaluate Trader Agent."""
    logger.info("Evaluating Trader Agent...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "trader_agent"), exist_ok=True)
    
    # Initialize metrics
    actions = []
    rewards = []
    positions = []
    
    # Reset trader agent
    trader_agent.position = Position.FLAT
    trader_agent.position_size = 0
    trader_agent.entry_price = 0.0
    trader_agent.pnl = 0.0
    trader_agent.returns = []
    trader_agent.trades = []
    
    # Convert features to numpy
    features_np = features.cpu().numpy()
    
    # Evaluate on each sample
    for i in range(len(features_np)):
        # Get state
        state = features_np[i]
        
        # Select action
        action = trader_agent.act(state)
        actions.append(action)
        
        # Apply action and get reward
        price = 100.0  # Placeholder price
        pnl, position = trader_agent.update_position(action, price)
        rewards.append(pnl)
        positions.append(position.value)
    
    # Analyze actions
    action_counts = np.bincount(actions, minlength=3)
    action_percentages = action_counts / len(actions) * 100
    
    logger.info(f"Action distribution: BUY={action_percentages[0]:.2f}%, "
               f"SELL={action_percentages[1]:.2f}%, "
               f"HOLD={action_percentages[2]:.2f}%")
    
    # Plot action distribution
    plt.figure(figsize=(10, 6))
    plt.bar(["BUY", "SELL", "HOLD"], action_counts)
    plt.title("Action Distribution")
    plt.xlabel("Action")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "trader_agent", "action_distribution.png"))
    plt.close()
    
    # Plot cumulative PnL
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(trader_agent.returns))
    plt.title("Cumulative PnL")
    plt.xlabel("Trade")
    plt.ylabel("PnL")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "trader_agent", "cumulative_pnl.png"))
    plt.close()
    
    # Get performance metrics
    performance_metrics = trader_agent.get_performance_metrics()
    
    # Save performance metrics
    with open(os.path.join(output_dir, "trader_agent", "performance_metrics.json"), "w") as f:
        json.dump(performance_metrics, f, indent=2)
    
    return actions, rewards, positions

def evaluate_bayesian_mixture(args, bom, features, output_dir):
    """Evaluate Bayesian Online Mixture."""
    if bom is None:
        logger.info("Skipping Bayesian Online Mixture evaluation (not requested)")
        return None
    
    logger.info("Evaluating Bayesian Online Mixture...")
    
    # Create output directory
    os.makedirs(os.path.join(output_dir, "bayesian_mixture"), exist_ok=True)
    
    # Initialize metrics
    means = []
    variances = []
    
    # Evaluate on each sample
    for i in range(len(features)):
        # Get state
        state = features[i:i+1]
        
        # Predict with uncertainty
        mean, variance = bom.predict(state)
        
        # Convert to numpy
        mean_np = mean.cpu().numpy()
        variance_np = variance.cpu().numpy()
        
        means.append(mean_np)
        variances.append(variance_np)
    
    # Convert to arrays
    means = np.concatenate(means, axis=0)
    variances = np.concatenate(variances, axis=0)
    
    # Analyze predictions
    prediction_stats = {
        "mean_mean": np.mean(means, axis=0).tolist(),
        "mean_std": np.std(means, axis=0).tolist(),
        "variance_mean": np.mean(variances, axis=0).tolist(),
        "variance_std": np.std(variances, axis=0).tolist(),
    }
    
    # Save prediction statistics
    with open(os.path.join(output_dir, "bayesian_mixture", "prediction_stats.json"), "w") as f:
        json.dump(prediction_stats, f, indent=2)
    
    # Plot mean heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(means, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Mean Predictions")
    plt.xlabel("Action")
    plt.ylabel("Sample")
    plt.savefig(os.path.join(output_dir, "bayesian_mixture", "mean_heatmap.png"))
    plt.close()
    
    # Plot variance heatmap
    plt.figure(figsize=(12, 8))
    plt.imshow(variances, aspect="auto", cmap="viridis")
    plt.colorbar()
    plt.title("Prediction Variance")
    plt.xlabel("Action")
    plt.ylabel("Sample")
    plt.savefig(os.path.join(output_dir, "bayesian_mixture", "variance_heatmap.png"))
    plt.close()
    
    # Plot uncertainty histogram
    plt.figure(figsize=(10, 6))
    plt.hist(np.mean(variances, axis=1), bins=50)
    plt.title("Uncertainty Distribution")
    plt.xlabel("Uncertainty")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "bayesian_mixture", "uncertainty_histogram.png"))
    plt.close()
    
    return means, variances

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    data_loader, train_loader, val_loader, test_loader = load_data(args)
    
    # Load models
    mg, tqe, sp3, trader_agent, adversary_agent, bom = load_models(args)
    
    # Evaluate Multiverse Generator
    samples = evaluate_multiverse_generator(args, mg, args.output_dir)
    
    # Evaluate Topo-Quantum Encoder
    features = evaluate_topo_quantum_encoder(args, tqe, samples, args.output_dir)
    
    # Evaluate Superposition Pool
    outputs = evaluate_superposition_pool(args, sp3, features, args.output_dir)
    
    # Evaluate Trader Agent
    actions, rewards, positions = evaluate_trader_agent(args, trader_agent, features, args.output_dir)
    
    # Evaluate Bayesian Online Mixture
    if args.use_bom:
        means, variances = evaluate_bayesian_mixture(args, bom, features, args.output_dir)
    
    logger.info("Evaluation completed successfully!")

if __name__ == "__main__":
    main()
