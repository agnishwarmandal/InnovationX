"""
Script to run the Bayesian Online Mixture (BOM) module.
"""

import os
import argparse
import logging
import time
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple, Union, Any

from mqtm.config import config
from mqtm.multiverse_generator.generator import MultiverseGenerator
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.superposition_pool.superposition_model import SuperpositionPool
from mqtm.adversarial_self_play.framework import ASPFramework, Action, Position, RewardType
from mqtm.adversarial_self_play.trader_agent import AdvancedTraderAgent
from mqtm.adversarial_self_play.adversary_agent import AdvancedAdversaryAgent
from mqtm.bayesian_mixture.online_learning import BayesianOnlineMixture

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("bom_module.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run BOM Module")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--asp_dir", type=str, default="models/asp",
                        help="Directory with ASP models")
    parser.add_argument("--bom_dir", type=str, default="models/bom",
                        help="Directory to save BOM models")
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="Number of iterations to run")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--num_models", type=int, default=3,
                        help="Number of models in the mixture")
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[128, 128],
                        help="Hidden dimensions for Bayesian networks")
    parser.add_argument("--prior_std", type=float, default=0.1,
                        help="Prior standard deviation")
    parser.add_argument("--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    
    return parser.parse_args()

def load_models(models_dir, asp_dir):
    """Load MQTM and ASP models."""
    logger.info(f"Loading models from {models_dir} and {asp_dir}...")
    
    # Create MQTM models
    mg = MultiverseGenerator(
        model_dir=os.path.join(models_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load MQTM models if available
    mg.load()
    
    tqe_path = os.path.join(models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(models_dir, "sp3.pt")
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
    
    # Load ASP agents if available
    trader_agent.load(os.path.join(asp_dir, "trader_agent.pt"))
    adversary_agent.load(os.path.join(asp_dir, "adversary_agent.pt"))
    
    # Move models to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    return mg, tqe, sp3, trader_agent, adversary_agent

def create_bom_module(args, input_dim, output_dim):
    """Create BOM module."""
    logger.info(f"Creating BOM module with input_dim={input_dim}, output_dim={output_dim}...")
    
    bom = BayesianOnlineMixture(
        input_dim=input_dim,
        hidden_dims=args.hidden_dims,
        output_dim=output_dim,
        num_models=args.num_models,
        prior_mean=0.0,
        prior_std=args.prior_std,
        learning_rate=args.learning_rate,
        device=config.hardware.device,
    )
    
    return bom

def run_online_learning(
    args,
    mg,
    tqe,
    sp3,
    trader_agent,
    adversary_agent,
    bom
):
    """Run online learning with the BOM module."""
    logger.info(f"Running online learning for {args.num_iterations} iterations...")
    
    # Create ASP framework
    framework = ASPFramework(
        trader_agent=trader_agent,
        adversary_agent=adversary_agent,
        multiverse_generator=mg,
        topo_quantum_encoder=tqe,
        superposition_pool=sp3,
        exchange_client=None,
        reward_type=RewardType.PNL,
        steps_per_actor=10,
        device=config.hardware.device,
    )
    
    # Initialize metrics
    metrics = {
        "losses": [],
        "rewards": [],
        "uncertainties": [],
    }
    
    # Run online learning
    for iteration in range(args.num_iterations):
        logger.info(f"Online learning iteration {iteration+1}/{args.num_iterations}")
        
        # Generate market data
        market_data, _ = mg.generate_samples(
            num_samples=args.batch_size,
            seq_len=config.data.history_length,
        )
        
        # Initialize batch metrics
        batch_losses = []
        batch_rewards = []
        batch_uncertainties = []
        
        # Run batch
        for batch_idx in range(args.batch_size):
            # Get market data for this batch
            batch_data = market_data[batch_idx:batch_idx+1]
            
            # Initialize state
            state = framework._preprocess_market_data(batch_data)
            
            # Trader selects action
            action = trader_agent.act(state)
            
            # Apply action and get reward
            price = batch_data[0, 3, -1]  # Close price
            pnl, _ = trader_agent.update_position(action, price)
            reward = trader_agent.calculate_reward(pnl, framework.reward_type)
            
            # Convert state and action to tensors
            state_tensor = torch.tensor(state, dtype=torch.float32, device=config.hardware.device)
            action_tensor = torch.tensor([action], dtype=torch.long, device=config.hardware.device)
            
            # Update BOM module
            loss = bom.update(state_tensor, action_tensor)
            
            # Predict with uncertainty
            mean, variance = bom.predict(state_tensor)
            uncertainty = variance.mean().item()
            
            # Update batch metrics
            batch_losses.append(loss)
            batch_rewards.append(reward)
            batch_uncertainties.append(uncertainty)
            
            # Generate next market data
            next_batch_data, _ = mg.generate_samples(
                num_samples=1,
                seq_len=config.data.history_length,
                conditioning=batch_data,
            )
            
            # Update state
            state = framework._preprocess_market_data(next_batch_data)
            batch_data = next_batch_data
        
        # Update metrics
        metrics["losses"].append(np.mean(batch_losses))
        metrics["rewards"].append(np.mean(batch_rewards))
        metrics["uncertainties"].append(np.mean(batch_uncertainties))
        
        # Log progress
        if (iteration + 1) % 10 == 0:
            logger.info(f"Loss: {metrics['losses'][-1]:.4f}")
            logger.info(f"Reward: {metrics['rewards'][-1]:.4f}")
            logger.info(f"Uncertainty: {metrics['uncertainties'][-1]:.4f}")
        
        # Save BOM module
        if (iteration + 1) % 100 == 0:
            bom.save(os.path.join(args.bom_dir, "bom_module.pt"))
            plot_metrics(metrics, args.bom_dir)
    
    # Save final BOM module
    bom.save(os.path.join(args.bom_dir, "bom_module.pt"))
    plot_metrics(metrics, args.bom_dir)
    
    return metrics

def test_uncertainty_estimation(
    args,
    mg,
    tqe,
    sp3,
    trader_agent,
    adversary_agent,
    bom
):
    """Test uncertainty estimation with the BOM module."""
    logger.info("Testing uncertainty estimation...")
    
    # Create ASP framework
    framework = ASPFramework(
        trader_agent=trader_agent,
        adversary_agent=adversary_agent,
        multiverse_generator=mg,
        topo_quantum_encoder=tqe,
        superposition_pool=sp3,
        exchange_client=None,
        reward_type=RewardType.PNL,
        steps_per_actor=10,
        device=config.hardware.device,
    )
    
    # Generate market data
    market_data, _ = mg.generate_samples(
        num_samples=100,
        seq_len=config.data.history_length,
    )
    
    # Initialize metrics
    uncertainty_metrics = {
        "states": [],
        "actions": [],
        "rewards": [],
        "uncertainties": [],
        "correct_predictions": [],
    }
    
    # Run test
    for i in range(100):
        # Get market data for this test
        batch_data = market_data[i:i+1]
        
        # Initialize state
        state = framework._preprocess_market_data(batch_data)
        
        # Trader selects action
        action = trader_agent.act(state)
        
        # Apply action and get reward
        price = batch_data[0, 3, -1]  # Close price
        pnl, _ = trader_agent.update_position(action, price)
        reward = trader_agent.calculate_reward(pnl, framework.reward_type)
        
        # Convert state to tensor
        state_tensor = torch.tensor(state, dtype=torch.float32, device=config.hardware.device)
        
        # Predict with uncertainty
        mean, variance = bom.predict(state_tensor)
        
        # Get predicted action
        predicted_action = mean.argmax().item()
        
        # Compute uncertainty
        uncertainty = variance.mean().item()
        
        # Update metrics
        uncertainty_metrics["states"].append(state)
        uncertainty_metrics["actions"].append(action)
        uncertainty_metrics["rewards"].append(reward)
        uncertainty_metrics["uncertainties"].append(uncertainty)
        uncertainty_metrics["correct_predictions"].append(predicted_action == action)
    
    # Analyze uncertainty vs. correctness
    correct_uncertainties = [u for u, c in zip(uncertainty_metrics["uncertainties"], uncertainty_metrics["correct_predictions"]) if c]
    incorrect_uncertainties = [u for u, c in zip(uncertainty_metrics["uncertainties"], uncertainty_metrics["correct_predictions"]) if not c]
    
    logger.info(f"Average uncertainty for correct predictions: {np.mean(correct_uncertainties):.4f}")
    logger.info(f"Average uncertainty for incorrect predictions: {np.mean(incorrect_uncertainties):.4f}")
    
    # Plot uncertainty vs. correctness
    plt.figure(figsize=(10, 6))
    plt.hist(correct_uncertainties, bins=20, alpha=0.5, label="Correct")
    plt.hist(incorrect_uncertainties, bins=20, alpha=0.5, label="Incorrect")
    plt.title("Uncertainty vs. Correctness")
    plt.xlabel("Uncertainty")
    plt.ylabel("Count")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(args.bom_dir, "uncertainty_vs_correctness.png"))
    plt.close()
    
    # Plot uncertainty vs. reward
    plt.figure(figsize=(10, 6))
    plt.scatter(uncertainty_metrics["uncertainties"], uncertainty_metrics["rewards"])
    plt.title("Uncertainty vs. Reward")
    plt.xlabel("Uncertainty")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(args.bom_dir, "uncertainty_vs_reward.png"))
    plt.close()
    
    return uncertainty_metrics

def plot_metrics(metrics, output_dir):
    """Plot online learning metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["losses"])
    plt.title("Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()
    
    # Create figure for reward
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["rewards"])
    plt.title("Reward")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "reward.png"))
    plt.close()
    
    # Create figure for uncertainty
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["uncertainties"])
    plt.title("Uncertainty")
    plt.xlabel("Iteration")
    plt.ylabel("Uncertainty")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "uncertainty.png"))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "metrics.json"), "w") as f:
        json.dump({
            "losses": metrics["losses"],
            "rewards": metrics["rewards"],
            "uncertainties": metrics["uncertainties"],
        }, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.bom_dir, exist_ok=True)
    
    # Load models
    mg, tqe, sp3, trader_agent, adversary_agent = load_models(args.models_dir, args.asp_dir)
    
    # Create BOM module
    input_dim = tqe.total_features_dim
    output_dim = 3  # BUY, SELL, HOLD
    
    bom = create_bom_module(args, input_dim, output_dim)
    
    # Load existing BOM module if available
    bom_path = os.path.join(args.bom_dir, "bom_module.pt")
    if os.path.exists(bom_path):
        bom.load(bom_path)
    
    # Run online learning
    metrics = run_online_learning(
        args, mg, tqe, sp3, trader_agent, adversary_agent, bom
    )
    
    # Test uncertainty estimation
    uncertainty_metrics = test_uncertainty_estimation(
        args, mg, tqe, sp3, trader_agent, adversary_agent, bom
    )
    
    logger.info("BOM module training completed")

if __name__ == "__main__":
    main()
