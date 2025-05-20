"""
Script to run the Meta-Gradient Introspection (MGI) module.
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
from mqtm.meta_gradient.introspection import MetaGradientIntrospection, HyperparameterType

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mgi_module.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MGI Module")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--asp_dir", type=str, default="models/asp",
                        help="Directory with ASP models")
    parser.add_argument("--mgi_dir", type=str, default="models/mgi",
                        help="Directory to save MGI models")
    parser.add_argument("--num_iterations", type=int, default=1000,
                        help="Number of iterations to run")
    parser.add_argument("--meta_batch_size", type=int, default=16,
                        help="Meta batch size")
    parser.add_argument("--meta_lr", type=float, default=1e-3,
                        help="Meta learning rate")
    parser.add_argument("--use_adaptive_lr", action="store_true",
                        help="Whether to use adaptive learning rate")
    parser.add_argument("--use_second_order", action="store_true",
                        help="Whether to use second-order optimization")
    parser.add_argument("--optimize_hyperparams", type=str, nargs="+",
                        default=["learning_rate", "risk_aversion", "stop_loss", "take_profit"],
                        help="Hyperparameters to optimize")
    
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

def create_mgi_module(args, models, hyperparams):
    """Create MGI module."""
    logger.info(f"Creating MGI module with {len(models)} models...")
    
    mgi = MetaGradientIntrospection(
        models=models,
        hyperparams=hyperparams,
        meta_lr=args.meta_lr,
        use_adaptive_lr=args.use_adaptive_lr,
        use_second_order=args.use_second_order,
        device=config.hardware.device,
    )
    
    return mgi

def run_meta_training(
    args,
    mg,
    tqe,
    sp3,
    trader_agent,
    adversary_agent,
    mgi
):
    """Run meta-training with the MGI module."""
    logger.info(f"Running meta-training for {args.num_iterations} iterations...")
    
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
    meta_metrics = {
        "meta_losses": [],
        "hyperparams": {name: [] for name in mgi.hyperparam_optimizer.hyperparams},
        "topology_changes": [],
        "trader_rewards": [],
        "adversary_rewards": [],
    }
    
    # Run meta-training
    for iteration in range(args.num_iterations):
        logger.info(f"Meta-training iteration {iteration+1}/{args.num_iterations}")
        
        # Generate market data
        market_data, _ = mg.generate_samples(
            num_samples=args.meta_batch_size,
            seq_len=config.data.history_length,
        )
        
        # Initialize meta batch
        meta_batch_losses = []
        trader_batch_rewards = []
        adversary_batch_rewards = []
        
        # Run meta batch
        for batch_idx in range(args.meta_batch_size):
            # Get market data for this batch
            batch_data = market_data[batch_idx:batch_idx+1]
            
            # Initialize state
            state = framework._preprocess_market_data(batch_data)
            
            # Run episode
            episode_reward = 0.0
            trader_rewards = []
            adversary_rewards = []
            
            for step in range(20):  # Short episodes for meta-training
                # Switch actors if needed
                if step % framework.steps_per_actor == 0:
                    framework.current_actor = "trader" if framework.current_actor == "adversary" else "adversary"
                
                # Trader's turn
                if framework.current_actor == "trader":
                    # Trader selects action
                    action = trader_agent.act(state)
                    
                    # Apply action and get reward
                    price = batch_data[0, 3, -1]  # Close price
                    pnl, _ = trader_agent.update_position(action, price)
                    reward = trader_agent.calculate_reward(pnl, framework.reward_type)
                    
                    # Update metrics
                    episode_reward += reward
                    trader_rewards.append(reward)
                
                # Adversary's turn
                else:
                    # Adversary selects action
                    action = adversary_agent.act(state)
                    
                    # Perturb market data
                    batch_data = adversary_agent.perturb_market(batch_data, action)
                    
                    # Adversary gets negative of trader's reward
                    price = batch_data[0, 3, -1]  # Close price
                    pnl, _ = trader_agent.update_position(Action.HOLD.value, price)
                    trader_reward = trader_agent.calculate_reward(pnl, framework.reward_type)
                    reward = -trader_reward
                    
                    # Update metrics
                    episode_reward += reward
                    adversary_rewards.append(reward)
                
                # Generate next market data
                next_batch_data, _ = mg.generate_samples(
                    num_samples=1,
                    seq_len=config.data.history_length,
                    conditioning=batch_data,
                )
                
                # Preprocess next state
                next_state = framework._preprocess_market_data(next_batch_data)
                
                # Update state
                state = next_state
                batch_data = next_batch_data
            
            # Compute meta loss
            meta_loss = -episode_reward  # Minimize negative reward
            meta_batch_losses.append(meta_loss)
            
            # Update batch metrics
            trader_batch_rewards.extend(trader_rewards)
            adversary_batch_rewards.extend(adversary_rewards)
        
        # Compute mean meta loss
        mean_meta_loss = torch.tensor(meta_batch_losses).mean()
        
        # Update MGI module
        updated_hyperparams = mgi.meta_step(mean_meta_loss, market_data[0])
        
        # Update models with new hyperparameters
        if "learning_rate" in updated_hyperparams:
            lr = updated_hyperparams["learning_rate"]
            for optimizer in trader_agent.policy_optimizer, trader_agent.value_optimizer, \
                            adversary_agent.policy_optimizer, adversary_agent.value_optimizer:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
        
        if "risk_aversion" in updated_hyperparams:
            trader_agent.risk_aversion = updated_hyperparams["risk_aversion"]
        
        if "stop_loss" in updated_hyperparams:
            trader_agent.risk_manager.stop_loss_pct = updated_hyperparams["stop_loss"]
        
        if "take_profit" in updated_hyperparams:
            trader_agent.risk_manager.take_profit_pct = updated_hyperparams["take_profit"]
        
        if "position_size" in updated_hyperparams:
            trader_agent.risk_manager.max_position_size = int(updated_hyperparams["position_size"])
        
        if "gamma" in updated_hyperparams:
            trader_agent.gamma = updated_hyperparams["gamma"]
            adversary_agent.gamma = updated_hyperparams["gamma"]
        
        if "kl_budget" in updated_hyperparams:
            adversary_agent.kl_budget = updated_hyperparams["kl_budget"]
        
        if "perturbation_strength" in updated_hyperparams:
            adversary_agent.perturbation_strength = updated_hyperparams["perturbation_strength"]
        
        # Update metrics
        meta_metrics["meta_losses"].append(mean_meta_loss.item())
        
        for name, value in updated_hyperparams.items():
            meta_metrics["hyperparams"][name].append(value)
        
        meta_metrics["topology_changes"].append(mgi.meta_state["topology_changes"])
        meta_metrics["trader_rewards"].append(np.mean(trader_batch_rewards))
        meta_metrics["adversary_rewards"].append(np.mean(adversary_batch_rewards))
        
        # Log progress
        if (iteration + 1) % 10 == 0:
            logger.info(f"Meta loss: {mean_meta_loss.item():.4f}")
            logger.info(f"Hyperparameters: {updated_hyperparams}")
            logger.info(f"Topology changes: {mgi.meta_state['topology_changes']}")
        
        # Save models and MGI module
        if (iteration + 1) % 100 == 0:
            save_models(args.mgi_dir, trader_agent, adversary_agent, mgi)
            plot_meta_metrics(meta_metrics, args.mgi_dir)
    
    # Save final models and MGI module
    save_models(args.mgi_dir, trader_agent, adversary_agent, mgi)
    plot_meta_metrics(meta_metrics, args.mgi_dir)
    
    return meta_metrics

def save_models(mgi_dir, trader_agent, adversary_agent, mgi):
    """Save models and MGI module."""
    os.makedirs(mgi_dir, exist_ok=True)
    
    # Save agents
    trader_agent.save(os.path.join(mgi_dir, "trader_agent.pt"))
    adversary_agent.save(os.path.join(mgi_dir, "adversary_agent.pt"))
    
    # Save MGI module
    mgi.save(os.path.join(mgi_dir, "mgi_module.pt"))
    
    logger.info(f"Saved models and MGI module to {mgi_dir}")

def plot_meta_metrics(meta_metrics, output_dir):
    """Plot meta-training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for meta loss
    plt.figure(figsize=(10, 6))
    plt.plot(meta_metrics["meta_losses"])
    plt.title("Meta Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "meta_loss.png"))
    plt.close()
    
    # Create figure for hyperparameters
    plt.figure(figsize=(12, 8))
    for name, values in meta_metrics["hyperparams"].items():
        plt.plot(values, label=name)
    plt.title("Hyperparameter Evolution")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "hyperparams.png"))
    plt.close()
    
    # Create figure for topology changes
    plt.figure(figsize=(10, 6))
    plt.plot(meta_metrics["topology_changes"])
    plt.title("Topology Changes")
    plt.xlabel("Iteration")
    plt.ylabel("Count")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "topology_changes.png"))
    plt.close()
    
    # Create figure for rewards
    plt.figure(figsize=(10, 6))
    plt.plot(meta_metrics["trader_rewards"], label="Trader")
    plt.plot(meta_metrics["adversary_rewards"], label="Adversary")
    plt.title("Agent Rewards")
    plt.xlabel("Iteration")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "rewards.png"))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "meta_metrics.json"), "w") as f:
        json.dump({
            "meta_losses": meta_metrics["meta_losses"],
            "hyperparams": {name: values for name, values in meta_metrics["hyperparams"].items()},
            "topology_changes": meta_metrics["topology_changes"],
            "trader_rewards": meta_metrics["trader_rewards"],
            "adversary_rewards": meta_metrics["adversary_rewards"],
        }, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.mgi_dir, exist_ok=True)
    
    # Load models
    mg, tqe, sp3, trader_agent, adversary_agent = load_models(args.models_dir, args.asp_dir)
    
    # Define hyperparameters to optimize
    hyperparams = {}
    
    for param_name in args.optimize_hyperparams:
        if param_name == "learning_rate":
            hyperparams["learning_rate"] = 1e-4
        elif param_name == "risk_aversion":
            hyperparams["risk_aversion"] = 0.1
        elif param_name == "stop_loss":
            hyperparams["stop_loss"] = 0.02
        elif param_name == "take_profit":
            hyperparams["take_profit"] = 0.05
        elif param_name == "position_size":
            hyperparams["position_size"] = 10.0
        elif param_name == "gamma":
            hyperparams["gamma"] = 0.99
        elif param_name == "kl_budget":
            hyperparams["kl_budget"] = 0.05
        elif param_name == "perturbation_strength":
            hyperparams["perturbation_strength"] = 0.1
    
    # Create MGI module
    models = [
        trader_agent.policy_network,
        trader_agent.value_network,
        adversary_agent.policy_network,
        adversary_agent.value_network,
    ]
    
    mgi = create_mgi_module(args, models, hyperparams)
    
    # Run meta-training
    meta_metrics = run_meta_training(
        args, mg, tqe, sp3, trader_agent, adversary_agent, mgi
    )
    
    logger.info("MGI module training completed")

if __name__ == "__main__":
    main()
