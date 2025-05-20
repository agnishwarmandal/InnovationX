"""
Script to train the Adversarial Self-Play (ASP) framework.
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
from mqtm.utils.memory_optimization import MemoryOptimizer
from mqtm.utils.performance_profiling import Timer, TorchProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("train_asp.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train ASP Framework")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--asp_dir", type=str, default="models/asp",
                        help="Directory to save ASP models")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to train on (if None, use all available)")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum number of steps per episode")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate for training")
    parser.add_argument("--reward_type", type=str, 
                        choices=["pnl", "sharpe", "sortino", "calmar", "custom"],
                        default="pnl", help="Type of reward to use")
    parser.add_argument("--risk_aversion", type=float, default=0.1,
                        help="Risk aversion parameter")
    parser.add_argument("--max_position_size", type=int, default=10,
                        help="Maximum position size")
    parser.add_argument("--stop_loss_pct", type=float, default=0.02,
                        help="Stop loss percentage")
    parser.add_argument("--take_profit_pct", type=float, default=0.05,
                        help="Take profit percentage")
    parser.add_argument("--kl_budget", type=float, default=0.05,
                        help="KL divergence budget for perturbation")
    parser.add_argument("--perturbation_strength", type=float, default=0.1,
                        help="Strength of perturbation")
    parser.add_argument("--use_regime_detection", action="store_true",
                        help="Whether to use regime detection")
    parser.add_argument("--use_causal_model", action="store_true",
                        help="Whether to use causal model for perturbation")
    parser.add_argument("--optimize_memory", action="store_true",
                        help="Whether to optimize memory usage")
    parser.add_argument("--profile_performance", action="store_true",
                        help="Whether to profile performance")
    
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
    
    # Create Multiverse Generator
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.models_dir, "multiverse_generator"),
    )
    
    # Create Topo-Quantum Encoder
    tqe = TopoQuantumEncoder()
    
    # Create Superposition Pool
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load models if available
    mg.load()
    
    tqe_path = os.path.join(args.models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(args.models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Move models to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
    # Optimize memory if requested
    if args.optimize_memory:
        logger.info("Optimizing memory usage...")
        MemoryOptimizer.optimize_model_memory(tqe)
        MemoryOptimizer.optimize_model_memory(sp3)
    
    return mg, tqe, sp3

def create_agents(args, state_dim):
    """Create trader and adversary agents."""
    logger.info(f"Creating agents with state_dim={state_dim}...")
    
    # Map reward type string to enum
    reward_type_map = {
        "pnl": RewardType.PNL,
        "sharpe": RewardType.SHARPE,
        "sortino": RewardType.SORTINO,
        "calmar": RewardType.CALMAR,
        "custom": RewardType.CUSTOM,
    }
    reward_type = reward_type_map[args.reward_type]
    
    # Create trader agent
    trader_agent = AdvancedTraderAgent(
        state_dim=state_dim,
        action_dim=3,  # BUY, SELL, HOLD
        hidden_dim=128,
        learning_rate=args.learning_rate,
        gamma=0.99,
        risk_aversion=args.risk_aversion,
        max_position_size=args.max_position_size,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        use_regime_detection=args.use_regime_detection,
        device=config.hardware.device,
    )
    
    # Create adversary agent
    adversary_agent = AdvancedAdversaryAgent(
        state_dim=state_dim,
        action_dim=8,  # Number of market scenarios
        hidden_dim=128,
        learning_rate=args.learning_rate,
        gamma=0.99,
        kl_budget=args.kl_budget,
        perturbation_strength=args.perturbation_strength,
        use_causal_model=args.use_causal_model,
        device=config.hardware.device,
    )
    
    return trader_agent, adversary_agent

def train_asp_framework(args, mg, tqe, sp3, trader_agent, adversary_agent):
    """Train the ASP framework."""
    logger.info("Training ASP framework...")
    
    # Create ASP framework
    framework = ASPFramework(
        trader_agent=trader_agent,
        adversary_agent=adversary_agent,
        multiverse_generator=mg,
        topo_quantum_encoder=tqe,
        superposition_pool=sp3,
        exchange_client=None,
        reward_type=RewardType[args.reward_type.upper()],
        steps_per_actor=10,
        device=config.hardware.device,
    )
    
    # Create output directory
    os.makedirs(args.asp_dir, exist_ok=True)
    
    # Load existing models if available
    if os.path.exists(os.path.join(args.asp_dir, "trader_agent.pt")):
        framework.load_models(args.asp_dir)
    
    # Create progress bar
    progress_bar = tqdm(total=args.num_episodes, desc="Training ASP")
    
    # Initialize metrics
    metrics = {
        "episode_rewards": [],
        "trader_metrics": {"policy_loss": [], "value_loss": [], "rewards": []},
        "adversary_metrics": {"policy_loss": [], "value_loss": [], "rewards": []},
    }
    
    # Train framework
    for episode in range(args.num_episodes):
        # Generate initial market data
        market_data, _ = mg.generate_samples(
            num_samples=1,
            seq_len=config.data.history_length,
        )
        
        # Initialize episode state
        state = framework._preprocess_market_data(market_data)
        episode_reward = 0.0
        
        # Run episode
        for step in range(args.max_steps):
            # Switch actors if needed
            if step % framework.steps_per_actor == 0:
                framework.current_actor = "trader" if framework.current_actor == "adversary" else "adversary"
            
            # Trader's turn
            if framework.current_actor == "trader":
                # Trader selects action
                action = trader_agent.act(state)
                
                # Apply action and get reward
                price = market_data[0, 3, -1]  # Close price
                pnl, _ = trader_agent.update_position(action, price)
                reward = trader_agent.calculate_reward(pnl, framework.reward_type)
                
                # Update metrics
                episode_reward += reward
                metrics["trader_metrics"]["rewards"].append(reward)
            
            # Adversary's turn
            else:
                # Adversary selects action
                action = adversary_agent.act(state)
                
                # Perturb market data
                market_data = adversary_agent.perturb_market(market_data, action)
                
                # Adversary gets negative of trader's reward
                price = market_data[0, 3, -1]  # Close price
                pnl, _ = trader_agent.update_position(Action.HOLD.value, price)
                trader_reward = trader_agent.calculate_reward(pnl, framework.reward_type)
                reward = -trader_reward
                
                # Update metrics
                episode_reward += reward
                metrics["adversary_metrics"]["rewards"].append(reward)
            
            # Generate next market data
            next_market_data, _ = mg.generate_samples(
                num_samples=1,
                seq_len=config.data.history_length,
                conditioning=market_data,
            )
            
            # Preprocess next state
            next_state = framework._preprocess_market_data(next_market_data)
            
            # Store experience
            done = (step == args.max_steps - 1)
            
            if framework.current_actor == "trader":
                trader_agent.add_experience(state, action, reward, next_state, done)
            else:
                adversary_agent.add_experience(state, action, reward, next_state, done)
            
            # Update state
            state = next_state
            market_data = next_market_data
            
            # Update agents
            if step % 10 == 0:
                if len(trader_agent.memory) >= args.batch_size:
                    trader_metrics = trader_agent.update(args.batch_size)
                    metrics["trader_metrics"]["policy_loss"].append(trader_metrics["policy_loss"])
                    metrics["trader_metrics"]["value_loss"].append(trader_metrics["value_loss"])
                
                if len(adversary_agent.memory) >= args.batch_size:
                    adversary_metrics = adversary_agent.update(args.batch_size)
                    metrics["adversary_metrics"]["policy_loss"].append(adversary_metrics["policy_loss"])
                    metrics["adversary_metrics"]["value_loss"].append(adversary_metrics["value_loss"])
        
        # End of episode
        metrics["episode_rewards"].append(episode_reward)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "reward": f"{episode_reward:.4f}",
            "trader_loss": f"{metrics['trader_metrics']['policy_loss'][-1] if metrics['trader_metrics']['policy_loss'] else 0:.4f}",
            "adversary_loss": f"{metrics['adversary_metrics']['policy_loss'][-1] if metrics['adversary_metrics']['policy_loss'] else 0:.4f}",
        })
        
        # Log progress
        if (episode + 1) % 10 == 0:
            logger.info(f"Episode {episode+1}/{args.num_episodes}, Reward: {episode_reward:.4f}")
        
        # Save models
        if (episode + 1) % 100 == 0:
            framework._save_models(args.asp_dir)
            plot_training_metrics(metrics, args.asp_dir)
    
    # Save final models
    framework._save_models(args.asp_dir)
    plot_training_metrics(metrics, args.asp_dir)
    
    # Close progress bar
    progress_bar.close()
    
    logger.info("ASP training completed")
    
    return framework, metrics

def plot_training_metrics(metrics, output_dir):
    """Plot training metrics."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for episode rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["episode_rewards"])
    plt.title("Episode Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "episode_rewards.png"))
    plt.close()
    
    # Create figure for policy loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["trader_metrics"]["policy_loss"], label="Trader")
    plt.plot(metrics["adversary_metrics"]["policy_loss"], label="Adversary")
    plt.title("Policy Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "policy_loss.png"))
    plt.close()
    
    # Create figure for value loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["trader_metrics"]["value_loss"], label="Trader")
    plt.plot(metrics["adversary_metrics"]["value_loss"], label="Adversary")
    plt.title("Value Loss")
    plt.xlabel("Update")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "value_loss.png"))
    plt.close()
    
    # Create figure for rewards
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["trader_metrics"]["rewards"], label="Trader")
    plt.plot(metrics["adversary_metrics"]["rewards"], label="Adversary")
    plt.title("Agent Rewards")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "agent_rewards.png"))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "training_metrics.json"), "w") as f:
        json.dump({
            "episode_rewards": metrics["episode_rewards"],
            "trader_metrics": {
                "policy_loss": metrics["trader_metrics"]["policy_loss"],
                "value_loss": metrics["trader_metrics"]["value_loss"],
            },
            "adversary_metrics": {
                "policy_loss": metrics["adversary_metrics"]["policy_loss"],
                "value_loss": metrics["adversary_metrics"]["value_loss"],
            },
        }, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.asp_dir, exist_ok=True)
    
    # Load data
    data_loader, train_loader, val_loader, test_loader = load_data(args)
    
    # Load models
    mg, tqe, sp3 = load_models(args)
    
    # Create agents
    state_dim = tqe.total_features_dim
    trader_agent, adversary_agent = create_agents(args, state_dim)
    
    # Profile performance if requested
    if args.profile_performance:
        logger.info("Profiling performance...")
        
        # Create profiler
        profiler = TorchProfiler()
        
        # Profile trader agent
        with profiler.profile_model(
            model=trader_agent.policy_network,
            inputs=torch.randn(args.batch_size, state_dim, device=config.hardware.device),
            trace_filename="profiles/trader_policy_profile",
        ):
            pass
        
        # Profile adversary agent
        with profiler.profile_model(
            model=adversary_agent.policy_network,
            inputs=torch.randn(args.batch_size, state_dim, device=config.hardware.device),
            trace_filename="profiles/adversary_policy_profile",
        ):
            pass
    
    # Train ASP framework
    framework, metrics = train_asp_framework(
        args, mg, tqe, sp3, trader_agent, adversary_agent
    )
    
    logger.info("ASP framework training completed")

if __name__ == "__main__":
    main()
