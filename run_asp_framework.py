"""
Script to run the Adversarial Self-Play (ASP) framework.
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
from mqtm.adversarial_self_play.trader_agent import AdvancedTraderAgent, RiskManager
from mqtm.adversarial_self_play.adversary_agent import AdvancedAdversaryAgent, MarketScenario
from mqtm.adversarial_self_play.delta_exchange_client import DeltaExchangeClient

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("asp_framework.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run ASP Framework")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--asp_dir", type=str, default="models/asp",
                        help="Directory to save ASP models")
    parser.add_argument("--mode", type=str, choices=["train", "backtest", "live"], default="train",
                        help="Mode to run the framework in")
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of episodes to train for")
    parser.add_argument("--max_steps", type=int, default=200,
                        help="Maximum number of steps per episode")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training")
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
    
    # Delta Exchange parameters
    parser.add_argument("--api_key", type=str, default="",
                        help="Delta Exchange API key")
    parser.add_argument("--api_secret", type=str, default="",
                        help="Delta Exchange API secret")
    parser.add_argument("--product_id", type=int, default=0,
                        help="Product ID to trade")
    parser.add_argument("--testnet", action="store_true",
                        help="Whether to use testnet")
    
    return parser.parse_args()

def load_models(models_dir):
    """Load MQTM models."""
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
    mg.load()
    
    tqe_path = os.path.join(models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
    
    sp3_path = os.path.join(models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
    
    # Move to device
    tqe.to(config.hardware.device)
    sp3.to(config.hardware.device)
    
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
        learning_rate=1e-4,
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
        action_dim=len(MarketScenario),
        hidden_dim=128,
        learning_rate=1e-4,
        gamma=0.99,
        kl_budget=args.kl_budget,
        perturbation_strength=args.perturbation_strength,
        use_causal_model=args.use_causal_model,
        device=config.hardware.device,
    )
    
    return trader_agent, adversary_agent

def create_exchange_client(args):
    """Create Delta Exchange client."""
    if not args.api_key or not args.api_secret:
        logger.warning("API key and secret not provided, skipping exchange client creation")
        return None
    
    logger.info(f"Creating Delta Exchange client with testnet={args.testnet}...")
    
    client = DeltaExchangeClient(
        api_key=args.api_key,
        api_secret=args.api_secret,
        testnet=args.testnet,
    )
    
    return client

def train_asp_framework(args, mg, tqe, sp3, trader_agent, adversary_agent, exchange_client=None):
    """Train the ASP framework."""
    logger.info("Training ASP framework...")
    
    # Create ASP framework
    framework = ASPFramework(
        trader_agent=trader_agent,
        adversary_agent=adversary_agent,
        multiverse_generator=mg,
        topo_quantum_encoder=tqe,
        superposition_pool=sp3,
        exchange_client=exchange_client,
        reward_type=RewardType[args.reward_type.upper()],
        steps_per_actor=10,
        device=config.hardware.device,
    )
    
    # Load existing models if available
    if os.path.exists(os.path.join(args.asp_dir, "trader_agent.pt")):
        framework.load_models(args.asp_dir)
    
    # Train framework
    metrics = framework.train(
        num_episodes=args.num_episodes,
        max_steps_per_episode=args.max_steps,
        batch_size=args.batch_size,
        save_interval=100,
        model_dir=args.asp_dir,
    )
    
    # Plot training metrics
    plot_training_metrics(metrics, args.asp_dir)
    
    return framework, metrics

def backtest_asp_framework(args, mg, tqe, sp3, trader_agent, adversary_agent):
    """Backtest the ASP framework."""
    logger.info("Backtesting ASP framework...")
    
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
    
    # Load existing models
    framework.load_models(args.asp_dir)
    
    # Generate market data for backtesting
    num_samples = 100
    seq_len = config.data.history_length
    
    logger.info(f"Generating {num_samples} market scenarios for backtesting...")
    
    # Generate initial market data
    market_data, _ = mg.generate_samples(
        num_samples=num_samples,
        seq_len=seq_len,
    )
    
    # Initialize backtest results
    backtest_results = {
        "pnl": [],
        "returns": [],
        "trades": [],
        "scenarios": [],
    }
    
    # Run backtest
    for i in range(num_samples):
        logger.info(f"Backtesting scenario {i+1}/{num_samples}...")
        
        # Get market data for this scenario
        scenario_data = market_data[i:i+1]
        
        # Initialize trader state
        trader_agent.position = Position.FLAT
        trader_agent.position_size = 0
        trader_agent.entry_price = 0.0
        trader_agent.pnl = 0.0
        trader_agent.returns = []
        trader_agent.trades = []
        
        # Initialize scenario state
        state = framework._preprocess_market_data(scenario_data)
        total_pnl = 0.0
        
        # Run scenario
        for step in range(args.max_steps):
            # Trader selects action
            action = trader_agent.act(state)
            
            # Apply action and get reward
            price = scenario_data[0, 3, -1]  # Close price
            pnl, position = trader_agent.update_position(action, price)
            total_pnl += pnl
            
            # Generate next market data
            next_scenario_data, _ = mg.generate_samples(
                num_samples=1,
                seq_len=seq_len,
                conditioning=scenario_data,
            )
            
            # Update state
            state = framework._preprocess_market_data(next_scenario_data)
            scenario_data = next_scenario_data
        
        # Record results
        backtest_results["pnl"].append(total_pnl)
        backtest_results["returns"].extend(trader_agent.returns)
        backtest_results["trades"].extend(trader_agent.trades)
        backtest_results["scenarios"].append({
            "id": i,
            "pnl": total_pnl,
            "num_trades": len(trader_agent.trades),
        })
    
    # Calculate performance metrics
    performance_metrics = calculate_performance_metrics(backtest_results)
    
    # Plot backtest results
    plot_backtest_results(backtest_results, performance_metrics, args.asp_dir)
    
    return backtest_results, performance_metrics

def live_trade_asp_framework(args, mg, tqe, sp3, trader_agent, adversary_agent, exchange_client):
    """Run live trading using the ASP framework."""
    if exchange_client is None:
        raise ValueError("Exchange client is required for live trading")
    
    logger.info("Starting live trading...")
    
    # Create ASP framework
    framework = ASPFramework(
        trader_agent=trader_agent,
        adversary_agent=adversary_agent,
        multiverse_generator=mg,
        topo_quantum_encoder=tqe,
        superposition_pool=sp3,
        exchange_client=exchange_client,
        reward_type=RewardType[args.reward_type.upper()],
        steps_per_actor=10,
        device=config.hardware.device,
    )
    
    # Load existing models
    framework.load_models(args.asp_dir)
    
    # Run live trading
    framework.live_trade(
        product_id=args.product_id,
        interval_seconds=60,
        max_position_size=args.max_position_size,
        stop_loss_pct=args.stop_loss_pct,
        take_profit_pct=args.take_profit_pct,
        max_runtime_hours=24.0,
    )

def calculate_performance_metrics(backtest_results):
    """Calculate performance metrics from backtest results."""
    pnl = backtest_results["pnl"]
    returns = backtest_results["returns"]
    trades = backtest_results["trades"]
    
    if not pnl or not returns:
        return {
            "total_pnl": 0.0,
            "sharpe_ratio": 0.0,
            "sortino_ratio": 0.0,
            "win_rate": 0.0,
            "max_drawdown": 0.0,
            "avg_trade_pnl": 0.0,
        }
    
    # Calculate total PnL
    total_pnl = sum(pnl)
    
    # Calculate Sharpe ratio
    returns_array = np.array(returns)
    sharpe_ratio = returns_array.mean() / (returns_array.std() + 1e-8)
    
    # Calculate Sortino ratio
    downside_returns = returns_array[returns_array < 0]
    sortino_ratio = returns_array.mean() / (downside_returns.std() + 1e-8)
    
    # Calculate win rate
    winning_trades = [t for t in trades if t.get("pnl", 0) > 0]
    win_rate = len(winning_trades) / max(1, len([t for t in trades if "pnl" in t]))
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumsum(returns_array)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = peak - cumulative_returns
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0
    
    # Calculate average trade PnL
    trade_pnls = [t.get("pnl", 0) for t in trades if "pnl" in t]
    avg_trade_pnl = sum(trade_pnls) / max(1, len(trade_pnls))
    
    return {
        "total_pnl": total_pnl,
        "sharpe_ratio": sharpe_ratio,
        "sortino_ratio": sortino_ratio,
        "win_rate": win_rate,
        "max_drawdown": max_drawdown,
        "avg_trade_pnl": avg_trade_pnl,
    }

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

def plot_backtest_results(backtest_results, performance_metrics, output_dir):
    """Plot backtest results."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create figure for PnL distribution
    plt.figure(figsize=(10, 6))
    plt.hist(backtest_results["pnl"], bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("PnL Distribution")
    plt.xlabel("PnL")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "pnl_distribution.png"))
    plt.close()
    
    # Create figure for returns distribution
    plt.figure(figsize=(10, 6))
    plt.hist(backtest_results["returns"], bins=20)
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title("Returns Distribution")
    plt.xlabel("Return")
    plt.ylabel("Frequency")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "returns_distribution.png"))
    plt.close()
    
    # Create figure for cumulative PnL
    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(backtest_results["returns"]))
    plt.title("Cumulative PnL")
    plt.xlabel("Trade")
    plt.ylabel("Cumulative PnL")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "cumulative_pnl.png"))
    plt.close()
    
    # Create figure for performance metrics
    plt.figure(figsize=(10, 6))
    metrics = [
        performance_metrics["total_pnl"],
        performance_metrics["sharpe_ratio"],
        performance_metrics["sortino_ratio"],
        performance_metrics["win_rate"] * 100,
        performance_metrics["avg_trade_pnl"]
    ]
    labels = [
        "Total PnL",
        "Sharpe Ratio",
        "Sortino Ratio",
        "Win Rate (%)",
        "Avg Trade PnL"
    ]
    plt.bar(labels, metrics)
    plt.title("Performance Metrics")
    plt.xticks(rotation=45)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "performance_metrics.png"))
    plt.close()
    
    # Save metrics to JSON
    with open(os.path.join(output_dir, "performance_metrics.json"), "w") as f:
        json.dump(performance_metrics, f, indent=2)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.asp_dir, exist_ok=True)
    
    # Load MQTM models
    mg, tqe, sp3 = load_models(args.models_dir)
    
    # Determine state dimension
    state_dim = tqe.total_features_dim
    
    # Create agents
    trader_agent, adversary_agent = create_agents(args, state_dim)
    
    # Create exchange client if needed
    exchange_client = None
    if args.mode == "live":
        exchange_client = create_exchange_client(args)
    
    # Run in specified mode
    if args.mode == "train":
        framework, metrics = train_asp_framework(
            args, mg, tqe, sp3, trader_agent, adversary_agent, exchange_client
        )
    elif args.mode == "backtest":
        backtest_results, performance_metrics = backtest_asp_framework(
            args, mg, tqe, sp3, trader_agent, adversary_agent
        )
    elif args.mode == "live":
        if exchange_client is None:
            logger.error("Exchange client is required for live trading")
            return
        
        live_trade_asp_framework(
            args, mg, tqe, sp3, trader_agent, adversary_agent, exchange_client
        )
    
    logger.info(f"ASP framework {args.mode} completed")

if __name__ == "__main__":
    main()
