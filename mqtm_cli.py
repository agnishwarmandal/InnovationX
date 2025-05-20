"""
Command-line interface for the MQTM system.
"""

import os
import sys
import argparse
import logging
import subprocess
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("mqtm_cli.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MQTM Command-Line Interface")
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Data command
    data_parser = subparsers.add_parser("data", help="Download and process data")
    data_parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                            help="Trading symbols to use")
    data_parser.add_argument("--days", type=int, default=30,
                            help="Number of days of data to download")
    
    # Train command
    train_parser = subparsers.add_parser("train", help="Train MQTM models")
    train_parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                             help="Trading symbols to use")
    train_parser.add_argument("--batch_size", type=int, default=32,
                             help="Batch size for training")
    train_parser.add_argument("--epochs", type=int, default=50,
                             help="Number of training epochs")
    train_parser.add_argument("--learning_rate", type=float, default=1e-4,
                             help="Learning rate")
    train_parser.add_argument("--output_dir", type=str, default="models",
                             help="Directory to save models")
    
    # Generate command
    generate_parser = subparsers.add_parser("generate", help="Generate synthetic data")
    generate_parser.add_argument("--num_samples", type=int, default=1000,
                                help="Number of samples to generate")
    generate_parser.add_argument("--sigma_multiplier", type=float, default=1.5,
                                help="Sigma multiplier for generation")
    generate_parser.add_argument("--output_dir", type=str, default="samples",
                                help="Directory to save samples")
    
    # Visualize command
    visualize_parser = subparsers.add_parser("visualize", help="Visualize MQTM components")
    visualize_parser.add_argument("--component", type=str, required=True,
                                 choices=["tqe", "sp3", "mg"],
                                 help="Component to visualize")
    visualize_parser.add_argument("--symbol", type=str, default="BTCUSDT",
                                 help="Trading symbol to use")
    visualize_parser.add_argument("--output_dir", type=str, default="visualizations",
                                 help="Directory to save visualizations")
    
    # Trade command
    trade_parser = subparsers.add_parser("trade", help="Run trading example")
    trade_parser.add_argument("--symbol", type=str, default="BTCUSDT",
                             help="Trading symbol to use")
    trade_parser.add_argument("--mode", type=str, default="backtest",
                             choices=["backtest", "paper", "live"],
                             help="Trading mode: backtest, paper, or live")
    trade_parser.add_argument("--days", type=int, default=7,
                             help="Number of days for backtesting")
    trade_parser.add_argument("--capital", type=float, default=10000.0,
                             help="Initial capital")
    trade_parser.add_argument("--risk_per_trade", type=float, default=0.01,
                             help="Risk per trade (fraction of capital)")
    
    # Pipeline command
    pipeline_parser = subparsers.add_parser("pipeline", help="Run complete MQTM pipeline")
    pipeline_parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                                help="Trading symbols to use")
    pipeline_parser.add_argument("--download_data", action="store_true",
                                help="Download data from Binance")
    pipeline_parser.add_argument("--days", type=int, default=30,
                                help="Number of days of data to download")
    pipeline_parser.add_argument("--batch_size", type=int, default=32,
                                help="Batch size for training")
    pipeline_parser.add_argument("--epochs", type=int, default=10,
                                help="Number of training epochs")
    pipeline_parser.add_argument("--output_dir", type=str, default="pipeline_output",
                                help="Directory to save output")
    
    return parser.parse_args()

def run_data_command(args):
    """Run data download and processing command."""
    logger.info("Running data command...")
    
    # Build command
    cmd = [
        "python", "train_mqtm.py",
        "--mode", "train",
        "--download_data",
        "--days", str(args.days),
        "--symbols"
    ] + args.symbols
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_train_command(args):
    """Run training command."""
    logger.info("Running train command...")
    
    # Build command
    cmd = [
        "python", "train_mqtm.py",
        "--mode", "train",
        "--symbols"
    ] + args.symbols + [
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--learning_rate", str(args.learning_rate),
        "--output_dir", args.output_dir
    ]
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_generate_command(args):
    """Run generation command."""
    logger.info("Running generate command...")
    
    # Build command
    cmd = [
        "python", "train_mqtm.py",
        "--mode", "generate",
        "--num_samples", str(args.num_samples),
        "--sigma_multiplier", str(args.sigma_multiplier),
        "--output_dir", args.output_dir
    ]
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_visualize_command(args):
    """Run visualization command."""
    logger.info("Running visualize command...")
    
    # Build command based on component
    if args.component == "tqe":
        cmd = [
            "python", "visualize_tqe.py",
            "--symbol", args.symbol,
            "--output_dir", args.output_dir
        ]
    elif args.component == "sp3":
        cmd = [
            "python", "visualize_sp3.py",
            "--output_dir", args.output_dir
        ]
    elif args.component == "mg":
        cmd = [
            "python", "visualize_mg.py",
            "--symbol", args.symbol,
            "--output_dir", args.output_dir
        ]
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_trade_command(args):
    """Run trading command."""
    logger.info("Running trade command...")
    
    # Build command
    cmd = [
        "python", "trading_example.py",
        "--symbol", args.symbol,
        "--mode", args.mode,
        "--days", str(args.days),
        "--capital", str(args.capital),
        "--risk_per_trade", str(args.risk_per_trade)
    ]
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def run_pipeline_command(args):
    """Run pipeline command."""
    logger.info("Running pipeline command...")
    
    # Build command
    cmd = [
        "python", "run_pipeline.py",
        "--symbols"
    ] + args.symbols + [
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs)
    ]
    
    if args.download_data:
        cmd.append("--download_data")
        cmd.extend(["--days", str(args.days)])
    
    # Run command
    logger.info(f"Executing command: {' '.join(cmd)}")
    subprocess.run(cmd)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Check if a command was provided
    if args.command is None:
        print("Error: No command specified.")
        print("Run 'python mqtm_cli.py --help' for usage information.")
        sys.exit(1)
    
    # Run the appropriate command
    if args.command == "data":
        run_data_command(args)
    elif args.command == "train":
        run_train_command(args)
    elif args.command == "generate":
        run_generate_command(args)
    elif args.command == "visualize":
        run_visualize_command(args)
    elif args.command == "trade":
        run_trade_command(args)
    elif args.command == "pipeline":
        run_pipeline_command(args)
    else:
        print(f"Error: Unknown command '{args.command}'.")
        sys.exit(1)
    
    logger.info("Command completed successfully.")

if __name__ == "__main__":
    main()
