"""
Script to test the efficient data loading pipeline.
"""

import os
import argparse
import logging
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from mqtm.config import config
from mqtm.data_engineering.data_fetcher import BinanceDataFetcher
from mqtm.data_engineering.data_processor import OHLCVProcessor
from mqtm.data_engineering.efficient_dataloader import (
    create_efficient_dataloaders, PrefetchDataset, StreamingOHLCVDataset, BalancedBatchSampler
)
from mqtm.utils.performance_profiling import Timer, DataLoaderProfiler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test Efficient DataLoader")
    
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--download", action="store_true",
                        help="Download data from Binance")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days of data to download")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for data loading")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of worker processes")
    parser.add_argument("--output_dir", type=str, default="dataloader_test",
                        help="Directory to save test results")
    
    return parser.parse_args()

def download_data(symbols, days):
    """Download historical OHLCV data."""
    logger.info(f"Downloading {days} days of data for {symbols}...")
    
    # Create data fetcher
    fetcher = BinanceDataFetcher(symbols=symbols)
    
    # Download data
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        save=True
    )
    
    return data

def process_data(symbols):
    """Process OHLCV data."""
    logger.info(f"Processing data for {symbols}...")
    
    # Create data processor
    processor = OHLCVProcessor(symbols=symbols)
    
    # Process data
    data = processor.process_all_symbols()
    
    return data

def test_dataloader_configurations(symbols, batch_size, num_workers, output_dir):
    """Test different dataloader configurations."""
    logger.info("Testing different dataloader configurations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define configurations to test
    configs = [
        {"name": "Standard", "prefetch": False, "balanced_batches": False, "streaming": False},
        {"name": "Prefetch", "prefetch": True, "balanced_batches": False, "streaming": False},
        {"name": "Balanced", "prefetch": False, "balanced_batches": True, "streaming": False},
        {"name": "Prefetch+Balanced", "prefetch": True, "balanced_batches": True, "streaming": False},
        {"name": "Streaming", "prefetch": False, "balanced_batches": False, "streaming": True},
    ]
    
    # Test each configuration
    results = {}
    
    for config_dict in configs:
        name = config_dict["name"]
        logger.info(f"Testing {name} configuration...")
        
        # Create data loaders
        with Timer(f"{name} dataloader creation"):
            train_loader, val_loader = create_efficient_dataloaders(
                symbols=symbols,
                batch_size=batch_size,
                num_workers=num_workers,
                prefetch=config_dict["prefetch"],
                balanced_batches=config_dict["balanced_batches"],
                streaming=config_dict["streaming"],
            )
        
        # Profile data loaders
        train_stats = DataLoaderProfiler.profile_dataloader(
            dataloader=train_loader,
            num_batches=10,
        )
        
        val_stats = DataLoaderProfiler.profile_dataloader(
            dataloader=val_loader,
            num_batches=10,
        )
        
        # Store results
        results[name] = {
            "train": train_stats,
            "val": val_stats,
        }
        
        # Check label distribution if balanced batches
        if config_dict["balanced_batches"]:
            logger.info("Checking label distribution...")
            
            # Count labels in batches
            label_counts = {}
            
            for i, (_, y, _) in enumerate(train_loader):
                if i >= 10:
                    break
                
                for label in y:
                    label_item = label.item()
                    if label_item not in label_counts:
                        label_counts[label_item] = 0
                    label_counts[label_item] += 1
            
            logger.info(f"Label counts: {label_counts}")
    
    # Plot results
    plot_dataloader_results(results, output_dir)
    
    return results

def test_memory_usage(symbols, batch_size, num_workers, output_dir):
    """Test memory usage of different dataloader configurations."""
    logger.info("Testing memory usage of different dataloader configurations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define configurations to test
    configs = [
        {"name": "Standard", "prefetch": False, "balanced_batches": False, "streaming": False},
        {"name": "Prefetch", "prefetch": True, "balanced_batches": False, "streaming": False},
        {"name": "Streaming", "prefetch": False, "balanced_batches": False, "streaming": True},
    ]
    
    # Test each configuration
    results = {}
    
    for config_dict in configs:
        name = config_dict["name"]
        logger.info(f"Testing {name} configuration...")
        
        # Create data loaders
        train_loader, _ = create_efficient_dataloaders(
            symbols=symbols,
            batch_size=batch_size,
            num_workers=num_workers,
            prefetch=config_dict["prefetch"],
            balanced_batches=config_dict["balanced_batches"],
            streaming=config_dict["streaming"],
        )
        
        # Measure memory usage
        import psutil
        import gc
        
        # Force garbage collection
        gc.collect()
        
        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        
        # Iterate through batches
        memory_usage = []
        
        for i, _ in enumerate(train_loader):
            if i >= 10:
                break
            
            # Measure memory usage
            memory = process.memory_info().rss / (1024 * 1024)  # MB
            memory_usage.append(memory)
        
        # Compute statistics
        memory_usage = np.array(memory_usage)
        
        results[name] = {
            "initial_memory": initial_memory,
            "min_memory": np.min(memory_usage),
            "max_memory": np.max(memory_usage),
            "mean_memory": np.mean(memory_usage),
            "std_memory": np.std(memory_usage),
        }
        
        logger.info(f"{name} memory usage (MB): {results[name]}")
        
        # Force garbage collection
        gc.collect()
    
    # Plot results
    plot_memory_results(results, output_dir)
    
    return results

def test_throughput(symbols, batch_size, num_workers, output_dir):
    """Test throughput of different dataloader configurations."""
    logger.info("Testing throughput of different dataloader configurations...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Define configurations to test
    configs = [
        {"name": "Standard", "prefetch": False, "balanced_batches": False, "streaming": False},
        {"name": "Prefetch", "prefetch": True, "balanced_batches": False, "streaming": False},
        {"name": "Balanced", "prefetch": False, "balanced_batches": True, "streaming": False},
        {"name": "Prefetch+Balanced", "prefetch": True, "balanced_batches": True, "streaming": False},
        {"name": "Streaming", "prefetch": False, "balanced_batches": False, "streaming": True},
    ]
    
    # Define batch sizes to test
    batch_sizes = [8, 16, 32, 64, 128]
    
    # Test each configuration
    results = {}
    
    for config_dict in configs:
        name = config_dict["name"]
        logger.info(f"Testing {name} configuration...")
        
        batch_results = {}
        
        for bs in batch_sizes:
            logger.info(f"Testing batch size {bs}...")
            
            # Create data loaders
            train_loader, _ = create_efficient_dataloaders(
                symbols=symbols,
                batch_size=bs,
                num_workers=num_workers,
                prefetch=config_dict["prefetch"],
                balanced_batches=config_dict["balanced_batches"],
                streaming=config_dict["streaming"],
            )
            
            # Measure throughput
            start_time = time.time()
            num_samples = 0
            
            for i, (X, _, _) in enumerate(train_loader):
                if i >= 100:
                    break
                
                num_samples += X.shape[0]
            
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Compute throughput
            throughput = num_samples / elapsed_time
            
            batch_results[bs] = throughput
            
            logger.info(f"Batch size {bs}: {throughput:.2f} samples/second")
        
        results[name] = batch_results
    
    # Plot results
    plot_throughput_results(results, output_dir)
    
    return results

def plot_dataloader_results(results, output_dir):
    """Plot dataloader profiling results."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot batches per second
    x = np.arange(len(results))
    width = 0.35
    
    train_throughput = [results[name]["train"]["batches_per_second"] for name in results]
    val_throughput = [results[name]["val"]["batches_per_second"] for name in results]
    
    ax.bar(x - width/2, train_throughput, width, label="Train")
    ax.bar(x + width/2, val_throughput, width, label="Validation")
    
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Batches per second")
    ax.set_title("DataLoader Throughput")
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataloader_throughput.png"))
    
    # Create figure for batch time
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot batch time
    train_time = [results[name]["train"]["avg_batch_time"] for name in results]
    val_time = [results[name]["val"]["avg_batch_time"] for name in results]
    
    ax.bar(x - width/2, train_time, width, label="Train")
    ax.bar(x + width/2, val_time, width, label="Validation")
    
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Average batch time (seconds)")
    ax.set_title("DataLoader Batch Time")
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataloader_batch_time.png"))

def plot_memory_results(results, output_dir):
    """Plot memory usage results."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot memory usage
    x = np.arange(len(results))
    width = 0.35
    
    initial_memory = [results[name]["initial_memory"] for name in results]
    max_memory = [results[name]["max_memory"] for name in results]
    
    ax.bar(x - width/2, initial_memory, width, label="Initial")
    ax.bar(x + width/2, max_memory, width, label="Maximum")
    
    ax.set_xlabel("Configuration")
    ax.set_ylabel("Memory usage (MB)")
    ax.set_title("DataLoader Memory Usage")
    ax.set_xticks(x)
    ax.set_xticklabels(list(results.keys()))
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataloader_memory.png"))

def plot_throughput_results(results, output_dir):
    """Plot throughput results."""
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot throughput
    for name, batch_results in results.items():
        batch_sizes = list(batch_results.keys())
        throughput = list(batch_results.values())
        
        ax.plot(batch_sizes, throughput, marker='o', label=name)
    
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Throughput (samples/second)")
    ax.set_title("DataLoader Throughput vs. Batch Size")
    ax.legend()
    ax.grid(True)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "dataloader_throughput_vs_batch_size.png"))

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Download data if requested
    if args.download:
        download_data(args.symbols, args.days)
    
    # Process data
    process_data(args.symbols)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Test dataloader configurations
    test_dataloader_configurations(
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    
    # Test memory usage
    test_memory_usage(
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    
    # Test throughput
    test_throughput(
        symbols=args.symbols,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        output_dir=args.output_dir,
    )
    
    logger.info(f"DataLoader testing complete. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
