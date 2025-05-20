"""
Script to run all MQTM enhancements in sequence.
"""

import os
import argparse
import logging
import subprocess
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("enhancements.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Run MQTM Enhancements")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--output_dir", type=str, default="enhanced_models",
                        help="Directory to save enhanced models")
    parser.add_argument("--symbols", type=str, nargs="+", default=["BTCUSDT", "ETHUSDT"],
                        help="Trading symbols to use")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for testing")
    parser.add_argument("--skip_optimization", action="store_true",
                        help="Skip memory optimization")
    parser.add_argument("--skip_profiling", action="store_true",
                        help="Skip performance profiling")
    parser.add_argument("--skip_dataloader", action="store_true",
                        help="Skip dataloader testing")
    parser.add_argument("--skip_mg", action="store_true",
                        help="Skip Multiverse Generator enhancement")
    parser.add_argument("--skip_tqe", action="store_true",
                        help="Skip Topo-Quantum Encoder enhancement")
    parser.add_argument("--skip_sp3", action="store_true",
                        help="Skip Superposition Pool enhancement")
    
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and log output."""
    logger.info(f"Running {description}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    start_time = time.time()
    
    try:
        # Run command
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            universal_newlines=True,
            bufsize=1
        )
        
        # Read output
        for line in iter(process.stdout.readline, ""):
            logger.info(line.strip())
        
        # Wait for process to complete
        return_code = process.wait()
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        if return_code == 0:
            logger.info(f"{description} completed successfully in {elapsed_time:.2f} seconds")
            return True
        else:
            logger.error(f"{description} failed with return code {return_code}")
            return False
    
    except Exception as e:
        logger.error(f"{description} failed with exception: {e}")
        return False

def run_memory_optimization(args):
    """Run memory optimization."""
    if args.skip_optimization:
        logger.info("Skipping memory optimization")
        return True
    
    # Build command
    cmd = [
        "python", "optimize_models.py",
        "--models_dir", args.models_dir,
        "--output_dir", args.output_dir,
        "--batch_size", str(args.batch_size),
    ]
    
    # Run command
    return run_command(cmd, "Memory Optimization")

def run_performance_profiling(args):
    """Run performance profiling."""
    if args.skip_profiling:
        logger.info("Skipping performance profiling")
        return True
    
    # Build command
    cmd = [
        "python", "profile_models.py",
        "--models_dir", args.models_dir,
        "--symbols"
    ] + args.symbols + [
        "--batch_size", str(args.batch_size),
        "--num_batches", "10",
        "--profile_dir", "profiles",
    ]
    
    # Run command
    return run_command(cmd, "Performance Profiling")

def run_dataloader_testing(args):
    """Run dataloader testing."""
    if args.skip_dataloader:
        logger.info("Skipping dataloader testing")
        return True
    
    # Build command
    cmd = [
        "python", "test_dataloader.py",
        "--symbols"
    ] + args.symbols + [
        "--batch_size", str(args.batch_size),
        "--num_workers", "4",
        "--output_dir", "dataloader_test",
    ]
    
    # Run command
    return run_command(cmd, "DataLoader Testing")

def run_multiverse_generator_enhancement(args):
    """Run Multiverse Generator enhancement."""
    if args.skip_mg:
        logger.info("Skipping Multiverse Generator enhancement")
        return True
    
    # Build command
    cmd = [
        "python", "enhance_multiverse_generator.py",
        "--model_dir", os.path.join(args.models_dir, "multiverse_generator"),
        "--output_dir", os.path.join(args.output_dir, "multiverse_generator"),
        "--hidden_dim", "128",
        "--num_layers", "4",
        "--num_heads", "4",
        "--head_dim", "32",
        "--dropout", "0.1",
        "--num_timesteps", "100",
        "--noise_schedule", "cosine",
        "--batch_size", str(args.batch_size),
        "--num_samples", "10",
    ]
    
    # Run command
    return run_command(cmd, "Multiverse Generator Enhancement")

def run_topo_quantum_encoder_enhancement(args):
    """Run Topo-Quantum Encoder enhancement."""
    if args.skip_tqe:
        logger.info("Skipping Topo-Quantum Encoder enhancement")
        return True
    
    # Build command
    cmd = [
        "python", "enhance_topo_quantum_encoder.py",
        "--model_path", os.path.join(args.models_dir, "tqe.pt"),
        "--output_path", os.path.join(args.output_dir, "tqe.pt"),
        "--use_advanced_persistence",
        "--use_advanced_wavelets",
        "--max_homology_dim", "2",
        "--wavelet_families", "morl", "mexh", "gaus1", "cgau1",
        "--batch_size", str(args.batch_size),
    ]
    
    # Run command
    return run_command(cmd, "Topo-Quantum Encoder Enhancement")

def run_superposition_pool_enhancement(args):
    """Run Superposition Pool enhancement."""
    if args.skip_sp3:
        logger.info("Skipping Superposition Pool enhancement")
        return True
    
    # Build command
    cmd = [
        "python", "enhance_superposition_pool.py",
        "--model_path", os.path.join(args.models_dir, "sp3.pt"),
        "--output_path", os.path.join(args.output_dir, "sp3.pt"),
        "--use_advanced_unitary",
        "--use_entanglement",
        "--unitary_method", "cayley",
        "--batch_size", str(args.batch_size),
        "--input_dim", "112",
    ]
    
    # Run command
    return run_command(cmd, "Superposition Pool Enhancement")

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create profiles directory
    os.makedirs("profiles", exist_ok=True)
    
    # Create visualizations directory
    os.makedirs("visualizations", exist_ok=True)
    
    # Run memory optimization
    if not run_memory_optimization(args):
        logger.warning("Memory optimization failed, continuing with other enhancements")
    
    # Run performance profiling
    if not run_performance_profiling(args):
        logger.warning("Performance profiling failed, continuing with other enhancements")
    
    # Run dataloader testing
    if not run_dataloader_testing(args):
        logger.warning("DataLoader testing failed, continuing with other enhancements")
    
    # Run Multiverse Generator enhancement
    if not run_multiverse_generator_enhancement(args):
        logger.warning("Multiverse Generator enhancement failed, continuing with other enhancements")
    
    # Run Topo-Quantum Encoder enhancement
    if not run_topo_quantum_encoder_enhancement(args):
        logger.warning("Topo-Quantum Encoder enhancement failed, continuing with other enhancements")
    
    # Run Superposition Pool enhancement
    if not run_superposition_pool_enhancement(args):
        logger.warning("Superposition Pool enhancement failed")
    
    logger.info("All enhancements completed")

if __name__ == "__main__":
    main()
