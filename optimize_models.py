"""
Script to apply memory optimization to MQTM models.
"""

import os
import argparse
import logging
import torch

from mqtm.config import config
from mqtm.utils.memory_optimization import MemoryOptimizer, MixedPrecisionTrainer, DynamicBatchSizer
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
    parser = argparse.ArgumentParser(description="Optimize MQTM Models")
    
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--output_dir", type=str, default="optimized_models",
                        help="Directory to save optimized models")
    parser.add_argument("--batch_size", type=int, default=config.hardware.batch_size,
                        help="Batch size for testing")
    
    return parser.parse_args()

def optimize_multiverse_generator(mg, output_dir):
    """Optimize Multiverse Generator."""
    logger.info("Optimizing Multiverse Generator...")
    
    # Log initial memory usage
    MemoryOptimizer.log_memory_usage("Before optimization: ")
    
    # Optimize diffusion model
    if mg.diffusion_model and mg.diffusion_model.model:
        MemoryOptimizer.optimize_model_memory(mg.diffusion_model.model)
    
    # Clear GPU memory
    MemoryOptimizer.clear_gpu_memory()
    
    # Log memory usage after optimization
    MemoryOptimizer.log_memory_usage("After optimization: ")
    
    # Save optimized model
    mg_dir = os.path.join(output_dir, "multiverse_generator")
    os.makedirs(mg_dir, exist_ok=True)
    mg.model_dir = mg_dir
    mg.save()
    
    logger.info(f"Saved optimized Multiverse Generator to {mg_dir}")

def optimize_topo_quantum_encoder(tqe, output_dir):
    """Optimize Topo-Quantum Encoder."""
    logger.info("Optimizing Topo-Quantum Encoder...")
    
    # Log initial memory usage
    MemoryOptimizer.log_memory_usage("Before optimization: ")
    
    # Optimize model
    MemoryOptimizer.optimize_model_memory(tqe)
    
    # Clear GPU memory
    MemoryOptimizer.clear_gpu_memory()
    
    # Log memory usage after optimization
    MemoryOptimizer.log_memory_usage("After optimization: ")
    
    # Save optimized model
    tqe_path = os.path.join(output_dir, "tqe.pt")
    tqe.save(tqe_path)
    
    logger.info(f"Saved optimized Topo-Quantum Encoder to {tqe_path}")

def optimize_superposition_pool(sp3, output_dir):
    """Optimize Superposition Pool."""
    logger.info("Optimizing Superposition Pool...")
    
    # Log initial memory usage
    MemoryOptimizer.log_memory_usage("Before optimization: ")
    
    # Optimize model
    MemoryOptimizer.optimize_model_memory(sp3)
    
    # Optimize each head
    for i, head in enumerate(sp3.heads):
        logger.info(f"Optimizing head {i}...")
        MemoryOptimizer.optimize_model_memory(head)
    
    # Clear GPU memory
    MemoryOptimizer.clear_gpu_memory()
    
    # Log memory usage after optimization
    MemoryOptimizer.log_memory_usage("After optimization: ")
    
    # Save optimized model
    sp3_path = os.path.join(output_dir, "sp3.pt")
    sp3.save(sp3_path)
    
    logger.info(f"Saved optimized Superposition Pool to {sp3_path}")

def test_memory_usage(mg, tqe, sp3, batch_size):
    """Test memory usage with random inputs."""
    logger.info(f"Testing memory usage with batch size {batch_size}...")
    
    # Create dynamic batch sizer
    batch_sizer = DynamicBatchSizer(
        initial_batch_size=batch_size,
        min_batch_size=4,
        max_batch_size=128,
    )
    
    # Log initial memory usage
    MemoryOptimizer.log_memory_usage("Initial: ")
    
    # Create random inputs
    seq_len = config.data.history_length
    
    # Test Multiverse Generator
    if mg.diffusion_model and mg.diffusion_model.model:
        logger.info("Testing Multiverse Generator...")
        
        # Update batch size
        batch_size = batch_sizer.update_batch_size()
        
        # Create random inputs
        x = torch.randn(batch_size, 5, seq_len, device=config.hardware.device)
        timesteps = torch.randint(0, mg.diffusion_model.num_timesteps, (batch_size,), device=config.hardware.device)
        causal_graph = torch.randn(batch_size, mg.num_latent_factors, mg.num_latent_factors, device=config.hardware.device)
        
        # Forward pass
        with torch.no_grad():
            mg.diffusion_model.model(x, timesteps, causal_graph)
        
        # Log memory usage
        MemoryOptimizer.log_memory_usage("After MG forward pass: ")
    
    # Test Topo-Quantum Encoder
    logger.info("Testing Topo-Quantum Encoder...")
    
    # Update batch size
    batch_size = batch_sizer.update_batch_size()
    
    # Create random inputs
    x = torch.randn(batch_size, 5, seq_len, device=config.hardware.device)
    
    # Forward pass
    with torch.no_grad():
        features = tqe(x)
    
    # Log memory usage
    MemoryOptimizer.log_memory_usage("After TQE forward pass: ")
    
    # Test Superposition Pool
    logger.info("Testing Superposition Pool...")
    
    # Update batch size
    batch_size = batch_sizer.update_batch_size()
    
    # Create random inputs
    regime = torch.rand(batch_size, 2, device=config.hardware.device)
    
    # Forward pass
    with torch.no_grad():
        outputs = sp3(features, regime)
    
    # Log memory usage
    MemoryOptimizer.log_memory_usage("After SP3 forward pass: ")
    
    # Clear GPU memory
    MemoryOptimizer.clear_gpu_memory()
    
    # Log final memory usage
    MemoryOptimizer.log_memory_usage("Final: ")
    
    # Return optimal batch size
    return batch_sizer.batch_size

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load models
    logger.info(f"Loading models from {args.models_dir}...")
    
    # Create models
    mg = MultiverseGenerator(
        model_dir=os.path.join(args.models_dir, "multiverse_generator"),
    )
    
    tqe = TopoQuantumEncoder()
    
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
    
    # Optimize models
    optimize_multiverse_generator(mg, args.output_dir)
    optimize_topo_quantum_encoder(tqe, args.output_dir)
    optimize_superposition_pool(sp3, args.output_dir)
    
    # Test memory usage
    optimal_batch_size = test_memory_usage(mg, tqe, sp3, args.batch_size)
    
    logger.info(f"Optimization complete. Optimal batch size: {optimal_batch_size}")
    logger.info(f"Optimized models saved to {args.output_dir}")

if __name__ == "__main__":
    main()
