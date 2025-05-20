"""
Demo script for the MQTM system.
"""

import os
import argparse
import logging
import time
import random
from datetime import datetime

import numpy as np
import torch

from mqtm.config import config
from mqtm.monitoring.progress_tracker import HyperRandomTrainingTracker
from mqtm.utils.hyper_random import HyperRandomTraining

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MQTM Demo")
    
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps to simulate")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs to simulate")
    parser.add_argument("--speed", type=float, default=0.01,
                        help="Speed of simulation (seconds per step)")
    
    return parser.parse_args()

def simulate_multiverse_generator(tracker, hyper_random, steps, epochs, speed):
    """Simulate Multiverse Generator training."""
    logger.info("Simulating Multiverse Generator training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=0,  # Multiverse Generator
            module_progress=progress,
            metrics={
                "loss": loss,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(1.0 - loss)
        
        # Sleep to simulate work
        time.sleep(speed)

def simulate_topo_quantum_encoder(tracker, hyper_random, steps, epochs, speed):
    """Simulate Topo-Quantum Encoder training."""
    logger.info("Simulating Topo-Quantum Encoder training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        accuracy = min(95, 50 + step / steps * 45 + 5 * (2 * hyper_random.current_entropy - 1))
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=1,  # Topo-Quantum Encoder
            module_progress=progress,
            metrics={
                "loss": loss,
                "accuracy": accuracy,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(accuracy)
        
        # Sleep to simulate work
        time.sleep(speed)

def simulate_superposition_pool(tracker, hyper_random, steps, epochs, speed):
    """Simulate Superposition Pool training."""
    logger.info("Simulating Superposition Pool training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        accuracy = min(95, 50 + step / steps * 45 + 5 * (2 * hyper_random.current_entropy - 1))
        sharpe = min(3.0, 0.5 + step / steps * 2.5 + 0.5 * (2 * hyper_random.current_entropy - 1))
        hit_rate = min(0.6, 0.5 + step / steps * 0.1 + 0.05 * (2 * hyper_random.current_entropy - 1))
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=2,  # Superposition Pool
            module_progress=progress,
            metrics={
                "loss": loss,
                "accuracy": accuracy,
                "sharpe": sharpe,
                "hit_rate": hit_rate,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(accuracy)
        
        # Sleep to simulate work
        time.sleep(speed)

def simulate_adversarial_self_play(tracker, hyper_random, steps, epochs, speed):
    """Simulate Adversarial Self-Play training."""
    logger.info("Simulating Adversarial Self-Play training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        reward = min(10.0, 0.0 + step / steps * 10.0 + 2.0 * (2 * hyper_random.current_entropy - 1))
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=3,  # Adversarial Self-Play
            module_progress=progress,
            metrics={
                "loss": loss,
                "reward": reward,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(reward / 10.0 * 100.0)
        
        # Sleep to simulate work
        time.sleep(speed)

def simulate_meta_gradient_introspection(tracker, hyper_random, steps, epochs, speed):
    """Simulate Meta-Gradient Introspection training."""
    logger.info("Simulating Meta-Gradient Introspection training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        learning_rate = max(1e-6, 1e-4 * (1.0 - step / steps * 0.9))
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=4,  # Meta-Gradient Introspection
            module_progress=progress,
            metrics={
                "loss": loss,
                "learning_rate": learning_rate,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(1.0 - loss)
        
        # Sleep to simulate work
        time.sleep(speed)

def simulate_bayesian_online_mixture(tracker, hyper_random, steps, epochs, speed):
    """Simulate Bayesian Online Mixture training."""
    logger.info("Simulating Bayesian Online Mixture training...")
    
    # Simulate training
    for step in range(steps):
        # Calculate epoch and progress
        epoch = step // (steps // epochs)
        progress = (step % (steps // epochs)) / (steps // epochs) * 100
        
        # Generate random metrics
        loss = max(0.1, 1.0 - step / steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
        entropy = max(0.1, 1.0 - step / steps * 0.5)
        
        # Update progress tracker
        tracker.update(
            step=step,
            epoch=epoch,
            module_idx=5,  # Bayesian Online Mixture
            module_progress=progress,
            metrics={
                "loss": loss,
                "entropy": entropy,
            },
            randomness_stats=hyper_random.get_randomness_stats(),
        )
        
        # Update hyper-random state
        if step % 10 == 0:
            hyper_random.update_randomness_state(1.0 - loss)
        
        # Sleep to simulate work
        time.sleep(speed)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Create hyper-random training utility
    hyper_random = HyperRandomTraining()
    
    # Create progress tracker
    tracker = HyperRandomTrainingTracker(
        total_steps=args.steps,
        num_epochs=args.epochs,
        description="MQTM Demo"
    )
    
    # Start progress tracker
    tracker.start()
    
    try:
        # Simulate each module
        modules = [
            simulate_multiverse_generator,
            simulate_topo_quantum_encoder,
            simulate_superposition_pool,
            simulate_adversarial_self_play,
            simulate_meta_gradient_introspection,
            simulate_bayesian_online_mixture,
        ]
        
        # Choose a random module to simulate
        module_idx = random.randint(0, len(modules) - 1)
        modules[module_idx](tracker, hyper_random, args.steps, args.epochs, args.speed)
    
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user.")
    
    finally:
        # Stop progress tracker
        tracker.stop()

if __name__ == "__main__":
    main()
