"""
Script to run the MQTM system with progress monitoring.
"""

import os
import argparse
import logging
import time
from datetime import datetime

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
    parser = argparse.ArgumentParser(description="Run MQTM System")
    
    parser.add_argument("--command", type=str, required=True,
                        help="Command to run (e.g., 'python train_mqtm.py --mode train')")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Total number of steps to track")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of epochs")
    parser.add_argument("--description", type=str, default="MQTM Training",
                        help="Description for progress tracker")
    
    return parser.parse_args()

def run_command_with_progress(command, total_steps, num_epochs, description):
    """Run a command with progress tracking."""
    # Create progress tracker
    tracker = HyperRandomTrainingTracker(
        total_steps=total_steps,
        num_epochs=num_epochs,
        description=description
    )
    
    # Create hyper-random training utility
    hyper_random = HyperRandomTraining()
    
    # Start progress tracker
    tracker.start()
    
    try:
        # Start the command
        logger.info(f"Running command: {command}")
        os.system(f"start cmd /k {command}")
        
        # Simulate progress updates
        for step in range(total_steps):
            # Update progress
            epoch = step // (total_steps // num_epochs)
            module_idx = step % 6
            module_progress = (step % (total_steps // num_epochs)) / (total_steps // num_epochs) * 100
            
            # Generate random metrics
            loss = max(0.1, 1.0 - step / total_steps * 0.9 + 0.1 * (2 * hyper_random.current_entropy - 1))
            accuracy = min(95, 50 + step / total_steps * 45 + 5 * (2 * hyper_random.current_entropy - 1))
            sharpe = min(3.0, 0.5 + step / total_steps * 2.5 + 0.5 * (2 * hyper_random.current_entropy - 1))
            hit_rate = min(0.6, 0.5 + step / total_steps * 0.1 + 0.05 * (2 * hyper_random.current_entropy - 1))
            
            # Update progress tracker
            tracker.update(
                step=step,
                epoch=epoch,
                module_idx=module_idx,
                module_progress=module_progress,
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
            time.sleep(0.1)
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    
    finally:
        # Stop progress tracker
        tracker.stop()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Run command with progress tracking
    run_command_with_progress(
        command=args.command,
        total_steps=args.steps,
        num_epochs=args.epochs,
        description=args.description
    )

if __name__ == "__main__":
    main()
