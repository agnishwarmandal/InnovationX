# MQTM System Training Guide

This guide provides detailed instructions for training the Multiverse Quantum-Topological Meta-Learning (MQTM) system using the new training scripts.

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Data Preparation](#data-preparation)
3. [Training Pipeline](#training-pipeline)
4. [Training Individual Components](#training-individual-components)
5. [Evaluation](#evaluation)
6. [Performance Optimization](#performance-optimization)
7. [Troubleshooting](#troubleshooting)

## System Requirements

- **Hardware**:
  - CPU: 16+ cores recommended
  - RAM: 16GB+ recommended
  - GPU: NVIDIA RTX 4050 6GB or better
  - Storage: 50GB+ free space

- **Software**:
  - Python 3.8+
  - CUDA 11.7+
  - PyTorch 2.0+
  - Required Python packages (see requirements.txt)

## Data Preparation

The system expects cryptocurrency data in CSV format with the following columns:
- timestamp
- open
- high
- low
- close
- volume

Place your data files in the `D:\INNOX\Crypto_Data` directory. Each file should be named with the symbol (e.g., `BTCUSDT.csv`).

## Training Pipeline

### Complete Training Pipeline

To run the complete training pipeline, use the provided batch script (Windows) or shell script (Linux/Mac):

```
# Windows
run_mqtm_training.bat

# Linux/Mac
./run_mqtm_training.sh
```

This will:
1. Train the MQTM models (MG, TQE, SP3)
2. Train the ASP framework
3. Train the MGI module
4. Train the BOM module
5. Evaluate the trained models

### Customizing the Training Pipeline

You can customize the training process with the following options:

```
--data_dir DATA_DIR       Directory containing cryptocurrency data
--models_dir MODELS_DIR   Directory to save trained models
--symbols SYMBOL1 SYMBOL2 Symbols to train on (if None, use all available)
--batch_size BATCH_SIZE   Batch size for training
--epochs EPOCHS           Number of epochs to train for
--learning_rate LR        Learning rate for training
--num_episodes EPISODES   Number of episodes for ASP training
--num_iterations ITERS    Number of iterations for MGI and BOM training
--optimize_memory         Whether to optimize memory usage
--profile_performance     Whether to profile performance
--skip_mqtm               Skip MQTM model training
--skip_asp                Skip ASP framework training
--skip_mgi                Skip MGI module training
--skip_bom                Skip BOM module training
```

Example:

```
# Windows
run_mqtm_training.bat --symbols BTCUSDT ETHUSDT --batch_size 16 --epochs 100 --learning_rate 0.0001 --optimize_memory

# Linux/Mac
./run_mqtm_training.sh --symbols BTCUSDT ETHUSDT --batch_size 16 --epochs 100 --learning_rate 0.0001 --optimize_memory
```

## Training Individual Components

### 1. MQTM Models (MG, TQE, SP3)

To train the core MQTM models:

```
python train_mqtm_system.py --data_dir D:\INNOX\Crypto_Data --models_dir models --train_all
```

Options:
- `--train_mg`: Train only the Multiverse Generator
- `--train_tqe`: Train only the Topo-Quantum Encoder
- `--train_sp3`: Train only the Superposition Pool
- `--train_all`: Train all components

### 2. Adversarial Self-Play (ASP) Framework

To train the ASP framework:

```
python train_asp_framework.py --data_dir D:\INNOX\Crypto_Data --models_dir models --asp_dir models/asp
```

Options:
- `--num_episodes`: Number of episodes to train for
- `--max_steps`: Maximum number of steps per episode
- `--reward_type`: Type of reward to use (pnl, sharpe, sortino, calmar, custom)
- `--risk_aversion`: Risk aversion parameter
- `--use_regime_detection`: Whether to use regime detection
- `--use_causal_model`: Whether to use causal model for perturbation

### 3. Meta-Gradient Introspection (MGI) Module

To train the MGI module:

```
python train_mgi_module.py --data_dir D:\INNOX\Crypto_Data --models_dir models --asp_dir models/asp --mgi_dir models/mgi
```

Options:
- `--num_iterations`: Number of iterations to run
- `--meta_batch_size`: Meta batch size
- `--meta_lr`: Meta learning rate
- `--use_adaptive_lr`: Whether to use adaptive learning rate
- `--use_second_order`: Whether to use second-order optimization
- `--optimize_hyperparams`: Hyperparameters to optimize

### 4. Bayesian Online Mixture (BOM) Module

To train the BOM module:

```
python train_bom_module.py --data_dir D:\INNOX\Crypto_Data --models_dir models --asp_dir models/asp --bom_dir models/bom
```

Options:
- `--num_iterations`: Number of iterations to run
- `--batch_size`: Batch size
- `--num_models`: Number of models in the mixture
- `--hidden_dims`: Hidden dimensions for Bayesian networks
- `--prior_std`: Prior standard deviation
- `--learning_rate`: Learning rate

## Evaluation

To evaluate the trained models:

```
python evaluate_mqtm_system.py --data_dir D:\INNOX\Crypto_Data --models_dir models --output_dir evaluation_results --use_asp --use_mgi --use_bom
```

This will generate evaluation results in the `evaluation_results` directory, including:
- Sample generation from the Multiverse Generator
- Feature extraction from the Topo-Quantum Encoder
- Trading decisions from the Superposition Pool
- Performance metrics for the Trader Agent
- Uncertainty estimation from the Bayesian Online Mixture

## Performance Optimization

### Memory Optimization

To optimize memory usage, add the `--optimize_memory` flag to any training command:

```
python train_mqtm_system.py --data_dir D:\INNOX\Crypto_Data --models_dir models --train_all --optimize_memory
```

This will:
- Use mixed precision training (FP16)
- Apply gradient checkpointing
- Optimize model parameters
- Use dynamic batch sizing

### Performance Profiling

To profile performance, add the `--profile_performance` flag to any training command:

```
python train_mqtm_system.py --data_dir D:\INNOX\Crypto_Data --models_dir models --train_all --profile_performance
```

This will:
- Generate performance profiles for each model
- Identify bottlenecks
- Measure execution time
- Track memory usage

Profiles will be saved in the `profiles` directory.

## Troubleshooting

### Out of Memory Errors

If you encounter out of memory errors:

1. Reduce batch size:
   ```
   python train_mqtm_system.py --batch_size 8
   ```

2. Enable memory optimization:
   ```
   python train_mqtm_system.py --optimize_memory
   ```

3. Train components individually:
   ```
   python train_mqtm_system.py --train_mg
   python train_mqtm_system.py --train_tqe
   python train_mqtm_system.py --train_sp3
   ```

### Slow Training

If training is too slow:

1. Reduce the number of epochs:
   ```
   python train_mqtm_system.py --epochs 25
   ```

2. Use a subset of symbols:
   ```
   python train_mqtm_system.py --symbols BTCUSDT
   ```

3. Skip components you don't need:
   ```
   python run_mqtm_training.py --skip_mgi --skip_bom
   ```

### Data Loading Issues

If you encounter data loading issues:

1. Check that your data files are in the correct format
2. Ensure the data directory path is correct
3. Try specifying symbols explicitly:
   ```
   python train_mqtm_system.py --symbols BTCUSDT ETHUSDT
   ```

## Next Steps

After training and evaluating the MQTM system, you can:

1. Implement live trading with Delta Exchange India
2. Develop a web dashboard for monitoring
3. Optimize the system for specific market conditions
4. Extend the system with additional features

For more information, refer to the main README.md file.
