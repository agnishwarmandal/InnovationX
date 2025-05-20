# Multiverse Quantum-Topological Meta-Learning (MQTM)

MQTM is a novel training paradigm designed for crypto-futures prediction, combining causal counterfactuals, quantum-inspired superposition, persistent homology, and adversarial self-play.

## Overview

The MQTM system consists of several key components:

1. **Multiverse Generator (MG)**: Generates synthetic OHLCV data conditioned on latent causal graphs.
2. **Topo-Quantum Encoder (TQE)**: Extracts topological and complex wavelet features from OHLCV data.
3. **Superposition Parameter Pool (SP³)**: Maintains complex-valued neural network weights in quantum-inspired superposition.
4. **Adversarial Self-Play (ASP)**: Implements a self-play mechanism between a trader and an adversary.
5. **Meta-Gradient Introspection (MGI)**: Optimizes learning rates based on topological drift.
6. **Bayesian Online Mixture (BOM)**: Maintains a posterior over SP³ heads.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA-compatible GPU (NVIDIA GeForce RTX 4050 6GB or better)

### Setup

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/mqtm.git
   cd mqtm
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Install additional dependencies for topological data analysis:
   ```
   pip install giotto-tda ripser persim
   ```

## Usage

### Data Preparation

The system expects cryptocurrency data in CSV format with the following columns:
- `timestamp`: Time of the candlestick
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

Place your data files in the `D:\INNOX\Crypto_Data` directory, with each file named after the cryptocurrency symbol (e.g., `BTC_USDT_5m.csv`).

### Robust Training

To train the model on all available datasets:

```bash
python run_training.py --data_dir="D:\INNOX\Crypto_Data" --models_dir="models/robust_training" --batch_size=16 --epochs=50 --learning_rate=1e-5
```

Additional training options:
- `--max_files`: Maximum number of files to load (default: all files)
- `--batch_size`: Batch size for training (default: 16)
- `--epochs`: Number of epochs to train for (default: 50)
- `--learning_rate`: Learning rate for training (default: 1e-5)

### Monitoring

To monitor the training progress and system resources:

```bash
python monitor_training.py --models_dir="models/robust_training" --interval=60 --output_dir="monitor_output"
```

### Testing

To test the trained model on all datasets:

```bash
python test_model.py --data_dir="D:\INNOX\Crypto_Data" --model_path="models/robust_training/best_model.pt" --output_dir="test_results"
```

## System Architecture

### Robust Core Prediction Model

The core prediction model is designed for numerical stability and robust training:

- Batch normalization at multiple layers to prevent exploding/vanishing gradients
- Gradient clipping to limit gradient magnitudes
- Learning rate scheduling to adapt to training dynamics
- Mixed precision training for improved performance
- Weight decay for regularization and preventing overfitting

### Data Processing Pipeline

The data processing pipeline ensures high-quality input data:

- Automatic detection and handling of different data formats
- Normalization of price and volume data
- Calculation of technical indicators (returns, volatility, moving averages, MACD, RSI)
- Creation of fixed-length input sequences for consistent training

### Training Process

The training process is designed for robustness and efficiency:

- Progress bars with real-time metrics
- Automatic early stopping based on validation performance
- Checkpointing to save training progress
- Resource monitoring to prevent crashes
- Comprehensive logging for debugging

### Future Components

After the core prediction model is trained and validated, the following components will be implemented:

#### Multiverse Generator (MG)

The MG will combine a diffusion model with a causal graph learner to generate synthetic but causally-consistent OHLCV streams.

#### Topo-Quantum Encoder (TQE)

The TQE will extract rich features from OHLCV data using persistent homology and complex-valued wavelet coefficients.

#### Superposition Parameter Pool (SP³)

The SP³ will implement complex-valued neural networks with quantum-inspired superposition.

## Hardware Considerations

The implementation is optimized for systems with limited GPU memory (RTX 4050 6GB VRAM) and 16-core CPU:

- Mixed precision training to reduce memory usage
- Gradient clipping to prevent numerical instability
- Efficient batch processing with adjustable batch size
- Dynamic resource monitoring to prevent crashes
- Automatic batch size adjustment based on available memory

## Progress Monitoring

The system includes a comprehensive progress monitoring system:

- Real-time progress bars with loss and accuracy metrics
- Resource usage tracking (CPU, RAM, GPU)
- Training metrics visualization (loss, accuracy, learning rate)
- Automatic detection of numerical instability
- Comprehensive logging for debugging

## Performance Metrics

The system evaluates performance using the following metrics:

- **Accuracy**: Percentage of correct predictions
- **Precision**: Percentage of true positives among positive predictions
- **Recall**: Percentage of true positives among actual positives
- **F1 Score**: Harmonic mean of precision and recall
- **ROC AUC**: Area under the Receiver Operating Characteristic curve

## Visualization Tools

The system includes several visualization tools:

- **Loss Curves**: Visualize training and validation loss over time
- **Accuracy Curves**: Visualize training and validation accuracy over time
- **Learning Rate Curves**: Visualize learning rate adjustments over time
- **Confusion Matrix**: Visualize true vs. predicted classes
- **ROC Curve**: Visualize model discrimination ability
- **Performance by Symbol**: Compare model performance across different cryptocurrencies

## Hyper-Random Training

The implementation includes a hyper-random training utility that introduces controlled randomness in:

- Data sampling across all 136 datasets
- Sequence creation with varying lengths and offsets
- Batch composition with diverse symbols
- Training schedules with dynamic learning rates

## Delta Exchange Integration

After comprehensive training and backtesting, the system will be integrated with Delta Exchange India for live trading:

- **API Integration**: Secure connection to Delta Exchange API
- **Order Management**: Automated order placement and management
- **Risk Management**: Position sizing and stop-loss implementation
- **Performance Monitoring**: Real-time tracking of trading performance

**Note**: API credentials will only be provided after satisfactory backtesting performance.

## Training and Testing Pipeline

The complete training and testing pipeline consists of the following steps:

1. **Data Loading**: Load all 136 cryptocurrency datasets
2. **Data Preprocessing**: Normalize data and calculate technical indicators
3. **Model Training**: Train the robust core prediction model
4. **Model Validation**: Validate the model on held-out data
5. **Model Testing**: Test the model on unseen data
6. **Performance Analysis**: Analyze model performance across different cryptocurrencies
7. **Backtesting**: Simulate trading with the model on historical data
8. **Live Trading**: Deploy the model for live trading on Delta Exchange

## License

This project is proprietary and confidential. All rights reserved.

## Acknowledgements

This implementation is based on the MQTM concept, which combines ideas from deep learning, quantum computing, topological data analysis, and reinforcement learning.

## Disclaimer

This software is provided for research and educational purposes only. Trading cryptocurrencies involves significant risk. Always conduct thorough research before making investment decisions.
