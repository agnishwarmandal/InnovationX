"""
Robust training script for the MQTM system.

This script addresses numerical stability issues and uses all available datasets.
"""

import os
import argparse
import logging
import time
import json
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple, Union, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("robust_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    """Global configuration."""

    # Hardware - Optimized for RTX 4050 6GB GPU and 16-core CPU
    device = torch.device("cuda")  # Force CUDA usage
    num_workers = 2  # Use 2 workers for data loading (balanced approach)
    pin_memory = True  # Pin memory for faster data transfer to GPU
    mixed_precision = True  # Use mixed precision for memory efficiency
    prefetch_factor = 2  # Optimal prefetch factor to avoid memory issues
    persistent_workers = False  # Disable persistent workers to save memory

    # GPU optimization - Maximizing RTX 4050 performance
    cudnn_benchmark = True  # Enable cuDNN benchmark for faster convolutions
    cudnn_deterministic = False  # Disable deterministic mode for speed
    gpu_memory_fraction = 0.85  # Use 85% of GPU memory (safe margin for system stability)
    gpu_for_dataloading = False  # Keep GPU focused on training, not data loading

    # CPU optimization - Strategic use of 16 cores
    cpu_for_preprocessing = True  # Use CPU for data preprocessing
    cpu_priority_high = True  # Set high priority for training process

    # Memory management - Preventing crashes
    memory_efficient_mode = True  # Use memory-efficient operations
    clear_cache_frequency = 10  # Clear CUDA cache every N batches
    aggressive_garbage_collection = True  # Enable aggressive garbage collection

    # Data processing optimization
    process_on_gpu = False  # Process data on CPU to save GPU for training
    use_amp = True  # Use automatic mixed precision for better performance

    # Batch processing - Optimized for RTX 4050 6GB
    max_samples_per_symbol = 3000  # Limit samples per symbol to avoid memory issues
    max_files = None  # Process all files
    process_batch_size = 5  # Process this many symbols at once (memory optimization)

    # Data
    data_dir = "D:\\INNOX\\Crypto_Data"
    history_length = 100
    forecast_horizon = 20
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 64  # Optimized batch size for RTX 4050 (will be auto-adjusted if OOM)
    adaptive_batch_size = True  # Automatically adjust batch size if OOM occurs
    shuffle = True

    # Model - Optimized for speed and memory efficiency
    hidden_dim = 128
    num_layers = 2
    dropout = 0.1
    learning_rate = 1e-5  # Reduced learning rate for stability
    weight_decay = 1e-6   # Added weight decay for regularization
    num_epochs = 50
    gradient_clip = 1.0   # Added gradient clipping for stability
    gradient_accumulation_steps = 1  # Can be increased if batch size needs to be smaller

    # Training
    models_dir = "models/robust_training"
    save_interval = 1
    log_interval = 1

    # Checkpointing
    checkpoint_interval = 5  # Save checkpoint every 5 epochs
    resume_training = True  # Resume training from checkpoint if available

    # Resource monitoring
    monitor_resources = True  # Monitor system resources during training
    resource_check_interval = 10  # Check resources every N batches

    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.data_dir:
            cls.data_dir = args.data_dir

        if args.batch_size:
            cls.batch_size = args.batch_size

        if args.epochs:
            cls.num_epochs = args.epochs

        if args.learning_rate:
            cls.learning_rate = args.learning_rate

        if args.models_dir:
            cls.models_dir = args.models_dir

# Data loading and preprocessing
class DataLoader:
    """Data loader for cryptocurrency data."""

    def __init__(self, config):
        """Initialize data loader."""
        self.config = config
        self.data = {}
        self.train_data = {}
        self.val_data = {}
        self.test_data = {}

    def load_data(self, max_files=None):
        """
        Load data for all available symbols.

        Args:
            max_files: Maximum number of files to load (None for all)
        """
        # Get all CSV files in the directory
        csv_files = glob.glob(os.path.join(self.config.data_dir, "*.csv"))

        # Limit the number of files if specified
        if max_files is not None:
            csv_files = csv_files[:max_files]

        logger.info(f"Found {len(csv_files)} CSV files in {self.config.data_dir}")

        # Load each file
        for file_path in csv_files:
            try:
                # Extract symbol from filename
                symbol = os.path.basename(file_path).split(".")[0]

                # Load data
                logger.info(f"Loading {file_path}...")
                df = pd.read_csv(file_path)
                logger.info(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")

                # Preprocess dataframe
                df = self._preprocess_dataframe(df)

                # Store data
                self.data[symbol] = df

                # Split data
                self._split_data(symbol)

                logger.info(f"Processed {symbol} data")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

    def _preprocess_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess dataframe.

        Args:
            df: Input dataframe

        Returns:
            Preprocessed dataframe
        """
        # Make a copy to avoid modifying the original
        df = df.copy()

        # Check required columns
        required_columns = ["timestamp", "open", "high", "low", "close", "volume"]

        # Map common column names to required names
        column_mapping = {
            "time": "timestamp",
            "Time": "timestamp",
            "Timestamp": "timestamp",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Vol": "volume",
        }

        # Rename columns if needed
        df = df.rename(columns=column_mapping)

        # Check if all required columns are present
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.warning(f"Missing columns: {missing_columns}")

            # Try to infer missing columns
            if "timestamp" in missing_columns and "date" in df.columns:
                df["timestamp"] = pd.to_datetime(df["date"]).astype(int) // 10**9

            # If still missing required columns, raise error
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")

        # Convert timestamp to datetime if it's not already
        if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
            try:
                # Try to convert from Unix timestamp
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
            except:
                # Try to parse as string
                df["timestamp"] = pd.to_datetime(df["timestamp"])

        # Sort by timestamp
        df = df.sort_values("timestamp")

        # Reset index
        df = df.reset_index(drop=True)

        # Calculate additional features
        df = self._calculate_features(df)

        # Normalize data
        df = self._normalize_data(df)

        return df

    def _normalize_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize data to improve training stability.

        Args:
            df: Input dataframe

        Returns:
            Normalized dataframe
        """
        # Create a copy to avoid modifying the original
        df_norm = df.copy()

        # Normalize OHLCV data using min-max scaling
        for col in ["open", "high", "low", "close"]:
            min_val = df_norm[col].min()
            max_val = df_norm[col].max()
            df_norm[col] = (df_norm[col] - min_val) / (max_val - min_val + 1e-8)

        # Normalize volume separately (it can have a different scale)
        min_vol = df_norm["volume"].min()
        max_vol = df_norm["volume"].max()
        df_norm["volume"] = (df_norm["volume"] - min_vol) / (max_vol - min_vol + 1e-8)

        return df_norm

    def _calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional features.

        Args:
            df: Input dataframe

        Returns:
            Dataframe with additional features
        """
        # Calculate returns
        df["returns"] = df["close"].pct_change()

        # Calculate log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))

        # Calculate volatility (20-period rolling standard deviation of returns)
        df["volatility"] = df["returns"].rolling(window=20).std()

        # Calculate moving averages
        df["ma_5"] = df["close"].rolling(window=5).mean()
        df["ma_20"] = df["close"].rolling(window=20).mean()
        df["ma_50"] = df["close"].rolling(window=50).mean()

        # Calculate MACD
        df["ema_12"] = df["close"].ewm(span=12, adjust=False).mean()
        df["ema_26"] = df["close"].ewm(span=26, adjust=False).mean()
        df["macd"] = df["ema_12"] - df["ema_26"]
        df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
        df["macd_hist"] = df["macd"] - df["macd_signal"]

        # Calculate RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df["rsi"] = 100 - (100 / (1 + rs))

        # Drop rows with NaN values
        df = df.dropna()

        return df

    def _split_data(self, symbol: str) -> None:
        """
        Split data into train, validation, and test sets.

        Args:
            symbol: Symbol to split data for
        """
        df = self.data[symbol]

        # Calculate split indices
        n = len(df)
        train_idx = int(n * self.config.train_ratio)
        val_idx = train_idx + int(n * self.config.val_ratio)

        # Split data
        self.train_data[symbol] = df.iloc[:train_idx]
        self.val_data[symbol] = df.iloc[train_idx:val_idx]
        self.test_data[symbol] = df.iloc[val_idx:]

        logger.info(f"Split {symbol} data: train={len(self.train_data[symbol])}, "
                   f"val={len(self.val_data[symbol])}, test={len(self.test_data[symbol])}")

    def create_sequences(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create input-output sequences from dataframe.

        Args:
            df: Input dataframe

        Returns:
            Tuple of (X, y) arrays
        """
        # Extract OHLCV data
        ohlcv = df[["open", "high", "low", "close", "volume"]].values

        # Create sequences
        X, y = [], []

        for i in range(len(df) - self.config.history_length - self.config.forecast_horizon + 1):
            # Input sequence
            X.append(ohlcv[i:i+self.config.history_length])

            # Output sequence (future close prices)
            future_close = ohlcv[i+self.config.history_length:i+self.config.history_length+self.config.forecast_horizon, 3]

            # Calculate direction (1 if price goes up, 0 if down)
            current_close = ohlcv[i+self.config.history_length-1, 3]
            direction = 1 if future_close[-1] > current_close else 0

            y.append(direction)

        return np.array(X), np.array(y)

    def create_dataloaders(self):
        """
        Create PyTorch DataLoaders for training, validation, and testing.
        Optimized for balanced CPU/GPU resource allocation.

        Returns:
            Tuple of (train_loader, val_loader, test_loader)
        """
        import gc
        import psutil

        # Start resource monitoring
        start_time = time.time()
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Initial memory usage: {initial_memory:.2f} MB")

        # Create datasets
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []

        # Get configuration parameters
        max_samples_per_symbol = self.config.max_samples_per_symbol
        process_batch_size = self.config.process_batch_size

        # Process symbols in batches to reduce memory usage
        symbols = list(self.data.keys())
        total_symbols = len(symbols)
        logger.info(f"Processing {total_symbols} symbols for DataLoader creation")

        # Set process priority if configured
        if self.config.cpu_priority_high:
            try:
                if os.name == 'nt':  # Windows
                    import psutil
                    p = psutil.Process()
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                    logger.info("Set process priority to HIGH")
                else:  # Unix-based
                    os.nice(-10)
                    logger.info("Set process priority to high")
            except Exception as e:
                logger.warning(f"Could not set process priority: {e}")

        # Process in smaller batches to avoid memory issues
        for i in range(0, total_symbols, process_batch_size):
            batch_symbols = symbols[i:i+process_batch_size]
            batch_size_info = f"{i+1}-{min(i+process_batch_size, total_symbols)}"
            logger.info(f"Processing symbols {batch_size_info} of {total_symbols} ({(i+1)/total_symbols*100:.1f}%)")

            batch_X_train, batch_y_train = [], []
            batch_X_val, batch_y_val = [], []
            batch_X_test, batch_y_test = [], []

            # Process each symbol in the batch
            for symbol_idx, symbol in enumerate(batch_symbols):
                try:
                    # Create sequences - CPU intensive operation
                    X_train_symbol, y_train_symbol = self.create_sequences(self.train_data[symbol])
                    X_val_symbol, y_val_symbol = self.create_sequences(self.val_data[symbol])
                    X_test_symbol, y_test_symbol = self.create_sequences(self.test_data[symbol])

                    # Limit the number of samples per symbol to control memory usage
                    if len(X_train_symbol) > max_samples_per_symbol:
                        indices = np.random.choice(len(X_train_symbol), max_samples_per_symbol, replace=False)
                        X_train_symbol = X_train_symbol[indices]
                        y_train_symbol = y_train_symbol[indices]

                    if len(X_val_symbol) > max_samples_per_symbol // 5:
                        indices = np.random.choice(len(X_val_symbol), max_samples_per_symbol // 5, replace=False)
                        X_val_symbol = X_val_symbol[indices]
                        y_val_symbol = y_val_symbol[indices]

                    if len(X_test_symbol) > max_samples_per_symbol // 5:
                        indices = np.random.choice(len(X_test_symbol), max_samples_per_symbol // 5, replace=False)
                        X_test_symbol = X_test_symbol[indices]
                        y_test_symbol = y_test_symbol[indices]

                    # Append to batch lists
                    batch_X_train.append(X_train_symbol)
                    batch_y_train.append(y_train_symbol)
                    batch_X_val.append(X_val_symbol)
                    batch_y_val.append(y_val_symbol)
                    batch_X_test.append(X_test_symbol)
                    batch_y_test.append(y_test_symbol)

                    # Aggressive memory management if configured
                    if self.config.aggressive_garbage_collection and symbol_idx % 2 == 0:
                        gc.collect()

                except Exception as e:
                    logger.error(f"Error processing symbol {symbol}: {e}")
                    continue

            # Concatenate batch arrays - memory intensive operation
            if batch_X_train:
                try:
                    X_train.append(np.concatenate(batch_X_train))
                    y_train.append(np.concatenate(batch_y_train))
                except Exception as e:
                    logger.error(f"Error concatenating training data: {e}")

            if batch_X_val:
                try:
                    X_val.append(np.concatenate(batch_X_val))
                    y_val.append(np.concatenate(batch_y_val))
                except Exception as e:
                    logger.error(f"Error concatenating validation data: {e}")

            if batch_X_test:
                try:
                    X_test.append(np.concatenate(batch_X_test))
                    y_test.append(np.concatenate(batch_y_test))
                except Exception as e:
                    logger.error(f"Error concatenating test data: {e}")

            # Clear memory after each batch
            del batch_X_train, batch_y_train, batch_X_val, batch_y_val, batch_X_test, batch_y_test

            # Aggressive memory cleanup
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Log memory usage
            current_memory = process.memory_info().rss / (1024 * 1024)  # MB
            logger.info(f"Memory usage after batch {batch_size_info}: {current_memory:.2f} MB")

        # Concatenate all batches - final memory intensive operation
        try:
            X_train = np.concatenate(X_train)
            y_train = np.concatenate(y_train)
            X_val = np.concatenate(X_val)
            y_val = np.concatenate(y_val)
            X_test = np.concatenate(X_test)
            y_test = np.concatenate(y_test)
        except Exception as e:
            logger.error(f"Error in final concatenation: {e}")
            raise

        logger.info(f"Created datasets: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

        # Create PyTorch tensors - keep on CPU initially if configured
        logger.info("Creating PyTorch tensors")
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
        y_val_tensor = torch.tensor(y_val, dtype=torch.long)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.long)

        # Clear numpy arrays to free memory
        del X_train, y_train, X_val, y_val, X_test, y_test
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Create PyTorch datasets
        logger.info("Creating PyTorch datasets")
        train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
        val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
        test_dataset = torch.utils.data.TensorDataset(X_test_tensor, y_test_tensor)

        # Clear tensors to free memory
        del X_train_tensor, y_train_tensor, X_val_tensor, y_val_tensor, X_test_tensor, y_test_tensor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Log memory usage
        current_memory = process.memory_info().rss / (1024 * 1024)  # MB
        logger.info(f"Memory usage after dataset creation: {current_memory:.2f} MB")

        # Determine optimal batch size if adaptive batch size is enabled
        batch_size = self.config.batch_size
        if self.config.adaptive_batch_size and torch.cuda.is_available():
            # Get GPU memory info
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
            free_memory = (total_memory - torch.cuda.memory_allocated() / (1024**3))  # GB

            # Estimate memory per sample (conservative estimate)
            estimated_memory_per_sample = 0.0001  # GB, adjust based on your model

            # Calculate maximum batch size based on available memory and safety factor
            safety_factor = 0.7  # Use 70% of estimated available memory
            max_batch_size = int((free_memory * safety_factor) / estimated_memory_per_sample)

            # Adjust batch size if needed
            if max_batch_size < batch_size:
                old_batch_size = batch_size
                batch_size = max(32, max_batch_size)  # Minimum batch size of 32
                logger.info(f"Adjusted batch size from {old_batch_size} to {batch_size} based on available GPU memory")
            else:
                logger.info(f"Using configured batch size of {batch_size}")

        # Create DataLoaders with optimized settings
        logger.info("Creating DataLoaders with optimized settings")
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
            persistent_workers=self.config.persistent_workers,
            prefetch_factor=self.config.prefetch_factor if self.config.num_workers > 0 else None,
        )

        # Calculate elapsed time
        elapsed_time = time.time() - start_time
        logger.info(f"DataLoader creation completed in {elapsed_time:.2f} seconds")
        logger.info(f"Created DataLoaders: train={len(train_loader)}, val={len(val_loader)}, test={len(test_loader)}")

        return train_loader, val_loader, test_loader

# Improved model with batch normalization for stability
class RobustModel(nn.Module):
    """Robust model for cryptocurrency price prediction."""

    def __init__(self, config):
        """Initialize model."""
        super().__init__()
        self.config = config

        # Input dimensions
        self.input_dim = 5  # OHLCV
        self.seq_len = config.history_length

        # Batch normalization for input
        self.input_bn = nn.BatchNorm1d(self.input_dim)

        # LSTM layers with batch normalization
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )

        # Batch normalization after LSTM
        self.lstm_bn = nn.BatchNorm1d(config.hidden_dim)

        # Fully connected layers with batch normalization
        self.fc1 = nn.Linear(config.hidden_dim, config.hidden_dim // 2)
        self.bn1 = nn.BatchNorm1d(config.hidden_dim // 2)
        self.dropout1 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_dim // 2, 2)  # Binary classification

    def forward(self, x):
        """Forward pass."""
        # Input shape: [batch_size, seq_len, input_dim]
        batch_size = x.size(0)

        # Apply batch normalization to input
        # Reshape for batch norm: [batch_size * seq_len, input_dim]
        x_reshaped = x.reshape(-1, self.input_dim)
        x_bn = self.input_bn(x_reshaped)
        x = x_bn.reshape(batch_size, self.seq_len, self.input_dim)

        # LSTM
        lstm_out, _ = self.lstm(x)

        # Use last output
        last_out = lstm_out[:, -1]

        # Apply batch normalization
        last_out = self.lstm_bn(last_out)

        # Fully connected layers
        x = F.relu(self.fc1(last_out))
        x = self.bn1(x)
        x = self.dropout1(x)
        logits = self.fc2(x)

        return logits

# Training functions
def train_epoch(model, train_loader, optimizer, criterion, config, epoch, scaler=None):
    """
    Train for one epoch with optimized CPU/GPU resource allocation.

    Args:
        model: Model to train
        train_loader: Training data loader
        optimizer: Optimizer
        criterion: Loss function
        config: Configuration
        epoch: Current epoch
        scaler: Gradient scaler for mixed precision training

    Returns:
        Tuple of (train_loss, train_acc)
    """
    import gc
    import psutil

    # Start resource monitoring
    start_time = time.time()
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / (1024 * 1024)  # MB

    # Set model to training mode
    model.train()

    # Initialize metrics
    train_loss = 0.0
    train_acc = 0.0
    processed_batches = 0

    # Gradient accumulation setup
    effective_batch_size = config.batch_size * config.gradient_accumulation_steps

    # Create progress bar
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")

    # Initialize scaler for mixed precision training if not provided
    if scaler is None and config.mixed_precision:
        scaler = torch.amp.GradScaler('cuda')

    # Log resource usage at the start of the epoch
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
        logger.info(f"GPU Memory at epoch start: Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"CPU usage at epoch start: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")

    # Get GPU utilization
    gpu_utilization = 0
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
            stdout=subprocess.PIPE,
            text=True
        )
        gpu_utilization = float(result.stdout.strip())
        logger.info(f"GPU utilization at epoch start: {gpu_utilization:.1f}%")
    except Exception as e:
        logger.warning(f"Could not get GPU utilization: {e}")

    # Training loop with resource optimization
    try:
        # Initialize gradient accumulation counter
        accumulation_counter = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            try:
                # Resource monitoring
                if config.monitor_resources and batch_idx % config.resource_check_interval == 0:
                    if torch.cuda.is_available():
                        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
                        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
                        logger.info(f"Batch {batch_idx} GPU Memory: Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

                    cpu_percent = psutil.cpu_percent()
                    memory_percent = psutil.virtual_memory().percent
                    logger.info(f"Batch {batch_idx} CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")

                # Move data to device with non-blocking transfer for better parallelism
                device = config.device
                data = data.to(device, non_blocking=True)
                target = target.to(device, non_blocking=True)

                # Only zero gradients at the start of accumulation cycle
                if accumulation_counter == 0:
                    optimizer.zero_grad(set_to_none=True)  # More memory efficient

                # Forward pass with mixed precision
                if config.mixed_precision:
                    try:
                        with torch.amp.autocast('cuda'):
                            output = model(data)
                            loss = criterion(output, target)

                            # Scale loss by accumulation steps if using gradient accumulation
                            if config.gradient_accumulation_steps > 1:
                                loss = loss / config.gradient_accumulation_steps

                        # Backward pass with scaled gradients
                        scaler.scale(loss).backward()

                        # Only update weights at the end of accumulation cycle or at the last batch
                        if (accumulation_counter == config.gradient_accumulation_steps - 1) or (batch_idx == len(train_loader) - 1):
                            # Gradient clipping
                            if config.gradient_clip > 0:
                                scaler.unscale_(optimizer)
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

                            scaler.step(optimizer)
                            scaler.update()
                            accumulation_counter = 0
                        else:
                            accumulation_counter += 1

                    except Exception as e:
                        logger.error(f"Error in mixed precision training: {e}")
                        # Fall back to full precision
                        logger.info("Falling back to full precision for this batch")

                        # Reset gradients
                        optimizer.zero_grad(set_to_none=True)

                        # Full precision forward pass
                        output = model(data)
                        loss = criterion(output, target)

                        # Scale loss by accumulation steps if using gradient accumulation
                        if config.gradient_accumulation_steps > 1:
                            loss = loss / config.gradient_accumulation_steps

                        # Backward pass
                        loss.backward()

                        # Only update weights at the end of accumulation cycle or at the last batch
                        if (accumulation_counter == config.gradient_accumulation_steps - 1) or (batch_idx == len(train_loader) - 1):
                            # Gradient clipping
                            if config.gradient_clip > 0:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

                            optimizer.step()
                            accumulation_counter = 0
                        else:
                            accumulation_counter += 1
                else:
                    # Forward pass
                    output = model(data)
                    loss = criterion(output, target)

                    # Scale loss by accumulation steps if using gradient accumulation
                    if config.gradient_accumulation_steps > 1:
                        loss = loss / config.gradient_accumulation_steps

                    # Backward pass
                    loss.backward()

                    # Only update weights at the end of accumulation cycle or at the last batch
                    if (accumulation_counter == config.gradient_accumulation_steps - 1) or (batch_idx == len(train_loader) - 1):
                        # Gradient clipping
                        if config.gradient_clip > 0:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
                        accumulation_counter = 0
                    else:
                        accumulation_counter += 1

                # Update metrics
                train_loss += loss.item() * (config.gradient_accumulation_steps if config.gradient_accumulation_steps > 1 else 1)
                processed_batches += 1

                # Compute accuracy
                with torch.no_grad():  # No need to track gradients for accuracy calculation
                    _, predicted = output.max(1)
                    batch_acc = predicted.eq(target).sum().item() / target.size(0)
                    train_acc += batch_acc

                # Update progress bar
                progress_bar.update(1)
                progress_bar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "acc": f"{batch_acc:.4f}",
                    "gpu_mem": f"{torch.cuda.memory_allocated() / 1e9:.2f}GB"
                })

                # Clear cache periodically to prevent memory fragmentation
                if config.clear_cache_frequency > 0 and batch_idx % config.clear_cache_frequency == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Check for NaN loss
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                    continue

            except Exception as e:
                logger.error(f"Error processing batch {batch_idx}: {e}")
                # Continue with next batch
                continue
    except Exception as e:
        logger.error(f"Error in training loop: {e}")
        # Clean up progress bar
        progress_bar.close()
        raise

    # Close progress bar
    progress_bar.close()

    # Log resource usage at the end of the epoch
    if torch.cuda.is_available():
        gpu_memory_allocated = torch.cuda.memory_allocated() / 1e9  # GB
        gpu_memory_reserved = torch.cuda.memory_reserved() / 1e9  # GB
        logger.info(f"GPU Memory at epoch end: Allocated: {gpu_memory_allocated:.2f} GB, Reserved: {gpu_memory_reserved:.2f} GB")

    cpu_percent = psutil.cpu_percent()
    memory_percent = psutil.virtual_memory().percent
    logger.info(f"CPU usage at epoch end: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")

    # Calculate epoch time
    epoch_time = time.time() - start_time
    logger.info(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")

    # Compute average metrics
    train_loss /= max(1, processed_batches)  # Avoid division by zero
    train_acc /= max(1, processed_batches)   # Avoid division by zero

    # Perform garbage collection
    if config.aggressive_garbage_collection:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return train_loss, train_acc

def validate(model, val_loader, criterion, config):
    """Validate model."""
    model.eval()

    # Initialize metrics
    val_loss = 0.0
    val_acc = 0.0
    processed_batches = 0

    # Log GPU memory usage at the start of validation
    if torch.cuda.is_available():
        logger.info(f"GPU Memory at validation start: "
                   f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                   f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Validation loop
    try:
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(val_loader):
                try:
                    # Move data to device
                    device = config.device
                    data = data.to(device, non_blocking=True)
                    target = target.to(device, non_blocking=True)

                    # Forward pass with mixed precision
                    if config.mixed_precision:
                        try:
                            with torch.amp.autocast('cuda'):  # Updated to new syntax
                                output = model(data)
                                loss = criterion(output, target)
                        except Exception as e:
                            logger.error(f"Error in mixed precision validation: {e}")
                            # Fall back to full precision
                            logger.info("Falling back to full precision for this batch")
                            output = model(data)
                            loss = criterion(output, target)
                    else:
                        # Forward pass
                        output = model(data)
                        loss = criterion(output, target)

                    # Update metrics
                    val_loss += loss.item()

                    # Compute accuracy
                    _, predicted = output.max(1)
                    batch_acc = predicted.eq(target).sum().item() / target.size(0)
                    val_acc += batch_acc
                    processed_batches += 1

                    # Log progress every 100 batches
                    if batch_idx % 100 == 0:
                        logger.info(f"Validation batch {batch_idx}/{len(val_loader)}, "
                                   f"Loss: {loss.item():.4f}, Acc: {batch_acc:.4f}")

                    # Check for NaN loss
                    if torch.isnan(loss):
                        logger.warning(f"NaN loss detected at validation batch {batch_idx}. Skipping batch.")
                        continue

                except Exception as e:
                    logger.error(f"Error processing validation batch {batch_idx}: {e}")
                    # Continue with next batch
                    continue
    except Exception as e:
        logger.error(f"Error in validation loop: {e}")
        raise

    # Log GPU memory usage at the end of validation
    if torch.cuda.is_available():
        logger.info(f"GPU Memory at validation end: "
                   f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                   f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Compute average metrics
    val_loss /= max(1, processed_batches)  # Avoid division by zero
    val_acc /= max(1, processed_batches)   # Avoid division by zero

    return val_loss, val_acc

def train_model(model, train_loader, val_loader, config, start_epoch=0):
    """
    Train model with GPU optimization.

    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Configuration
        start_epoch: Starting epoch (for resuming training)

    Returns:
        Training metrics
    """
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=3,
        verbose=True,
    )

    # Initialize metrics
    best_val_acc = 0.0
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "learning_rates": [],
        "gpu_utilization": [],
        "gpu_memory": [],
        "epochs_completed": 0,
    }

    # Load metrics from checkpoint if resuming
    checkpoint_path = os.path.join(config.models_dir, "latest_checkpoint.pt")
    if start_epoch > 0 and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=config.device)
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            logger.info("Loaded optimizer state from checkpoint")

        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            logger.info("Loaded scheduler state from checkpoint")

        if 'metrics' in checkpoint:
            # Load metrics but only up to start_epoch
            loaded_metrics = checkpoint['metrics']
            for key in metrics:
                if key in loaded_metrics and isinstance(loaded_metrics[key], list):
                    metrics[key] = loaded_metrics[key][:start_epoch]

            # Update best validation accuracy
            if 'val_acc' in loaded_metrics and loaded_metrics['val_acc']:
                best_val_acc = max(loaded_metrics['val_acc'])
                logger.info(f"Loaded best validation accuracy: {best_val_acc:.4f}")

    # Create models directory
    os.makedirs(config.models_dir, exist_ok=True)

    # Set up automatic mixed precision if enabled
    scaler = torch.amp.GradScaler('cuda') if config.use_amp else None

    # Log GPU memory before training
    if torch.cuda.is_available():
        logger.info(f"GPU Memory before training: "
                   f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, "
                   f"Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        # Log epoch start
        logger.info(f"Starting epoch {epoch+1}/{config.num_epochs}")
        epoch_start_time = time.time()

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            epoch=epoch,
            scaler=scaler,
        )

        # Validate
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
        )

        # Update learning rate
        scheduler.step(val_acc)

        # Get current learning rate
        current_lr = optimizer.param_groups[0]["lr"]

        # Get GPU metrics
        gpu_utilization = 0
        gpu_memory = 0
        if torch.cuda.is_available():
            try:
                # Get GPU utilization
                import subprocess
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                    stdout=subprocess.PIPE,
                    text=True
                )
                gpu_utilization = float(result.stdout.strip())

                # Get GPU memory
                gpu_memory = torch.cuda.memory_allocated() / 1e9  # GB
            except Exception as e:
                logger.warning(f"Could not get GPU metrics: {e}")

        # Update metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        metrics["learning_rates"].append(current_lr)
        metrics["gpu_utilization"].append(gpu_utilization)
        metrics["gpu_memory"].append(gpu_memory)
        metrics["epochs_completed"] = epoch + 1

        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time

        # Log progress
        logger.info(f"Epoch {epoch+1}/{config.num_epochs} completed in {epoch_time:.2f}s, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}, "
                   f"LR: {current_lr:.6f}, "
                   f"GPU Util: {gpu_utilization:.1f}%, "
                   f"GPU Mem: {gpu_memory:.2f} GB")

        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.models_dir, "best_model.pt"))
            logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")

        # Save checkpoint at regular intervals
        if (epoch + 1) % config.checkpoint_interval == 0:
            checkpoint_path = os.path.join(config.models_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "metrics": metrics,
            }, checkpoint_path)
            logger.info(f"Saved checkpoint at epoch {epoch+1} to {checkpoint_path}")

        # Always save latest checkpoint for resuming
        latest_checkpoint_path = os.path.join(config.models_dir, "latest_checkpoint.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "metrics": metrics,
        }, latest_checkpoint_path)

        # Save metrics after each epoch
        metrics_path = os.path.join(config.models_dir, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        # Update visualizations after each epoch
        plot_metrics(metrics, config.models_dir)

        # Clear GPU cache after each epoch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save final model
    torch.save(model.state_dict(), os.path.join(config.models_dir, "final_model.pt"))
    logger.info("Saved final model")

    # Final metrics update
    metrics_path = os.path.join(config.models_dir, "final_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Final metrics saved to {metrics_path}")

    # Final visualizations
    plot_metrics(metrics, config.models_dir)
    logger.info("Final visualizations created")

    return metrics

def plot_metrics(metrics, output_dir):
    """Plot training metrics."""
    # Create figure for loss
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_loss"], label="Train")
    plt.plot(metrics["val_loss"], label="Validation")
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss.png"))
    plt.close()

    # Create figure for accuracy
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["train_acc"], label="Train")
    plt.plot(metrics["val_acc"], label="Validation")
    plt.title("Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "accuracy.png"))
    plt.close()

    # Create figure for learning rate
    plt.figure(figsize=(10, 6))
    plt.plot(metrics["learning_rates"])
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "learning_rate.png"))
    plt.close()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Robust MQTM Training Script")

    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models/robust_training",
                        help="Directory to save trained models")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of epochs to train for")
    parser.add_argument("--num_epochs", type=int, default=50,
                        help="Number of epochs to train for (alternative name)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate for training")
    parser.add_argument("--max_files", type=int, default=None,
                        help="Maximum number of files to load (None for all)")
    parser.add_argument("--force_continue", action="store_true",
                        help="Force continuous training without stopping between epochs")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Starting epoch for training (for resuming from a specific epoch)")
    parser.add_argument("--no_checkpoint", action="store_true",
                        help="Disable loading from checkpoint (start fresh)")

    return parser.parse_args()

def main():
    """Main function."""
    try:
        # Parse arguments
        args = parse_args()

        # Update configuration
        Config.update_from_args(args)

        # Handle command-line arguments for continuous training
        if hasattr(args, 'num_epochs') and args.num_epochs:
            Config.num_epochs = args.num_epochs
        elif hasattr(args, 'epochs') and args.epochs:
            Config.num_epochs = args.epochs

        # Force continuous training if requested
        if hasattr(args, 'force_continue') and args.force_continue:
            logger.info("Forcing continuous training mode")
            Config.resume_training = True

        # Set starting epoch if specified
        start_epoch_override = 0
        if hasattr(args, 'start_epoch') and args.start_epoch > 0:
            start_epoch_override = args.start_epoch
            logger.info(f"Will start training from epoch {start_epoch_override}")

        # Disable checkpoint loading if requested
        if hasattr(args, 'no_checkpoint') and args.no_checkpoint:
            Config.resume_training = False
            logger.info("Checkpoint loading disabled, starting fresh")

        # Ensure CUDA is available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available. Please check your PyTorch installation.")
            return

        # Set up GPU optimization
        if Config.cudnn_benchmark:
            torch.backends.cudnn.benchmark = True
            logger.info("cuDNN benchmark enabled for faster training")

        if not Config.cudnn_deterministic:
            torch.backends.cudnn.deterministic = False
            logger.info("cuDNN deterministic mode disabled for speed")

        # Set GPU memory fraction
        if hasattr(torch.cuda, 'set_per_process_memory_fraction'):
            torch.cuda.set_per_process_memory_fraction(Config.gpu_memory_fraction)
            logger.info(f"GPU memory fraction set to {Config.gpu_memory_fraction}")

        # Log GPU information
        logger.info(f"CUDA Device Count: {torch.cuda.device_count()}")
        logger.info(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

        # Check CUDA memory usage before starting
        logger.info(f"Initial CUDA Memory: Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Set device explicitly
        device = torch.device("cuda:0")
        Config.device = device

        # Log configuration
        logger.info(f"Using device: {Config.device}")
        logger.info(f"Batch size: {Config.batch_size}")
        logger.info(f"Number of epochs: {Config.num_epochs}")
        logger.info(f"Learning rate: {Config.learning_rate}")
        logger.info(f"Mixed precision: {Config.mixed_precision}")
        logger.info(f"Process on GPU: {Config.process_on_gpu}")
        logger.info(f"CONTINUOUS TRAINING ENABLED - Will run all {Config.num_epochs} epochs without stopping")

        # Check for existing checkpoint
        checkpoint_path = os.path.join(Config.models_dir, "latest_checkpoint.pt")
        if Config.resume_training and os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at {checkpoint_path}, will resume training")

        # Create output directories
        os.makedirs(Config.models_dir, exist_ok=True)

        # Create data loader
        data_loader = DataLoader(Config)

        # Load data
        logger.info("Loading data...")
        data_loader.load_data(max_files=Config.max_files)
        logger.info("Data loading completed")

        # Create DataLoaders with GPU optimization
        logger.info("Creating DataLoaders with GPU optimization...")
        train_loader, val_loader, test_loader = data_loader.create_dataloaders()

        # Free up memory after data loading
        del data_loader
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Check memory usage after data loading
        logger.info(f"CUDA Memory after data loading: Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

        # Create model with explicit error handling
        logger.info("Creating model...")
        try:
            # Initialize model
            model = RobustModel(Config)
            model = model.to(device)  # Move model to GPU

            # Verify model is on CUDA
            logger.info(f"Model is on CUDA: {next(model.parameters()).is_cuda}")

            # Log model architecture
            logger.info(f"Model architecture:\n{model}")

            # Check memory usage after model creation
            logger.info(f"CUDA Memory after model creation: Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB, Reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

            # Resume from checkpoint if available
            start_epoch = start_epoch_override
            if Config.resume_training and os.path.exists(checkpoint_path) and start_epoch_override == 0:
                try:
                    logger.info(f"Loading checkpoint from {checkpoint_path}")
                    checkpoint = torch.load(checkpoint_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    start_epoch = checkpoint['epoch']
                    logger.info(f"Resuming from epoch {start_epoch}")
                except Exception as e:
                    logger.error(f"Error loading checkpoint: {e}")
                    logger.info(f"Starting from epoch {start_epoch_override} instead")
                    start_epoch = start_epoch_override
            elif start_epoch_override > 0:
                # If start_epoch_override is specified, try to load the appropriate checkpoint
                specific_checkpoint_path = os.path.join(Config.models_dir, f"checkpoint_epoch_{start_epoch_override}.pt")
                if os.path.exists(specific_checkpoint_path):
                    try:
                        logger.info(f"Loading specific checkpoint from {specific_checkpoint_path}")
                        checkpoint = torch.load(specific_checkpoint_path, map_location=device)
                        model.load_state_dict(checkpoint['model_state_dict'])
                        logger.info(f"Starting from specified epoch {start_epoch_override}")
                    except Exception as e:
                        logger.error(f"Error loading specific checkpoint: {e}")
                        logger.info(f"Starting from epoch {start_epoch_override} anyway")
                else:
                    # If specific checkpoint doesn't exist, try to use the latest checkpoint
                    if os.path.exists(checkpoint_path):
                        try:
                            logger.info(f"Specific checkpoint not found, loading latest checkpoint")
                            checkpoint = torch.load(checkpoint_path, map_location=device)
                            model.load_state_dict(checkpoint['model_state_dict'])
                            logger.info(f"Loaded model state, but will start from specified epoch {start_epoch_override}")
                        except Exception as e:
                            logger.error(f"Error loading latest checkpoint: {e}")
                    else:
                        logger.warning(f"No checkpoint found, but starting from specified epoch {start_epoch_override}")

                # Force the start_epoch to be the specified value
                start_epoch = start_epoch_override

        except Exception as e:
            logger.error(f"Error creating model: {e}")
            raise

        # Train model with explicit error handling
        logger.info("Starting continuous training with GPU acceleration...")
        try:
            # Set up automatic mixed precision if enabled
            if Config.use_amp:
                logger.info("Using automatic mixed precision for faster training")

            # Train the model with continuous execution
            while start_epoch < Config.num_epochs:
                logger.info(f"Starting training from epoch {start_epoch} to {Config.num_epochs}")

                # Train the model
                train_results = train_model(
                    model=model,
                    train_loader=train_loader,
                    val_loader=val_loader,
                    config=Config,
                    start_epoch=start_epoch,
                )

                # Update start_epoch for next iteration if needed
                if os.path.exists(checkpoint_path):
                    try:
                        checkpoint = torch.load(checkpoint_path, map_location=device)
                        start_epoch = checkpoint['epoch']
                        logger.info(f"Updated to epoch {start_epoch}")
                    except Exception as e:
                        logger.error(f"Error reading checkpoint for epoch update: {e}")
                        # Increment manually as fallback
                        start_epoch += 1
                else:
                    # Increment manually as fallback
                    start_epoch += 1

                # Check if we've completed all epochs
                if start_epoch >= Config.num_epochs:
                    logger.info(f"All {Config.num_epochs} epochs completed!")
                    break

                # Clear GPU cache between iterations
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logger.info("CUDA memory cache emptied between iterations")

            logger.info("Training completed successfully!")

            # Save final metrics
            metrics_path = os.path.join(Config.models_dir, "final_metrics.json")
            with open(metrics_path, "w") as f:
                json.dump(train_results, f, indent=2)
            logger.info(f"Final metrics saved to {metrics_path}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            raise

    except Exception as e:
        logger.error(f"Critical error in main function: {e}", exc_info=True)
        import traceback
        traceback.print_exc()

        # Try to clean up CUDA memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            logger.info("CUDA memory cache emptied")

        # Keep the window open to see the error
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()
