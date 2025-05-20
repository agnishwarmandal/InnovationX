"""
Optimized training script for the MQTM system.
"""

import os
import argparse
import logging
import time
import json
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
        logging.FileHandler("optimized_training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Global configuration
class Config:
    """Global configuration."""
    
    # Hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_workers = min(os.cpu_count() or 1, 4)
    pin_memory = True
    mixed_precision = True
    
    # Data
    data_dir = "D:\\INNOX\\Crypto_Data"
    symbols = ["BTC_USDT_5m", "ETH_USDT_5m"]
    history_length = 100
    forecast_horizon = 20
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15
    batch_size = 16
    shuffle = True
    
    # Model
    hidden_dim = 128
    num_layers = 2
    dropout = 0.1
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 50
    
    # Training
    models_dir = "models"
    save_interval = 5
    log_interval = 1
    
    @classmethod
    def update_from_args(cls, args):
        """Update configuration from command line arguments."""
        if args.data_dir:
            cls.data_dir = args.data_dir
        
        if args.symbols:
            cls.symbols = args.symbols
        
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
    
    def load_data(self):
        """Load data for all symbols."""
        for symbol in self.config.symbols:
            file_path = os.path.join(self.config.data_dir, f"{symbol}.csv")
            
            # Check if file exists
            if not os.path.exists(file_path):
                logger.warning(f"File not found: {file_path}")
                continue
            
            # Load CSV file
            logger.info(f"Loading {file_path}...")
            try:
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
    
    def _preprocess_dataframe(self, df):
        """Preprocess dataframe."""
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
        
        return df
    
    def _calculate_features(self, df):
        """Calculate additional features."""
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
    
    def _split_data(self, symbol):
        """Split data into train, validation, and test sets."""
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
    
    def create_sequences(self, df):
        """Create input-output sequences from dataframe."""
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
        """Create PyTorch DataLoaders."""
        # Create datasets
        X_train, y_train = [], []
        X_val, y_val = [], []
        X_test, y_test = [], []
        
        for symbol in self.config.symbols:
            if symbol not in self.data:
                continue
            
            # Create sequences
            X_train_symbol, y_train_symbol = self.create_sequences(self.train_data[symbol])
            X_val_symbol, y_val_symbol = self.create_sequences(self.val_data[symbol])
            X_test_symbol, y_test_symbol = self.create_sequences(self.test_data[symbol])
            
            # Append to lists
            X_train.append(X_train_symbol)
            y_train.append(y_train_symbol)
            X_val.append(X_val_symbol)
            y_val.append(y_val_symbol)
            X_test.append(X_test_symbol)
            y_test.append(y_test_symbol)
        
        # Concatenate arrays
        X_train = np.concatenate(X_train)
        y_train = np.concatenate(y_train)
        X_val = np.concatenate(X_val)
        y_val = np.concatenate(y_val)
        X_test = np.concatenate(X_test)
        y_test = np.concatenate(y_test)
        
        logger.info(f"Created datasets: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")
        
        # Create PyTorch datasets
        train_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long)
        )
        
        val_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_val, dtype=torch.float32),
            torch.tensor(y_val, dtype=torch.long)
        )
        
        test_dataset = torch.utils.data.TensorDataset(
            torch.tensor(X_test, dtype=torch.float32),
            torch.tensor(y_test, dtype=torch.long)
        )
        
        # Create DataLoaders
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=self.config.shuffle,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        
        return train_loader, val_loader, test_loader

# Simple model for testing
class SimpleModel(nn.Module):
    """Simple model for testing."""
    
    def __init__(self, config):
        """Initialize model."""
        super().__init__()
        self.config = config
        
        # Input dimensions
        self.input_dim = 5  # OHLCV
        self.seq_len = config.history_length
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=config.hidden_dim,
            num_layers=config.num_layers,
            batch_first=True,
            dropout=config.dropout if config.num_layers > 1 else 0,
        )
        
        # Fully connected layers
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim // 2, 2),  # Binary classification
        )
    
    def forward(self, x):
        """Forward pass."""
        # LSTM
        lstm_out, _ = self.lstm(x)
        
        # Use last output
        last_out = lstm_out[:, -1]
        
        # Fully connected layers
        logits = self.fc(last_out)
        
        return logits

# Training functions
def train_epoch(model, train_loader, optimizer, criterion, config, epoch):
    """Train for one epoch."""
    model.train()
    
    # Initialize metrics
    train_loss = 0.0
    train_acc = 0.0
    
    # Create progress bar
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    # Initialize scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
    
    # Training loop
    for batch_idx, (data, target) in enumerate(train_loader):
        # Move data to device
        data = data.to(config.device)
        target = target.to(config.device)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass with mixed precision
        if config.mixed_precision:
            with torch.cuda.amp.autocast():
                output = model(data)
                loss = criterion(output, target)
            
            # Backward pass with scaled gradients
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
        
        # Update metrics
        train_loss += loss.item()
        
        # Compute accuracy
        _, predicted = output.max(1)
        train_acc += predicted.eq(target).sum().item() / target.size(0)
        
        # Update progress bar
        progress_bar.update(1)
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{predicted.eq(target).sum().item() / target.size(0):.4f}",
        })
    
    # Close progress bar
    progress_bar.close()
    
    # Compute average metrics
    train_loss /= len(train_loader)
    train_acc /= len(train_loader)
    
    return train_loss, train_acc

def validate(model, val_loader, criterion, config):
    """Validate model."""
    model.eval()
    
    # Initialize metrics
    val_loss = 0.0
    val_acc = 0.0
    
    # Validation loop
    with torch.no_grad():
        for data, target in val_loader:
            # Move data to device
            data = data.to(config.device)
            target = target.to(config.device)
            
            # Forward pass
            output = model(data)
            loss = criterion(output, target)
            
            # Update metrics
            val_loss += loss.item()
            
            # Compute accuracy
            _, predicted = output.max(1)
            val_acc += predicted.eq(target).sum().item() / target.size(0)
    
    # Compute average metrics
    val_loss /= len(val_loader)
    val_acc /= len(val_loader)
    
    return val_loss, val_acc

def train_model(model, train_loader, val_loader, config):
    """Train model."""
    # Create criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    
    # Initialize metrics
    best_val_acc = 0.0
    metrics = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }
    
    # Create models directory
    os.makedirs(config.models_dir, exist_ok=True)
    
    # Training loop
    for epoch in range(config.num_epochs):
        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model=model,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            config=config,
            epoch=epoch,
        )
        
        # Validate
        val_loss, val_acc = validate(
            model=model,
            val_loader=val_loader,
            criterion=criterion,
            config=config,
        )
        
        # Update metrics
        metrics["train_loss"].append(train_loss)
        metrics["train_acc"].append(train_acc)
        metrics["val_loss"].append(val_loss)
        metrics["val_acc"].append(val_acc)
        
        # Log progress
        logger.info(f"Epoch {epoch+1}/{config.num_epochs}, "
                   f"Train Loss: {train_loss:.4f}, "
                   f"Train Acc: {train_acc:.4f}, "
                   f"Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.4f}")
        
        # Save model if it's the best so far
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(config.models_dir, "best_model.pt"))
            logger.info(f"Saved best model with validation accuracy: {val_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_acc": train_acc,
                "val_acc": val_acc,
                "metrics": metrics,
            }, os.path.join(config.models_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            logger.info(f"Saved checkpoint at epoch {epoch+1}")
    
    # Save final model
    torch.save(model.state_dict(), os.path.join(config.models_dir, "final_model.pt"))
    logger.info("Saved final model")
    
    # Save metrics
    with open(os.path.join(config.models_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    
    # Plot metrics
    plot_metrics(metrics, config.models_dir)
    
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized Training Script")
    
    parser.add_argument("--data_dir", type=str, default="D:\\INNOX\\Crypto_Data",
                        help="Directory containing cryptocurrency data")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory to save trained models")
    parser.add_argument("--symbols", type=str, nargs="+", default=None,
                        help="Symbols to train on (if None, use default)")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs to train for")
    parser.add_argument("--learning_rate", type=float, default=None,
                        help="Learning rate for training")
    
    return parser.parse_args()

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Update configuration
    Config.update_from_args(args)
    
    # Log configuration
    logger.info(f"Using device: {Config.device}")
    logger.info(f"Batch size: {Config.batch_size}")
    logger.info(f"Number of epochs: {Config.num_epochs}")
    logger.info(f"Learning rate: {Config.learning_rate}")
    
    # Create data loader
    data_loader = DataLoader(Config)
    
    # Load data
    data_loader.load_data()
    
    # Create DataLoaders
    train_loader, val_loader, test_loader = data_loader.create_dataloaders()
    
    # Create model
    model = SimpleModel(Config)
    model.to(Config.device)
    
    # Log model architecture
    logger.info(f"Model architecture:\n{model}")
    
    # Train model
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=Config,
    )
    
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()
