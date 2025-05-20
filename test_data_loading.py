"""
Script to test data loading and basic model functionality.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# Data directory
DATA_DIR = "D:\\INNOX\\Crypto_Data"
SYMBOLS = ["BTC_USDT_5m", "ETH_USDT_5m"]
BATCH_SIZE = 16
HISTORY_LENGTH = 100
FORECAST_HORIZON = 20

def load_csv_file(file_path):
    """Load a CSV file."""
    print(f"Loading {file_path}...")
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} rows with columns: {df.columns.tolist()}")
        return df
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def preprocess_dataframe(df):
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
        print(f"Missing columns: {missing_columns}")

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
    df = calculate_features(df)

    return df

def calculate_features(df):
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

def create_sequences(df, history_length=HISTORY_LENGTH, forecast_horizon=FORECAST_HORIZON):
    """Create input-output sequences from dataframe."""
    # Extract OHLCV data
    ohlcv = df[["open", "high", "low", "close", "volume"]].values

    # Create sequences
    X, y = [], []

    for i in range(len(df) - history_length - forecast_horizon + 1):
        # Input sequence
        X.append(ohlcv[i:i+history_length])

        # Output sequence (future close prices)
        future_close = ohlcv[i+history_length:i+history_length+forecast_horizon, 3]

        # Calculate direction (1 if price goes up, 0 if down)
        current_close = ohlcv[i+history_length-1, 3]
        direction = 1 if future_close[-1] > current_close else 0

        y.append(direction)

    return np.array(X), np.array(y)

def create_dataloaders(data_dict, batch_size=BATCH_SIZE):
    """Create PyTorch DataLoaders."""
    # Create datasets
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []

    for symbol, datasets in data_dict.items():
        # Append to lists
        X_train.append(datasets["X_train"])
        y_train.append(datasets["y_train"])
        X_val.append(datasets["X_val"])
        y_val.append(datasets["y_val"])
        X_test.append(datasets["X_test"])
        y_test.append(datasets["y_test"])

    # Concatenate arrays
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)
    X_val = np.concatenate(X_val)
    y_val = np.concatenate(y_val)
    X_test = np.concatenate(X_test)
    y_test = np.concatenate(y_test)

    print(f"Created datasets: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}")

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
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader

def main():
    """Main function."""
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Check available GPU memory
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"Available GPU memory: {gpu_memory:.2f} GB")

    # Load data
    data_dict = {}

    for symbol in SYMBOLS:
        file_path = os.path.join(DATA_DIR, f"{symbol}.csv")

        # Load CSV file
        df = load_csv_file(file_path)

        if df is not None:
            # Preprocess dataframe
            df = preprocess_dataframe(df)

            # Split data
            n = len(df)
            train_idx = int(n * 0.7)
            val_idx = train_idx + int(n * 0.15)

            train_df = df.iloc[:train_idx]
            val_df = df.iloc[train_idx:val_idx]
            test_df = df.iloc[val_idx:]

            print(f"Split {symbol} data: train={len(train_df)}, val={len(val_df)}, test={len(test_df)}")

            # Create sequences
            X_train, y_train = create_sequences(train_df)
            X_val, y_val = create_sequences(val_df)
            X_test, y_test = create_sequences(test_df)

            print(f"Created sequences for {symbol}: X_train={X_train.shape}, y_train={y_train.shape}")

            # Store in dictionary
            data_dict[symbol] = {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
            }

    # Create DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(data_dict)

    # Test data loading
    print("Testing data loading...")
    start_time = time.time()

    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        # Move data to device
        data = data.to(device)
        target = target.to(device)

        # Print batch information
        if batch_idx == 0:
            print(f"Batch shape: {data.shape}, Target shape: {target.shape}")

        # Limit to 10 batches for testing
        if batch_idx >= 10:
            break

    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Data loading test completed in {elapsed_time:.2f} seconds")

    # Estimate training time
    num_batches = len(train_loader)
    time_per_batch = elapsed_time / min(10, num_batches)

    # Estimate time for full training
    epochs = 50
    estimated_training_time = time_per_batch * num_batches * epochs

    print(f"Estimated training time for MQTM models (50 epochs): {estimated_training_time/3600:.2f} hours")

    # Estimate time for ASP training
    num_episodes = 1000
    estimated_asp_time = time_per_batch * 20 * num_episodes  # Assuming 20 steps per episode

    print(f"Estimated training time for ASP framework (1000 episodes): {estimated_asp_time/3600:.2f} hours")

    # Estimate time for MGI training
    num_iterations = 1000
    estimated_mgi_time = time_per_batch * 20 * num_iterations  # Assuming 20 steps per iteration

    print(f"Estimated training time for MGI module (1000 iterations): {estimated_mgi_time/3600:.2f} hours")

    # Estimate time for BOM training
    estimated_bom_time = time_per_batch * num_iterations  # Assuming 1 step per iteration

    print(f"Estimated training time for BOM module (1000 iterations): {estimated_bom_time/3600:.2f} hours")

    # Estimate total training time
    total_training_time = estimated_training_time + estimated_asp_time + estimated_mgi_time + estimated_bom_time

    print(f"Estimated total training time: {total_training_time/3600:.2f} hours")

    # Plot sample data
    print("Plotting sample data...")

    for symbol in SYMBOLS:
        if symbol in data_dict:
            # Get sample data
            X = data_dict[symbol]["X_train"][0]

            # Plot OHLC
            plt.figure(figsize=(12, 8))

            plt.subplot(2, 1, 1)
            plt.plot(X[:, 0], label="Open")
            plt.plot(X[:, 1], label="High")
            plt.plot(X[:, 2], label="Low")
            plt.plot(X[:, 3], label="Close")
            plt.title(f"{symbol} - OHLC")
            plt.legend()
            plt.grid(True)

            plt.subplot(2, 1, 2)
            plt.plot(X[:, 4], label="Volume")
            plt.title(f"{symbol} - Volume")
            plt.legend()
            plt.grid(True)

            plt.tight_layout()
            plt.savefig(f"{symbol}_sample.png")
            plt.close()

if __name__ == "__main__":
    main()
