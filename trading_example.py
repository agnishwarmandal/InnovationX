"""
Example script demonstrating how to use the MQTM system for trading.
"""

import os
import argparse
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import ccxt

from mqtm.config import config
from mqtm.data_engineering.data_fetcher import BinanceDataFetcher
from mqtm.data_engineering.data_processor import OHLCVProcessor
from mqtm.topo_quantum_encoder.encoder import TopoQuantumEncoder
from mqtm.superposition_pool.superposition_model import SuperpositionPool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="MQTM Trading Example")
    
    parser.add_argument("--symbol", type=str, default="BTCUSDT",
                        help="Trading symbol to use")
    parser.add_argument("--models_dir", type=str, default="models",
                        help="Directory with trained models")
    parser.add_argument("--mode", type=str, default="backtest", choices=["backtest", "paper", "live"],
                        help="Trading mode: backtest, paper, or live")
    parser.add_argument("--days", type=int, default=7,
                        help="Number of days for backtesting")
    parser.add_argument("--capital", type=float, default=10000.0,
                        help="Initial capital")
    parser.add_argument("--risk_per_trade", type=float, default=0.01,
                        help="Risk per trade (fraction of capital)")
    
    return parser.parse_args()

def load_models(models_dir):
    """Load trained MQTM models."""
    logger.info(f"Loading models from {models_dir}...")
    
    # Create models
    tqe = TopoQuantumEncoder()
    
    sp3 = SuperpositionPool(
        input_dim=tqe.total_features_dim,
    )
    
    # Load models if available
    tqe_path = os.path.join(models_dir, "tqe.pt")
    if os.path.exists(tqe_path):
        tqe.load(tqe_path)
        logger.info(f"Loaded TQE from {tqe_path}")
    else:
        logger.warning(f"TQE model not found at {tqe_path}")
    
    sp3_path = os.path.join(models_dir, "sp3.pt")
    if os.path.exists(sp3_path):
        sp3.load(sp3_path)
        logger.info(f"Loaded SP3 from {sp3_path}")
    else:
        logger.warning(f"SP3 model not found at {sp3_path}")
    
    return tqe, sp3

def compute_regime_features(ohlcv_df):
    """Compute regime features from OHLCV data."""
    # Compute volatility percentile
    returns = ohlcv_df["close"].pct_change()
    volatility = returns.rolling(20).std().fillna(0)
    volatility_percentile = volatility.rolling(100).rank(pct=True).fillna(0.5)
    
    # Simulate funding rate (in practice, would fetch from exchange)
    # For simplicity, we use a random value
    funding_sign = np.random.choice([-1, 1], size=len(ohlcv_df))
    
    # Create regime features
    regime = np.column_stack((volatility_percentile, funding_sign))
    
    return regime

def preprocess_data(ohlcv_df, window_size=120):
    """Preprocess OHLCV data for prediction."""
    # Create sliding windows
    X = []
    timestamps = []
    
    for i in range(len(ohlcv_df) - window_size + 1):
        # Extract window
        window = ohlcv_df.iloc[i:i+window_size]
        
        # Extract OHLCV
        open_price = window["open"].values
        high = window["high"].values
        low = window["low"].values
        close = window["close"].values
        volume = window["volume"].values
        
        # Stack OHLCV
        ohlcv = np.stack([open_price, high, low, close, volume])
        
        # Add to list
        X.append(ohlcv)
        timestamps.append(window.index[-1])
    
    # Convert to numpy array
    X = np.array(X)
    
    # Compute regime features
    regime = compute_regime_features(ohlcv_df.iloc[window_size-1:])
    
    return X, regime, timestamps

def make_predictions(tqe, sp3, X, regime):
    """Make predictions using the MQTM models."""
    # Convert to torch tensor
    X_torch = torch.tensor(X, dtype=torch.float32, device=config.hardware.device)
    regime_torch = torch.tensor(regime, dtype=torch.float32, device=config.hardware.device)
    
    # Extract features with TQE
    tqe.eval()
    with torch.no_grad():
        features = tqe(X_torch)
    
    # Make predictions with SP3
    sp3.eval()
    with torch.no_grad():
        outputs = sp3(features, regime_torch)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = torch.argmax(probabilities, dim=1)
    
    # Convert to numpy
    probabilities = probabilities.cpu().numpy()
    predictions = predictions.cpu().numpy()
    
    # Map predictions to labels
    labels = ["Down", "Flat", "Up"]
    prediction_labels = [labels[p] for p in predictions]
    
    return predictions, probabilities, prediction_labels

def backtest(tqe, sp3, symbol, days, capital, risk_per_trade):
    """Run a backtest."""
    logger.info(f"Running backtest for {symbol} over {days} days...")
    
    # Fetch historical data
    fetcher = BinanceDataFetcher(symbols=[symbol])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    data = fetcher.fetch_historical_data(
        start_date=start_date,
        end_date=end_date,
        save=False
    )
    
    # Get OHLCV data
    ohlcv_df = data[symbol]
    
    # Preprocess data
    X, regime, timestamps = preprocess_data(ohlcv_df)
    
    # Make predictions
    predictions, probabilities, prediction_labels = make_predictions(tqe, sp3, X, regime)
    
    # Run backtest
    balance = capital
    position = None
    position_size = 0
    entry_price = 0
    trades = []
    
    for i in range(len(predictions)):
        timestamp = timestamps[i]
        prediction = predictions[i]
        confidence = probabilities[i, prediction]
        
        # Get current price
        current_price = ohlcv_df.loc[timestamp, "close"]
        
        # Check if we have an open position
        if position is not None:
            # Check if we should close the position
            if (position == "long" and prediction == 0) or (position == "short" and prediction == 2):
                # Close position
                pnl = position_size * (current_price - entry_price) if position == "long" else position_size * (entry_price - current_price)
                balance += pnl
                
                # Record trade
                trades.append({
                    "timestamp": timestamp,
                    "action": "close",
                    "position": position,
                    "price": current_price,
                    "size": position_size,
                    "pnl": pnl,
                    "balance": balance
                })
                
                # Reset position
                position = None
                position_size = 0
                entry_price = 0
        
        # Check if we should open a position
        if position is None and prediction != 1 and confidence >= 0.55:
            # Calculate position size
            position_size = balance * risk_per_trade / 0.03  # Assume 3% stop loss
            
            # Open position
            if prediction == 2:  # Up
                position = "long"
            else:  # Down
                position = "short"
            
            entry_price = current_price
            
            # Record trade
            trades.append({
                "timestamp": timestamp,
                "action": "open",
                "position": position,
                "price": current_price,
                "size": position_size,
                "balance": balance
            })
    
    # Close any open position at the end
    if position is not None:
        # Get last price
        last_price = ohlcv_df.iloc[-1]["close"]
        
        # Close position
        pnl = position_size * (last_price - entry_price) if position == "long" else position_size * (entry_price - last_price)
        balance += pnl
        
        # Record trade
        trades.append({
            "timestamp": timestamps[-1],
            "action": "close",
            "position": position,
            "price": last_price,
            "size": position_size,
            "pnl": pnl,
            "balance": balance
        })
    
    # Calculate performance metrics
    if trades:
        trades_df = pd.DataFrame(trades)
        
        # Calculate returns
        returns = []
        for i in range(0, len(trades_df), 2):
            if i + 1 < len(trades_df):
                open_trade = trades_df.iloc[i]
                close_trade = trades_df.iloc[i+1]
                returns.append(close_trade["pnl"] / (open_trade["balance"] * risk_per_trade))
        
        returns = np.array(returns)
        
        # Calculate metrics
        num_trades = len(returns)
        win_rate = np.mean(returns > 0) * 100
        avg_return = np.mean(returns) * 100
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        max_drawdown = np.min(np.minimum.accumulate(np.concatenate([[0], returns])))
        
        logger.info(f"Backtest results:")
        logger.info(f"  Initial capital: ${capital:.2f}")
        logger.info(f"  Final balance: ${balance:.2f}")
        logger.info(f"  Return: {(balance/capital - 1) * 100:.2f}%")
        logger.info(f"  Number of trades: {num_trades}")
        logger.info(f"  Win rate: {win_rate:.2f}%")
        logger.info(f"  Average return per trade: {avg_return:.2f}%")
        logger.info(f"  Sharpe ratio: {sharpe:.2f}")
        logger.info(f"  Maximum drawdown: {max_drawdown * 100:.2f}%")
        
        # Plot equity curve
        plt.figure(figsize=(12, 6))
        plt.plot(trades_df["timestamp"], trades_df["balance"])
        plt.title("Equity Curve")
        plt.xlabel("Time")
        plt.ylabel("Balance ($)")
        plt.grid(True)
        plt.savefig("equity_curve.png")
        logger.info("Saved equity curve to equity_curve.png")
    else:
        logger.info("No trades executed during backtest period.")

def paper_trade(tqe, sp3, symbol, capital, risk_per_trade):
    """Run paper trading."""
    logger.info(f"Running paper trading for {symbol}...")
    
    # Create exchange instance
    exchange = ccxt.binance({
        'enableRateLimit': True,
    })
    
    # Initialize state
    balance = capital
    position = None
    position_size = 0
    entry_price = 0
    trades = []
    
    # Trading loop
    try:
        while True:
            # Fetch latest OHLCV data
            ohlcv = exchange.fetch_ohlcv(symbol, '5m', limit=120)
            
            # Convert to DataFrame
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            # Preprocess data
            X = np.array([np.stack([
                df["open"].values,
                df["high"].values,
                df["low"].values,
                df["close"].values,
                df["volume"].values
            ])])
            
            # Compute regime features
            regime = compute_regime_features(df)
            
            # Make prediction
            predictions, probabilities, prediction_labels = make_predictions(tqe, sp3, X, regime)
            
            # Get latest prediction
            prediction = predictions[-1]
            confidence = probabilities[-1, prediction]
            prediction_label = prediction_labels[-1]
            
            # Get current price
            current_price = df.iloc[-1]["close"]
            
            # Log prediction
            logger.info(f"Timestamp: {df.index[-1]}")
            logger.info(f"Current price: {current_price}")
            logger.info(f"Prediction: {prediction_label} (confidence: {confidence:.2f})")
            
            # Check if we have an open position
            if position is not None:
                # Check if we should close the position
                if (position == "long" and prediction == 0) or (position == "short" and prediction == 2):
                    # Close position
                    pnl = position_size * (current_price - entry_price) if position == "long" else position_size * (entry_price - current_price)
                    balance += pnl
                    
                    # Record trade
                    trades.append({
                        "timestamp": df.index[-1],
                        "action": "close",
                        "position": position,
                        "price": current_price,
                        "size": position_size,
                        "pnl": pnl,
                        "balance": balance
                    })
                    
                    logger.info(f"Closed {position} position at {current_price}")
                    logger.info(f"PnL: {pnl:.2f}")
                    logger.info(f"Balance: {balance:.2f}")
                    
                    # Reset position
                    position = None
                    position_size = 0
                    entry_price = 0
            
            # Check if we should open a position
            if position is None and prediction != 1 and confidence >= 0.55:
                # Calculate position size
                position_size = balance * risk_per_trade / 0.03  # Assume 3% stop loss
                
                # Open position
                if prediction == 2:  # Up
                    position = "long"
                else:  # Down
                    position = "short"
                
                entry_price = current_price
                
                # Record trade
                trades.append({
                    "timestamp": df.index[-1],
                    "action": "open",
                    "position": position,
                    "price": current_price,
                    "size": position_size,
                    "balance": balance
                })
                
                logger.info(f"Opened {position} position at {current_price}")
                logger.info(f"Position size: {position_size:.2f}")
            
            # Save trades to file
            with open("paper_trades.json", "w") as f:
                json.dump(trades, f, indent=2, default=str)
            
            # Wait for next candle
            logger.info("Waiting for next candle...")
            time.sleep(60)  # Wait 1 minute
    
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user.")
        
        # Save final trades to file
        with open("paper_trades.json", "w") as f:
            json.dump(trades, f, indent=2, default=str)

def main():
    """Main function."""
    # Parse arguments
    args = parse_args()
    
    # Load models
    tqe, sp3 = load_models(args.models_dir)
    
    # Run in specified mode
    if args.mode == "backtest":
        backtest(tqe, sp3, args.symbol, args.days, args.capital, args.risk_per_trade)
    elif args.mode == "paper":
        paper_trade(tqe, sp3, args.symbol, args.capital, args.risk_per_trade)
    elif args.mode == "live":
        logger.warning("Live trading not implemented in this example.")
    
    logger.info("Trading example completed.")

if __name__ == "__main__":
    main()
