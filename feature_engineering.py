"""
Advanced Feature Engineering for MQTM Cryptocurrency Trading

This module implements a comprehensive set of technical indicators and their relationships
for cryptocurrency price prediction, focusing on creating robust features for the
Multiverse Quantum-Topological Meta-Learning (MQTM) architecture.
"""

import numpy as np
import pandas as pd
import talib
from scipy import stats
import pywt
from sklearn.preprocessing import StandardScaler, RobustScaler
import warnings
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("feature_engineering.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Suppress specific warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="invalid value encountered in double_scalars")
warnings.filterwarnings("ignore", category=RuntimeWarning, message="divide by zero encountered")

class AdvancedFeatureEngineering:
    """Advanced feature engineering for cryptocurrency data."""

    def __init__(self, use_gpu=True):
        """Initialize feature engineering module."""
        self.use_gpu = use_gpu
        self.scaler = RobustScaler()  # More robust to outliers than StandardScaler
        self.feature_groups = {
            'price_volume': True,
            'moving_averages': True,
            'momentum': True,
            'trend_bands': True,
            'volatility': True,
            'volume_flow': True,
            'cycles': True,
            'entropy': True,
            'spectral': True,  # Enable for MQTM
            'regime': True,
            'relationships': True,
            'topological': True  # MQTM-specific features
        }

    def compute_features(self, df, include_categories=None, exclude_categories=None):
        """Compute all features for a dataframe."""
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Determine which categories to include
        active_categories = {}
        for category, default_state in self.feature_groups.items():
            if include_categories is not None and category in include_categories:
                active_categories[category] = True
            elif exclude_categories is not None and category in exclude_categories:
                active_categories[category] = False
            else:
                active_categories[category] = default_state

        # Log which feature groups are being computed
        logger.info(f"Computing features with the following groups: {active_categories}")

        # Compute features by category
        try:
            if active_categories.get('price_volume', False):
                result = self._add_price_volume_features(result)

            if active_categories.get('moving_averages', False):
                result = self._add_moving_average_features(result)

            if active_categories.get('momentum', False):
                result = self._add_momentum_features(result)

            if active_categories.get('trend_bands', False):
                result = self._add_trend_band_features(result)

            if active_categories.get('volatility', False):
                result = self._add_volatility_features(result)

            if active_categories.get('volume_flow', False):
                result = self._add_volume_flow_features(result)

            if active_categories.get('cycles', False):
                result = self._add_cycle_features(result)

            if active_categories.get('entropy', False):
                result = self._add_entropy_features(result)

            if active_categories.get('spectral', False):
                result = self._add_spectral_features(result)

            if active_categories.get('regime', False):
                result = self._add_regime_features(result)

            if active_categories.get('relationships', False):
                result = self._add_relationship_features(result)

            if active_categories.get('topological', False):
                result = self._add_topological_features(result)

            # Drop NaN values created by indicators that need lookback periods
            initial_rows = len(result)
            result = result.dropna()
            dropped_rows = initial_rows - len(result)
            if dropped_rows > 0:
                logger.info(f"Dropped {dropped_rows} rows with NaN values ({dropped_rows/initial_rows:.2%} of data)")

            # Log the final feature count
            logger.info(f"Generated {len(result.columns)} features")

            return result

        except Exception as e:
            logger.error(f"Error computing features: {e}")
            raise

    def _add_price_volume_features(self, df):
        """Add price and volume primitive features."""
        logger.info("Adding price and volume features")

        # Typical Price, Median Price, Weighted Close
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['median_price'] = (df['high'] + df['low']) / 2
        df['weighted_close'] = (df['high'] + df['low'] + 2 * df['close']) / 4

        # Bar True Range, Log Return, Percent Change
        df['true_range'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=1)
        df['log_return'] = np.log(df['close'] / df['close'].shift(1))
        df['pct_change'] = df['close'].pct_change()

        # Volume features
        df['dollar_volume'] = df['close'] * df['volume']
        if 'number_of_trades' in df.columns:
            df['avg_trade_size'] = df['volume'] / df['number_of_trades']

        # VWAP and deviation
        df['vwap'] = (df['typical_price'] * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap']

        return df

    def _add_moving_average_features(self, df):
        """Add moving average derived features."""
        logger.info("Adding moving average features")

        # Simple Moving Averages
        for n in [5, 8, 13, 21, 34, 55, 89]:
            df[f'sma_{n}'] = talib.SMA(df['close'].values, timeperiod=n)

        # Exponential Moving Averages
        for n in [5, 8, 13, 21, 34]:
            df[f'ema_{n}'] = talib.EMA(df['close'].values, timeperiod=n)

        # Double and Triple EMAs
        df['dema_21'] = talib.DEMA(df['close'].values, timeperiod=21)
        df['tema_21'] = talib.TEMA(df['close'].values, timeperiod=21)

        # Weighted Moving Averages
        df['wma_21'] = talib.WMA(df['close'].values, timeperiod=21)

        # Hull Moving Average (HMA)
        n = 21
        df[f'hma_{n}'] = talib.WMA(2 * talib.WMA(df['close'].values, timeperiod=n//2) -
                                  talib.WMA(df['close'].values, timeperiod=n),
                                  timeperiod=int(np.sqrt(n)))

        # Kaufman Adaptive Moving Average
        df['kama_21'] = talib.KAMA(df['close'].values, timeperiod=21)

        # Moving Average Crossovers
        df['sma_cross_8_21'] = np.where(df['sma_8'] > df['sma_21'], 1, -1)
        df['ema_cross_8_21'] = np.where(df['ema_8'] > df['ema_21'], 1, -1)

        return df

    def _add_momentum_features(self, df):
        """Add momentum and oscillator features."""
        logger.info("Adding momentum features")

        # RSI
        for n in [7, 14, 21]:
            df[f'rsi_{n}'] = talib.RSI(df['close'].values, timeperiod=n)

        # Stochastic
        df['stoch_k_14'], df['stoch_d_14'] = talib.STOCH(df['high'].values, df['low'].values,
                                                        df['close'].values, fastk_period=14,
                                                        slowk_period=3, slowk_matype=0,
                                                        slowd_period=3, slowd_matype=0)

        # Williams %R
        df['willr_14'] = talib.WILLR(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)

        # ROC, CMO, TRIX
        df['roc_10'] = talib.ROC(df['close'].values, timeperiod=10)
        df['cmo_14'] = talib.CMO(df['close'].values, timeperiod=14)
        df['trix_21'] = talib.TRIX(df['close'].values, timeperiod=21)

        # MACD
        df['macd'], df['macd_signal'], df['macd_hist'] = talib.MACD(df['close'].values,
                                                                   fastperiod=12, slowperiod=26,
                                                                   signalperiod=9)

        # Awesome Oscillator
        df['ao'] = talib.SMA(df['median_price'].values, timeperiod=5) - talib.SMA(df['median_price'].values, timeperiod=34)

        # Stochastic RSI
        for n in [14]:
            if f'rsi_{n}' in df.columns:
                df[f'stoch_rsi_{n}'] = talib.STOCHRSI(df['close'].values, timeperiod=n)[0]

        # Money Flow Index
        df['mfi_14'] = talib.MFI(df['high'].values, df['low'].values, df['close'].values, df['volume'].values, timeperiod=14)

        return df

    def _add_trend_band_features(self, df):
        """Add trend and filter band features."""
        logger.info("Adding trend band features")

        # Bollinger Bands
        for n in [20]:
            for std in [2.0]:
                upper, middle, lower = talib.BBANDS(df['close'].values, timeperiod=n,
                                                  nbdevup=std, nbdevdn=std, matype=0)
                df[f'bb_upper_{n}_{int(std*10)}'] = upper
                df[f'bb_middle_{n}_{int(std*10)}'] = middle
                df[f'bb_lower_{n}_{int(std*10)}'] = lower
                df[f'bb_width_{n}_{int(std*10)}'] = (upper - lower) / middle
                df[f'bb_pct_{n}_{int(std*10)}'] = (df['close'] - lower) / (upper - lower)

        # ADX, +DI, -DI
        df['adx'] = talib.ADX(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['plus_di'] = talib.PLUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['minus_di'] = talib.MINUS_DI(df['high'].values, df['low'].values, df['close'].values, timeperiod=14)
        df['adx_trend'] = np.where(df['plus_di'] > df['minus_di'], 1, -1)

        return df

    def _add_volatility_features(self, df):
        """Add volatility and risk metrics."""
        logger.info("Adding volatility features")

        # ATR and Normalized ATR
        for n in [7, 14, 21]:
            df[f'atr_{n}'] = talib.ATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=n)
            df[f'natr_{n}'] = talib.NATR(df['high'].values, df['low'].values, df['close'].values, timeperiod=n)

        # Historical Volatility
        for n in [10, 21]:
            df[f'volatility_{n}'] = df['log_return'].rolling(n).std() * np.sqrt(252)

        return df

    def _add_volume_flow_features(self, df):
        """Add volume flow and money flow features."""
        logger.info("Adding volume flow features")

        # On-Balance Volume
        df['obv'] = talib.OBV(df['close'].values, df['volume'].values)

        # Accumulation/Distribution Line
        df['adl'] = talib.AD(df['high'].values, df['low'].values, df['close'].values, df['volume'].values)

        # Chaikin A/D Oscillator
        df['chaikin_ad_osc'] = talib.ADOSC(df['high'].values, df['low'].values, df['close'].values,
                                          df['volume'].values, fastperiod=3, slowperiod=10)

        return df

    def _add_cycle_features(self, df):
        """Add cycle, Hilbert and fractal analytics."""
        logger.info("Adding cycle features")

        # Fisher Transform of price
        price_series = (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min())
        price_series = 0.5 * np.log((1 + price_series) / (1 - price_series))
        df['fisher_transform'] = price_series

        # Fisher Transform of RSI
        if 'rsi_14' in df.columns:
            rsi_series = (df['rsi_14'] - 5) / 95  # Scale RSI from 0-100 to 0-1
            rsi_series = 0.5 * np.log((1 + rsi_series) / (1 - rsi_series))
            df['fisher_rsi'] = rsi_series

        return df

    def _add_entropy_features(self, df):
        """Add entropy and information theory features."""
        logger.info("Adding entropy features")

        # Shannon Entropy of returns
        def shannon_entropy(x, bins=10):
            counts, _ = np.histogram(x, bins=bins)
            probs = counts / len(x)
            probs = probs[probs > 0]  # Remove zeros
            return -np.sum(probs * np.log2(probs)) if len(probs) > 0 else 0

        # Calculate Shannon entropy for returns over different windows
        for n in [20]:
            df[f'shannon_entropy_{n}'] = df['log_return'].rolling(n).apply(shannon_entropy, raw=True)

        return df

    def _add_spectral_features(self, df):
        """Add spectral and signal processing features."""
        logger.info("Adding spectral features")

        # This is a placeholder - spectral features are computationally expensive
        # and will be implemented in a separate module if needed
        return df

    def _add_regime_features(self, df):
        """Add regime switch and statistical state features."""
        logger.info("Adding regime features")

        # Z-Score of Close vs Rolling Mean
        for n in [20, 50]:
            rolling_mean = df['close'].rolling(n).mean()
            rolling_std = df['close'].rolling(n).std()
            df[f'zscore_{n}'] = (df['close'] - rolling_mean) / rolling_std

        # Z-Score of RSI vs history
        if 'rsi_14' in df.columns:
            rolling_mean = df['rsi_14'].rolling(50).mean()
            rolling_std = df['rsi_14'].rolling(50).std()
            df['rsi_zscore'] = (df['rsi_14'] - rolling_mean) / rolling_std

        # Quantile Position
        for n in [20, 50]:
            min_val = df['close'].rolling(n).min()
            max_val = df['close'].rolling(n).max()
            df[f'quantile_pos_{n}'] = (df['close'] - min_val) / (max_val - min_val)

        return df

    def _add_relationship_features(self, df):
        """Add parametric inter- and intra-indicator relationships."""
        logger.info("Adding relationship features")

        # Basic Algebraic Relationships

        # Ratio of short-term to long-term moving averages
        if 'sma_8' in df.columns and 'sma_21' in df.columns:
            df['sma_ratio_8_21'] = df['sma_8'] / df['sma_21']

        # Spread between RSI periods
        if 'rsi_7' in df.columns and 'rsi_21' in df.columns:
            df['rsi_spread_7_21'] = df['rsi_7'] - df['rsi_21']

        # Volatility-adjusted momentum
        if 'roc_10' in df.columns and 'volatility_21' in df.columns:
            df['vol_adj_momentum'] = df['roc_10'] / df['volatility_21']

        # Trend-momentum fusion
        if 'rsi_14' in df.columns and 'adx' in df.columns:
            df['trend_momentum'] = df['rsi_14'] * df['adx'] / 100

        # ATR-scaled distance from moving average
        if 'sma_21' in df.columns and 'atr_14' in df.columns:
            df['atr_scaled_sma_dist'] = (df['close'] - df['sma_21']) / df['atr_14']

        return df

    def _add_topological_features(self, df):
        """Add topological features specific to MQTM architecture."""
        logger.info("Adding topological features")

        # This implements a simplified version of the topological features
        # that will be expanded in the full MQTM architecture

        # Rolling correlation matrix determinant (measure of feature space volume)
        if len(df) >= 50:  # Need enough data for correlation
            try:
                # Use OHLCV as base features
                base_cols = ['open', 'high', 'low', 'close', 'volume']

                # Calculate rolling correlation matrix determinant
                def rolling_corr_det(x):
                    corr_matrix = np.corrcoef(x.T)
                    # Add small value to diagonal for numerical stability
                    np.fill_diagonal(corr_matrix, corr_matrix.diagonal() + 1e-8)
                    try:
                        return np.linalg.det(corr_matrix)
                    except:
                        return np.nan

                # Apply to rolling windows
                for n in [50]:
                    df[f'topo_corr_det_{n}'] = df[base_cols].rolling(n).apply(
                        rolling_corr_det, raw=False
                    )

                # Calculate persistence measure (simplified version of persistent homology)
                # This measures how long price patterns "persist" over time
                def persistence_measure(x):
                    # Calculate peaks and valleys
                    peaks = []
                    valleys = []
                    for i in range(1, len(x)-1):
                        if x[i] > x[i-1] and x[i] > x[i+1]:
                            peaks.append((i, x[i]))
                        elif x[i] < x[i-1] and x[i] < x[i+1]:
                            valleys.append((i, x[i]))

                    # Calculate persistence pairs
                    pairs = []
                    for valley_idx, valley_val in valleys:
                        # Find the next peak
                        next_peaks = [(i, v) for i, v in peaks if i > valley_idx]
                        if next_peaks:
                            next_peak = min(next_peaks, key=lambda p: p[0])
                            pairs.append((valley_val, next_peak[1]))

                    # Calculate persistence (height of each pair)
                    if pairs:
                        persistences = [peak - valley for valley, peak in pairs]
                        return np.mean(persistences)
                    else:
                        return 0

                # Apply persistence measure to price
                for n in [50]:
                    df[f'topo_persistence_{n}'] = df['close'].rolling(n).apply(
                        persistence_measure, raw=True
                    )

                # Calculate phase space embedding (simplified)
                # This is a basic implementation of delay embedding from chaos theory
                lag = 5
                if len(df) > lag:
                    df['phase_x'] = df['close']
                    df['phase_y'] = df['close'].shift(lag)

                    # Calculate phase space velocity
                    df['phase_velocity'] = np.sqrt(
                        df['close'].diff()**2 + df['close'].shift(lag).diff()**2
                    )

                    # Calculate phase space acceleration
                    df['phase_acceleration'] = df['phase_velocity'].diff()

            except Exception as e:
                logger.warning(f"Could not compute topological features: {e}")

        return df