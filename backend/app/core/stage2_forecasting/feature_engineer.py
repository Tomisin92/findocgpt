# # backend/app/core/stage2_forecasting/feature_engineer.py
# """
# Feature engineering for financial time series
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Any
# import talib

# class FeatureEngineer:
#     def __init__(self):
#         self.technical_indicators = [
#             "sma_20", "sma_50", "ema_12", "ema_26", 
#             "rsi", "macd", "bollinger_upper", "bollinger_lower",
#             "atr", "stoch_k", "stoch_d"
#         ]
    
#     async def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Create comprehensive feature set for modeling"""
        
#         features = data.copy()
        
#         # Price-based features
#         features = self._add_price_features(features)
        
#         # Technical indicators
#         features = await self._add_technical_indicators(features)
        
#         # Volume features
#         features = self._add_volume_features(features)
        
#         # Volatility features
#         features = self._add_volatility_features(features)
        
#         # Lag features
#         features = self._add_lag_features(features)
        
#         return features.dropna()
    
#     def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Add price-based features"""
        
#         # Returns
#         data["returns"] = data["Close"].pct_change()
#         data["log_returns"] = np.log(data["Close"] / data["Close"].shift(1))
        
#         # Price ratios
#         data["high_low_ratio"] = data["High"] / data["Low"]
#         data["close_open_ratio"] = data["Close"] / data["Open"]
        
#         # Price position within day's range
#         data["price_position"] = (data["Close"] - data["Low"]) / (data["High"] - data["Low"])
        
#         return data
    
#     async def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Add technical indicators"""
        
#         # Moving averages
#         data["sma_20"] = data["Close"].rolling(window=20).mean()
#         data["sma_50"] = data["Close"].rolling(window=50).mean()
#         data["ema_12"] = data["Close"].ewm(span=12).mean()
#         data["ema_26"] = data["Close"].ewm(span=26).mean()
        
#         # RSI
#         delta = data["Close"].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / loss
#         data["rsi"] = 100 - (100 / (1 + rs))
        
#         # MACD
#         data["macd"] = data["ema_12"] - data["ema_26"]
#         data["macd_signal"] = data["macd"].ewm(span=9).mean()
#         data["macd_histogram"] = data["macd"] - data["macd_signal"]
        
#         # Bollinger Bands
#         bb_period = 20
#         bb_std = 2
#         bb_middle = data["Close"].rolling(window=bb_period).mean()
#         bb_std_dev = data["Close"].rolling(window=bb_period).std()
#         data["bollinger_upper"] = bb_middle + (bb_std_dev * bb_std)
#         data["bollinger_lower"] = bb_middle - (bb_std_dev * bb_std)
#         data["bollinger_width"] = data["bollinger_upper"] - data["bollinger_lower"]
        
#         return data
    
#     def _add_volume_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Add volume-based features"""
        
#         # Volume moving averages
#         data["volume_sma_20"] = data["Volume"].rolling(window=20).mean()
#         data["volume_ratio"] = data["Volume"] / data["volume_sma_20"]
        
#         # Price-volume features
#         data["price_volume"] = data["Close"] * data["Volume"]
#         data["vwap"] = (data["price_volume"].rolling(window=20).sum() / 
#                        data["Volume"].rolling(window=20).sum())
        
#         return data
    
#     def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Add volatility features"""
        
#         # Historical volatility
#         data["volatility_20"] = data["returns"].rolling(window=20).std()
#         data["volatility_50"] = data["returns"].rolling(window=50).std()
        
#         # True Range and ATR
#         data["true_range"] = np.maximum(
#             data["High"] - data["Low"],
#             np.maximum(
#                 abs(data["High"] - data["Close"].shift(1)),
#                 abs(data["Low"] - data["Close"].shift(1))
#             )
#         )
#         data["atr"] = data["true_range"].rolling(window=14).mean()
        
#         return data
    
#     def _add_lag_features(self, data: pd.DataFrame) -> pd.DataFrame:
#         """Add lagged features"""
        
#         lag_periods = [1, 2, 3, 5, 10]
        
#         for lag in lag_periods:
#             data[f"close_lag_{lag}"] = data["Close"].shift(lag)
#             data[f"volume_lag_{lag}"] = data["Volume"].shift(lag)
#             data[f"returns_lag_{lag}"] = data["returns"].shift(lag)
        
#         return data
    
#     async def calculate_technical_indicators(self, symbol: str) -> Dict[str, Any]:
#         """Calculate current technical indicators for a symbol"""
        
#         # This would normally get data from DataCollector
#         # Simplified implementation
        
#         indicators = {
#             "values": {
#                 "sma_20": 145.32,
#                 "sma_50": 142.18,
#                 "rsi": 68.5,
#                 "macd": 1.23,
#                 "bollinger_upper": 148.9,
#                 "bollinger_lower": 141.2,
#                 "atr": 2.45
#             },
#             "signals": self._generate_signals(),
#             "trends": self._analyze_trends(),
#             "levels": self._identify_support_resistance()
#         }
        
#         return indicators
    
#     def _generate_signals(self) -> Dict[str, str]:
#         """Generate trading signals from indicators"""
#         return {
#             "trend": "bullish",
#             "momentum": "positive",
#             "volatility": "normal",
#             "volume": "increasing"
#         }
    
#     def _analyze_trends(self) -> Dict[str, Any]:
#         """Analyze price trends"""
#         return {
#             "short_term": "uptrend",
#             "medium_term": "uptrend",
#             "long_term": "sideways",
#             "strength": 0.75
#         }
    
#     def _identify_support_resistance(self) -> Dict[str, float]:
#         """Identify support and resistance levels"""
#         return {
#             "support_1": 138.50,
#             "support_2": 135.20,
#             "resistance_1": 150.80,
#             "resistance_2": 155.40
#         }




# backend/app/core/stage2_forecasting/feature_engineer.py
"""
Feature Engineering for Financial Forecasting
Using pandas and numpy instead of talib
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf

class FeatureEngineer:
    """Feature engineering for financial time series"""
    
    def __init__(self):
        self.features = {}
    
    def create_technical_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create technical analysis features using pandas"""
        df = data.copy()
        
        if 'Close' in df.columns:
            # Moving averages
            df['SMA_5'] = df['Close'].rolling(window=5).mean()
            df['SMA_10'] = df['Close'].rolling(window=10).mean()
            df['SMA_20'] = df['Close'].rolling(window=20).mean()
            df['SMA_50'] = df['Close'].rolling(window=50).mean()
            
            # Exponential moving averages
            df['EMA_12'] = df['Close'].ewm(span=12).mean()
            df['EMA_26'] = df['Close'].ewm(span=26).mean()
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
            df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
            
            # RSI
            df['RSI'] = self._calculate_rsi(df['Close'])
            
            # Bollinger Bands
            bb_window = 20
            df['BB_Middle'] = df['Close'].rolling(window=bb_window).mean()
            bb_std = df['Close'].rolling(window=bb_window).std()
            df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
            df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
            df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']
            df['BB_Position'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
            
            # Price-based features
            df['Price_Change'] = df['Close'].pct_change()
            df['Price_Change_2d'] = df['Close'].pct_change(2)
            df['Price_Change_5d'] = df['Close'].pct_change(5)
            
            # Volatility
            df['Volatility_10d'] = df['Price_Change'].rolling(window=10).std()
            df['Volatility_20d'] = df['Price_Change'].rolling(window=20).std()
        
        if 'Volume' in df.columns:
            # Volume features
            df['Volume_SMA_10'] = df['Volume'].rolling(window=10).mean()
            df['Volume_Ratio'] = df['Volume'] / df['Volume_SMA_10']
            df['Volume_Change'] = df['Volume'].pct_change()
        
        if 'High' in df.columns and 'Low' in df.columns and 'Close' in df.columns:
            # High-Low features
            df['High_Low_Ratio'] = df['High'] / df['Low']
            df['Close_High_Ratio'] = df['Close'] / df['High']
            df['Close_Low_Ratio'] = df['Close'] / df['Low']
            
            # True Range and ATR
            df['True_Range'] = self._calculate_true_range(df)
            df['ATR'] = df['True_Range'].rolling(window=14).mean()
        
        return df
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> pd.Series:
        """Calculate RSI using pandas"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def _calculate_true_range(self, df: pd.DataFrame) -> pd.Series:
        """Calculate True Range"""
        high_low = df['High'] - df['Low']
        high_close_prev = np.abs(df['High'] - df['Close'].shift(1))
        low_close_prev = np.abs(df['Low'] - df['Close'].shift(1))
        true_range = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        return true_range

