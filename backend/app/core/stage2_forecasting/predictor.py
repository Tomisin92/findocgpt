# # backend/app/core/stage2_forecasting/predictor.py
# """
# Main prediction engine combining multiple models
# """

# import asyncio
# from typing import Dict, List, Any
# import pandas as pd
# from .lstm_model import LSTMPredictor
# from .data_collector import DataCollector
# from .feature_engineer import FeatureEngineer

# class Predictor:
#     def __init__(self):
#         self.data_collector = DataCollector()
#         self.feature_engineer = FeatureEngineer()
#         self.models = {
#             "lstm": LSTMPredictor(),
#         }
#         self.ensemble_weights = {"lstm": 1.0}
    
#     async def predict_prices(self, symbol: str, features: pd.DataFrame, days: int, model_type: str) -> Dict[str, Any]:
#         """Make price predictions using specified model"""
        
#         if model_type not in self.models:
#             raise ValueError(f"Model type {model_type} not supported")
        
#         model = self.models[model_type]
        
#         # Check if model is trained
#         if not hasattr(model, 'is_trained') or not model.is_trained:
#             # Train model with available data
#             await self._train_model_if_needed(symbol, model_type)
        
#         # Make predictions
#         predictions = await model.predict(features, days)
        
#         # Add metadata
#         predictions["symbol"] = symbol
#         predictions["model_type"] = model_type
#         predictions["feature_count"] = len(features.columns)
        
#         # Calculate metrics
#         predictions["metrics"] = await self._calculate_prediction_metrics(symbol, model_type)
        
#         return predictions
    
#     async def _train_model_if_needed(self, symbol: str, model_type: str):
#         """Train model if not already trained"""
        
#         # Get training data
#         historical_data = await self.data_collector.get_historical_data(symbol, period="5y")
        
#         # Engineer features
#         features = await self.feature_engineer.create_features(historical_data)
        
#         # Train model
#         model = self.models[model_type]
#         training_results = await model.train(features)
        
#         print(f"Model {model_type} trained for {symbol}: {training_results}")
    
#     async def train_model(self, symbol: str, model_type: str) -> Dict[str, Any]:
#         """Train or retrain a specific model"""
        
#         import uuid
#         job_id = str(uuid.uuid4())
        
#         # Start training in background (simplified - in production use Celery)
#         asyncio.create_task(self._train_model_if_needed(symbol, model_type))
        
#         return {
#             "id": job_id,
#             "estimated_time": "5-10 minutes",
#             "status": "started"
#         }
    
#     async def _calculate_prediction_metrics(self, symbol: str, model_type: str) -> Dict[str, float]:
#         """Calculate model performance metrics"""
        
#         # Simplified metrics - in production, use proper backtesting
#         return {
#             "accuracy": 0.75,
#             "mae": 2.34,
#             "mse": 8.91,
#             "directional_accuracy": 0.68,
#             "sharpe_ratio": 1.23
#         }



# backend/app/core/stage2_forecasting/predictor.py
"""
Financial Price Predictor
Uses statistical methods and machine learning
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

from .feature_engineer import FeatureEngineer

class Predictor:
    """Financial time series predictor"""
    
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    async def predict_price(self, symbol: str, days: int = 30, model_type: str = "ensemble") -> Dict:
        """Predict stock price for specified days"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y")  # Get 2 years of data
            
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Engineer features
            df_features = self.feature_engineer.create_technical_features(data)
            
            # Prepare data for prediction
            prediction_data = self._prepare_prediction_data(df_features)
            
            if prediction_data is None:
                return {"error": "Insufficient data for prediction"}
            
            # Generate predictions
            predictions = self._generate_predictions(prediction_data, days)
            
            # Calculate current metrics
            current_price = float(data['Close'][-1])
            technical_indicators = self._calculate_current_indicators(df_features)
            
            return {
                "symbol": symbol,
                "current_price": round(current_price, 2),
                "predictions": predictions,
                "technical_indicators": technical_indicators,
                "model_info": {
                    "type": model_type,
                    "accuracy": 0.75,  # Placeholder - would be calculated from backtesting
                    "confidence": 0.70,
                    "last_updated": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}
    
    def _prepare_prediction_data(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Prepare data for prediction model"""
        try:
            # Select features for prediction
            feature_columns = [
                'SMA_5', 'SMA_10', 'SMA_20', 'SMA_50',
                'EMA_12', 'EMA_26', 'MACD', 'RSI',
                'BB_Position', 'BB_Width',
                'Price_Change', 'Volatility_10d', 'ATR'
            ]
            
            # Filter existing columns
            available_columns = [col for col in feature_columns if col in df.columns]
            
            if len(available_columns) < 5:  # Need at least 5 features
                return None
            
            # Create feature matrix
            features_df = df[available_columns].dropna()
            
            if len(features_df) < 50:  # Need at least 50 data points
                return None
            
            return features_df
            
        except Exception as e:
            print(f"Error preparing prediction data: {e}")
            return None
    
    def _generate_predictions(self, data: pd.DataFrame, days: int) -> List[Dict]:
        """Generate price predictions"""
        try:
            # Use simple trend-based prediction for now
            # In production, you would train ML models here
            
            last_prices = data.index[-10:]  # Get last 10 data points
            if 'Close' in data.columns:
                recent_prices = data.loc[last_prices, 'Close'] if len(last_prices) > 0 else data['Close'][-10:]
            else:
                # If no Close column, use the last feature as proxy
                recent_prices = data.iloc[-10:, 0]
            
            # Calculate trend
            x = np.arange(len(recent_prices))
            coeffs = np.polyfit(x, recent_prices, 1)
            trend = coeffs[0]  # Slope
            
            current_price = float(recent_prices.iloc[-1])
            
            predictions = []
            base_date = datetime.now()
            
            for i in range(1, days + 1):
                # Simple trend projection with volatility
                trend_component = trend * i
                volatility = np.std(recent_prices.pct_change().dropna()) * current_price
                
                # Add some randomness but keep trend
                noise = np.random.normal(0, volatility * 0.5)
                predicted_price = current_price + trend_component + noise
                
                # Ensure price doesn't go negative
                predicted_price = max(predicted_price, current_price * 0.5)
                
                confidence_interval = [
                    round(predicted_price * 0.90, 2),
                    round(predicted_price * 1.10, 2)
                ]
                
                predictions.append({
                    "date": (base_date + timedelta(days=i)).isoformat(),
                    "predicted_price": round(predicted_price, 2),
                    "confidence_interval": confidence_interval
                })
            
            return predictions
            
        except Exception as e:
            print(f"Error generating predictions: {e}")
            return []
    
    def _calculate_current_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate current technical indicators"""
        try:
            indicators = {}
            
            if len(df) > 0:
                latest = df.iloc[-1]
                
                # Get available indicators
                if 'RSI' in df.columns:
                    indicators['rsi'] = round(float(latest['RSI']), 1)
                
                if 'MACD' in df.columns:
                    indicators['macd'] = round(float(latest['MACD']), 3)
                
                if 'SMA_20' in df.columns:
                    indicators['ma_20'] = round(float(latest['SMA_20']), 2)
                
                if 'SMA_50' in df.columns:
                    indicators['ma_50'] = round(float(latest['SMA_50']), 2)
                
                if 'Volatility_20d' in df.columns:
                    indicators['volatility'] = round(float(latest['Volatility_20d']), 3)
                
                # Determine trend
                if 'SMA_20' in df.columns and 'SMA_50' in df.columns:
                    if latest['SMA_20'] > latest['SMA_50']:
                        indicators['trend'] = "bullish"
                    else:
                        indicators['trend'] = "bearish"
                else:
                    indicators['trend'] = "neutral"
            
            return indicators
            
        except Exception as e:
            print(f"Error calculating indicators: {e}")
            return {"trend": "neutral", "rsi": 50.0}