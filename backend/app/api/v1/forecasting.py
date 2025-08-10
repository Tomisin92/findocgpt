# backend/app/api/v1/forecasting.py
"""
Stage 2: Forecasting API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from datetime import datetime, timedelta
from app.core.stage2_forecasting.predictor import Predictor
from app.core.stage2_forecasting.data_collector import DataCollector
from app.core.stage2_forecasting.feature_engineer import FeatureEngineer

router = APIRouter()

# Initialize components
predictor = Predictor()
data_collector = DataCollector()
feature_engineer = FeatureEngineer()

@router.get("/predict/{symbol}")
async def predict_price(symbol: str, days: int = 30, model_type: str = "lstm"):
    """Predict stock price for specified days"""
    try:
        # Get historical data
        historical_data = await data_collector.get_historical_data(symbol)
        
        # Engineer features
        features = await feature_engineer.create_features(historical_data)
        
        # Make predictions
        predictions = await predictor.predict_prices(symbol, features, days, model_type)
        
        return {
            "symbol": symbol,
            "model_type": model_type,
            "prediction_period": days,
            "predictions": predictions["prices"],
            "confidence_intervals": predictions["confidence"],
            "model_metrics": predictions["metrics"],
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/technical-indicators/{symbol}")
async def get_technical_indicators(symbol: str):
    """Get technical indicators for a symbol"""
    try:
        # Calculate technical indicators
        indicators = await feature_engineer.calculate_technical_indicators(symbol)
        
        return {
            "symbol": symbol,
            "indicators": indicators["values"],
            "signals": indicators["signals"],
            "trend_analysis": indicators["trends"],
            "support_resistance": indicators["levels"],
            "last_updated": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_forecasting_model(symbol: str, model_type: str = "lstm"):
    """Train or retrain forecasting model"""
    try:
        # Start model training
        training_job = await predictor.train_model(symbol, model_type)
        
        return {
            "message": f"Training {model_type} model for {symbol}",
            "job_id": training_job["id"],
            "status": "started",
            "estimated_time": training_job["estimated_time"],
            "progress_endpoint": f"/api/v1/forecasting/training-status/{training_job['id']}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))