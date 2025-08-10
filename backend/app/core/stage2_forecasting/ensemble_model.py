# backend/app/core/stage2_forecasting/ensemble_model.py
"""
Ensemble model combining multiple forecasting approaches
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Any, Tuple, Optional
import joblib
from datetime import datetime
import asyncio

from .lstm_model import LSTMPredictor
from .feature_engineer import FeatureEngineer

class EnsemblePredictor:
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_engineer = FeatureEngineer()
        
        # Initialize individual models
        self.lstm_model = LSTMPredictor()
        self.rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.lr_model = LinearRegression()
        
    async def train(self, data: pd.DataFrame, target_column: str = "Close") -> Dict[str, Any]:
        """Train ensemble model with multiple algorithms"""
        
        # Prepare features
        features_df = await self.feature_engineer.create_features(data)
        features_df = features_df.dropna()
        
        # Separate features and target
        feature_columns = [col for col in features_df.columns if col != target_column]
        X = features_df[feature_columns].values
        y = features_df[target_column].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        train_size = int(0.8 * len(X_scaled))
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Train individual models
        model_performances = {}
        
        # 1. Train LSTM
        print("Training LSTM model...")
        lstm_results = await self.lstm_model.train(features_df, target_column)
        lstm_predictions = await self._get_lstm_predictions(features_df, len(y_test))
        lstm_mae = np.mean(np.abs(lstm_predictions - y_test))
        model_performances['lstm'] = {'mae': lstm_mae, 'model': self.lstm_model}
        
        # 2. Train Random Forest
        print("Training Random Forest model...")
        self.rf_model.fit(X_train, y_train)
        rf_predictions = self.rf_model.predict(X_test)
        rf_mae = np.mean(np.abs(rf_predictions - y_test))
        model_performances['random_forest'] = {'mae': rf_mae, 'model': self.rf_model}
        
        # 3. Train Gradient Boosting
        print("Training Gradient Boosting model...")
        self.gb_model.fit(X_train, y_train)
        gb_predictions = self.gb_model.predict(X_test)
        gb_mae = np.mean(np.abs(gb_predictions - y_test))
        model_performances['gradient_boosting'] = {'mae': gb_mae, 'model': self.gb_model}
        
        # 4. Train Linear Regression
        print("Training Linear Regression model...")
        self.lr_model.fit(X_train, y_train)
        lr_predictions = self.lr_model.predict(X_test)
        lr_mae = np.mean(np.abs(lr_predictions - y_test))
        model_performances['linear_regression'] = {'mae': lr_mae, 'model': self.lr_model}
        
        # Calculate ensemble weights based on inverse MAE
        total_inverse_mae = sum(1/perf['mae'] for perf in model_performances.values())
        
        for model_name, perf in model_performances.items():
            weight = (1/perf['mae']) / total_inverse_mae
            self.weights[model_name] = weight
            self.models[model_name] = perf['model']
        
        # Test ensemble performance
        ensemble_predictions = await self._make_ensemble_prediction(X_test)
        ensemble_mae = np.mean(np.abs(ensemble_predictions - y_test))
        
        self.is_trained = True
        
        return {
            'individual_performances': {name: perf['mae'] for name, perf in model_performances.items()},
            'ensemble_mae': ensemble_mae,
            'model_weights': self.weights,
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features_used': len(feature_columns)
        }
    
    async def _get_lstm_predictions(self, data: pd.DataFrame, n_predictions: int) -> np.ndarray:
        """Get predictions from LSTM model"""
        if not self.lstm_model.is_trained:
            return np.zeros(n_predictions)  # Return zeros if LSTM not trained
        
        # Use last part of data for predictions
        lstm_result = await self.lstm_model.predict(data, n_predictions)
        return np.array(lstm_result['predictions'])
    
    async def _make_ensemble_prediction(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble prediction from all models"""
        predictions = {}
        
        # Get predictions from each model
        if 'random_forest' in self.models:
            predictions['random_forest'] = self.models['random_forest'].predict(X)
        
        if 'gradient_boosting' in self.models:
            predictions['gradient_boosting'] = self.models['gradient_boosting'].predict(X)
        
        if 'linear_regression' in self.models:
            predictions['linear_regression'] = self.models['linear_regression'].predict(X)
        
        # LSTM predictions would need special handling due to sequence requirements
        # For simplicity, using other models' average for missing LSTM
        if 'lstm' not in predictions and len(predictions) > 0:
            predictions['lstm'] = np.mean(list(predictions.values()), axis=0)
        
        # Weighted ensemble
        ensemble_pred = np.zeros(len(X))
        total_weight = 0
        
        for model_name, pred in predictions.items():
            weight = self.weights.get(model_name, 0.25)  # Default equal weight
            ensemble_pred += weight * pred
            total_weight += weight
        
        # Normalize if weights don't sum to 1
        if total_weight > 0:
            ensemble_pred /= total_weight
        
        return ensemble_pred
    
    async def predict(self, data: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
        """Make ensemble predictions for future periods"""
        
        if not self.is_trained:
            raise ValueError("Ensemble model must be trained before making predictions")
        
        # Prepare features
        features_df = await self.feature_engineer.create_features(data)
        features_df = features_df.dropna()
        
        # Get feature columns (excluding target)
        feature_columns = [col for col in features_df.columns if col not in ['Close', 'Date']]
        
        predictions = []
        confidence_intervals = []
        model_contributions = {name: [] for name in self.models.keys()}
        
        # Make predictions for each day
        current_features = features_df[feature_columns].iloc[-1:].values
        current_features_scaled = self.scaler.transform(current_features)
        
        for day in range(days):
            # Get individual model predictions
            individual_predictions = {}
            
            # Random Forest
            if 'random_forest' in self.models:
                rf_pred = self.models['random_forest'].predict(current_features_scaled)[0]
                individual_predictions['random_forest'] = rf_pred
                model_contributions['random_forest'].append(rf_pred)
            
            # Gradient Boosting
            if 'gradient_boosting' in self.models:
                gb_pred = self.models['gradient_boosting'].predict(current_features_scaled)[0]
                individual_predictions['gradient_boosting'] = gb_pred
                model_contributions['gradient_boosting'].append(gb_pred)
            
            # Linear Regression
            if 'linear_regression' in self.models:
                lr_pred = self.models['linear_regression'].predict(current_features_scaled)[0]
                individual_predictions['linear_regression'] = lr_pred
                model_contributions['linear_regression'].append(lr_pred)
            
            # LSTM prediction (simplified)
            if 'lstm' in self.models:
                # For demo purposes, use trend from other models
                if individual_predictions:
                    lstm_pred = np.mean(list(individual_predictions.values()))
                    individual_predictions['lstm'] = lstm_pred
                    model_contributions['lstm'].append(lstm_pred)
            
            # Calculate weighted ensemble prediction
            ensemble_pred = 0
            total_weight = 0
            
            for model_name, pred in individual_predictions.items():
                weight = self.weights.get(model_name, 0.25)
                ensemble_pred += weight * pred
                total_weight += weight
            
            if total_weight > 0:
                ensemble_pred /= total_weight
            
            predictions.append(ensemble_pred)
            
            # Calculate confidence interval based on model disagreement
            pred_values = list(individual_predictions.values())
            if len(pred_values) > 1:
                pred_std = np.std(pred_values)
                lower_bound = ensemble_pred - 1.96 * pred_std
                upper_bound = ensemble_pred + 1.96 * pred_std
            else:
                # Default 5% confidence interval
                lower_bound = ensemble_pred * 0.95
                upper_bound = ensemble_pred * 1.05
            
            confidence_intervals.append([lower_bound, upper_bound])
            
            # Update features for next prediction (simplified)
            # In practice, you'd need more sophisticated feature updating
            current_features_scaled = current_features_scaled  # Keep same for demo
        
        return {
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'model_contributions': model_contributions,
            'model_weights': self.weights,
            'ensemble_type': 'weighted_average',
            'models_used': list(self.models.keys()),
            'prediction_days': days
        }
    
    def get_feature_importance(self) -> Dict[str, Dict[str, float]]:
        """Get feature importance from tree-based models"""
        
        importance_dict = {}
        
        # Random Forest importance
        if 'random_forest' in self.models and hasattr(self.models['random_forest'], 'feature_importances_'):
            importance_dict['random_forest'] = {
                f'feature_{i}': imp for i, imp in enumerate(self.models['random_forest'].feature_importances_)
            }
        
        # Gradient Boosting importance
        if 'gradient_boosting' in self.models and hasattr(self.models['gradient_boosting'], 'feature_importances_'):
            importance_dict['gradient_boosting'] = {
                f'feature_{i}': imp for i, imp in enumerate(self.models['gradient_boosting'].feature_importances_)
            }
        
        return importance_dict
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get performance metrics for each model"""
        
        return {
            'model_weights': self.weights,
            'models_trained': list(self.models.keys()),
            'ensemble_strategy': 'weighted_average_by_inverse_mae',
            'is_trained': self.is_trained
        }
    
    def save_ensemble(self, base_path: str):
        """Save the entire ensemble model"""
        
        if not self.is_trained:
            raise ValueError("Cannot save untrained ensemble")
        
        # Save individual models
        joblib.dump(self.rf_model, f"{base_path}_random_forest.pkl")
        joblib.dump(self.gb_model, f"{base_path}_gradient_boosting.pkl")
        joblib.dump(self.lr_model, f"{base_path}_linear_regression.pkl")
        joblib.dump(self.scaler, f"{base_path}_scaler.pkl")
        
        # Save LSTM separately
        if self.lstm_model.is_trained:
            self.lstm_model.save_model(f"{base_path}_lstm.pth")
        
        # Save ensemble metadata
        ensemble_metadata = {
            'weights': self.weights,
            'models': list(self.models.keys()),
            'is_trained': self.is_trained,
            'save_timestamp': datetime.now().isoformat()
        }
        
        import json
        with open(f"{base_path}_ensemble_metadata.json", 'w') as f:
            json.dump(ensemble_metadata, f, indent=2)
    
    def load_ensemble(self, base_path: str):
        """Load the entire ensemble model"""
        
        # Load individual models
        self.rf_model = joblib.load(f"{base_path}_random_forest.pkl")
        self.gb_model = joblib.load(f"{base_path}_gradient_boosting.pkl")
        self.lr_model = joblib.load(f"{base_path}_linear_regression.pkl")
        self.scaler = joblib.load(f"{base_path}_scaler.pkl")
        
        # Load ensemble metadata
        import json
        with open(f"{base_path}_ensemble_metadata.json", 'r') as f:
            metadata = json.load(f)
        
        self.weights = metadata['weights']
        self.is_trained = metadata['is_trained']
        
        # Reconstruct models dict
        self.models = {
            'random_forest': self.rf_model,
            'gradient_boosting': self.gb_model,
            'linear_regression': self.lr_model
        }
        
        # Load LSTM if exists
        try:
            # You'd need to know the input size to load LSTM
            # This is simplified - in practice, save/load input size in metadata
            input_size = 20  # Default, should be saved in metadata
            self.lstm_model.load_model(f"{base_path}_lstm.pth", input_size)
            self.models['lstm'] = self.lstm_model
        except FileNotFoundError:
            print("LSTM model not found, continuing without it")

class WeightedEnsemble:
    """
    Simple weighted ensemble for combining multiple model predictions
    """
    
    def __init__(self, models: Dict[str, Any], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {name: w/total_weight for name, w in self.weights.items()}
    
    async def predict(self, features: np.ndarray) -> Dict[str, Any]:
        """Make ensemble prediction"""
        
        predictions = {}
        
        # Get predictions from each model
        for name, model in self.models.items():
            if hasattr(model, 'predict'):
                if asyncio.iscoroutinefunction(model.predict):
                    pred = await model.predict(features)
                else:
                    pred = model.predict(features)
                predictions[name] = pred
        
        # Weighted average
        if predictions:
            ensemble_pred = np.zeros_like(list(predictions.values())[0])
            
            for name, pred in predictions.items():
                weight = self.weights.get(name, 0)
                if isinstance(pred, dict) and 'predictions' in pred:
                    pred = pred['predictions']
                ensemble_pred += weight * np.array(pred)
            
            # Calculate uncertainty based on prediction variance
            pred_values = [np.array(pred['predictions']) if isinstance(pred, dict) else np.array(pred) 
                          for pred in predictions.values()]
            
            if len(pred_values) > 1:
                pred_std = np.std(pred_values, axis=0)
                confidence_intervals = [
                    [pred - 1.96*std, pred + 1.96*std] 
                    for pred, std in zip(ensemble_pred, pred_std)
                ]
            else:
                confidence_intervals = [
                    [pred*0.95, pred*1.05] for pred in ensemble_pred
                ]
            
            return {
                'predictions': ensemble_pred.tolist(),
                'confidence_intervals': confidence_intervals,
                'individual_predictions': predictions,
                'weights_used': self.weights
            }
        
        return {'predictions': [], 'error': 'No valid predictions from models'}