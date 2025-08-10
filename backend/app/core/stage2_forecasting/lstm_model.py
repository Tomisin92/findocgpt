# backend/app/core/stage2_forecasting/lstm_model.py
"""
LSTM model for stock price prediction
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from typing import Dict, List, Any, Tuple
import joblib

class LSTMModel(nn.Module):
    def __init__(self, input_size: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        # Initialize hidden state
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # LSTM forward pass
        out, _ = self.lstm(x, (h0, c0))
        out = self.dropout(out[:, -1, :])  # Take last time step
        out = self.linear(out)
        
        return out

class LSTMPredictor:
    def __init__(self):
        self.model = None
        self.scaler = MinMaxScaler()
        self.sequence_length = 60
        self.is_trained = False
        
    async def train(self, data: pd.DataFrame, target_column: str = "Close") -> Dict[str, Any]:
        """Train LSTM model on historical data"""
        
        # Prepare data
        X, y = self._prepare_sequences(data, target_column)
        
        # Split data
        train_size = int(0.8 * len(X))
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to tensors
        X_train = torch.FloatTensor(X_train)
        y_train = torch.FloatTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.FloatTensor(y_test)
        
        # Initialize model
        input_size = X_train.shape[2]
        self.model = LSTMModel(input_size)
        
        # Training parameters
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Training loop
        epochs = 100
        train_losses = []
        
        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()
            
            outputs = self.model(X_train)
            loss = criterion(outputs, y_train.unsqueeze(1))
            
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            
            if epoch % 20 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Evaluate on test set
        self.model.eval()
        with torch.no_grad():
            test_outputs = self.model(X_test)
            test_loss = criterion(test_outputs, y_test.unsqueeze(1))
        
        self.is_trained = True
        
        return {
            "train_loss": train_losses[-1],
            "test_loss": test_loss.item(),
            "epochs": epochs,
            "model_parameters": sum(p.numel() for p in self.model.parameters())
        }
    
    def _prepare_sequences(self, data: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare sequences for LSTM training"""
        
        # Select features
        feature_columns = [col for col in data.columns if col not in ['Date']]
        features = data[feature_columns].values
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Create sequences
        X, y = [], []
        
        for i in range(self.sequence_length, len(scaled_features)):
            X.append(scaled_features[i-self.sequence_length:i])
            y.append(scaled_features[i, data.columns.get_loc(target_column)])
        
        return np.array(X), np.array(y)
    
    async def predict(self, data: pd.DataFrame, days: int = 30) -> Dict[str, Any]:
        """Make price predictions"""
        
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")
        
        # Prepare last sequence
        feature_columns = [col for col in data.columns if col not in ['Date']]
        features = data[feature_columns].values
        scaled_features = self.scaler.transform(features)
        
        # Get last sequence
        last_sequence = scaled_features[-self.sequence_length:]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        self.model.eval()
        
        for _ in range(days):
            # Reshape for model input
            input_tensor = torch.FloatTensor(current_sequence).unsqueeze(0)
            
            # Make prediction
            with torch.no_grad():
                pred = self.model(input_tensor)
                predictions.append(pred.item())
            
            # Update sequence (simplified - use prediction for next input)
            new_row = current_sequence[-1].copy()
            new_row[0] = pred.item()  # Assuming Close price is first column
            
            current_sequence = np.vstack([current_sequence[1:], new_row])
        
        # Inverse transform predictions
        # Create dummy array for inverse transform
        dummy_features = np.zeros((len(predictions), scaled_features.shape[1]))
        dummy_features[:, 0] = predictions  # Put predictions in first column
        
        inverse_transformed = self.scaler.inverse_transform(dummy_features)
        final_predictions = inverse_transformed[:, 0]
        
        return {
            "predictions": final_predictions.tolist(),
            "confidence_intervals": self._calculate_confidence_intervals(final_predictions),
            "model_type": "LSTM",
            "prediction_days": days
        }
    
    def _calculate_confidence_intervals(self, predictions: np.ndarray) -> List[List[float]]:
        """Calculate confidence intervals for predictions"""
        
        # Simplified confidence interval calculation
        confidence_intervals = []
        
        for i, pred in enumerate(predictions):
            # Increasing uncertainty over time
            uncertainty = 0.02 + (i * 0.001)  # 2% base + 0.1% per day
            
            lower = pred * (1 - uncertainty)
            upper = pred * (1 + uncertainty)
            
            confidence_intervals.append([lower, upper])
        
        return confidence_intervals
    
    def save_model(self, path: str):
        """Save trained model and scaler"""
        if self.model and self.is_trained:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'scaler': self.scaler,
                'sequence_length': self.sequence_length
            }, path)
    
    def load_model(self, path: str, input_size: int):
        """Load trained model and scaler"""
        checkpoint = torch.load(path)
        
        self.model = LSTMModel(input_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.sequence_length = checkpoint['sequence_length']
        self.is_trained = True