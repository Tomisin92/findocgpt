# backend/app/core/stage1_document_ai/anomaly_detector.py
"""
Financial anomaly detection
"""

import asyncio
from typing import Dict, List, Any
import numpy as np

class AnomalyDetector:
    def __init__(self):
        self.anomaly_threshold = 2.0  # Standard deviations
        self.metrics_to_monitor = [
            "revenue_growth", "profit_margin", "debt_ratio", 
            "current_ratio", "roe", "roa"
        ]
    
    async def detect_anomalies(self, symbol: str) -> List[Dict[str, Any]]:
        """Detect anomalies in financial metrics"""
        
        # Get financial data
        financial_data = await self._get_financial_data(symbol)
        
        anomalies = []
        
        for metric in self.metrics_to_monitor:
            if metric in financial_data:
                anomaly = await self._detect_metric_anomaly(metric, financial_data[metric])
                if anomaly:
                    anomalies.append(anomaly)
        
        return anomalies
    
    async def _get_financial_data(self, symbol: str) -> Dict[str, List[float]]:
        """Get historical financial data for analysis"""
        
        # Simplified financial data
        return {
            "revenue_growth": [5.2, 7.1, 8.3, -2.1, 6.8],  # Anomaly at index 3
            "profit_margin": [12.5, 13.1, 12.8, 11.9, 12.3],
            "debt_ratio": [0.45, 0.42, 0.48, 0.51, 0.47],
            "current_ratio": [1.8, 1.9, 1.7, 1.6, 1.8],
            "roe": [15.2, 16.1, 14.8, 13.9, 15.5],
            "roa": [8.1, 8.5, 7.9, 7.2, 8.3]
        }
    
    async def _detect_metric_anomaly(self, metric: str, values: List[float]) -> Dict[str, Any]:
        """Detect anomalies in a specific metric"""
        
        if len(values) < 3:
            return None
        
        # Calculate statistics
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        # Find anomalies
        anomalies = []
        for i, value in enumerate(values):
            z_score = abs(value - mean_val) / std_val if std_val > 0 else 0
            
            if z_score > self.anomaly_threshold:
                anomalies.append({
                    "period": i,
                    "value": value,
                    "expected_range": [mean_val - std_val, mean_val + std_val],
                    "z_score": z_score,
                    "severity": "high" if z_score > 3 else "medium"
                })
        
        if anomalies:
            return {
                "metric": metric,
                "anomalies": anomalies,
                "description": await self._get_metric_description(metric),
                "impact": await self._assess_impact(metric, anomalies)
            }
        
        return None
    
    async def _get_metric_description(self, metric: str) -> str:
        """Get description of what the metric measures"""
        descriptions = {
            "revenue_growth": "Year-over-year revenue growth percentage",
            "profit_margin": "Net profit margin percentage",
            "debt_ratio": "Total debt to total assets ratio",
            "current_ratio": "Current assets to current liabilities ratio",
            "roe": "Return on equity percentage",
            "roa": "Return on assets percentage"
        }
        return descriptions.get(metric, f"Financial metric: {metric}")
    
    async def _assess_impact(self, metric: str, anomalies: List[Dict[str, Any]]) -> str:
        """Assess the impact of detected anomalies"""
        if not anomalies:
            return "No impact"
        
        severity_levels = [a["severity"] for a in anomalies]
        
        if "high" in severity_levels:
            return "High impact - requires immediate attention"
        elif "medium" in severity_levels:
            return "Medium impact - monitor closely"
        else:
            return "Low impact - normal variation"
    
    def calculate_risk_level(self, anomalies: List[Dict[str, Any]]) -> str:
        """Calculate overall risk level based on anomalies"""
        if not anomalies:
            return "low"
        
        high_severity_count = sum(1 for a in anomalies 
                                 if any(sub_a["severity"] == "high" 
                                       for sub_a in a.get("anomalies", [])))
        
        if high_severity_count >= 2:
            return "high"
        elif high_severity_count == 1 or len(anomalies) > 2:
            return "medium"
        else:
            return "low"
    
    def get_recommendations(self, anomalies: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations based on detected anomalies"""
        if not anomalies:
            return ["Continue monitoring financial metrics"]
        
        recommendations = []
        
        for anomaly in anomalies:
            metric = anomaly["metric"]
            if metric == "revenue_growth":
                recommendations.append("Investigate revenue decline drivers")
            elif metric == "profit_margin":
                recommendations.append("Review cost structure and operational efficiency")
            elif metric == "debt_ratio":
                recommendations.append("Monitor debt levels and repayment capacity")
        
        return recommendations or ["Conduct detailed financial analysis"]