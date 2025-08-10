# # backend/app/core/stage3_strategy/decision_engine.py
# """
# Investment decision making engine
# """

# import asyncio
# from typing import Dict, List, Any
# from datetime import datetime
# import numpy as np

# class DecisionEngine:
#     def __init__(self):
#         self.decision_weights = {
#             "technical": 0.3,
#             "fundamental": 0.4,
#             "sentiment": 0.2,
#             "risk": 0.1
#         }
        
#     async def get_recommendation(self, symbol: str) -> Dict[str, Any]:
#         """Get comprehensive investment recommendation"""
        
#         # Gather all analysis components
#         technical_score = await self._get_technical_score(symbol)
#         fundamental_score = await self._get_fundamental_score(symbol)
#         sentiment_score = await self._get_sentiment_score(symbol)
#         risk_score = await self._get_risk_score(symbol)
        
#         # Calculate weighted score
#         overall_score = (
#             technical_score * self.decision_weights["technical"] +
#             fundamental_score * self.decision_weights["fundamental"] +
#             sentiment_score * self.decision_weights["sentiment"] +
#             risk_score * self.decision_weights["risk"]
#         )
        
#         # Make decision
#         recommendation = await self._make_decision(overall_score, symbol)
        
#         return {
#             "action": recommendation["action"],
#             "confidence": recommendation["confidence"],
#             "target_price": recommendation["target_price"],
#             "stop_loss": recommendation["stop_loss"],
#             "reasoning": recommendation["reasoning"],
#             "risks": recommendation["risks"],
#             "time_horizon": recommendation["time_horizon"],
#             "scores": {
#                 "technical": technical_score,
#                 "fundamental": fundamental_score,
#                 "sentiment": sentiment_score,
#                 "risk": risk_score,
#                 "overall": overall_score
#             },
#             "timestamp": datetime.now().isoformat()
#         }
    
#     async def _get_technical_score(self, symbol: str) -> float:
#         """Calculate technical analysis score"""
        
#         # Simplified technical scoring
#         indicators = {
#             "rsi": 65,  # Slightly overbought
#             "macd": 1.2,  # Positive momentum
#             "sma_trend": "upward",  # Price above moving averages
#             "volume": "increasing"  # Good volume confirmation
#         }
        
#         score = 0.0
        
#         # RSI scoring
#         if 30 <= indicators["rsi"] <= 70:
#             score += 0.3
#         elif indicators["rsi"] < 30:
#             score += 0.4  # Oversold - potential buy
#         else:
#             score += 0.1  # Overbought
        
#         # MACD scoring
#         if indicators["macd"] > 0:
#             score += 0.3
        
#         # Trend scoring
#         if indicators["sma_trend"] == "upward":
#             score += 0.2
        
#         # Volume scoring
#         if indicators["volume"] == "increasing":
#             score += 0.2
        
#         return min(score, 1.0)
    
#     async def _get_fundamental_score(self, symbol: str) -> float:
#         """Calculate fundamental analysis score"""
        
#         # Simplified fundamental scoring
#         metrics = {
#             "pe_ratio": 18.5,
#             "peg_ratio": 1.2,
#             "debt_to_equity": 0.3,
#             "roe": 0.15,
#             "revenue_growth": 0.08
#         }
        
#         score = 0.0
        
#         # P/E ratio scoring
#         if 10 <= metrics["pe_ratio"] <= 25:
#             score += 0.25
        
#         # PEG ratio scoring
#         if metrics["peg_ratio"] <= 1.5:
#             score += 0.25
        
#         # Debt-to-equity scoring
#         if metrics["debt_to_equity"] <= 0.5:
#             score += 0.2
        
#         # ROE scoring
#         if metrics["roe"] >= 0.12:
#             score += 0.15
        
#         # Revenue growth scoring
#         if metrics["revenue_growth"] >= 0.05:
#             score += 0.15
        
#         return min(score, 1.0)
    
#     async def _get_sentiment_score(self, symbol: str) -> float:
#         """Calculate sentiment analysis score"""
        
#         # Simplified sentiment scoring
#         sentiment_data = {
#             "news_sentiment": 0.7,  # Positive
#             "analyst_rating": 4.2,  # Buy/Strong Buy
#             "social_sentiment": 0.6,  # Slightly positive
#             "insider_trading": "neutral"
#         }
        
#         score = 0.0
        
#         # News sentiment
#         score += sentiment_data["news_sentiment"] * 0.4
        
#         # Analyst rating (scale 1-5, where 5 is strong buy)
#         score += (sentiment_data["analyst_rating"] / 5) * 0.3
        
#         # Social sentiment
#         score += sentiment_data["social_sentiment"] * 0.2
        
#         # Insider trading
#         if sentiment_data["insider_trading"] == "buying":
#             score += 0.1
        
#         return min(score, 1.0)
    
#     async def _get_risk_score(self, symbol: str) -> float:
#         """Calculate risk assessment score (higher is better/less risky)"""
        
#         risk_factors = {
#             "volatility": 0.25,  # 25% annualized volatility
#             "beta": 1.2,
#             "market_correlation": 0.7,
#             "liquidity": "high",
#             "sector_risk": "medium"
#         }
        
#         score = 1.0  # Start with max score, subtract for risks
        
#         # Volatility penalty
#         if risk_factors["volatility"] > 0.3:
#             score -= 0.2
#         elif risk_factors["volatility"] > 0.2:
#             score -= 0.1
        
#         # Beta penalty
#         if risk_factors["beta"] > 1.5:
#             score -= 0.2
#         elif risk_factors["beta"] > 1.2:
#             score -= 0.1
        
#         # Liquidity bonus
#         if risk_factors["liquidity"] == "high":
#             score += 0.1
        
#         return max(score, 0.0)
    
#     async def _make_decision(self, overall_score: float, symbol: str) -> Dict[str, Any]:
#         """Make final investment decision based on overall score"""
        
#         if overall_score >= 0.7:
#             action = "BUY"
#             confidence = 0.85
#             target_multiplier = 1.15
#             stop_multiplier = 0.92
#             time_horizon = "3-6 months"
#             reasoning = [
#                 "Strong technical indicators",
#                 "Solid fundamental metrics",
#                 "Positive market sentiment"
#             ]
#         elif overall_score >= 0.5:
#             action = "HOLD"
#             confidence = 0.65
#             target_multiplier = 1.08
#             stop_multiplier = 0.95
#             time_horizon = "1-3 months"
#             reasoning = [
#                 "Mixed signals across analysis dimensions",
#                 "Wait for clearer trend confirmation"
#             ]
#         else:
#             action = "SELL"
#             confidence = 0.75
#             target_multiplier = 0.95
#             stop_multiplier = 1.05
#             time_horizon = "1-2 months"
#             reasoning = [
#                 "Weak technical outlook",
#                 "Concerning fundamental trends",
#                 "Negative sentiment indicators"
#             ]
        
#         # Calculate target prices (simplified)
#         current_price = 150.0  # Would get from real data
        
#         return {
#             "action": action,
#             "confidence": confidence,
#             "target_price": current_price * target_multiplier,
#             "stop_loss": current_price * stop_multiplier,
#             "reasoning": reasoning,
#             "risks": await self._identify_risks(symbol),
#             "time_horizon": time_horizon
#         }
    
#     async def _identify_risks(self, symbol: str) -> List[str]:
#         """Identify key risks for the investment"""
        
#         return [
#             "Market volatility risk",
#             "Sector-specific headwinds",
#             "Regulatory changes",
#             "Economic downturn impact"
#         ]


"""Investment Decision Engine"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import yfinance as yf

class DecisionEngine:
    """Investment decision making engine"""
    
    def __init__(self):
        self.risk_free_rate = 0.02
    
    async def get_recommendation(self, symbol: str, analysis_data: Optional[Dict] = None) -> Dict:
        """Get investment recommendation for a symbol"""
        try:
            if analysis_data is None:
                analysis_data = await self._gather_basic_data(symbol)
            
            # Simple scoring system
            fundamental_score = self._score_fundamentals(analysis_data)
            technical_score = self._score_technical(analysis_data)
            
            overall_score = (fundamental_score + technical_score) / 2
            
            # Generate recommendation
            if overall_score >= 7:
                recommendation = "BUY"
                confidence = 0.8
            elif overall_score >= 5:
                recommendation = "HOLD"
                confidence = 0.6
            else:
                recommendation = "SELL"
                confidence = 0.7
            
            return {
                "symbol": symbol,
                "recommendation": recommendation,
                "confidence": confidence,
                "overall_score": round(overall_score, 1),
                "fundamental_score": round(fundamental_score, 1),
                "technical_score": round(technical_score, 1)
            }
            
        except Exception as e:
            return {"error": f"Decision engine failed: {str(e)}"}
    
    async def _gather_basic_data(self, symbol: str) -> Dict:
        """Gather basic data for analysis"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                "symbol": symbol,
                "current_price": info.get("currentPrice", 100),
                "pe_ratio": info.get("trailingPE", 20),
                "market_cap": info.get("marketCap", 1000000000)
            }
        except:
            return {"symbol": symbol, "current_price": 100}
    
    def _score_fundamentals(self, data: Dict) -> float:
        """Score fundamental metrics"""
        score = 5.0  # Base score
        
        pe_ratio = data.get("pe_ratio", 20)
        if pe_ratio < 15:
            score += 2
        elif pe_ratio > 30:
            score -= 2
        
        return max(0, min(10, score))
    
    def _score_technical(self, data: Dict) -> float:
        """Score technical metrics"""
        # Simple technical scoring
        return 6.0  # Placeholder