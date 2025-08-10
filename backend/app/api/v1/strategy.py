# backend/app/api/v1/strategy.py
"""
Stage 3: Investment Strategy API endpoints
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict, Any
from app.core.stage3_strategy.decision_engine import DecisionEngine
from app.core.stage3_strategy.portfolio_optimizer import PortfolioOptimizer
from app.core.stage3_strategy.risk_manager import RiskManager
from app.core.stage3_strategy.backtester import Backtester

router = APIRouter()

# Initialize components
decision_engine = DecisionEngine()
portfolio_optimizer = PortfolioOptimizer()
risk_manager = RiskManager()
backtester = Backtester()

@router.get("/recommendation/{symbol}")
async def get_investment_recommendation(symbol: str):
    """Get buy/sell/hold recommendation"""
    try:
        # Get comprehensive recommendation
        recommendation = await decision_engine.get_recommendation(symbol)
        
        return {
            "symbol": symbol,
            "recommendation": recommendation["action"],
            "confidence": recommendation["confidence"],
            "target_price": recommendation["target_price"],
            "stop_loss": recommendation["stop_loss"],
            "reasoning": recommendation["reasoning"],
            "risk_factors": recommendation["risks"],
            "time_horizon": recommendation["time_horizon"],
            "last_updated": recommendation["timestamp"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/portfolio/optimize")
async def optimize_portfolio(symbols: List[str], investment_amount: float, risk_tolerance: str = "moderate"):
    """Optimize portfolio allocation"""
    try:
        # Optimize portfolio
        optimization = await portfolio_optimizer.optimize(symbols, investment_amount, risk_tolerance)
        
        return {
            "total_investment": investment_amount,
            "risk_tolerance": risk_tolerance,
            "allocations": optimization["allocations"],
            "expected_return": optimization["expected_return"],
            "expected_risk": optimization["expected_risk"],
            "sharpe_ratio": optimization["sharpe_ratio"],
            "diversification_score": optimization["diversification"],
            "rebalancing_frequency": optimization["rebalancing"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/backtest/{symbol}")
async def run_backtest(symbol: str, start_date: str, end_date: str, strategy: str = "momentum"):
    """Run strategy backtest"""
    try:
        # Run backtesting
        results = await backtester.run_backtest(symbol, start_date, end_date, strategy)
        
        return {
            "symbol": symbol,
            "strategy": strategy,
            "period": f"{start_date} to {end_date}",
            "performance": {
                "total_return": results["total_return"],
                "annual_return": results["annual_return"],
                "max_drawdown": results["max_drawdown"],
                "sharpe_ratio": results["sharpe_ratio"],
                "win_rate": results["win_rate"]
            },
            "trades": results["trades"],
            "risk_metrics": results["risk_metrics"],
            "benchmark_comparison": results["benchmark"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
