# backend/app/core/stage3_strategy/risk_manager.py
"""
Risk Management Module for Investment Strategy
Calculates various risk metrics and provides risk assessment
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskManager:
    """Comprehensive risk management for investment strategies"""
    
    def __init__(self):
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        self.market_symbol = "SPY"  # S&P 500 as market benchmark
        
    async def calculate_portfolio_risk(self, 
                                     symbols: List[str], 
                                     weights: Optional[List[float]] = None,
                                     period: str = "1y") -> Dict:
        """Calculate comprehensive portfolio risk metrics"""
        try:
            if weights is None:
                weights = [1.0 / len(symbols)] * len(symbols)  # Equal weights
            
            # Get historical data for all symbols
            portfolio_data = await self._get_portfolio_data(symbols, period)
            
            if portfolio_data.empty:
                return {"error": "No data available for portfolio analysis"}
            
            # Calculate portfolio returns
            portfolio_returns = self._calculate_portfolio_returns(portfolio_data, weights)
            
            # Calculate individual stock metrics
            individual_metrics = await self._calculate_individual_metrics(symbols, period)
            
            # Calculate portfolio-level metrics
            portfolio_metrics = self._calculate_portfolio_metrics(portfolio_returns)
            
            # Calculate risk-adjusted metrics
            risk_adjusted_metrics = self._calculate_risk_adjusted_metrics(portfolio_returns)
            
            # Calculate correlation matrix
            correlation_matrix = self._calculate_correlation_matrix(portfolio_data)
            
            # Risk assessment
            risk_assessment = self._assess_overall_risk(portfolio_metrics, individual_metrics)
            
            return {
                "portfolio_metrics": portfolio_metrics,
                "individual_metrics": individual_metrics,
                "risk_adjusted_metrics": risk_adjusted_metrics,
                "correlation_matrix": correlation_matrix,
                "risk_assessment": risk_assessment,
                "symbols": symbols,
                "weights": weights,
                "analysis_date": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Portfolio risk calculation failed: {str(e)}"}
    
    async def calculate_var(self, 
                          symbol: str, 
                          confidence_level: float = 0.95,
                          time_horizon: int = 1,
                          portfolio_value: float = 10000) -> Dict:
        """Calculate Value at Risk (VaR) for a single asset"""
        try:
            # Get historical data
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y")
            
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calculate daily returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate VaR using historical method
            var_historical = np.percentile(returns, (1 - confidence_level) * 100)
            
            # Calculate VaR using parametric method
            mean_return = returns.mean()
            std_return = returns.std()
            var_parametric = mean_return - std_return * np.sqrt(time_horizon) * \
                           self._inverse_normal(confidence_level)
            
            # Calculate Expected Shortfall (Conditional VaR)
            expected_shortfall = returns[returns <= var_historical].mean()
            
            # Convert to dollar amounts
            var_historical_dollar = abs(var_historical * portfolio_value)
            var_parametric_dollar = abs(var_parametric * portfolio_value)
            expected_shortfall_dollar = abs(expected_shortfall * portfolio_value)
            
            return {
                "symbol": symbol,
                "confidence_level": confidence_level,
                "time_horizon_days": time_horizon,
                "portfolio_value": portfolio_value,
                "var_historical_percent": round(var_historical * 100, 2),
                "var_parametric_percent": round(var_parametric * 100, 2),
                "var_historical_dollar": round(var_historical_dollar, 2),
                "var_parametric_dollar": round(var_parametric_dollar, 2),
                "expected_shortfall_percent": round(expected_shortfall * 100, 2),
                "expected_shortfall_dollar": round(expected_shortfall_dollar, 2),
                "interpretation": self._interpret_var(var_historical, confidence_level)
            }
            
        except Exception as e:
            return {"error": f"VaR calculation failed: {str(e)}"}
    
    async def calculate_beta(self, symbol: str, market_symbol: str = None) -> Dict:
        """Calculate beta coefficient relative to market"""
        try:
            if market_symbol is None:
                market_symbol = self.market_symbol
            
            # Get data for both asset and market
            asset_ticker = yf.Ticker(symbol)
            market_ticker = yf.Ticker(market_symbol)
            
            asset_data = asset_ticker.history(period="2y")
            market_data = market_ticker.history(period="2y")
            
            if asset_data.empty or market_data.empty:
                return {"error": "Insufficient data for beta calculation"}
            
            # Calculate returns
            asset_returns = asset_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = asset_returns.index.intersection(market_returns.index)
            asset_returns = asset_returns.loc[common_dates]
            market_returns = market_returns.loc[common_dates]
            
            # Calculate beta
            covariance = np.cov(asset_returns, market_returns)[0, 1]
            market_variance = np.var(market_returns)
            beta = covariance / market_variance if market_variance != 0 else 1.0
            
            # Calculate correlation
            correlation = np.corrcoef(asset_returns, market_returns)[0, 1]
            
            # Calculate R-squared
            r_squared = correlation ** 2
            
            return {
                "symbol": symbol,
                "market_symbol": market_symbol,
                "beta": round(beta, 3),
                "correlation": round(correlation, 3),
                "r_squared": round(r_squared, 3),
                "interpretation": self._interpret_beta(beta),
                "data_points": len(asset_returns)
            }
            
        except Exception as e:
            return {"error": f"Beta calculation failed: {str(e)}"}
    
    async def calculate_sharpe_ratio(self, symbol: str, period: str = "1y") -> Dict:
        """Calculate Sharpe ratio for risk-adjusted returns"""
        try:
            ticker = yf.Ticker(symbol)
            data = ticker.history(period="2y")
            
            if data.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calculate returns
            returns = data['Close'].pct_change().dropna()
            
            # Calculate annualized metrics
            annual_return = returns.mean() * 252  # 252 trading days
            annual_volatility = returns.std() * np.sqrt(252)
            
            # Calculate Sharpe ratio
            excess_return = annual_return - self.risk_free_rate
            sharpe_ratio = excess_return / annual_volatility if annual_volatility != 0 else 0
            
            # Calculate other risk metrics
            max_drawdown = self._calculate_max_drawdown(data['Close'])
            sortino_ratio = self._calculate_sortino_ratio(returns)
            
            return {
                "symbol": symbol,
                "annual_return": round(annual_return * 100, 2),
                "annual_volatility": round(annual_volatility * 100, 2),
                "sharpe_ratio": round(sharpe_ratio, 3),
                "sortino_ratio": round(sortino_ratio, 3),
                "max_drawdown": round(max_drawdown * 100, 2),
                "risk_free_rate": round(self.risk_free_rate * 100, 2),
                "interpretation": self._interpret_sharpe_ratio(sharpe_ratio)
            }
            
        except Exception as e:
            return {"error": f"Sharpe ratio calculation failed: {str(e)}"}
    
    def assess_position_size(self, 
                           symbol: str,
                           portfolio_value: float,
                           risk_tolerance: str = "moderate",
                           max_position_size: float = 0.1) -> Dict:
        """Recommend position size based on risk management rules"""
        try:
            # Risk tolerance mapping
            risk_multipliers = {
                "conservative": 0.5,
                "moderate": 1.0,
                "aggressive": 1.5
            }
            
            multiplier = risk_multipliers.get(risk_tolerance, 1.0)
            
            # Base position size (percentage of portfolio)
            base_position_size = min(max_position_size, 0.05 * multiplier)
            
            # Calculate dollar amount
            position_value = portfolio_value * base_position_size
            
            # Risk management rules
            rules = [
                f"Maximum position size: {max_position_size:.1%} of portfolio",
                f"Risk tolerance: {risk_tolerance}",
                "Diversify across at least 10-15 positions",
                "Consider correlation with existing holdings"
            ]
            
            return {
                "symbol": symbol,
                "recommended_position_size_percent": round(base_position_size * 100, 2),
                "recommended_position_value": round(position_value, 2),
                "portfolio_value": portfolio_value,
                "risk_tolerance": risk_tolerance,
                "max_position_size_percent": round(max_position_size * 100, 2),
                "risk_management_rules": rules
            }
            
        except Exception as e:
            return {"error": f"Position sizing calculation failed: {str(e)}"}
    
    # Helper methods
    async def _get_portfolio_data(self, symbols: List[str], period: str) -> pd.DataFrame:
        """Get historical data for portfolio symbols"""
        try:
            data_frames = []
            
            for symbol in symbols:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                if not data.empty:
                    data_frames.append(data['Close'].rename(symbol))
            
            if data_frames:
                return pd.concat(data_frames, axis=1).dropna()
            else:
                return pd.DataFrame()
                
        except Exception as e:
            print(f"Error getting portfolio data: {e}")
            return pd.DataFrame()
    
    def _calculate_portfolio_returns(self, data: pd.DataFrame, weights: List[float]) -> pd.Series:
        """Calculate weighted portfolio returns"""
        returns = data.pct_change().dropna()
        portfolio_returns = (returns * weights).sum(axis=1)
        return portfolio_returns
    
    async def _calculate_individual_metrics(self, symbols: List[str], period: str) -> Dict:
        """Calculate risk metrics for individual stocks"""
        individual_metrics = {}
        
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                data = ticker.history(period=period)
                
                if not data.empty:
                    returns = data['Close'].pct_change().dropna()
                    
                    metrics = {
                        "volatility": round(returns.std() * np.sqrt(252) * 100, 2),
                        "max_drawdown": round(self._calculate_max_drawdown(data['Close']) * 100, 2),
                        "sharpe_ratio": round(self._calculate_simple_sharpe(returns), 3)
                    }
                    
                    individual_metrics[symbol] = metrics
                    
            except Exception as e:
                individual_metrics[symbol] = {"error": str(e)}
        
        return individual_metrics
    
    def _calculate_portfolio_metrics(self, returns: pd.Series) -> Dict:
        """Calculate portfolio-level risk metrics"""
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        return {
            "annual_return_percent": round(annual_return * 100, 2),
            "annual_volatility_percent": round(annual_volatility * 100, 2),
            "total_return_percent": round((1 + returns).cumprod().iloc[-1] * 100 - 100, 2),
            "max_drawdown_percent": round(self._calculate_max_drawdown_from_returns(returns) * 100, 2)
        }
    
    def _calculate_risk_adjusted_metrics(self, returns: pd.Series) -> Dict:
        """Calculate risk-adjusted performance metrics"""
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        # Sharpe ratio
        sharpe_ratio = (annual_return - self.risk_free_rate) / annual_volatility if annual_volatility != 0 else 0
        
        # Sortino ratio
        sortino_ratio = self._calculate_sortino_ratio(returns)
        
        # Calmar ratio
        max_dd = self._calculate_max_drawdown_from_returns(returns)
        calmar_ratio = annual_return / abs(max_dd) if max_dd != 0 else 0
        
        return {
            "sharpe_ratio": round(sharpe_ratio, 3),
            "sortino_ratio": round(sortino_ratio, 3),
            "calmar_ratio": round(calmar_ratio, 3)
        }
    
    def _calculate_correlation_matrix(self, data: pd.DataFrame) -> Dict:
        """Calculate correlation matrix for portfolio assets"""
        if data.empty:
            return {}
        
        returns = data.pct_change().dropna()
        correlation_matrix = returns.corr()
        
        # Convert to dictionary format
        corr_dict = {}
        for i, symbol1 in enumerate(correlation_matrix.columns):
            corr_dict[symbol1] = {}
            for j, symbol2 in enumerate(correlation_matrix.columns):
                corr_dict[symbol1][symbol2] = round(correlation_matrix.iloc[i, j], 3)
        
        return corr_dict
    
    def _assess_overall_risk(self, portfolio_metrics: Dict, individual_metrics: Dict) -> Dict:
        """Provide overall risk assessment"""
        portfolio_vol = portfolio_metrics.get("annual_volatility_percent", 0)
        max_dd = portfolio_metrics.get("max_drawdown_percent", 0)
        
        # Risk level assessment
        if portfolio_vol < 15 and abs(max_dd) < 10:
            risk_level = "Low"
        elif portfolio_vol < 25 and abs(max_dd) < 20:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        # Risk factors
        risk_factors = []
        if portfolio_vol > 30:
            risk_factors.append("High volatility")
        if abs(max_dd) > 25:
            risk_factors.append("Large maximum drawdown")
        if len(individual_metrics) < 5:
            risk_factors.append("Insufficient diversification")
        
        return {
            "overall_risk_level": risk_level,
            "risk_factors": risk_factors,
            "recommendation": self._get_risk_recommendation(risk_level, risk_factors)
        }
    
    def _calculate_max_drawdown(self, prices: pd.Series) -> float:
        """Calculate maximum drawdown from price series"""
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        return drawdown.min()
    
    def _calculate_max_drawdown_from_returns(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown from return series"""
        cumulative = (1 + returns).cumprod()
        peak = cumulative.expanding().max()
        drawdown = (cumulative - peak) / peak
        return drawdown.min()
    
    def _calculate_sortino_ratio(self, returns: pd.Series) -> float:
        """Calculate Sortino ratio (downside deviation only)"""
        annual_return = returns.mean() * 252
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252)
        
        if downside_deviation != 0:
            return (annual_return - self.risk_free_rate) / downside_deviation
        else:
            return 0
    
    def _calculate_simple_sharpe(self, returns: pd.Series) -> float:
        """Simple Sharpe ratio calculation"""
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        
        if annual_volatility != 0:
            return (annual_return - self.risk_free_rate) / annual_volatility
        else:
            return 0
    
    def _inverse_normal(self, confidence_level: float) -> float:
        """Approximate inverse normal distribution for VaR calculation"""
        # Simple approximation for common confidence levels
        if confidence_level >= 0.99:
            return 2.33
        elif confidence_level >= 0.95:
            return 1.645
        elif confidence_level >= 0.90:
            return 1.28
        else:
            return 1.0
    
    def _interpret_var(self, var_value: float, confidence_level: float) -> str:
        """Interpret VaR results"""
        confidence_pct = int(confidence_level * 100)
        var_pct = abs(var_value * 100)
        
        if var_pct < 2:
            risk_level = "low"
        elif var_pct < 5:
            risk_level = "moderate"
        else:
            risk_level = "high"
        
        return f"With {confidence_pct}% confidence, daily losses should not exceed {var_pct:.1f}%. Risk level: {risk_level}."
    
    def _interpret_beta(self, beta: float) -> str:
        """Interpret beta coefficient"""
        if beta < 0.8:
            return "Low sensitivity to market movements (defensive)"
        elif beta < 1.2:
            return "Similar sensitivity to market movements"
        else:
            return "High sensitivity to market movements (aggressive)"
    
    def _interpret_sharpe_ratio(self, sharpe: float) -> str:
        """Interpret Sharpe ratio"""
        if sharpe < 0:
            return "Poor risk-adjusted returns"
        elif sharpe < 1:
            return "Below average risk-adjusted returns"
        elif sharpe < 2:
            return "Good risk-adjusted returns"
        else:
            return "Excellent risk-adjusted returns"
    
    def _get_risk_recommendation(self, risk_level: str, risk_factors: List[str]) -> str:
        """Get risk management recommendation"""
        if risk_level == "Low":
            return "Portfolio risk is well-managed. Consider gradual position increases if appropriate."
        elif risk_level == "Moderate":
            return "Portfolio risk is acceptable but monitor closely. Consider diversification improvements."
        else:
            return "Portfolio risk is high. Consider reducing position sizes and improving diversification."