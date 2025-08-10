# backend/app/utils/financial_metrics.py
"""
Financial metrics calculations and analysis utilities
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import warnings
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

class FinancialMetrics:
    """Comprehensive financial metrics calculation utilities"""
    
    @staticmethod
    def calculate_returns(prices: pd.Series, method: str = 'simple') -> pd.Series:
        """Calculate returns from price series"""
        if method == 'simple':
            return prices.pct_change()
        elif method == 'log':
            return np.log(prices / prices.shift(1))
        else:
            raise ValueError("Method must be 'simple' or 'log'")
    
    @staticmethod
    def calculate_log_returns(prices: pd.Series) -> pd.Series:
        """Calculate logarithmic returns"""
        return np.log(prices / prices.shift(1))
    
    @staticmethod
    def calculate_cumulative_returns(returns: pd.Series) -> pd.Series:
        """Calculate cumulative returns"""
        return (1 + returns).cumprod() - 1
    
    @staticmethod
    def calculate_volatility(returns: pd.Series, periods: int = 252) -> float:
        """Calculate annualized volatility"""
        if len(returns) < 2:
            return 0.0
        return returns.std() * np.sqrt(periods)
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """Calculate Sharpe ratio"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() * periods - risk_free_rate
        volatility = FinancialMetrics.calculate_volatility(returns, periods)
        
        return excess_returns / volatility if volatility > 0 else 0.0
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """Calculate Sortino ratio (downside deviation)"""
        if len(returns) < 2:
            return 0.0
        
        excess_returns = returns.mean() * periods - risk_free_rate
        downside_returns = returns[returns < 0]
        
        if len(downside_returns) == 0:
            return float('inf') if excess_returns > 0 else 0.0
        
        downside_deviation = downside_returns.std() * np.sqrt(periods)
        return excess_returns / downside_deviation if downside_deviation > 0 else 0.0
    
    @staticmethod
    def calculate_max_drawdown(prices: pd.Series) -> Dict[str, float]:
        """Calculate maximum drawdown and related metrics"""
        if len(prices) < 2:
            return {'max_drawdown': 0.0, 'max_drawdown_duration': 0, 'recovery_time': 0}
        
        # Calculate running maximum
        peak = prices.expanding().max()
        drawdown = (prices - peak) / peak
        
        max_dd = drawdown.min()
        
        # Find drawdown periods
        drawdown_start = None
        drawdown_end = None
        max_duration = 0
        current_duration = 0
        
        for i, dd in enumerate(drawdown):
            if dd < 0:
                if drawdown_start is None:
                    drawdown_start = i
                current_duration += 1
            else:
                if drawdown_start is not None:
                    drawdown_end = i
                    max_duration = max(max_duration, current_duration)
                    current_duration = 0
                    drawdown_start = None
        
        # Handle case where drawdown continues to end
        if drawdown_start is not None:
            max_duration = max(max_duration, current_duration)
        
        return {
            'max_drawdown': abs(max_dd),
            'max_drawdown_duration': max_duration,
            'drawdown_series': drawdown
        }
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, periods: int = 252) -> float:
        """Calculate Calmar ratio (annual return / max drawdown)"""
        if len(returns) < 2:
            return 0.0
        
        annual_return = returns.mean() * periods
        prices = (1 + returns).cumprod()
        max_dd = FinancialMetrics.calculate_max_drawdown(prices)['max_drawdown']
        
        return annual_return / max_dd if max_dd > 0 else 0.0
    
    @staticmethod
    def calculate_var(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Value at Risk"""
        if len(returns) < 2:
            return 0.0
        return np.percentile(returns.dropna(), confidence_level * 100)
    
    @staticmethod
    def calculate_cvar(returns: pd.Series, confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        if len(returns) < 2:
            return 0.0
        
        var = FinancialMetrics.calculate_var(returns, confidence_level)
        return returns[returns <= var].mean()
    
    @staticmethod
    def calculate_beta(asset_returns: pd.Series, market_returns: pd.Series) -> float:
        """Calculate beta coefficient"""
        # Align series and drop NaN values
        aligned_data = pd.concat([asset_returns, market_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 1.0
        
        asset_clean = aligned_data.iloc[:, 0]
        market_clean = aligned_data.iloc[:, 1]
        
        covariance = np.cov(asset_clean, market_clean)[0][1]
        market_variance = np.var(market_clean)
        
        return covariance / market_variance if market_variance > 0 else 1.0
    
    @staticmethod
    def calculate_alpha(asset_returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02) -> float:
        """Calculate Jensen's alpha"""
        if len(asset_returns) < 2 or len(market_returns) < 2:
            return 0.0
        
        beta = FinancialMetrics.calculate_beta(asset_returns, market_returns)
        asset_return = asset_returns.mean() * 252
        market_return = market_returns.mean() * 252
        
        expected_return = risk_free_rate + beta * (market_return - risk_free_rate)
        return asset_return - expected_return
    
    @staticmethod
    def calculate_information_ratio(portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        # Align series
        aligned_data = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return 0.0
        
        excess_returns = aligned_data.iloc[:, 0] - aligned_data.iloc[:, 1]
        tracking_error = excess_returns.std()
        
        return excess_returns.mean() / tracking_error if tracking_error > 0 else 0.0
    
    @staticmethod
    def calculate_treynor_ratio(returns: pd.Series, market_returns: pd.Series, risk_free_rate: float = 0.02, periods: int = 252) -> float:
        """Calculate Treynor ratio"""
        if len(returns) < 2:
            return 0.0
        
        beta = FinancialMetrics.calculate_beta(returns, market_returns)
        excess_return = returns.mean() * periods - risk_free_rate
        
        return excess_return / beta if beta != 0 else 0.0
    
    @staticmethod
    def calculate_omega_ratio(returns: pd.Series, threshold: float = 0.0) -> float:
        """Calculate Omega ratio"""
        if len(returns) < 2:
            return 1.0
        
        excess_returns = returns - threshold
        positive_returns = excess_returns[excess_returns > 0].sum()
        negative_returns = abs(excess_returns[excess_returns < 0].sum())
        
        return positive_returns / negative_returns if negative_returns > 0 else float('inf')
    
    @staticmethod
    def calculate_tail_ratio(returns: pd.Series, percentile: float = 0.05) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)"""
        if len(returns) < 2:
            return 1.0
        
        upper_tail = np.percentile(returns.dropna(), (1 - percentile) * 100)
        lower_tail = abs(np.percentile(returns.dropna(), percentile * 100))
        
        return upper_tail / lower_tail if lower_tail > 0 else float('inf')
    
    @staticmethod
    def calculate_skewness(returns: pd.Series) -> float:
        """Calculate skewness of returns"""
        if len(returns) < 3:
            return 0.0
        return stats.skew(returns.dropna())
    
    @staticmethod
    def calculate_kurtosis(returns: pd.Series) -> float:
        """Calculate kurtosis of returns"""
        if len(returns) < 4:
            return 0.0
        return stats.kurtosis(returns.dropna())
    
    @staticmethod
    def calculate_hit_ratio(predictions: pd.Series, actual: pd.Series) -> float:
        """Calculate hit ratio for predictions (directional accuracy)"""
        if len(predictions) != len(actual) or len(predictions) < 2:
            return 0.0
        
        pred_direction = np.sign(predictions.diff())
        actual_direction = np.sign(actual.diff())
        
        hits = (pred_direction == actual_direction).sum()
        total = len(pred_direction) - 1  # -1 because diff() reduces length by 1
        
        return hits / total if total > 0 else 0.0
    
    @staticmethod
    def calculate_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate correlation matrix for multiple assets"""
        return returns_df.corr()
    
    @staticmethod
    def calculate_portfolio_metrics(weights: np.ndarray, returns: pd.DataFrame, risk_free_rate: float = 0.02) -> Dict[str, float]:
        """Calculate portfolio-level metrics"""
        if len(weights) != len(returns.columns):
            raise ValueError("Number of weights must match number of assets")
        
        # Portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1)
        
        # Expected return
        expected_return = portfolio_returns.mean() * 252
        
        # Portfolio volatility
        cov_matrix = returns.cov() * 252
        portfolio_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        portfolio_volatility = np.sqrt(portfolio_variance)
        
        # Sharpe ratio
        sharpe = (expected_return - risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0.0
        
        # Max drawdown
        portfolio_prices = (1 + portfolio_returns).cumprod()
        max_dd_info = FinancialMetrics.calculate_max_drawdown(portfolio_prices)
        
        return {
            'expected_return': expected_return,
            'volatility': portfolio_volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd_info['max_drawdown'],
            'var_5': FinancialMetrics.calculate_var(portfolio_returns, 0.05),
            'cvar_5': FinancialMetrics.calculate_cvar(portfolio_returns, 0.05)
        }

class TechnicalIndicators:
    """Technical analysis indicators"""
    
    @staticmethod
    def sma(prices: pd.Series, window: int) -> pd.Series:
        """Simple Moving Average"""
        return prices.rolling(window=window).mean()
    
    @staticmethod
    def ema(prices: pd.Series, window: int) -> pd.Series:
        """Exponential Moving Average"""
        return prices.ewm(span=window).mean()
    
    @staticmethod
    def rsi(prices: pd.Series, window: int = 14) -> pd.Series:
        """Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    @staticmethod
    def bollinger_bands(prices: pd.Series, window: int = 20, num_std: float = 2.0) -> Dict[str, pd.Series]:
        """Bollinger Bands"""
        sma = prices.rolling(window=window).mean()
        std = prices.rolling(window=window).std()
        
        return {
            'upper': sma + (std * num_std),
            'middle': sma,
            'lower': sma - (std * num_std)
        }
    
    @staticmethod
    def macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Dict[str, pd.Series]:
        """MACD (Moving Average Convergence Divergence)"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal).mean()
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
        """Average True Range"""
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        return true_range.rolling(window=window).mean()
    
    @staticmethod
    def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, k_window: int = 14, d_window: int = 3) -> Dict[str, pd.Series]:
        """Stochastic Oscillator"""
        lowest_low = low.rolling(window=k_window).min()
        highest_high = high.rolling(window=k_window).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(window=d_window).mean()
        
        return {
            'k': k_percent,
            'd': d_percent
        }

class PerformanceAnalyzer:
    """Portfolio and strategy performance analysis"""
    
    @staticmethod
    def comprehensive_analysis(returns: pd.Series, benchmark_returns: pd.Series = None, risk_free_rate: float = 0.02) -> Dict[str, Any]:
        """Comprehensive performance analysis"""
        
        metrics = {}
        
        # Basic metrics
        metrics['total_return'] = FinancialMetrics.calculate_cumulative_returns(returns).iloc[-1] if len(returns) > 0 else 0.0
        metrics['annualized_return'] = returns.mean() * 252
        metrics['volatility'] = FinancialMetrics.calculate_volatility(returns)
        metrics['sharpe_ratio'] = FinancialMetrics.calculate_sharpe_ratio(returns, risk_free_rate)
        metrics['sortino_ratio'] = FinancialMetrics.calculate_sortino_ratio(returns, risk_free_rate)
        
        # Risk metrics
        prices = (1 + returns).cumprod()
        drawdown_info = FinancialMetrics.calculate_max_drawdown(prices)
        metrics['max_drawdown'] = drawdown_info['max_drawdown']
        metrics['calmar_ratio'] = FinancialMetrics.calculate_calmar_ratio(returns)
        
        # Distribution metrics
        metrics['skewness'] = FinancialMetrics.calculate_skewness(returns)
        metrics['kurtosis'] = FinancialMetrics.calculate_kurtosis(returns)
        
        # Risk measures
        metrics['var_5'] = FinancialMetrics.calculate_var(returns, 0.05)
        metrics['cvar_5'] = FinancialMetrics.calculate_cvar(returns, 0.05)
        metrics['omega_ratio'] = FinancialMetrics.calculate_omega_ratio(returns)
        
        # Benchmark comparison
        if benchmark_returns is not None:
            metrics['beta'] = FinancialMetrics.calculate_beta(returns, benchmark_returns)
            metrics['alpha'] = FinancialMetrics.calculate_alpha(returns, benchmark_returns, risk_free_rate)
            metrics['information_ratio'] = FinancialMetrics.calculate_information_ratio(returns, benchmark_returns)
            metrics['treynor_ratio'] = FinancialMetrics.calculate_treynor_ratio(returns, benchmark_returns, risk_free_rate)
        
        return metrics
    
    @staticmethod
    def rolling_metrics(returns: pd.Series, window: int = 252) -> pd.DataFrame:
        """Calculate rolling performance metrics"""
        
        rolling_data = pd.DataFrame(index=returns.index)
        
        rolling_data['rolling_return'] = returns.rolling(window=window).mean() * 252
        rolling_data['rolling_volatility'] = returns.rolling(window=window).std() * np.sqrt(252)
        rolling_data['rolling_sharpe'] = rolling_data['rolling_return'] / rolling_data['rolling_volatility']
        
        # Rolling max drawdown
        prices = (1 + returns).cumprod()
        rolling_max = prices.rolling(window=window).max()
        rolling_dd = (prices - rolling_max) / rolling_max
        rolling_data['rolling_max_drawdown'] = rolling_dd.rolling(window=window).min().abs()
        
        return rolling_data.dropna()

class ModelEvaluationMetrics:
    """Metrics for evaluating financial models"""
    
    @staticmethod
    def prediction_accuracy_metrics(predictions: pd.Series, actual: pd.Series) -> Dict[str, float]:
        """Calculate prediction accuracy metrics"""
        
        # Align series
        aligned_data = pd.concat([predictions, actual], axis=1).dropna()
        
        if len(aligned_data) < 2:
            return {
                'mae': float('inf'),
                'mse': float('inf'),
                'rmse': float('inf'),
                'mape': float('inf'),
                'hit_ratio': 0.0,
                'correlation': 0.0
            }
        
        pred = aligned_data.iloc[:, 0]
        act = aligned_data.iloc[:, 1]
        
        # Error metrics
        mae = mean_absolute_error(act, pred)
        mse = mean_squared_error(act, pred)
        rmse = np.sqrt(mse)
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((act - pred) / act)) * 100 if (act != 0).all() else float('inf')
        
        # Directional accuracy
        hit_ratio = FinancialMetrics.calculate_hit_ratio(pred, act)
        
        # Correlation
        correlation = pred.corr(act)
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': rmse,
            'mape': mape,
            'hit_ratio': hit_ratio,
            'correlation': correlation
        }
    
    @staticmethod
    def backtesting_metrics(strategy_returns: pd.Series, benchmark_returns: pd.Series = None) -> Dict[str, Any]:
        """Comprehensive backtesting metrics"""
        
        metrics = PerformanceAnalyzer.comprehensive_analysis(strategy_returns, benchmark_returns)
        
        # Additional backtesting-specific metrics
        win_rate = (strategy_returns > 0).mean()
        loss_rate = (strategy_returns < 0).mean()
        
        positive_returns = strategy_returns[strategy_returns > 0]
        negative_returns = strategy_returns[strategy_returns < 0]
        
        avg_win = positive_returns.mean() if len(positive_returns) > 0 else 0.0
        avg_loss = abs(negative_returns.mean()) if len(negative_returns) > 0 else 0.0
        
        profit_factor = (positive_returns.sum() / abs(negative_returns.sum())) if negative_returns.sum() != 0 else float('inf')
        
        metrics.update({
            'win_rate': win_rate,
            'loss_rate': loss_rate,
            'average_win': avg_win,
            'average_loss': avg_loss,
            'profit_factor': profit_factor,
            'total_trades': len(strategy_returns),
            'winning_trades': len(positive_returns),
            'losing_trades': len(negative_returns)
        })
        
        return metrics

# Utility functions for common calculations
def annualize_metric(metric_value: float, frequency: str = 'daily') -> float:
    """Annualize a metric based on frequency"""
    frequency_map = {
        'daily': 252,
        'weekly': 52,
        'monthly': 12,
        'quarterly': 4,
        'annual': 1
    }
    
    periods = frequency_map.get(frequency, 252)
    return metric_value * periods

def efficient_frontier_point(returns: pd.DataFrame, target_return: float) -> Dict[str, Any]:
    """Calculate efficient frontier point for given target return"""
    from scipy.optimize import minimize
    
    n_assets = len(returns.columns)
    mean_returns = returns.mean() * 252
    cov_matrix = returns.cov() * 252
    
    # Objective function: minimize portfolio variance
    def portfolio_variance(weights):
        return np.dot(weights.T, np.dot(cov_matrix, weights))
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},  # Weights sum to 1
        {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target_return}  # Target return
    ]
    
    # Bounds
    bounds = tuple((0, 1) for _ in range(n_assets))
    
    # Initial guess
    x0 = np.array([1/n_assets] * n_assets)
    
    # Optimize
    result = minimize(portfolio_variance, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    
    if result.success:
        optimal_weights = result.x
        portfolio_return = np.dot(optimal_weights, mean_returns)
        portfolio_risk = np.sqrt(portfolio_variance(optimal_weights))
        
        return {
            'weights': optimal_weights,
            'expected_return': portfolio_return,
            'volatility': portfolio_risk,
            'sharpe_ratio': portfolio_return / portfolio_risk if portfolio_risk > 0 else 0
        }
    else:
        return {'error': 'Optimization failed'}

def calculate_portfolio_attribution(weights: np.ndarray, returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate portfolio attribution analysis"""
    
    # Individual asset contributions
    asset_contributions = returns.multiply(weights, axis=1)
    
    # Portfolio return
    portfolio_return = asset_contributions.sum(axis=1)
    
    # Attribution DataFrame
    attribution = pd.DataFrame({
        'asset': returns.columns,
        'weight': weights,
        'asset_return': returns.mean() * 252,
        'contribution': asset_contributions.mean() * 252,
        'relative_contribution': (asset_contributions.mean() * 252) / (portfolio_return.mean() * 252) * 100
    })
    
    return attribution