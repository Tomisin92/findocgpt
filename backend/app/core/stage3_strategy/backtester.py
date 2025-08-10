# # backend/app/core/stage3_strategy/backtester.py
# """
# Strategy backtesting engine
# """

# import pandas as pd
# import numpy as np
# from typing import Dict, List, Any
# from datetime import datetime, timedelta
# import asyncio

# class Backtester:
#     def __init__(self):
#         self.initial_capital = 100000  # $100k default
#         self.commission = 0.001  # 0.1% commission
        
#     async def run_backtest(self, symbol: str, start_date: str, end_date: str, strategy: str) -> Dict[str, Any]:
#         """Run comprehensive strategy backtest"""
        
#         # Get historical data
#         data = await self._get_historical_data(symbol, start_date, end_date)
        
#         # Generate trading signals
#         signals = await self._generate_signals(data, strategy)
        
#         # Execute trades
#         trades = await self._execute_trades(data, signals)
        
#         # Calculate performance metrics
#         performance = await self._calculate_performance(trades, data)
        
#         # Calculate risk metrics
#         risk_metrics = await self._calculate_risk_metrics(trades)
        
#         # Compare to benchmark
#         benchmark = await self._compare_to_benchmark(trades, data, symbol)
        
#         return {
#             "total_return": performance["total_return"],
#             "annual_return": performance["annual_return"],
#             "max_drawdown": risk_metrics["max_drawdown"],
#             "sharpe_ratio": risk_metrics["sharpe_ratio"],
#             "win_rate": performance["win_rate"],
#             "trades": len(trades),
#             "risk_metrics": risk_metrics,
#             "benchmark": benchmark,
#             "trade_details": trades[-10:]  # Last 10 trades
#         }
    
#     async def _get_historical_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
#         """Get historical price data for backtesting"""
        
#         # Simplified historical data generation
#         date_range = pd.date_range(start=start_date, end=end_date, freq="D")
        
#         # Generate synthetic price data with trend and volatility
#         np.random.seed(42)  # For reproducible results
#         returns = np.random.normal(0.0005, 0.02, len(date_range))  # 0.05% daily return, 2% volatility
        
#         # Add some trend
#         trend = np.linspace(0, 0.3, len(date_range))  # 30% total trend
#         returns += trend / len(date_range)
        
#         # Calculate prices
#         prices = [100.0]  # Starting price
#         for ret in returns[1:]:
#             prices.append(prices[-1] * (1 + ret))
        
#         data = pd.DataFrame({
#             "Date": date_range,
#             "Close": prices,
#             "High": [p * 1.02 for p in prices],
#             "Low": [p * 0.98 for p in prices],
#             "Volume": np.random.randint(1000000, 5000000, len(date_range))
#         })
        
#         data.set_index("Date", inplace=True)
#         return data
    
#     async def _generate_signals(self, data: pd.DataFrame, strategy: str) -> pd.DataFrame:
#         """Generate trading signals based on strategy"""
        
#         signals = data.copy()
#         signals["signal"] = 0
#         signals["position"] = 0
        
#         if strategy == "momentum":
#             # Simple momentum strategy
#             signals["sma_short"] = data["Close"].rolling(window=20).mean()
#             signals["sma_long"] = data["Close"].rolling(window=50).mean()
            
#             # Buy when short MA crosses above long MA
#             signals.loc[signals["sma_short"] > signals["sma_long"], "signal"] = 1
#             signals.loc[signals["sma_short"] < signals["sma_long"], "signal"] = -1
            
#         elif strategy == "mean_reversion":
#             # Mean reversion strategy
#             signals["sma"] = data["Close"].rolling(window=30).mean()
#             signals["std"] = data["Close"].rolling(window=30).std()
#             signals["z_score"] = (data["Close"] - signals["sma"]) / signals["std"]
            
#             # Buy when oversold, sell when overbought
#             signals.loc[signals["z_score"] < -2, "signal"] = 1
#             signals.loc[signals["z_score"] > 2, "signal"] = -1
            
#         # Convert signals to positions
#         signals["position"] = signals["signal"].shift(1).fillna(0)
        
#         return signals
    
#     async def _execute_trades(self, data: pd.DataFrame, signals: pd.DataFrame) -> List[Dict[str, Any]]:
#         """Execute trades based on signals"""
        
#         trades = []
#         cash = self.initial_capital
#         position = 0
#         shares = 0
        
#         for date, row in signals.iterrows():
#             current_price = row["Close"]
#             signal = row["signal"]
            
#             # Execute trade if signal changed
#             if signal != position:
#                 if position != 0:  # Close existing position
#                     # Sell
#                     cash += shares * current_price * (1 - self.commission)
                    
#                     trades.append({
#                         "date": date,
#                         "action": "sell",
#                         "price": current_price,
#                         "shares": shares,
#                         "value": shares * current_price,
#                         "cash": cash
#                     })
                    
#                     shares = 0
                
#                 if signal != 0:  # Open new position
#                     # Buy
#                     shares = int(cash / (current_price * (1 + self.commission)))
#                     cash -= shares * current_price * (1 + self.commission)
                    
#                     trades.append({
#                         "date": date,
#                         "action": "buy",
#                         "price": current_price,
#                         "shares": shares,
#                         "value": shares * current_price,
#                         "cash": cash
#                     })
                
#                 position = signal
        
#         return trades
    
#     async def _calculate_performance(self, trades: List[Dict[str, Any]], data: pd.DataFrame) -> Dict[str, float]:
#         """Calculate performance metrics"""
        
#         if not trades:
#             return {"total_return": 0.0, "annual_return": 0.0, "win_rate": 0.0}
        
#         # Calculate returns for each trade pair
#         trade_returns = []
#         buy_trades = [t for t in trades if t["action"] == "buy"]
#         sell_trades = [t for t in trades if t["action"] == "sell"]
        
#         for buy, sell in zip(buy_trades, sell_trades):
#             trade_return = (sell["price"] - buy["price"]) / buy["price"]
#             trade_returns.append(trade_return)
        
#         # Final portfolio value
#         final_price = data["Close"].iloc[-1]
#         last_trade = trades[-1]
        
#         if last_trade["action"] == "buy":
#             # Still holding position
#             final_value = last_trade["cash"] + last_trade["shares"] * final_price
#         else:
#             # Cash position
#             final_value = last_trade["cash"]
        
#         # Performance metrics
#         total_return = (final_value - self.initial_capital) / self.initial_capital
        
#         # Annualize return
#         start_date = trades[0]["date"]
#         end_date = trades[-1]["date"]
#         days = (end_date - start_date).days
#         annual_return = (1 + total_return) ** (365 / days) - 1 if days > 0 else 0
        
#         # Win rate
#         winning_trades = [r for r in trade_returns if r > 0]
#         win_rate = len(winning_trades) / len(trade_returns) if trade_returns else 0
        
#         return {
#             "total_return": total_return,
#             "annual_return": annual_return,
#             "win_rate": win_rate
#         }
    
#     async def _calculate_risk_metrics(self, trades: List[Dict[str, Any]]) -> Dict[str, float]:
#         """Calculate risk metrics"""
        
#         if len(trades) < 2:
#             return {"max_drawdown": 0.0, "sharpe_ratio": 0.0, "volatility": 0.0}
        
#         # Calculate portfolio values over time
#         portfolio_values = []
        
#         for trade in trades:
#             if trade["action"] == "buy":
#                 portfolio_value = trade["cash"] + trade["shares"] * trade["price"]
#             else:
#                 portfolio_value = trade["cash"]
#             portfolio_values.append(portfolio_value)
        
#         # Calculate returns
#         returns = []
#         for i in range(1, len(portfolio_values)):
#             ret = (portfolio_values[i] - portfolio_values[i-1]) / portfolio_values[i-1]
#             returns.append(ret)
        
#         # Max drawdown
#         peak = portfolio_values[0]
#         max_drawdown = 0.0
        
#         for value in portfolio_values:
#             if value > peak:
#                 peak = value
#             drawdown = (peak - value) / peak
#             max_drawdown = max(max_drawdown, drawdown)
        
#         # Sharpe ratio
#         if returns:
#             avg_return = np.mean(returns)
#             volatility = np.std(returns)
#             sharpe_ratio = avg_return / volatility if volatility > 0 else 0
#         else:
#             volatility = 0
#             sharpe_ratio = 0
        
#         return {
#             "max_drawdown": max_drawdown,
#             "sharpe_ratio": sharpe_ratio,
#             "volatility": volatility
#         }
    
#     async def _compare_to_benchmark(self, trades: List[Dict[str, Any]], data: pd.DataFrame, symbol: str) -> Dict[str, float]:
#         """Compare strategy performance to buy-and-hold benchmark"""
        
#         if not trades:
#             return {"benchmark_return": 0.0, "excess_return": 0.0}
        
#         # Buy-and-hold return
#         start_price = data["Close"].iloc[0]
#         end_price = data["Close"].iloc[-1]
#         benchmark_return = (end_price - start_price) / start_price
        
#         # Strategy return
#         strategy_return = 0.0
#         if trades:
#             final_value = trades[-1]["cash"] if trades[-1]["action"] == "sell" else trades[-1]["cash"] + trades[-1]["shares"] * end_price
#             strategy_return = (final_value - self.initial_capital) / self.initial_capital
        
#         excess_return = strategy_return - benchmark_return
        
#         return {
#             "benchmark_return": benchmark_return,
#             "strategy_return": strategy_return,
#             "excess_return": excess_return,
#             "outperformed": excess_return > 0
#         }



# backend/app/core/stage3_strategy/backtester.py
"""Backtesting Framework"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime, timedelta

class Backtester:
    """Simple backtesting framework"""
    
    def __init__(self):
        self.initial_capital = 10000
    
    def run_backtest(self, symbol: str, strategy: str = "buy_and_hold") -> Dict:
        """Run a simple backtest"""
        try:
            # Placeholder backtest results
            return {
                "symbol": symbol,
                "strategy": strategy,
                "initial_capital": self.initial_capital,
                "final_value": 12000,
                "total_return": 0.20,
                "annual_return": 0.15,
                "max_drawdown": -0.10,
                "sharpe_ratio": 1.2,
                "number_of_trades": 5
            }
        except Exception as e:
            return {"error": f"Backtest failed: {str(e)}"}