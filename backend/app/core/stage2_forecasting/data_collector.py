# backend/app/core/stage2_forecasting/data_collector.py
"""
Financial data collection from various sources
"""

import asyncio
import yfinance as yf
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

class DataCollector:
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
    
    async def get_historical_data(self, symbol: str, period: str = "2y") -> pd.DataFrame:
        """Get historical price data for a symbol"""
        
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if (datetime.now() - timestamp).seconds < self.cache_timeout:
                return cached_data
        
        try:
            # Fetch data using yfinance
            ticker = yf.Ticker(symbol)
            data = ticker.history(period=period)
            
            # Cache the data
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            raise Exception(f"Failed to fetch data for {symbol}: {str(e)}")
    
    async def get_company_info(self, symbol: str) -> Dict[str, Any]:
        """Get company information and fundamentals"""
        
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Extract key metrics
            key_metrics = {
                "market_cap": info.get("marketCap"),
                "pe_ratio": info.get("trailingPE"),
                "forward_pe": info.get("forwardPE"),
                "peg_ratio": info.get("pegRatio"),
                "price_to_book": info.get("priceToBook"),
                "debt_to_equity": info.get("debtToEquity"),
                "roe": info.get("returnOnEquity"),
                "roa": info.get("returnOnAssets"),
                "profit_margin": info.get("profitMargins"),
                "revenue": info.get("totalRevenue"),
                "earnings_growth": info.get("earningsGrowth")
            }
            
            return {
                "symbol": symbol,
                "company_name": info.get("longName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "metrics": key_metrics,
                "last_updated": datetime.now().isoformat()
            }
            
        except Exception as e:
            raise Exception(f"Failed to fetch company info for {symbol}: {str(e)}")
    
    async def get_economic_indicators(self) -> Dict[str, Any]:
        """Get key economic indicators"""
        
        indicators = {}
        
        try:
            # S&P 500
            sp500 = yf.Ticker("^GSPC")
            sp500_data = sp500.history(period="1d")
            indicators["sp500"] = {
                "current": sp500_data["Close"].iloc[-1],
                "change": sp500_data["Close"].iloc[-1] - sp500_data["Open"].iloc[-1]
            }
            
            # VIX
            vix = yf.Ticker("^VIX")
            vix_data = vix.history(period="1d")
            indicators["vix"] = {
                "current": vix_data["Close"].iloc[-1],
                "level": "high" if vix_data["Close"].iloc[-1] > 25 else "normal"
            }
            
            return indicators
            
        except Exception as e:
            return {"error": f"Failed to fetch economic indicators: {str(e)}"}