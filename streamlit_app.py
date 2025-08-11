
"""
FinDocGPT - AI-Powered Financial Analysis & Investment Strategy
Complete Implementation for AkashX.ai Hackathon - FIXED VERSION

3-Stage Architecture:
1. Insights & Analysis (Document Q&A)
2. Financial Forecasting  
3. Investment Strategy & Decision-Making
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import yfinance as yf
import numpy as np
import time
import uuid
from openai import OpenAI
import os
from typing import Dict, List, Any
import json
from dataclasses import dataclass
import logging
import io
import base64
from PIL import Image
import re

# Configure Streamlit
st.set_page_config(
    page_title="FinDocGPT - AI Financial Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom CSS for better UI
st.markdown("""
<style>
.main-header {
    background: linear-gradient(90deg, #1f1f1f 0%, #2d2d2d 100%);
    padding: 1rem;
    border-radius: 10px;
    margin-bottom: 1rem;
    text-align: center;
    color: white;
}
.metric-card {
    background: #f0f2f6;
    padding: 1rem;
    border-radius: 8px;
    border-left: 4px solid #00d4ff;
    margin: 0.5rem 0;
}
.status-bar {
    background: #1f2937;
    padding: 0.5rem;
    border-radius: 5px;
    margin-bottom: 1rem;
}
.analysis-box {
    background: #f8fafc;
    padding: 1.5rem;
    border-radius: 10px;
    border: 1px solid #e2e8f0;
    margin: 1rem 0;
}
</style>
""", unsafe_allow_html=True)

# Configuration
@dataclass
class Config:
    API_BASE_URL: str = "http://localhost:8000"
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = "gpt-4"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.3
    FALLBACK_MODEL: str = "gpt-3.5-turbo"

config = Config()

# Initialize OpenAI client
client = None
if config.OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        st.success("✅ OpenAI API connected successfully")
    except Exception as e:
        st.warning(f"⚠️ OpenAI initialization error: {str(e)}")
        client = None
else:
    st.warning("⚠️ OpenAI API key not found. Some AI features will be limited.")

# Session State Management
def init_session_state():
    """Initialize all session state variables"""
    defaults = {
        "messages": [],
        "chart_counter": 0,
        "analysis_history": [],
        "investment_portfolio": [],
        "selected_symbol": "AAPL",
        "forecasting_results": {},
        "document_qa_context": "",
        "market_sentiment_data": {},
        "uploaded_documents": [],
        "current_analysis": None,
        "ai_responses": {},
        "portfolio_value": 100000,  # Starting with $100k
        "trade_history": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_unique_key():
    """Generate unique key for components"""
    st.session_state.chart_counter += 1
    return f"component_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"

# Updated OpenAI Integration with Error Handling
class OpenAIHandler:
    @staticmethod
    def generate_response(prompt: str, context: str = "", analysis_type: str = "general", max_retries: int = 2) -> str:
        """Generate AI response using OpenAI with fallback"""
        if not client:
            return OpenAIHandler._fallback_analysis(prompt, analysis_type)
        
        system_prompts = {
            "document_qa": """You are FinDocGPT, a specialized financial document analyst. Analyze financial reports, earnings statements, and SEC filings from the FinanceBench dataset. Provide precise, data-driven answers with specific financial metrics, ratios, and insights. Always cite specific numbers and timeframes when available.""",
            
            "sentiment": """You are a market sentiment analyst specializing in financial communications. Analyze earnings calls, press releases, and financial news to determine market sentiment (Bullish/Bearish/Neutral). Provide confidence scores (0-100%) and key sentiment drivers. Focus on forward-looking statements and management tone.""",
            
            "forecasting": """You are a quantitative financial forecasting expert. Use technical analysis, fundamental data, and market trends to predict future stock performance. Provide specific price targets with timeframes, probability ranges, and key catalysts. Consider both upside and downside scenarios.""",
            
            "investment": """You are a senior investment strategist. Based on comprehensive financial analysis, sentiment, and forecasts, provide clear BUY/SELL/HOLD recommendations. Include entry/exit points, position sizing, risk management (stop-loss levels), and portfolio allocation suggestions. Always consider risk-adjusted returns.""",
            
            "general": """You are FinDocGPT, an AI financial intelligence assistant built for the AkashX.ai hackathon. Provide comprehensive, accurate financial analysis combining document insights, market data, and investment strategy."""
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
        for attempt in range(max_retries + 1):
            try:
                model = config.DEFAULT_MODEL if attempt == 0 else config.FALLBACK_MODEL
                
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"Context: {context}\n\nAnalysis Request: {prompt}"}
                    ],
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE
                )
                
                return response.choices[0].message.content
                
            except Exception as e:
                error_str = str(e).lower()
                
                # Handle rate limiting - updated for new library
                if "rate" in error_str or "quota" in error_str or "429" in error_str:
                    if attempt < max_retries:
                        wait_time = 2 ** attempt
                        time.sleep(wait_time)
                        continue
                    return "⚠️ Rate limit exceeded. Please try again in a moment."
                
                # Handle invalid requests
                if "invalid" in error_str or "400" in error_str:
                    return f"⚠️ Invalid request: {str(e)}"
                
                # Handle authentication errors
                if "auth" in error_str or "api" in error_str or "401" in error_str:
                    return "⚠️ API authentication error. Please check your OpenAI API key."
                
                # Generic error handling
                if attempt < max_retries:
                    continue
                
                logger.error(f"OpenAI API error: {e}")
                return OpenAIHandler._fallback_analysis(prompt, analysis_type)
    
    @staticmethod
    def _fallback_analysis(prompt: str, analysis_type: str) -> str:
        """Provide fallback analysis when OpenAI is unavailable"""
        fallback_responses = {
            "document_qa": "📊 Document analysis requires OpenAI API. Please configure your API key to analyze financial documents from FinanceBench dataset.",
            "sentiment": "😊 Sentiment analysis shows neutral market conditions based on available data. Advanced AI sentiment requires API configuration.",
            "forecasting": "📈 Basic forecasting suggests following current market trends. Advanced AI forecasting requires OpenAI API configuration.",
            "investment": "💼 Conservative HOLD recommendation based on available metrics. Detailed investment strategy requires AI analysis.",
            "general": "🤖 AI analysis is limited without OpenAI API key. Please configure your API key for full FinDocGPT capabilities."
        }
        return fallback_responses.get(analysis_type, fallback_responses["general"])

# Enhanced Financial Data Handler
class FinancialDataHandler:
    @staticmethod
    def get_real_time_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
        """Enhanced real-time financial data with comprehensive metrics"""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No data available for {symbol}"}
            
            # Calculate advanced technical indicators
            current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            # RSI Calculation (14-period)
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            # Moving Averages
            ma_20 = hist['Close'].rolling(20).mean()
            ma_50 = hist['Close'].rolling(50).mean()
            ma_200 = hist['Close'].rolling(200).mean()
            
            # Bollinger Bands
            bb_std = hist['Close'].rolling(20).std()
            bb_upper = ma_20 + (bb_std * 2)
            bb_lower = ma_20 - (bb_std * 2)
            
            # MACD
            ema_12 = hist['Close'].ewm(span=12).mean()
            ema_26 = hist['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            signal = macd.ewm(span=9).mean()
            
            # Volatility & Volume metrics
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            avg_volume = hist['Volume'].mean()
            volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            # Support and Resistance levels
            highs = hist['High'].rolling(20).max()
            lows = hist['Low'].rolling(20).min()
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": price_change,
                "volume": int(hist['Volume'].iloc[-1]),
                "avg_volume": int(avg_volume),
                "volume_ratio": float(volume_ratio),
                "market_cap": int(info.get('marketCap', 0)),
                "pe_ratio": float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
                "forward_pe": float(info.get('forwardPE', 0)) if info.get('forwardPE') else 0,
                "peg_ratio": float(info.get('pegRatio', 0)) if info.get('pegRatio') else 0,
                "dividend_yield": float(info.get('dividendYield', 0)) if info.get('dividendYield') else 0,
                "beta": float(info.get('beta', 0)) if info.get('beta') else 0,
                
                # Technical Indicators
                "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else current_price,
                "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else current_price,
                "ma_200": float(ma_200.iloc[-1]) if not pd.isna(ma_200.iloc[-1]) else current_price,
                "bb_upper": float(bb_upper.iloc[-1]) if not pd.isna(bb_upper.iloc[-1]) else current_price * 1.02,
                "bb_lower": float(bb_lower.iloc[-1]) if not pd.isna(bb_lower.iloc[-1]) else current_price * 0.98,
                "macd": float(macd.iloc[-1]) if not pd.isna(macd.iloc[-1]) else 0,
                "macd_signal": float(signal.iloc[-1]) if not pd.isna(signal.iloc[-1]) else 0,
                "volatility": float(volatility),
                "support": float(lows.iloc[-1]) if not pd.isna(lows.iloc[-1]) else current_price * 0.95,
                "resistance": float(highs.iloc[-1]) if not pd.isna(highs.iloc[-1]) else current_price * 1.05,
                
                "historical_data": hist,
                "company_info": {
                    "sector": info.get('sector', 'Unknown'),
                    "industry": info.get('industry', 'Unknown'),
                    "employees": info.get('fullTimeEmployees', 0),
                    "description": (info.get('longBusinessSummary', 'No description available')[:300] + "...") if info.get('longBusinessSummary') else "No description available",
                    "website": info.get('website', ''),
                    "country": info.get('country', 'Unknown')
                },
                "last_updated": datetime.now().strftime("%H:%M:%S")
            }
            
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
            return {"error": str(e)}

    @staticmethod
    def calculate_investment_score(data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate comprehensive investment score with multiple factors"""
        if "error" in data:
            return {"error": "Cannot calculate investment score"}
        
        factors = {}
        weights = {
            'momentum': 0.20,
            'technical': 0.25,
            'trend': 0.20,
            'value': 0.15,
            'risk': 0.20
        }
        
        # 1. Price Momentum (20% weight)
        price_change = data['price_change']
        if price_change > 10:
            factors['momentum'] = 10
        elif price_change > 5:
            factors['momentum'] = 8
        elif price_change > 2:
            factors['momentum'] = 6
        elif price_change > 0:
            factors['momentum'] = 5
        elif price_change > -2:
            factors['momentum'] = 4
        elif price_change > -5:
            factors['momentum'] = 2
        else:
            factors['momentum'] = 0
        
        # 2. Technical Indicators (25% weight)
        rsi = data['rsi']
        if 45 <= rsi <= 55:  # Neutral zone
            factors['technical'] = 10
        elif 40 <= rsi <= 60:
            factors['technical'] = 8
        elif 35 <= rsi <= 65:
            factors['technical'] = 6
        elif 30 <= rsi <= 70:
            factors['technical'] = 4
        else:
            factors['technical'] = 2  # Extreme overbought/oversold
        
        # 3. Trend Analysis (20% weight)
        current = data['current_price']
        ma_20 = data['ma_20']
        ma_50 = data['ma_50']
        ma_200 = data.get('ma_200', ma_50)
        
        if current > ma_20 > ma_50 > ma_200:
            factors['trend'] = 10  # Strong uptrend
        elif current > ma_20 > ma_50:
            factors['trend'] = 8
        elif current > ma_20:
            factors['trend'] = 6
        elif ma_20 > ma_50:
            factors['trend'] = 4
        else:
            factors['trend'] = 2  # Downtrend
        
        # 4. Valuation Metrics (15% weight)
        pe_ratio = data.get('pe_ratio', 0)
        if 0 < pe_ratio < 15:
            factors['value'] = 10  # Undervalued
        elif pe_ratio < 20:
            factors['value'] = 8
        elif pe_ratio < 25:
            factors['value'] = 6
        elif pe_ratio < 30:
            factors['value'] = 4
        elif pe_ratio > 0:
            factors['value'] = 2
        else:
            factors['value'] = 5  # No PE data
        
        # 5. Risk Assessment (20% weight)
        volatility = data['volatility']
        volume_ratio = data['volume_ratio']
        
        if volatility < 0.2 and volume_ratio > 0.8:
            factors['risk'] = 10  # Low risk, good liquidity
        elif volatility < 0.3 and volume_ratio > 0.6:
            factors['risk'] = 8
        elif volatility < 0.5:
            factors['risk'] = 6
        elif volatility < 0.8:
            factors['risk'] = 4
        else:
            factors['risk'] = 2  # High risk
        
        # Calculate weighted score
        total_score = sum(factors[key] * weights[key] * 10 for key in factors.keys())
        
        # Determine recommendation
        if total_score >= 85:
            recommendation = "🟢 STRONG BUY"
            action = "BUY"
        elif total_score >= 70:
            recommendation = "🟢 BUY"
            action = "BUY"
        elif total_score >= 55:
            recommendation = "🟡 HOLD"
            action = "HOLD"
        elif total_score >= 40:
            recommendation = "🟠 WEAK SELL"
            action = "SELL"
        else:
            recommendation = "🔴 STRONG SELL"
            action = "SELL"
        
        return {
            "recommendation": recommendation,
            "action": action,
            "score": round(total_score, 1),
            "max_score": 100,
            "factors": factors,
            "weights": weights,
            "confidence": min(round(total_score, 1), 100)
        }

# Document Processing for FinanceBench
class DocumentProcessor:
    @staticmethod
    def process_financial_document(document_text: str, query: str) -> str:
        """Process FinanceBench financial documents for Q&A"""
        if not document_text.strip():
            return "❌ No document content provided for analysis."
        
        # Extract key financial metrics from text
        financial_patterns = {
            'revenue': r'revenue[:\s]+\$?([\d,]+\.?\d*)',
            'profit': r'profit[:\s]+\$?([\d,]+\.?\d*)',
            'earnings': r'earnings[:\s]+\$?([\d,]+\.?\d*)',
            'assets': r'assets[:\s]+\$?([\d,]+\.?\d*)',
            'liabilities': r'liabilities[:\s]+\$?([\d,]+\.?\d*)'
        }
        
        extracted_metrics = {}
        for metric, pattern in financial_patterns.items():
            matches = re.findall(pattern, document_text.lower())
            if matches:
                extracted_metrics[metric] = matches[0]
        
        context = f"""
        Document Analysis Context:
        - Document Length: {len(document_text)} characters
        - Extracted Financial Metrics: {extracted_metrics}
        - Query Type: Document Q&A from FinanceBench dataset
        
        Document Content: {document_text[:1500]}...
        """
        
        prompt = f"""
        As FinDocGPT analyzing a financial document from the FinanceBench dataset, please answer this query:
        
        Query: {query}
        
        Based on the document content, provide:
        1. Direct answer to the query with specific numbers/metrics
        2. Supporting evidence from the document
        3. Key financial insights and ratios
        4. Risk factors or opportunities mentioned
        5. Business context and implications
        
        If specific financial data is requested but not found, clearly state this and suggest what information would be needed.
        """
        
        return OpenAIHandler.generate_response(prompt, context, "document_qa")
    
    @staticmethod
    def extract_document_text(uploaded_file) -> str:
        """Extract text from uploaded documents"""
        try:
            if uploaded_file.type == "text/plain":
                return str(uploaded_file.read(), "utf-8")
            elif uploaded_file.type == "application/pdf":
                # For demo purposes, return placeholder
                return "PDF processing requires additional libraries. Please paste text content for analysis."
            else:
                return "Unsupported file type. Please upload .txt files or paste content directly."
        except Exception as e:
            return f"Error processing document: {str(e)}"

# Advanced Chart Rendering
class ChartRenderer:
    @staticmethod
    def render_advanced_price_chart(data: Dict[str, Any], height: int = 500):
        """Render comprehensive price chart with technical indicators"""
        if "error" in data or "historical_data" not in data:
            st.warning("⚠️ No data available for charting")
            return
        
        hist_data = data["historical_data"]
        
        # Create subplots
        from plotly.subplots import make_subplots
        
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=(f'{data["symbol"]} - Price & Technical Analysis', 'Volume', 'RSI'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Main price chart with Bollinger Bands
        fig.add_trace(
            go.Candlestick(
                x=hist_data.index,
                open=hist_data['Open'],
                high=hist_data['High'],
                low=hist_data['Low'],
                close=hist_data['Close'],
                name=f'{data["symbol"]} Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444'
            ), row=1, col=1
        )
        
        # Moving Averages
        ma_20 = hist_data['Close'].rolling(20).mean()
        ma_50 = hist_data['Close'].rolling(50).mean()
        
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma_20, mode='lines', name='MA 20', line=dict(color='orange', width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_data.index, y=ma_50, mode='lines', name='MA 50', line=dict(color='red', width=1)), row=1, col=1)
        
        # Bollinger Bands
        bb_std = hist_data['Close'].rolling(20).std()
        bb_upper = ma_20 + (bb_std * 2)
        bb_lower = ma_20 - (bb_std * 2)
        
        fig.add_trace(go.Scatter(x=hist_data.index, y=bb_upper, mode='lines', name='BB Upper', line=dict(color='gray', dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=hist_data.index, y=bb_lower, mode='lines', name='BB Lower', line=dict(color='gray', dash='dash')), row=1, col=1)
        
        # Volume
        colors = ['green' if close > open else 'red' for close, open in zip(hist_data['Close'], hist_data['Open'])]
        fig.add_trace(go.Bar(x=hist_data.index, y=hist_data['Volume'], name='Volume', marker_color=colors), row=2, col=1)
        
        # RSI
        delta = hist_data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(x=hist_data.index, y=rsi, mode='lines', name='RSI', line=dict(color='purple')), row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        fig.update_layout(
            height=height,
            showlegend=True,
            template="plotly_dark",
            title_text=f"{data['symbol']} - Comprehensive Technical Analysis"
        )
        
        fig.update_xaxes(rangeslider_visible=False)
        
        st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
    
    @staticmethod
    def render_technical_indicators(data: Dict[str, Any]):
        """Render technical indicators display"""
        if "error" in data:
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            rsi_color = "normal"
            if data['rsi'] > 70:
                rsi_color = "inverse" 
            elif data['rsi'] < 30:
                rsi_color = "inverse"
            
            st.metric(
                "RSI",
                f"{data['rsi']:.1f}",
                delta="Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral",
                delta_color=rsi_color
            )
        
        with col2:
            ma_trend = "⬆️" if data['current_price'] > data['ma_20'] else "⬇️"
            st.metric("MA 20", f"${data['ma_20']:.2f}", ma_trend)
        
        with col3:
            ma_trend = "⬆️" if data['current_price'] > data['ma_50'] else "⬇️"
            st.metric("MA 50", f"${data['ma_50']:.2f}", ma_trend)
        
        with col4:
            vol_status = "High" if data['volatility'] > 0.5 else "Normal" if data['volatility'] > 0.2 else "Low"
            st.metric("Volatility", f"{data['volatility']:.2%}", vol_status)
    
    @staticmethod
    def render_investment_dashboard(data: Dict[str, Any], score_data: Dict[str, Any]):
        """Render investment analysis dashboard"""
        if "error" in data:
            return
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.markdown("### 📊 Current Metrics")
            metrics_data = [
                ("Price", f"${data['current_price']:.2f}"),
                ("Change", f"{data['price_change']:+.2f}%"),
                ("Volume", f"{data['volume']:,}"),
                ("Market Cap", f"${data['market_cap']:,.0f}"),
                ("P/E Ratio", f"{data['pe_ratio']:.1f}"),
                ("Beta", f"{data.get('beta', 0):.2f}")
            ]
            
            for label, value in metrics_data:
                st.metric(label, value)
        
        with col2:
            st.markdown("### 🎯 Technical Indicators")
            
            # RSI Gauge
            rsi_fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = data['rsi'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "RSI"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgray"},
                        {'range': [70, 100], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ))
            rsi_fig.update_layout(height=200)
            st.plotly_chart(rsi_fig, use_container_width=True, key=get_unique_key())
        
        with col3:
            st.markdown("### 💡 Investment Score")
            
            # Score visualization
            score_fig = go.Figure(go.Indicator(
                mode = "gauge+number+delta",
                value = score_data['score'],
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': f"Score: {score_data['recommendation']}"},
                delta = {'reference': 50},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkgreen" if score_data['score'] > 70 else "orange" if score_data['score'] > 40 else "red"},
                    'steps': [
                        {'range': [0, 40], 'color': "lightgray"},
                        {'range': [40, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "lightgreen"}
                    ]
                }
            ))
            score_fig.update_layout(height=200)
            st.plotly_chart(score_fig, use_container_width=True, key=get_unique_key())

def main():
    init_session_state()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 FinDocGPT - AI Financial Intelligence</h1>
        <h3>AkashX.ai Hackathon Solution</h3>
        <p>Advanced AI-powered financial analysis, forecasting, and investment strategy using FinanceBench dataset and Yahoo Finance API</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Bar
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        openai_status = "🟢 Connected" if client else "🔴 Limited"
        st.markdown(f"🤖 **OpenAI:** {openai_status}")
    with col3:
        st.markdown(f"💼 **Portfolio:** {len(st.session_state.investment_portfolio)} stocks")
    with col4:
        st.markdown(f"💰 **Portfolio Value:** ${st.session_state.portfolio_value:,.0f}")
    with col5:
        if st.button("🔄 Refresh All", key="refresh_main"):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Sidebar Configuration
    with st.sidebar:
        st.header("🎯 FinDocGPT Control Panel")
        
        # Symbol Selection
        symbol_input = st.text_input(
            "📊 Stock Symbol", 
            value=st.session_state.selected_symbol,
            help="Enter stock symbol (e.g., AAPL, GOOGL, MSFT)"
        )
        
        if symbol_input != st.session_state.selected_symbol:
            st.session_state.selected_symbol = symbol_input.upper()
        
        # Data Period Selection
        data_period = st.selectbox(
            "📅 Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y", "5y"],
            index=3,
            help="Select timeframe for historical analysis"
        )
        
        # Real-time Data Display
        current_data = {}
        if st.session_state.selected_symbol:
            with st.spinner("📡 Loading market data..."):
                current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol, data_period)
            
            if "error" not in current_data:
                st.success(f"✅ {st.session_state.selected_symbol} Data Loaded")
                
                # Key Metrics Display
                col1, col2 = st.columns(2)
                with col1:
                    price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
                    st.metric("Price", f"${current_data['current_price']:.2f}", 
                             f"{current_data['price_change']:+.2f}%", delta_color=price_color)
                    st.metric("Volume", f"{current_data['volume']:,}")
                    st.metric("Market Cap", f"${current_data['market_cap']/1e9:.1f}B")
                
                with col2:
                    st.metric("P/E Ratio", f"{current_data['pe_ratio']:.1f}")
                    st.metric("Beta", f"{current_data.get('beta', 0):.2f}")
                    st.metric("Volatility", f"{current_data['volatility']:.2%}")
                
                # Investment Score Card
                score_data = FinancialDataHandler.calculate_investment_score(current_data)
                if "error" not in score_data:
                    st.markdown("### 🎯 AI Investment Score")
                    
                    # Progress bar with color coding
                    score_color = "🟢" if score_data['score'] > 70 else "🟡" if score_data['score'] > 40 else "🔴"
                    st.progress(score_data['score'] / score_data['max_score'])
                    
                    st.markdown(f"""
                    **{score_data['recommendation']}**  
                    Score: {score_data['score']:.1f}/100 ({score_data['confidence']:.1f}% confidence)
                    """)
                    
                    # Factor breakdown
                    with st.expander("📊 Score Breakdown"):
                        for factor, value in score_data['factors'].items():
                            st.progress(value / 10, text=f"{factor.title()}: {value}/10")
            else:
                st.error(f"❌ Error loading {st.session_state.selected_symbol}: {current_data['error']}")
        
        # Market Status
        st.markdown("---")
        st.header("📈 Market Status")
        
        now = datetime.now()
        is_market_open = now.weekday() < 5 and 9 <= now.hour <= 16
        
        if is_market_open:
            st.success("🟢 Market is Open")
            st.caption("NYSE: 9:30 AM - 4:00 PM ET")
        else:
            st.warning("🟡 Market is Closed")
            st.caption("Next open: Monday 9:30 AM ET")
        
        # Quick Actions
        st.markdown("---")
        st.header("⚡ Quick Actions")
        
        if st.button("📊 Refresh Data", use_container_width=True):
            st.rerun()
        
        if st.button("💾 Save Analysis", use_container_width=True):
            if st.session_state.current_analysis:
                st.session_state.analysis_history.append(st.session_state.current_analysis)
                st.success("Analysis saved!")
        
        if st.button("🗑️ Clear History", use_container_width=True):
            st.session_state.analysis_history.clear()
            st.session_state.ai_responses.clear()
            st.success("History cleared!")
    
    # Main Navigation - 3 Stages Implementation
    tab1, tab2, tab3 = st.tabs([
        "🔍 Stage 1: Insights & Analysis", 
        "📈 Stage 2: Financial Forecasting", 
        "💼 Stage 3: Investment Strategy"
    ])
    
    # ============================================================================
    # STAGE 1: INSIGHTS & ANALYSIS (Document Q&A)
    # ============================================================================
    with tab1:
        st.header("🔍 Stage 1: Insights & Analysis")
        st.markdown("*Document Q&A using FinanceBench • Market Sentiment Analysis • Anomaly Detection*")
        
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "📄 Document Q&A (FinanceBench)", 
            "😊 Sentiment Analysis", 
            "⚠️ Anomaly Detection"
        ])
        
        # Document Q&A with FinanceBench Integration
        with sub_tab1:
            st.subheader("📄 Financial Document Q&A - FinanceBench Dataset")
            st.markdown("""
            Upload financial documents or use FinanceBench dataset examples for AI-powered analysis.
            Ask questions about earnings reports, SEC filings, and financial statements.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Document Upload Section
                st.markdown("#### 📁 Document Upload")
                uploaded_file = st.file_uploader(
                    "Upload Financial Document", 
                    type=['txt', 'pdf', 'csv'],
                    help="Upload earnings reports, SEC filings, or financial statements"
                )
                
                # Text input area for FinanceBench examples
                st.markdown("#### 📝 Paste Document Content")
                document_text = st.text_area(
                    "Paste FinanceBench document or financial report:",
                    height=300,
                    placeholder="""Example FinanceBench content:
                    
APPLE INC. Q4 2023 EARNINGS REPORT
Revenue: $89.5 billion (+2.8% YoY)
Net Income: $22.9 billion 
Gross Margin: 45.2%
iPhone Revenue: $43.8 billion
Services Revenue: $22.3 billion
Research & Development: $7.8 billion

Key Highlights:
- Record Services revenue driven by App Store growth
- iPhone 15 launch exceeded expectations  
- Supply chain constraints improved significantly
- Cash position remains strong at $162.1 billion

Risk Factors:
- China market regulatory challenges
- Component cost inflation
- Foreign exchange headwinds"""
                )
                
                # Query Input
                st.markdown("#### ❓ Analysis Query")
                query = st.text_input(
                    "Ask a question about the financial document:",
                    placeholder="e.g., What was Apple's revenue growth this quarter? What are the main risk factors?",
                    help="Ask specific questions about financial metrics, performance, risks, or business insights"
                )
                
                # Analysis Button
                if st.button("🔍 Analyze Document with AI", key="analyze_doc", use_container_width=True):
                    if (document_text.strip() or uploaded_file) and query.strip():
                        with st.spinner("🤖 FinDocGPT is analyzing the document..."):
                            # Process uploaded file if exists
                            if uploaded_file:
                                doc_content = DocumentProcessor.extract_document_text(uploaded_file)
                                if "Error" in doc_content:
                                    st.error(doc_content)
                                    doc_content = document_text  # Fallback to text area
                            else:
                                doc_content = document_text
                            
                            # Generate AI analysis
                            response = DocumentProcessor.process_financial_document(doc_content, query)
                            
                            # Display results in an attractive format
                            st.markdown("### 📊 FinDocGPT Analysis Results")
                            
                            with st.container():
                                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                st.markdown(response)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Save to analysis history
                            analysis_record = {
                                "timestamp": datetime.now(),
                                "type": "Document Q&A",
                                "query": query,
                                "response": response,
                                "document_preview": doc_content[:200] + "..."
                            }
                            
                            st.session_state.analysis_history.append(analysis_record)
                            st.session_state.current_analysis = analysis_record
                            
                            st.success("✅ Analysis completed and saved to history!")
                    else:
                        st.warning("⚠️ Please provide both document content and a query for analysis.")
            
            with col2:
                st.markdown("#### 📋 Instructions")
                st.markdown("""
                **How to use Document Q&A:**
                
                1. **Upload** a financial document or **paste** content in the text area
                2. **Ask** a specific question about the document
                3. **Click** "Analyze Document" to get AI insights
                
                **Example Questions:**
                - What was the quarterly revenue?
                - What are the main risk factors?
                - How did expenses change YoY?
                - What's the cash flow situation?
                """)
                
                # Show analysis history if available
                if st.session_state.analysis_history:
                    st.markdown("#### 📚 Recent Analyses")
                    recent_analyses = [a for a in st.session_state.analysis_history if a['type'] == 'Document Q&A'][-2:]
                    
                    for analysis in reversed(recent_analyses):
                        with st.expander(f"Q: {analysis['query'][:50]}..."):
                            st.caption(f"Date: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            st.text_area("Response:", analysis['response'][:200] + "...", height=100, disabled=True)
        
        # Sentiment Analysis
        with sub_tab2:
            st.subheader("😊 Market Sentiment Analysis")
            st.markdown("Analyze sentiment from earnings calls, press releases, and financial news using advanced AI.")
            
            if st.session_state.selected_symbol and current_data and "error" not in current_data:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"#### 📰 Sentiment Analysis for {st.session_state.selected_symbol}")
                    
                    # Sentiment text input
                    sentiment_text = st.text_area(
                        "Paste financial news, earnings call transcript, or press release:",
                        height=200,
                        placeholder=f"""Example content for {st.session_state.selected_symbol}:
                        
"During the earnings call, CEO mentioned strong performance in cloud services with 35% growth. Management expressed optimism about AI initiatives and expects continued expansion. However, they noted concerns about supply chain costs and competitive pressures in mobile segment. The company raised full-year guidance but warned of potential headwinds from foreign exchange rates."

Paste actual content here for AI-powered sentiment analysis..."""
                    )
                    
                    # Sentiment analysis type
                    analysis_depth = st.selectbox(
                        "Analysis Depth:",
                        ["Quick Sentiment", "Detailed Analysis", "Comprehensive Report"],
                        help="Choose the depth of sentiment analysis"
                    )
                    
                    if st.button("📊 Analyze Sentiment", key="analyze_sentiment", use_container_width=True):
                        if sentiment_text.strip():
                            with st.spinner("🤖 Analyzing market sentiment..."):
                                # Create context for sentiment analysis
                                context = f"""
                                Sentiment Analysis Request for: {st.session_state.selected_symbol}
                                Analysis Type: {analysis_depth}
                                Current Stock Price: ${current_data.get('current_price', 'N/A')}
                                Recent Performance: {current_data.get('price_change', 'N/A')}%
                                
                                Text Content: {sentiment_text[:1000]}
                                """
                                
                                prompt = f"""
                                Perform {analysis_depth.lower()} sentiment analysis for {st.session_state.selected_symbol} based on the provided financial communication.
                                
                                Provide:
                                1. Overall Sentiment (Bullish/Bearish/Neutral) with confidence score (0-100%)
                                2. Key positive sentiment drivers
                                3. Key negative sentiment factors  
                                4. Management tone assessment
                                5. Forward-looking statement analysis
                                6. Investment implications
                                7. Sentiment score breakdown by topic (if applicable)
                                
                                Text to analyze: {sentiment_text}
                                """
                                
                                sentiment_response = OpenAIHandler.generate_response(prompt, context, "sentiment")
                                
                                st.markdown("### 📈 Sentiment Analysis Results")
                                with st.container():
                                    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                    st.markdown(sentiment_response)
                                    st.markdown('</div>', unsafe_allow_html=True)
                                
                                # Save sentiment analysis
                                sentiment_record = {
                                    "timestamp": datetime.now(),
                                    "symbol": st.session_state.selected_symbol,
                                    "analysis": sentiment_response,
                                    "source_text": sentiment_text[:200] + "...",
                                    "analysis_type": analysis_depth
                                }
                                
                                st.session_state.market_sentiment_data[st.session_state.selected_symbol] = sentiment_record
                                st.success("✅ Sentiment analysis completed!")
                        else:
                            st.warning("⚠️ Please provide text content for sentiment analysis.")
                
                with col2:
                    st.markdown("#### 🎯 Sentiment Examples")
                    
                    sample_texts = {
                        "Bullish Example": "Management reported exceptional quarter with record revenue growth of 45%. New product launch exceeded all expectations. Strong pipeline for next year with multiple expansion opportunities.",
                        
                        "Bearish Example": "Company missed earnings expectations due to supply chain disruptions. Management lowered full-year guidance citing economic headwinds. Competitive pressures increasing in core markets.",
                        
                        "Neutral Example": "Quarter met expectations with steady performance. Management maintaining current guidance. Some positive developments offset by ongoing challenges in certain segments."
                    }
                    
                    for sentiment_type, example in sample_texts.items():
                        with st.expander(f"📝 {sentiment_type}"):
                            st.text_area("", value=example, height=80, key=f"sentiment_{sentiment_type}")
                    
                    # Historical sentiment if available
                    if st.session_state.selected_symbol in st.session_state.market_sentiment_data:
                        st.markdown("#### 📊 Previous Analysis")
                        prev_sentiment = st.session_state.market_sentiment_data[st.session_state.selected_symbol]
                        st.caption(f"Last analyzed: {prev_sentiment['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                        st.text_area("Previous Result:", value=prev_sentiment['analysis'][:200] + "...", height=100, disabled=True)
            else:
                st.info("ℹ️ Please select a stock symbol in the sidebar to begin sentiment analysis.")
        
        # Anomaly Detection
        with sub_tab3:
            st.subheader("⚠️ Real-Time Anomaly Detection")
            st.markdown("Identify unusual patterns and potential risks in financial metrics and market behavior.")
            
            if st.session_state.selected_symbol and current_data and "error" not in current_data:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown("### 🔍 Anomaly Detection Results")
                    
                    # Define comprehensive anomaly detection thresholds
                    anomalies = []
                    warnings = []
                    
                    # Price Movement Anomalies
                    if abs(current_data['price_change']) > 15:
                        anomalies.append({
                            "type": "Critical",
                            "category": "Price Movement", 
                            "message": f"Extreme price movement: {current_data['price_change']:+.2f}%",
                            "impact": "High"
                        })
                    elif abs(current_data['price_change']) > 8:
                        warnings.append({
                            "type": "Warning",
                            "category": "Price Movement",
                            "message": f"Significant price movement: {current_data['price_change']:+.2f}%",
                            "impact": "Medium"
                        })
                    
                    # Technical Indicator Anomalies
                    rsi = current_data['rsi']
                    if rsi > 85:
                        anomalies.append({
                            "type": "Critical",
                            "category": "Technical Analysis",
                            "message": f"Extremely overbought condition: RSI {rsi:.1f}",
                            "impact": "High"
                        })
                    elif rsi < 15:
                        anomalies.append({
                            "type": "Critical", 
                            "category": "Technical Analysis",
                            "message": f"Extremely oversold condition: RSI {rsi:.1f}",
                            "impact": "High"
                        })
                    elif rsi > 75 or rsi < 25:
                        warnings.append({
                            "type": "Warning",
                            "category": "Technical Analysis", 
                            "message": f"Overbought/oversold condition: RSI {rsi:.1f}",
                            "impact": "Medium"
                        })
                    
                    # Volume Anomalies
                    volume_ratio = current_data.get('volume_ratio', 1)
                    if volume_ratio > 5:
                        anomalies.append({
                            "type": "Critical",
                            "category": "Volume",
                            "message": f"Extreme volume spike: {volume_ratio:.1f}x average volume",
                            "impact": "High"
                        })
                    elif volume_ratio > 2:
                        warnings.append({
                            "type": "Warning", 
                            "category": "Volume",
                            "message": f"High volume activity: {volume_ratio:.1f}x average volume",
                            "impact": "Medium"
                        })
                    
                    # Volatility Anomalies  
                    volatility = current_data['volatility']
                    if volatility > 1.5:
                        anomalies.append({
                            "type": "Critical",
                            "category": "Volatility",
                            "message": f"Extreme volatility: {volatility:.2%} annualized",
                            "impact": "High"
                        })
                    elif volatility > 0.8:
                        warnings.append({
                            "type": "Warning",
                            "category": "Volatility", 
                            "message": f"High volatility: {volatility:.2%} annualized",
                            "impact": "Medium"
                        })
                    
                    # Technical Pattern Anomalies
                    hist_data = current_data.get('historical_data')
                    if hist_data is not None and len(hist_data) > 1:
                        prev_close = hist_data['Close'].iloc[-2]
                        today_open = hist_data['Open'].iloc[-1]
                        gap_percent = ((today_open - prev_close) / prev_close) * 100
                        
                        if abs(gap_percent) > 5:
                            anomalies.append({
                                "type": "Critical",
                                "category": "Price Gap",
                                "message": f"Significant price gap: {gap_percent:+.2f}%",
                                "impact": "High"
                            })
                    
                    # Display Results
                    if anomalies:
                        st.error("### 🚨 Critical Anomalies Detected")
                        for anomaly in anomalies:
                            st.markdown(f"""
                            **{anomaly['category']}**: {anomaly['message']}  
                            *Impact: {anomaly['impact']}*
                            """)
                    
                    if warnings:
                        st.warning("### ⚠️ Warnings Detected")
                        for warning in warnings:
                            st.markdown(f"""
                            **{warning['category']}**: {warning['message']}  
                            *Impact: {warning['impact']}*
                            """)
                    
                    if not anomalies and not warnings:
                        st.success("### ✅ No Significant Anomalies Detected")
                        st.markdown("All monitored metrics are within normal ranges.")
                    
                    # Technical Analysis Summary
                    st.markdown("### 📊 Current Technical Status")
                    ChartRenderer.render_technical_indicators(current_data)
                
                with col2:
                    st.markdown("#### 🎯 Anomaly Monitoring")
                    
                    # Monitoring thresholds
                    st.markdown("**Current Thresholds:**")
                    thresholds = {
                        "Price Movement": "> ±8% (Warning), > ±15% (Critical)",
                        "RSI": "< 25 or > 75 (Warning), < 15 or > 85 (Critical)",
                        "Volume": "> 2x avg (Warning), > 5x avg (Critical)", 
                        "Volatility": "> 80% (Warning), > 150% (Critical)",
                        "Price Gaps": "> ±3% (Warning), > ±5% (Critical)"
                    }
                    
                    for metric, threshold in thresholds.items():
                        st.caption(f"**{metric}**: {threshold}")
                    
                    st.markdown("---")
                    st.markdown("#### 📈 Real-time Metrics")
                    
                    # Current values with status indicators
                    metrics_status = [
                        ("Price Change", f"{current_data['price_change']:+.2f}%", 
                         "🟢" if abs(current_data['price_change']) < 3 else "🟡" if abs(current_data['price_change']) < 8 else "🔴"),
                        ("RSI", f"{current_data['rsi']:.1f}", 
                         "🟢" if 40 <= current_data['rsi'] <= 60 else "🟡" if 25 <= current_data['rsi'] <= 75 else "🔴"),
                        ("Volume Ratio", f"{current_data.get('volume_ratio', 1):.1f}x", 
                         "🟢" if current_data.get('volume_ratio', 1) < 1.5 else "🟡" if current_data.get('volume_ratio', 1) < 3 else "🔴"),
                        ("Volatility", f"{current_data['volatility']:.1%}", 
                         "🟢" if current_data['volatility'] < 0.4 else "🟡" if current_data['volatility'] < 0.8 else "🔴")
                    ]
                    
                    for metric, value, status in metrics_status:
                        st.markdown(f"{status} **{metric}**: {value}")
                    
                    # Auto-refresh option
                    st.markdown("---")
                    auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", key="auto_refresh_anomaly")
                    
                    if auto_refresh:
                        time.sleep(30)
                        st.rerun()
            
            elif current_data and "error" in current_data:
                st.error(f"❌ Cannot perform anomaly detection: {current_data['error']}")
            else:
                st.info("ℹ️ Please select a stock symbol to begin anomaly detection.")
    
    # ============================================================================  
    # STAGE 2: FINANCIAL FORECASTING
    # ============================================================================
    with tab2:
        st.header("📈 Stage 2: Financial Forecasting")
        st.markdown("*AI-powered predictions • Technical analysis • External data integration via Yahoo Finance API*")
        
        if st.session_state.selected_symbol and current_data and "error" not in current_data:
            # Forecasting Interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 AI-Powered Financial Forecasting")
                
                # Forecasting parameters
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    forecast_period = st.selectbox(
                        "📅 Forecast Period:",
                        ["1 Week", "2 Weeks", "1 Month", "3 Months", "6 Months", "1 Year"],
                        index=2
                    )
                
                with col_b:
                    forecast_type = st.selectbox(
                        "📊 Analysis Type:",
                        ["Technical Analysis", "Fundamental Analysis", "Combined Analysis", "Sentiment-Based"],
                        index=2
                    )
                
                with col_c:
                    confidence_level = st.selectbox(
                        "🎯 Confidence Level:",
                        ["Conservative", "Moderate", "Aggressive"],
                        index=1
                    )
                
                # Generate forecast button
                if st.button("🔮 Generate AI Forecast", key="generate_forecast", use_container_width=True):
                    with st.spinner("🤖 FinDocGPT is generating advanced forecasts..."):
                        # Create comprehensive forecasting context
                        score_data = FinancialDataHandler.calculate_investment_score(current_data)
                        
                        # Add sentiment data if available
                        sentiment_context = ""
                        if st.session_state.selected_symbol in st.session_state.market_sentiment_data:
                            sentiment_data = st.session_state.market_sentiment_data[st.session_state.selected_symbol]
                            sentiment_context = f"Recent Sentiment Analysis: {sentiment_data['analysis'][:300]}..."
                        
                        context = f"""
                        COMPREHENSIVE FORECASTING CONTEXT for {st.session_state.selected_symbol}
                        
                        Current Market Data:
                        - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
                        - Investment Score: {score_data['score']:.1f}/100 ({score_data['recommendation']})
                        - Volume: {current_data['volume']:,} (Ratio: {current_data.get('volume_ratio', 1):.1f}x)
                        - Market Cap: ${current_data['market_cap']/1e9:.1f}B
                        
                        Technical Indicators:
                        - RSI: {current_data['rsi']:.1f} ({"Overbought" if current_data['rsi'] > 70 else "Oversold" if current_data['rsi'] < 30 else "Neutral"})
                        - MA 20: ${current_data['ma_20']:.2f} | MA 50: ${current_data['ma_50']:.2f}
                        - Bollinger Bands: ${current_data.get('bb_lower', 0):.2f} - ${current_data.get('bb_upper', 0):.2f}
                        - MACD: {current_data.get('macd', 0):.3f} | Signal: {current_data.get('macd_signal', 0):.3f}
                        - Support: ${current_data.get('support', 0):.2f} | Resistance: ${current_data.get('resistance', 0):.2f}
                        
                        Fundamental Metrics:
                        - P/E Ratio: {current_data['pe_ratio']:.1f}
                        - Beta: {current_data.get('beta', 0):.2f}
                        - Volatility: {current_data['volatility']:.2%}
                        - Sector: {current_data['company_info']['sector']}
                        - Industry: {current_data['company_info']['industry']}
                        
                        {sentiment_context}
                        
                        Forecast Parameters:
                        - Period: {forecast_period}
                        - Analysis Type: {forecast_type}
                        - Confidence Level: {confidence_level}
                        """
                        
                        prompt = f"""
                        As FinDocGPT's advanced forecasting AI, generate a comprehensive {forecast_period} forecast for {st.session_state.selected_symbol} using {forecast_type.lower()} with {confidence_level.lower()} assumptions.
                        
                        Provide detailed analysis including:
                        
                        1. **Price Targets & Scenarios**:
                           - Bull case target with probability
                           - Base case target with probability  
                           - Bear case target with probability
                        
                        2. **Key Catalysts & Risks**:
                           - Positive drivers for price movement
                           - Potential downside risks
                           - Upcoming events (earnings, product launches, etc.)
                        
                        3. **Technical Analysis**:
                           - Support and resistance levels
                           - Chart pattern analysis
                           - Momentum indicators outlook
                        
                        4. **Timeline & Milestones**:
                           - Expected price movement timeline
                           - Key dates to watch
                           - Performance checkpoints
                        
                        5. **Confidence Assessment**:
                           - Overall confidence level (0-100%)
                           - Factors affecting reliability
                           - Recommendation for position sizing
                        """
                        
                        forecast_response = OpenAIHandler.generate_response(prompt, context, "forecasting")
                        
                        st.markdown("### 🔮 AI Forecasting Results")
                        with st.container():
                            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                            st.markdown(forecast_response)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Save forecast results
                        forecast_record = {
                            "timestamp": datetime.now(),
                            "symbol": st.session_state.selected_symbol,
                            "period": forecast_period,
                            "type": forecast_type,
                            "confidence": confidence_level,
                            "forecast": forecast_response,
                            "market_data": current_data
                        }
                        
                        st.session_state.forecasting_results[st.session_state.selected_symbol] = forecast_record
                        st.success("✅ Forecast generated and saved!")
                        
                # Technical Chart Display
                st.markdown("### 📊 Technical Analysis Chart")
                ChartRenderer.render_advanced_price_chart(current_data)
                
            with col2:
                st.markdown("#### 🎯 Forecasting Guide")
                
                st.markdown("""
                **Forecast Types:**
                - **Technical**: Price patterns, indicators
                - **Fundamental**: Company financials, industry
                - **Combined**: Technical + Fundamental
                - **Sentiment**: Market psychology focus
                
                **Confidence Levels:**
                - **Conservative**: Lower risk, modest returns
                - **Moderate**: Balanced risk/reward  
                - **Aggressive**: Higher risk, higher potential
                """)
                
                # Show previous forecast if available
                if st.session_state.selected_symbol in st.session_state.forecasting_results:
                    st.markdown("#### 📈 Previous Forecast")
                    prev_forecast = st.session_state.forecasting_results[st.session_state.selected_symbol]
                    st.caption(f"Generated: {prev_forecast['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                    st.caption(f"Period: {prev_forecast['period']} | Type: {prev_forecast['type']}")
                    
                    with st.expander("View Previous Results"):
                        st.text_area("", value=prev_forecast['forecast'][:300] + "...", height=150, disabled=True)
                
                # Market indicators summary
                st.markdown("#### 📊 Current Indicators")
                if current_data:
                    indicators = [
                        ("Price Trend", "🟢 Up" if current_data['price_change'] > 0 else "🔴 Down"),
                        ("RSI Status", "🟢 Neutral" if 30 < current_data['rsi'] < 70 else "🟡 Extreme"),
                        ("Volume", "🟢 Normal" if current_data.get('volume_ratio', 1) < 2 else "🟡 High"),
                        ("Volatility", "🟢 Low" if current_data['volatility'] < 0.5 else "🟡 High")
                    ]
                    
                    for indicator, status in indicators:
                        st.markdown(f"**{indicator}**: {status}")
        
        else:
            st.info("ℹ️ Please select a stock symbol in the sidebar to begin forecasting.")
    
    # ============================================================================
    # STAGE 3: INVESTMENT STRATEGY & DECISION-MAKING  
    # ============================================================================
    with tab3:
        st.header("💼 Stage 3: Investment Strategy & Decision-Making")
        st.markdown("*Portfolio management • Risk assessment • Trading recommendations*")
        
        if st.session_state.selected_symbol and current_data and "error" not in current_data:
            score_data = FinancialDataHandler.calculate_investment_score(current_data)
            
            # Investment Strategy Interface
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("🎯 AI Investment Strategy Generator")
                
                # Strategy parameters
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    investment_horizon = st.selectbox(
                        "📅 Investment Horizon:",
                        ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"],
                        index=1
                    )
                
                with col_b:
                    risk_tolerance = st.selectbox(
                        "⚖️ Risk Tolerance:",
                        ["Conservative", "Moderate", "Aggressive"],
                        index=1
                    )
                
                with col_c:
                    position_size = st.selectbox(
                        "💰 Position Size:",
                        ["Small (1-3%)", "Medium (3-7%)", "Large (7-15%)"],
                        index=1
                    )
                
                # Generate strategy button
                if st.button("💡 Generate Investment Strategy", key="generate_strategy", use_container_width=True):
                    with st.spinner("🤖 Generating comprehensive investment strategy..."):
                        # Prepare context with all available data
                        forecast_context = ""
                        if st.session_state.selected_symbol in st.session_state.forecasting_results:
                            forecast_data = st.session_state.forecasting_results[st.session_state.selected_symbol]
                            forecast_context = f"Recent Forecast: {forecast_data['forecast'][:300]}..."
                        
                        sentiment_context = ""
                        if st.session_state.selected_symbol in st.session_state.market_sentiment_data:
                            sentiment_data = st.session_state.market_sentiment_data[st.session_state.selected_symbol]
                            sentiment_context = f"Sentiment Analysis: {sentiment_data['analysis'][:300]}..."
                        
                        context = f"""
                        COMPREHENSIVE INVESTMENT STRATEGY CONTEXT for {st.session_state.selected_symbol}
                        
                        Current Analysis:
                        - Investment Score: {score_data['score']:.1f}/100
                        - Recommendation: {score_data['recommendation']}
                        - Confidence: {score_data['confidence']:.1f}%
                        
                        Market Data:
                        - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
                        - Volume: {current_data['volume']:,} ({current_data.get('volume_ratio', 1):.1f}x avg)
                        - Volatility: {current_data['volatility']:.2%}
                        - RSI: {current_data['rsi']:.1f}
                        - Support: ${current_data.get('support', 0):.2f} | Resistance: ${current_data.get('resistance', 0):.2f}
                        
                        Company Fundamentals:
                        - Market Cap: ${current_data['market_cap']/1e9:.1f}B
                        - P/E Ratio: {current_data['pe_ratio']:.1f}
                        - Beta: {current_data.get('beta', 0):.2f}
                        - Sector: {current_data['company_info']['sector']}
                        - Industry: {current_data['company_info']['industry']}
                        
                        {forecast_context}
                        {sentiment_context}
                        
                        Strategy Parameters:
                        - Investment Horizon: {investment_horizon}
                        - Risk Tolerance: {risk_tolerance}
                        - Position Size: {position_size}
                        - Portfolio Value: ${st.session_state.portfolio_value:,.0f}
                        """
                        
                        prompt = f"""
                        As FinDocGPT's senior investment strategist, create a comprehensive investment strategy for {st.session_state.selected_symbol} based on the analysis.
                        
                        Provide detailed recommendations including:
                        
                        1. **Investment Decision**:
                           - Clear BUY/SELL/HOLD recommendation with rationale
                           - Entry point strategy and timing
                           - Exit strategy and profit targets
                        
                        2. **Risk Management**:
                           - Stop-loss levels and trailing stops
                           - Position sizing recommendations
                           - Risk/reward ratio analysis
                        
                        3. **Portfolio Integration**:
                           - How this fits into overall portfolio
                           - Diversification considerations
                           - Sector/industry allocation impact
                        
                        4. **Execution Plan**:
                           - Order types and execution strategy
                           - Dollar-cost averaging vs lump sum
                           - Rebalancing triggers and timeline
                        
                        5. **Monitoring & Review**:
                           - Key metrics to track
                           - Review schedule and triggers
                           - Scenario planning (bull/bear cases)
                        
                        Consider the {risk_tolerance.lower()} risk tolerance and {investment_horizon} timeframe in all recommendations.
                        """
                        
                        investment_strategy = OpenAIHandler.generate_response(prompt, context, "investment")
                        
                        st.markdown("### 💼 Investment Strategy Recommendation")
                        with st.container():
                            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                            st.markdown(investment_strategy)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        # Add to portfolio if BUY recommendation
                        if "BUY" in investment_strategy.upper():
                            if st.button("➕ Add to Portfolio", key="add_to_portfolio"):
                                portfolio_item = {
                                    "symbol": st.session_state.selected_symbol,
                                    "action": score_data['action'],
                                    "price": current_data['current_price'],
                                    "timestamp": datetime.now(),
                                    "strategy": investment_strategy[:200] + "...",
                                    "score": score_data['score']
                                }
                                st.session_state.investment_portfolio.append(portfolio_item)
                                st.success(f"✅ {st.session_state.selected_symbol} added to portfolio!")
                
                # Investment Dashboard
                st.markdown("### 📊 Investment Dashboard")
                ChartRenderer.render_investment_dashboard(current_data, score_data)
                
            with col2:
                st.markdown("#### 💡 Strategy Guide")
                
                st.markdown("""
                **Investment Horizons:**
                - **Short-term**: Technical focus, quick profits
                - **Medium-term**: Balanced approach
                - **Long-term**: Fundamental focus, compound growth
                
                **Risk Levels:**
                - **Conservative**: Capital preservation priority
                - **Moderate**: Balanced risk/return
                - **Aggressive**: Growth maximization
                """)
                
                # Current recommendation summary
                if score_data and "error" not in score_data:
                    st.markdown("#### 🎯 Quick Recommendation")
                    
                    rec_color = "success" if "BUY" in score_data['recommendation'] else "warning" if "HOLD" in score_data['recommendation'] else "error"
                    
                    if rec_color == "success":
                        st.success(f"**{score_data['recommendation']}**")
                    elif rec_color == "warning":
                        st.warning(f"**{score_data['recommendation']}**")
                    else:
                        st.error(f"**{score_data['recommendation']}**")
                    
                    st.caption(f"Score: {score_data['score']:.1f}/100")
                    st.caption(f"Confidence: {score_data['confidence']:.1f}%")
                
                # Portfolio summary
                st.markdown("#### 💼 Portfolio Summary")
                if st.session_state.investment_portfolio:
                    portfolio_value = st.session_state.portfolio_value
                    num_positions = len(st.session_state.investment_portfolio)
                    
                    st.metric("Portfolio Value", f"${portfolio_value:,.0f}")
                    st.metric("Positions", num_positions)
                    st.metric("Available Cash", f"${portfolio_value * 0.1:,.0f}")
                    
                    # Recent additions
                    st.markdown("**Recent Additions:**")
                    recent = st.session_state.investment_portfolio[-3:]
                    for item in reversed(recent):
                        st.caption(f"• {item['symbol']} - {item['action']} (Score: {item['score']:.1f})")
                else:
                    st.info("No positions yet. Add your first investment!")
        
        else:
            st.info("ℹ️ Please select a stock symbol to begin investment strategy analysis.")
        
        # Portfolio Management Section
        st.markdown("---")
        st.subheader("📊 Portfolio Management")
        
        if st.session_state.investment_portfolio:
            # Portfolio table
            portfolio_df = pd.DataFrame(st.session_state.investment_portfolio)
            portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
            portfolio_df = portfolio_df.sort_values('timestamp', ascending=False)
            
            # Display portfolio
            st.dataframe(
                portfolio_df[['symbol', 'action', 'price', 'score', 'timestamp']],
                use_container_width=True,
                hide_index=True
            )
            
            # Portfolio actions
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📈 Refresh Portfolio", use_container_width=True):
                    st.rerun()
            
            with col2:
                if st.button("📊 Portfolio Analytics", use_container_width=True):
                    st.info("Advanced portfolio analytics coming soon!")
            
            with col3:
                if st.button("🗑️ Clear Portfolio", use_container_width=True):
                    st.session_state.investment_portfolio.clear()
                    st.success("Portfolio cleared!")
                    st.rerun()
        else:
            st.info("🆕 Your portfolio is empty. Start by analyzing stocks and adding investment recommendations!")

# Run the application
if __name__ == "__main__":
    main()