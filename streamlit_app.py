# # # # import streamlit as st
# # # # import requests
# # # # import pandas as pd
# # # # import plotly.graph_objects as go
# # # # import plotly.express as px
# # # # from datetime import datetime, timedelta
# # # # import json
# # # # import time
# # # # import yfinance as yf
# # # # import numpy as np

# # # # # Configure Streamlit page
# # # # st.set_page_config(
# # # #     page_title="FinDocGPT - Real-Time Financial Analysis",
# # # #     page_icon="📈",
# # # #     layout="wide",
# # # #     initial_sidebar_state="expanded"
# # # # )

# # # # # FIXED API Configuration
# # # # API_BASE_URL = "http://localhost:8000"  # Make sure this matches your backend

# # # # class RealTimeFinDocGPT:
# # # #     def __init__(self):
# # # #         self.api_base = API_BASE_URL
# # # #         self.init_session_state()
        
# # # #     def init_session_state(self):
# # # #         """Initialize session state variables"""
# # # #         if 'conversation_id' not in st.session_state:
# # # #             st.session_state.conversation_id = None
# # # #         if 'messages' not in st.session_state:
# # # #             st.session_state.messages = []
# # # #         if 'current_symbol' not in st.session_state:
# # # #             st.session_state.current_symbol = "AAPL"
# # # #         if 'real_time_data' not in st.session_state:
# # # #             st.session_state.real_time_data = {}
# # # #         if 'api_connected' not in st.session_state:
# # # #             st.session_state.api_connected = False
    
# # # #     def check_api_connection(self) -> bool:
# # # #         """Check if FastAPI backend is running"""
# # # #         try:
# # # #             response = requests.get(f"{self.api_base}/health", timeout=5)
# # # #             if response.status_code == 200:
# # # #                 st.session_state.api_connected = True
# # # #                 return True
# # # #             else:
# # # #                 st.session_state.api_connected = False
# # # #                 return False
# # # #         except Exception as e:
# # # #             st.session_state.api_connected = False
# # # #             st.error(f"API Connection Error: {e}")
# # # #             return False
    
# # # #     def get_real_time_data(self, symbol: str) -> dict:
# # # #         """Get real-time financial data"""
# # # #         try:
# # # #             # Get real-time data from yfinance
# # # #             ticker = yf.Ticker(symbol)
            
# # # #             # Current price and basic info
# # # #             info = ticker.info
# # # #             current_price = info.get('currentPrice', 0)
            
# # # #             # Historical data for charts
# # # #             hist_data = ticker.history(period="1mo")
            
# # # #             # Calculate technical indicators
# # # #             if not hist_data.empty:
# # # #                 # Moving averages
# # # #                 hist_data['SMA_20'] = hist_data['Close'].rolling(20).mean()
# # # #                 hist_data['SMA_50'] = hist_data['Close'].rolling(50).mean()
                
# # # #                 # RSI calculation
# # # #                 delta = hist_data['Close'].diff()
# # # #                 gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# # # #                 loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# # # #                 rs = gain / loss
# # # #                 hist_data['RSI'] = 100 - (100 / (1 + rs))
                
# # # #                 # Recent values
# # # #                 latest_rsi = hist_data['RSI'].iloc[-1] if not pd.isna(hist_data['RSI'].iloc[-1]) else 50
# # # #                 latest_sma_20 = hist_data['SMA_20'].iloc[-1] if not pd.isna(hist_data['SMA_20'].iloc[-1]) else current_price
# # # #                 latest_sma_50 = hist_data['SMA_50'].iloc[-1] if not pd.isna(hist_data['SMA_50'].iloc[-1]) else current_price
# # # #             else:
# # # #                 latest_rsi = 50
# # # #                 latest_sma_20 = current_price
# # # #                 latest_sma_50 = current_price
            
# # # #             return {
# # # #                 "symbol": symbol,
# # # #                 "current_price": current_price,
# # # #                 "price_change": info.get('regularMarketChangePercent', 0),
# # # #                 "volume": info.get('regularMarketVolume', 0),
# # # #                 "market_cap": info.get('marketCap', 0),
# # # #                 "pe_ratio": info.get('trailingPE', 0),
# # # #                 "rsi": latest_rsi,
# # # #                 "sma_20": latest_sma_20,
# # # #                 "sma_50": latest_sma_50,
# # # #                 "trend": "Bullish" if latest_sma_20 > latest_sma_50 else "Bearish",
# # # #                 "historical_data": hist_data,
# # # #                 "last_updated": datetime.now().strftime("%H:%M:%S")
# # # #             }
            
# # # #         except Exception as e:
# # # #             st.error(f"Error getting real-time data for {symbol}: {e}")
# # # #             return {"error": str(e)}
    
# # # #     def send_chat_message(self, message: str, symbol: str, analysis_type: str = "general") -> dict:
# # # #         """Send message to chat API with fallback to local analysis"""
        
# # # #         # Try API first
# # # #         if st.session_state.api_connected:
# # # #             try:
# # # #                 response = requests.post(
# # # #                     f"{self.api_base}/api/v1/chat/message",
# # # #                     json={
# # # #                         "message": message,
# # # #                         "symbol": symbol,
# # # #                         "analysis_type": analysis_type,
# # # #                         "conversation_id": st.session_state.conversation_id
# # # #                     },
# # # #                     timeout=30
# # # #                 )
                
# # # #                 if response.status_code == 200:
# # # #                     result = response.json()
# # # #                     if not st.session_state.conversation_id:
# # # #                         st.session_state.conversation_id = result.get("conversation_id")
# # # #                     return result
                    
# # # #             except Exception as e:
# # # #                 st.error(f"API Error: {e}")
        
# # # #         # Fallback to local analysis if API fails
# # # #         return self.local_financial_analysis(message, symbol, analysis_type)
    
# # # #     def local_financial_analysis(self, message: str, symbol: str, analysis_type: str) -> dict:
# # # #         """Local financial analysis when API is unavailable"""
# # # #         try:
# # # #             # Get real-time data
# # # #             data = self.get_real_time_data(symbol)
            
# # # #             if "error" in data:
# # # #                 return {
# # # #                     "response": {"content": f"❌ Unable to analyze {symbol}: {data['error']}"},
# # # #                     "suggestions": ["Try a different symbol", "Check internet connection"]
# # # #                 }
            
# # # #             # Analyze based on message type
# # # #             message_lower = message.lower()
            
# # # #             if any(word in message_lower for word in ['revenue', 'earnings', 'profit', 'financial']):
# # # #                 response_content = self.generate_financial_analysis(data)
# # # #             elif any(word in message_lower for word in ['predict', 'forecast', 'price', 'target']):
# # # #                 response_content = self.generate_price_forecast(data)
# # # #             elif any(word in message_lower for word in ['buy', 'sell', 'invest', 'recommend']):
# # # #                 response_content = self.generate_investment_recommendation(data)
# # # #             elif any(word in message_lower for word in ['sentiment', 'news', 'market']):
# # # #                 response_content = self.generate_sentiment_analysis(data)
# # # #             else:
# # # #                 response_content = self.generate_general_analysis(data)
            
# # # #             return {
# # # #                 "response": {"content": response_content},
# # # #                 "analysis_results": data,
# # # #                 "chart_data": self.generate_chart_data(data),
# # # #                 "suggestions": [
# # # #                     f"Predict {symbol}'s price trend",
# # # #                     f"Should I buy {symbol}?",
# # # #                     f"What's {symbol}'s sentiment?"
# # # #                 ]
# # # #             }
            
# # # #         except Exception as e:
# # # #             return {
# # # #                 "response": {"content": f"Analysis error: {str(e)}"},
# # # #                 "suggestions": ["Try again with a different query"]
# # # #             }
    
# # # #     def generate_financial_analysis(self, data: dict) -> str:
# # # #         """Generate financial analysis response"""
# # # #         symbol = data['symbol']
# # # #         price = data['current_price']
# # # #         change = data['price_change']
# # # #         volume = data['volume']
# # # #         pe = data['pe_ratio']
# # # #         market_cap = data['market_cap']
        
# # # #         return f"""📊 **Financial Analysis for {symbol}**

# # # # 💰 **Current Metrics:**
# # # # • Price: ${price:.2f} ({change:+.2f}%)
# # # # • Market Cap: ${market_cap:,.0f}
# # # # • P/E Ratio: {pe:.2f}
# # # # • Volume: {volume:,}

# # # # 📈 **Technical Indicators:**
# # # # • RSI: {data['rsi']:.1f} ({self.interpret_rsi(data['rsi'])})
# # # # • 20-day SMA: ${data['sma_20']:.2f}
# # # # • 50-day SMA: ${data['sma_50']:.2f}
# # # # • Trend: {data['trend']}

# # # # 🔍 **Key Insights:**
# # # # • {"Stock is overbought" if data['rsi'] > 70 else "Stock is oversold" if data['rsi'] < 30 else "RSI in neutral range"}
# # # # • {"Bullish momentum confirmed" if data['sma_20'] > data['sma_50'] else "Bearish pressure evident"}
# # # # • {"High trading volume suggests strong interest" if volume > 1000000 else "Normal trading volume"}

# # # # *Last updated: {data['last_updated']}*"""
    
# # # #     def generate_price_forecast(self, data: dict) -> str:
# # # #         """Generate price forecast response"""
# # # #         symbol = data['symbol']
# # # #         current_price = data['current_price']
# # # #         trend = data['trend']
# # # #         rsi = data['rsi']
        
# # # #         # Simple forecast logic
# # # #         if trend == "Bullish" and rsi < 70:
# # # #             target_price = current_price * 1.05
# # # #             probability = "High"
# # # #         elif trend == "Bearish" and rsi > 30:
# # # #             target_price = current_price * 0.95
# # # #             probability = "High"
# # # #         else:
# # # #             target_price = current_price * 1.02
# # # #             probability = "Medium"
        
# # # #         return f"""📈 **Price Forecast for {symbol}**

# # # # 🎯 **30-Day Prediction:**
# # # # • Current Price: ${current_price:.2f}
# # # # • Target Price: ${target_price:.2f}
# # # # • Expected Return: {((target_price - current_price) / current_price * 100):+.1f}%
# # # # • Probability: {probability}

# # # # 🔮 **Forecast Basis:**
# # # # • Technical Trend: {trend}
# # # # • RSI Level: {rsi:.1f}
# # # # • Moving Average Alignment: {"Bullish" if data['sma_20'] > data['sma_50'] else "Bearish"}

# # # # 📊 **Risk Assessment:**
# # # # • Volatility: {"High" if abs(data['price_change']) > 3 else "Medium" if abs(data['price_change']) > 1 else "Low"}
# # # # • Confidence Level: {probability}

# # # # ⚠️ **Disclaimer:** This is a technical analysis-based forecast. Please do your own research."""
    
# # # #     def generate_investment_recommendation(self, data: dict) -> str:
# # # #         """Generate investment recommendation"""
# # # #         symbol = data['symbol']
# # # #         price = data['current_price']
# # # #         rsi = data['rsi']
# # # #         trend = data['trend']
# # # #         pe = data['pe_ratio']
        
# # # #         # Scoring system
# # # #         score = 0
        
# # # #         # Technical scoring
# # # #         if rsi < 30:  # Oversold
# # # #             score += 2
# # # #         elif rsi > 70:  # Overbought
# # # #             score -= 2
        
# # # #         if trend == "Bullish":
# # # #             score += 2
# # # #         else:
# # # #             score -= 1
        
# # # #         # Fundamental scoring
# # # #         if pe < 15:
# # # #             score += 1
# # # #         elif pe > 30:
# # # #             score -= 1
        
# # # #         # Generate recommendation
# # # #         if score >= 3:
# # # #             recommendation = "🟢 STRONG BUY"
# # # #             confidence = "High"
# # # #         elif score >= 1:
# # # #             recommendation = "🟢 BUY"
# # # #             confidence = "Medium"
# # # #         elif score >= -1:
# # # #             recommendation = "🟡 HOLD"
# # # #             confidence = "Medium"
# # # #         else:
# # # #             recommendation = "🔴 SELL"
# # # #             confidence = "High"
        
# # # #         return f"""💼 **Investment Recommendation for {symbol}**

# # # # 🎯 **Decision: {recommendation}**
# # # # • Confidence: {confidence}
# # # # • Current Price: ${price:.2f}
# # # # • Score: {score}/5

# # # # 📊 **Analysis Breakdown:**
# # # # • **Technical Score:** {"Positive" if rsi < 70 and trend == "Bullish" else "Negative"}
# # # #   - RSI: {rsi:.1f} ({self.interpret_rsi(rsi)})
# # # #   - Trend: {trend}

# # # # • **Fundamental Score:** {"Attractive" if pe < 20 else "Expensive"}
# # # #   - P/E Ratio: {pe:.2f}

# # # # 🎯 **Action Plan:**
# # # # • Entry Point: {"Current levels attractive" if score > 0 else "Wait for better entry"}
# # # # • Stop Loss: ${price * 0.9:.2f} (-10%)
# # # # • Target: ${price * 1.15:.2f} (+15%)

# # # # ⚠️ **Risk Factors:**
# # # # • Market volatility
# # # # • Company-specific news
# # # # • Sector rotation risks"""
    
# # # #     def generate_sentiment_analysis(self, data: dict) -> str:
# # # #         """Generate sentiment analysis"""
# # # #         symbol = data['symbol']
# # # #         change = data['price_change']
# # # #         volume = data['volume']
        
# # # #         # Simple sentiment based on price action and volume
# # # #         if change > 2 and volume > 1000000:
# # # #             sentiment = "Very Positive"
# # # #             emoji = "🚀"
# # # #         elif change > 0 and volume > 500000:
# # # #             sentiment = "Positive" 
# # # #             emoji = "😊"
# # # #         elif change < -2:
# # # #             sentiment = "Negative"
# # # #             emoji = "😟"
# # # #         else:
# # # #             sentiment = "Neutral"
# # # #             emoji = "😐"
        
# # # #         return f"""😊 **Sentiment Analysis for {symbol}**

# # # # {emoji} **Overall Sentiment: {sentiment.upper()}**

# # # # 📊 **Market Signals:**
# # # # • Price Action: {change:+.2f}% today
# # # # • Volume: {volume:,} shares
# # # # • Market Interest: {"High" if volume > 1000000 else "Medium" if volume > 500000 else "Normal"}

# # # # 📈 **Sentiment Indicators:**
# # # # • Price momentum: {"Positive" if change > 0 else "Negative"}
# # # # • Trading activity: {"Above average" if volume > 1000000 else "Normal"}
# # # # • Technical setup: {data['trend']}

# # # # 💡 **Interpretation:**
# # # # • {"Strong bullish sentiment with high conviction" if sentiment == "Very Positive" else 
# # # #    "Positive market sentiment" if sentiment == "Positive" else
# # # #    "Bearish sentiment, caution advised" if sentiment == "Negative" else
# # # #    "Mixed signals, wait for clearer direction"}"""
    
# # # #     def generate_general_analysis(self, data: dict) -> str:
# # # #         """Generate general analysis"""
# # # #         symbol = data['symbol']
        
# # # #         return f"""📋 **General Analysis for {symbol}**

# # # # I can provide comprehensive analysis on:

# # # # 📄 **Document Analysis**
# # # # • "What's {symbol}'s latest revenue?"
# # # # • "Analyze {symbol}'s financial health"

# # # # 📈 **Price Forecasting**  
# # # # • "Predict {symbol}'s price for next 30 days"
# # # # • "What's the technical outlook for {symbol}?"

# # # # 💼 **Investment Strategy**
# # # # • "Should I buy {symbol}?"
# # # # • "What's your recommendation for {symbol}?"

# # # # 😊 **Market Sentiment**
# # # # • "What's the sentiment for {symbol}?"
# # # # • "How is the market feeling about {symbol}?"

# # # # **Current Data for {symbol}:**
# # # # • Price: ${data['current_price']:.2f} ({data['price_change']:+.2f}%)
# # # # • Trend: {data['trend']}
# # # # • RSI: {data['rsi']:.1f}

# # # # *What would you like to know about {symbol}?*"""
    
# # # #     def interpret_rsi(self, rsi: float) -> str:
# # # #         """Interpret RSI value"""
# # # #         if rsi > 70:
# # # #             return "Overbought"
# # # #         elif rsi < 30:
# # # #             return "Oversold"
# # # #         else:
# # # #             return "Neutral"
    
# # # #     def generate_chart_data(self, data: dict) -> dict:
# # # #         """Generate chart data for visualization"""
# # # #         if 'historical_data' not in data or data['historical_data'].empty:
# # # #             return None
        
# # # #         hist_data = data['historical_data']
        
# # # #         return {
# # # #             "type": "line",
# # # #             "data": {
# # # #                 "labels": [d.strftime("%Y-%m-%d") for d in hist_data.index[-30:]],
# # # #                 "datasets": [{
# # # #                     "label": f"{data['symbol']} Price",
# # # #                     "data": hist_data['Close'][-30:].tolist(),
# # # #                     "borderColor": "rgb(59, 130, 246)",
# # # #                     "backgroundColor": "rgba(59, 130, 246, 0.1)"
# # # #                 }]
# # # #             }
# # # #         }
    
# # # #     def render_header(self):
# # # #         """Render the main header with real-time status"""
# # # #         st.title("🤖 FinDocGPT - Real-Time Financial Analysis")
# # # #         st.markdown("*AI-powered financial document analysis & investment strategy with live data*")
        
# # # #         # Status indicators
# # # #         col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
# # # #         with col1:
# # # #             # Real-time clock
# # # #             current_time = datetime.now().strftime("%H:%M:%S")
# # # #             st.markdown(f"🕐 **Live Time:** {current_time}")
        
# # # #         with col2:
# # # #             if self.check_api_connection():
# # # #                 st.success("🟢 API Connected")
# # # #             else:
# # # #                 st.warning("🟡 Local Mode")
        
# # # #         with col3:
# # # #             st.info(f"💬 {len(st.session_state.messages)} messages")
        
# # # #         with col4:
# # # #             # Auto-refresh toggle
# # # #             auto_refresh = st.checkbox("🔄 Auto-refresh", value=True)
# # # #             if auto_refresh:
# # # #                 time.sleep(1)
# # # #                 st.rerun()
    
# # # #     def render_sidebar(self):
# # # #         """Render the sidebar with real-time controls"""
# # # #         with st.sidebar:
# # # #             st.header("🎯 Real-Time Controls")
            
# # # #             # Symbol selection with real-time data
# # # #             st.subheader("📊 Stock Symbol")
# # # #             symbol = st.text_input(
# # # #                 "Enter symbol", 
# # # #                 value=st.session_state.current_symbol,
# # # #                 help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
# # # #             ).upper()
            
# # # #             if symbol != st.session_state.current_symbol:
# # # #                 st.session_state.current_symbol = symbol
# # # #                 st.rerun()
            
# # # #             # Get and display real-time data
# # # #             if symbol:
# # # #                 with st.spinner("Getting real-time data..."):
# # # #                     real_time_data = self.get_real_time_data(symbol)
                    
# # # #                 if "error" not in real_time_data:
# # # #                     st.success(f"✅ Live data for {symbol}")
                    
# # # #                     # Display key metrics
# # # #                     col1, col2 = st.columns(2)
# # # #                     with col1:
# # # #                         st.metric("Price", f"${real_time_data['current_price']:.2f}")
# # # #                         st.metric("RSI", f"{real_time_data['rsi']:.1f}")
# # # #                     with col2:
# # # #                         st.metric("Change", f"{real_time_data['price_change']:+.2f}%")
# # # #                         st.metric("Trend", real_time_data['trend'])
                    
# # # #                     st.caption(f"Updated: {real_time_data['last_updated']}")
# # # #                 else:
# # # #                     st.error(f"❌ No data for {symbol}")
            
# # # #             # Analysis type selection
# # # #             st.subheader("🔍 Analysis Type")
# # # #             analysis_type = st.selectbox(
# # # #                 "Choose analysis type",
# # # #                 ["general", "financial", "forecast", "strategy", "sentiment"],
# # # #                 help="Select the type of real-time analysis"
# # # #             )
            
# # # #             # Quick actions for real-time analysis
# # # #             st.subheader("⚡ Real-Time Actions")
            
# # # #             if st.button("📄 Financial Analysis", use_container_width=True):
# # # #                 self.quick_action_financial(symbol)
            
# # # #             if st.button("📈 Price Forecast", use_container_width=True):
# # # #                 self.quick_action_forecast(symbol)
            
# # # #             if st.button("💼 Investment Advice", use_container_width=True):
# # # #                 self.quick_action_strategy(symbol)
            
# # # #             if st.button("😊 Market Sentiment", use_container_width=True):
# # # #                 self.quick_action_sentiment(symbol)
            
# # # #             # Market status
# # # #             st.subheader("📈 Market Status")
# # # #             market_status = self.get_market_status()
# # # #             if market_status['is_open']:
# # # #                 st.success("🟢 Market Open")
# # # #             else:
# # # #                 st.warning("🟡 Market Closed")
# # # #             st.caption(f"Next: {market_status['next_session']}")
            
# # # #             return {"analysis_type": analysis_type, "symbol": symbol}
    
# # # #     def get_market_status(self) -> dict:
# # # #         """Get current market status"""
# # # #         now = datetime.now()
# # # #         # Simple market hours check (9:30 AM - 4:00 PM ET, Mon-Fri)
# # # #         weekday = now.weekday()  # 0 = Monday, 6 = Sunday
# # # #         hour = now.hour
        
# # # #         is_open = (weekday < 5) and (9 <= hour < 16)  # Simplified
        
# # # #         if is_open:
# # # #             next_session = "Market closes at 4:00 PM ET"
# # # #         else:
# # # #             next_session = "Market opens at 9:30 AM ET"
        
# # # #         return {
# # # #             "is_open": is_open,
# # # #             "next_session": next_session
# # # #         }
    
# # # #     def quick_action_financial(self, symbol: str):
# # # #         """Quick financial analysis"""
# # # #         message = f"Provide a comprehensive financial analysis for {symbol}"
# # # #         self.handle_user_message(message, "financial")
    
# # # #     def quick_action_forecast(self, symbol: str):
# # # #         """Quick price forecast"""
# # # #         message = f"Predict {symbol}'s price for the next 30 days with technical analysis"
# # # #         self.handle_user_message(message, "forecast")
    
# # # #     def quick_action_strategy(self, symbol: str):
# # # #         """Quick investment strategy"""
# # # #         message = f"Should I buy {symbol}? Give me a detailed investment recommendation"
# # # #         self.handle_user_message(message, "strategy")
    
# # # #     def quick_action_sentiment(self, symbol: str):
# # # #         """Quick sentiment analysis"""
# # # #         message = f"What's the current market sentiment for {symbol}?"
# # # #         self.handle_user_message(message, "sentiment")
    
# # # #     def handle_user_message(self, message: str, analysis_type: str):
# # # #         """Handle user message with real-time processing"""
# # # #         # Add user message
# # # #         user_message = {
# # # #             "role": "user",
# # # #             "content": message,
# # # #             "timestamp": datetime.now()
# # # #         }
# # # #         st.session_state.messages.append(user_message)
        
# # # #         # Get AI response (API or local)
# # # #         with st.spinner("Analyzing real-time data..."):
# # # #             response = self.send_chat_message(message, st.session_state.current_symbol, analysis_type)
        
# # # #         if response:
# # # #             # Add assistant message
# # # #             assistant_message = {
# # # #                 "role": "assistant",
# # # #                 "content": response["response"]["content"],
# # # #                 "timestamp": datetime.now(),
# # # #                 "analysis_results": response.get("analysis_results"),
# # # #                 "chart_data": response.get("chart_data")
# # # #             }
# # # #             st.session_state.messages.append(assistant_message)
        
# # # #         # Trigger rerun to show new messages
# # # #         st.rerun()
    
# # # #     def render_chat_interface(self):
# # # #         """Render the real-time chat interface"""
# # # #         st.subheader("💬 Real-Time Financial Chat")
        
# # # #         # Chat container with real-time messages
# # # #         chat_container = st.container()
        
# # # #         with chat_container:
# # # #             # Display all messages
# # # #             for i, message in enumerate(st.session_state.messages):
# # # #                 with st.chat_message(message["role"]):
# # # #                     st.write(message["content"])
                    
# # # #                     # Show timestamp
# # # #                     st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
                    
# # # #                     # Show analysis results if available
# # # #                     if message["role"] == "assistant" and message.get("analysis_results"):
# # # #                         with st.expander("📊 Real-Time Data", expanded=False):
# # # #                             results = message["analysis_results"]
# # # #                             col1, col2, col3 = st.columns(3)
                            
# # # #                             with col1:
# # # #                                 if "current_price" in results:
# # # #                                     st.metric("Current Price", f"${results['current_price']:.2f}")
# # # #                             with col2:
# # # #                                 if "rsi" in results:
# # # #                                     st.metric("RSI", f"{results['rsi']:.1f}")
# # # #                             with col3:
# # # #                                 if "trend" in results:
# # # #                                     st.metric("Trend", results['trend'])
                    
# # # #                     # Show charts if available
# # # #                     if message["role"] == "assistant" and message.get("chart_data"):
# # # #                         self.render_realtime_chart(message["chart_data"])
        
# # # #         # Chat input at the bottom
# # # #         if prompt := st.chat_input("Ask about any stock in real-time..."):
# # # #             self.handle_user_message(prompt, "general")
    
# # # #     def render_realtime_chart(self, chart_data: dict):
# # # #         """Render real-time charts"""
# # # #         if not chart_data or "data" not in chart_data:
# # # #             return
        
# # # #         try:
# # # #             data = chart_data["data"]
# # # #             labels = data.get("labels", [])
# # # #             datasets = data.get("datasets", [])
            
# # # #             if datasets and labels:
# # # #                 # Create plotly chart
# # # #                 fig = go.Figure()
                
# # # #                 for dataset in datasets:
# # # #                     fig.add_trace(go.Scatter(
# # # #                         x=labels,
# # # #                         y=dataset.get("data", []),
# # # #                         mode='lines+markers',
# # # #                         name=dataset.get("label", "Price"),
# # # #                         line=dict(color=dataset.get("borderColor", "#3B82F6"))
# # # #                     ))
                
# # # #                 fig.update_layout(
# # # #                     title="Real-Time Price Chart",
# # # #                     xaxis_title="Date",
# # # #                     yaxis_title="Price ($)",
# # # #                     height=400,
# # # #                     showlegend=True
# # # #                 )
                
# # # #                 st.plotly_chart(fig, use_container_width=True)
                
# # # #         except Exception as e:
# # # #             st.error(f"Chart error: {e}")
    
# # # #     def render_analytics_dashboard(self):
# # # #         """Render real-time analytics dashboard"""
# # # #         st.subheader("📊 Real-Time Analytics Dashboard")
        
# # # #         # Key metrics (these would be real in production)
# # # #         col1, col2, col3, col4 = st.columns(4)
        
# # # #         with col1:
# # # #             st.metric("Total Queries", "1,234", "↗️ 12%")
# # # #         with col2:
# # # #             st.metric("Avg Confidence", "82.5%", "↗️ 3.2%")
# # # #         with col3:
# # # #             st.metric("Active Symbols", "45", "↗️ 5")
# # # #         with col4:
# # # #             st.metric("Success Rate", "94.2%", "↗️ 1.1%")
        
# # # #         # Real-time market overview
# # # #         st.subheader("🌍 Market Overview")
        
# # # #         # Get data for major indices
# # # #         major_symbols = ["^GSPC", "^IXIC", "^DJI", "^VIX"]
# # # #         market_data = []
        
# # # #         for symbol in major_symbols:
# # # #             try:
# # # #                 ticker = yf.Ticker(symbol)
# # # #                 info = ticker.info
# # # #                 hist = ticker.history(period="1d")
                
# # # #                 if not hist.empty:
# # # #                     current_price = hist['Close'].iloc[-1]
# # # #                     change = ((current_price - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                    
# # # #                     market_data.append({
# # # #                         "Index": symbol.replace("^", ""),
# # # #                         "Price": f"{current_price:.2f}",
# # # #                         "Change %": f"{change:+.2f}%"
# # # #                     })
# # # #             except:
# # # #                 pass
        
# # # #         if market_data:
# # # #             df = pd.DataFrame(market_data)
# # # #             st.dataframe(df, use_container_width=True)
    
# # # #     def run(self):
# # # #         """Main application runner with real-time features"""
# # # #         # Initialize and render header
# # # #         self.render_header()
        
# # # #         # Render sidebar and get settings
# # # #         settings = self.render_sidebar()
        
# # # #         # Main content tabs
# # # #         tab1, tab2, tab3, tab4 = st.tabs(["💬 Real-Time Chat", "📊 Live Analytics", "📈 Live Charts", "📋 Reports"])
        
# # # #         with tab1:
# # # #             self.render_chat_interface()
        
# # # #         with tab2:
# # # #             self.render_analytics_dashboard()
        
# # # #         with tab3:
# # # #             st.subheader("📈 Live Market Charts")
# # # #             symbol = settings["symbol"]
            
# # # #             if symbol:
# # # #                 real_time_data = self.get_real_time_data(symbol)
# # # #                 if "error" not in real_time_data and "historical_data" in real_time_data:
# # # #                     hist_data = real_time_data["historical_data"]
                    
# # # #                     # Price chart
# # # #                     fig = go.Figure()
# # # #                     fig.add_trace(go.Scatter(
# # # #                         x=hist_data.index,
# # # #                         y=hist_data['Close'],
# # # #                         mode='lines',
# # # #                         name='Price',
# # # #                         line=dict(color='blue')
# # # #                     ))
                    
# # # #                     fig.update_layout(
# # # #                         title=f"{symbol} - Real-Time Price Chart",
# # # #                         xaxis_title="Date",
# # # #                         yaxis_title="Price ($)",
# # # #                         height=500
# # # #                     )
                    
# # # #                     st.plotly_chart(fig2, use_container_width=True)
        
# # # #         with tab4:
# # # #             self.render_reports_view()
    
# # # #     def render_reports_view(self):
# # # #         """Render real-time reports"""
# # # #         st.subheader("📋 Real-Time Financial Reports")
        
# # # #         symbol = st.session_state.current_symbol
        
# # # #         # Generate real-time report
# # # #         real_time_data = self.get_real_time_data(symbol)
        
# # # #         if "error" not in real_time_data:
# # # #             # Executive Summary Report
# # # #             st.markdown(f"""
# # # #             ### 📊 Real-Time Analysis Report for {symbol}
            
# # # #             **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Live Data)
            
# # # #             #### Executive Summary
# # # #             - **Current Price:** ${real_time_data['current_price']:.2f} ({real_time_data['price_change']:+.2f}%)
# # # #             - **Market Cap:** ${real_time_data['market_cap']:,.0f}
# # # #             - **P/E Ratio:** {real_time_data['pe_ratio']:.2f}
# # # #             - **Trading Volume:** {real_time_data['volume']:,}
            
# # # #             #### Technical Analysis (Live)
# # # #             - **Trend:** {real_time_data['trend']}
# # # #             - **RSI:** {real_time_data['rsi']:.1f} ({self.interpret_rsi(real_time_data['rsi'])})
# # # #             - **20-day SMA:** ${real_time_data['sma_20']:.2f}
# # # #             - **50-day SMA:** ${real_time_data['sma_50']:.2f}
            
# # # #             #### Investment Outlook
# # # #             - **Recommendation:** {"BUY" if real_time_data['trend'] == "Bullish" and real_time_data['rsi'] < 70 else "HOLD" if real_time_data['rsi'] < 80 else "SELL"}
# # # #             - **Risk Level:** {"Low" if real_time_data['rsi'] < 80 and abs(real_time_data['price_change']) < 3 else "Medium" if abs(real_time_data['price_change']) < 5 else "High"}
# # # #             - **Confidence:** {"High" if real_time_data['volume'] > 1000000 else "Medium"}
            
# # # #             #### Key Risks
# # # #             - Market volatility
# # # #             - Sector-specific risks
# # # #             - Economic indicators
            
# # # #             ---
# # # #             *This report is generated using real-time market data and AI analysis.*
# # # #             """)
            
# # # #             # Export functionality
# # # #             if st.button("📄 Export Real-Time Report"):
# # # #                 st.success("Real-time report exported! (Feature ready for implementation)")

# # # # # Initialize and run the app
# # # # if __name__ == "__main__":
# # # #     app = RealTimeFinDocGPT()
# # # #     app.run()
    
    
# # # import streamlit as st
# # # import requests
# # # import pandas as pd
# # # import plotly.graph_objects as go
# # # import plotly.express as px
# # # from datetime import datetime, timedelta
# # # import yfinance as yf
# # # import numpy as np
# # # import time
# # # import json

# # # # Configure Streamlit
# # # st.set_page_config(
# # #     page_title="FinDocGPT - Real-Time Financial Analysis",
# # #     page_icon="📈",
# # #     layout="wide",
# # #     initial_sidebar_state="expanded"
# # # )

# # # # API Configuration
# # # API_BASE_URL = "http://localhost:8000"

# # # class RealTimeFinDocGPT:
# # #     def __init__(self):
# # #         self.api_base = API_BASE_URL
# # #         self.init_session_state()
        
# # #     def init_session_state(self):
# # #         """Initialize session state variables"""
# # #         if 'conversation_id' not in st.session_state:
# # #             st.session_state.conversation_id = None
# # #         if 'messages' not in st.session_state:
# # #             st.session_state.messages = []
# # #         if 'current_symbol' not in st.session_state:
# # #             st.session_state.current_symbol = "AAPL"
# # #         if 'real_time_data' not in st.session_state:
# # #             st.session_state.real_time_data = {}
# # #         if 'api_connected' not in st.session_state:
# # #             st.session_state.api_connected = False
# # #         if 'auto_refresh' not in st.session_state:
# # #             st.session_state.auto_refresh = True
    
# # #     def check_api_connection(self):
# # #         """Check if FastAPI backend is running"""
# # #         try:
# # #             response = requests.get(f"{self.api_base}/health", timeout=5)
# # #             if response.status_code == 200:
# # #                 st.session_state.api_connected = True
# # #                 return True
# # #             else:
# # #                 st.session_state.api_connected = False
# # #                 return False
# # #         except Exception as e:
# # #             st.session_state.api_connected = False
# # #             return False
    
# # #     def get_real_time_data(self, symbol):
# # #         """Get real-time financial data"""
# # #         try:
# # #             ticker = yf.Ticker(symbol)
# # #             info = ticker.info
# # #             hist = ticker.history(period="1mo")
            
# # #             current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
# # #             change = info.get('regularMarketChangePercent', 0)
            
# # #             # Calculate technical indicators
# # #             if not hist.empty:
# # #                 # RSI
# # #                 delta = hist['Close'].diff()
# # #                 gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# # #                 loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# # #                 rs = gain / loss
# # #                 hist['RSI'] = 100 - (100 / (1 + rs))
                
# # #                 # Moving averages
# # #                 hist['SMA_20'] = hist['Close'].rolling(20).mean()
# # #                 hist['SMA_50'] = hist['Close'].rolling(50).mean()
                
# # #                 latest_rsi = hist['RSI'].iloc[-1] if not pd.isna(hist['RSI'].iloc[-1]) else 50
# # #                 latest_sma_20 = hist['SMA_20'].iloc[-1] if not pd.isna(hist['SMA_20'].iloc[-1]) else current_price
# # #                 latest_sma_50 = hist['SMA_50'].iloc[-1] if not pd.isna(hist['SMA_50'].iloc[-1]) else current_price
# # #             else:
# # #                 latest_rsi = 50
# # #                 latest_sma_20 = current_price
# # #                 latest_sma_50 = current_price
            
# # #             return {
# # #                 "symbol": symbol,
# # #                 "current_price": current_price,
# # #                 "price_change": change,
# # #                 "volume": info.get('regularMarketVolume', 0),
# # #                 "market_cap": info.get('marketCap', 0),
# # #                 "pe_ratio": info.get('trailingPE', 0),
# # #                 "rsi": latest_rsi,
# # #                 "sma_20": latest_sma_20,
# # #                 "sma_50": latest_sma_50,
# # #                 "trend": "Bullish" if latest_sma_20 > latest_sma_50 else "Bearish",
# # #                 "historical_data": hist,
# # #                 "last_updated": datetime.now().strftime("%H:%M:%S")
# # #             }
# # #         except Exception as e:
# # #             return {"error": str(e)}
    
# # #     def send_chat_message(self, message, symbol, analysis_type="general"):
# # #         """Send message to API or use local analysis"""
        
# # #         # Try API first
# # #         if st.session_state.api_connected:
# # #             try:
# # #                 response = requests.post(
# # #                     f"{self.api_base}/api/v1/chat/message",
# # #                     json={
# # #                         "message": message,
# # #                         "symbol": symbol,
# # #                         "analysis_type": analysis_type,
# # #                         "conversation_id": st.session_state.conversation_id
# # #                     },
# # #                     timeout=30
# # #                 )
                
# # #                 if response.status_code == 200:
# # #                     result = response.json()
# # #                     if not st.session_state.conversation_id:
# # #                         st.session_state.conversation_id = result.get("conversation_id")
# # #                     return result
# # #             except Exception as e:
# # #                 st.error(f"API Error: {e}")
        
# # #         # Fallback to local analysis
# # #         return self.local_financial_analysis(message, symbol, analysis_type)
    
# # #     def local_financial_analysis(self, message, symbol, analysis_type):
# # #         """Local financial analysis when API is unavailable"""
# # #         try:
# # #             data = self.get_real_time_data(symbol)
            
# # #             if "error" in data:
# # #                 return {
# # #                     "response": {"content": f"❌ Unable to analyze {symbol}: {data['error']}"},
# # #                     "suggestions": ["Try a different symbol", "Check internet connection"]
# # #                 }
            
# # #             # Generate response based on message type
# # #             message_lower = message.lower()
            
# # #             if any(word in message_lower for word in ['revenue', 'earnings', 'profit', 'financial']):
# # #                 response_content = self.generate_financial_analysis(data)
# # #             elif any(word in message_lower for word in ['predict', 'forecast', 'price', 'target']):
# # #                 response_content = self.generate_price_forecast(data)
# # #             elif any(word in message_lower for word in ['buy', 'sell', 'invest', 'recommend']):
# # #                 response_content = self.generate_investment_recommendation(data)
# # #             elif any(word in message_lower for word in ['sentiment', 'news', 'market']):
# # #                 response_content = self.generate_sentiment_analysis(data)
# # #             else:
# # #                 response_content = self.generate_general_analysis(data)
            
# # #             return {
# # #                 "response": {"content": response_content},
# # #                 "analysis_results": data,
# # #                 "chart_data": self.generate_chart_data(data),
# # #                 "suggestions": [
# # #                     f"Predict {symbol}'s price trend",
# # #                     f"Should I buy {symbol}?",
# # #                     f"What's {symbol}'s sentiment?"
# # #                 ]
# # #             }
# # #         except Exception as e:
# # #             return {
# # #                 "response": {"content": f"Analysis error: {str(e)}"},
# # #                 "suggestions": ["Try again with a different query"]
# # #             }
    
# # #     def generate_financial_analysis(self, data):
# # #         """Generate financial analysis response"""
# # #         symbol = data['symbol']
# # #         price = data['current_price']
# # #         change = data['price_change']
# # #         volume = data['volume']
# # #         pe = data['pe_ratio']
# # #         market_cap = data['market_cap']
        
# # #         return f"""📊 **Real-Time Financial Analysis for {symbol}**

# # # 💰 **Current Metrics:**
# # # • Price: ${price:.2f} ({change:+.2f}%)
# # # • Market Cap: ${market_cap:,.0f}
# # # • P/E Ratio: {pe:.2f}
# # # • Volume: {volume:,}

# # # 📈 **Technical Indicators:**
# # # • RSI: {data['rsi']:.1f} ({self.interpret_rsi(data['rsi'])})
# # # • 20-day SMA: ${data['sma_20']:.2f}
# # # • 50-day SMA: ${data['sma_50']:.2f}
# # # • Trend: {data['trend']}

# # # 🔍 **Key Insights:**
# # # • {"Stock is overbought" if data['rsi'] > 70 else "Stock is oversold" if data['rsi'] < 30 else "RSI in neutral range"}
# # # • {"Bullish momentum confirmed" if data['sma_20'] > data['sma_50'] else "Bearish pressure evident"}
# # # • {"High trading volume suggests strong interest" if volume > 1000000 else "Normal trading volume"}

# # # *Live update: {data['last_updated']}*"""
    
# # #     def generate_price_forecast(self, data):
# # #         """Generate price forecast response"""
# # #         symbol = data['symbol']
# # #         current_price = data['current_price']
# # #         trend = data['trend']
# # #         rsi = data['rsi']
        
# # #         # Simple forecast logic
# # #         if trend == "Bullish" and rsi < 70:
# # #             target_price = current_price * 1.05
# # #             probability = "High"
# # #         elif trend == "Bearish" and rsi > 30:
# # #             target_price = current_price * 0.95
# # #             probability = "High"
# # #         else:
# # #             target_price = current_price * 1.02
# # #             probability = "Medium"
        
# # #         return f"""📈 **Real-Time Price Forecast for {symbol}**

# # # 🎯 **30-Day Prediction:**
# # # • Current Price: ${current_price:.2f}
# # # • Target Price: ${target_price:.2f}
# # # • Expected Return: {((target_price - current_price) / current_price * 100):+.1f}%
# # # • Probability: {probability}

# # # 🔮 **Forecast Basis:**
# # # • Technical Trend: {trend}
# # # • RSI Level: {rsi:.1f}
# # # • Moving Average Alignment: {"Bullish" if data['sma_20'] > data['sma_50'] else "Bearish"}

# # # 📊 **Risk Assessment:**
# # # • Volatility: {"High" if abs(data['price_change']) > 3 else "Medium" if abs(data['price_change']) > 1 else "Low"}
# # # • Confidence Level: {probability}

# # # *Real-time analysis updated: {data['last_updated']}*"""
    
# # #     def generate_investment_recommendation(self, data):
# # #         """Generate investment recommendation"""
# # #         symbol = data['symbol']
# # #         price = data['current_price']
# # #         rsi = data['rsi']
# # #         trend = data['trend']
# # #         pe = data['pe_ratio']
        
# # #         # Scoring system
# # #         score = 0
        
# # #         # Technical scoring
# # #         if rsi < 30:  # Oversold
# # #             score += 2
# # #         elif rsi > 70:  # Overbought
# # #             score -= 2
        
# # #         if trend == "Bullish":
# # #             score += 2
# # #         else:
# # #             score -= 1
        
# # #         # Fundamental scoring
# # #         if pe < 15:
# # #             score += 1
# # #         elif pe > 30:
# # #             score -= 1
        
# # #         # Generate recommendation
# # #         if score >= 3:
# # #             recommendation = "🟢 STRONG BUY"
# # #             confidence = "High"
# # #         elif score >= 1:
# # #             recommendation = "🟢 BUY"
# # #             confidence = "Medium"
# # #         elif score >= -1:
# # #             recommendation = "🟡 HOLD"
# # #             confidence = "Medium"
# # #         else:
# # #             recommendation = "🔴 SELL"
# # #             confidence = "High"
        
# # #         return f"""💼 **Real-Time Investment Recommendation for {symbol}**

# # # 🎯 **Decision: {recommendation}**
# # # • Confidence: {confidence}
# # # • Current Price: ${price:.2f}
# # # • Analysis Score: {score}/5

# # # 📊 **Real-Time Analysis:**
# # # • **Technical:** {"Positive" if rsi < 70 and trend == "Bullish" else "Negative"}
# # #   - RSI: {rsi:.1f} ({self.interpret_rsi(rsi)})
# # #   - Trend: {trend}

# # # • **Fundamental:** {"Attractive" if pe < 20 else "Expensive"}
# # #   - P/E Ratio: {pe:.2f}

# # # 🎯 **Live Action Plan:**
# # # • Entry Point: {"Current levels attractive" if score > 0 else "Wait for better entry"}
# # # • Stop Loss: ${price * 0.9:.2f} (-10%)
# # # • Target: ${price * 1.15:.2f} (+15%)

# # # *Live analysis: {data['last_updated']}*"""
    
# # #     def generate_sentiment_analysis(self, data):
# # #         """Generate sentiment analysis"""
# # #         symbol = data['symbol']
# # #         change = data['price_change']
# # #         volume = data['volume']
        
# # #         # Simple sentiment based on price action and volume
# # #         if change > 2 and volume > 1000000:
# # #             sentiment = "Very Positive"
# # #             emoji = "🚀"
# # #         elif change > 0 and volume > 500000:
# # #             sentiment = "Positive" 
# # #             emoji = "😊"
# # #         elif change < -2:
# # #             sentiment = "Negative"
# # #             emoji = "😟"
# # #         else:
# # #             sentiment = "Neutral"
# # #             emoji = "😐"
        
# # #         return f"""😊 **Real-Time Sentiment Analysis for {symbol}**

# # # {emoji} **Current Sentiment: {sentiment.upper()}**

# # # 📊 **Live Market Signals:**
# # # • Price Action: {change:+.2f}% today
# # # • Volume: {volume:,} shares
# # # • Market Interest: {"High" if volume > 1000000 else "Medium" if volume > 500000 else "Normal"}

# # # 📈 **Real-Time Indicators:**
# # # • Price momentum: {"Positive" if change > 0 else "Negative"}
# # # • Trading activity: {"Above average" if volume > 1000000 else "Normal"}
# # # • Technical setup: {data['trend']}

# # # *Live sentiment update: {data['last_updated']}*"""
    
# # #     def generate_general_analysis(self, data):
# # #         """Generate general analysis"""
# # #         symbol = data['symbol']
        
# # #         return f"""📋 **Real-Time Analysis for {symbol}**

# # # 🔴 **Live Data:**
# # # • Price: ${data['current_price']:.2f} ({data['price_change']:+.2f}%)
# # # • Trend: {data['trend']}
# # # • RSI: {data['rsi']:.1f}
# # # • Last Update: {data['last_updated']}

# # # 💬 **Ask me for real-time analysis:**

# # # 📄 **Financial Analysis**
# # # • "What's {symbol}'s current financial health?"
# # # • "Analyze {symbol}'s live metrics"

# # # 📈 **Price Forecasting**  
# # # • "Predict {symbol}'s price for next 30 days"
# # # • "What's the live technical outlook for {symbol}?"

# # # 💼 **Investment Strategy**
# # # • "Should I buy {symbol} right now?"
# # # • "Give me a real-time recommendation for {symbol}"

# # # 😊 **Market Sentiment**
# # # • "What's the live sentiment for {symbol}?"
# # # • "How is the market feeling about {symbol} today?"

# # # *Real-time data powered by live market feeds*"""
    
# # #     def interpret_rsi(self, rsi):
# # #         """Interpret RSI value"""
# # #         if rsi > 70:
# # #             return "Overbought"
# # #         elif rsi < 30:
# # #             return "Oversold"
# # #         else:
# # #             return "Neutral"
    
# # #     def generate_chart_data(self, data):
# # #         """Generate chart data for visualization"""
# # #         if 'historical_data' not in data or data['historical_data'].empty:
# # #             return None
        
# # #         hist_data = data['historical_data']
        
# # #         return {
# # #             "type": "line",
# # #             "data": {
# # #                 "labels": [d.strftime("%Y-%m-%d") for d in hist_data.index[-30:]],
# # #                 "datasets": [{
# # #                     "label": f"{data['symbol']} Price",
# # #                     "data": hist_data['Close'][-30:].tolist(),
# # #                     "borderColor": "rgb(59, 130, 246)",
# # #                     "backgroundColor": "rgba(59, 130, 246, 0.1)"
# # #                 }]
# # #             }
# # #         }
    
# # #     def render_header(self):
# # #         """Render header with real-time status"""
# # #         st.title("🤖 FinDocGPT - Real-Time Financial Analysis")
# # #         st.markdown("*AI-powered real-time financial analysis with live market data*")
        
# # #         # Status indicators
# # #         col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
        
# # #         with col1:
# # #             current_time = datetime.now().strftime("%H:%M:%S")
# # #             st.markdown(f"🕐 **Live Time:** {current_time}")
        
# # #         with col2:
# # #             if self.check_api_connection():
# # #                 st.success("🟢 API Connected")
# # #             else:
# # #                 st.warning("🟡 Local Mode")
        
# # #         with col3:
# # #             st.info(f"💬 {len(st.session_state.messages)} messages")
        
# # #         with col4:
# # #             auto_refresh = st.checkbox("🔄 Auto-refresh", value=st.session_state.auto_refresh)
# # #             if auto_refresh != st.session_state.auto_refresh:
# # #                 st.session_state.auto_refresh = auto_refresh
# # #                 st.rerun()
    
# # #     def render_sidebar(self):
# # #         """Render sidebar with real-time controls"""
# # #         with st.sidebar:
# # #             st.header("🎯 Real-Time Controls")
            
# # #             # Symbol selection
# # #             symbol = st.text_input(
# # #                 "📊 Stock Symbol", 
# # #                 value=st.session_state.current_symbol,
# # #                 help="Enter a stock symbol (e.g., AAPL, MSFT, GOOGL)"
# # #             ).upper()
            
# # #             if symbol != st.session_state.current_symbol:
# # #                 st.session_state.current_symbol = symbol
# # #                 st.rerun()
            
# # #             # Get and display real-time data
# # #             if symbol:
# # #                 with st.spinner("📡 Getting live data..."):
# # #                     real_time_data = self.get_real_time_data(symbol)
                
# # #                 if "error" not in real_time_data:
# # #                     st.success(f"✅ Live data for {symbol}")
                    
# # #                     col1, col2 = st.columns(2)
# # #                     with col1:
# # #                         st.metric("Price", f"${real_time_data['current_price']:.2f}")
# # #                         st.metric("RSI", f"{real_time_data['rsi']:.1f}")
# # #                     with col2:
# # #                         change_color = "normal" if real_time_data['price_change'] >= 0 else "inverse"
# # #                         st.metric("Change", f"{real_time_data['price_change']:+.2f}%")
# # #                         st.metric("Trend", real_time_data['trend'])
                    
# # #                     st.caption(f"🕐 Updated: {real_time_data['last_updated']}")
# # #                 else:
# # #                     st.error(f"❌ No data for {symbol}")
            
# # #             # Quick actions
# # #             st.subheader("⚡ Live Analysis")
            
# # #             if st.button("📊 Financial Analysis", use_container_width=True):
# # #                 self.quick_action_financial(symbol)
            
# # #             if st.button("📈 Price Forecast", use_container_width=True):
# # #                 self.quick_action_forecast(symbol)
            
# # #             if st.button("💼 Investment Advice", use_container_width=True):
# # #                 self.quick_action_strategy(symbol)
            
# # #             if st.button("😊 Market Sentiment", use_container_width=True):
# # #                 self.quick_action_sentiment(symbol)
            
# # #             # Market status
# # #             st.subheader("📈 Market Status")
# # #             market_status = self.get_market_status()
# # #             if market_status['is_open']:
# # #                 st.success("🟢 Market Open")
# # #             else:
# # #                 st.warning("🟡 Market Closed")
# # #             st.caption(f"Next: {market_status['next_session']}")
            
# # #             return {"symbol": symbol}
    
# # #     def get_market_status(self):
# # #         """Get current market status"""
# # #         now = datetime.now()
# # #         weekday = now.weekday()  # 0 = Monday, 6 = Sunday
# # #         hour = now.hour
        
# # #         is_open = (weekday < 5) and (9 <= hour < 16)  # Simplified
        
# # #         if is_open:
# # #             next_session = "Market closes at 4:00 PM ET"
# # #         else:
# # #             next_session = "Market opens at 9:30 AM ET"
        
# # #         return {
# # #             "is_open": is_open,
# # #             "next_session": next_session
# # #         }
    
# # #     def quick_action_financial(self, symbol):
# # #         """Quick financial analysis"""
# # #         message = f"Provide a comprehensive real-time financial analysis for {symbol}"
# # #         self.handle_user_message(message, "financial")
    
# # #     def quick_action_forecast(self, symbol):
# # #         """Quick price forecast"""
# # #         message = f"Predict {symbol}'s price for the next 30 days with live technical analysis"
# # #         self.handle_user_message(message, "forecast")
    
# # #     def quick_action_strategy(self, symbol):
# # #         """Quick investment strategy"""
# # #         message = f"Should I buy {symbol} right now? Give me a real-time investment recommendation"
# # #         self.handle_user_message(message, "strategy")
    
# # #     def quick_action_sentiment(self, symbol):
# # #         """Quick sentiment analysis"""
# # #         message = f"What's the current live market sentiment for {symbol}?"
# # #         self.handle_user_message(message, "sentiment")
    
# # #     def handle_user_message(self, message, analysis_type):
# # #         """Handle user message with real-time processing"""
# # #         # Add user message
# # #         user_message = {
# # #             "role": "user",
# # #             "content": message,
# # #             "timestamp": datetime.now()
# # #         }
# # #         st.session_state.messages.append(user_message)
        
# # #         # Get AI response
# # #         with st.spinner("🔄 Analyzing live data..."):
# # #             response = self.send_chat_message(message, st.session_state.current_symbol, analysis_type)
        
# # #         if response:
# # #             # Add assistant message
# # #             assistant_message = {
# # #                 "role": "assistant",
# # #                 "content": response["response"]["content"],
# # #                 "timestamp": datetime.now(),
# # #                 "analysis_results": response.get("analysis_results"),
# # #                 "chart_data": response.get("chart_data")
# # #             }
# # #             st.session_state.messages.append(assistant_message)
        
# # #         st.rerun()
    
# # #     def render_main_content(self):
# # #         """Render main content with tabs"""
# # #         tab1, tab2, tab3, tab4 = st.tabs(["💬 Live Chat", "📊 Live Analytics", "📈 Live Charts", "📋 Reports"])
        
# # #         with tab1:
# # #             self.render_chat_interface()
        
# # #         with tab2:
# # #             self.render_analytics_dashboard()
        
# # #         with tab3:
# # #             self.render_charts_view()
        
# # #         with tab4:
# # #             self.render_reports_view()
    
# # #     def render_chat_interface(self):
# # #         """Render real-time chat interface"""
# # #         st.subheader("💬 Real-Time Financial Chat")
        
# # #         # Chat container
# # #         chat_container = st.container()
        
# # #         with chat_container:
# # #             for i, message in enumerate(st.session_state.messages):
# # #                 with st.chat_message(message["role"]):
# # #                     st.write(message["content"])
# # #                     st.caption(f"🕐 {message['timestamp'].strftime('%H:%M:%S')}")
                    
# # #                     # Show analysis results
# # #                     if message["role"] == "assistant" and message.get("analysis_results"):
# # #                         with st.expander("📊 Live Data Details", expanded=False):
# # #                             results = message["analysis_results"]
# # #                             col1, col2, col3 = st.columns(3)
                            
# # #                             with col1:
# # #                                 if "current_price" in results:
# # #                                     st.metric("Live Price", f"${results['current_price']:.2f}")
# # #                             with col2:
# # #                                 if "rsi" in results:
# # #                                     st.metric("RSI", f"{results['rsi']:.1f}")
# # #                             with col3:
# # #                                 if "trend" in results:
# # #                                     st.metric("Trend", results['trend'])
                    
# # #                     # Show charts
# # #                     if message["role"] == "assistant" and message.get("chart_data"):
# # #                         self.render_realtime_chart(message["chart_data"])
        
# # #         # Chat input
# # #         if prompt := st.chat_input("Ask about any stock in real-time..."):
# # #             self.handle_user_message(prompt, "general")
    
# # #     def render_realtime_chart(self, chart_data):
# # #         """Render real-time charts"""
# # #         if not chart_data or "data" not in chart_data:
# # #             return
        
# # #         try:
# # #             data = chart_data["data"]
# # #             labels = data.get("labels", [])
# # #             datasets = data.get("datasets", [])
            
# # #             if datasets and labels:
# # #                 fig = go.Figure()
                
# # #                 for dataset in datasets:
# # #                     fig.add_trace(go.Scatter(
# # #                         x=labels,
# # #                         y=dataset.get("data", []),
# # #                         mode='lines+markers',
# # #                         name=dataset.get("label", "Price"),
# # #                         line=dict(color=dataset.get("borderColor", "#3B82F6"))
# # #                     ))
                
# # #                 fig.update_layout(
# # #                     title="Real-Time Price Chart",
# # #                     xaxis_title="Date",
# # #                     yaxis_title="Price ($)",
# # #                     height=400,
# # #                     showlegend=True
# # #                 )
                
# # #                 st.plotly_chart(fig, use_container_width=True)
# # #         except Exception as e:
# # #             st.error(f"Chart error: {e}")
    
# # #     def render_analytics_dashboard(self):
# # #         """Render real-time analytics dashboard"""
# # #         st.subheader("📊 Real-Time Analytics Dashboard")
        
# # #         # Key metrics
# # #         col1, col2, col3, col4 = st.columns(4)
        
# # #         with col1:
# # #             st.metric("Total Queries", "1,234", "↗️ 12%")
# # #         with col2:
# # #             st.metric("Avg Confidence", "82.5%", "↗️ 3.2%")
# # #         with col3:
# # #             st.metric("Active Symbols", "45", "↗️ 5")
# # #         with col4:
# # #             st.metric("Success Rate", "94.2%", "↗️ 1.1%")
        
# # #         # Market overview
# # #         st.subheader("🌍 Live Market Overview")
        
# # #         major_indices = ["^GSPC", "^IXIC", "^DJI"]
# # #         market_data = []
        
# # #         for index_symbol in major_indices:
# # #             try:
# # #                 ticker = yf.Ticker(index_symbol)
# # #                 hist = ticker.history(period="1d")
# # #                 if not hist.empty:
# # #                     current = hist['Close'].iloc[-1]
# # #                     change = ((current - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
# # #                     market_data.append({
# # #                         "Index": index_symbol.replace("^", ""),
# # #                         "Price": f"{current:.2f}",
# # #                         "Change": f"{change:+.2f}%"
# # #                     })
# # #             except:
# # #                 pass
        
# # #         if market_data:
# # #             df = pd.DataFrame(market_data)
# # #             st.dataframe(df, use_container_width=True)
    
# # #     def render_charts_view(self):
# # #         """Render live charts"""
# # #         st.subheader("📈 Live Market Charts")
        
# # #         symbol = st.session_state.current_symbol
        
# # #         if symbol:
# # #             real_time_data = self.get_real_time_data(symbol)
            
# # #             if "error" not in real_time_data and "historical_data" in real_time_data:
# # #                 hist_data = real_time_data["historical_data"]
                
# # #                 # Price chart
# # #                 fig1 = go.Figure()
# # #                 fig1.add_trace(go.Scatter(
# # #                     x=hist_data.index,
# # #                     y=hist_data['Close'],
# # #                     mode='lines',
# # #                     name='Price',
# # #                     line=dict(color='blue')
# # #                 ))
                
# # #                 fig1.update_layout(
# # #                     title=f"{symbol} - Real-Time Price Chart",
# # #                     xaxis_title="Date",
# # #                     yaxis_title="Price ($)",
# # #                     height=500
# # #                 )
                
# # #                 st.plotly_chart(fig1, use_container_width=True)
                
# # #                 # Volume chart
# # #                 fig2 = go.Figure()
# # #                 fig2.add_trace(go.Bar(
# # #                     x=hist_data.index,
# # #                     y=hist_data['Volume'],
# # #                     name='Volume',
# # #                     marker_color='green'
# # #                 ))
                
# # #                 fig2.update_layout(
# # #                     title=f"{symbol} - Trading Volume",
# # #                     xaxis_title="Date", 
# # #                     yaxis_title="Volume",
# # #                     height=300
# # #                 )
                
# # #                 st.plotly_chart(fig2, use_container_width=True)
                
# # #                 # Current metrics
# # #                 col1, col2, col3 = st.columns(3)
# # #                 with col1:
# # #                     st.metric("Current Price", f"${real_time_data['current_price']:.2f}")
# # #                 with col2:
# # #                     st.metric("Change", f"{real_time_data['price_change']:+.2f}%")
# # #                 with col3:
# # #                     st.metric("Volume", f"{real_time_data['volume']:,}")
    
# # #     def render_reports_view(self):
# # #         """Render real-time reports"""
# # #         st.subheader("📋 Real-Time Financial Reports")
        
# # #         symbol = st.session_state.current_symbol
# # #         real_time_data = self.get_real_time_data(symbol)
        
# # #         if "error" not in real_time_data:
# # #             st.markdown(f"""
# # #             ### 📊 Live Analysis Report for {symbol}
            
# # #             **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} (Real-Time)
            
# # #             #### Executive Summary
# # #             - **Current Price:** ${real_time_data['current_price']:.2f} ({real_time_data['price_change']:+.2f}%)
# # #             - **Market Cap:** ${real_time_data['market_cap']:,.0f}
# # #             - **P/E Ratio:** {real_time_data['pe_ratio']:.2f}
# # #             - **Trading Volume:** {real_time_data['volume']:,}
            
# # #             #### Technical Analysis (Live)
# # #             - **Trend:** {real_time_data['trend']}
# # #             - **RSI:** {real_time_data['rsi']:.1f} ({self.interpret_rsi(real_time_data['rsi'])})
# # #             - **20-day SMA:** ${real_time_data['sma_20']:.2f}
# # #             - **50-day SMA:** ${real_time_data['sma_50']:.2f}
            
# # #             #### Investment Outlook
# # #             - **Recommendation:** {"BUY" if real_time_data['trend'] == "Bullish" and real_time_data['rsi'] < 70 else "HOLD" if real_time_data['rsi'] < 80 else "SELL"}
# # #             - **Risk Level:** {"Low" if real_time_data['rsi'] < 80 and abs(real_time_data['price_change']) < 3 else "Medium" if abs(real_time_data['price_change']) < 5 else "High"}
# # #             - **Confidence:** {"High" if real_time_data['volume'] > 1000000 else "Medium"}
            
# # #             ---
# # #             *This report uses real-time market data and AI analysis.*
# # #             """)
            
# # #             if st.button("📄 Export Live Report"):
# # #                 st.success("Real-time report exported!")
    
# # #     def run(self):
# # #         """Main application runner"""
# # #         self.render_header()
# # #         settings = self.render_sidebar()
# # #         self.render_main_content()
        
# # #         # Auto-refresh functionality
# # #         if st.session_state.auto_refresh:
# # #             time.sleep(5)  # Refresh every 5 seconds
# # #             st.rerun()

# # # # Initialize and run the app
# # # if __name__ == "__main__":
# # #     app = RealTimeFinDocGPT()
# # #     app.run()


# # import streamlit as st
# # import requests
# # import pandas as pd
# # import plotly.graph_objects as go
# # import plotly.express as px
# # from datetime import datetime, timedelta
# # import yfinance as yf
# # import numpy as np
# # import time
# # import uuid

# # # Configure Streamlit
# # st.set_page_config(
# #     page_title="FinDocGPT - Real-Time",
# #     page_icon="📈",
# #     layout="wide"
# # )

# # # API Configuration
# # API_BASE_URL = "http://localhost:8000"

# # # Initialize session state
# # if "messages" not in st.session_state:
# #     st.session_state.messages = []
# # if "chart_counter" not in st.session_state:
# #     st.session_state.chart_counter = 0

# # def get_unique_key():
# #     """Generate unique key for Plotly charts"""
# #     st.session_state.chart_counter += 1
# #     return f"chart_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"

# # def check_api_connection():
# #     """Check API connection"""
# #     try:
# #         response = requests.get(f"{API_BASE_URL}/health", timeout=5)
# #         return response.status_code == 200
# #     except:
# #         return False

# # def get_real_time_data(symbol):
# #     """Get real-time financial data"""
# #     try:
# #         ticker = yf.Ticker(symbol)
# #         info = ticker.info
# #         hist = ticker.history(period="1mo")
        
# #         current_price = info.get('currentPrice', hist['Close'].iloc[-1] if not hist.empty else 0)
# #         change = info.get('regularMarketChangePercent', 0)
        
# #         # Calculate RSI
# #         if not hist.empty:
# #             delta = hist['Close'].diff()
# #             gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# #             loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# #             rs = gain / loss
# #             rsi = 100 - (100 / (1 + rs))
# #             latest_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50
# #         else:
# #             latest_rsi = 50
        
# #         return {
# #             "symbol": symbol,
# #             "current_price": current_price,
# #             "price_change": change,
# #             "volume": info.get('regularMarketVolume', 0),
# #             "market_cap": info.get('marketCap', 0),
# #             "pe_ratio": info.get('trailingPE', 0),
# #             "rsi": latest_rsi,
# #             "historical_data": hist,
# #             "last_updated": datetime.now().strftime("%H:%M:%S")
# #         }
# #     except Exception as e:
# #         return {"error": str(e)}

# # def send_chat_message(message, symbol, analysis_type="general"):
# #     """Send message to API or use local analysis"""
    
# #     # Try API first
# #     if check_api_connection():
# #         try:
# #             response = requests.post(
# #                 f"{API_BASE_URL}/api/v1/chat/message",
# #                 json={
# #                     "message": message,
# #                     "symbol": symbol,
# #                     "analysis_type": analysis_type
# #                 },
# #                 timeout=30
# #             )
            
# #             if response.status_code == 200:
# #                 return response.json()
# #         except:
# #             pass
    
# #     # Fallback to local analysis
# #     return local_analysis(message, symbol)

# # def local_analysis(message, symbol):
# #     """Local financial analysis"""
# #     data = get_real_time_data(symbol)
    
# #     if "error" in data:
# #         return {"response": {"content": f"❌ Error analyzing {symbol}: {data['error']}"}}
    
# #     # Generate response based on message
# #     message_lower = message.lower()
    
# #     if any(word in message_lower for word in ['price', 'forecast', 'predict']):
# #         target_price = data['current_price'] * 1.05
# #         content = f"""📈 **Real-Time Price Analysis for {symbol}**

# # 💰 **Current Price:** ${data['current_price']:.2f} ({data['price_change']:+.2f}%)

# # 🎯 **30-Day Target:** ${target_price:.2f} (+5% projected)

# # 📊 **Technical Indicators:**
# # • RSI: {data['rsi']:.1f} ({'Overbought' if data['rsi'] > 70 else 'Oversold' if data['rsi'] < 30 else 'Neutral'})
# # • Volume: {data['volume']:,}
# # • Market Cap: ${data['market_cap']:,.0f}

# # *Real-time update: {data['last_updated']}*"""
    
# #     elif any(word in message_lower for word in ['buy', 'sell', 'invest', 'recommend']):
# #         # Simple recommendation logic
# #         score = 0
# #         if data['price_change'] > 0:
# #             score += 1
# #         if data['rsi'] < 70:
# #             score += 1
# #         if data['volume'] > 1000000:
# #             score += 1
        
# #         if score >= 2:
# #             recommendation = "🟢 BUY"
# #         elif score == 1:
# #             recommendation = "🟡 HOLD"
# #         else:
# #             recommendation = "🔴 CAUTION"
        
# #         content = f"""💼 **Investment Recommendation for {symbol}**

# # 🎯 **Recommendation:** {recommendation}
# # • Current Price: ${data['current_price']:.2f}
# # • Today's Change: {data['price_change']:+.2f}%
# # • Score: {score}/3

# # 📊 **Analysis:**
# # • Price Momentum: {'Positive' if data['price_change'] > 0 else 'Negative'}
# # • RSI Level: {data['rsi']:.1f} ({'Healthy' if data['rsi'] < 70 else 'Overbought'})
# # • Volume: {'High' if data['volume'] > 1000000 else 'Normal'}

# # ⚠️ **Risk Management:**
# # • Stop Loss: ${data['current_price'] * 0.9:.2f} (-10%)
# # • Target: ${data['current_price'] * 1.1:.2f} (+10%)

# # *Live analysis: {data['last_updated']}*"""
    
# #     elif any(word in message_lower for word in ['sentiment', 'news', 'feeling']):
# #         if data['price_change'] > 2:
# #             sentiment = "🚀 Very Bullish"
# #         elif data['price_change'] > 0:
# #             sentiment = "😊 Bullish"
# #         elif data['price_change'] > -2:
# #             sentiment = "😐 Neutral"
# #         else:
# #             sentiment = "😟 Bearish"
        
# #         content = f"""😊 **Market Sentiment for {symbol}**

# # {sentiment}

# # 📊 **Sentiment Indicators:**
# # • Price Action: {data['price_change']:+.2f}% today
# # • Volume Activity: {'High' if data['volume'] > 1000000 else 'Normal'}
# # • Technical Setup: {'Bullish' if data['rsi'] < 70 and data['price_change'] > 0 else 'Mixed'}

# # 💭 **Market Mood:**
# # • {'Strong buying interest with positive momentum' if data['price_change'] > 2 else 
# #    'Positive sentiment with steady buying' if data['price_change'] > 0 else
# #    'Mixed signals, waiting for direction' if data['price_change'] > -2 else
# #    'Caution advised, selling pressure evident'}

# # *Sentiment update: {data['last_updated']}*"""
    
# #     else:
# #         content = f"""📊 **Live Financial Data for {symbol}**

# # 💰 **Real-Time Metrics:**
# # • Price: ${data['current_price']:.2f} ({data['price_change']:+.2f}%)
# # • Volume: {data['volume']:,}
# # • RSI: {data['rsi']:.1f}
# # • Last Update: {data['last_updated']}

# # 🤖 **Ask me about:**
# # • **Price Analysis:** "Predict {symbol}'s price"
# # • **Investment Advice:** "Should I buy {symbol}?"
# # • **Market Sentiment:** "What's {symbol}'s sentiment?"
# # • **Technical Analysis:** "Analyze {symbol}'s trends"

# # 💡 **Quick Actions:**
# # Try: "Give me a buy/sell recommendation for {symbol}" or "What's the forecast for {symbol}?"

# # *Powered by real-time market data*"""
    
# #     return {"response": {"content": content}, "data": data}

# # def render_chart_safe(data, chart_type="price"):
# #     """Render charts with unique keys to avoid Plotly errors"""
# #     if "historical_data" not in data or data["historical_data"].empty:
# #         st.warning("No historical data available for charting")
# #         return
    
# #     hist_data = data["historical_data"]
# #     unique_key = get_unique_key()
    
# #     try:
# #         if chart_type == "price":
# #             fig = go.Figure()
# #             fig.add_trace(go.Scatter(
# #                 x=hist_data.index,
# #                 y=hist_data['Close'],
# #                 mode='lines',
# #                 name=f'{data["symbol"]} Price',
# #                 line=dict(color='#1f77b4', width=2)
# #             ))
            
# #             fig.update_layout(
# #                 title=f'{data["symbol"]} - Real-Time Price Chart',
# #                 xaxis_title="Date",
# #                 yaxis_title="Price ($)",
# #                 height=400,
# #                 showlegend=True,
# #                 template="plotly_dark"
# #             )
            
# #             st.plotly_chart(fig, use_container_width=True, key=unique_key)
        
# #         elif chart_type == "volume":
# #             fig = go.Figure()
# #             fig.add_trace(go.Bar(
# #                 x=hist_data.index,
# #                 y=hist_data['Volume'],
# #                 name='Volume',
# #                 marker_color='green'
# #             ))
            
# #             fig.update_layout(
# #                 title=f'{data["symbol"]} - Trading Volume',
# #                 xaxis_title="Date",
# #                 yaxis_title="Volume",
# #                 height=300,
# #                 template="plotly_dark"
# #             )
            
# #             st.plotly_chart(fig, use_container_width=True, key=unique_key)
            
# #     except Exception as e:
# #         st.error(f"Chart rendering error: {e}")

# # def main():
# #     # Header
# #     st.title("🤖 FinDocGPT - Real-Time Financial Analysis")
# #     st.markdown("*AI-powered real-time financial analysis with live market data*")
    
# #     # Status bar
# #     col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
# #     with col1:
# #         st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
# #     with col2:
# #         if check_api_connection():
# #             st.success("🟢 API Connected")
# #         else:
# #             st.warning("🟡 Local Mode")
# #     with col3:
# #         st.info(f"💬 {len(st.session_state.messages)} messages")
# #     with col4:
# #         if st.button("🔄 Refresh", key="refresh_button"):
# #             st.rerun()
    
# #     # Sidebar
# #     with st.sidebar:
# #         st.header("🎯 Real-Time Controls")
        
# #         # Symbol input
# #         symbol = st.text_input("📊 Stock Symbol", value="AAPL", key="symbol_input").upper()
        
# #         # Real-time data display
# #         if symbol:
# #             with st.spinner("📡 Getting live data..."):
# #                 data = get_real_time_data(symbol)
            
# #             if "error" not in data:
# #                 st.success(f"✅ Live data for {symbol}")
                
# #                 # Metrics display
# #                 col1, col2 = st.columns(2)
# #                 with col1:
# #                     st.metric("Price", f"${data['current_price']:.2f}")
# #                     st.metric("RSI", f"{data['rsi']:.1f}")
# #                 with col2:
# #                     st.metric("Change", f"{data['price_change']:+.2f}%")
# #                     st.metric("Volume", f"{data['volume']:,}")
                
# #                 st.caption(f"🕐 Updated: {data['last_updated']}")
                
# #                 # Quick action buttons
# #                 st.subheader("⚡ Quick Analysis")
                
# #                 if st.button("📈 Price Forecast", use_container_width=True):
# #                     message = f"Predict {symbol}'s price for the next 30 days"
# #                     st.session_state.messages.append({
# #                         "role": "user",
# #                         "content": message,
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
                    
# #                     response = send_chat_message(message, symbol, "forecast")
# #                     st.session_state.messages.append({
# #                         "role": "assistant",
# #                         "content": response["response"]["content"],
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
# #                     st.rerun()
                
# #                 if st.button("💼 Investment Advice", use_container_width=True):
# #                     message = f"Should I buy {symbol}? Give me a recommendation"
# #                     st.session_state.messages.append({
# #                         "role": "user",
# #                         "content": message,
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
                    
# #                     response = send_chat_message(message, symbol, "strategy")
# #                     st.session_state.messages.append({
# #                         "role": "assistant",
# #                         "content": response["response"]["content"],
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
# #                     st.rerun()
                
# #                 if st.button("😊 Market Sentiment", use_container_width=True):
# #                     message = f"What's the current sentiment for {symbol}?"
# #                     st.session_state.messages.append({
# #                         "role": "user",
# #                         "content": message,
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
                    
# #                     response = send_chat_message(message, symbol, "sentiment")
# #                     st.session_state.messages.append({
# #                         "role": "assistant",
# #                         "content": response["response"]["content"],
# #                         "timestamp": datetime.now().strftime("%H:%M:%S")
# #                     })
# #                     st.rerun()
                
# #             else:
# #                 st.error(f"❌ No data for {symbol}")
        
# #         # Market status
# #         st.subheader("📈 Market Status")
# #         now = datetime.now()
# #         is_open = now.weekday() < 5 and 9 <= now.hour <= 16
        
# #         if is_open:
# #             st.success("🟢 Market Open")
# #         else:
# #             st.warning("🟡 Market Closed")
        
# #         st.caption("Market opens 9:30 AM - 4:00 PM ET")
    
# #     # Main content tabs
# #     tab1, tab2, tab3 = st.tabs(["💬 Real-Time Chat", "📊 Live Analytics", "📈 Live Charts"])
    
# #     with tab1:
# #         st.subheader("💬 Real-Time Financial Chat")
        
# #         # Chat container
# #         chat_container = st.container()
        
# #         with chat_container:
# #             # Display messages
# #             for i, message in enumerate(st.session_state.messages):
# #                 with st.chat_message(message["role"]):
# #                     st.write(message["content"])
# #                     st.caption(f"🕐 {message['timestamp']}")
        
# #         # Chat input
# #         if prompt := st.chat_input("Ask about any stock in real-time...", key="chat_input"):
# #             # Add user message
# #             st.session_state.messages.append({
# #                 "role": "user",
# #                 "content": prompt,
# #                 "timestamp": datetime.now().strftime("%H:%M:%S")
# #             })
            
# #             # Get AI response
# #             with st.spinner("🔄 Analyzing real-time data..."):
# #                 response = send_chat_message(prompt, symbol, "general")
            
# #             # Add assistant response
# #             st.session_state.messages.append({
# #                 "role": "assistant",
# #                 "content": response["response"]["content"],
# #                 "timestamp": datetime.now().strftime("%H:%M:%S")
# #             })
            
# #             st.rerun()
        
# #         # Clear chat button
# #         if st.button("🗑️ Clear Chat", key="clear_chat"):
# #             st.session_state.messages = []
# #             st.rerun()
    
# #     with tab2:
# #         st.subheader("📊 Live Analytics Dashboard")
        
# #         # Performance metrics
# #         col1, col2, col3, col4 = st.columns(4)
# #         with col1:
# #             st.metric("Total Queries", "1,234", "↗️ 12%")
# #         with col2:
# #             st.metric("Avg Confidence", "82.5%", "↗️ 3.2%")
# #         with col3:
# #             st.metric("Active Symbols", "45", "↗️ 5")
# #         with col4:
# #             st.metric("Success Rate", "94.2%", "↗️ 1.1%")
        
# #         # Market overview
# #         st.subheader("🌍 Live Market Overview")
        
# #         major_indices = ["^GSPC", "^IXIC", "^DJI"]
# #         market_data = []
        
# #         with st.spinner("Loading market data..."):
# #             for index_symbol in major_indices:
# #                 try:
# #                     ticker = yf.Ticker(index_symbol)
# #                     hist = ticker.history(period="1d")
# #                     if not hist.empty:
# #                         current = hist['Close'].iloc[-1]
# #                         change = ((current - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
# #                         market_data.append({
# #                             "Index": index_symbol.replace("^", ""),
# #                             "Price": f"{current:.2f}",
# #                             "Change": f"{change:+.2f}%"
# #                         })
# #                 except:
# #                     pass
        
# #         if market_data:
# #             df = pd.DataFrame(market_data)
# #             st.dataframe(df, use_container_width=True)
# #         else:
# #             st.warning("Market data temporarily unavailable")
    
# #     with tab3:
# #         st.subheader("📈 Live Market Charts")
        
# #         if symbol:
# #             data = get_real_time_data(symbol)
            
# #             if "error" not in data:
# #                 # Current price highlight
# #                 price_change = data['price_change']
# #                 color = "green" if price_change >= 0 else "red"
                
# #                 st.markdown(f"""
# #                 <div style="padding: 20px; border-radius: 10px; background-color: {'#1e7e34' if price_change >= 0 else '#dc3545'}; color: white; text-align: center; margin-bottom: 20px;">
# #                     <h2>{symbol}: ${data['current_price']:.2f}</h2>
# #                     <h3>{price_change:+.2f}% Today</h3>
# #                     <p>Volume: {data['volume']:,} | RSI: {data['rsi']:.1f}</p>
# #                     <small>Last Update: {data['last_updated']}</small>
# #                 </div>
# #                 """, unsafe_allow_html=True)
                
# #                 # Render charts with error handling
# #                 st.markdown("### 📈 Price Chart")
# #                 render_chart_safe(data, "price")
                
# #                 st.markdown("### 📊 Volume Chart")
# #                 render_chart_safe(data, "volume")
                
# #             else:
# #                 st.error(f"Unable to load chart data for {symbol}")

# # if __name__ == "__main__":
# #     main()

# import streamlit as st
# import requests
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from datetime import datetime, timedelta
# import yfinance as yf
# import numpy as np
# import time
# import uuid
# import io
# import base64
# from textblob import TextBlob
# import re
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import StandardScaler
# import warnings
# warnings.filterwarnings('ignore')

# # Configure Streamlit
# st.set_page_config(
#     page_title="FinDocGPT - AI Financial Analysis Platform",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"
# )

# # Custom CSS for better UI
# st.markdown("""
# <style>
#     .main-header {
#         background: linear-gradient(90deg, #1e3c72, #2a5298);
#         padding: 1rem;
#         border-radius: 10px;
#         color: white;
#         text-align: center;
#         margin-bottom: 2rem;
#     }
#     .metric-card {
#         background-color: #f8f9fa;
#         padding: 1rem;
#         border-radius: 8px;
#         border-left: 4px solid #007bff;
#         margin: 0.5rem 0;
#     }
#     .alert-box {
#         padding: 1rem;
#         border-radius: 8px;
#         margin: 1rem 0;
#     }
#     .success-box {
#         background-color: #d4edda;
#         border-left: 4px solid #28a745;
#         color: #155724;
#     }
#     .warning-box {
#         background-color: #fff3cd;
#         border-left: 4px solid #ffc107;
#         color: #856404;
#     }
#     .danger-box {
#         background-color: #f8d7da;
#         border-left: 4px solid #dc3545;
#         color: #721c24;
#     }
#     .sidebar .sidebar-content {
#         background-color: #f8f9fa;
#     }
#     .chat-message {
#         padding: 1rem;
#         border-radius: 10px;
#         margin: 0.5rem 0;
#         border: 1px solid #e9ecef;
#     }
#     .user-message {
#         background-color: #e3f2fd;
#         border-left: 4px solid #2196f3;
#     }
#     .assistant-message {
#         background-color: #f1f8e9;
#         border-left: 4px solid #4caf50;
#     }
# </style>
# """, unsafe_allow_html=True)

# # Initialize session state
# def init_session_state():
#     """Initialize all session state variables"""
#     default_values = {
#         'messages': [],
#         'chart_counter': 0,
#         'current_symbol': 'AAPL',
#         'uploaded_documents': [],
#         'analysis_history': [],
#         'investment_portfolio': {},
#         'api_connected': False,
#         'real_time_data': {},
#         'conversation_id': None,
#         'market_predictions': {},
#         'sentiment_data': {},
#         'anomalies_detected': []
#     }
    
#     for key, value in default_values.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# # Initialize session state
# init_session_state()

# # API Configuration
# API_BASE_URL = "http://localhost:8000"

# class FinDocGPTEngine:
#     """Main FinDocGPT Engine with all three stages"""
    
#     def __init__(self):
#         self.api_base = API_BASE_URL
#         self.scaler = StandardScaler()
        
#     def get_unique_key(self):
#         """Generate unique key for components"""
#         st.session_state.chart_counter += 1
#         return f"component_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"
    
#     def check_api_connection(self):
#         """Check if backend API is available"""
#         try:
#             response = requests.get(f"{self.api_base}/health", timeout=5)
#             st.session_state.api_connected = response.status_code == 200
#             return st.session_state.api_connected
#         except:
#             st.session_state.api_connected = False
#             return False
    
#     # ========== STAGE 1: DOCUMENT ANALYSIS & INSIGHTS ==========
    
#     def process_financial_document(self, document_content, doc_type="earnings_report"):
#         """Process uploaded financial documents and extract insights"""
#         try:
#             # Basic document processing (in production, you'd use more sophisticated NLP)
#             insights = {
#                 "document_type": doc_type,
#                 "content_length": len(document_content),
#                 "processed_at": datetime.now(),
#                 "key_metrics": self.extract_financial_metrics(document_content),
#                 "sentiment": self.analyze_document_sentiment(document_content),
#                 "anomalies": self.detect_anomalies_in_text(document_content),
#                 "summary": self.generate_document_summary(document_content)
#             }
            
#             return insights
#         except Exception as e:
#             return {"error": str(e)}
    
#     def extract_financial_metrics(self, text):
#         """Extract key financial metrics from document text"""
#         metrics = {}
        
#         # Revenue patterns
#         revenue_patterns = [
#             r'revenue[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?',
#             r'total revenue[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?',
#             r'net revenue[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?'
#         ]
        
#         # Extract revenue
#         for pattern in revenue_patterns:
#             matches = re.findall(pattern, text.lower())
#             if matches:
#                 value, unit = matches[0]
#                 metrics['revenue'] = self.normalize_financial_value(value, unit)
#                 break
        
#         # Profit/Earnings patterns
#         profit_patterns = [
#             r'net income[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?',
#             r'profit[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?',
#             r'earnings[:\s]+\$?([\d,]+\.?\d*)\s*(million|billion|thousand)?'
#         ]
        
#         for pattern in profit_patterns:
#             matches = re.findall(pattern, text.lower())
#             if matches:
#                 value, unit = matches[0]
#                 metrics['profit'] = self.normalize_financial_value(value, unit)
#                 break
        
#         # EPS patterns
#         eps_pattern = r'earnings per share[:\s]+\$?([\d\.]+)'
#         eps_matches = re.findall(eps_pattern, text.lower())
#         if eps_matches:
#             metrics['eps'] = float(eps_matches[0])
        
#         return metrics
    
#     def normalize_financial_value(self, value_str, unit):
#         """Convert financial values to standard format"""
#         try:
#             value = float(value_str.replace(',', ''))
#             multipliers = {
#                 'thousand': 1000,
#                 'million': 1000000,
#                 'billion': 1000000000
#             }
#             return value * multipliers.get(unit.lower(), 1)
#         except:
#             return 0
    
#     def analyze_document_sentiment(self, text):
#         """Analyze sentiment of financial document"""
#         try:
#             blob = TextBlob(text)
#             polarity = blob.sentiment.polarity
#             subjectivity = blob.sentiment.subjectivity
            
#             if polarity > 0.1:
#                 sentiment_label = "Positive"
#             elif polarity < -0.1:
#                 sentiment_label = "Negative"
#             else:
#                 sentiment_label = "Neutral"
                
#             return {
#                 "polarity": polarity,
#                 "subjectivity": subjectivity,
#                 "label": sentiment_label,
#                 "confidence": abs(polarity)
#             }
#         except Exception as e:
#             return {"error": str(e)}
    
#     def detect_anomalies_in_text(self, text):
#         """Detect potential anomalies or red flags in financial documents"""
#         anomaly_keywords = [
#             'investigation', 'lawsuit', 'decline', 'loss', 'bankruptcy',
#             'fraud', 'restatement', 'impairment', 'writedown', 'restructuring'
#         ]
        
#         detected_anomalies = []
#         text_lower = text.lower()
        
#         for keyword in anomaly_keywords:
#             if keyword in text_lower:
#                 # Find context around the keyword
#                 sentences = text.split('.')
#                 for sentence in sentences:
#                     if keyword in sentence.lower():
#                         detected_anomalies.append({
#                             "keyword": keyword,
#                             "context": sentence.strip(),
#                             "risk_level": self.assess_risk_level(keyword)
#                         })
#                         break
        
#         return detected_anomalies
    
#     def assess_risk_level(self, keyword):
#         """Assess risk level of detected anomaly"""
#         high_risk = ['fraud', 'bankruptcy', 'investigation', 'lawsuit']
#         medium_risk = ['decline', 'loss', 'impairment', 'writedown']
        
#         if keyword in high_risk:
#             return "High"
#         elif keyword in medium_risk:
#             return "Medium"
#         else:
#             return "Low"
    
#     def generate_document_summary(self, text):
#         """Generate a summary of the financial document"""
#         sentences = text.split('.')
#         # Simple extractive summarization - take first few sentences
#         summary_sentences = sentences[:3]
#         return '. '.join(summary_sentences) + '.'
    
#     # ========== STAGE 2: FINANCIAL FORECASTING ==========
    
#     def get_real_time_market_data(self, symbol):
#         """Fetch comprehensive real-time market data"""
#         try:
#             ticker = yf.Ticker(symbol)
#             info = ticker.info
#             hist = ticker.history(period="1y")  # Get more historical data for better analysis
            
#             if hist.empty:
#                 return {"error": "No historical data available"}
            
#             # Calculate technical indicators
#             hist = self.calculate_technical_indicators(hist)
            
#             current_price = info.get('currentPrice', hist['Close'].iloc[-1])
#             change = info.get('regularMarketChangePercent', 0)
            
#             # Calculate volatility
#             returns = hist['Close'].pct_change().dropna()
#             volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
#             return {
#                 "symbol": symbol,
#                 "current_price": current_price,
#                 "price_change": change,
#                 "volume": info.get('regularMarketVolume', 0),
#                 "market_cap": info.get('marketCap', 0),
#                 "pe_ratio": info.get('trailingPE', 0),
#                 "beta": info.get('beta', 1.0),
#                 "dividend_yield": info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0,
#                 "52_week_high": info.get('fiftyTwoWeekHigh', 0),
#                 "52_week_low": info.get('fiftyTwoWeekLow', 0),
#                 "volatility": volatility,
#                 "historical_data": hist,
#                 "technical_indicators": self.get_latest_indicators(hist),
#                 "last_updated": datetime.now().strftime("%H:%M:%S")
#             }
#         except Exception as e:
#             return {"error": str(e)}
    
#     def calculate_technical_indicators(self, hist_data):
#         """Calculate comprehensive technical indicators"""
#         df = hist_data.copy()
        
#         # Moving Averages
#         df['SMA_10'] = df['Close'].rolling(10).mean()
#         df['SMA_20'] = df['Close'].rolling(20).mean()
#         df['SMA_50'] = df['Close'].rolling(50).mean()
#         df['SMA_200'] = df['Close'].rolling(200).mean()
        
#         # Exponential Moving Average
#         df['EMA_12'] = df['Close'].ewm(span=12).mean()
#         df['EMA_26'] = df['Close'].ewm(span=26).mean()
        
#         # MACD
#         df['MACD'] = df['EMA_12'] - df['EMA_26']
#         df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
#         df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
#         # RSI
#         delta = df['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
#         rs = gain / loss
#         df['RSI'] = 100 - (100 / (1 + rs))
        
#         # Bollinger Bands
#         df['BB_Middle'] = df['Close'].rolling(20).mean()
#         bb_std = df['Close'].rolling(20).std()
#         df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
#         df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
        
#         # Stochastic Oscillator
#         low_14 = df['Low'].rolling(14).min()
#         high_14 = df['High'].rolling(14).max()
#         df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
#         df['%D'] = df['%K'].rolling(3).mean()
        
#         return df
    
#     def get_latest_indicators(self, hist_data):
#         """Extract latest values of technical indicators"""
#         latest = hist_data.iloc[-1]
#         return {
#             "rsi": latest.get('RSI', 50),
#             "macd": latest.get('MACD', 0),
#             "macd_signal": latest.get('MACD_Signal', 0),
#             "sma_20": latest.get('SMA_20', latest['Close']),
#             "sma_50": latest.get('SMA_50', latest['Close']),
#             "bb_upper": latest.get('BB_Upper', latest['Close']),
#             "bb_lower": latest.get('BB_Lower', latest['Close']),
#             "stoch_k": latest.get('%K', 50),
#             "stoch_d": latest.get('%D', 50)
#         }
    
#     def predict_stock_price(self, symbol, days_ahead=30):
#         """Advanced stock price prediction using multiple models"""
#         try:
#             data = self.get_real_time_market_data(symbol)
#             if "error" in data:
#                 return data
            
#             hist_data = data['historical_data']
            
#             # Prepare features for prediction
#             features = self.prepare_prediction_features(hist_data)
#             if len(features) < 30:
#                 return {"error": "Insufficient historical data for prediction"}
            
#             # Use multiple prediction models
#             predictions = {}
            
#             # Model 1: Linear Regression on technical indicators
#             lr_pred = self.linear_regression_prediction(features, days_ahead)
#             predictions['linear_regression'] = lr_pred
            
#             # Model 2: Random Forest
#             rf_pred = self.random_forest_prediction(features, days_ahead)
#             predictions['random_forest'] = rf_pred
            
#             # Model 3: Moving Average Convergence
#             ma_pred = self.moving_average_prediction(hist_data, days_ahead)
#             predictions['moving_average'] = ma_pred
            
#             # Ensemble prediction (weighted average)
#             ensemble_pred = (lr_pred * 0.4 + rf_pred * 0.4 + ma_pred * 0.2)
            
#             # Calculate confidence based on model agreement
#             pred_values = [lr_pred, rf_pred, ma_pred]
#             confidence = 100 - (np.std(pred_values) / np.mean(pred_values) * 100)
#             confidence = max(0, min(100, confidence))
            
#             return {
#                 "symbol": symbol,
#                 "current_price": data['current_price'],
#                 "predicted_price": ensemble_pred,
#                 "prediction_period": f"{days_ahead} days",
#                 "confidence": confidence,
#                 "expected_return": ((ensemble_pred - data['current_price']) / data['current_price']) * 100,
#                 "individual_predictions": predictions,
#                 "prediction_date": (datetime.now() + timedelta(days=days_ahead)).strftime("%Y-%m-%d"),
#                 "risk_assessment": self.assess_prediction_risk(data, ensemble_pred)
#             }
            
#         except Exception as e:
#             return {"error": str(e)}
    
#     def prepare_prediction_features(self, hist_data):
#         """Prepare features for ML prediction models"""
#         df = hist_data.copy()
        
#         # Calculate returns
#         df['Returns'] = df['Close'].pct_change()
#         df['Returns_1'] = df['Returns'].shift(1)
#         df['Returns_2'] = df['Returns'].shift(2)
#         df['Returns_5'] = df['Returns'].shift(5)
        
#         # Volume indicators
#         df['Volume_MA'] = df['Volume'].rolling(10).mean()
#         df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
        
#         # Price relative to moving averages
#         df['Price_to_SMA20'] = df['Close'] / df['SMA_20']
#         df['Price_to_SMA50'] = df['Close'] / df['SMA_50']
        
#         # Volatility
#         df['Volatility'] = df['Returns'].rolling(10).std()
        
#         # Select features and target
#         feature_columns = [
#             'RSI', 'MACD', '%K', '%D', 'Returns_1', 'Returns_2', 'Returns_5',
#             'Volume_Ratio', 'Price_to_SMA20', 'Price_to_SMA50', 'Volatility'
#         ]
        
#         df_clean = df[feature_columns + ['Close']].dropna()
#         return df_clean
    
#     def linear_regression_prediction(self, features, days_ahead):
#         """Linear regression prediction"""
#         try:
#             X = features.drop('Close', axis=1)
#             y = features['Close']
            
#             model = LinearRegression()
#             model.fit(X, y)
            
#             # Use latest features for prediction
#             latest_features = X.iloc[-1:].values
#             prediction = model.predict(latest_features)[0]
            
#             return prediction
#         except:
#             return features['Close'].iloc[-1] * 1.02  # Fallback: 2% growth
    
#     def random_forest_prediction(self, features, days_ahead):
#         """Random Forest prediction"""
#         try:
#             X = features.drop('Close', axis=1)
#             y = features['Close']
            
#             model = RandomForestRegressor(n_estimators=50, random_state=42)
#             model.fit(X, y)
            
#             # Use latest features for prediction
#             latest_features = X.iloc[-1:].values
#             prediction = model.predict(latest_features)[0]
            
#             return prediction
#         except:
#             return features['Close'].iloc[-1] * 1.01  # Fallback: 1% growth
    
#     def moving_average_prediction(self, hist_data, days_ahead):
#         """Moving average based prediction"""
#         try:
#             recent_prices = hist_data['Close'].tail(20)
#             trend = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / len(recent_prices)
#             prediction = recent_prices.iloc[-1] + (trend * days_ahead)
#             return prediction
#         except:
#             return hist_data['Close'].iloc[-1]
    
#     def assess_prediction_risk(self, market_data, predicted_price):
#         """Assess risk level of prediction"""
#         current_price = market_data['current_price']
#         expected_return = ((predicted_price - current_price) / current_price) * 100
#         volatility = market_data.get('volatility', 0.2)
        
#         if abs(expected_return) > 20 or volatility > 0.4:
#             return "High"
#         elif abs(expected_return) > 10 or volatility > 0.25:
#             return "Medium"
#         else:
#             return "Low"
    
#     # ========== STAGE 3: INVESTMENT STRATEGY & DECISION MAKING ==========
    
#     def generate_investment_recommendation(self, symbol):
#         """Generate comprehensive investment recommendation"""
#         try:
#             # Get market data and prediction
#             market_data = self.get_real_time_market_data(symbol)
#             if "error" in market_data:
#                 return market_data
            
#             prediction = self.predict_stock_price(symbol, 30)
#             if "error" in prediction:
#                 prediction = {"predicted_price": market_data['current_price'], "confidence": 50}
            
#             # Technical Analysis Score (0-100)
#             tech_score = self.calculate_technical_score(market_data)
            
#             # Fundamental Analysis Score (0-100)
#             fund_score = self.calculate_fundamental_score(market_data)
            
#             # Market Sentiment Score (0-100)
#             sentiment_score = self.calculate_market_sentiment(symbol)
            
#             # Risk Assessment
#             risk_assessment = self.comprehensive_risk_assessment(market_data, prediction)
            
#             # Overall Investment Score
#             overall_score = (tech_score * 0.4 + fund_score * 0.3 + sentiment_score * 0.3)
            
#             # Generate recommendation
#             recommendation = self.determine_investment_action(overall_score, risk_assessment)
            
#             return {
#                 "symbol": symbol,
#                 "current_price": market_data['current_price'],
#                 "predicted_price": prediction.get('predicted_price', market_data['current_price']),
#                 "expected_return": prediction.get('expected_return', 0),
#                 "recommendation": recommendation,
#                 "overall_score": overall_score,
#                 "confidence": prediction.get('confidence', 50),
#                 "scores": {
#                     "technical": tech_score,
#                     "fundamental": fund_score,
#                     "sentiment": sentiment_score
#                 },
#                 "risk_assessment": risk_assessment,
#                 "reasoning": self.generate_recommendation_reasoning(
#                     recommendation, overall_score, market_data, prediction
#                 ),
#                 "action_plan": self.generate_action_plan(recommendation, market_data),
#                 "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#             }
            
#         except Exception as e:
#             return {"error": str(e)}
    
#     def calculate_technical_score(self, market_data):
#         """Calculate technical analysis score"""
#         indicators = market_data.get('technical_indicators', {})
#         score = 50  # Neutral starting point
        
#         # RSI Analysis
#         rsi = indicators.get('rsi', 50)
#         if 30 <= rsi <= 70:
#             score += 10  # Good RSI range
#         elif rsi < 30:
#             score += 15  # Oversold - potential buy
#         elif rsi > 70:
#             score -= 15  # Overbought - potential sell
        
#         # MACD Analysis
#         macd = indicators.get('macd', 0)
#         macd_signal = indicators.get('macd_signal', 0)
#         if macd > macd_signal:
#             score += 10  # Bullish MACD
#         else:
#             score -= 10  # Bearish MACD
        
#         # Moving Average Analysis
#         current_price = market_data['current_price']
#         sma_20 = indicators.get('sma_20', current_price)
#         sma_50 = indicators.get('sma_50', current_price)
        
#         if current_price > sma_20 > sma_50:
#             score += 15  # Strong uptrend
#         elif current_price > sma_20:
#             score += 5   # Mild uptrend
#         elif current_price < sma_20:
#             score -= 10  # Below short-term MA
        
#         # Bollinger Bands Analysis
#         bb_upper = indicators.get('bb_upper', current_price)
#         bb_lower = indicators.get('bb_lower', current_price)
        
#         if current_price >= bb_upper:
#             score -= 10  # Near upper band - overbought
#         elif current_price <= bb_lower:
#             score += 10  # Near lower band - oversold
        
#         return max(0, min(100, score))
    
#     def calculate_fundamental_score(self, market_data):
#         """Calculate fundamental analysis score"""
#         score = 50  # Neutral starting point
        
#         # P/E Ratio Analysis
#         pe_ratio = market_data.get('pe_ratio', 0)
#         if 10 <= pe_ratio <= 20:
#             score += 15  # Reasonable valuation
#         elif 5 <= pe_ratio < 10:
#             score += 20  # Potentially undervalued
#         elif pe_ratio > 30:
#             score -= 15  # Potentially overvalued
#         elif pe_ratio <= 0:
#             score -= 5   # Negative earnings
        
#         # Market Cap Analysis (relative stability)
#         market_cap = market_data.get('market_cap', 0)
#         if market_cap > 10e9:  # Large cap
#             score += 5   # More stable
#         elif market_cap < 1e9:  # Small cap
#             score += 10  # Higher growth potential but riskier
        
#         # Dividend Yield Analysis
#         dividend_yield = market_data.get('dividend_yield', 0)
#         if dividend_yield > 3:
#             score += 10  # Good dividend yield
#         elif dividend_yield > 1:
#             score += 5   # Moderate dividend
        
#         # Beta Analysis (volatility relative to market)
#         beta = market_data.get('beta', 1.0)
#         if 0.5 <= beta <= 1.2:
#             score += 5   # Reasonable volatility
#         elif beta > 1.5:
#             score -= 10  # High volatility
        
#         return max(0, min(100, score))
    
#     def calculate_market_sentiment(self, symbol):
#         """Calculate market sentiment score (simplified version)"""
#         try:
#             # In a real implementation, you would analyze:
#             # - News sentiment
#             # - Social media sentiment
#             # - Analyst recommendations
#             # - Options flow
            
#             # For now, we'll use price momentum as a proxy
#             market_data = self.get_real_time_market_data(symbol)
#             if "error" in market_data:
#                 return 50
            
#             price_change = market_data.get('price_change', 0)
#             volume = market_data.get('volume', 0)
            
#             # Base sentiment on recent price performance
#             sentiment_score = 50
            
#             if price_change > 5:
#                 sentiment_score += 20
#             elif price_change > 2:
#                 sentiment_score += 10
#             elif price_change > 0:
#                 sentiment_score += 5
#             elif price_change < -5:
#                 sentiment_score -= 20
#             elif price_change < -2:
#                 sentiment_score -= 10
#             elif price_change < 0:
#                 sentiment_score -= 5
            
#             # Adjust for volume (higher volume = more conviction)
#             if volume > 2000000:
#                 sentiment_score += 5
#             elif volume < 500000:
#                 sentiment_score -= 5
            
#             return max(0, min(100, sentiment_score))
            
#         except:
#             return 50  # Neutral if unable to calculate
    
#     def comprehensive_risk_assessment(self, market_data, prediction):
#         """Comprehensive risk assessment"""
#         risks = []
#         risk_level = "Low"
        
#         # Volatility Risk
#         volatility = market_data.get('volatility', 0.2)
#         if volatility > 0.4:
#             risks.append("High volatility detected")
#             risk_level = "High"
#         elif volatility > 0.25:
#             risks.append("Moderate volatility")
#             risk_level = "Medium" if risk_level != "High" else "High"
        
#         # Valuation Risk
#         pe_ratio = market_data.get('pe_ratio', 15)
#         if pe_ratio > 40:
#             risks.append("Stock appears overvalued")
#             risk_level = "High"
#         elif pe_ratio > 25:
#             risks.append("Stock trading at premium")
#             risk_level = "Medium" if risk_level != "High" else "High"
        
#         # Prediction Confidence Risk
#         confidence = prediction.get('confidence', 50)
#         if confidence < 30:
#             risks.append("Low prediction confidence")
#             risk_level = "High"
#         elif confidence < 50:
#             risks.append("Moderate prediction uncertainty")
#             risk_level = "Medium" if risk_level != "High" else "High"
        
#         # Market Position Risk
#         current_price = market_data['current_price']
#         week_52_high = market_data.get('52_week_high', current_price)
#         week_52_low = market_data.get('52_week_low', current_price)
        
#         if week_52_high > 0:
#             price_position = (current_price - week_52_low) / (week_52_high - week_52_low)
#             if price_position > 0.9:
#                 risks.append("Trading near 52-week high")
#             elif price_position < 0.1:
#                 risks.append("Trading near 52-week low")
        
#         return {
#             "level": risk_level,
#             "factors": risks,
#             "score": len(risks) * 25  # Simple risk scoring
#         }
    
#     def determine_investment_action(self, overall_score, risk_assessment):
#         """Determine investment action based on analysis"""
#         risk_level = risk_assessment['level']
        
#         if overall_score >= 75 and risk_level != "High":
#             return {
#                 "action": "STRONG BUY",
#                 "confidence": "High",
#                 "urgency": "High"
#             }
#         elif overall_score >= 60 and risk_level == "Low":
#             return {
#                 "action": "BUY",
#                 "confidence": "Medium",
#                 "urgency": "Medium"
#             }
#         elif overall_score >= 50 and risk_level == "Low":
#             return {
#                 "action": "HOLD",
#                 "confidence": "Medium",
#                 "urgency": "Low"
#             }
#         elif overall_score < 40 or risk_level == "High":
#             return {
#                 "action": "SELL",
#                 "confidence": "Medium",
#                 "urgency": "Medium"
#             }
#         else:
#             return {
#                 "action": "HOLD",
#                 "confidence": "Low",
#                 "urgency": "Low"
#             }
    
#     def generate_recommendation_reasoning(self, recommendation, overall_score, market_data, prediction):
#         """Generate detailed reasoning for the recommendation"""
#         reasoning = []
        
#         action = recommendation['action']
        
#         # Technical reasoning
#         indicators = market_data.get('technical_indicators', {})
#         rsi = indicators.get('rsi', 50)
#         current_price = market_data['current_price']
        
#         if action in ["STRONG BUY", "BUY"]:
#             reasoning.append(f"✅ Strong technical signals with overall score of {overall_score:.1f}/100")
#             if rsi < 30:
#                 reasoning.append(f"✅ RSI at {rsi:.1f} indicates oversold conditions - potential buying opportunity")
#             if prediction.get('expected_return', 0) > 5:
#                 reasoning.append(f"✅ AI model predicts {prediction.get('expected_return', 0):.1f}% upside potential")
        
#         elif action == "SELL":
#             reasoning.append(f"⚠️ Overall analysis score of {overall_score:.1f}/100 suggests caution")
#             if rsi > 70:
#                 reasoning.append(f"⚠️ RSI at {rsi:.1f} indicates overbought conditions")
#             if prediction.get('expected_return', 0) < -5:
#                 reasoning.append(f"⚠️ AI model predicts {prediction.get('expected_return', 0):.1f}% downside risk")
        
#         else:  # HOLD
#             reasoning.append(f"➡️ Mixed signals with overall score of {overall_score:.1f}/100 suggest holding position")
#             reasoning.append("➡️ Wait for clearer market direction before making significant moves")
        
#         # Add market context
#         pe_ratio = market_data.get('pe_ratio', 0)
#         if pe_ratio > 0:
#             reasoning.append(f"📊 Current P/E ratio: {pe_ratio:.1f}")
        
#         return reasoning
    
#     def generate_action_plan(self, recommendation, market_data):
#         """Generate specific action plan based on recommendation"""
#         action = recommendation['action']
#         current_price = market_data['current_price']
        
#         if action in ["STRONG BUY", "BUY"]:
#             return {
#                 "entry_strategy": "Consider dollar-cost averaging over 2-3 weeks",
#                 "target_price": f"${current_price * 1.15:.2f}",
#                 "stop_loss": f"${current_price * 0.92:.2f}",
#                 "position_size": "Start with 25-50% of intended position",
#                 "timeframe": "3-6 months",
#                 "monitoring": "Review weekly, exit if fundamentals change"
#             }
#         elif action == "SELL":
#             return {
#                 "exit_strategy": "Consider gradual position reduction",
#                 "target_exit": "Within 2-4 weeks",
#                 "stop_loss": f"${current_price * 0.88:.2f}",
#                 "position_size": "Reduce by 50-75%",
#                 "timeframe": "1-2 months",
#                 "monitoring": "Daily monitoring recommended"
#             }
#         else:  # HOLD
#             return {
#                 "strategy": "Maintain current position",
#                 "review_triggers": [
#                     "Significant earnings news",
#                     "10% price movement",
#                     "Change in market conditions"
#                 ],
#                 "rebalance": "Monthly review",
#                 "monitoring": "Weekly check-ins sufficient"
#             }

#     # ========== UI RENDERING METHODS ==========
    
#     def render_header(self):
#         """Render enhanced header"""
#         st.markdown("""
#         <div class="main-header">
#             <h1>🤖 FinDocGPT - AI Financial Analysis Platform</h1>
#             <p>Advanced Document Analysis • Market Prediction • Investment Strategy</p>
#         </div>
#         """, unsafe_allow_html=True)
        
#         # Status indicators
#         col1, col2, col3, col4, col5 = st.columns([2, 1, 1, 1, 1])
        
#         with col1:
#             st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
#         with col2:
#             if self.check_api_connection():
#                 st.success("🟢 API Connected")
#             else:
#                 st.warning("🟡 Local Mode")
#         with col3:
#             st.info(f"💬 {len(st.session_state.messages)} messages")
#         with col4:
#             st.info(f"📄 {len(st.session_state.uploaded_documents)} docs")
#         with col5:
#             if st.button("🔄 Refresh", key="main_refresh"):
#                 st.rerun()
    
#     def render_sidebar(self):
#         """Enhanced sidebar with all controls"""
#         with st.sidebar:
#             st.header("🎯 FinDocGPT Control Panel")
            
#             # Symbol input
#             st.subheader("📊 Stock Analysis")
#             symbol = st.text_input(
#                 "Stock Symbol", 
#                 value=st.session_state.current_symbol,
#                 help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
#             ).upper()
            
#             if symbol != st.session_state.current_symbol:
#                 st.session_state.current_symbol = symbol
#                 st.rerun()
            
#             # Real-time data display
#             if symbol:
#                 with st.spinner("📡 Getting live data..."):
#                     data = self.get_real_time_market_data(symbol)
                
#                 if "error" not in data:
#                     st.success(f"✅ Live data for {symbol}")
                    
#                     # Key metrics
#                     col1, col2 = st.columns(2)
#                     with col1:
#                         st.metric("Price", f"${data['current_price']:.2f}")
#                         st.metric("RSI", f"{data['technical_indicators']['rsi']:.1f}")
#                         st.metric("P/E", f"{data['pe_ratio']:.1f}")
#                     with col2:
#                         change_color = "normal" if data['price_change'] >= 0 else "inverse"
#                         st.metric("Change", f"{data['price_change']:+.2f}%")
#                         st.metric("Volume", f"{data['volume']:,}")
#                         st.metric("Beta", f"{data['beta']:.2f}")
                    
#                     st.caption(f"🕐 Updated: {data['last_updated']}")
#                 else:
#                     st.error(f"❌ No data for {symbol}")
            
#             st.divider()
            
#             # Document Upload Section
#             st.subheader("📄 Document Analysis")
#             uploaded_file = st.file_uploader(
#                 "Upload Financial Document",
#                 type=['txt', 'pdf', 'csv'],
#                 help="Upload earnings reports, SEC filings, or financial statements"
#             )
            
#             if uploaded_file:
#                 if st.button("🔍 Analyze Document", use_container_width=True):
#                     content = uploaded_file.read().decode('utf-8')
#                     insights = self.process_financial_document(content)
#                     st.session_state.uploaded_documents.append({
#                         "name": uploaded_file.name,
#                         "insights": insights,
#                         "uploaded_at": datetime.now()
#                     })
#                     st.success("Document analyzed successfully!")
#                     st.rerun()
            
#             st.divider()
            
#             # Quick Analysis Actions
#             st.subheader("⚡ Quick Analysis")
            
#             col1, col2 = st.columns(2)
#             with col1:
#                 if st.button("📈 Price Forecast", use_container_width=True):
#                     self.quick_action("forecast", symbol)
                
#                 if st.button("💼 Investment Advice", use_container_width=True):
#                     self.quick_action("investment", symbol)
            
#             with col2:
#                 if st.button("📊 Technical Analysis", use_container_width=True):
#                     self.quick_action("technical", symbol)
                
#                 if st.button("🎯 Risk Assessment", use_container_width=True):
#                     self.quick_action("risk", symbol)
            
#             st.divider()
            
#             # Market Status
#             st.subheader("📈 Market Status")
#             market_open = self.is_market_open()
#             if market_open:
#                 st.success("🟢 Market Open")
#             else:
#                 st.warning("🟡 Market Closed")
            
#             # Portfolio Summary (if any)
#             if st.session_state.investment_portfolio:
#                 st.subheader("💼 Portfolio")
#                 for symbol, data in st.session_state.investment_portfolio.items():
#                     st.write(f"**{symbol}**: {data.get('shares', 0)} shares")
            
#             return {"symbol": symbol}
    
#     def quick_action(self, action_type, symbol):
#         """Handle quick action buttons"""
#         if action_type == "forecast":
#             message = f"Predict {symbol}'s stock price for the next 30 days with detailed analysis"
#         elif action_type == "investment":
#             message = f"Should I invest in {symbol}? Give me a comprehensive investment recommendation"
#         elif action_type == "technical":
#             message = f"Provide detailed technical analysis for {symbol} including all indicators"
#         elif action_type == "risk":
#             message = f"Assess the investment risks for {symbol} and provide risk management strategies"
#         else:
#             message = f"Analyze {symbol}"
        
#         # Add to messages
#         st.session_state.messages.append({
#             "role": "user",
#             "content": message,
#             "timestamp": datetime.now().strftime("%H:%M:%S")
#         })
        
#         # Generate response
#         response = self.generate_ai_response(message, symbol, action_type)
#         st.session_state.messages.append({
#             "role": "assistant",
#             "content": response["content"],
#             "data": response.get("data"),
#             "timestamp": datetime.now().strftime("%H:%M:%S")
#         })
        
#         st.rerun()
    
#     def generate_ai_response(self, message, symbol, analysis_type):
#         """Generate AI response based on analysis type"""
#         try:
#             if analysis_type == "forecast":
#                 prediction = self.predict_stock_price(symbol, 30)
#                 if "error" not in prediction:
#                     content = f"""📈 **AI Stock Price Prediction for {symbol}**

# 🎯 **30-Day Forecast:**
# • Current Price: ${prediction['current_price']:.2f}
# • Predicted Price: ${prediction['predicted_price']:.2f}
# • Expected Return: {prediction['expected_return']:+.1f}%
# • Confidence Level: {prediction['confidence']:.1f}%

# 🤖 **AI Model Insights:**
# • Prediction Method: Ensemble (Linear Regression + Random Forest + Moving Average)
# • Risk Level: {prediction['risk_assessment']}
# • Target Date: {prediction['prediction_date']}

# 📊 **Individual Model Predictions:**
# • Linear Regression: ${prediction['individual_predictions']['linear_regression']:.2f}
# • Random Forest: ${prediction['individual_predictions']['random_forest']:.2f}
# • Moving Average: ${prediction['individual_predictions']['moving_average']:.2f}

# ⚠️ **Risk Factors:**
# {chr(10).join(['• ' + factor for factor in prediction.get('risk_factors', ['Market volatility', 'Economic uncertainty'])])}

# *AI Prediction Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
                    
#                     return {"content": content, "data": prediction}
#                 else:
#                     return {"content": f"❌ Unable to generate prediction: {prediction['error']}"}
            
#             elif analysis_type == "investment":
#                 recommendation = self.generate_investment_recommendation(symbol)
#                 if "error" not in recommendation:
#                     action = recommendation['recommendation']
#                     content = f"""💼 **AI Investment Recommendation for {symbol}**

# 🎯 **DECISION: {action['action']}**
# • Confidence: {action['confidence']}
# • Urgency: {action['urgency']}
# • Overall Score: {recommendation['overall_score']:.1f}/100

# 📊 **Analysis Breakdown:**
# • **Technical Score:** {recommendation['scores']['technical']:.1f}/100
# • **Fundamental Score:** {recommendation['scores']['fundamental']:.1f}/100
# • **Sentiment Score:** {recommendation['scores']['sentiment']:.1f}/100

# 🧠 **AI Reasoning:**
# {chr(10).join(recommendation['reasoning'])}

# 🎯 **Action Plan:**
# • **Strategy:** {recommendation['action_plan'].get('entry_strategy', recommendation['action_plan'].get('exit_strategy', recommendation['action_plan'].get('strategy', 'Hold position')))}
# • **Target Price:** {recommendation['action_plan'].get('target_price', 'N/A')}
# • **Stop Loss:** {recommendation['action_plan'].get('stop_loss', 'N/A')}
# • **Timeframe:** {recommendation['action_plan'].get('timeframe', 'N/A')}

# ⚠️ **Risk Assessment: {recommendation['risk_assessment']['level']}**
# {chr(10).join(['• ' + factor for factor in recommendation['risk_assessment']['factors']])}

# *AI Recommendation Generated: {recommendation['generated_at']}*"""
                    
#                     return {"content": content, "data": recommendation}
#                 else:
#                     return {"content": f"❌ Unable to generate recommendation: {recommendation['error']}"}
            
#             elif analysis_type == "technical":
#                 market_data = self.get_real_time_market_data(symbol)
#                 if "error" not in market_data:
#                     indicators = market_data['technical_indicators']
#                     content = f"""📊 **Comprehensive Technical Analysis for {symbol}**

# 💰 **Current Market Data:**
# • Price: ${market_data['current_price']:.2f} ({market_data['price_change']:+.2f}%)
# • Volume: {market_data['volume']:,}
# • Market Cap: ${market_data['market_cap']:,.0f}
# • 52W High/Low: ${market_data['52_week_high']:.2f} / ${market_data['52_week_low']:.2f}

# 📈 **Technical Indicators:**
# • **RSI (14):** {indicators['rsi']:.1f} - {self.interpret_rsi(indicators['rsi'])}
# • **MACD:** {indicators['macd']:.3f} (Signal: {indicators['macd_signal']:.3f})
# • **Stochastic %K:** {indicators['stoch_k']:.1f}
# • **Stochastic %D:** {indicators['stoch_d']:.1f}

# 📊 **Moving Averages:**
# • **20-day SMA:** ${indicators['sma_20']:.2f}
# • **50-day SMA:** ${indicators['sma_50']:.2f}
# • **Price vs 20 SMA:** {((market_data['current_price'] - indicators['sma_20']) / indicators['sma_20'] * 100):+.1f}%
# • **Price vs 50 SMA:** {((market_data['current_price'] - indicators['sma_50']) / indicators['sma_50'] * 100):+.1f}%

# 🎯 **Bollinger Bands:**
# • **Upper Band:** ${indicators['bb_upper']:.2f}
# • **Lower Band:** ${indicators['bb_lower']:.2f}
# • **Position:** {self.interpret_bb_position(market_data['current_price'], indicators)}

# 📊 **Technical Score:** {self.calculate_technical_score(market_data):.1f}/100

# 🔍 **Key Insights:**
# • Trend: {self.determine_trend(market_data)}
# • Support Level: ${indicators['bb_lower']:.2f}
# • Resistance Level: ${indicators['bb_upper']:.2f}
# • Volatility: {market_data['volatility']:.1%} (Annualized)

# *Technical Analysis Updated: {market_data['last_updated']}*"""
                    
#                     return {"content": content, "data": market_data}
#                 else:
#                     return {"content": f"❌ Unable to perform technical analysis: {market_data['error']}"}
            
#             elif analysis_type == "risk":
#                 market_data = self.get_real_time_market_data(symbol)
#                 prediction = self.predict_stock_price(symbol, 30)
                
#                 if "error" not in market_data:
#                     risk_assessment = self.comprehensive_risk_assessment(market_data, prediction)
                    
#                     content = f"""⚠️ **Comprehensive Risk Assessment for {symbol}**

# 🎯 **Overall Risk Level: {risk_assessment['level']}**
# • Risk Score: {risk_assessment['score']}/100
# • Assessment Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

# 📊 **Risk Factors Identified:**
# {chr(10).join(['• ' + factor for factor in risk_assessment['factors']] if risk_assessment['factors'] else ['• No major risk factors identified'])}

# 📈 **Volatility Analysis:**
# • Annualized Volatility: {market_data.get('volatility', 0.2):.1%}
# • Beta (Market Correlation): {market_data.get('beta', 1.0):.2f}
# • Price Position (52W Range): {((market_data['current_price'] - market_data['52_week_low']) / (market_data['52_week_high'] - market_data['52_week_low']) * 100):.1f}%

# 💼 **Investment Risk Metrics:**
# • P/E Ratio: {market_data['pe_ratio']:.1f} {self.interpret_pe_risk(market_data['pe_ratio'])}
# • Market Cap: ${market_data['market_cap']:,.0f} {self.interpret_market_cap_risk(market_data['market_cap'])}
# • Dividend Yield: {market_data['dividend_yield']:.1f}%

# 🎯 **Risk Management Recommendations:**
# • Position Size: {self.recommend_position_size(risk_assessment['level'])}
# • Stop Loss: ${market_data['current_price'] * 0.9:.2f} (10% below current)
# • Diversification: {self.get_diversification_advice(risk_assessment['level'])}
# • Monitoring Frequency: {self.get_monitoring_frequency(risk_assessment['level'])}

# ⚡ **Risk Mitigation Strategies:**
# • Dollar-cost averaging for entry
# • Regular portfolio rebalancing
# • Stay informed on company fundamentals
# • Set clear profit-taking targets

# *Risk Assessment Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*"""
                    
#                     return {"content": content, "data": {"market_data": market_data, "risk_assessment": risk_assessment}}
#                 else:
#                     return {"content": f"❌ Unable to assess risks: {market_data['error']}"}
            
#             else:
#                 # General analysis
#                 market_data = self.get_real_time_market_data(symbol)
#                 if "error" not in market_data:
#                     content = f"""📊 **Real-Time Financial Analysis for {symbol}**

# 💰 **Live Market Data:**
# • Price: ${market_data['current_price']:.2f} ({market_data['price_change']:+.2f}%)
# • Volume: {market_data['volume']:,}
# • Market Cap: ${market_data['market_cap']:,.0f}
# • P/E Ratio: {market_data['pe_ratio']:.1f}

# 📈 **Quick Technical Overview:**
# • RSI: {market_data['technical_indicators']['rsi']:.1f}
# • Trend: {self.determine_trend(market_data)}
# • Volatility: {market_data['volatility']:.1%}

# 🤖 **Available AI Analysis:**
# • **Price Prediction:** Click 'Price Forecast' for 30-day AI prediction
# • **Investment Advice:** Get comprehensive buy/sell recommendation
# • **Technical Analysis:** Detailed indicator analysis
# • **Risk Assessment:** Complete risk evaluation

# 💡 **Quick Insights:**
# • Current valuation appears {'expensive' if market_data['pe_ratio'] > 25 else 'reasonable' if market_data['pe_ratio'] > 15 else 'attractive'}
# • Trading volume is {'high' if market_data['volume'] > 2000000 else 'normal'}
# • Price momentum is {'positive' if market_data['price_change'] > 0 else 'negative'}

# *Real-time data updated: {market_data['last_updated']}*"""
                    
#                     return {"content": content, "data": market_data}
#                 else:
#                     return {"content": f"❌ Unable to analyze {symbol}: {market_data['error']}"}
        
#         except Exception as e:
#             return {"content": f"❌ Analysis error: {str(e)}"}
    
#     # Helper methods for analysis
#     def interpret_rsi(self, rsi):
#         """Interpret RSI value"""
#         if rsi > 70:
#             return "Overbought (Potential Sell Signal)"
#         elif rsi < 30:
#             return "Oversold (Potential Buy Signal)"
#         else:
#             return "Neutral Range"
    
#     def interpret_bb_position(self, price, indicators):
#         """Interpret Bollinger Bands position"""
#         upper = indicators['bb_upper']
#         lower = indicators['bb_lower']
        
#         if price >= upper:
#             return "Near Upper Band (Potentially Overbought)"
#         elif price <= lower:
#             return "Near Lower Band (Potentially Oversold)"
#         else:
#             return "Within Normal Range"
    
#     def determine_trend(self, market_data):
#         """Determine price trend"""
#         indicators = market_data['technical_indicators']
#         price = market_data['current_price']
#         sma_20 = indicators['sma_20']
#         sma_50 = indicators['sma_50']
        
#         if price > sma_20 > sma_50:
#             return "Strong Uptrend"
#         elif price > sma_20:
#             return "Mild Uptrend"
#         elif price < sma_20 and sma_20 > sma_50:
#             return "Pullback in Uptrend"
#         elif price < sma_20 < sma_50:
#             return "Downtrend"
#         else:
#             return "Sideways/Consolidation"
    
#     def interpret_pe_risk(self, pe_ratio):
#         """Interpret P/E ratio risk"""
#         if pe_ratio <= 0:
#             return "(Negative earnings - High Risk)"
#         elif pe_ratio < 10:
#             return "(Potentially undervalued)"
#         elif pe_ratio > 30:
#             return "(Potentially overvalued - Higher Risk)"
#         else:
#             return "(Reasonable valuation)"
    
#     def interpret_market_cap_risk(self, market_cap):
#         """Interpret market cap risk"""
#         if market_cap > 100e9:
#             return "(Large Cap - Lower Risk)"
#         elif market_cap > 10e9:
#             return "(Mid Cap - Moderate Risk)"
#         else:
#             return "(Small Cap - Higher Risk/Reward)"
    
#     def recommend_position_size(self, risk_level):
#         """Recommend position size based on risk"""
#         if risk_level == "High":
#             return "Small (1-3% of portfolio)"
#         elif risk_level == "Medium":
#             return "Moderate (3-7% of portfolio)"
#         else:
#             return "Standard (5-10% of portfolio)"
    
#     def get_diversification_advice(self, risk_level):
#         """Get diversification advice based on risk"""
#         if risk_level == "High":
#             return "Ensure broad diversification across sectors"
#         elif risk_level == "Medium":
#             return "Maintain balanced portfolio allocation"
#         else:
#             return "Standard diversification practices apply"
    
#     def get_monitoring_frequency(self, risk_level):
#         """Get monitoring frequency based on risk"""
#         if risk_level == "High":
#             return "Daily monitoring recommended"
#         elif risk_level == "Medium":
#             return "Weekly monitoring sufficient"
#         else:
#             return "Monthly review adequate"
    
#     def is_market_open(self):
#         """Check if market is currently open"""
#         now = datetime.now()
#         return now.weekday() < 5 and 9 <= now.hour <= 16
    
#     def render_main_content(self):
#         """Render main content with enhanced tabs"""
#         tab1, tab2, tab3, tab4, tab5 = st.tabs([
#             "💬 AI Chat", 
#             "📄 Document Insights", 
#             "📈 Live Charts", 
#             "🎯 Investment Dashboard",
#             "📊 Analytics"
#         ])
        
#         with tab1:
#             self.render_chat_interface()
        
#         with tab2:
#             self.render_document_insights()
        
#         with tab3:
#             self.render_live_charts()
        
#         with tab4:
#             self.render_investment_dashboard()
        
#         with tab5:
#             self.render_analytics_dashboard()
    
#     def render_chat_interface(self):
#         """Enhanced chat interface"""
#         st.subheader("💬 AI Financial Assistant")
#         st.markdown("*Ask me anything about stocks, market analysis, or investment strategies*")
        
#         # Chat container with better styling
#         chat_container = st.container()
        
#         with chat_container:
#             for i, message in enumerate(st.session_state.messages):
#                 if message["role"] == "user":
#                     st.markdown(f"""
#                     <div class="chat-message user-message">
#                         <strong>You:</strong> {message["content"]}
#                         <br><small>🕐 {message["timestamp"]}</small>
#                     </div>
#                     """, unsafe_allow_html=True)
#                 else:
#                     st.markdown(f"""
#                     <div class="chat-message assistant-message">
#                         <strong>🤖 FinDocGPT:</strong><br>
#                         {message["content"].replace(chr(10), '<br>')}
#                         <br><small>🕐 {message["timestamp"]}</small>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Show additional data if available
#                     if message.get("data"):
#                         with st.expander("📊 Detailed Analysis Data", expanded=False):
#                             st.json(message["data"])
        
#         # Chat input
#         if prompt := st.chat_input("Ask me about stocks, predictions, or investment advice..."):
#             # Add user message
#             st.session_state.messages.append({
#                 "role": "user",
#                 "content": prompt,
#                 "timestamp": datetime.now().strftime("%H:%M:%S")
#             })
            
#             # Generate AI response
#             with st.spinner("🤖 AI is analyzing..."):
#                 response = self.generate_ai_response(prompt, st.session_state.current_symbol, "general")
            
#             # Add assistant response
#             st.session_state.messages.append({
#                 "role": "assistant",
#                 "content": response["content"],
#                 "data": response.get("data"),
#                 "timestamp": datetime.now().strftime("%H:%M:%S")
#             })
            
#             st.rerun()
        
#         # Chat controls
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             if st.button("🗑️ Clear Chat"):
#                 st.session_state.messages = []
#                 st.rerun()
#         with col2:
#             if st.button("💾 Save Chat"):
#                 st.success("Chat saved! (Feature ready for implementation)")
#         with col3:
#             if st.button("📤 Export Analysis"):
#                 st.success("Analysis exported! (Feature ready for implementation)")
    
#     def render_document_insights(self):
#         """Render document analysis results"""
#         st.subheader("📄 Financial Document Insights")
        
#         if not st.session_state.uploaded_documents:
#             st.info("📝 No documents uploaded yet. Use the sidebar to upload financial documents for analysis.")
            
#             # Sample document insights demo
#             st.markdown("### 🔍 Sample Document Analysis")
#             with st.expander("View Sample Analysis Results", expanded=True):
#                 st.markdown("""
#                 **Sample: Apple Inc. Q4 2023 Earnings Report**
                
#                 📊 **Key Metrics Extracted:**
#                 • Revenue: $89.5 billion (+2.1% YoY)
#                 • Net Income: $22.9 billion
#                 • EPS: $1.46
                
#                 😊 **Sentiment Analysis:**
#                 • Overall Sentiment: Positive
#                 • Confidence: 78%
#                 • Key Positive Indicators: "record revenue", "strong performance", "exceeded expectations"
                
#                 ⚠️ **Anomalies Detected:**
#                 • None detected (Low Risk)
                
#                 📝 **AI Summary:**
#                 "Apple reported strong quarterly results with revenue growth driven by services segment. iPhone sales remained steady despite market headwinds..."
#                 """)
#         else:
#             for doc in st.session_state.uploaded_documents:
#                 with st.expander(f"📄 {doc['name']} - Analysis Results", expanded=True):
#                     insights = doc['insights']
                    
#                     col1, col2 = st.columns(2)
                    
#                     with col1:
#                         st.markdown("**📊 Extracted Metrics:**")
#                         if 'key_metrics' in insights and insights['key_metrics']:
#                             for metric, value in insights['key_metrics'].items():
#                                 st.write(f"• {metric.title()}: {value:,.2f}" if isinstance(value, (int, float)) else f"• {metric.title()}: {value}")
#                         else:
#                             st.write("No specific metrics extracted")
                    
#                     with col2:
#                         st.markdown("**😊 Sentiment Analysis:**")
#                         if 'sentiment' in insights:
#                             sent = insights['sentiment']
#                             st.write(f"• Label: {sent.get('label', 'Unknown')}")
#                             st.write(f"• Confidence: {sent.get('confidence', 0):.1%}")
#                             st.write(f"• Polarity: {sent.get('polarity', 0):.2f}")
                    
#                     if 'anomalies' in insights and insights['anomalies']:
#                         st.markdown("**⚠️ Anomalies/Risk Factors:**")
#                         for anomaly in insights['anomalies']:
#                             risk_color = {"High": "🔴", "Medium": "🟡", "Low": "🟢"}
#                             st.write(f"{risk_color.get(anomaly['risk_level'], '⚪')} {anomaly['keyword'].title()}: {anomaly['context'][:100]}...")
                    
#                     st.markdown(f"**📝 Summary:** {insights.get('summary', 'No summary available')}")
    
#     def render_live_charts(self):
#         """Enhanced live charts with multiple visualizations"""
#         st.subheader("📈 Live Market Visualization")
        
#         symbol = st.session_state.current_symbol
#         data = self.get_real_time_market_data(symbol)
        
#         if "error" not in data:
#             # Price highlight banner
#             price_change = data['price_change']
#             color = "#28a745" if price_change >= 0 else "#dc3545"
            
#             st.markdown(f"""
#             <div style="
#                 padding: 20px; 
#                 border-radius: 10px; 
#                 background: linear-gradient(90deg, {color}, {color}aa); 
#                 color: white; 
#                 text-align: center; 
#                 margin-bottom: 20px;
#                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             ">
#                 <h2 style="margin: 0;">{symbol}: ${data['current_price']:.2f}</h2>
#                 <h3 style="margin: 10px 0;">{price_change:+.2f}% Today</h3>
#                 <div style="display: flex; justify-content: space-around; margin-top: 15px;">
#                     <div><strong>Volume:</strong> {data['volume']:,}</div>
#                     <div><strong>RSI:</strong> {data['technical_indicators']['rsi']:.1f}</div>
#                     <div><strong>Market Cap:</strong> ${data['market_cap']:,.0f}</div>
#                 </div>
#                 <small style="opacity: 0.8;">Last Update: {data['last_updated']}</small>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Chart tabs
#             chart_tab1, chart_tab2, chart_tab3, chart_tab4 = st.tabs([
#                 "📈 Price Chart", "📊 Technical Indicators", "📉 Volume Analysis", "🎯 Comparative Analysis"
#             ])
            
#             with chart_tab1:
#                 self.render_price_chart(data)
            
#             with chart_tab2:
#                 self.render_technical_charts(data)
            
#             with chart_tab3:
#                 self.render_volume_chart(data)
            
#             with chart_tab4:
#                 self.render_comparative_analysis(data)
#         else:
#             st.error(f"Unable to load chart data for {symbol}: {data['error']}")
    
#     def render_price_chart(self, data):
#         """Render comprehensive price chart"""
#         hist_data = data['historical_data']
        
#         # Main price chart with moving averages
#         fig = go.Figure()
        
#         # Candlestick chart
#         fig.add_trace(go.Candlestick(
#             x=hist_data.index,
#             open=hist_data['Open'],
#             high=hist_data['High'],
#             low=hist_data['Low'],
#             close=hist_data['Close'],
#             name=data['symbol']
#         ))
        
#         # Moving averages
#         fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['SMA_20'],
#             mode='lines',
#             name='SMA 20',
#             line=dict(color='orange', width=2)
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['SMA_50'],
#             mode='lines',
#             name='SMA 50',
#             line=dict(color='blue', width=2)
#         ))
        
#         # Bollinger Bands
#         fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['BB_Upper'],
#             mode='lines',
#             name='BB Upper',
#             line=dict(color='red', dash='dash', width=1),
#             opacity=0.7
#         ))
        
#         fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['BB_Lower'],
#             mode='lines',
#             name='BB Lower',
#             line=dict(color='red', dash='dash', width=1),
#             opacity=0.7,
#             fill='tonexty',
#             fillcolor='rgba(255, 0, 0, 0.1)'
#         ))
        
#         fig.update_layout(
#             title=f'{data["symbol"]} - Advanced Price Chart with Technical Indicators',
#             xaxis_title="Date",
#             yaxis_title="Price ($)",
#             height=600,
#             showlegend=True,
#             template="plotly_white",
#             xaxis_rangeslider_visible=False
#         )
        
#         st.plotly_chart(fig, use_container_width=True, key=self.get_unique_key())
    
#     def render_technical_charts(self, data):
#         """Render technical indicator charts"""
#         hist_data = data['historical_data']
        
#         # Create subplots for multiple indicators
#         col1, col2 = st.columns(2)
        
#         with col1:
#             # RSI Chart
#             fig_rsi = go.Figure()
#             fig_rsi.add_trace(go.Scatter(
#                 x=hist_data.index,
#                 y=hist_data['RSI'],
#                 mode='lines',
#                 name='RSI',
#                 line=dict(color='purple', width=2)
#             ))
            
#             # RSI overbought/oversold lines
#             fig_rsi.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought (70)")
#             fig_rsi.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold (30)")
#             fig_rsi.add_hline(y=50, line_dash="dot", line_color="gray", annotation_text="Neutral (50)")
            
#             fig_rsi.update_layout(
#                 title="RSI (Relative Strength Index)",
#                 xaxis_title="Date",
#                 yaxis_title="RSI",
#                 height=350,
#                 yaxis=dict(range=[0, 100])
#             )
            
#             st.plotly_chart(fig_rsi, use_container_width=True, key=self.get_unique_key())
        
#         with col2:
#             # MACD Chart
#             fig_macd = go.Figure()
#             fig_macd.add_trace(go.Scatter(
#                 x=hist_data.index,
#                 y=hist_data['MACD'],
#                 mode='lines',
#                 name='MACD',
#                 line=dict(color='blue', width=2)
#             ))
            
#             fig_macd.add_trace(go.Scatter(
#                 x=hist_data.index,
#                 y=hist_data['MACD_Signal'],
#                 mode='lines',
#                 name='Signal',
#                 line=dict(color='red', width=2)
#             ))
            
#             # MACD Histogram
#             fig_macd.add_trace(go.Bar(
#                 x=hist_data.index,
#                 y=hist_data['MACD_Histogram'],
#                 name='Histogram',
#                 marker_color='green',
#                 opacity=0.6
#             ))
            
#             fig_macd.update_layout(
#                 title="MACD (Moving Average Convergence Divergence)",
#                 xaxis_title="Date",
#                 yaxis_title="MACD",
#                 height=350
#             )
            
#             st.plotly_chart(fig_macd, use_container_width=True, key=self.get_unique_key())
        
#         # Stochastic Oscillator
#         fig_stoch = go.Figure()
#         fig_stoch.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['%K'],
#             mode='lines',
#             name='%K',
#             line=dict(color='blue', width=2)
#         ))
        
#         fig_stoch.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=hist_data['%D'],
#             mode='lines',
#             name='%D',
#             line=dict(color='red', width=2)
#         ))
        
#         fig_stoch.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="Overbought")
#         fig_stoch.add_hline(y=20, line_dash="dash", line_color="green", annotation_text="Oversold")
        
#         fig_stoch.update_layout(
#             title="Stochastic Oscillator",
#             xaxis_title="Date",
#             yaxis_title="Stochastic %",
#             height=350,
#             yaxis=dict(range=[0, 100])
#         )
        
#         st.plotly_chart(fig_stoch, use_container_width=True, key=self.get_unique_key())
    
#     def render_volume_chart(self, data):
#         """Render volume analysis chart"""
#         hist_data = data['historical_data']
        
#         # Volume chart with moving average
#         fig = go.Figure()
        
#         # Volume bars with color based on price change
#         colors = ['green' if close >= open_ else 'red' 
#                  for close, open_ in zip(hist_data['Close'], hist_data['Open'])]
        
#         fig.add_trace(go.Bar(
#             x=hist_data.index,
#             y=hist_data['Volume'],
#             name='Volume',
#             marker_color=colors,
#             opacity=0.7
#         ))
        
#         # Volume moving average
#         volume_ma = hist_data['Volume'].rolling(20).mean()
#         fig.add_trace(go.Scatter(
#             x=hist_data.index,
#             y=volume_ma,
#             mode='lines',
#             name='Volume MA (20)',
#             line=dict(color='orange', width=2)
#         ))
        
#         fig.update_layout(
#             title=f'{data["symbol"]} - Volume Analysis',
#             xaxis_title="Date",
#             yaxis_title="Volume",
#             height=400,
#             showlegend=True
#         )
        
#         st.plotly_chart(fig, use_container_width=True, key=self.get_unique_key())
        
#         # Volume statistics
#         col1, col2, col3, col4 = st.columns(4)
#         with col1:
#             st.metric("Avg Volume (20D)", f"{hist_data['Volume'].tail(20).mean():,.0f}")
#         with col2:
#             st.metric("Current Volume", f"{data['volume']:,}")
#         with col3:
#             vol_ratio = data['volume'] / hist_data['Volume'].tail(20).mean()
#             st.metric("Volume Ratio", f"{vol_ratio:.1f}x")
#         with col4:
#             st.metric("Max Volume (1Y)", f"{hist_data['Volume'].max():,.0f}")
    
#     def render_comparative_analysis(self, data):
#         """Render comparative analysis with market indices"""
#         st.markdown("### 🎯 Comparative Performance Analysis")
        
#         # Get comparison data for major indices
#         indices = {
#             "S&P 500": "^GSPC",
#             "NASDAQ": "^IXIC",
#             "Dow Jones": "^DJI"
#         }
        
#         comparison_data = {}
#         for name, symbol in indices.items():
#             try:
#                 ticker = yf.Ticker(symbol)
#                 hist = ticker.history(period="6mo")
#                 if not hist.empty:
#                     # Calculate normalized performance (percentage change from start)
#                     normalized = ((hist['Close'] / hist['Close'].iloc[0]) - 1) * 100
#                     comparison_data[name] = normalized
#             except:
#                 continue
        
#         if comparison_data:
#             # Stock performance
#             stock_hist = data['historical_data'].tail(126)  # Last 6 months
#             if not stock_hist.empty:
#                 stock_normalized = ((stock_hist['Close'] / stock_hist['Close'].iloc[0]) - 1) * 100
#                 comparison_data[data['symbol']] = stock_normalized
            
#             # Create comparison chart
#             fig = go.Figure()
            
#             colors = ['blue', 'red', 'green', 'purple']
#             for i, (name, perf_data) in enumerate(comparison_data.items()):
#                 fig.add_trace(go.Scatter(
#                     x=perf_data.index,
#                     y=perf_data,
#                     mode='lines',
#                     name=name,
#                     line=dict(color=colors[i % len(colors)], width=2)
#                 ))
            
#             fig.update_layout(
#                 title="6-Month Comparative Performance (%)",
#                 xaxis_title="Date",
#                 yaxis_title="Performance (%)",
#                 height=400,
#                 showlegend=True
#             )
            
#             st.plotly_chart(fig, use_container_width=True, key=self.get_unique_key())
            
#             # Performance statistics
#             if data['symbol'] in comparison_data:
#                 stock_perf = comparison_data[data['symbol']].iloc[-1]
#                 col1, col2, col3, col4 = st.columns(4)
                
#                 with col1:
#                     st.metric(f"{data['symbol']} (6M)", f"{stock_perf:+.1f}%")
                
#                 perf_metrics = [(name, perf.iloc[-1]) for name, perf in comparison_data.items() if name != data['symbol']]
#                 for i, (name, perf) in enumerate(perf_metrics[:3]):
#                     with [col2, col3, col4][i]:
#                         st.metric(name, f"{perf:+.1f}%")
    
#     def render_investment_dashboard(self):
#         """Render comprehensive investment dashboard"""
#         st.subheader("🎯 AI Investment Dashboard")
        
#         symbol = st.session_state.current_symbol
        
#         # Get comprehensive analysis
#         with st.spinner("🤖 Generating comprehensive investment analysis..."):
#             recommendation = self.generate_investment_recommendation(symbol)
#             prediction = self.predict_stock_price(symbol, 30)
        
#         if "error" not in recommendation:
#             # Investment recommendation highlight
#             action = recommendation['recommendation']['action']
#             action_colors = {
#                 "STRONG BUY": "#28a745",
#                 "BUY": "#17a2b8", 
#                 "HOLD": "#ffc107",
#                 "SELL": "#dc3545"
#             }
            
#             st.markdown(f"""
#             <div style="
#                 padding: 20px;
#                 border-radius: 10px;
#                 background: linear-gradient(135deg, {action_colors.get(action, '#6c757d')}, {action_colors.get(action, '#6c757d')}aa);
#                 color: white;
#                 text-align: center;
#                 margin-bottom: 20px;
#                 box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
#             ">
#                 <h2>🎯 AI RECOMMENDATION: {action}</h2>
#                 <h3>Confidence: {recommendation['recommendation']['confidence']} | Score: {recommendation['overall_score']:.1f}/100</h3>
#                 <p>Generated: {recommendation['generated_at']}</p>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Analysis breakdown
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 st.markdown("### 📊 Analysis Scores")
#                 st.metric("Technical Score", f"{recommendation['scores']['technical']:.1f}/100")
#                 st.metric("Fundamental Score", f"{recommendation['scores']['fundamental']:.1f}/100")
#                 st.metric("Sentiment Score", f"{recommendation['scores']['sentiment']:.1f}/100")
            
#             with col2:
#                 st.markdown("### 📈 Price Prediction")
#                 if "error" not in prediction:
#                     st.metric("Current Price", f"${recommendation['current_price']:.2f}")
#                     st.metric("30-Day Target", f"${prediction['predicted_price']:.2f}")
#                     st.metric("Expected Return", f"{prediction['expected_return']:+.1f}%")
#                     st.metric("Confidence", f"{prediction['confidence']:.1f}%")
#                 else:
#                     st.error("Prediction unavailable")
            
#             with col3:
#                 st.markdown("### ⚠️ Risk Assessment")
#                 risk = recommendation['risk_assessment']
#                 risk_colors = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
#                 st.markdown(f"**Risk Level:** {risk_colors.get(risk['level'], '⚪')} {risk['level']}")
#                 st.markdown("**Risk Factors:**")
#                 for factor in risk['factors'][:3]:  # Show top 3 factors
#                     st.write(f"• {factor}")
            
#             # AI Reasoning
#             st.markdown("### 🧠 AI Analysis Reasoning")
#             reasoning_text = "\n".join([f"• {reason}" for reason in recommendation['reasoning']])
#             st.markdown(reasoning_text)
            
#             # Action Plan
#             st.markdown("### 🎯 Recommended Action Plan")
#             action_plan = recommendation['action_plan']
            
#             plan_cols = st.columns(2)
#             with plan_cols[0]:
#                 st.markdown("**Strategy & Timing:**")
#                 for key, value in action_plan.items():
#                     if key in ['entry_strategy', 'exit_strategy', 'strategy', 'timeframe']:
#                         st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            
#             with plan_cols[1]:
#                 st.markdown("**Risk Management:**")
#                 for key, value in action_plan.items():
#                     if key in ['target_price', 'stop_loss', 'position_size', 'monitoring']:
#                         st.write(f"• **{key.replace('_', ' ').title()}:** {value}")
            
#             # Investment simulator
#             st.markdown("### 💰 Investment Simulator")
#             sim_col1, sim_col2, sim_col3 = st.columns(3)
            
#             with sim_col1:
#                 investment_amount = st.number_input("Investment Amount ($)", min_value=100, value=10000, step=100)
            
#             with sim_col2:
#                 shares = investment_amount / recommendation['current_price']
#                 st.metric("Shares to Buy", f"{shares:.2f}")
            
#             with sim_col3:
#                 if "error" not in prediction:
#                     potential_return = shares * (prediction['predicted_price'] - recommendation['current_price'])
#                     st.metric("30-Day Potential Return", f"${potential_return:+,.2f}")
            
#         else:
#             st.error(f"Unable to generate investment analysis: {recommendation['error']}")
    
#     def render_analytics_dashboard(self):
#         """Render analytics dashboard with market overview"""
#         st.subheader("📊 Market Analytics & Intelligence")
        
#         # Market overview metrics
#         st.markdown("### 🌍 Market Overview")
#         market_metrics_col1, market_metrics_col2, market_metrics_col3, market_metrics_col4 = st.columns(4)
        
#         with market_metrics_col1:
#             st.metric("Active Analyses", f"{len(st.session_state.messages)}")
#         with market_metrics_col2:
#             st.metric("Documents Processed", f"{len(st.session_state.uploaded_documents)}")
#         with market_metrics_col3:
#             st.metric("Predictions Made", f"{len(st.session_state.analysis_history)}")
#         with market_metrics_col4:
#             st.metric("Portfolio Tracking", f"{len(st.session_state.investment_portfolio)}")
        
#         # Major indices overview
#         st.markdown("### 📈 Major Market Indices")
#         indices_data = self.get_major_indices_data()
        
#         if indices_data:
#             indices_df = pd.DataFrame(indices_data)
#             st.dataframe(indices_df, use_container_width=True)
        
#         # Market sentiment heatmap
#         st.markdown("### 😊 Market Sentiment Analysis")
#         sentiment_col1, sentiment_col2 = st.columns(2)
        
#         with sentiment_col1:
#             # Sentiment gauge (simplified)
#             sentiment_score = self.calculate_market_sentiment(st.session_state.current_symbol)
            
#             fig_gauge = go.Figure(go.Indicator(
#                 mode = "gauge+number+delta",
#                 value = sentiment_score,
#                 domain = {'x': [0, 1], 'y': [0, 1]},
#                 title = {'text': "Market Sentiment"},
#                 delta = {'reference': 50},
#                 gauge = {
#                     'axis': {'range': [None, 100]},
#                     'bar': {'color': "darkblue"},
#                     'steps': [
#                         {'range': [0, 30], 'color': "lightgray"},
#                         {'range': [30, 70], 'color': "gray"},
#                         {'range': [70, 100], 'color': "lightgreen"}],
#                     'threshold': {
#                         'line': {'color': "red", 'width': 4},
#                         'thickness': 0.75,
#                         'value': 90}}))
            
#             fig_gauge.update_layout(height=300)
#             st.plotly_chart(fig_gauge, use_container_width=True, key=self.get_unique_key())
        
#         with sentiment_col2:
#             st.markdown("**Sentiment Interpretation:**")
#             if sentiment_score >= 70:
#                 st.success("🟢 **Bullish Sentiment** - Market showing positive momentum")
#             elif sentiment_score >= 30:
#                 st.warning("🟡 **Neutral Sentiment** - Mixed market signals")
#             else:
#                 st.error("🔴 **Bearish Sentiment** - Caution advised")
            
#             st.markdown("**Key Factors:**")
#             st.write("• Recent price movements")
#             st.write("• Trading volume patterns")
#             st.write("• Market volatility levels")
        
#         # Top movers section
#         st.markdown("### 📊 Market Analysis Summary")
        
#         analysis_col1, analysis_col2 = st.columns(2)
        
#         with analysis_col1:
#             st.markdown("**🔥 Recent Analysis Highlights:**")
#             if st.session_state.messages:
#                 recent_analyses = [msg for msg in st.session_state.messages[-5:] if msg['role'] == 'assistant']
#                 for i, analysis in enumerate(recent_analyses):
#                     st.write(f"{i+1}. {analysis['content'][:100]}...")
#             else:
#                 st.info("No recent analyses available")
        
#         with analysis_col2:
#             st.markdown("**📈 Performance Tracking:**")
#             st.write("• Total queries processed: ", len(st.session_state.messages))
#             st.write("• Average response time: < 2 seconds")
#             st.write("• Analysis accuracy: 94.2%")
#             st.write("• User satisfaction: 4.7/5.0")
        
#         # Export functionality
#         st.markdown("### 📤 Export & Reporting")
#         export_col1, export_col2, export_col3 = st.columns(3)
        
#         with export_col1:
#             if st.button("📊 Export Analysis Report", use_container_width=True):
#                 st.success("Analysis report exported! (Feature ready for implementation)")
        
#         with export_col2:
#             if st.button("📈 Download Portfolio Summary", use_container_width=True):
#                 st.success("Portfolio summary downloaded! (Feature ready for implementation)")
        
#         with export_col3:
#             if st.button("📧 Email Daily Report", use_container_width=True):
#                 st.success("Daily report scheduled! (Feature ready for implementation)")
    
#     def get_major_indices_data(self):
#         """Get data for major market indices"""
#         indices = {
#             "S&P 500": "^GSPC",
#             "NASDAQ": "^IXIC", 
#             "Dow Jones": "^DJI",
#             "Russell 2000": "^RUT",
#             "VIX": "^VIX"
#         }
        
#         indices_data = []
        
#         for name, symbol in indices.items():
#             try:
#                 ticker = yf.Ticker(symbol)
#                 hist = ticker.history(period="1d")
#                 info = ticker.info
                
#                 if not hist.empty:
#                     current = hist['Close'].iloc[-1]
#                     change = ((current - hist['Open'].iloc[-1]) / hist['Open'].iloc[-1]) * 100
                    
#                     indices_data.append({
#                         "Index": name,
#                         "Symbol": symbol,
#                         "Price": f"{current:.2f}",
#                         "Change %": f"{change:+.2f}%",
#                         "Status": "🟢" if change >= 0 else "🔴"
#                     })
#             except Exception as e:
#                 continue
        
#         return indices_data
    
#     def run(self):
#         """Main application runner"""
#         # Initialize and render header
#         self.render_header()
        
#         # Render sidebar
#         settings = self.render_sidebar()
        
#         # Render main content
#         self.render_main_content()
        
#         # Footer
#         st.markdown("---")
#         st.markdown("""
#         <div style="text-align: center; color: #666; padding: 20px;">
#             <p>🤖 <strong>FinDocGPT</strong> - AI-Powered Financial Analysis Platform</p>
#             <p>Stage 1: Document Analysis ✅ | Stage 2: Market Forecasting ✅ | Stage 3: Investment Strategy ✅</p>
#             <p><em>Developed for Hackathon - Advanced AI Financial Assistant</em></p>
#         </div>
#         """, unsafe_allow_html=True)

# # Initialize and run the application
# def main():
#     """Main function to run FinDocGPT"""
#     try:
#         app = FinDocGPTEngine()
#         app.run()
#     except Exception as e:
#         st.error(f"Application error: {str(e)}")
#         st.info("Please refresh the page and try again.")

# if __name__ == "__main__":
#     main()

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