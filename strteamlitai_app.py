# # """
# # FinDocGPT - AI-Powered Financial Intelligence
# # Premium Professional Interface - AkashX.ai Hackathon Solution
# # """

# # import streamlit as st
# # import pandas as pd
# # import plotly.graph_objects as go
# # import plotly.express as px
# # from plotly.subplots import make_subplots
# # from datetime import datetime, timedelta
# # import yfinance as yf
# # import numpy as np
# # import time
# # import uuid
# # from openai import OpenAI
# # import os
# # from typing import Dict, List, Any
# # import json
# # from dataclasses import dataclass
# # import logging
# # import io
# # import re

# # # PDF processing imports
# # try:
# #     import PyPDF2
# #     import pdfplumber
# #     PDF_AVAILABLE = True
# # except ImportError:
# #     PDF_AVAILABLE = False

# # # Configure Streamlit
# # st.set_page_config(
# #     page_title="FinDocGPT - AI Financial Intelligence",
# #     page_icon="🤖",
# #     layout="wide",
# #     initial_sidebar_state="expanded"
# # )

# # # Premium CSS Styling
# # st.markdown("""
# # <style>
# # /* Import Google Fonts */
# # @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# # /* Global Styling */
# # .main {
# #     font-family: 'Inter', sans-serif;
# #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #     min-height: 100vh;
# # }

# # /* Header Styling */
# # .premium-header {
# #     background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
# #     padding: 2rem;
# #     border-radius: 20px;
# #     margin-bottom: 2rem;
# #     text-align: center;
# #     color: white;
# #     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
# #     position: relative;
# #     overflow: hidden;
# # }

# # .premium-header::before {
# #     content: '';
# #     position: absolute;
# #     top: 0;
# #     left: 0;
# #     right: 0;
# #     bottom: 0;
# #     background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
# #     opacity: 0.3;
# # }

# # .premium-header h1 {
# #     font-size: 3rem;
# #     font-weight: 700;
# #     margin: 0;
# #     text-shadow: 0 2px 4px rgba(0,0,0,0.3);
# #     position: relative;
# #     z-index: 1;
# # }

# # .premium-header h3 {
# #     font-size: 1.2rem;
# #     font-weight: 400;
# #     margin: 0.5rem 0;
# #     opacity: 0.9;
# #     position: relative;
# #     z-index: 1;
# # }

# # .premium-header p {
# #     font-size: 1rem;
# #     margin: 1rem 0 0 0;
# #     opacity: 0.8;
# #     position: relative;
# #     z-index: 1;
# # }

# # /* Status Bar */
# # .status-bar {
# #     background: rgba(255, 255, 255, 0.95);
# #     backdrop-filter: blur(10px);
# #     padding: 1rem;
# #     border-radius: 15px;
# #     margin-bottom: 2rem;
# #     box-shadow: 0 8px 32px rgba(0,0,0,0.1);
# #     border: 1px solid rgba(255,255,255,0.2);
# # }

# # .status-item {
# #     display: flex;
# #     align-items: center;
# #     gap: 0.5rem;
# #     font-weight: 500;
# #     color: #2d3748;
# # }

# # /* Card Styling */
# # .premium-card {
# #     background: rgba(255, 255, 255, 0.95);
# #     backdrop-filter: blur(20px);
# #     border-radius: 20px;
# #     padding: 2rem;
# #     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
# #     border: 1px solid rgba(255,255,255,0.2);
# #     margin-bottom: 2rem;
# #     transition: all 0.3s ease;
# # }

# # .premium-card:hover {
# #     transform: translateY(-5px);
# #     box-shadow: 0 25px 50px rgba(0,0,0,0.15);
# # }

# # /* Metric Cards */
# # .metric-card {
# #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #     color: white;
# #     padding: 1.5rem;
# #     border-radius: 15px;
# #     text-align: center;
# #     box-shadow: 0 10px 30px rgba(102,126,234,0.3);
# #     transition: all 0.3s ease;
# # }

# # .metric-card:hover {
# #     transform: translateY(-3px);
# #     box-shadow: 0 15px 40px rgba(102,126,234,0.4);
# # }

# # .metric-value {
# #     font-size: 2rem;
# #     font-weight: 700;
# #     margin-bottom: 0.5rem;
# # }

# # .metric-label {
# #     font-size: 0.9rem;
# #     opacity: 0.9;
# #     font-weight: 500;
# # }

# # .metric-change {
# #     font-size: 0.8rem;
# #     margin-top: 0.5rem;
# #     padding: 0.3rem 0.6rem;
# #     border-radius: 20px;
# #     background: rgba(255,255,255,0.2);
# # }

# # /* Sidebar Styling */
# # .css-1d391kg {
# #     background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
# # }

# # .sidebar-content {
# #     color: white;
# # }

# # /* Investment Score */
# # .score-excellent {
# #     background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
# # }

# # .score-good {
# #     background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
# # }

# # .score-fair {
# #     background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
# # }

# # .score-poor {
# #     background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
# # }

# # /* Analysis Box */
# # .analysis-box {
# #     background: rgba(255, 255, 255, 0.95);
# #     backdrop-filter: blur(20px);
# #     padding: 2rem;
# #     border-radius: 20px;
# #     border: 1px solid rgba(255,255,255,0.2);
# #     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
# #     margin: 2rem 0;
# # }

# # /* Buttons */
# # .stButton > button {
# #     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
# #     color: white;
# #     border: none;
# #     border-radius: 12px;
# #     padding: 0.75rem 2rem;
# #     font-weight: 600;
# #     font-size: 1rem;
# #     transition: all 0.3s ease;
# #     box-shadow: 0 4px 15px rgba(102,126,234,0.3);
# # }

# # .stButton > button:hover {
# #     transform: translateY(-2px);
# #     box-shadow: 0 8px 25px rgba(102,126,234,0.4);
# # }

# # /* Tabs */
# # .stTabs [data-baseweb="tab-list"] {
# #     background: rgba(255, 255, 255, 0.1);
# #     border-radius: 15px;
# #     padding: 0.5rem;
# #     backdrop-filter: blur(10px);
# # }

# # .stTabs [data-baseweb="tab"] {
# #     background: transparent;
# #     border-radius: 10px;
# #     color: white;
# #     font-weight: 500;
# #     padding: 1rem 2rem;
# #     transition: all 0.3s ease;
# # }

# # .stTabs [aria-selected="true"] {
# #     background: rgba(255, 255, 255, 0.2);
# #     color: white;
# # }

# # /* Charts */
# # .chart-container {
# #     background: rgba(255, 255, 255, 0.95);
# #     border-radius: 20px;
# #     padding: 2rem;
# #     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
# #     backdrop-filter: blur(20px);
# # }

# # /* File Upload */
# # .uploadedFile {
# #     background: rgba(255, 255, 255, 0.1);
# #     border: 2px dashed rgba(255, 255, 255, 0.3);
# #     border-radius: 15px;
# #     padding: 2rem;
# #     text-align: center;
# #     color: white;
# # }

# # /* Recommendations */
# # .recommendation-buy {
# #     background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
# #     color: white;
# #     padding: 1rem 2rem;
# #     border-radius: 15px;
# #     text-align: center;
# #     font-weight: 600;
# #     font-size: 1.1rem;
# #     box-shadow: 0 10px 30px rgba(72,187,120,0.3);
# # }

# # .recommendation-hold {
# #     background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
# #     color: white;
# #     padding: 1rem 2rem;
# #     border-radius: 15px;
# #     text-align: center;
# #     font-weight: 600;
# #     font-size: 1.1rem;
# #     box-shadow: 0 10px 30px rgba(237,137,54,0.3);
# # }

# # .recommendation-sell {
# #     background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
# #     color: white;
# #     padding: 1rem 2rem;
# #     border-radius: 15px;
# #     text-align: center;
# #     font-weight: 600;
# #     font-size: 1.1rem;
# #     box-shadow: 0 10px 30px rgba(245,101,101,0.3);
# # }

# # /* Progress Bars */
# # .stProgress .st-bo {
# #     background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
# #     height: 12px;
# #     border-radius: 6px;
# # }

# # /* Text Input */
# # .stTextInput > div > div > input {
# #     background: rgba(255, 255, 255, 0.1);
# #     border: 1px solid rgba(255, 255, 255, 0.2);
# #     border-radius: 10px;
# #     color: white;
# #     backdrop-filter: blur(10px);
# # }

# # /* Select Box */
# # .stSelectbox > div > div > select {
# #     background: rgba(255, 255, 255, 0.1);
# #     border: 1px solid rgba(255, 255, 255, 0.2);
# #     border-radius: 10px;
# #     color: white;
# #     backdrop-filter: blur(10px);
# # }

# # /* Hide Streamlit Branding */
# # #MainMenu {visibility: hidden;}
# # footer {visibility: hidden;}
# # header {visibility: hidden;}
# # </style>
# # """, unsafe_allow_html=True)

# # # Configuration
# # @dataclass
# # class Config:
# #     OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
# #     DEFAULT_MODEL: str = "gpt-4"
# #     MAX_TOKENS: int = 2000
# #     TEMPERATURE: float = 0.3

# # config = Config()

# # # Initialize OpenAI client
# # client = None
# # if config.OPENAI_API_KEY:
# #     try:
# #         client = OpenAI(api_key=config.OPENAI_API_KEY)
# #     except Exception as e:
# #         st.warning(f"⚠️ OpenAI initialization error: {str(e)}")
# #         client = None

# # # Session State Management
# # def init_session_state():
# #     defaults = {
# #         "selected_symbol": "AAPL",
# #         "analysis_history": [],
# #         "investment_portfolio": [],
# #         "portfolio_value": 100000,
# #         "current_analysis": None,
# #         "chart_counter": 0
# #     }
    
# #     for key, value in defaults.items():
# #         if key not in st.session_state:
# #             st.session_state[key] = value

# # def get_unique_key():
# #     st.session_state.chart_counter += 1
# #     return f"component_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"

# # # Enhanced OpenAI Handler
# # class OpenAIHandler:
# #     @staticmethod
# #     def generate_response(prompt: str, context: str = "", analysis_type: str = "general") -> str:
# #         if not client:
# #             return OpenAIHandler._fallback_analysis(prompt, analysis_type)
        
# #         system_prompts = {
# #             "document_qa": "You are FinDocGPT, a specialized financial document analyst. Provide precise, data-driven answers with specific financial metrics and insights.",
# #             "sentiment": "You are a market sentiment analyst. Analyze financial communications to determine market sentiment with confidence scores.",
# #             "forecasting": "You are a quantitative financial forecasting expert. Provide specific price targets with timeframes and probability ranges.",
# #             "investment": "You are a senior investment strategist. Provide clear BUY/SELL/HOLD recommendations with entry/exit points and risk management.",
# #             "general": "You are FinDocGPT, an AI financial intelligence assistant. Provide comprehensive financial analysis."
# #         }
        
# #         system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
# #         # Context length management
# #         max_context_chars = 6000
# #         if len(context) > max_context_chars:
# #             context = context[:max_context_chars] + "\n[Content truncated...]"
# #         if len(prompt) > max_context_chars:
# #             prompt = prompt[:max_context_chars] + "\n[Query truncated...]"
        
# #         try:
# #             response = client.chat.completions.create(
# #                 model=config.DEFAULT_MODEL,
# #                 messages=[
# #                     {"role": "system", "content": system_prompt},
# #                     {"role": "user", "content": f"Context: {context}\n\nRequest: {prompt}"}
# #                 ],
# #                 max_tokens=config.MAX_TOKENS,
# #                 temperature=config.TEMPERATURE
# #             )
# #             return response.choices[0].message.content
# #         except Exception as e:
# #             if "context_length_exceeded" in str(e).lower():
# #                 return "⚠️ Document too large for analysis. Please try with a smaller section."
# #             return f"⚠️ AI analysis error: {str(e)}"
    
# #     @staticmethod
# #     def _fallback_analysis(prompt: str, analysis_type: str) -> str:
# #         fallbacks = {
# #             "document_qa": "📊 Document analysis requires OpenAI API configuration.",
# #             "sentiment": "😊 Sentiment analysis shows neutral market conditions.",
# #             "forecasting": "📈 Basic forecasting suggests following current trends.",
# #             "investment": "💼 Conservative HOLD recommendation based on available data.",
# #             "general": "🤖 Limited analysis without OpenAI API key."
# #         }
# #         return fallbacks.get(analysis_type, fallbacks["general"])

# # # Enhanced Document Processor
# # class DocumentProcessor:
# #     @staticmethod
# #     def extract_document_text(uploaded_file) -> str:
# #         try:
# #             file_type = uploaded_file.type
# #             file_name = uploaded_file.name.lower()
            
# #             if file_type == "text/plain" or file_name.endswith('.txt'):
# #                 return DocumentProcessor._process_txt_file(uploaded_file)
# #             elif file_type == "application/pdf" or file_name.endswith('.pdf'):
# #                 if PDF_AVAILABLE:
# #                     return DocumentProcessor._process_pdf_file(uploaded_file)
# #                 else:
# #                     return "❌ PDF processing unavailable. Install: pip install PyPDF2 pdfplumber"
# #             elif file_type in ["text/csv", "application/vnd.ms-excel"] or file_name.endswith('.csv'):
# #                 return DocumentProcessor._process_csv_file(uploaded_file)
# #             else:
# #                 return f"❌ Unsupported file type: {file_type}"
# #         except Exception as e:
# #             return f"❌ Error processing file: {str(e)}"
    
# #     @staticmethod
# #     def _process_pdf_file(uploaded_file) -> str:
# #         try:
# #             pdf_bytes = uploaded_file.read()
# #             with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
# #                 text_pages = []
                
# #                 for i, page in enumerate(pdf.pages[:10]):  # Limit pages
# #                     page_text = page.extract_text()
# #                     if page_text:
# #                         text_pages.append(page_text[:1500])  # Limit page content
                
# #                 if text_pages:
# #                     full_text = "\n\n".join(text_pages)
# #                     if len(full_text) > 15000:
# #                         full_text = full_text[:15000] + "\n[Document truncated...]"
                    
# #                     financial_metrics = DocumentProcessor._extract_financial_metrics(full_text)
                    
# #                     return f"""📄 PDF Document Analysis:
# # 📊 File Statistics:
# # - Pages processed: {len(text_pages)}/{len(pdf.pages)}
# # - Characters: {len(full_text):,}

# # 💰 Financial Data:
# # {DocumentProcessor._format_financial_metrics(financial_metrics)}

# # 📝 Content:
# # {full_text}"""
# #                 else:
# #                     return "❌ Could not extract text from PDF."
# #         except Exception as e:
# #             return f"❌ PDF processing error: {str(e)}"
    
# #     @staticmethod
# #     def _process_txt_file(uploaded_file) -> str:
# #         try:
# #             encodings = ['utf-8', 'latin-1', 'cp1252']
# #             content = None
            
# #             for encoding in encodings:
# #                 try:
# #                     uploaded_file.seek(0)
# #                     content = uploaded_file.read().decode(encoding)
# #                     break
# #                 except UnicodeDecodeError:
# #                     continue
            
# #             if content is None:
# #                 return "❌ Could not decode text file."
            
# #             if len(content) > 15000:
# #                 content = content[:15000] + "\n[Content truncated...]"
            
# #             financial_metrics = DocumentProcessor._extract_financial_metrics(content)
            
# #             return f"""📄 TXT Document Analysis:
# # 📊 Statistics: {len(content):,} characters, {len(content.split()):,} words

# # 💰 Financial Data:
# # {DocumentProcessor._format_financial_metrics(financial_metrics)}

# # 📝 Content:
# # {content}"""
# #         except Exception as e:
# #             return f"❌ TXT processing error: {str(e)}"
    
# #     @staticmethod
# #     def _process_csv_file(uploaded_file) -> str:
# #         try:
# #             df = pd.read_csv(uploaded_file)
            
# #             if len(df) > 500:
# #                 df = df.head(500)
# #                 note = f"\n[Showing first 500 rows of {len(df)} total]"
# #             else:
# #                 note = ""
            
# #             financial_keywords = ['revenue', 'sales', 'income', 'profit', 'cost', 'price', 'amount', 'value']
# #             financial_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in financial_keywords)]
            
# #             return f"""📊 CSV Dataset Analysis:
# # 📈 Overview: {len(df)} rows, {len(df.columns)} columns
# # 💰 Financial columns: {', '.join(financial_cols[:5]) if financial_cols else 'None detected'}

# # 📋 Sample Data:
# # {df.head().to_string()}

# # 📝 Full Dataset:
# # {df.to_string()[:8000]}{note}"""
# #         except Exception as e:
# #             return f"❌ CSV processing error: {str(e)}"
    
# #     @staticmethod
# #     def _extract_financial_metrics(text: str) -> Dict[str, float]:
# #         metrics = {}
# #         patterns = {
# #             'revenue': r'(?:revenue|sales)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
# #             'net_income': r'net\s+(?:income|earnings)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
# #             'assets': r'total\s+assets[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
# #             'cash': r'cash[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?'
# #         }
        
# #         text_lower = text.lower()
# #         for metric, pattern in patterns.items():
# #             matches = re.findall(pattern, text_lower)
# #             if matches:
# #                 try:
# #                     value = float(matches[0].replace(',', ''))
# #                     if 'billion' in text_lower or ' b' in text_lower:
# #                         value *= 1_000_000_000
# #                     elif 'million' in text_lower or ' m' in text_lower:
# #                         value *= 1_000_000
# #                     metrics[metric] = value
# #                 except ValueError:
# #                     continue
# #         return metrics
    
# #     @staticmethod
# #     def _format_financial_metrics(metrics: Dict[str, float]) -> str:
# #         if not metrics:
# #             return "No quantifiable metrics detected"
        
# #         formatted = []
# #         for metric, value in metrics.items():
# #             name = metric.replace('_', ' ').title()
# #             if value >= 1_000_000_000:
# #                 formatted.append(f"• {name}: ${value/1_000_000_000:.1f}B")
# #             elif value >= 1_000_000:
# #                 formatted.append(f"• {name}: ${value/1_000_000:.1f}M")
# #             else:
# #                 formatted.append(f"• {name}: ${value:,.0f}")
# #         return "\n".join(formatted)
    
# #     @staticmethod
# #     def process_financial_document(document_text: str, query: str) -> str:
# #         if not document_text.strip():
# #             return "❌ No document content provided."
        
# #         if document_text.startswith("❌"):
# #             return document_text
        
# #         context = f"Document Analysis: {document_text[:4000]}..."
# #         prompt = f"Analyze this financial document and answer: {query}\n\nDocument: {document_text[:6000]}..."
        
# #         try:
# #             response = OpenAIHandler.generate_response(prompt, context, "document_qa")
# #             return f"## 📋 Financial Document Analysis\n\n{response}"
# #         except Exception as e:
# #             return f"❌ Analysis error: {str(e)}"

# # # Enhanced Financial Data Handler
# # class FinancialDataHandler:
# #     @staticmethod
# #     def get_real_time_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
# #         try:
# #             ticker = yf.Ticker(symbol)
# #             info = ticker.info
# #             hist = ticker.history(period=period)
            
# #             if hist.empty:
# #                 return {"error": f"No data available for {symbol}"}
            
# #             current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
# #             prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
# #             price_change = ((current_price - prev_close) / prev_close) * 100
            
# #             # Technical indicators
# #             delta = hist['Close'].diff()
# #             gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# #             loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# #             rs = gain / loss
# #             rsi = 100 - (100 / (1 + rs))
            
# #             ma_20 = hist['Close'].rolling(20).mean()
# #             ma_50 = hist['Close'].rolling(50).mean()
            
# #             volatility = hist['Close'].pct_change().std() * np.sqrt(252)
# #             avg_volume = hist['Volume'].mean()
# #             volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
# #             return {
# #                 "symbol": symbol,
# #                 "current_price": current_price,
# #                 "price_change": price_change,
# #                 "volume": int(hist['Volume'].iloc[-1]),
# #                 "market_cap": int(info.get('marketCap', 0)),
# #                 "pe_ratio": float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
# #                 "beta": float(info.get('beta', 0)) if info.get('beta') else 0,
# #                 "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
# #                 "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else current_price,
# #                 "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else current_price,
# #                 "volatility": float(volatility),
# #                 "volume_ratio": float(volume_ratio),
# #                 "historical_data": hist,
# #                 "company_info": {
# #                     "sector": info.get('sector', 'Unknown'),
# #                     "industry": info.get('industry', 'Unknown'),
# #                     "description": info.get('longBusinessSummary', 'No description')[:200] + "..."
# #                 }
# #             }
# #         except Exception as e:
# #             return {"error": str(e)}
    
# #     @staticmethod
# #     def calculate_investment_score(data: Dict[str, Any]) -> Dict[str, Any]:
# #         if "error" in data:
# #             return {"error": "Cannot calculate score"}
        
# #         # Scoring factors
# #         score = 50  # Base score
        
# #         # Price momentum
# #         if data['price_change'] > 5:
# #             score += 15
# #         elif data['price_change'] > 0:
# #             score += 10
# #         elif data['price_change'] > -5:
# #             score += 5
        
# #         # RSI
# #         rsi = data['rsi']
# #         if 40 <= rsi <= 60:
# #             score += 15
# #         elif 30 <= rsi <= 70:
# #             score += 10
# #         else:
# #             score += 5
        
# #         # Trend
# #         if data['current_price'] > data['ma_20'] > data['ma_50']:
# #             score += 15
# #         elif data['current_price'] > data['ma_20']:
# #             score += 10
        
# #         # Valuation
# #         pe = data['pe_ratio']
# #         if 10 < pe < 20:
# #             score += 10
# #         elif 5 < pe < 30:
# #             score += 5
        
# #         # Risk
# #         if data['volatility'] < 0.3:
# #             score += 10
# #         elif data['volatility'] < 0.5:
# #             score += 5
        
# #         # Determine recommendation
# #         if score >= 80:
# #             recommendation = "STRONG BUY"
# #             color_class = "score-excellent"
# #         elif score >= 65:
# #             recommendation = "BUY"
# #             color_class = "score-good"
# #         elif score >= 50:
# #             recommendation = "HOLD"
# #             color_class = "score-fair"
# #         else:
# #             recommendation = "SELL"
# #             color_class = "score-poor"
        
# #         return {
# #             "score": min(score, 100),
# #             "recommendation": recommendation,
# #             "color_class": color_class,
# #             "confidence": min(score * 0.8, 100)
# #         }

# # # Premium Chart Renderer
# # class PremiumChartRenderer:
# #     @staticmethod
# #     def render_premium_price_chart(data: Dict[str, Any]):
# #         if "error" in data:
# #             st.error("📈 Unable to load chart data")
# #             return
        
# #         hist = data["historical_data"]
        
# #         # Create sophisticated chart
# #         fig = make_subplots(
# #             rows=3, cols=1,
# #             shared_xaxes=True,
# #             vertical_spacing=0.05,
# #             subplot_titles=[
# #                 f'{data["symbol"]} - Price Action & Technical Analysis', 
# #                 'Volume Profile', 
# #                 'RSI Momentum'
# #             ],
# #             row_heights=[0.6, 0.2, 0.2]
# #         )
        
# #         # Candlestick chart
# #         fig.add_trace(
# #             go.Candlestick(
# #                 x=hist.index,
# #                 open=hist['Open'],
# #                 high=hist['High'],
# #                 low=hist['Low'],
# #                 close=hist['Close'],
# #                 name=f'{data["symbol"]} Price',
# #                 increasing_line_color='#00ff88',
# #                 decreasing_line_color='#ff4444',
# #                 increasing_fillcolor='rgba(0,255,136,0.7)',
# #                 decreasing_fillcolor='rgba(255,68,68,0.7)'
# #             ), row=1, col=1
# #         )
        
# #         # Moving averages
# #         ma_20 = hist['Close'].rolling(20).mean()
# #         ma_50 = hist['Close'].rolling(50).mean()
        
# #         fig.add_trace(go.Scatter(
# #             x=hist.index, y=ma_20, 
# #             mode='lines', name='MA 20',
# #             line=dict(color='#ffa500', width=2)
# #         ), row=1, col=1)
        
# #         fig.add_trace(go.Scatter(
# #             x=hist.index, y=ma_50, 
# #             mode='lines', name='MA 50',
# #             line=dict(color='#ff6b6b', width=2)
# #         ), row=1, col=1)
        
# #         # Volume
# #         colors = ['rgba(0,255,136,0.7)' if close > open else 'rgba(255,68,68,0.7)' 
# #                  for close, open in zip(hist['Close'], hist['Open'])]
        
# #         fig.add_trace(go.Bar(
# #             x=hist.index, y=hist['Volume'], 
# #             name='Volume', marker_color=colors,
# #             showlegend=False
# #         ), row=2, col=1)
        
# #         # RSI
# #         delta = hist['Close'].diff()
# #         gain = (delta.where(delta > 0, 0)).rolling(14).mean()
# #         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
# #         rs = gain / loss
# #         rsi = 100 - (100 / (1 + rs))
        
# #         fig.add_trace(go.Scatter(
# #             x=hist.index, y=rsi, 
# #             mode='lines', name='RSI',
# #             line=dict(color='#9f7aea', width=3)
# #         ), row=3, col=1)
        
# #         # RSI levels
# #         fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.8)", row=3, col=1)
# #         fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.8)", row=3, col=1)
# #         fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.5)", row=3, col=1)
        
# #         # Update layout
# #         fig.update_layout(
# #             height=600,
# #             template="plotly_dark",
# #             showlegend=True,
# #             title_text=f"{data['symbol']} - Advanced Technical Analysis",
# #             title_font=dict(size=24, color='white'),
# #             plot_bgcolor='rgba(0,0,0,0)',
# #             paper_bgcolor='rgba(0,0,0,0)',
# #             font=dict(color='white', family='Inter'),
# #             legend=dict(
# #                 bgcolor='rgba(255,255,255,0.1)',
# #                 bordercolor='rgba(255,255,255,0.2)',
# #                 borderwidth=1
# #             )
# #         )
        
# #         fig.update_xaxes(
# #             rangeslider_visible=False,
# #             gridcolor='rgba(255,255,255,0.1)',
# #             showgrid=True
# #         )
# #         fig.update_yaxes(
# #             gridcolor='rgba(255,255,255,0.1)',
# #             showgrid=True
# #         )
        
# #         st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
    
# #     @staticmethod
# #     def render_investment_dashboard(data: Dict[str, Any], score_data: Dict[str, Any]):
# #         """Render premium investment dashboard"""
# #         col1, col2, col3 = st.columns([1, 1, 1])
        
# #         # Current Metrics
# #         with col1:
# #             st.markdown("### 📊 Current Metrics")
            
# #             # Price card
# #             price_change_color = "#00ff88" if data['price_change'] >= 0 else "#ff4444"
# #             st.markdown(f"""
# #             <div class="metric-card">
# #                 <div class="metric-value">${data['current_price']:.2f}</div>
# #                 <div class="metric-label">Current Price</div>
# #                 <div class="metric-change" style="background: {price_change_color};">
# #                     {data['price_change']:+.2f}%
# #                 </div>
# #             </div>
# #             """, unsafe_allow_html=True)
            
# #             # Volume
# #             st.markdown(f"""
# #             <div class="metric-card">
# #                 <div class="metric-value">{data['volume']:,}</div>
# #                 <div class="metric-label">Volume</div>
# #                 <div class="metric-change">
# #                     {data['volume_ratio']:.1f}x Avg
# #                 </div>
# #             </div>
# #             """, unsafe_allow_html=True)
            
# #             # Market Cap
# #             market_cap_display = f"${data['market_cap']/1e9:.1f}B" if data['market_cap'] > 1e9 else f"${data['market_cap']/1e6:.1f}M"
# #             st.markdown(f"""
# #             <div class="metric-card">
# #                 <div class="metric-value">{market_cap_display}</div>
# #                 <div class="metric-label">Market Cap</div>
# #             </div>
# #             """, unsafe_allow_html=True)
        
# #         # Technical Indicators
# #         with col2:
# #             st.markdown("### 🎯 Technical Indicators")
            
# #             # RSI Gauge
# #             rsi_color = "#ff4444" if data['rsi'] > 70 or data['rsi'] < 30 else "#00ff88"
# #             rsi_status = "Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral"
            
# #             rsi_fig = go.Figure(go.Indicator(
# #                 mode="gauge+number",
# #                 value=data['rsi'],
# #                 domain={'x': [0, 1], 'y': [0, 1]},
# #                 title={'text': "RSI", 'font': {'color': 'white', 'size': 20}},
# #                 number={'font': {'color': 'white', 'size': 24}},
# #                 gauge={
# #                     'axis': {'range': [None, 100], 'tickcolor': 'white'},
# #                     'bar': {'color': rsi_color, 'thickness': 0.3},
# #                     'bgcolor': 'rgba(255,255,255,0.1)',
# #                     'borderwidth': 2,
# #                     'bordercolor': 'rgba(255,255,255,0.3)',
# #                     'steps': [
# #                         {'range': [0, 30], 'color': 'rgba(0,255,136,0.3)'},
# #                         {'range': [70, 100], 'color': 'rgba(255,68,68,0.3)'}
# #                     ],
# #                     'threshold': {
# #                         'line': {'color': "white", 'width': 4},
# #                         'thickness': 0.8,
# #                         'value': 50
# #                     }
# #                 }
# #             ))
            
# #             rsi_fig.update_layout(
# #                 height=250,
# #                 plot_bgcolor='rgba(0,0,0,0)',
# #                 paper_bgcolor='rgba(0,0,0,0)',
# #                 font={'color': 'white', 'family': 'Inter'}
# #             )
            
# #             st.plotly_chart(rsi_fig, use_container_width=True, key=get_unique_key())
            
# #             st.markdown(f"""
# #             <div style="text-align: center; color: {rsi_color}; font-weight: 600; margin-top: -20px;">
# #                 {rsi_status} ({data['rsi']:.1f})
# #             </div>
# #             """, unsafe_allow_html=True)
            
# #             # Additional indicators
# #             ma_trend = "🟢 Bullish" if data['current_price'] > data['ma_20'] else "🔴 Bearish"
# #             vol_status = "🟡 High" if data['volatility'] > 0.5 else "🟢 Normal"
            
# #             st.markdown(f"""
# #             <div style="margin-top: 1rem;">
# #                 <div style="margin: 0.5rem 0;"><strong>MA Trend:</strong> {ma_trend}</div>
# #                 <div style="margin: 0.5rem 0;"><strong>Volatility:</strong> {vol_status} ({data['volatility']:.1%})</div>
# #                 <div style="margin: 0.5rem 0;"><strong>P/E Ratio:</strong> {data['pe_ratio']:.1f}</div>
# #             </div>
# #             """, unsafe_allow_html=True)
        
# #         # Investment Score
# #         with col3:
# #             st.markdown("### 💡 AI Investment Score")
            
# #             # Score gauge
# #             score_fig = go.Figure(go.Indicator(
# #                 mode="gauge+number",
# #                 value=score_data['score'],
# #                 domain={'x': [0, 1], 'y': [0, 1]},
# #                 title={'text': "Investment Score", 'font': {'color': 'white', 'size': 20}},
# #                 number={'font': {'color': 'white', 'size': 24}},
# #                 gauge={
# #                     'axis': {'range': [None, 100], 'tickcolor': 'white'},
# #                     'bar': {'color': '#667eea', 'thickness': 0.3},
# #                     'bgcolor': 'rgba(255,255,255,0.1)',
# #                     'borderwidth': 2,
# #                     'bordercolor': 'rgba(255,255,255,0.3)',
# #                     'steps': [
# #                         {'range': [0, 40], 'color': 'rgba(245,101,101,0.3)'},
# #                         {'range': [40, 70], 'color': 'rgba(237,137,54,0.3)'},
# #                         {'range': [70, 100], 'color': 'rgba(72,187,120,0.3)'}
# #                     ]
# #                 }
# #             ))
            
# #             score_fig.update_layout(
# #                 height=250,
# #                 plot_bgcolor='rgba(0,0,0,0)',
# #                 paper_bgcolor='rgba(0,0,0,0)',
# #                 font={'color': 'white', 'family': 'Inter'}
# #             )
            
# #             st.plotly_chart(score_fig, use_container_width=True, key=get_unique_key())
            
# #             # Recommendation
# #             rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
# #             st.markdown(f"""
# #             <div class="{rec_class}" style="margin-top: -20px;">
# #                 {score_data['recommendation']}
# #             </div>
# #             """, unsafe_allow_html=True)
            
# #             st.markdown(f"""
# #             <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
# #                 <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
# #                 <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
# #             </div>
# #             """, unsafe_allow_html=True)

# # def main():
# #     init_session_state()
    
# #     # Premium Header
# #     st.markdown("""
# #     <div class="premium-header">
# #         <h1>🤖 FinDocGPT</h1>
# #         <h3>AI-Powered Financial Intelligence</h3>
# #         <p>Advanced financial analysis, forecasting, and investment strategy using FinanceBench dataset and Yahoo Finance API</p>
# #     </div>
# #     """, unsafe_allow_html=True)
    
# #     # Status Bar
# #     st.markdown('<div class="status-bar">', unsafe_allow_html=True)
# #     col1, col2, col3, col4 = st.columns(4)
    
# #     with col1:
# #         st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
# #     with col2:
# #         openai_status = "🟢 Connected" if client else "🔴 Limited"
# #         st.markdown(f"🤖 **OpenAI:** {openai_status}")
# #     with col3:
# #         if PDF_AVAILABLE:
# #             st.markdown("📄 **Documents:** TXT, PDF, CSV")
# #         else:
# #             st.markdown("📄 **Documents:** TXT, CSV Only")
# #     with col4:
# #         if st.button("🔄 Refresh All", key="refresh_main"):
# #             st.rerun()
    
# #     st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Premium Sidebar
# #     with st.sidebar:
# #         st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
# #         st.header("🎯 FinDocGPT Control Panel")
        
# #         # Stock Symbol Selection
# #         symbol_input = st.text_input(
# #             "📊 Stock Symbol", 
# #             value=st.session_state.selected_symbol,
# #             help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
# #         )
        
# #         if symbol_input != st.session_state.selected_symbol:
# #             st.session_state.selected_symbol = symbol_input.upper()
        
# #         # Analysis Period
# #         data_period = st.selectbox(
# #             "📅 Analysis Period",
# #             ["1mo", "3mo", "6mo", "1y", "2y"],
# #             index=3
# #         )
        
# #         # Real-time Data Display
# #         if st.session_state.selected_symbol:
# #             with st.spinner("📡 Loading market data..."):
# #                 current_data = FinancialDataHandler.get_real_time_data(
# #                     st.session_state.selected_symbol, data_period
# #                 )
            
# #             if "error" not in current_data:
# #                 st.success(f"✅ {st.session_state.selected_symbol} Data Loaded")
                
# #                 # Investment Score
# #                 score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
# #                 # Score display
# #                 st.markdown(f"""
# #                 <div class="metric-card {score_data['color_class']}">
# #                     <div class="metric-value">{score_data['recommendation']}</div>
# #                     <div class="metric-label">AI Investment Score</div>
# #                     <div class="metric-change">
# #                         Score: {score_data['score']:.1f}/100 ({score_data['confidence']:.1f}% confidence)
# #                     </div>
# #                 </div>
# #                 """, unsafe_allow_html=True)
                
# #                 # Key Metrics
# #                 col1, col2 = st.columns(2)
# #                 with col1:
# #                     price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
# #                     st.metric("Price", f"${current_data['current_price']:.2f}", 
# #                              f"{current_data['price_change']:+.2f}%", delta_color=price_color)
# #                     st.metric("Volume", f"{current_data['volume']:,}")
# #                     st.metric("Market Cap", f"${current_data['market_cap']/1e9:.1f}B")
                
# #                 with col2:
# #                     st.metric("P/E Ratio", f"{current_data['pe_ratio']:.1f}")
# #                     st.metric("Beta", f"{current_data.get('beta', 0):.2f}")
# #                     st.metric("RSI", f"{current_data['rsi']:.1f}")
# #             else:
# #                 st.error(f"❌ Error: {current_data['error']}")
        
# #         # Market Status
# #         st.markdown("---")
# #         st.header("📈 Market Status")
        
# #         now = datetime.now()
# #         is_market_open = now.weekday() < 5 and 9 <= now.hour <= 16
        
# #         if is_market_open:
# #             st.success("🟢 Market is Open")
# #             st.caption("NYSE: 9:30 AM - 4:00 PM ET")
# #         else:
# #             st.warning("🟡 Market is Closed")
# #             st.caption("Next open: Monday 9:30 AM ET")
        
# #         # Quick Actions
# #         st.markdown("---")
# #         st.header("⚡ Quick Actions")
        
# #         if st.button("📊 Refresh Data", use_container_width=True):
# #             st.rerun()
        
# #         if st.button("💾 Save Analysis", use_container_width=True):
# #             if st.session_state.current_analysis:
# #                 st.session_state.analysis_history.append(st.session_state.current_analysis)
# #                 st.success("Analysis saved!")
        
# #         if st.button("🗑️ Clear History", use_container_width=True):
# #             st.session_state.analysis_history.clear()
# #             st.success("History cleared!")
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Main Navigation Tabs
# #     tab1, tab2, tab3 = st.tabs([
# #         "🔍 Stage 1: Insights & Analysis", 
# #         "📈 Stage 2: Financial Forecasting", 
# #         "💼 Stage 3: Investment Strategy"
# #     ])
    
# #     # Stage 1: Document Q&A
# #     with tab1:
# #         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
# #         st.header("🔍 Stage 1: Document Insights & Analysis")
# #         st.markdown("*Document Q&A using FinanceBench • Market Sentiment Analysis • Anomaly Detection*")
        
# #         # Sub-tabs for Stage 1
# #         sub_tab1, sub_tab2, sub_tab3 = st.tabs([
# #             "📄 Document Q&A (FinanceBench)", 
# #             "😊 Sentiment Analysis", 
# #             "⚠️ Anomaly Detection"
# #         ])
        
# #         # Document Q&A Sub-tab
# #         with sub_tab1:
# #             st.subheader("📄 Financial Document Q&A - FinanceBench Dataset")
# #             st.markdown("""
# #             Upload financial documents or use FinanceBench dataset examples for AI-powered analysis.
# #             Ask questions about earnings reports, SEC filings, and financial statements.
# #             """)
            
# #             col1, col2 = st.columns([2, 1])
            
# #             with col1:
# #                 # Document Upload
# #                 uploaded_file = st.file_uploader(
# #                     "📁 Upload Financial Document", 
# #                     type=['txt', 'pdf', 'csv'],
# #                     help="Upload earnings reports, SEC filings, or financial datasets"
# #                 )
                
# #                 # Text input area
# #                 document_text = st.text_area(
# #                     "📝 Or paste document content:",
# #                     height=200,
# #                     placeholder="""Example FinanceBench content:

# # APPLE INC. Q4 2023 EARNINGS REPORT
# # Revenue: $89.5 billion (+2.8% YoY)
# # Net Income: $22.9 billion 
# # Gross Margin: 45.2%
# # iPhone Revenue: $43.8 billion
# # Services Revenue: $22.3 billion

# # Key Highlights:
# # - Record Services revenue driven by App Store growth
# # - iPhone 15 launch exceeded expectations  
# # - Supply chain constraints improved significantly

# # Risk Factors:
# # - China market regulatory challenges
# # - Component cost inflation"""
# #                 )
                
# #                 # Query Input
# #                 query = st.text_input(
# #                     "❓ Ask a question about the financial document:",
# #                     placeholder="e.g., What was the revenue growth? What are the main risk factors?",
# #                     help="Ask specific questions about financial metrics, performance, risks, or business insights"
# #                 )
                
# #                 # Analysis Button
# #                 if st.button("🔍 Analyze Document with AI", key="analyze_doc", use_container_width=True):
# #                     if (document_text.strip() or uploaded_file) and query.strip():
# #                         with st.spinner("🤖 FinDocGPT is analyzing the document..."):
# #                             # Process uploaded file if exists
# #                             if uploaded_file:
# #                                 doc_content = DocumentProcessor.extract_document_text(uploaded_file)
# #                                 if "❌" in doc_content:
# #                                     st.error(doc_content)
# #                                     doc_content = document_text  # Fallback to text area
# #                             else:
# #                                 doc_content = document_text
                            
# #                             # Generate AI analysis
# #                             response = DocumentProcessor.process_financial_document(doc_content, query)
                            
# #                             # Display results
# #                             st.markdown("### 📊 FinDocGPT Analysis Results")
                            
# #                             with st.container():
# #                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
# #                                 st.markdown(response)
# #                                 st.markdown('</div>', unsafe_allow_html=True)
                            
# #                             # Save to history
# #                             analysis_record = {
# #                                 "timestamp": datetime.now(),
# #                                 "type": "Document Q&A",
# #                                 "query": query,
# #                                 "response": response,
# #                                 "document_preview": doc_content[:200] + "..."
# #                             }
                            
# #                             st.session_state.analysis_history.append(analysis_record)
# #                             st.session_state.current_analysis = analysis_record
                            
# #                             st.success("✅ Analysis completed and saved to history!")
# #                     else:
# #                         st.warning("⚠️ Please provide both document content and a query for analysis.")
            
# #             with col2:
# #                 st.subheader("📋 Instructions")
# #                 st.markdown("""
# #                 **How to use Document Q&A:**
                
# #                 1. **Upload** a financial document or **paste** content
# #                 2. **Ask** a specific question about the document
# #                 3. **Click** "Analyze Document" to get AI insights
                
# #                 **Supported File Types:**
# #                 - ✅ **TXT**: Financial reports, transcripts
# #                 - ✅ **PDF**: 10-K filings, annual reports  
# #                 - ✅ **CSV**: Financial datasets, metrics
                
# #                 **Example Questions:**
# #                 - What was the quarterly revenue?
# #                 - What are the main risk factors?
# #                 - How did expenses change YoY?
# #                 - What's the cash flow situation?
# #                 """)
                
# #                 # Show recent analysis
# #                 if st.session_state.analysis_history:
# #                     st.subheader("📚 Recent Analyses")
# #                     recent = [a for a in st.session_state.analysis_history if a['type'] == 'Document Q&A'][-2:]
                    
# #                     for analysis in reversed(recent):
# #                         with st.expander(f"Q: {analysis['query'][:30]}..."):
# #                             st.caption(f"Date: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
# #                             st.text_area("Response:", analysis['response'][:150] + "...", height=80, disabled=True)
        
# #         # Sentiment Analysis Sub-tab
# #         with sub_tab2:
# #             st.subheader("😊 Market Sentiment Analysis")
# #             st.markdown("Analyze sentiment from earnings calls, press releases, and financial news using advanced AI.")
            
# #             if st.session_state.selected_symbol:
# #                 current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
# #                 if "error" not in current_data:
# #                     col1, col2 = st.columns([2, 1])
                    
# #                     with col1:
# #                         st.markdown(f"#### 📰 Sentiment Analysis for {st.session_state.selected_symbol}")
                        
# #                         # Sentiment text input
# #                         sentiment_text = st.text_area(
# #                             "Paste financial news, earnings call transcript, or press release:",
# #                             height=200,
# #                             placeholder=f"""Example content for {st.session_state.selected_symbol}:
                            
# # "During the earnings call, CEO mentioned strong performance in cloud services with 35% growth. Management expressed optimism about AI initiatives and expects continued expansion. However, they noted concerns about supply chain costs and competitive pressures in mobile segment. The company raised full-year guidance but warned of potential headwinds from foreign exchange rates."

# # Paste actual content here for AI-powered sentiment analysis..."""
# #                         )
                        
# #                         # Sentiment analysis type
# #                         analysis_depth = st.selectbox(
# #                             "Analysis Depth:",
# #                             ["Quick Sentiment", "Detailed Analysis", "Comprehensive Report"],
# #                             help="Choose the depth of sentiment analysis"
# #                         )
                        
# #                         if st.button("📊 Analyze Sentiment", key="analyze_sentiment", use_container_width=True):
# #                             if sentiment_text.strip():
# #                                 with st.spinner("🤖 Analyzing market sentiment..."):
# #                                     # Create context for sentiment analysis
# #                                     context = f"""
# #                                     Sentiment Analysis Request for: {st.session_state.selected_symbol}
# #                                     Analysis Type: {analysis_depth}
# #                                     Current Stock Price: ${current_data.get('current_price', 'N/A')}
# #                                     Recent Performance: {current_data.get('price_change', 'N/A')}%
                                    
# #                                     Text Content: {sentiment_text[:1000]}
# #                                     """
                                    
# #                                     prompt = f"""
# #                                     Perform {analysis_depth.lower()} sentiment analysis for {st.session_state.selected_symbol} based on the provided financial communication.
                                    
# #                                     Provide:
# #                                     1. Overall Sentiment (Bullish/Bearish/Neutral) with confidence score (0-100%)
# #                                     2. Key positive sentiment drivers
# #                                     3. Key negative sentiment factors  
# #                                     4. Management tone assessment
# #                                     5. Forward-looking statement analysis
# #                                     6. Investment implications
# #                                     7. Sentiment score breakdown by topic (if applicable)
                                    
# #                                     Text to analyze: {sentiment_text}
# #                                     """
                                    
# #                                     sentiment_response = OpenAIHandler.generate_response(prompt, context, "sentiment")
                                    
# #                                     st.markdown("### 📈 Sentiment Analysis Results")
# #                                     with st.container():
# #                                         st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
# #                                         st.markdown(sentiment_response)
# #                                         st.markdown('</div>', unsafe_allow_html=True)
                                    
# #                                     st.success("✅ Sentiment analysis completed!")
# #                             else:
# #                                 st.warning("⚠️ Please provide text content for sentiment analysis.")
                    
# #                     with col2:
# #                         st.markdown("#### 🎯 Sentiment Examples")
                        
# #                         sample_texts = {
# #                             "Bullish Example": "Management reported exceptional quarter with record revenue growth of 45%. New product launch exceeded all expectations. Strong pipeline for next year with multiple expansion opportunities.",
                            
# #                             "Bearish Example": "Company missed earnings expectations due to supply chain disruptions. Management lowered full-year guidance citing economic headwinds. Competitive pressures increasing in core markets.",
                            
# #                             "Neutral Example": "Quarter met expectations with steady performance. Management maintaining current guidance. Some positive developments offset by ongoing challenges in certain segments."
# #                         }
                        
# #                         for sentiment_type, example in sample_texts.items():
# #                             with st.expander(f"📝 {sentiment_type}"):
# #                                 st.text_area("", value=example, height=80, key=f"sentiment_{sentiment_type}")
# #                 else:
# #                     st.error(f"Cannot load data for {st.session_state.selected_symbol}")
# #             else:
# #                 st.info("ℹ️ Please select a stock symbol in the sidebar to begin sentiment analysis.")
        
# #         # Anomaly Detection Sub-tab
# #         with sub_tab3:
# #             st.subheader("⚠️ Real-Time Anomaly Detection")
# #             st.markdown("Identify unusual patterns and potential risks in financial metrics and market behavior.")
            
# #             if st.session_state.selected_symbol:
# #                 current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
# #                 if "error" not in current_data:
# #                     col1, col2 = st.columns([2, 1])
                    
# #                     with col1:
# #                         st.markdown("### 🔍 Anomaly Detection Results")
                        
# #                         # Define comprehensive anomaly detection thresholds
# #                         anomalies = []
# #                         warnings = []
                        
# #                         # Price Movement Anomalies
# #                         if abs(current_data['price_change']) > 15:
# #                             anomalies.append({
# #                                 "type": "Critical",
# #                                 "category": "Price Movement", 
# #                                 "message": f"Extreme price movement: {current_data['price_change']:+.2f}%",
# #                                 "impact": "High"
# #                             })
# #                         elif abs(current_data['price_change']) > 8:
# #                             warnings.append({
# #                                 "type": "Warning",
# #                                 "category": "Price Movement",
# #                                 "message": f"Significant price movement: {current_data['price_change']:+.2f}%",
# #                                 "impact": "Medium"
# #                             })
                        
# #                         # Technical Indicator Anomalies
# #                         rsi = current_data['rsi']
# #                         if rsi > 85:
# #                             anomalies.append({
# #                                 "type": "Critical",
# #                                 "category": "Technical Analysis",
# #                                 "message": f"Extremely overbought condition: RSI {rsi:.1f}",
# #                                 "impact": "High"
# #                             })
# #                         elif rsi < 15:
# #                             anomalies.append({
# #                                 "type": "Critical", 
# #                                 "category": "Technical Analysis",
# #                                 "message": f"Extremely oversold condition: RSI {rsi:.1f}",
# #                                 "impact": "High"
# #                             })
# #                         elif rsi > 75 or rsi < 25:
# #                             warnings.append({
# #                                 "type": "Warning",
# #                                 "category": "Technical Analysis", 
# #                                 "message": f"Overbought/oversold condition: RSI {rsi:.1f}",
# #                                 "impact": "Medium"
# #                             })
                        
# #                         # Volume Anomalies
# #                         volume_ratio = current_data.get('volume_ratio', 1)
# #                         if volume_ratio > 5:
# #                             anomalies.append({
# #                                 "type": "Critical",
# #                                 "category": "Volume",
# #                                 "message": f"Extreme volume spike: {volume_ratio:.1f}x average volume",
# #                                 "impact": "High"
# #                             })
# #                         elif volume_ratio > 2:
# #                             warnings.append({
# #                                 "type": "Warning", 
# #                                 "category": "Volume",
# #                                 "message": f"High volume activity: {volume_ratio:.1f}x average volume",
# #                                 "impact": "Medium"
# #                             })
                        
# #                         # Volatility Anomalies  
# #                         volatility = current_data['volatility']
# #                         if volatility > 1.5:
# #                             anomalies.append({
# #                                 "type": "Critical",
# #                                 "category": "Volatility",
# #                                 "message": f"Extreme volatility: {volatility:.2%} annualized",
# #                                 "impact": "High"
# #                             })
# #                         elif volatility > 0.8:
# #                             warnings.append({
# #                                 "type": "Warning",
# #                                 "category": "Volatility", 
# #                                 "message": f"High volatility: {volatility:.2%} annualized",
# #                                 "impact": "Medium"
# #                             })
                        
# #                         # Display Results
# #                         if anomalies:
# #                             st.error("### 🚨 Critical Anomalies Detected")
# #                             for anomaly in anomalies:
# #                                 st.markdown(f"""
# #                                 **{anomaly['category']}**: {anomaly['message']}  
# #                                 *Impact: {anomaly['impact']}*
# #                                 """)
                        
# #                         if warnings:
# #                             st.warning("### ⚠️ Warnings Detected")
# #                             for warning in warnings:
# #                                 st.markdown(f"""
# #                                 **{warning['category']}**: {warning['message']}  
# #                                 *Impact: {warning['impact']}*
# #                                 """)
                        
# #                         if not anomalies and not warnings:
# #                             st.success("### ✅ No Significant Anomalies Detected")
# #                             st.markdown("All monitored metrics are within normal ranges.")
                        
# #                         # Technical Analysis Summary
# #                         st.markdown("### 📊 Current Technical Status")
                        
# #                         col_a, col_b, col_c, col_d = st.columns(4)
                        
# #                         with col_a:
# #                             price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
# #                             st.metric("Price Change", f"{current_data['price_change']:+.2f}%", delta_color=price_color)
                        
# #                         with col_b:
# #                             rsi_status = "Overbought" if current_data['rsi'] > 70 else "Oversold" if current_data['rsi'] < 30 else "Neutral"
# #                             st.metric("RSI", f"{current_data['rsi']:.1f}", rsi_status)
                        
# #                         with col_c:
# #                             vol_status = "High" if current_data.get('volume_ratio', 1) > 2 else "Normal"
# #                             st.metric("Volume", f"{current_data['volume']:,}", f"{volume_ratio:.1f}x ({vol_status})")
                        
# #                         with col_d:
# #                             vol_status = "High" if current_data['volatility'] > 0.5 else "Normal"
# #                             st.metric("Volatility", f"{current_data['volatility']:.2%}", vol_status)
                    
# #                     with col2:
# #                         st.markdown("#### 🎯 Anomaly Monitoring")
                        
# #                         # Monitoring thresholds
# #                         st.markdown("**Current Thresholds:**")
# #                         thresholds = {
# #                             "Price Movement": "> ±8% (Warning), > ±15% (Critical)",
# #                             "RSI": "< 25 or > 75 (Warning), < 15 or > 85 (Critical)",
# #                             "Volume": "> 2x avg (Warning), > 5x avg (Critical)", 
# #                             "Volatility": "> 80% (Warning), > 150% (Critical)"
# #                         }
                        
# #                         for metric, threshold in thresholds.items():
# #                             st.caption(f"**{metric}**: {threshold}")
                        
# #                         st.markdown("---")
# #                         st.markdown("#### 📈 Real-time Status")
                        
# #                         # Current values with status indicators
# #                         metrics_status = [
# #                             ("Price Change", f"{current_data['price_change']:+.2f}%", 
# #                              "🟢" if abs(current_data['price_change']) < 3 else "🟡" if abs(current_data['price_change']) < 8 else "🔴"),
# #                             ("RSI", f"{current_data['rsi']:.1f}", 
# #                              "🟢" if 40 <= current_data['rsi'] <= 60 else "🟡" if 25 <= current_data['rsi'] <= 75 else "🔴"),
# #                             ("Volume Ratio", f"{current_data.get('volume_ratio', 1):.1f}x", 
# #                              "🟢" if current_data.get('volume_ratio', 1) < 1.5 else "🟡" if current_data.get('volume_ratio', 1) < 3 else "🔴"),
# #                             ("Volatility", f"{current_data['volatility']:.1%}", 
# #                              "🟢" if current_data['volatility'] < 0.4 else "🟡" if current_data['volatility'] < 0.8 else "🔴")
# #                         ]
                        
# #                         for metric, value, status in metrics_status:
# #                             st.markdown(f"{status} **{metric}**: {value}")
                        
# #                         # Auto-refresh option
# #                         st.markdown("---")
# #                         auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", key="auto_refresh_anomaly")
                        
# #                         if auto_refresh:
# #                             time.sleep(30)
# #                             st.rerun()
# #                 else:
# #                     st.error(f"Cannot load data for {st.session_state.selected_symbol}")
# #             else:
# #                 st.info("ℹ️ Please select a stock symbol in the sidebar to begin anomaly detection.")
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Stage 2: Financial Forecasting
# #     with tab2:
# #         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
# #         st.header("📈 Stage 2: Financial Forecasting")
# #         st.markdown("*AI-powered predictions • Advanced technical analysis • Market data integration*")
        
# #         if st.session_state.selected_symbol:
# #             current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
# #             if "error" not in current_data:
# #                 # Premium Chart
# #                 st.subheader("📊 Advanced Technical Analysis")
# #                 with st.container():
# #                     st.markdown('<div class="chart-container">', unsafe_allow_html=True)
# #                     PremiumChartRenderer.render_premium_price_chart(current_data)
# #                     st.markdown('</div>', unsafe_allow_html=True)
                
# #                 col1, col2 = st.columns([2, 1])
                
# #                 with col1:
# #                     st.subheader("🎯 AI-Powered Financial Forecasting")
                    
# #                     # Forecasting parameters
# #                     col_a, col_b = st.columns(2)
                    
# #                     with col_a:
# #                         forecast_period = st.selectbox(
# #                             "📅 Forecast Period:",
# #                             ["1 Week", "1 Month", "3 Months", "6 Months"],
# #                             index=1
# #                         )
                    
# #                     with col_b:
# #                         forecast_type = st.selectbox(
# #                             "📊 Analysis Type:",
# #                             ["Technical Analysis", "Combined Analysis"],
# #                             index=1
# #                         )
                    
# #                     # Generate forecast button
# #                     if st.button("🔮 Generate AI Forecast", key="generate_forecast", use_container_width=True):
# #                         with st.spinner("🤖 Generating advanced forecast..."):
                            
# #                             context = f"""
# #                             FORECASTING CONTEXT for {st.session_state.selected_symbol}
                            
# #                             Current Data:
# #                             - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
# #                             - Volume: {current_data['volume']:,}
# #                             - RSI: {current_data['rsi']:.1f}
# #                             - MA 20: ${current_data['ma_20']:.2f}
# #                             - MA 50: ${current_data['ma_50']:.2f}
# #                             - Volatility: {current_data['volatility']:.2%}
# #                             - Sector: {current_data['company_info']['sector']}
# #                             """
                            
# #                             prompt = f"""
# #                             Generate a comprehensive {forecast_period} forecast for {st.session_state.selected_symbol} using {forecast_type.lower()}.
                            
# #                             Provide detailed analysis including:
# #                             1. **Price Targets**: Bull case, base case, and bear case scenarios with specific prices and probabilities
# #                             2. **Key Catalysts**: List 3-5 positive drivers and 3-5 potential risks
# #                             3. **Technical Analysis**: Support/resistance levels, momentum indicators, and chart patterns
# #                             4. **Timeline**: Expected price movement schedule with key milestones
# #                             5. **Confidence Assessment**: Overall confidence level (0-100%) and reliability factors
# #                             6. **Risk Management**: Suggested stop-loss levels and position sizing recommendations
# #                             """
                            
# #                             forecast_response = OpenAIHandler.generate_response(prompt, context, "forecasting")
                            
# #                             st.markdown("### 🔮 AI Forecasting Results")
# #                             with st.container():
# #                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
# #                                 st.markdown(forecast_response)
# #                                 st.markdown('</div>', unsafe_allow_html=True)
                            
# #                             st.success("✅ Advanced forecast generated!")
                
# #                 with col2:
# #                     st.subheader("🎯 Forecasting Guide")
                    
# #                     st.markdown("""
# #                     **Forecast Types:**
# #                     - **Technical**: Price patterns, indicators
# #                     - **Combined**: Technical + Fundamental
                    
# #                     **Confidence Levels:**
# #                     - **Conservative**: Lower risk, modest returns
# #                     - **Moderate**: Balanced risk/reward  
# #                     - **Aggressive**: Higher risk, higher potential
# #                     """)
                    
# #                     # Current indicators
# #                     st.subheader("📊 Current Indicators")
                    
# #                     indicators = [
# #                         ("Price Trend", "🟢 Up" if current_data['price_change'] > 0 else "🔴 Down"),
# #                         ("RSI Status", "🟢 Neutral" if 30 < current_data['rsi'] < 70 else "🟡 Extreme"),
# #                         ("Volume", "🟢 Normal" if current_data.get('volume_ratio', 1) < 2 else "🟡 High"),
# #                         ("Volatility", "🟢 Low" if current_data['volatility'] < 0.5 else "🟡 High")
# #                     ]
                    
# #                     for indicator, status in indicators:
# #                         st.markdown(f"**{indicator}**: {status}")
# #             else:
# #                 st.error(f"Cannot load data for {st.session_state.selected_symbol}")
# #         else:
# #             st.info("Please select a stock symbol in the sidebar to begin forecasting.")
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Stage 3: Investment Strategy
# #     with tab3:
# #         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
# #         st.header("💼 Stage 3: Investment Strategy & Decision-Making")
# #         st.markdown("*Professional portfolio management • Advanced risk assessment • AI-powered trading recommendations*")
        
# #         if st.session_state.selected_symbol:
# #             current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
# #             if "error" not in current_data:
# #                 score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
# #                 # Premium Investment Dashboard
# #                 st.subheader("📊 Professional Investment Dashboard")
# #                 PremiumChartRenderer.render_investment_dashboard(current_data, score_data)
                
# #                 col1, col2 = st.columns([2, 1])
                
# #                 with col1:
# #                     st.subheader("🎯 AI Investment Strategy Generator")
                    
# #                     # Strategy parameters
# #                     col_a, col_b = st.columns(2)
                    
# #                     with col_a:
# #                         investment_horizon = st.selectbox(
# #                             "📅 Investment Horizon:",
# #                             ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"],
# #                             index=1
# #                         )
                    
# #                     with col_b:
# #                         risk_tolerance = st.selectbox(
# #                             "⚖️ Risk Tolerance:",
# #                             ["Conservative", "Moderate", "Aggressive"],
# #                             index=1
# #                         )
                    
# #                     # Generate strategy button
# #                     if st.button("💡 Generate Professional Investment Strategy", key="generate_strategy", use_container_width=True):
# #                         with st.spinner("🤖 Generating comprehensive investment strategy..."):
                            
# #                             context = f"""
# #                             INVESTMENT STRATEGY CONTEXT for {st.session_state.selected_symbol}
                            
# #                             Market Data:
# #                             - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
# #                             - P/E Ratio: {current_data['pe_ratio']:.1f}
# #                             - Beta: {current_data.get('beta', 0):.2f}
# #                             - Market Cap: ${current_data['market_cap']/1e9:.1f}B
# #                             - RSI: {current_data['rsi']:.1f}
# #                             - Volatility: {current_data['volatility']:.2%}
# #                             - Investment Score: {score_data['score']:.1f}/100
# #                             - AI Recommendation: {score_data['recommendation']}
                            
# #                             Parameters:
# #                             - Investment Horizon: {investment_horizon}
# #                             - Risk Tolerance: {risk_tolerance}
# #                             - Portfolio Value: ${st.session_state.portfolio_value:,.0f}
# #                             """
                            
# #                             prompt = f"""
# #                             Create a comprehensive, professional-grade investment strategy for {st.session_state.selected_symbol}.
                            
# #                             Provide detailed analysis including:
                            
# #                             1. **Executive Summary & Investment Decision**:
# #                                - Clear BUY/SELL/HOLD recommendation with detailed rationale
# #                                - Investment thesis in 2-3 sentences
# #                                - Expected return potential and timeframe
                            
# #                             2. **Entry Strategy & Timing**:
# #                                - Optimal entry price points and conditions
# #                                - Dollar-cost averaging vs. lump sum recommendations
# #                                - Market timing considerations and triggers
                            
# #                             3. **Risk Management Framework**:
# #                                - Stop-loss levels with specific prices
# #                                - Position sizing as % of portfolio (given ${st.session_state.portfolio_value:,.0f} portfolio)
# #                                - Risk/reward ratio analysis
# #                                - Hedging strategies if applicable
                            
# #                             4. **Exit Strategy & Profit Taking**:
# #                                - Primary profit target prices
# #                                - Partial profit-taking levels
# #                                - Review and rebalancing triggers
                            
# #                             5. **Portfolio Integration**:
# #                                - How this position fits into overall allocation
# #                                - Diversification impact and sector exposure
# #                                - Correlation with existing holdings
                            
# #                             6. **Monitoring & Review Framework**:
# #                                - Key metrics to track weekly/monthly
# #                                - Fundamental and technical review schedule
# #                                - Scenario planning (bull/bear case actions)
                            
# #                             Consider the {risk_tolerance.lower()} risk profile and {investment_horizon} timeframe in all recommendations.
# #                             Provide specific, actionable advice that a professional investment advisor would give.
# #                             """
                            
# #                             investment_strategy = OpenAIHandler.generate_response(prompt, context, "investment")
                            
# #                             st.markdown("### 💼 Professional Investment Strategy")
# #                             with st.container():
# #                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
# #                                 st.markdown(investment_strategy)
# #                                 st.markdown('</div>', unsafe_allow_html=True)
                            
# #                             # Add to portfolio option
# #                             if "BUY" in investment_strategy.upper():
# #                                 if st.button("➕ Add to Portfolio", key="add_to_portfolio"):
# #                                     portfolio_item = {
# #                                         "symbol": st.session_state.selected_symbol,
# #                                         "action": score_data['recommendation'],
# #                                         "price": current_data['current_price'],
# #                                         "timestamp": datetime.now(),
# #                                         "strategy": investment_strategy[:200] + "...",
# #                                         "score": score_data['score']
# #                                     }
# #                                     st.session_state.investment_portfolio.append(portfolio_item)
# #                                     st.success(f"✅ {st.session_state.selected_symbol} added to portfolio!")
                            
# #                             st.success("✅ Professional investment strategy generated!")
                
# #                 with col2:
# #                     st.subheader("💡 Strategy Guide")
                    
# #                     st.markdown("""
# #                     **Investment Horizons:**
# #                     - **Short-term**: Technical focus, quick profits
# #                     - **Medium-term**: Balanced approach
# #                     - **Long-term**: Fundamental focus, compound growth
                    
# #                     **Risk Levels:**
# #                     - **Conservative**: Capital preservation priority
# #                     - **Moderate**: Balanced risk/return
# #                     - **Aggressive**: Growth maximization
# #                     """)
                    
# #                     # Quick recommendation
# #                     st.subheader("🎯 Quick AI Recommendation")
                    
# #                     rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
# #                     st.markdown(f"""
# #                     <div class="{rec_class}">
# #                         {score_data['recommendation']}
# #                     </div>
# #                     """, unsafe_allow_html=True)
                    
# #                     st.markdown(f"""
# #                     <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
# #                         <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
# #                         <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
# #                     </div>
# #                     """, unsafe_allow_html=True)
                    
# #                     # Portfolio summary - Only show if there are actual positions
# #                     if st.session_state.investment_portfolio:
# #                         st.subheader("💼 Portfolio Summary")
# #                         st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.0f}")
# #                         st.metric("Total Positions", len(st.session_state.investment_portfolio))
                        
# #                         st.markdown("**Recent Positions:**")
# #                         recent = st.session_state.investment_portfolio[-3:]
# #                         for item in reversed(recent):
# #                             st.markdown(f"• {item['symbol']} - {item['action']} (Score: {item['score']:.1f})")
# #                     else:
# #                         st.subheader("💼 Portfolio")
# #                         st.info("No positions yet. Generate investment strategies to build your portfolio!")
# #             else:
# #                 st.error(f"Cannot load data for {st.session_state.selected_symbol}")
# #         else:
# #             st.info("Please select a stock symbol in the sidebar to begin investment analysis.")
        
# #         # Portfolio Management Section
# #         if st.session_state.investment_portfolio:
# #             st.markdown("---")
# #             st.subheader("📊 Portfolio Management")
            
# #             # Portfolio table
# #             portfolio_df = pd.DataFrame(st.session_state.investment_portfolio)
# #             portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
# #             portfolio_df = portfolio_df.sort_values('timestamp', ascending=False)
            
# #             # Display portfolio
# #             st.dataframe(
# #                 portfolio_df[['symbol', 'action', 'price', 'score', 'timestamp']],
# #                 use_container_width=True,
# #                 hide_index=True
# #             )
            
# #             # Portfolio actions
# #             col1, col2, col3 = st.columns(3)
            
# #             with col1:
# #                 if st.button("📈 Portfolio Analytics", use_container_width=True):
# #                     st.info("Advanced portfolio analytics - Coming soon!")
            
# #             with col2:
# #                 if st.button("📊 Performance Report", use_container_width=True):
# #                     st.info("Performance reporting - Coming soon!")
            
# #             with col3:
# #                 if st.button("🗑️ Clear Portfolio", use_container_width=True):
# #                     st.session_state.investment_portfolio.clear()
# #                     st.success("Portfolio cleared!")
# #                     st.rerun()
        
# #         st.markdown('</div>', unsafe_allow_html=True)
    
# #     # Footer Status
# #     st.markdown("---")
# #     col1, col2 = st.columns(2)
    
# #     with col1:
# #         if not PDF_AVAILABLE:
# #             st.error("""
# #             🚨 **PDF Processing Not Available**
# #             Install: `pip install PyPDF2 pdfplumber`
# #             """)
# #         else:
# #             st.success("✅ All document formats supported (TXT, PDF, CSV)")
    
# #     with col2:
# #         if not client:
# #             st.warning("⚠️ OpenAI API not configured. Set OPENAI_API_KEY environment variable for full AI capabilities.")
# #         else:
# #             st.success("✅ AI analysis fully enabled")

# # # Run the application
# # if __name__ == "__main__":
# #     main()

# """
# FinDocGPT - AI-Powered Financial Intelligence
# Premium Professional Interface - AkashX.ai Hackathon Solution
# """

# import streamlit as st
# import pandas as pd
# import plotly.graph_objects as go
# import plotly.express as px
# from plotly.subplots import make_subplots
# from datetime import datetime, timedelta
# import yfinance as yf
# import numpy as np
# import time
# import uuid
# from openai import OpenAI
# import os
# from typing import Dict, List, Any
# import json
# from dataclasses import dataclass
# import logging
# import io
# import re

# # PDF processing imports
# try:
#     import PyPDF2
#     import pdfplumber
#     PDF_AVAILABLE = True
# except ImportError:
#     PDF_AVAILABLE = False

# # Configure Streamlit
# st.set_page_config(
#     page_title="FinDocGPT - AI Financial Intelligence",
#     page_icon="🤖",
#     layout="wide",
#     initial_sidebar_state="expanded"  # Force sidebar to be expanded
# )

# # Premium CSS Styling
# st.markdown("""
# <style>
# /* Import Google Fonts */
# @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

# /* Global Styling */
# .main {
#     font-family: 'Inter', sans-serif;
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     min-height: 100vh;
# }

# /* Force sidebar to be visible */
# .css-1d391kg {
#     background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
#     min-width: 300px !important;
#     max-width: 350px !important;
# }

# /* Sidebar visibility fix */
# [data-testid="stSidebar"] {
#     display: block !important;
#     visibility: visible !important;
#     opacity: 1 !important;
#     transform: translateX(0) !important;
# }

# /* Prevent sidebar from collapsing */
# [data-testid="stSidebar"][aria-expanded="false"] {
#     display: block !important;
#     transform: translateX(0) !important;
# }

# /* Header Styling */
# .premium-header {
#     background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
#     padding: 2rem;
#     border-radius: 20px;
#     margin-bottom: 2rem;
#     text-align: center;
#     color: white;
#     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#     position: relative;
#     overflow: hidden;
# }

# .premium-header::before {
#     content: '';
#     position: absolute;
#     top: 0;
#     left: 0;
#     right: 0;
#     bottom: 0;
#     background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
#     opacity: 0.3;
# }

# .premium-header h1 {
#     font-size: 3rem;
#     font-weight: 700;
#     margin: 0;
#     text-shadow: 0 2px 4px rgba(0,0,0,0.3);
#     position: relative;
#     z-index: 1;
# }

# .premium-header h3 {
#     font-size: 1.2rem;
#     font-weight: 400;
#     margin: 0.5rem 0;
#     opacity: 0.9;
#     position: relative;
#     z-index: 1;
# }

# .premium-header p {
#     font-size: 1rem;
#     margin: 1rem 0 0 0;
#     opacity: 0.8;
#     position: relative;
#     z-index: 1;
# }

# /* Status Bar */
# .status-bar {
#     background: rgba(255, 255, 255, 0.95);
#     backdrop-filter: blur(10px);
#     padding: 1rem;
#     border-radius: 15px;
#     margin-bottom: 2rem;
#     box-shadow: 0 8px 32px rgba(0,0,0,0.1);
#     border: 1px solid rgba(255,255,255,0.2);
# }

# .status-item {
#     display: flex;
#     align-items: center;
#     gap: 0.5rem;
#     font-weight: 500;
#     color: #2d3748;
# }

# /* Card Styling */
# .premium-card {
#     background: rgba(255, 255, 255, 0.95);
#     backdrop-filter: blur(20px);
#     border-radius: 20px;
#     padding: 2rem;
#     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#     border: 1px solid rgba(255,255,255,0.2);
#     margin-bottom: 2rem;
#     transition: all 0.3s ease;
# }

# .premium-card:hover {
#     transform: translateY(-5px);
#     box-shadow: 0 25px 50px rgba(0,0,0,0.15);
# }

# /* Metric Cards */
# .metric-card {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     color: white;
#     padding: 1.5rem;
#     border-radius: 15px;
#     text-align: center;
#     box-shadow: 0 10px 30px rgba(102,126,234,0.3);
#     transition: all 0.3s ease;
# }

# .metric-card:hover {
#     transform: translateY(-3px);
#     box-shadow: 0 15px 40px rgba(102,126,234,0.4);
# }

# .metric-value {
#     font-size: 2rem;
#     font-weight: 700;
#     margin-bottom: 0.5rem;
# }

# .metric-label {
#     font-size: 0.9rem;
#     opacity: 0.9;
#     font-weight: 500;
# }

# .metric-change {
#     font-size: 0.8rem;
#     margin-top: 0.5rem;
#     padding: 0.3rem 0.6rem;
#     border-radius: 20px;
#     background: rgba(255,255,255,0.2);
# }

# /* Sidebar Styling */
# .sidebar-content {
#     color: white;
# }

# /* Investment Score */
# .score-excellent {
#     background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
# }

# .score-good {
#     background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
# }

# .score-fair {
#     background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
# }

# .score-poor {
#     background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
# }

# /* Analysis Box */
# .analysis-box {
#     background: rgba(255, 255, 255, 0.95);
#     backdrop-filter: blur(20px);
#     padding: 2rem;
#     border-radius: 20px;
#     border: 1px solid rgba(255,255,255,0.2);
#     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#     margin: 2rem 0;
# }

# /* Buttons */
# .stButton > button {
#     background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
#     color: white;
#     border: none;
#     border-radius: 12px;
#     padding: 0.75rem 2rem;
#     font-weight: 600;
#     font-size: 1rem;
#     transition: all 0.3s ease;
#     box-shadow: 0 4px 15px rgba(102,126,234,0.3);
# }

# .stButton > button:hover {
#     transform: translateY(-2px);
#     box-shadow: 0 8px 25px rgba(102,126,234,0.4);
# }

# /* Tabs */
# .stTabs [data-baseweb="tab-list"] {
#     background: rgba(255, 255, 255, 0.1);
#     border-radius: 15px;
#     padding: 0.5rem;
#     backdrop-filter: blur(10px);
# }

# .stTabs [data-baseweb="tab"] {
#     background: transparent;
#     border-radius: 10px;
#     color: white;
#     font-weight: 500;
#     padding: 1rem 2rem;
#     transition: all 0.3s ease;
# }

# .stTabs [aria-selected="true"] {
#     background: rgba(255, 255, 255, 0.2);
#     color: white;
# }

# /* Charts */
# .chart-container {
#     background: rgba(255, 255, 255, 0.95);
#     border-radius: 20px;
#     padding: 2rem;
#     box-shadow: 0 20px 40px rgba(0,0,0,0.1);
#     backdrop-filter: blur(20px);
# }

# /* File Upload */
# .uploadedFile {
#     background: rgba(255, 255, 255, 0.1);
#     border: 2px dashed rgba(255, 255, 255, 0.3);
#     border-radius: 15px;
#     padding: 2rem;
#     text-align: center;
#     color: white;
# }

# /* Recommendations */
# .recommendation-buy {
#     background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
#     color: white;
#     padding: 1rem 2rem;
#     border-radius: 15px;
#     text-align: center;
#     font-weight: 600;
#     font-size: 1.1rem;
#     box-shadow: 0 10px 30px rgba(72,187,120,0.3);
# }

# .recommendation-hold {
#     background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
#     color: white;
#     padding: 1rem 2rem;
#     border-radius: 15px;
#     text-align: center;
#     font-weight: 600;
#     font-size: 1.1rem;
#     box-shadow: 0 10px 30px rgba(237,137,54,0.3);
# }

# .recommendation-sell {
#     background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
#     color: white;
#     padding: 1rem 2rem;
#     border-radius: 15px;
#     text-align: center;
#     font-weight: 600;
#     font-size: 1.1rem;
#     box-shadow: 0 10px 30px rgba(245,101,101,0.3);
# }

# /* Progress Bars */
# .stProgress .st-bo {
#     background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
#     height: 12px;
#     border-radius: 6px;
# }

# /* Text Input */
# .stTextInput > div > div > input {
#     background: rgba(255, 255, 255, 0.1);
#     border: 1px solid rgba(255, 255, 255, 0.2);
#     border-radius: 10px;
#     color: white;
#     backdrop-filter: blur(10px);
# }

# /* Select Box */
# .stSelectbox > div > div > select {
#     background: rgba(255, 255, 255, 0.1);
#     border: 1px solid rgba(255, 255, 255, 0.2);
#     border-radius: 10px;
#     color: white;
#     backdrop-filter: blur(10px);
# }

# /* Hide Streamlit Branding */
# #MainMenu {visibility: hidden;}
# footer {visibility: hidden;}
# header {visibility: hidden;}
# </style>

# <script>
# // Force sidebar to stay visible
# setTimeout(function() {
#     const sidebar = document.querySelector('[data-testid="stSidebar"]');
#     if (sidebar) {
#         sidebar.style.display = 'block';
#         sidebar.style.transform = 'translateX(0)';
#         sidebar.style.visibility = 'visible';
#         sidebar.style.opacity = '1';
#     }
# }, 1000);
# </script>
# """, unsafe_allow_html=True)

# # Configuration
# @dataclass
# class Config:
#     OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
#     DEFAULT_MODEL: str = "gpt-4"
#     MAX_TOKENS: int = 2000
#     TEMPERATURE: float = 0.3

# config = Config()

# # Initialize OpenAI client
# client = None
# if config.OPENAI_API_KEY:
#     try:
#         client = OpenAI(api_key=config.OPENAI_API_KEY)
#     except Exception as e:
#         st.warning(f"⚠️ OpenAI initialization error: {str(e)}")
#         client = None

# # Session State Management
# def init_session_state():
#     defaults = {
#         "selected_symbol": "AAPL",
#         "analysis_history": [],
#         "investment_portfolio": [],
#         "portfolio_value": 100000,
#         "current_analysis": None,
#         "chart_counter": 0
#     }
    
#     for key, value in defaults.items():
#         if key not in st.session_state:
#             st.session_state[key] = value

# def get_unique_key():
#     st.session_state.chart_counter += 1
#     return f"component_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"

# # Enhanced OpenAI Handler
# class OpenAIHandler:
#     @staticmethod
#     def generate_response(prompt: str, context: str = "", analysis_type: str = "general") -> str:
#         if not client:
#             return OpenAIHandler._fallback_analysis(prompt, analysis_type)
        
#         system_prompts = {
#             "document_qa": "You are FinDocGPT, a specialized financial document analyst. Provide precise, data-driven answers with specific financial metrics and insights.",
#             "sentiment": "You are a market sentiment analyst. Analyze financial communications to determine market sentiment with confidence scores.",
#             "forecasting": "You are a quantitative financial forecasting expert. Provide specific price targets with timeframes and probability ranges.",
#             "investment": "You are a senior investment strategist. Provide clear BUY/SELL/HOLD recommendations with entry/exit points and risk management.",
#             "general": "You are FinDocGPT, an AI financial intelligence assistant. Provide comprehensive financial analysis."
#         }
        
#         system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
#         # Context length management
#         max_context_chars = 6000
#         if len(context) > max_context_chars:
#             context = context[:max_context_chars] + "\n[Content truncated...]"
#         if len(prompt) > max_context_chars:
#             prompt = prompt[:max_context_chars] + "\n[Query truncated...]"
        
#         try:
#             response = client.chat.completions.create(
#                 model=config.DEFAULT_MODEL,
#                 messages=[
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": f"Context: {context}\n\nRequest: {prompt}"}
#                 ],
#                 max_tokens=config.MAX_TOKENS,
#                 temperature=config.TEMPERATURE
#             )
#             return response.choices[0].message.content
#         except Exception as e:
#             if "context_length_exceeded" in str(e).lower():
#                 return "⚠️ Document too large for analysis. Please try with a smaller section."
#             return f"⚠️ AI analysis error: {str(e)}"
    
#     @staticmethod
#     def _fallback_analysis(prompt: str, analysis_type: str) -> str:
#         fallbacks = {
#             "document_qa": "📊 Document analysis requires OpenAI API configuration.",
#             "sentiment": "😊 Sentiment analysis shows neutral market conditions.",
#             "forecasting": "📈 Basic forecasting suggests following current trends.",
#             "investment": "💼 Conservative HOLD recommendation based on available data.",
#             "general": "🤖 Limited analysis without OpenAI API key."
#         }
#         return fallbacks.get(analysis_type, fallbacks["general"])

# # Enhanced Document Processor
# class DocumentProcessor:
#     @staticmethod
#     def extract_document_text(uploaded_file) -> str:
#         try:
#             file_type = uploaded_file.type
#             file_name = uploaded_file.name.lower()
            
#             if file_type == "text/plain" or file_name.endswith('.txt'):
#                 return DocumentProcessor._process_txt_file(uploaded_file)
#             elif file_type == "application/pdf" or file_name.endswith('.pdf'):
#                 if PDF_AVAILABLE:
#                     return DocumentProcessor._process_pdf_file(uploaded_file)
#                 else:
#                     return "❌ PDF processing unavailable. Install: pip install PyPDF2 pdfplumber"
#             elif file_type in ["text/csv", "application/vnd.ms-excel"] or file_name.endswith('.csv'):
#                 return DocumentProcessor._process_csv_file(uploaded_file)
#             else:
#                 return f"❌ Unsupported file type: {file_type}"
#         except Exception as e:
#             return f"❌ Error processing file: {str(e)}"
    
#     @staticmethod
#     def _process_pdf_file(uploaded_file) -> str:
#         try:
#             pdf_bytes = uploaded_file.read()
#             with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
#                 text_pages = []
                
#                 for i, page in enumerate(pdf.pages[:10]):  # Limit pages
#                     page_text = page.extract_text()
#                     if page_text:
#                         text_pages.append(page_text[:1500])  # Limit page content
                
#                 if text_pages:
#                     full_text = "\n\n".join(text_pages)
#                     if len(full_text) > 15000:
#                         full_text = full_text[:15000] + "\n[Document truncated...]"
                    
#                     financial_metrics = DocumentProcessor._extract_financial_metrics(full_text)
                    
#                     return f"""📄 PDF Document Analysis:
# 📊 File Statistics:
# - Pages processed: {len(text_pages)}/{len(pdf.pages)}
# - Characters: {len(full_text):,}

# 💰 Financial Data:
# {DocumentProcessor._format_financial_metrics(financial_metrics)}

# 📝 Content:
# {full_text}"""
#                 else:
#                     return "❌ Could not extract text from PDF."
#         except Exception as e:
#             return f"❌ PDF processing error: {str(e)}"
    
#     @staticmethod
#     def _process_txt_file(uploaded_file) -> str:
#         try:
#             encodings = ['utf-8', 'latin-1', 'cp1252']
#             content = None
            
#             for encoding in encodings:
#                 try:
#                     uploaded_file.seek(0)
#                     content = uploaded_file.read().decode(encoding)
#                     break
#                 except UnicodeDecodeError:
#                     continue
            
#             if content is None:
#                 return "❌ Could not decode text file."
            
#             if len(content) > 15000:
#                 content = content[:15000] + "\n[Content truncated...]"
            
#             financial_metrics = DocumentProcessor._extract_financial_metrics(content)
            
#             return f"""📄 TXT Document Analysis:
# 📊 Statistics: {len(content):,} characters, {len(content.split()):,} words

# 💰 Financial Data:
# {DocumentProcessor._format_financial_metrics(financial_metrics)}

# 📝 Content:
# {content}"""
#         except Exception as e:
#             return f"❌ TXT processing error: {str(e)}"
    
#     @staticmethod
#     def _process_csv_file(uploaded_file) -> str:
#         try:
#             df = pd.read_csv(uploaded_file)
            
#             if len(df) > 500:
#                 df = df.head(500)
#                 note = f"\n[Showing first 500 rows of {len(df)} total]"
#             else:
#                 note = ""
            
#             financial_keywords = ['revenue', 'sales', 'income', 'profit', 'cost', 'price', 'amount', 'value']
#             financial_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in financial_keywords)]
            
#             return f"""📊 CSV Dataset Analysis:
# 📈 Overview: {len(df)} rows, {len(df.columns)} columns
# 💰 Financial columns: {', '.join(financial_cols[:5]) if financial_cols else 'None detected'}

# 📋 Sample Data:
# {df.head().to_string()}

# 📝 Full Dataset:
# {df.to_string()[:8000]}{note}"""
#         except Exception as e:
#             return f"❌ CSV processing error: {str(e)}"
    
#     @staticmethod
#     def _extract_financial_metrics(text: str) -> Dict[str, float]:
#         metrics = {}
#         patterns = {
#             'revenue': r'(?:revenue|sales)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
#             'net_income': r'net\s+(?:income|earnings)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
#             'assets': r'total\s+assets[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
#             'cash': r'cash[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?'
#         }
        
#         text_lower = text.lower()
#         for metric, pattern in patterns.items():
#             matches = re.findall(pattern, text_lower)
#             if matches:
#                 try:
#                     value = float(matches[0].replace(',', ''))
#                     if 'billion' in text_lower or ' b' in text_lower:
#                         value *= 1_000_000_000
#                     elif 'million' in text_lower or ' m' in text_lower:
#                         value *= 1_000_000
#                     metrics[metric] = value
#                 except ValueError:
#                     continue
#         return metrics
    
#     @staticmethod
#     def _format_financial_metrics(metrics: Dict[str, float]) -> str:
#         if not metrics:
#             return "No quantifiable metrics detected"
        
#         formatted = []
#         for metric, value in metrics.items():
#             name = metric.replace('_', ' ').title()
#             if value >= 1_000_000_000:
#                 formatted.append(f"• {name}: ${value/1_000_000_000:.1f}B")
#             elif value >= 1_000_000:
#                 formatted.append(f"• {name}: ${value/1_000_000:.1f}M")
#             else:
#                 formatted.append(f"• {name}: ${value:,.0f}")
#         return "\n".join(formatted)
    
#     @staticmethod
#     def process_financial_document(document_text: str, query: str) -> str:
#         if not document_text.strip():
#             return "❌ No document content provided."
        
#         if document_text.startswith("❌"):
#             return document_text
        
#         context = f"Document Analysis: {document_text[:4000]}..."
#         prompt = f"Analyze this financial document and answer: {query}\n\nDocument: {document_text[:6000]}..."
        
#         try:
#             response = OpenAIHandler.generate_response(prompt, context, "document_qa")
#             return f"## 📋 Financial Document Analysis\n\n{response}"
#         except Exception as e:
#             return f"❌ Analysis error: {str(e)}"

# # Enhanced Financial Data Handler
# class FinancialDataHandler:
#     @staticmethod
#     def get_real_time_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
#         try:
#             ticker = yf.Ticker(symbol)
#             info = ticker.info
#             hist = ticker.history(period=period)
            
#             if hist.empty:
#                 return {"error": f"No data available for {symbol}"}
            
#             current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
#             prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
#             price_change = ((current_price - prev_close) / prev_close) * 100
            
#             # Technical indicators
#             delta = hist['Close'].diff()
#             gain = (delta.where(delta > 0, 0)).rolling(14).mean()
#             loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
#             rs = gain / loss
#             rsi = 100 - (100 / (1 + rs))
            
#             ma_20 = hist['Close'].rolling(20).mean()
#             ma_50 = hist['Close'].rolling(50).mean()
            
#             volatility = hist['Close'].pct_change().std() * np.sqrt(252)
#             avg_volume = hist['Volume'].mean()
#             volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
#             return {
#                 "symbol": symbol,
#                 "current_price": current_price,
#                 "price_change": price_change,
#                 "volume": int(hist['Volume'].iloc[-1]),
#                 "market_cap": int(info.get('marketCap', 0)),
#                 "pe_ratio": float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
#                 "beta": float(info.get('beta', 0)) if info.get('beta') else 0,
#                 "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
#                 "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else current_price,
#                 "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else current_price,
#                 "volatility": float(volatility),
#                 "volume_ratio": float(volume_ratio),
#                 "historical_data": hist,
#                 "company_info": {
#                     "sector": info.get('sector', 'Unknown'),
#                     "industry": info.get('industry', 'Unknown'),
#                     "description": info.get('longBusinessSummary', 'No description')[:200] + "..."
#                 }
#             }
#         except Exception as e:
#             return {"error": str(e)}
    
#     @staticmethod
#     def calculate_investment_score(data: Dict[str, Any]) -> Dict[str, Any]:
#         if "error" in data:
#             return {"error": "Cannot calculate score"}
        
#         # Scoring factors
#         score = 50  # Base score
        
#         # Price momentum
#         if data['price_change'] > 5:
#             score += 15
#         elif data['price_change'] > 0:
#             score += 10
#         elif data['price_change'] > -5:
#             score += 5
        
#         # RSI
#         rsi = data['rsi']
#         if 40 <= rsi <= 60:
#             score += 15
#         elif 30 <= rsi <= 70:
#             score += 10
#         else:
#             score += 5
        
#         # Trend
#         if data['current_price'] > data['ma_20'] > data['ma_50']:
#             score += 15
#         elif data['current_price'] > data['ma_20']:
#             score += 10
        
#         # Valuation
#         pe = data['pe_ratio']
#         if 10 < pe < 20:
#             score += 10
#         elif 5 < pe < 30:
#             score += 5
        
#         # Risk
#         if data['volatility'] < 0.3:
#             score += 10
#         elif data['volatility'] < 0.5:
#             score += 5
        
#         # Determine recommendation
#         if score >= 80:
#             recommendation = "STRONG BUY"
#             color_class = "score-excellent"
#         elif score >= 65:
#             recommendation = "BUY"
#             color_class = "score-good"
#         elif score >= 50:
#             recommendation = "HOLD"
#             color_class = "score-fair"
#         else:
#             recommendation = "SELL"
#             color_class = "score-poor"
        
#         return {
#             "score": min(score, 100),
#             "recommendation": recommendation,
#             "color_class": color_class,
#             "confidence": min(score * 0.8, 100)
#         }

# # Premium Chart Renderer
# class PremiumChartRenderer:
#     @staticmethod
#     def render_premium_price_chart(data: Dict[str, Any]):
#         if "error" in data:
#             st.error("📈 Unable to load chart data")
#             return
        
#         hist = data["historical_data"]
        
#         # Create sophisticated chart
#         fig = make_subplots(
#             rows=3, cols=1,
#             shared_xaxes=True,
#             vertical_spacing=0.05,
#             subplot_titles=[
#                 f'{data["symbol"]} - Price Action & Technical Analysis', 
#                 'Volume Profile', 
#                 'RSI Momentum'
#             ],
#             row_heights=[0.6, 0.2, 0.2]
#         )
        
#         # Candlestick chart
#         fig.add_trace(
#             go.Candlestick(
#                 x=hist.index,
#                 open=hist['Open'],
#                 high=hist['High'],
#                 low=hist['Low'],
#                 close=hist['Close'],
#                 name=f'{data["symbol"]} Price',
#                 increasing_line_color='#00ff88',
#                 decreasing_line_color='#ff4444',
#                 increasing_fillcolor='rgba(0,255,136,0.7)',
#                 decreasing_fillcolor='rgba(255,68,68,0.7)'
#             ), row=1, col=1
#         )
        
#         # Moving averages
#         ma_20 = hist['Close'].rolling(20).mean()
#         ma_50 = hist['Close'].rolling(50).mean()
        
#         fig.add_trace(go.Scatter(
#             x=hist.index, y=ma_20, 
#             mode='lines', name='MA 20',
#             line=dict(color='#ffa500', width=2)
#         ), row=1, col=1)
        
#         fig.add_trace(go.Scatter(
#             x=hist.index, y=ma_50, 
#             mode='lines', name='MA 50',
#             line=dict(color='#ff6b6b', width=2)
#         ), row=1, col=1)
        
#         # Volume
#         colors = ['rgba(0,255,136,0.7)' if close > open else 'rgba(255,68,68,0.7)' 
#                  for close, open in zip(hist['Close'], hist['Open'])]
        
#         fig.add_trace(go.Bar(
#             x=hist.index, y=hist['Volume'], 
#             name='Volume', marker_color=colors,
#             showlegend=False
#         ), row=2, col=1)
        
#         # RSI
#         delta = hist['Close'].diff()
#         gain = (delta.where(delta > 0, 0)).rolling(14).mean()
#         loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
#         rs = gain / loss
#         rsi = 100 - (100 / (1 + rs))
        
#         fig.add_trace(go.Scatter(
#             x=hist.index, y=rsi, 
#             mode='lines', name='RSI',
#             line=dict(color='#9f7aea', width=3)
#         ), row=3, col=1)
        
#         # RSI levels
#         fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.8)", row=3, col=1)
#         fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.8)", row=3, col=1)
#         fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.5)", row=3, col=1)
        
#         # Update layout
#         fig.update_layout(
#             height=600,
#             template="plotly_dark",
#             showlegend=True,
#             title_text=f"{data['symbol']} - Advanced Technical Analysis",
#             title_font=dict(size=24, color='white'),
#             plot_bgcolor='rgba(0,0,0,0)',
#             paper_bgcolor='rgba(0,0,0,0)',
#             font=dict(color='white', family='Inter'),
#             legend=dict(
#                 bgcolor='rgba(255,255,255,0.1)',
#                 bordercolor='rgba(255,255,255,0.2)',
#                 borderwidth=1
#             )
#         )
        
#         fig.update_xaxes(
#             rangeslider_visible=False,
#             gridcolor='rgba(255,255,255,0.1)',
#             showgrid=True
#         )
#         fig.update_yaxes(
#             gridcolor='rgba(255,255,255,0.1)',
#             showgrid=True
#         )
        
#         st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
    
#     @staticmethod
#     def render_investment_dashboard(data: Dict[str, Any], score_data: Dict[str, Any]):
#         """Render premium investment dashboard"""
#         col1, col2, col3 = st.columns([1, 1, 1])
        
#         # Current Metrics
#         with col1:
#             st.markdown("### 📊 Current Metrics")
            
#             # Price card
#             price_change_color = "#00ff88" if data['price_change'] >= 0 else "#ff4444"
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">${data['current_price']:.2f}</div>
#                 <div class="metric-label">Current Price</div>
#                 <div class="metric-change" style="background: {price_change_color};">
#                     {data['price_change']:+.2f}%
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Volume
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{data['volume']:,}</div>
#                 <div class="metric-label">Volume</div>
#                 <div class="metric-change">
#                     {data['volume_ratio']:.1f}x Avg
#                 </div>
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Market Cap
#             market_cap_display = f"${data['market_cap']/1e9:.1f}B" if data['market_cap'] > 1e9 else f"${data['market_cap']/1e6:.1f}M"
#             st.markdown(f"""
#             <div class="metric-card">
#                 <div class="metric-value">{market_cap_display}</div>
#                 <div class="metric-label">Market Cap</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Technical Indicators
#         with col2:
#             st.markdown("### 🎯 Technical Indicators")
            
#             # RSI Gauge
#             rsi_color = "#ff4444" if data['rsi'] > 70 or data['rsi'] < 30 else "#00ff88"
#             rsi_status = "Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral"
            
#             rsi_fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=data['rsi'],
#                 domain={'x': [0, 1], 'y': [0, 1]},
#                 title={'text': "RSI", 'font': {'color': 'white', 'size': 20}},
#                 number={'font': {'color': 'white', 'size': 24}},
#                 gauge={
#                     'axis': {'range': [None, 100], 'tickcolor': 'white'},
#                     'bar': {'color': rsi_color, 'thickness': 0.3},
#                     'bgcolor': 'rgba(255,255,255,0.1)',
#                     'borderwidth': 2,
#                     'bordercolor': 'rgba(255,255,255,0.3)',
#                     'steps': [
#                         {'range': [0, 30], 'color': 'rgba(0,255,136,0.3)'},
#                         {'range': [70, 100], 'color': 'rgba(255,68,68,0.3)'}
#                     ],
#                     'threshold': {
#                         'line': {'color': "white", 'width': 4},
#                         'thickness': 0.8,
#                         'value': 50
#                     }
#                 }
#             ))
            
#             rsi_fig.update_layout(
#                 height=250,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font={'color': 'white', 'family': 'Inter'}
#             )
            
#             st.plotly_chart(rsi_fig, use_container_width=True, key=get_unique_key())
            
#             st.markdown(f"""
#             <div style="text-align: center; color: {rsi_color}; font-weight: 600; margin-top: -20px;">
#                 {rsi_status} ({data['rsi']:.1f})
#             </div>
#             """, unsafe_allow_html=True)
            
#             # Additional indicators
#             ma_trend = "🟢 Bullish" if data['current_price'] > data['ma_20'] else "🔴 Bearish"
#             vol_status = "🟡 High" if data['volatility'] > 0.5 else "🟢 Normal"
            
#             st.markdown(f"""
#             <div style="margin-top: 1rem;">
#                 <div style="margin: 0.5rem 0;"><strong>MA Trend:</strong> {ma_trend}</div>
#                 <div style="margin: 0.5rem 0;"><strong>Volatility:</strong> {vol_status} ({data['volatility']:.1%})</div>
#                 <div style="margin: 0.5rem 0;"><strong>P/E Ratio:</strong> {data['pe_ratio']:.1f}</div>
#             </div>
#             """, unsafe_allow_html=True)
        
#         # Investment Score
#         with col3:
#             st.markdown("### 💡 AI Investment Score")
            
#             # Score gauge
#             score_fig = go.Figure(go.Indicator(
#                 mode="gauge+number",
#                 value=score_data['score'],
#                 domain={'x': [0, 1], 'y': [0, 1]},
#                 title={'text': "Investment Score", 'font': {'color': 'white', 'size': 20}},
#                 number={'font': {'color': 'white', 'size': 24}},
#                 gauge={
#                     'axis': {'range': [None, 100], 'tickcolor': 'white'},
#                     'bar': {'color': '#667eea', 'thickness': 0.3},
#                     'bgcolor': 'rgba(255,255,255,0.1)',
#                     'borderwidth': 2,
#                     'bordercolor': 'rgba(255,255,255,0.3)',
#                     'steps': [
#                         {'range': [0, 40], 'color': 'rgba(245,101,101,0.3)'},
#                         {'range': [40, 70], 'color': 'rgba(237,137,54,0.3)'},
#                         {'range': [70, 100], 'color': 'rgba(72,187,120,0.3)'}
#                     ]
#                 }
#             ))
            
#             score_fig.update_layout(
#                 height=250,
#                 plot_bgcolor='rgba(0,0,0,0)',
#                 paper_bgcolor='rgba(0,0,0,0)',
#                 font={'color': 'white', 'family': 'Inter'}
#             )
            
#             st.plotly_chart(score_fig, use_container_width=True, key=get_unique_key())
            
#             # Recommendation
#             rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
#             st.markdown(f"""
#             <div class="{rec_class}" style="margin-top: -20px;">
#                 {score_data['recommendation']}
#             </div>
#             """, unsafe_allow_html=True)
            
#             st.markdown(f"""
#             <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
#                 <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
#                 <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
#             </div>
#             """, unsafe_allow_html=True)

# def main():
#     init_session_state()
    
#     # Premium Header
#     st.markdown("""
#     <div class="premium-header">
#         <h1>🤖 FinDocGPT</h1>
#         <h3>AI-Powered Financial Intelligence</h3>
#         <p>Advanced financial analysis, forecasting, and investment strategy using FinanceBench dataset and Yahoo Finance API</p>
#     </div>
#     """, unsafe_allow_html=True)
    
#     # Status Bar with Sidebar Control
#     st.markdown('<div class="status-bar">', unsafe_allow_html=True)
#     col1, col2, col3, col4, col5 = st.columns(5)
    
#     with col1:
#         st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
#     with col2:
#         openai_status = "🟢 Connected" if client else "🔴 Limited"
#         st.markdown(f"🤖 **OpenAI:** {openai_status}")
#     with col3:
#         if PDF_AVAILABLE:
#             st.markdown("📄 **Documents:** TXT, PDF, CSV")
#         else:
#             st.markdown("📄 **Documents:** TXT, CSV Only")
#     with col4:
#         # Add sidebar toggle help
#         if st.button("📊 Show Sidebar", key="show_sidebar_help"):
#             st.info("""
#             **To show the sidebar:**
#             1. Look for a '>' arrow at the top-left
#             2. Click it once to expand
#             3. Don't click again or it will collapse
            
#             **If sidebar is still hidden:**
#             - Press Ctrl+Shift+] (Windows) or Cmd+Shift+] (Mac)
#             - Or refresh the page (F5)
#             """)
#     with col5:
#         if st.button("🔄 Refresh All", key="refresh_main"):
#             st.rerun()
    
#     st.markdown('</div>', unsafe_allow_html=True)
    
#     # Premium Sidebar - Force visibility with toggle arrows
#     with st.sidebar:
#         # Add prominent expand/collapse arrows at the top
#         st.markdown("""
#         <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
#             <div style="font-size: 1.5rem; color: white;">⏪ FinDocGPT Sidebar ⏩</div>
#             <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">Click « or » arrows to collapse/expand</div>
#         </div>
#         """, unsafe_allow_html=True)
        
#         st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
#         # Add sidebar toggle button at the top
#         col1, col2, col3 = st.columns([1, 2, 1])
#         with col1:
#             if st.button("«", key="collapse_sidebar", help="Collapse sidebar"):
#                 st.info("Click the » arrow to expand sidebar again")
#         with col2:
#             st.markdown("**Control Panel**")
#         with col3:
#             if st.button("»", key="expand_sidebar", help="Expand sidebar"):
#                 st.success("Sidebar expanded!")
        
#         # Add a visible indicator that sidebar is working
#         st.success("✅ Sidebar is active!")
        
#         st.header("🎯 FinDocGPT Control Panel")
        
#         # Stock Symbol Selection
#         symbol_input = st.text_input(
#             "📊 Stock Symbol", 
#             value=st.session_state.selected_symbol,
#             help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
#         )
        
#         if symbol_input != st.session_state.selected_symbol:
#             st.session_state.selected_symbol = symbol_input.upper()
        
#         # Analysis Period
#         data_period = st.selectbox(
#             "📅 Analysis Period",
#             ["1mo", "3mo", "6mo", "1y", "2y"],
#             index=3
#         )
        
#         # Real-time Data Display
#         if st.session_state.selected_symbol:
#             with st.spinner("📡 Loading market data..."):
#                 current_data = FinancialDataHandler.get_real_time_data(
#                     st.session_state.selected_symbol, data_period
#                 )
            
#             if "error" not in current_data:
#                 st.success(f"✅ {st.session_state.selected_symbol} Data Loaded")
                
#                 # Investment Score
#                 score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
#                 # Score display
#                 st.markdown(f"""
#                 <div class="metric-card {score_data['color_class']}">
#                     <div class="metric-value">{score_data['recommendation']}</div>
#                     <div class="metric-label">AI Investment Score</div>
#                     <div class="metric-change">
#                         Score: {score_data['score']:.1f}/100 ({score_data['confidence']:.1f}% confidence)
#                     </div>
#                 </div>
#                 """, unsafe_allow_html=True)
                
#                 # Key Metrics
#                 col1, col2 = st.columns(2)
#                 with col1:
#                     price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
#                     st.metric("Price", f"${current_data['current_price']:.2f}", 
#                              f"{current_data['price_change']:+.2f}%", delta_color=price_color)
#                     st.metric("Volume", f"{current_data['volume']:,}")
#                     st.metric("Market Cap", f"${current_data['market_cap']/1e9:.1f}B")
                
#                 with col2:
#                     st.metric("P/E Ratio", f"{current_data['pe_ratio']:.1f}")
#                     st.metric("Beta", f"{current_data.get('beta', 0):.2f}")
#                     st.metric("RSI", f"{current_data['rsi']:.1f}")
#             else:
#                 st.error(f"❌ Error: {current_data['error']}")
        
#         # Market Status
#         st.markdown("---")
#         st.header("📈 Market Status")
        
#         now = datetime.now()
#         is_market_open = now.weekday() < 5 and 9 <= now.hour <= 16
        
#         if is_market_open:
#             st.success("🟢 Market is Open")
#             st.caption("NYSE: 9:30 AM - 4:00 PM ET")
#         else:
#             st.warning("🟡 Market is Closed")
#             st.caption("Next open: Monday 9:30 AM ET")
        
#         # Quick Actions
#         st.markdown("---")
#         st.header("⚡ Quick Actions")
        
#         if st.button("📊 Refresh Data", use_container_width=True):
#             st.rerun()
        
#         if st.button("💾 Save Analysis", use_container_width=True):
#             if st.session_state.current_analysis:
#                 st.session_state.analysis_history.append(st.session_state.current_analysis)
#                 st.success("Analysis saved!")
        
#         if st.button("🗑️ Clear History", use_container_width=True):
#             st.session_state.analysis_history.clear()
#             st.success("History cleared!")
        
#         # Enhanced sidebar visibility help with arrows
#         st.markdown("---")
#         st.markdown("#### 🔧 Sidebar Navigation")
#         st.markdown("""
#         **Double Arrow Controls:**
#         - **« (Left Arrow)**: Collapse sidebar  
#         - **» (Right Arrow)**: Expand sidebar
#         - **⏪ ⏩**: Visual indicators in header
#         """)
#         st.caption("Look for « » arrows at the very top-left of the page")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Main Navigation Tabs
#     tab1, tab2, tab3 = st.tabs([
#         "🔍 Stage 1: Insights & Analysis", 
#         "📈 Stage 2: Financial Forecasting", 
#         "💼 Stage 3: Investment Strategy"
#     ])
    
#     # Stage 1: Document Q&A
#     with tab1:
#         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
#         st.header("🔍 Stage 1: Document Insights & Analysis")
#         st.markdown("*Document Q&A using FinanceBench • Market Sentiment Analysis • Anomaly Detection*")
        
#         # Sub-tabs for Stage 1
#         sub_tab1, sub_tab2, sub_tab3 = st.tabs([
#             "📄 Document Q&A (FinanceBench)", 
#             "😊 Sentiment Analysis", 
#             "⚠️ Anomaly Detection"
#         ])
        
#         # Document Q&A Sub-tab
#         with sub_tab1:
#             st.subheader("📄 Financial Document Q&A - FinanceBench Dataset")
#             st.markdown("""
#             Upload financial documents or use FinanceBench dataset examples for AI-powered analysis.
#             Ask questions about earnings reports, SEC filings, and financial statements.
#             """)
            
#             col1, col2 = st.columns([2, 1])
            
#             with col1:
#                 # Document Upload
#                 uploaded_file = st.file_uploader(
#                     "📁 Upload Financial Document", 
#                     type=['txt', 'pdf', 'csv'],
#                     help="Upload earnings reports, SEC filings, or financial datasets"
#                 )
                
#                 # Text input area
#                 document_text = st.text_area(
#                     "📝 Or paste document content:",
#                     height=200,
#                     placeholder="""Example FinanceBench content:

# APPLE INC. Q4 2023 EARNINGS REPORT
# Revenue: $89.5 billion (+2.8% YoY)
# Net Income: $22.9 billion 
# Gross Margin: 45.2%
# iPhone Revenue: $43.8 billion
# Services Revenue: $22.3 billion

# Key Highlights:
# - Record Services revenue driven by App Store growth
# - iPhone 15 launch exceeded expectations  
# - Supply chain constraints improved significantly

# Risk Factors:
# - China market regulatory challenges
# - Component cost inflation"""
#                 )
                
#                 # Query Input
#                 query = st.text_input(
#                     "❓ Ask a question about the financial document:",
#                     placeholder="e.g., What was the revenue growth? What are the main risk factors?",
#                     help="Ask specific questions about financial metrics, performance, risks, or business insights"
#                 )
                
#                 # Analysis Button
#                 if st.button("🔍 Analyze Document with AI", key="analyze_doc", use_container_width=True):
#                     if (document_text.strip() or uploaded_file) and query.strip():
#                         with st.spinner("🤖 FinDocGPT is analyzing the document..."):
#                             # Process uploaded file if exists
#                             if uploaded_file:
#                                 doc_content = DocumentProcessor.extract_document_text(uploaded_file)
#                                 if "❌" in doc_content:
#                                     st.error(doc_content)
#                                     doc_content = document_text  # Fallback to text area
#                             else:
#                                 doc_content = document_text
                            
#                             # Generate AI analysis
#                             response = DocumentProcessor.process_financial_document(doc_content, query)
                            
#                             # Display results
#                             st.markdown("### 📊 FinDocGPT Analysis Results")
                            
#                             with st.container():
#                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
#                                 st.markdown(response)
#                                 st.markdown('</div>', unsafe_allow_html=True)
                            
#                             # Save to history
#                             analysis_record = {
#                                 "timestamp": datetime.now(),
#                                 "type": "Document Q&A",
#                                 "query": query,
#                                 "response": response,
#                                 "document_preview": doc_content[:200] + "..."
#                             }
                            
#                             st.session_state.analysis_history.append(analysis_record)
#                             st.session_state.current_analysis = analysis_record
                            
#                             st.success("✅ Analysis completed and saved to history!")
#                     else:
#                         st.warning("⚠️ Please provide both document content and a query for analysis.")
            
#             with col2:
#                 st.subheader("📋 Instructions")
#                 st.markdown("""
#                 **How to use Document Q&A:**
                
#                 1. **Upload** a financial document or **paste** content
#                 2. **Ask** a specific question about the document
#                 3. **Click** "Analyze Document" to get AI insights
                
#                 **Supported File Types:**
#                 - ✅ **TXT**: Financial reports, transcripts
#                 - ✅ **PDF**: 10-K filings, annual reports  
#                 - ✅ **CSV**: Financial datasets, metrics
                
#                 **Example Questions:**
#                 - What was the quarterly revenue?
#                 - What are the main risk factors?
#                 - How did expenses change YoY?
#                 - What's the cash flow situation?
#                 """)
                
#                 # Show recent analysis
#                 if st.session_state.analysis_history:
#                     st.subheader("📚 Recent Analyses")
#                     recent = [a for a in st.session_state.analysis_history if a['type'] == 'Document Q&A'][-2:]
                    
#                     for analysis in reversed(recent):
#                         with st.expander(f"Q: {analysis['query'][:30]}..."):
#                             st.caption(f"Date: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
#                             st.text_area("Response:", analysis['response'][:150] + "...", height=80, disabled=True)
        
#         # Sentiment Analysis Sub-tab
#         with sub_tab2:
#             st.subheader("😊 Market Sentiment Analysis")
#             st.markdown("Analyze sentiment from earnings calls, press releases, and financial news using advanced AI.")
            
#             if st.session_state.selected_symbol:
#                 current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
#                 if "error" not in current_data:
#                     col1, col2 = st.columns([2, 1])
                    
#                     with col1:
#                         st.markdown(f"#### 📰 Sentiment Analysis for {st.session_state.selected_symbol}")
                        
#                         # Sentiment text input
#                         sentiment_text = st.text_area(
#                             "Paste financial news, earnings call transcript, or press release:",
#                             height=200,
#                             placeholder=f"""Example content for {st.session_state.selected_symbol}:
                            
# "During the earnings call, CEO mentioned strong performance in cloud services with 35% growth. Management expressed optimism about AI initiatives and expects continued expansion. However, they noted concerns about supply chain costs and competitive pressures in mobile segment. The company raised full-year guidance but warned of potential headwinds from foreign exchange rates."

# Paste actual content here for AI-powered sentiment analysis..."""
#                         )
                        
#                         # Sentiment analysis type
#                         analysis_depth = st.selectbox(
#                             "Analysis Depth:",
#                             ["Quick Sentiment", "Detailed Analysis", "Comprehensive Report"],
#                             help="Choose the depth of sentiment analysis"
#                         )
                        
#                         if st.button("📊 Analyze Sentiment", key="analyze_sentiment", use_container_width=True):
#                             if sentiment_text.strip():
#                                 with st.spinner("🤖 Analyzing market sentiment..."):
#                                     # Create context for sentiment analysis
#                                     context = f"""
#                                     Sentiment Analysis Request for: {st.session_state.selected_symbol}
#                                     Analysis Type: {analysis_depth}
#                                     Current Stock Price: ${current_data.get('current_price', 'N/A')}
#                                     Recent Performance: {current_data.get('price_change', 'N/A')}%
                                    
#                                     Text Content: {sentiment_text[:1000]}
#                                     """
                                    
#                                     prompt = f"""
#                                     Perform {analysis_depth.lower()} sentiment analysis for {st.session_state.selected_symbol} based on the provided financial communication.
                                    
#                                     Provide:
#                                     1. Overall Sentiment (Bullish/Bearish/Neutral) with confidence score (0-100%)
#                                     2. Key positive sentiment drivers
#                                     3. Key negative sentiment factors  
#                                     4. Management tone assessment
#                                     5. Forward-looking statement analysis
#                                     6. Investment implications
#                                     7. Sentiment score breakdown by topic (if applicable)
                                    
#                                     Text to analyze: {sentiment_text}
#                                     """
                                    
#                                     sentiment_response = OpenAIHandler.generate_response(prompt, context, "sentiment")
                                    
#                                     st.markdown("### 📈 Sentiment Analysis Results")
#                                     with st.container():
#                                         st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
#                                         st.markdown(sentiment_response)
#                                         st.markdown('</div>', unsafe_allow_html=True)
                                    
#                                     st.success("✅ Sentiment analysis completed!")
#                             else:
#                                 st.warning("⚠️ Please provide text content for sentiment analysis.")
                    
#                     with col2:
#                         st.markdown("#### 🎯 Sentiment Examples")
                        
#                         sample_texts = {
#                             "Bullish Example": "Management reported exceptional quarter with record revenue growth of 45%. New product launch exceeded all expectations. Strong pipeline for next year with multiple expansion opportunities.",
                            
#                             "Bearish Example": "Company missed earnings expectations due to supply chain disruptions. Management lowered full-year guidance citing economic headwinds. Competitive pressures increasing in core markets.",
                            
#                             "Neutral Example": "Quarter met expectations with steady performance. Management maintaining current guidance. Some positive developments offset by ongoing challenges in certain segments."
#                         }
                        
#                         for sentiment_type, example in sample_texts.items():
#                             with st.expander(f"📝 {sentiment_type}"):
#                                 st.text_area("", value=example, height=80, key=f"sentiment_{sentiment_type}")
#                 else:
#                     st.error(f"Cannot load data for {st.session_state.selected_symbol}")
#             else:
#                 st.info("ℹ️ Please select a stock symbol in the sidebar to begin sentiment analysis.")
        
#         # Anomaly Detection Sub-tab
#         with sub_tab3:
#             st.subheader("⚠️ Real-Time Anomaly Detection")
#             st.markdown("Identify unusual patterns and potential risks in financial metrics and market behavior.")
            
#             if st.session_state.selected_symbol:
#                 current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
#                 if "error" not in current_data:
#                     col1, col2 = st.columns([2, 1])
                    
#                     with col1:
#                         st.markdown("### 🔍 Anomaly Detection Results")
                        
#                         # Define comprehensive anomaly detection thresholds
#                         anomalies = []
#                         warnings = []
                        
#                         # Price Movement Anomalies
#                         if abs(current_data['price_change']) > 15:
#                             anomalies.append({
#                                 "type": "Critical",
#                                 "category": "Price Movement", 
#                                 "message": f"Extreme price movement: {current_data['price_change']:+.2f}%",
#                                 "impact": "High"
#                             })
#                         elif abs(current_data['price_change']) > 8:
#                             warnings.append({
#                                 "type": "Warning",
#                                 "category": "Price Movement",
#                                 "message": f"Significant price movement: {current_data['price_change']:+.2f}%",
#                                 "impact": "Medium"
#                             })
                        
#                         # Technical Indicator Anomalies
#                         rsi = current_data['rsi']
#                         if rsi > 85:
#                             anomalies.append({
#                                 "type": "Critical",
#                                 "category": "Technical Analysis",
#                                 "message": f"Extremely overbought condition: RSI {rsi:.1f}",
#                                 "impact": "High"
#                             })
#                         elif rsi < 15:
#                             anomalies.append({
#                                 "type": "Critical", 
#                                 "category": "Technical Analysis",
#                                 "message": f"Extremely oversold condition: RSI {rsi:.1f}",
#                                 "impact": "High"
#                             })
#                         elif rsi > 75 or rsi < 25:
#                             warnings.append({
#                                 "type": "Warning",
#                                 "category": "Technical Analysis", 
#                                 "message": f"Overbought/oversold condition: RSI {rsi:.1f}",
#                                 "impact": "Medium"
#                             })
                        
#                         # Volume Anomalies
#                         volume_ratio = current_data.get('volume_ratio', 1)
#                         if volume_ratio > 5:
#                             anomalies.append({
#                                 "type": "Critical",
#                                 "category": "Volume",
#                                 "message": f"Extreme volume spike: {volume_ratio:.1f}x average volume",
#                                 "impact": "High"
#                             })
#                         elif volume_ratio > 2:
#                             warnings.append({
#                                 "type": "Warning", 
#                                 "category": "Volume",
#                                 "message": f"High volume activity: {volume_ratio:.1f}x average volume",
#                                 "impact": "Medium"
#                             })
                        
#                         # Volatility Anomalies  
#                         volatility = current_data['volatility']
#                         if volatility > 1.5:
#                             anomalies.append({
#                                 "type": "Critical",
#                                 "category": "Volatility",
#                                 "message": f"Extreme volatility: {volatility:.2%} annualized",
#                                 "impact": "High"
#                             })
#                         elif volatility > 0.8:
#                             warnings.append({
#                                 "type": "Warning",
#                                 "category": "Volatility", 
#                                 "message": f"High volatility: {volatility:.2%} annualized",
#                                 "impact": "Medium"
#                             })
                        
#                         # Display Results
#                         if anomalies:
#                             st.error("### 🚨 Critical Anomalies Detected")
#                             for anomaly in anomalies:
#                                 st.markdown(f"""
#                                 **{anomaly['category']}**: {anomaly['message']}  
#                                 *Impact: {anomaly['impact']}*
#                                 """)
                        
#                         if warnings:
#                             st.warning("### ⚠️ Warnings Detected")
#                             for warning in warnings:
#                                 st.markdown(f"""
#                                 **{warning['category']}**: {warning['message']}  
#                                 *Impact: {warning['impact']}*
#                                 """)
                        
#                         if not anomalies and not warnings:
#                             st.success("### ✅ No Significant Anomalies Detected")
#                             st.markdown("All monitored metrics are within normal ranges.")
                        
#                         # Technical Analysis Summary
#                         st.markdown("### 📊 Current Technical Status")
                        
#                         col_a, col_b, col_c, col_d = st.columns(4)
                        
#                         with col_a:
#                             price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
#                             st.metric("Price Change", f"{current_data['price_change']:+.2f}%", delta_color=price_color)
                        
#                         with col_b:
#                             rsi_status = "Overbought" if current_data['rsi'] > 70 else "Oversold" if current_data['rsi'] < 30 else "Neutral"
#                             st.metric("RSI", f"{current_data['rsi']:.1f}", rsi_status)
                        
#                         with col_c:
#                             vol_status = "High" if current_data.get('volume_ratio', 1) > 2 else "Normal"
#                             st.metric("Volume", f"{current_data['volume']:,}", f"{volume_ratio:.1f}x ({vol_status})")
                        
#                         with col_d:
#                             vol_status = "High" if current_data['volatility'] > 0.5 else "Normal"
#                             st.metric("Volatility", f"{current_data['volatility']:.2%}", vol_status)
                    
#                     with col2:
#                         st.markdown("#### 🎯 Anomaly Monitoring")
                        
#                         # Monitoring thresholds
#                         st.markdown("**Current Thresholds:**")
#                         thresholds = {
#                             "Price Movement": "> ±8% (Warning), > ±15% (Critical)",
#                             "RSI": "< 25 or > 75 (Warning), < 15 or > 85 (Critical)",
#                             "Volume": "> 2x avg (Warning), > 5x avg (Critical)", 
#                             "Volatility": "> 80% (Warning), > 150% (Critical)"
#                         }
                        
#                         for metric, threshold in thresholds.items():
#                             st.caption(f"**{metric}**: {threshold}")
                        
#                         st.markdown("---")
#                         st.markdown("#### 📈 Real-time Status")
                        
#                         # Current values with status indicators
#                         metrics_status = [
#                             ("Price Change", f"{current_data['price_change']:+.2f}%", 
#                              "🟢" if abs(current_data['price_change']) < 3 else "🟡" if abs(current_data['price_change']) < 8 else "🔴"),
#                             ("RSI", f"{current_data['rsi']:.1f}", 
#                              "🟢" if 40 <= current_data['rsi'] <= 60 else "🟡" if 25 <= current_data['rsi'] <= 75 else "🔴"),
#                             ("Volume Ratio", f"{current_data.get('volume_ratio', 1):.1f}x", 
#                              "🟢" if current_data.get('volume_ratio', 1) < 1.5 else "🟡" if current_data.get('volume_ratio', 1) < 3 else "🔴"),
#                             ("Volatility", f"{current_data['volatility']:.1%}", 
#                              "🟢" if current_data['volatility'] < 0.4 else "🟡" if current_data['volatility'] < 0.8 else "🔴")
#                         ]
                        
#                         for metric, value, status in metrics_status:
#                             st.markdown(f"{status} **{metric}**: {value}")
                        
#                         # Auto-refresh option
#                         st.markdown("---")
#                         auto_refresh = st.checkbox("🔄 Auto-refresh (30s)", key="auto_refresh_anomaly")
                        
#                         if auto_refresh:
#                             time.sleep(30)
#                             st.rerun()
#                 else:
#                     st.error(f"Cannot load data for {st.session_state.selected_symbol}")
#             else:
#                 st.info("ℹ️ Please select a stock symbol in the sidebar to begin anomaly detection.")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Stage 2: Financial Forecasting
#     with tab2:
#         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
#         st.header("📈 Stage 2: Financial Forecasting")
#         st.markdown("*AI-powered predictions • Advanced technical analysis • Market data integration*")
        
#         if st.session_state.selected_symbol:
#             current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
#             if "error" not in current_data:
#                 # Premium Chart
#                 st.subheader("📊 Advanced Technical Analysis")
#                 with st.container():
#                     st.markdown('<div class="chart-container">', unsafe_allow_html=True)
#                     PremiumChartRenderer.render_premium_price_chart(current_data)
#                     st.markdown('</div>', unsafe_allow_html=True)
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.subheader("🎯 AI-Powered Financial Forecasting")
                    
#                     # Forecasting parameters
#                     col_a, col_b = st.columns(2)
                    
#                     with col_a:
#                         forecast_period = st.selectbox(
#                             "📅 Forecast Period:",
#                             ["1 Week", "1 Month", "3 Months", "6 Months"],
#                             index=1
#                         )
                    
#                     with col_b:
#                         forecast_type = st.selectbox(
#                             "📊 Analysis Type:",
#                             ["Technical Analysis", "Combined Analysis"],
#                             index=1
#                         )
                    
#                     # Generate forecast button
#                     if st.button("🔮 Generate AI Forecast", key="generate_forecast", use_container_width=True):
#                         with st.spinner("🤖 Generating advanced forecast..."):
                            
#                             context = f"""
#                             FORECASTING CONTEXT for {st.session_state.selected_symbol}
                            
#                             Current Data:
#                             - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
#                             - Volume: {current_data['volume']:,}
#                             - RSI: {current_data['rsi']:.1f}
#                             - MA 20: ${current_data['ma_20']:.2f}
#                             - MA 50: ${current_data['ma_50']:.2f}
#                             - Volatility: {current_data['volatility']:.2%}
#                             - Sector: {current_data['company_info']['sector']}
#                             """
                            
#                             prompt = f"""
#                             Generate a comprehensive {forecast_period} forecast for {st.session_state.selected_symbol} using {forecast_type.lower()}.
                            
#                             Provide detailed analysis including:
#                             1. **Price Targets**: Bull case, base case, and bear case scenarios with specific prices and probabilities
#                             2. **Key Catalysts**: List 3-5 positive drivers and 3-5 potential risks
#                             3. **Technical Analysis**: Support/resistance levels, momentum indicators, and chart patterns
#                             4. **Timeline**: Expected price movement schedule with key milestones
#                             5. **Confidence Assessment**: Overall confidence level (0-100%) and reliability factors
#                             6. **Risk Management**: Suggested stop-loss levels and position sizing recommendations
#                             """
                            
#                             forecast_response = OpenAIHandler.generate_response(prompt, context, "forecasting")
                            
#                             st.markdown("### 🔮 AI Forecasting Results")
#                             with st.container():
#                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
#                                 st.markdown(forecast_response)
#                                 st.markdown('</div>', unsafe_allow_html=True)
                            
#                             st.success("✅ Advanced forecast generated!")
                
#                 with col2:
#                     st.subheader("🎯 Forecasting Guide")
                    
#                     st.markdown("""
#                     **Forecast Types:**
#                     - **Technical**: Price patterns, indicators
#                     - **Combined**: Technical + Fundamental
                    
#                     **Confidence Levels:**
#                     - **Conservative**: Lower risk, modest returns
#                     - **Moderate**: Balanced risk/reward  
#                     - **Aggressive**: Higher risk, higher potential
#                     """)
                    
#                     # Current indicators
#                     st.subheader("📊 Current Indicators")
                    
#                     indicators = [
#                         ("Price Trend", "🟢 Up" if current_data['price_change'] > 0 else "🔴 Down"),
#                         ("RSI Status", "🟢 Neutral" if 30 < current_data['rsi'] < 70 else "🟡 Extreme"),
#                         ("Volume", "🟢 Normal" if current_data.get('volume_ratio', 1) < 2 else "🟡 High"),
#                         ("Volatility", "🟢 Low" if current_data['volatility'] < 0.5 else "🟡 High")
#                     ]
                    
#                     for indicator, status in indicators:
#                         st.markdown(f"**{indicator}**: {status}")
#             else:
#                 st.error(f"Cannot load data for {st.session_state.selected_symbol}")
#         else:
#             st.info("Please select a stock symbol in the sidebar to begin forecasting.")
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Stage 3: Investment Strategy
#     with tab3:
#         st.markdown('<div class="premium-card">', unsafe_allow_html=True)
#         st.header("💼 Stage 3: Investment Strategy & Decision-Making")
#         st.markdown("*Professional portfolio management • Advanced risk assessment • AI-powered trading recommendations*")
        
#         if st.session_state.selected_symbol:
#             current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
#             if "error" not in current_data:
#                 score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
#                 # Premium Investment Dashboard
#                 st.subheader("📊 Professional Investment Dashboard")
#                 PremiumChartRenderer.render_investment_dashboard(current_data, score_data)
                
#                 col1, col2 = st.columns([2, 1])
                
#                 with col1:
#                     st.subheader("🎯 AI Investment Strategy Generator")
                    
#                     # Strategy parameters
#                     col_a, col_b = st.columns(2)
                    
#                     with col_a:
#                         investment_horizon = st.selectbox(
#                             "📅 Investment Horizon:",
#                             ["Short-term (1-3 months)", "Medium-term (3-12 months)", "Long-term (1+ years)"],
#                             index=1
#                         )
                    
#                     with col_b:
#                         risk_tolerance = st.selectbox(
#                             "⚖️ Risk Tolerance:",
#                             ["Conservative", "Moderate", "Aggressive"],
#                             index=1
#                         )
                    
#                     # Generate strategy button
#                     if st.button("💡 Generate Professional Investment Strategy", key="generate_strategy", use_container_width=True):
#                         with st.spinner("🤖 Generating comprehensive investment strategy..."):
                            
#                             context = f"""
#                             INVESTMENT STRATEGY CONTEXT for {st.session_state.selected_symbol}
                            
#                             Market Data:
#                             - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
#                             - P/E Ratio: {current_data['pe_ratio']:.1f}
#                             - Beta: {current_data.get('beta', 0):.2f}
#                             - Market Cap: ${current_data['market_cap']/1e9:.1f}B
#                             - RSI: {current_data['rsi']:.1f}
#                             - Volatility: {current_data['volatility']:.2%}
#                             - Investment Score: {score_data['score']:.1f}/100
#                             - AI Recommendation: {score_data['recommendation']}
                            
#                             Parameters:
#                             - Investment Horizon: {investment_horizon}
#                             - Risk Tolerance: {risk_tolerance}
#                             - Portfolio Value: ${st.session_state.portfolio_value:,.0f}
#                             """
                            
#                             prompt = f"""
#                             Create a comprehensive, professional-grade investment strategy for {st.session_state.selected_symbol}.
                            
#                             Provide detailed analysis including:
                            
#                             1. **Executive Summary & Investment Decision**:
#                                - Clear BUY/SELL/HOLD recommendation with detailed rationale
#                                - Investment thesis in 2-3 sentences
#                                - Expected return potential and timeframe
                            
#                             2. **Entry Strategy & Timing**:
#                                - Optimal entry price points and conditions
#                                - Dollar-cost averaging vs. lump sum recommendations
#                                - Market timing considerations and triggers
                            
#                             3. **Risk Management Framework**:
#                                - Stop-loss levels with specific prices
#                                - Position sizing as % of portfolio (given ${st.session_state.portfolio_value:,.0f} portfolio)
#                                - Risk/reward ratio analysis
#                                - Hedging strategies if applicable
                            
#                             4. **Exit Strategy & Profit Taking**:
#                                - Primary profit target prices
#                                - Partial profit-taking levels
#                                - Review and rebalancing triggers
                            
#                             5. **Portfolio Integration**:
#                                - How this position fits into overall allocation
#                                - Diversification impact and sector exposure
#                                - Correlation with existing holdings
                            
#                             6. **Monitoring & Review Framework**:
#                                - Key metrics to track weekly/monthly
#                                - Fundamental and technical review schedule
#                                - Scenario planning (bull/bear case actions)
                            
#                             Consider the {risk_tolerance.lower()} risk profile and {investment_horizon} timeframe in all recommendations.
#                             Provide specific, actionable advice that a professional investment advisor would give.
#                             """
                            
#                             investment_strategy = OpenAIHandler.generate_response(prompt, context, "investment")
                            
#                             st.markdown("### 💼 Professional Investment Strategy")
#                             with st.container():
#                                 st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
#                                 st.markdown(investment_strategy)
#                                 st.markdown('</div>', unsafe_allow_html=True)
                            
#                             # Add to portfolio option
#                             if "BUY" in investment_strategy.upper():
#                                 if st.button("➕ Add to Portfolio", key="add_to_portfolio"):
#                                     portfolio_item = {
#                                         "symbol": st.session_state.selected_symbol,
#                                         "action": score_data['recommendation'],
#                                         "price": current_data['current_price'],
#                                         "timestamp": datetime.now(),
#                                         "strategy": investment_strategy[:200] + "...",
#                                         "score": score_data['score']
#                                     }
#                                     st.session_state.investment_portfolio.append(portfolio_item)
#                                     st.success(f"✅ {st.session_state.selected_symbol} added to portfolio!")
                            
#                             st.success("✅ Professional investment strategy generated!")
                
#                 with col2:
#                     st.subheader("💡 Strategy Guide")
                    
#                     st.markdown("""
#                     **Investment Horizons:**
#                     - **Short-term**: Technical focus, quick profits
#                     - **Medium-term**: Balanced approach
#                     - **Long-term**: Fundamental focus, compound growth
                    
#                     **Risk Levels:**
#                     - **Conservative**: Capital preservation priority
#                     - **Moderate**: Balanced risk/return
#                     - **Aggressive**: Growth maximization
#                     """)
                    
#                     # Quick recommendation
#                     st.subheader("🎯 Quick AI Recommendation")
                    
#                     rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
#                     st.markdown(f"""
#                     <div class="{rec_class}">
#                         {score_data['recommendation']}
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     st.markdown(f"""
#                     <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
#                         <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
#                         <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
#                     </div>
#                     """, unsafe_allow_html=True)
                    
#                     # Portfolio summary - Only show if there are actual positions
#                     if st.session_state.investment_portfolio:
#                         st.subheader("💼 Portfolio Summary")
#                         st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.0f}")
#                         st.metric("Total Positions", len(st.session_state.investment_portfolio))
                        
#                         st.markdown("**Recent Positions:**")
#                         recent = st.session_state.investment_portfolio[-3:]
#                         for item in reversed(recent):
#                             st.markdown(f"• {item['symbol']} - {item['action']} (Score: {item['score']:.1f})")
#                     else:
#                         st.subheader("💼 Portfolio")
#                         st.info("No positions yet. Generate investment strategies to build your portfolio!")
#             else:
#                 st.error(f"Cannot load data for {st.session_state.selected_symbol}")
#         else:
#             st.info("Please select a stock symbol in the sidebar to begin investment analysis.")
        
#         # Portfolio Management Section
#         if st.session_state.investment_portfolio:
#             st.markdown("---")
#             st.subheader("📊 Portfolio Management")
            
#             # Portfolio table
#             portfolio_df = pd.DataFrame(st.session_state.investment_portfolio)
#             portfolio_df['timestamp'] = pd.to_datetime(portfolio_df['timestamp'])
#             portfolio_df = portfolio_df.sort_values('timestamp', ascending=False)
            
#             # Display portfolio
#             st.dataframe(
#                 portfolio_df[['symbol', 'action', 'price', 'score', 'timestamp']],
#                 use_container_width=True,
#                 hide_index=True
#             )
            
#             # Portfolio actions
#             col1, col2, col3 = st.columns(3)
            
#             with col1:
#                 if st.button("📈 Portfolio Analytics", use_container_width=True):
#                     st.info("Advanced portfolio analytics - Coming soon!")
            
#             with col2:
#                 if st.button("📊 Performance Report", use_container_width=True):
#                     st.info("Performance reporting - Coming soon!")
            
#             with col3:
#                 if st.button("🗑️ Clear Portfolio", use_container_width=True):
#                     st.session_state.investment_portfolio.clear()
#                     st.success("Portfolio cleared!")
#                     st.rerun()
        
#         st.markdown('</div>', unsafe_allow_html=True)
    
#     # Footer Status
#     st.markdown("---")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if not PDF_AVAILABLE:
#             st.error("""
#             🚨 **PDF Processing Not Available**
#             Install: `pip install PyPDF2 pdfplumber`
#             """)
#         else:
#             st.success("✅ All document formats supported (TXT, PDF, CSV)")
    
#     with col2:
#         if not client:
#             st.warning("⚠️ OpenAI API not configured. Set OPENAI_API_KEY environment variable for full AI capabilities.")
#         else:
#             st.success("✅ AI analysis fully enabled")

# # Run the application
# if __name__ == "__main__":
#     main()


"""
FinDocGPT - AI-Powered Financial Intelligence
Premium Professional Interface - AkashX.ai Hackathon Solution
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
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
import re

# PDF processing imports
try:
    import PyPDF2
    import pdfplumber
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

# Configure Streamlit
st.set_page_config(
    page_title="FinDocGPT - AI Financial Intelligence",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium CSS Styling
st.markdown("""
<style>
/* Import Google Fonts */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* Global Styling */
.main {
    font-family: 'Inter', sans-serif;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
}

/* Force sidebar to be visible */
.css-1d391kg {
    background: linear-gradient(180deg, #1a202c 0%, #2d3748 100%);
    min-width: 300px !important;
    max-width: 350px !important;
}

/* Sidebar visibility fix */
[data-testid="stSidebar"] {
    display: block !important;
    visibility: visible !important;
    opacity: 1 !important;
    transform: translateX(0) !important;
}

/* Prevent sidebar from collapsing */
[data-testid="stSidebar"][aria-expanded="false"] {
    display: block !important;
    transform: translateX(0) !important;
}

/* Header Styling */
.premium-header {
    background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
    padding: 2rem;
    border-radius: 20px;
    margin-bottom: 2rem;
    text-align: center;
    color: white;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    position: relative;
    overflow: hidden;
}

.premium-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: url('data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><defs><pattern id="grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M 10 0 L 0 0 0 10" fill="none" stroke="rgba(255,255,255,0.1)" stroke-width="0.5"/></pattern></defs><rect width="100" height="100" fill="url(%23grid)"/></svg>');
    opacity: 0.3;
}

.premium-header h1 {
    font-size: 3rem;
    font-weight: 700;
    margin: 0;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

.premium-header h3 {
    font-size: 1.2rem;
    font-weight: 400;
    margin: 0.5rem 0;
    opacity: 0.9;
    position: relative;
    z-index: 1;
}

.premium-header p {
    font-size: 1rem;
    margin: 1rem 0 0 0;
    opacity: 0.8;
    position: relative;
    z-index: 1;
}

/* Status Bar */
.status-bar {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(10px);
    padding: 1rem;
    border-radius: 15px;
    margin-bottom: 2rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
}

.status-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-weight: 500;
    color: #2d3748;
}

/* Card Styling */
.premium-card {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    border: 1px solid rgba(255,255,255,0.2);
    margin-bottom: 2rem;
    transition: all 0.3s ease;
}

.premium-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 25px 50px rgba(0,0,0,0.15);
}

/* Metric Cards */
.metric-card {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 1.5rem;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0 10px 30px rgba(102,126,234,0.3);
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 15px 40px rgba(102,126,234,0.4);
}

.metric-value {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
}

.metric-label {
    font-size: 0.9rem;
    opacity: 0.9;
    font-weight: 500;
}

.metric-change {
    font-size: 0.8rem;
    margin-top: 0.5rem;
    padding: 0.3rem 0.6rem;
    border-radius: 20px;
    background: rgba(255,255,255,0.2);
}

/* Sidebar Styling */
.sidebar-content {
    color: white;
}

/* Investment Score */
.score-excellent {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
}

.score-good {
    background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%);
}

.score-fair {
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
}

.score-poor {
    background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
}

/* Analysis Box */
.analysis-box {
    background: rgba(255, 255, 255, 0.95);
    backdrop-filter: blur(20px);
    padding: 2rem;
    border-radius: 20px;
    border: 1px solid rgba(255,255,255,0.2);
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    margin: 2rem 0;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 2rem;
    font-weight: 600;
    font-size: 1rem;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.4);
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 15px;
    padding: 0.5rem;
    backdrop-filter: blur(10px);
}

.stTabs [data-baseweb="tab"] {
    background: transparent;
    border-radius: 10px;
    color: white;
    font-weight: 500;
    padding: 1rem 2rem;
    transition: all 0.3s ease;
}

.stTabs [aria-selected="true"] {
    background: rgba(255, 255, 255, 0.2);
    color: white;
}

/* Charts */
.chart-container {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 20px 40px rgba(0,0,0,0.1);
    backdrop-filter: blur(20px);
}

/* File Upload */
.uploadedFile {
    background: rgba(255, 255, 255, 0.1);
    border: 2px dashed rgba(255, 255, 255, 0.3);
    border-radius: 15px;
    padding: 2rem;
    text-align: center;
    color: white;
}

/* Recommendations */
.recommendation-buy {
    background: linear-gradient(135deg, #48bb78 0%, #38a169 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 10px 30px rgba(72,187,120,0.3);
}

.recommendation-strong-buy {
    background: linear-gradient(135deg, #38a169 0%, #2f855a 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 10px 30px rgba(56,161,105,0.3);
}

.recommendation-hold {
    background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 10px 30px rgba(237,137,54,0.3);
}

.recommendation-sell {
    background: linear-gradient(135deg, #f56565 0%, #e53e3e 100%);
    color: white;
    padding: 1rem 2rem;
    border-radius: 15px;
    text-align: center;
    font-weight: 600;
    font-size: 1.1rem;
    box-shadow: 0 10px 30px rgba(245,101,101,0.3);
}

/* Progress Bars */
.stProgress .st-bo {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    height: 12px;
    border-radius: 6px;
}

/* Text Input */
.stTextInput > div > div > input {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: white;
    backdrop-filter: blur(10px);
}

/* Select Box */
.stSelectbox > div > div > select {
    background: rgba(255, 255, 255, 0.1);
    border: 1px solid rgba(255, 255, 255, 0.2);
    border-radius: 10px;
    color: white;
    backdrop-filter: blur(10px);
}

/* Hide Streamlit Branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
</style>

<script>
// Force sidebar to stay visible
setTimeout(function() {
    const sidebar = document.querySelector('[data-testid="stSidebar"]');
    if (sidebar) {
        sidebar.style.display = 'block';
        sidebar.style.transform = 'translateX(0)';
        sidebar.style.visibility = 'visible';
        sidebar.style.opacity = '1';
    }
}, 1000);
</script>
""", unsafe_allow_html=True)

# Configuration
@dataclass
class Config:
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    DEFAULT_MODEL: str = "gpt-4"
    MAX_TOKENS: int = 2000
    TEMPERATURE: float = 0.3

config = Config()

# Initialize OpenAI client
client = None
if config.OPENAI_API_KEY:
    try:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
    except Exception as e:
        st.warning(f"⚠️ OpenAI initialization error: {str(e)}")
        client = None

# Session State Management
def init_session_state():
    defaults = {
        "selected_symbol": "AAPL",
        "analysis_history": [],
        "investment_portfolio": [],
        "portfolio_value": 100000,
        "current_analysis": None,
        "chart_counter": 0
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def get_unique_key():
    st.session_state.chart_counter += 1
    return f"component_{st.session_state.chart_counter}_{uuid.uuid4().hex[:8]}"

# Enhanced OpenAI Handler
class OpenAIHandler:
    @staticmethod
    def generate_response(prompt: str, context: str = "", analysis_type: str = "general") -> str:
        if not client:
            return OpenAIHandler._fallback_analysis(prompt, analysis_type)
        
        system_prompts = {
            "document_qa": "You are FinDocGPT, a specialized financial document analyst. Provide precise, data-driven answers with specific financial metrics and insights.",
            "sentiment": "You are a market sentiment analyst. Analyze financial communications to determine market sentiment with confidence scores.",
            "forecasting": "You are a quantitative financial forecasting expert. Provide specific price targets with timeframes and probability ranges.",
            "investment": "You are a senior investment strategist. Provide clear BUY/SELL/HOLD recommendations with entry/exit points and risk management.",
            "general": "You are FinDocGPT, an AI financial intelligence assistant. Provide comprehensive financial analysis."
        }
        
        system_prompt = system_prompts.get(analysis_type, system_prompts["general"])
        
        # Context length management
        max_context_chars = 6000
        if len(context) > max_context_chars:
            context = context[:max_context_chars] + "\n[Content truncated...]"
        if len(prompt) > max_context_chars:
            prompt = prompt[:max_context_chars] + "\n[Query truncated...]"
        
        try:
            response = client.chat.completions.create(
                model=config.DEFAULT_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Context: {context}\n\nRequest: {prompt}"}
                ],
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE
            )
            return response.choices[0].message.content
        except Exception as e:
            if "context_length_exceeded" in str(e).lower():
                return "⚠️ Document too large for analysis. Please try with a smaller section."
            return f"⚠️ AI analysis error: {str(e)}"
    
    @staticmethod
    def _fallback_analysis(prompt: str, analysis_type: str) -> str:
        fallbacks = {
            "document_qa": "📊 Document analysis requires OpenAI API configuration.",
            "sentiment": "😊 Sentiment analysis shows neutral market conditions.",
            "forecasting": "📈 Basic forecasting suggests following current trends.",
            "investment": "💼 Conservative HOLD recommendation based on available data.",
            "general": "🤖 Limited analysis without OpenAI API key."
        }
        return fallbacks.get(analysis_type, fallbacks["general"])

# Enhanced Document Processor
class DocumentProcessor:
    @staticmethod
    def extract_document_text(uploaded_file) -> str:
        try:
            file_type = uploaded_file.type
            file_name = uploaded_file.name.lower()
            
            if file_type == "text/plain" or file_name.endswith('.txt'):
                return DocumentProcessor._process_txt_file(uploaded_file)
            elif file_type == "application/pdf" or file_name.endswith('.pdf'):
                if PDF_AVAILABLE:
                    return DocumentProcessor._process_pdf_file(uploaded_file)
                else:
                    return "❌ PDF processing unavailable. Install: pip install PyPDF2 pdfplumber"
            elif file_type in ["text/csv", "application/vnd.ms-excel"] or file_name.endswith('.csv'):
                return DocumentProcessor._process_csv_file(uploaded_file)
            else:
                return f"❌ Unsupported file type: {file_type}"
        except Exception as e:
            return f"❌ Error processing file: {str(e)}"
    
    @staticmethod
    def _process_pdf_file(uploaded_file) -> str:
        try:
            pdf_bytes = uploaded_file.read()
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                text_pages = []
                
                for i, page in enumerate(pdf.pages[:10]):  # Limit pages
                    page_text = page.extract_text()
                    if page_text:
                        text_pages.append(page_text[:1500])  # Limit page content
                
                if text_pages:
                    full_text = "\n\n".join(text_pages)
                    if len(full_text) > 15000:
                        full_text = full_text[:15000] + "\n[Document truncated...]"
                    
                    financial_metrics = DocumentProcessor._extract_financial_metrics(full_text)
                    
                    return f"""📄 PDF Document Analysis:
📊 File Statistics:
- Pages processed: {len(text_pages)}/{len(pdf.pages)}
- Characters: {len(full_text):,}

💰 Financial Data:
{DocumentProcessor._format_financial_metrics(financial_metrics)}

📝 Content:
{full_text}"""
                else:
                    return "❌ Could not extract text from PDF."
        except Exception as e:
            return f"❌ PDF processing error: {str(e)}"
    
    @staticmethod
    def _process_txt_file(uploaded_file) -> str:
        try:
            encodings = ['utf-8', 'latin-1', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    uploaded_file.seek(0)
                    content = uploaded_file.read().decode(encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                return "❌ Could not decode text file."
            
            if len(content) > 15000:
                content = content[:15000] + "\n[Content truncated...]"
            
            financial_metrics = DocumentProcessor._extract_financial_metrics(content)
            
            return f"""📄 TXT Document Analysis:
📊 Statistics: {len(content):,} characters, {len(content.split()):,} words

💰 Financial Data:
{DocumentProcessor._format_financial_metrics(financial_metrics)}

📝 Content:
{content}"""
        except Exception as e:
            return f"❌ TXT processing error: {str(e)}"
    
    @staticmethod
    def _process_csv_file(uploaded_file) -> str:
        try:
            df = pd.read_csv(uploaded_file)
            
            if len(df) > 500:
                df = df.head(500)
                note = f"\n[Showing first 500 rows of {len(df)} total]"
            else:
                note = ""
            
            financial_keywords = ['revenue', 'sales', 'income', 'profit', 'cost', 'price', 'amount', 'value']
            financial_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in financial_keywords)]
            
            return f"""📊 CSV Dataset Analysis:
📈 Overview: {len(df)} rows, {len(df.columns)} columns
💰 Financial columns: {', '.join(financial_cols[:5]) if financial_cols else 'None detected'}

📋 Sample Data:
{df.head().to_string()}

📝 Full Dataset:
{df.to_string()[:8000]}{note}"""
        except Exception as e:
            return f"❌ CSV processing error: {str(e)}"
    
    @staticmethod
    def _extract_financial_metrics(text: str) -> Dict[str, float]:
        metrics = {}
        patterns = {
            'revenue': r'(?:revenue|sales)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'net_income': r'net\s+(?:income|earnings)[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'assets': r'total\s+assets[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?',
            'cash': r'cash[:\s]+\$?\s*([\d,]+\.?\d*)\s*(?:million|billion|m|b)?'
        }
        
        text_lower = text.lower()
        for metric, pattern in patterns.items():
            matches = re.findall(pattern, text_lower)
            if matches:
                try:
                    value = float(matches[0].replace(',', ''))
                    if 'billion' in text_lower or ' b' in text_lower:
                        value *= 1_000_000_000
                    elif 'million' in text_lower or ' m' in text_lower:
                        value *= 1_000_000
                    metrics[metric] = value
                except ValueError:
                    continue
        return metrics
    
    @staticmethod
    def _format_financial_metrics(metrics: Dict[str, float]) -> str:
        if not metrics:
            return "No quantifiable metrics detected"
        
        formatted = []
        for metric, value in metrics.items():
            name = metric.replace('_', ' ').title()
            if value >= 1_000_000_000:
                formatted.append(f"• {name}: ${value/1_000_000_000:.1f}B")
            elif value >= 1_000_000:
                formatted.append(f"• {name}: ${value/1_000_000:.1f}M")
            else:
                formatted.append(f"• {name}: ${value:,.0f}")
        return "\n".join(formatted)
    
    @staticmethod
    def process_financial_document(document_text: str, query: str) -> str:
        if not document_text.strip():
            return "❌ No document content provided."
        
        if document_text.startswith("❌"):
            return document_text
        
        context = f"Document Analysis: {document_text[:4000]}..."
        prompt = f"Analyze this financial document and answer: {query}\n\nDocument: {document_text[:6000]}..."
        
        try:
            response = OpenAIHandler.generate_response(prompt, context, "document_qa")
            return f"## 📋 Financial Document Analysis\n\n{response}"
        except Exception as e:
            return f"❌ Analysis error: {str(e)}"

# Enhanced Financial Data Handler
class FinancialDataHandler:
    @staticmethod
    def get_real_time_data(symbol: str, period: str = "1y") -> Dict[str, Any]:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            hist = ticker.history(period=period)
            
            if hist.empty:
                return {"error": f"No data available for {symbol}"}
            
            current_price = float(info.get('currentPrice', hist['Close'].iloc[-1]))
            prev_close = float(hist['Close'].iloc[-2]) if len(hist) > 1 else current_price
            price_change = ((current_price - prev_close) / prev_close) * 100
            
            # Technical indicators
            delta = hist['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            ma_20 = hist['Close'].rolling(20).mean()
            ma_50 = hist['Close'].rolling(50).mean()
            
            volatility = hist['Close'].pct_change().std() * np.sqrt(252)
            avg_volume = hist['Volume'].mean()
            volume_ratio = hist['Volume'].iloc[-1] / avg_volume if avg_volume > 0 else 1
            
            return {
                "symbol": symbol,
                "current_price": current_price,
                "price_change": price_change,
                "volume": int(hist['Volume'].iloc[-1]),
                "market_cap": int(info.get('marketCap', 0)),
                "pe_ratio": float(info.get('trailingPE', 0)) if info.get('trailingPE') else 0,
                "beta": float(info.get('beta', 0)) if info.get('beta') else 0,
                "rsi": float(rsi.iloc[-1]) if not pd.isna(rsi.iloc[-1]) else 50,
                "ma_20": float(ma_20.iloc[-1]) if not pd.isna(ma_20.iloc[-1]) else current_price,
                "ma_50": float(ma_50.iloc[-1]) if not pd.isna(ma_50.iloc[-1]) else current_price,
                "volatility": float(volatility),
                "volume_ratio": float(volume_ratio),
                "historical_data": hist,
                "company_info": {
                    "sector": info.get('sector', 'Unknown'),
                    "industry": info.get('industry', 'Unknown'),
                    "description": info.get('longBusinessSummary', 'No description')[:200] + "..."
                }
            }
        except Exception as e:
            return {"error": str(e)}
    
    @staticmethod
    def calculate_investment_score(data: Dict[str, Any]) -> Dict[str, Any]:
        if "error" in data:
            return {"error": "Cannot calculate score"}
        
        # Scoring factors
        score = 50  # Base score
        
        # Price momentum
        if data['price_change'] > 5:
            score += 15
        elif data['price_change'] > 0:
            score += 10
        elif data['price_change'] > -5:
            score += 5
        
        # RSI
        rsi = data['rsi']
        if 40 <= rsi <= 60:
            score += 15
        elif 30 <= rsi <= 70:
            score += 10
        else:
            score += 5
        
        # Trend
        if data['current_price'] > data['ma_20'] > data['ma_50']:
            score += 15
        elif data['current_price'] > data['ma_20']:
            score += 10
        
        # Valuation
        pe = data['pe_ratio']
        if 10 < pe < 20:
            score += 10
        elif 5 < pe < 30:
            score += 5
        
        # Risk
        if data['volatility'] < 0.3:
            score += 10
        elif data['volatility'] < 0.5:
            score += 5
        
        # Determine recommendation
        if score >= 80:
            recommendation = "STRONG BUY"
            color_class = "score-excellent"
        elif score >= 65:
            recommendation = "BUY"
            color_class = "score-good"
        elif score >= 50:
            recommendation = "HOLD"
            color_class = "score-fair"
        else:
            recommendation = "SELL"
            color_class = "score-poor"
        
        return {
            "score": min(score, 100),
            "recommendation": recommendation,
            "color_class": color_class,
            "confidence": min(score * 0.8, 100)
        }

# Premium Chart Renderer - FIXED VERSION
class PremiumChartRenderer:
    @staticmethod
    def render_premium_price_chart(data: Dict[str, Any], forecast_period: str = None):
        """Fixed version that accepts optional forecast_period parameter"""
        if "error" in data:
            st.error("📈 Unable to load chart data")
            return
        
        hist = data["historical_data"]
        
        # Create sophisticated chart
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=[
                f'{data["symbol"]} - Price Action & Technical Analysis', 
                'Volume Profile', 
                'RSI Momentum'
            ],
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=hist.index,
                open=hist['Open'],
                high=hist['High'],
                low=hist['Low'],
                close=hist['Close'],
                name=f'{data["symbol"]} Price',
                increasing_line_color='#00ff88',
                decreasing_line_color='#ff4444',
                increasing_fillcolor='rgba(0,255,136,0.7)',
                decreasing_fillcolor='rgba(255,68,68,0.7)'
            ), row=1, col=1
        )
        
        # Moving averages
        ma_20 = hist['Close'].rolling(20).mean()
        ma_50 = hist['Close'].rolling(50).mean()
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma_20, 
            mode='lines', name='MA 20',
            line=dict(color='#ffa500', width=2)
        ), row=1, col=1)
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=ma_50, 
            mode='lines', name='MA 50',
            line=dict(color='#ff6b6b', width=2)
        ), row=1, col=1)
        
        # Add forecast projection if forecast_period is provided
        if forecast_period:
            # Simple forecast based on trend (you can enhance this with more sophisticated models)
            last_close = hist['Close'].iloc[-1]
            forecast_days = {"1 Week": 7, "1 Month": 30, "3 Months": 90, "6 Months": 180}.get(forecast_period, 30)
            
            # Simple trend-based forecast
            trend = (hist['Close'].iloc[-1] - hist['Close'].iloc[-30]) / 30
            forecast_dates = pd.date_range(start=hist.index[-1] + pd.Timedelta(days=1), periods=forecast_days, freq='D')
            forecast_prices = [last_close + trend * i for i in range(1, forecast_days + 1)]
            
            fig.add_trace(go.Scatter(
                x=forecast_dates, y=forecast_prices,
                mode='lines', name=f'Forecast ({forecast_period})',
                line=dict(color='#9f7aea', width=2, dash='dash'),
                opacity=0.7
            ), row=1, col=1)
        
        # Volume
        colors = ['rgba(0,255,136,0.7)' if close > open else 'rgba(255,68,68,0.7)' 
                 for close, open in zip(hist['Close'], hist['Open'])]
        
        fig.add_trace(go.Bar(
            x=hist.index, y=hist['Volume'], 
            name='Volume', marker_color=colors,
            showlegend=False
        ), row=2, col=1)
        
        # RSI
        delta = hist['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(go.Scatter(
            x=hist.index, y=rsi, 
            mode='lines', name='RSI',
            line=dict(color='#9f7aea', width=3)
        ), row=3, col=1)
        
        # RSI levels
        fig.add_hline(y=70, line_dash="dash", line_color="rgba(255,68,68,0.8)", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="rgba(0,255,136,0.8)", row=3, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="rgba(255,255,255,0.5)", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            height=600,
            template="plotly_dark",
            showlegend=True,
            title_text=f"{data['symbol']} - Advanced Technical Analysis",
            title_font=dict(size=24, color='white'),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white', family='Inter'),
            legend=dict(
                bgcolor='rgba(255,255,255,0.1)',
                bordercolor='rgba(255,255,255,0.2)',
                borderwidth=1
            )
        )
        
        fig.update_xaxes(
            rangeslider_visible=False,
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        )
        fig.update_yaxes(
            gridcolor='rgba(255,255,255,0.1)',
            showgrid=True
        )
        
        st.plotly_chart(fig, use_container_width=True, key=get_unique_key())
    
    @staticmethod
    def render_investment_dashboard(data: Dict[str, Any], score_data: Dict[str, Any]):
        """Render premium investment dashboard"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        # Current Metrics
        with col1:
            st.markdown("### 📊 Current Metrics")
            
            # Price card
            price_change_color = "#00ff88" if data['price_change'] >= 0 else "#ff4444"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">${data['current_price']:.2f}</div>
                <div class="metric-label">Current Price</div>
                <div class="metric-change" style="background: {price_change_color};">
                    {data['price_change']:+.2f}%
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Volume
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{data['volume']:,}</div>
                <div class="metric-label">Volume</div>
                <div class="metric-change">
                    {data['volume_ratio']:.1f}x Avg
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Market Cap
            market_cap_display = f"${data['market_cap']/1e9:.1f}B" if data['market_cap'] > 1e9 else f"${data['market_cap']/1e6:.1f}M"
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{market_cap_display}</div>
                <div class="metric-label">Market Cap</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Technical Indicators
        with col2:
            st.markdown("### 🎯 Technical Indicators")
            
            # RSI Gauge
            rsi_color = "#ff4444" if data['rsi'] > 70 or data['rsi'] < 30 else "#00ff88"
            rsi_status = "Overbought" if data['rsi'] > 70 else "Oversold" if data['rsi'] < 30 else "Neutral"
            
            rsi_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=data['rsi'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "RSI", 'font': {'color': 'white', 'size': 20}},
                number={'font': {'color': 'white', 'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': rsi_color, 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.1)',
                    'borderwidth': 2,
                    'bordercolor': 'rgba(255,255,255,0.3)',
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(0,255,136,0.3)'},
                        {'range': [70, 100], 'color': 'rgba(255,68,68,0.3)'}
                    ],
                    'threshold': {
                        'line': {'color': "white", 'width': 4},
                        'thickness': 0.8,
                        'value': 50
                    }
                }
            ))
            
            rsi_fig.update_layout(
                height=250,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Inter'}
            )
            
            st.plotly_chart(rsi_fig, use_container_width=True, key=get_unique_key())
            
            st.markdown(f"""
            <div style="text-align: center; color: {rsi_color}; font-weight: 600; margin-top: -20px;">
                {rsi_status} ({data['rsi']:.1f})
            </div>
            """, unsafe_allow_html=True)
            
            # Additional indicators
            ma_trend = "🟢 Bullish" if data['current_price'] > data['ma_20'] else "🔴 Bearish"
            vol_status = "🟡 High" if data['volatility'] > 0.5 else "🟢 Normal"
            
            st.markdown(f"""
            <div style="margin-top: 1rem;">
                <div style="margin: 0.5rem 0;"><strong>MA Trend:</strong> {ma_trend}</div>
                <div style="margin: 0.5rem 0;"><strong>Volatility:</strong> {vol_status} ({data['volatility']:.1%})</div>
                <div style="margin: 0.5rem 0;"><strong>P/E Ratio:</strong> {data['pe_ratio']:.1f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        # Investment Score
        with col3:
            st.markdown("### 💡 AI Investment Score")
            
            # Score gauge
            score_fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=score_data['score'],
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Investment Score", 'font': {'color': 'white', 'size': 20}},
                number={'font': {'color': 'white', 'size': 24}},
                gauge={
                    'axis': {'range': [None, 100], 'tickcolor': 'white'},
                    'bar': {'color': '#667eea', 'thickness': 0.3},
                    'bgcolor': 'rgba(255,255,255,0.1)',
                    'borderwidth': 2,
                    'bordercolor': 'rgba(255,255,255,0.3)',
                    'steps': [
                        {'range': [0, 40], 'color': 'rgba(245,101,101,0.3)'},
                        {'range': [40, 70], 'color': 'rgba(237,137,54,0.3)'},
                        {'range': [70, 100], 'color': 'rgba(72,187,120,0.3)'}
                    ]
                }
            ))
            
            score_fig.update_layout(
                height=250,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font={'color': 'white', 'family': 'Inter'}
            )
            
            st.plotly_chart(score_fig, use_container_width=True, key=get_unique_key())
            
            # Recommendation
            rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
            st.markdown(f"""
            <div class="{rec_class}" style="margin-top: -20px;">
                {score_data['recommendation']}
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
                <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
                <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

def main():
    init_session_state()
    
    # Premium Header
    st.markdown("""
    <div class="premium-header">
        <h1>🤖 FinDocGPT</h1>
        <h3>AI-Powered Financial Intelligence</h3>
        <p>Advanced financial analysis, forecasting, and investment strategy using FinanceBench dataset and Yahoo Finance API</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Status Bar with Sidebar Control
    st.markdown('<div class="status-bar">', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"🕐 **Live Time:** {datetime.now().strftime('%H:%M:%S')}")
    with col2:
        openai_status = "🟢 Connected" if client else "🔴 Limited"
        st.markdown(f"🤖 **OpenAI:** {openai_status}")
    with col3:
        if PDF_AVAILABLE:
            st.markdown("📄 **Documents:** TXT, PDF, CSV")
        else:
            st.markdown("📄 **Documents:** TXT, CSV Only")
    with col4:
        if st.button("🔄 Refresh All", key="refresh_main"):
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Premium Sidebar - Force visibility with toggle arrows
    with st.sidebar:
        # Add prominent expand/collapse arrows at the top
        st.markdown("""
        <div style="text-align: center; padding: 1rem; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;">
            <div style="font-size: 1.5rem; color: white;">⏪ FinDocGPT Sidebar ⏩</div>
            <div style="font-size: 0.8rem; color: rgba(255,255,255,0.8);">Click « or » arrows to collapse/expand</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-content">', unsafe_allow_html=True)
        
        # Add sidebar toggle button at the top
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            if st.button("«", key="collapse_sidebar", help="Collapse sidebar"):
                st.info("Click the » arrow to expand sidebar again")
        with col2:
            st.markdown("**Control Panel**")
        with col3:
            if st.button("»", key="expand_sidebar", help="Expand sidebar"):
                st.success("Sidebar expanded!")
        
        # Add a visible indicator that sidebar is working
        st.success("✅ Sidebar is active!")
        
        st.header("🎯 FinDocGPT Control Panel")
        
        # Stock Symbol Selection
        symbol_input = st.text_input(
            "📊 Stock Symbol", 
            value=st.session_state.selected_symbol,
            help="Enter stock symbol (e.g., AAPL, MSFT, GOOGL)"
        )
        
        if symbol_input != st.session_state.selected_symbol:
            st.session_state.selected_symbol = symbol_input.upper()
        
        # Analysis Period
        data_period = st.selectbox(
            "📅 Analysis Period",
            ["1mo", "3mo", "6mo", "1y", "2y"],
            index=3
        )
        
        # Real-time Data Display
        if st.session_state.selected_symbol:
            with st.spinner("📡 Loading market data..."):
                current_data = FinancialDataHandler.get_real_time_data(
                    st.session_state.selected_symbol, data_period
                )
            
            if "error" not in current_data:
                st.success(f"✅ {st.session_state.selected_symbol} Data Loaded")
                
                # Investment Score
                score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
                # Score display
                st.markdown(f"""
                <div class="metric-card {score_data['color_class']}">
                    <div class="metric-value">{score_data['recommendation']}</div>
                    <div class="metric-label">AI Investment Score</div>
                    <div class="metric-change">
                        Score: {score_data['score']:.1f}/100 ({score_data['confidence']:.1f}% confidence)
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Key Metrics
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
                    st.metric("RSI", f"{current_data['rsi']:.1f}")
            else:
                st.error(f"❌ Error: {current_data['error']}")
        
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
            st.success("History cleared!")
        
        # Enhanced sidebar visibility help with arrows
        st.markdown("---")
        st.markdown("#### 🔧 Sidebar Navigation")
        st.markdown("""
        **Double Arrow Controls:**
        - **« (Left Arrow)**: Collapse sidebar  
        - **» (Right Arrow)**: Expand sidebar
        - **⏪ ⏩**: Visual indicators in header
        """)
        st.caption("Look for « » arrows at the very top-left of the page")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Main Navigation Tabs
    tab1, tab2, tab3 = st.tabs([
        "🔍 Stage 1: Insights & Analysis", 
        "📈 Stage 2: Financial Forecasting", 
        "💼 Stage 3: Investment Strategy"
    ])
    
    # Stage 1: Document Q&A
    with tab1:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.header("🔍 Stage 1: Document Insights & Analysis")
        st.markdown("*Document Q&A using FinanceBench • Market Sentiment Analysis • Anomaly Detection*")
        
        # Sub-tabs for Stage 1
        sub_tab1, sub_tab2, sub_tab3 = st.tabs([
            "📄 Document Q&A (FinanceBench)", 
            "😊 Sentiment Analysis", 
            "⚠️ Anomaly Detection"
        ])
        
        # Document Q&A Sub-tab
        with sub_tab1:
            st.subheader("📄 Financial Document Q&A - FinanceBench Dataset")
            st.markdown("""
            Upload financial documents or use FinanceBench dataset examples for AI-powered analysis.
            Ask questions about earnings reports, SEC filings, and financial statements.
            """)
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Document Upload
                uploaded_file = st.file_uploader(
                    "📁 Upload Financial Document", 
                    type=['txt', 'pdf', 'csv'],
                    help="Upload earnings reports, SEC filings, or financial datasets"
                )
                
                # Text input area
                document_text = st.text_area(
                    "📝 Or paste document content:",
                    height=200,
                    placeholder="""Example FinanceBench content:

APPLE INC. Q4 2023 EARNINGS REPORT
Revenue: $89.5 billion (+2.8% YoY)
Net Income: $22.9 billion 
Gross Margin: 45.2%
iPhone Revenue: $43.8 billion
Services Revenue: $22.3 billion

Key Highlights:
- Record Services revenue driven by App Store growth
- iPhone 15 launch exceeded expectations  
- Supply chain constraints improved significantly

Risk Factors:
- China market regulatory challenges
- Component cost inflation"""
                )
                
                # Query Input
                query = st.text_input(
                    "❓ Ask a question about the financial document:",
                    placeholder="e.g., What was the revenue growth? What are the main risk factors?",
                    help="Ask specific questions about financial metrics, performance, risks, or business insights"
                )
                
                # Analysis Button
                if st.button("🔍 Analyze Document with AI", key="analyze_doc", use_container_width=True):
                    if (document_text.strip() or uploaded_file) and query.strip():
                        with st.spinner("🤖 FinDocGPT is analyzing the document..."):
                            # Process uploaded file if exists
                            if uploaded_file:
                                doc_content = DocumentProcessor.extract_document_text(uploaded_file)
                                if "❌" in doc_content:
                                    st.error(doc_content)
                                    doc_content = document_text  # Fallback to text area
                            else:
                                doc_content = document_text
                            
                            # Generate AI analysis
                            response = DocumentProcessor.process_financial_document(doc_content, query)
                            
                            # Display results
                            st.markdown("### 📊 FinDocGPT Analysis Results")
                            
                            with st.container():
                                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                st.markdown(response)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Save to history
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
                st.subheader("📋 Instructions")
                st.markdown("""
                **How to use Document Q&A:**
                
                1. **Upload** a financial document or **paste** content
                2. **Ask** a specific question about the document
                3. **Click** "Analyze Document" to get AI insights
                
                **Supported File Types:**
                - ✅ **TXT**: Financial reports, transcripts
                - ✅ **PDF**: 10-K filings, annual reports  
                - ✅ **CSV**: Financial datasets, metrics
                
                **Example Questions:**
                - What was the quarterly revenue?
                - What are the main risk factors?
                - How did expenses change YoY?
                - What's the cash flow situation?
                """)
                
                # Show recent analysis
                if st.session_state.analysis_history:
                    st.subheader("📚 Recent Analyses")
                    recent = [a for a in st.session_state.analysis_history if a['type'] == 'Document Q&A'][-2:]
                    
                    for analysis in reversed(recent):
                        with st.expander(f"Q: {analysis['query'][:30]}..."):
                            st.caption(f"Date: {analysis['timestamp'].strftime('%Y-%m-%d %H:%M')}")
                            st.text_area("Response:", analysis['response'][:150] + "...", height=80, disabled=True)
        
        # Sentiment Analysis Sub-tab
        with sub_tab2:
            st.subheader("😊 Market Sentiment Analysis")
            st.markdown("Analyze sentiment from earnings calls, press releases, and financial news using advanced AI.")
            
            if st.session_state.selected_symbol:
                current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
                if "error" not in current_data:
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
                else:
                    st.error(f"Cannot load data for {st.session_state.selected_symbol}")
            else:
                st.info("ℹ️ Please select a stock symbol in the sidebar to begin sentiment analysis.")
        
        # Anomaly Detection Sub-tab
        with sub_tab3:
            st.subheader("⚠️ Real-Time Anomaly Detection")
            st.markdown("Identify unusual patterns and potential risks in financial metrics and market behavior.")
            
            if st.session_state.selected_symbol:
                current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
                
                if "error" not in current_data:
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
                        
                        col_a, col_b, col_c, col_d = st.columns(4)
                        
                        with col_a:
                            price_color = "normal" if current_data['price_change'] >= 0 else "inverse"
                            st.metric("Price Change", f"{current_data['price_change']:+.2f}%", delta_color=price_color)
                        
                        with col_b:
                            rsi_status = "Overbought" if current_data['rsi'] > 70 else "Oversold" if current_data['rsi'] < 30 else "Neutral"
                            st.metric("RSI", f"{current_data['rsi']:.1f}", rsi_status)
                        
                        with col_c:
                            vol_status = "High" if current_data.get('volume_ratio', 1) > 2 else "Normal"
                            st.metric("Volume", f"{current_data['volume']:,}", f"{volume_ratio:.1f}x ({vol_status})")
                        
                        with col_d:
                            vol_status = "High" if current_data['volatility'] > 0.5 else "Normal"
                            st.metric("Volatility", f"{current_data['volatility']:.2%}", vol_status)
                    
                    with col2:
                        st.markdown("#### 🎯 Anomaly Monitoring")
                        
                        # Monitoring thresholds
                        st.markdown("**Current Thresholds:**")
                        thresholds = {
                            "Price Movement": "> ±8% (Warning), > ±15% (Critical)",
                            "RSI": "< 25 or > 75 (Warning), < 15 or > 85 (Critical)",
                            "Volume": "> 2x avg (Warning), > 5x avg (Critical)", 
                            "Volatility": "> 80% (Warning), > 150% (Critical)"
                        }
                        
                        for metric, threshold in thresholds.items():
                            st.caption(f"**{metric}**: {threshold}")
                        
                        st.markdown("---")
                        st.markdown("#### 📈 Real-time Status")
                        
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
                else:
                    st.error(f"Cannot load data for {st.session_state.selected_symbol}")
            else:
                st.info("ℹ️ Please select a stock symbol in the sidebar to begin anomaly detection.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Stage 2: Financial Forecasting
    with tab2:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.header("📈 Stage 2: Financial Forecasting")
        st.markdown("*AI-powered predictions • Advanced technical analysis • Market data integration*")
        
        if st.session_state.selected_symbol:
            current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
            if "error" not in current_data:
                # Forecasting parameters - DEFINE THESE FIRST
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎯 AI-Powered Financial Forecasting")
                    
                    # Forecasting parameters
                    col_a, col_b, col_c = st.columns(3)
                    
                    with col_a:
                        forecast_period = st.selectbox(
                            "📅 Forecast Period:",
                            ["1 Week", "1 Month", "3 Months", "6 Months"],
                            index=1,
                            key="forecast_period_select"
                        )
                    
                    with col_b:
                        forecast_type = st.selectbox(
                            "📊 Analysis Type:",
                            ["Technical Analysis", "Combined Analysis"],
                            index=1
                        )
                    
                    with col_c:
                        confidence_level = st.selectbox(
                            "⚖️ Confidence Level:",
                            ["Conservative", "Moderate", "Aggressive"],
                            index=1,
                            help="Conservative: Lower risk predictions, Moderate: Balanced approach, Aggressive: Higher risk/reward scenarios"
                        )
                
                # Enhanced Chart with Integrated Forecast - NOW forecast_period is defined
                st.subheader("📊 Advanced Technical Analysis with Forecast")
                with st.container():
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    PremiumChartRenderer.render_premium_price_chart(current_data, forecast_period)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.subheader("🎯 Forecasting Guide")
                    
                    st.markdown("""
                    **Forecast Types:**
                    - **Technical**: Price patterns, indicators
                    - **Combined**: Technical + Fundamental
                    
                    **Confidence Levels:**
                    - **Conservative**: Lower risk, modest returns, higher probability scenarios (70%+)
                    - **Moderate**: Balanced risk/reward, moderate probability scenarios (50-70%)
                    - **Aggressive**: Higher risk, higher potential, includes lower probability high-reward scenarios (30-50%)
                    """)
                    
                    # Current indicators
                    st.subheader("📊 Current Indicators")
                    
                    indicators = [
                        ("Price Trend", "🟢 Up" if current_data['price_change'] > 0 else "🔴 Down"),
                        ("RSI Status", "🟢 Neutral" if 30 < current_data['rsi'] < 70 else "🟡 Extreme"),
                        ("Volume", "🟢 Normal" if current_data.get('volume_ratio', 1) < 2 else "🟡 High"),
                        ("Volatility", "🟢 Low" if current_data['volatility'] < 0.5 else "🟡 High")
                    ]
                    
                    for indicator, status in indicators:
                        st.markdown(f"**{indicator}**: {status}")
                
                # Generate forecast button
                if st.button("🔮 Generate AI Forecast", key="generate_forecast", use_container_width=True):
                    with st.spinner("🤖 Generating advanced forecast..."):
                        
                        context = f"""
                        FORECASTING CONTEXT for {st.session_state.selected_symbol}
                        
                        Current Data:
                        - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
                        - Volume: {current_data['volume']:,}
                        - RSI: {current_data['rsi']:.1f}
                        - MA 20: ${current_data['ma_20']:.2f}
                        - MA 50: ${current_data['ma_50']:.2f}
                        - Volatility: {current_data['volatility']:.2%}
                        - Sector: {current_data['company_info']['sector']}
                        """
                        
                        prompt = f"""
                        Generate a comprehensive {forecast_period} forecast for {st.session_state.selected_symbol} using {forecast_type.lower()} with {confidence_level.lower()} confidence level.
                        
                        Provide detailed analysis including:
                        1. **Price Targets**: Bull case, base case, and bear case scenarios with specific prices and probabilities (adjust probability ranges based on {confidence_level.lower()} approach)
                        2. **Key Catalysts**: List 3-5 positive drivers and 3-5 potential risks
                        3. **Technical Analysis**: Support/resistance levels, momentum indicators, and chart patterns
                        4. **Timeline**: Expected price movement schedule with key milestones
                        5. **Confidence Assessment**: Overall confidence level (0-100%) and reliability factors (consider {confidence_level.lower()} risk tolerance)
                        6. **Risk Management**: Suggested stop-loss levels and position sizing recommendations tailored to {confidence_level.lower()} investors
                        
                        **{confidence_level} Approach Guidelines:**
                        - Conservative: Focus on higher probability scenarios (70%+), smaller price targets, emphasis on capital preservation
                        - Moderate: Balanced probability scenarios (50-70%), moderate price targets, balanced risk/reward
                        - Aggressive: Include lower probability high-reward scenarios (30-50%), larger price targets, growth-focused
                        """
                        
                        forecast_response = OpenAIHandler.generate_response(prompt, context, "forecasting")
                        
                        st.markdown("### 🔮 AI Forecasting Results")
                        with st.container():
                            st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                            st.markdown(forecast_response)
                            st.markdown('</div>', unsafe_allow_html=True)
                        
                        st.success("✅ Advanced forecast generated!")
            else:
                st.error(f"Cannot load data for {st.session_state.selected_symbol}")
        else:
            st.info("Please select a stock symbol in the sidebar to begin forecasting.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Stage 3: Investment Strategy
    with tab3:
        st.markdown('<div class="premium-card">', unsafe_allow_html=True)
        st.header("💼 Stage 3: Investment Strategy & Decision-Making")
        st.markdown("*Professional portfolio management • Advanced risk assessment • AI-powered trading recommendations*")
        
        if st.session_state.selected_symbol:
            current_data = FinancialDataHandler.get_real_time_data(st.session_state.selected_symbol)
            
            if "error" not in current_data:
                score_data = FinancialDataHandler.calculate_investment_score(current_data)
                
                # Premium Investment Dashboard
                st.subheader("📊 Professional Investment Dashboard")
                PremiumChartRenderer.render_investment_dashboard(current_data, score_data)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.subheader("🎯 AI Investment Strategy Generator")
                    
                    # Strategy parameters
                    col_a, col_b = st.columns(2)
                    
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
                    
                    # Generate strategy button
                    if st.button("💡 Generate Professional Investment Strategy", key="generate_strategy", use_container_width=True):
                        with st.spinner("🤖 Generating comprehensive investment strategy..."):
                            
                            context = f"""
                            INVESTMENT STRATEGY CONTEXT for {st.session_state.selected_symbol}
                            
                            Market Data:
                            - Price: ${current_data['current_price']:.2f} ({current_data['price_change']:+.2f}%)
                            - P/E Ratio: {current_data['pe_ratio']:.1f}
                            - Beta: {current_data.get('beta', 0):.2f}
                            - Market Cap: ${current_data['market_cap']/1e9:.1f}B
                            - RSI: {current_data['rsi']:.1f}
                            - Volatility: {current_data['volatility']:.2%}
                            - Investment Score: {score_data['score']:.1f}/100
                            - AI Recommendation: {score_data['recommendation']}
                            
                            Parameters:
                            - Investment Horizon: {investment_horizon}
                            - Risk Tolerance: {risk_tolerance}
                            - Portfolio Value: ${st.session_state.portfolio_value:,.0f}
                            """
                            
                            prompt = f"""
                            Create a comprehensive, professional-grade investment strategy for {st.session_state.selected_symbol}.
                            
                            Provide detailed analysis including:
                            
                            1. **Executive Summary & Investment Decision**:
                               - Clear BUY/SELL/HOLD recommendation with detailed rationale
                               - Investment thesis in 2-3 sentences
                               - Expected return potential and timeframe
                            
                            2. **Entry Strategy & Timing**:
                               - Optimal entry price points and conditions
                               - Dollar-cost averaging vs. lump sum recommendations
                               - Market timing considerations and triggers
                            
                            3. **Risk Management Framework**:
                               - Stop-loss levels with specific prices
                               - Position sizing as % of portfolio (given ${st.session_state.portfolio_value:,.0f} portfolio)
                               - Risk/reward ratio analysis
                               - Hedging strategies if applicable
                            
                            4. **Exit Strategy & Profit Taking**:
                               - Primary profit target prices
                               - Partial profit-taking levels
                               - Review and rebalancing triggers
                            
                            5. **Portfolio Integration**:
                               - How this position fits into overall allocation
                               - Diversification impact and sector exposure
                               - Correlation with existing holdings
                            
                            6. **Monitoring & Review Framework**:
                               - Key metrics to track weekly/monthly
                               - Fundamental and technical review schedule
                               - Scenario planning (bull/bear case actions)
                            
                            Consider the {risk_tolerance.lower()} risk profile and {investment_horizon} timeframe in all recommendations.
                            Provide specific, actionable advice that a professional investment advisor would give.
                            """
                            
                            investment_strategy = OpenAIHandler.generate_response(prompt, context, "investment")
                            
                            st.markdown("### 💼 Professional Investment Strategy")
                            with st.container():
                                st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
                                st.markdown(investment_strategy)
                                st.markdown('</div>', unsafe_allow_html=True)
                            
                            # Add to portfolio option
                            if "BUY" in investment_strategy.upper():
                                if st.button("➕ Add to Portfolio", key="add_to_portfolio"):
                                    portfolio_item = {
                                        "symbol": st.session_state.selected_symbol,
                                        "action": score_data['recommendation'],
                                        "price": current_data['current_price'],
                                        "timestamp": datetime.now(),
                                        "strategy": investment_strategy[:200] + "...",
                                        "score": score_data['score']
                                    }
                                    st.session_state.investment_portfolio.append(portfolio_item)
                                    st.success(f"✅ {st.session_state.selected_symbol} added to portfolio!")
                            
                            st.success("✅ Professional investment strategy generated!")
                
                with col2:
                    st.subheader("💡 Strategy Guide")
                    
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
                    
                    # Quick recommendation
                    st.subheader("🎯 Quick AI Recommendation")
                    
                    rec_class = f"recommendation-{score_data['recommendation'].lower().replace(' ', '-')}"
                    st.markdown(f"""
                    <div class="{rec_class}">
                        {score_data['recommendation']}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"""
                    <div style="text-align: center; margin-top: 1rem; color: rgba(255,255,255,0.8);">
                        <div><strong>Score:</strong> {score_data['score']:.1f}/100</div>
                        <div><strong>Confidence:</strong> {score_data['confidence']:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Portfolio summary - Only show if there are actual positions
                    if st.session_state.investment_portfolio:
                        st.subheader("💼 Portfolio Summary")
                        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.0f}")
                        st.metric("Total Positions", len(st.session_state.investment_portfolio))
                        
                        st.markdown("**Recent Positions:**")
                        recent = st.session_state.investment_portfolio[-3:]
                        for item in reversed(recent):
                            st.markdown(f"• {item['symbol']} - {item['action']} (Score: {item['score']:.1f})")
                    else:
                        st.subheader("💼 Portfolio")
                        st.info("No positions yet. Generate investment strategies to build your portfolio!")
            else:
                st.error(f"Cannot load data for {st.session_state.selected_symbol}")
        else:
            st.info("Please select a stock symbol in the sidebar to begin investment analysis.")
        
        # Portfolio Management Section
        if st.session_state.investment_portfolio:
            st.markdown("---")
            st.subheader("📊 Portfolio Management")
            
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
                if st.button("📈 Portfolio Analytics", use_container_width=True):
                    st.info("Advanced portfolio analytics - Coming soon!")
            
            with col2:
                if st.button("📊 Performance Report", use_container_width=True):
                    st.info("Performance reporting - Coming soon!")
            
            with col3:
                if st.button("🗑️ Clear Portfolio", use_container_width=True):
                    st.session_state.investment_portfolio.clear()
                    st.success("Portfolio cleared!")
                    st.rerun()
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer Status
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        if not PDF_AVAILABLE:
            st.error("""
            🚨 **PDF Processing Not Available**
            Install: `pip install PyPDF2 pdfplumber`
            """)
        else:
            st.success("✅ All document formats supported (TXT, PDF, CSV)")
    
    with col2:
        if not client:
            st.warning("⚠️ OpenAI API not configured. Set OPENAI_API_KEY environment variable for full AI capabilities.")
        else:
            st.success("✅ AI analysis fully enabled")

# Run the application
if __name__ == "__main__":
    main()