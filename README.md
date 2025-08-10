# README.md
"""
# FinDocGPT 🚀

AI-powered financial document analysis & investment strategy platform built for hackathons and rapid prototyping.

![FinDocGPT](https://img.shields.io/badge/FinDocGPT-AI%20Finance-blue)
![Python](https://img.shields.io/badge/Python-3.8+-green)
![React](https://img.shields.io/badge/React-18+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

## 🎯 Overview

FinDocGPT combines three powerful AI-driven stages to transform financial analysis:

1. **📄 Document Intelligence**: Upload and analyze financial documents with Q&A, sentiment analysis, and anomaly detection
2. **📈 Price Forecasting**: LSTM models predict stock prices with technical indicators and confidence intervals  
3. **💼 Investment Strategy**: Get buy/sell recommendations, portfolio optimization, and comprehensive backtesting

## ⚡ Quick Start

```bash
# 1. Clone and setup
git clone <your-repo>
cd findocgpt

# 2. One-command setup
make dev-setup

# 3. Start development
make start-dev
```

Your servers will be running at:
- **Backend API**: http://localhost:8000
- **Frontend**: http://localhost:3000
- **API Docs**: http://localhost:8000/docs

## 🏗️ Architecture

### Stage 1: Document Intelligence
- **FinanceBench Integration**: 10,231 financial Q&A pairs
- **Smart Q&A Engine**: RAG-powered document questioning
- **Sentiment Analysis**: FinBERT-based sentiment scoring
- **Anomaly Detection**: Statistical outlier identification

### Stage 2: Forecasting Engine  
- **LSTM Models**: Deep learning price prediction
- **Technical Analysis**: 15+ indicators (RSI, MACD, Bollinger Bands)
- **Feature Engineering**: Advanced time series features
- **Confidence Intervals**: Uncertainty quantification

### Stage 3: Strategy Engine
- **Decision Engine**: Multi-factor investment recommendations
- **Portfolio Optimization**: Modern Portfolio Theory implementation
- **Risk Management**: VaR, drawdown, and volatility analysis
- **Backtesting**: Historical strategy validation

## 🛠️ Tech Stack

**Backend:**
- FastAPI + SQLAlchemy
- PyTorch + scikit-learn
- LangChain + Sentence Transformers
- yfinance + pandas

**Frontend:**
- React 18 + TypeScript
- TailwindCSS + Headless UI
- Recharts + React Query
- Vite build system

**Data:**
- FinanceBench dataset (Hugging Face)
- Yahoo Finance API
- SQLite/PostgreSQL
- Redis caching

## 📊 Dataset

- **FinanceBench**: 10,231 financial Q&A pairs with source documents
- **Market Data**: Real-time prices for 25+ major stocks  
- **Economic Indicators**: S&P 500, VIX, Treasury rates
- **Company Fundamentals**: P/E ratios, revenue growth, debt metrics

## 🚀 Development

### Available Commands

```bash
# Setup and installation
make setup              # Complete setup
make install-deps       # Install dependencies only
make download-data      # Download datasets only

# Development servers
make start-dev          # Start both backend + frontend
make start-backend      # Backend only
make start-frontend     # Frontend only

# Testing and validation
make test              # Run all tests
make test-backend      # Backend tests only
make lint              # Code linting
make format            # Code formatting

# Database operations
make db-init           # Initialize database
make db-reset          # Reset database

# Deployment
make build             # Production build
make deploy            # Docker deployment
```

### Project Structure

```
findocgpt/
├── backend/           # FastAPI application
│   ├── app/
│   │   ├── api/v1/    # API endpoints
│   │   ├── core/      # 3-stage implementations
│   │   ├── models/    # Database models
│   │   └── utils/     # Utilities
├── frontend/          # React application
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   └── services/
├── data/             # Datasets and models
├── notebooks/        # Jupyter analysis
├── scripts/          # Utility scripts
└── docs/            # Documentation
```

## 🔑 Configuration

### Required API Keys

1. **OpenAI** (Optional): For advanced document Q&A
   - Get key: https://platform.openai.com/api-keys
   
2. **Alpha Vantage** (Optional): For extended market data  
   - Get key: https://www.alphavantage.co/support/#api-key
   
3. **Yahoo Finance**: Free, no key required ✅

### Environment Setup

```bash
# Copy and edit environment file
cp .env.example .env

# Edit with your API keys
OPENAI_API_KEY=your_key_here
ALPHA_VANTAGE_API_KEY=your_key_here
```

## 📈 Features

### ✅ Document Analysis
- Upload financial PDFs, 10-Ks, earnings reports
- AI-powered Q&A with source citations
- Sentiment analysis with confidence scores
- Financial anomaly detection

### ✅ Price Forecasting  
- LSTM neural networks for price prediction
- Technical indicator calculation (RSI, MACD, etc.)
- Confidence intervals and uncertainty quantification
- Real-time data integration

### ✅ Investment Strategy
- Multi-factor buy/sell/hold recommendations
- Portfolio optimization with risk constraints
- Comprehensive backtesting framework
- Risk metrics (Sharpe ratio, VaR, drawdown)

### ✅ Interactive Dashboard
- Real-time charts and visualizations
- Responsive design with dark/light modes
- Export capabilities for reports
- Mobile-friendly interface

## 🧪 Testing

```bash
# Quick validation
make test

# Individual test suites
python scripts/quick_test.py    # Setup validation
python scripts/data_validator.py # Data integrity
pytest backend/tests/           # Unit tests
```

## 🚢 Deployment

### Docker Deployment
```bash
# Build and deploy
make deploy

# Or manually
docker-compose -f docker/docker-compose.yml up -d
```

### Manual Deployment
```bash
# Production build
make build

# Start production servers
gunicorn app.main:app --host 0.0.0.0 --port 8000
serve -s frontend/build -l 3000
```

## 📝 API Documentation

Once running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Key Endpoints

```bash
# Document Analysis
POST /api/v1/documents/upload
POST /api/v1/documents/question
GET  /api/v1/documents/sentiment/{symbol}

# Forecasting  
GET  /api/v1/forecasting/predict/{symbol}
GET  /api/v1/forecasting/technical-indicators/{symbol}

# Strategy
GET  /api/v1/strategy/recommendation/{symbol}
POST /api/v1/strategy/portfolio/optimize
GET  /api/v1/strategy/backtest/{symbol}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **FinanceBench**: PatronusAI for the financial Q&A dataset
- **Yahoo Finance**: Free financial data API
- **Hugging Face**: Model hosting and datasets platform
- **FastAPI & React**: Modern web development frameworks

---

**Built for hackathons, ready for production** 🚀

For questions and support, please open an issue or contact the development team.
"""
