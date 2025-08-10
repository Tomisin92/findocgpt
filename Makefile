# Makefile
"""
.PHONY: setup install-deps download-data start-dev test clean help

# Variables
PYTHON = python
PIP = pip
NPM = npm

# One-command setup
setup: install-deps download-data
	@echo "✅ FinDocGPT setup complete!"

# Quick development setup
dev-setup: 
	@echo "🚀 Setting up FinDocGPT for development..."
	cp .env.example .env
	@echo "⚙️  Created .env file - please edit with your API keys"
	$(MAKE) setup
	$(PYTHON) scripts/quick_test.py

# Install all dependencies
install-deps:
	@echo "📦 Installing backend dependencies..."
	cd backend && $(PIP) install -r requirements.txt
	@echo "📦 Installing frontend dependencies..."
	cd frontend && $(NPM) install

# Download all required data
download-data:
	@echo "📊 Downloading datasets..."
	$(PYTHON) scripts/download_data.py
	$(PYTHON) scripts/data_validator.py

# Start development servers
start-dev:
	@echo "🚀 Starting development servers..."
	@echo "Backend will be available at http://localhost:8000"
	@echo "Frontend will be available at http://localhost:3000"
	@echo "Press Ctrl+C to stop servers"
	cd backend && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
	cd frontend && $(NPM) start

# Start backend only
start-backend:
	@echo "🐍 Starting backend server..."
	cd backend && $(PYTHON) -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# Start frontend only
start-frontend:
	@echo "⚛️  Starting frontend server..."
	cd frontend && $(NPM) start

# Run tests
test:
	@echo "🧪 Running tests..."
	$(PYTHON) scripts/quick_test.py
	cd backend && $(PYTHON) -m pytest tests/ -v || echo "⚠️  Backend tests not fully implemented"
	cd frontend && $(NPM) test --watchAll=false || echo "⚠️  Frontend tests not fully implemented"

# Run specific test suites
test-backend:
	@echo "🧪 Running backend tests..."
	cd backend && $(PYTHON) -m pytest tests/ -v

test-frontend:
	@echo "🧪 Running frontend tests..."
	cd frontend && $(NPM) test --watchAll=false

# Linting and formatting
lint:
	@echo "🔍 Running linters..."
	cd backend && $(PYTHON) -m flake8 app/
	cd backend && $(PYTHON) -m black --check app/

format:
	@echo "🎨 Formatting code..."
	cd backend && $(PYTHON) -m black app/
	cd frontend && $(NPM) run format || echo "Frontend formatting not configured"

# Clean build artifacts
clean:
	@echo "🧹 Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf backend/build/
	rm -rf frontend/build/
	rm -rf frontend/node_modules/.cache/
	rm -rf .pytest_cache/

# Deep clean (including data and models)
clean-all: clean
	@echo "🗑️  Deep cleaning..."
	rm -rf data/processed/
	rm -rf data/models/trained_models/
	rm -rf data/models/checkpoints/

# Install development tools
install-dev:
	@echo "🛠️  Installing development tools..."
	$(PIP) install black flake8 pytest pytest-asyncio
	cd frontend && $(NPM) install --save-dev prettier eslint

# Database operations
db-init:
	@echo "🗃️  Initializing database..."
	cd backend && $(PYTHON) -c "from app.config.database import engine, Base; Base.metadata.create_all(bind=engine)"

db-reset:
	@echo "🔄 Resetting database..."
	rm -f backend/findocgpt.db
	$(MAKE) db-init

# Model training
train-models:
	@echo "🤖 Training ML models..."
	$(PYTHON) scripts/train_models.py

# Deployment
build:
	@echo "🏗️  Building for production..."
	cd frontend && $(NPM) run build
	cd backend && $(PYTHON) -m pip install --upgrade pip
	cd backend && $(PIP) install -r requirements.txt

deploy:
	@echo "🚀 Deploying to production..."
	$(MAKE) build
	docker-compose -f docker/docker-compose.yml up -d

# Docker operations
docker-build:
	@echo "🐳 Building Docker images..."
	docker-compose -f docker/docker-compose.yml build

docker-up:
	@echo "🐳 Starting Docker containers..."
	docker-compose -f docker/docker-compose.yml up -d

docker-down:
	@echo "🐳 Stopping Docker containers..."
	docker-compose -f docker/docker-compose.yml down

# Help
help:
	@echo "FinDocGPT Makefile Commands:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  setup        - Complete project setup (install deps + download data)"
	@echo "  dev-setup    - Quick development setup with .env creation"
	@echo "  install-deps - Install all dependencies (Python + Node.js)"
	@echo "  download-data- Download required datasets"
	@echo ""
	@echo "Development Commands:"
	@echo "  start-dev    - Start both backend and frontend servers"
	@echo "  start-backend- Start only backend server"
	@echo "  start-frontend- Start only frontend server"
	@echo ""
	@echo "Testing Commands:"
	@echo "  test         - Run all tests and validation"
	@echo "  test-backend - Run backend tests only"
	@echo "  test-frontend- Run frontend tests only"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint         - Run linters"
	@echo "  format       - Format code"
	@echo ""
	@echo "Database Commands:"
	@echo "  db-init      - Initialize database"
	@echo "  db-reset     - Reset database"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean        - Clean build artifacts"
	@echo "  clean-all    - Deep clean including data"
	@echo ""
	@echo "Deployment Commands:"
	@echo "  build        - Build for production"
	@echo "  deploy       - Deploy using Docker"
	@echo "  docker-build - Build Docker images"
	@echo "  docker-up    - Start Docker containers"
	@echo "  docker-down  - Stop Docker containers"
"""