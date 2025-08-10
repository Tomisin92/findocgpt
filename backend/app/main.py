# backend/app/main.py
"""
FinDocGPT FastAPI Application
Main entry point for the backend API
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn

from app.api.v1 import documents, forecasting, strategy, chat
from app.config.settings import get_settings

settings = get_settings()

app = FastAPI(
    title="FinDocGPT API",
    description="AI-powered financial document analysis & investment strategy",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(documents.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(forecasting.router, prefix="/api/v1/forecasting", tags=["forecasting"])
app.include_router(strategy.router, prefix="/api/v1/strategy", tags=["strategy"])
app.include_router(chat.router, prefix="/api/v1/chat", tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "FinDocGPT API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "1.0.0"}

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level=settings.LOG_LEVEL.lower()
    )