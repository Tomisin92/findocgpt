# backend/app/api/v1/documents.py
"""
Stage 1: Document Analysis API endpoints
"""

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from typing import List, Dict, Any
import json
from app.core.stage1_document_ai.document_processor import DocumentProcessor
from app.core.stage1_document_ai.qa_engine import QAEngine
from app.core.stage1_document_ai.sentiment_analyzer import SentimentAnalyzer
from app.core.stage1_document_ai.anomaly_detector import AnomalyDetector

router = APIRouter()

# Initialize processors
doc_processor = DocumentProcessor()
qa_engine = QAEngine()
sentiment_analyzer = SentimentAnalyzer()
anomaly_detector = AnomalyDetector()

@router.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload and process financial document"""
    try:
        # Read file content
        content = await file.read()
        
        # Process document
        processed_doc = await doc_processor.process_document(content, file.filename)
        
        return {
            "message": "Document uploaded successfully",
            "document_id": processed_doc["id"],
            "filename": file.filename,
            "size": file.size,
            "pages": processed_doc["pages"],
            "summary": processed_doc["summary"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/question")
async def ask_question(question: str, document_id: str = None):
    """Ask questions about financial documents"""
    try:
        # Get answer using Q&A engine
        result = await qa_engine.answer_question(question, document_id)
        
        return {
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "sources": result["sources"],
            "reasoning": result["reasoning"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/sentiment/{symbol}")
async def get_sentiment_analysis(symbol: str):
    """Get sentiment analysis for a company"""
    try:
        # Analyze sentiment from various sources
        sentiment_data = await sentiment_analyzer.analyze_company_sentiment(symbol)
        
        return {
            "symbol": symbol,
            "overall_sentiment": sentiment_data["overall"],
            "sentiment_score": sentiment_data["score"],
            "sources": sentiment_data["sources"],
            "key_themes": sentiment_data["themes"],
            "summary": sentiment_data["summary"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/anomalies/{symbol}")
async def detect_anomalies(symbol: str):
    """Detect financial anomalies"""
    try:
        # Detect anomalies in financial data
        anomalies = await anomaly_detector.detect_anomalies(symbol)
        
        return {
            "symbol": symbol,
            "anomalies_found": len(anomalies),
            "anomalies": anomalies,
            "risk_level": anomaly_detector.calculate_risk_level(anomalies),
            "recommendations": anomaly_detector.get_recommendations(anomalies)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
