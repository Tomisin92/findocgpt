"""
Database models for FinDocGPT
"""

from sqlalchemy import Column, Integer, String, Float, DateTime, Text, JSON, Boolean
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class Document(Base):
    __tablename__ = "documents"
    
    id = Column(String, primary_key=True)
    filename = Column(String, nullable=False)
    file_type = Column(String, nullable=False)
    content = Column(Text)
    summary = Column(Text)
    key_metrics = Column(JSON)
    upload_date = Column(DateTime, default=datetime.utcnow)
    processed = Column(Boolean, default=False)

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    symbol = Column(String, nullable=False)
    model_type = Column(String, nullable=False)
    prediction_date = Column(DateTime, default=datetime.utcnow)
    target_date = Column(DateTime, nullable=False)
    predicted_price = Column(Float, nullable=False)
    confidence = Column(Float)
    actual_price = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)

class Strategy(Base):
    __tablename__ = "strategies"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    description = Column(Text)
    parameters = Column(JSON)
    performance_metrics = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow)

class Portfolio(Base):
    __tablename__ = "portfolios"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String, nullable=False)
    symbols = Column(JSON)  # List of symbols
    weights = Column(JSON)  # Allocation weights
    total_value = Column(Float)
    risk_tolerance = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)
    rebalanced_at = Column(DateTime)