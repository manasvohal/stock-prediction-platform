from sqlalchemy import Column, Integer, String, Float, DateTime, JSON, ForeignKey, func, Text
from sqlalchemy.orm import relationship

from app.core.database import Base

class Fundamentals(Base):
    """Model for storing fundamental analysis data"""
    __tablename__ = "fundamentals"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, default=func.now())
    
    # Financial ratios
    pe_ratio = Column(Float)
    pb_ratio = Column(Float)
    ps_ratio = Column(Float)
    peg_ratio = Column(Float)
    debt_to_equity = Column(Float)
    current_ratio = Column(Float)
    quick_ratio = Column(Float)
    roe = Column(Float)  # Return on Equity
    roa = Column(Float)  # Return on Assets
    
    # Growth metrics
    revenue_growth = Column(Float)
    earnings_growth = Column(Float)
    
    # Financial data
    revenue = Column(Float)
    net_income = Column(Float)
    eps = Column(Float)
    dividend_yield = Column(Float)
    
    # Additional data
    raw_data = Column(JSON)  # Store additional data in JSON format
    
    # Relationship
    stock = relationship("Stock", back_populates="fundamentals")

class Analysis(Base):
    """Model for storing analysis results and predictions"""
    __tablename__ = "analysis"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(DateTime, default=func.now())
    
    # Technical indicators
    technical_indicators = Column(JSON)  # Store as JSON: MA, RSI, MACD, etc.
    
    # Sentiment analysis
    sentiment_score = Column(Float)  # -1 to 1 (negative to positive)
    sentiment_magnitude = Column(Float)  # 0 to infinity (strength of sentiment)
    news_summary = Column(Text)
    
    # Prediction results
    prediction = Column(String)  # "buy", "hold", "sell"
    confidence = Column(Float)  # 0 to 1
    predicted_return = Column(Float)  # Predicted percentage return
    explanation = Column(Text)  # Human-readable explanation
    feature_importance = Column(JSON)  # Store SHAP values or feature importance
    
    # Relationship
    stock = relationship("Stock", back_populates="analysis_results") 