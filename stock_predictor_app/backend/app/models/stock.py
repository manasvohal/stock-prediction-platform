from sqlalchemy import Column, Integer, String, Float, Date, DateTime, func, ForeignKey
from sqlalchemy.orm import relationship

from app.core.database import Base

class Stock(Base):
    """Stock model for storing basic stock information"""
    __tablename__ = "stocks"

    id = Column(Integer, primary_key=True, index=True)
    ticker = Column(String, unique=True, index=True, nullable=False)
    company_name = Column(String, nullable=False)
    sector = Column(String)
    industry = Column(String)
    country = Column(String)
    market_cap = Column(Float)
    last_updated = Column(DateTime, default=func.now(), onupdate=func.now())
    
    # Relationships
    price_history = relationship("PriceHistory", back_populates="stock", cascade="all, delete-orphan")
    fundamentals = relationship("Fundamentals", back_populates="stock", cascade="all, delete-orphan")
    analysis_results = relationship("Analysis", back_populates="stock", cascade="all, delete-orphan")

class PriceHistory(Base):
    """Model for storing historical price data"""
    __tablename__ = "price_history"
    
    id = Column(Integer, primary_key=True, index=True)
    stock_id = Column(Integer, ForeignKey("stocks.id"), nullable=False)
    date = Column(Date, nullable=False)
    open = Column(Float, nullable=False)
    high = Column(Float, nullable=False)
    low = Column(Float, nullable=False)
    close = Column(Float, nullable=False)
    adjusted_close = Column(Float, nullable=False)
    volume = Column(Integer, nullable=False)
    
    # Relationship
    stock = relationship("Stock", back_populates="price_history") 