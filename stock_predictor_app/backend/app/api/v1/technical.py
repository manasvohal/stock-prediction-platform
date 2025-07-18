from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.core.database import get_db
from app.services.technical_analysis import TechnicalAnalysis
from app.utils.json_encoder import convert_numpy_types

router = APIRouter()
technical_analyzer = TechnicalAnalysis()

@router.get("/{ticker}")
async def get_technical_analysis(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get technical analysis for a stock
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Technical analysis failed: {str(e)}")

@router.get("/{ticker}/indicators")
async def get_technical_indicators(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get technical indicators for a stock
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        
        # Extract indicators
        indicators = {
            "ticker": ticker,
            "period": period,
            "last_price": result.get("last_price"),
            "moving_averages": result.get("indicators", {}).get("moving_averages", {}),
            "rsi": result.get("indicators", {}).get("rsi"),
            "macd": result.get("indicators", {}).get("macd", {}),
            "bollinger_bands": result.get("indicators", {}).get("bollinger_bands", {}),
            "atr": result.get("indicators", {}).get("atr"),
            "volume": result.get("indicators", {}).get("volume", {})
        }
        
        return convert_numpy_types(indicators)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Technical indicators failed: {str(e)}")

@router.get("/{ticker}/trend")
async def get_trend_analysis(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get trend analysis for a stock
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        
        # Extract trend information
        trend_info = {
            "ticker": ticker,
            "period": period,
            "trend": result.get("trend"),
            "trend_strength": result.get("trend_strength"),
            "price_change": result.get("price_change"),
            "price_change_percent": result.get("price_change_percent"),
            "last_price": result.get("last_price")
        }
        
        return convert_numpy_types(trend_info)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Trend analysis failed: {str(e)}")

@router.get("/{ticker}/patterns")
async def get_chart_patterns(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get chart patterns for a stock
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        
        # Extract pattern information
        patterns = {
            "ticker": ticker,
            "period": period,
            "detected_patterns": result.get("patterns", {}).get("detected", []),
            "pattern_descriptions": result.get("patterns", {}).get("descriptions", {})
        }
        
        return convert_numpy_types(patterns)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Chart pattern analysis failed: {str(e)}")

@router.get("/{ticker}/support-resistance")
async def get_support_resistance(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get support and resistance levels for a stock
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        
        # Extract support and resistance levels
        levels = {
            "ticker": ticker,
            "period": period,
            "last_price": result.get("last_price"),
            "support_levels": result.get("support_resistance", {}).get("support", []),
            "resistance_levels": result.get("support_resistance", {}).get("resistance", [])
        }
        
        return convert_numpy_types(levels)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Support/resistance analysis failed: {str(e)}")

@router.get("/{ticker}/summary")
async def get_technical_summary(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a summary of technical analysis
    """
    try:
        result = technical_analyzer.analyze(ticker, period=period)
        
        # Extract summary
        summary = {
            "ticker": ticker,
            "period": period,
            "last_price": result.get("last_price"),
            "summary": result.get("summary"),
            "trend": result.get("trend"),
            "trend_strength": result.get("trend_strength"),
            "price_change_percent": result.get("price_change_percent")
        }
        
        return convert_numpy_types(summary)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Technical summary failed: {str(e)}") 