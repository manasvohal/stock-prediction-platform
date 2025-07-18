from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.core.database import get_db
from app.services.data_fetcher import DataFetcher
from app.utils.json_encoder import convert_numpy_types

router = APIRouter()
data_fetcher = DataFetcher()

@router.get("/search")
async def search_stocks(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Maximum number of results to return"),
    db: Session = Depends(get_db)
) -> List[Dict[str, Any]]:
    """
    Search for stocks by ticker or company name
    """
    try:
        # This would typically query a database or API
        # For now, we'll return a mock result
        results = [
            {"ticker": "AAPL", "name": "Apple Inc.", "exchange": "NASDAQ"},
            {"ticker": "MSFT", "name": "Microsoft Corporation", "exchange": "NASDAQ"},
            {"ticker": "GOOGL", "name": "Alphabet Inc.", "exchange": "NASDAQ"},
            {"ticker": "AMZN", "name": "Amazon.com, Inc.", "exchange": "NASDAQ"},
            {"ticker": "META", "name": "Meta Platforms, Inc.", "exchange": "NASDAQ"},
        ]
        
        # Filter by query
        filtered = [r for r in results if query.upper() in r["ticker"] or query.lower() in r["name"].lower()]
        
        # Limit results
        limited = filtered[:limit]
        
        return convert_numpy_types(limited)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@router.get("/{ticker}")
async def get_stock_info(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get basic stock information
    """
    try:
        result = data_fetcher.get_stock_info(ticker)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Stock not found: {str(e)}")

@router.get("/{ticker}/chart")
async def get_stock_chart(
    ticker: str, 
    period: str = Query("1y", description="Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get stock chart data
    """
    try:
        df = data_fetcher.get_historical_prices(ticker, period=period)
        
        # Convert DataFrame to dict
        chart_data = df.to_dict(orient="records")
        
        result = {
            "ticker": ticker,
            "period": period,
            "data": chart_data
        }
        
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Chart data not found: {str(e)}")

@router.get("/{ticker}/news")
async def get_stock_news(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get news articles for a stock
    """
    try:
        news = data_fetcher.get_news(ticker, days=days)
        
        result = {
            "ticker": ticker,
            "articles": news
        }
        
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"News not found: {str(e)}") 