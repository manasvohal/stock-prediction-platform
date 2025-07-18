from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.core.database import get_db
from app.services.sentiment_analysis import SentimentAnalysis
from app.utils.json_encoder import convert_numpy_types

router = APIRouter()
sentiment_analyzer = SentimentAnalysis()

@router.get("/{ticker}")
async def get_sentiment_analysis(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get sentiment analysis for a stock
    """
    try:
        result = sentiment_analyzer.analyze(ticker, days=days)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sentiment analysis failed: {str(e)}")

@router.get("/{ticker}/score")
async def get_sentiment_score(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get sentiment score for a stock
    """
    try:
        result = sentiment_analyzer.analyze(ticker, days=days)
        
        # Extract sentiment score
        score = {
            "ticker": ticker,
            "days": days,
            "sentiment_score": result.get("sentiment_score"),
            "sentiment_label": result.get("sentiment_label"),
            "company_name": result.get("company_name")
        }
        
        return convert_numpy_types(score)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sentiment score failed: {str(e)}")

@router.get("/{ticker}/articles")
async def get_sentiment_articles(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    limit: int = Query(10, description="Maximum number of articles to return"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get sentiment articles for a stock
    """
    try:
        result = sentiment_analyzer.analyze(ticker, days=days)
        
        # Extract articles
        articles = {
            "ticker": ticker,
            "days": days,
            "articles": result.get("articles", [])[:limit],
            "company_name": result.get("company_name")
        }
        
        return convert_numpy_types(articles)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sentiment articles failed: {str(e)}")

@router.get("/{ticker}/distribution")
async def get_sentiment_distribution(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get sentiment distribution for a stock
    """
    try:
        result = sentiment_analyzer.analyze(ticker, days=days)
        
        # Extract distribution
        distribution = {
            "ticker": ticker,
            "days": days,
            "positive": result.get("distribution", {}).get("positive", 0),
            "negative": result.get("distribution", {}).get("negative", 0),
            "neutral": result.get("distribution", {}).get("neutral", 0),
            "company_name": result.get("company_name")
        }
        
        return convert_numpy_types(distribution)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sentiment distribution failed: {str(e)}")

@router.get("/{ticker}/summary")
async def get_sentiment_summary(
    ticker: str, 
    days: int = Query(7, description="Number of days to look back for news"),
    db: Session = Depends(get_db)
) -> Dict[str, Any]:
    """
    Get a summary of sentiment analysis
    """
    try:
        result = sentiment_analyzer.analyze(ticker, days=days)
        
        # Extract summary
        summary = {
            "ticker": ticker,
            "days": days,
            "sentiment_label": result.get("sentiment_label"),
            "sentiment_score": result.get("sentiment_score"),
            "summary": result.get("summary"),
            "company_name": result.get("company_name"),
            "article_count": result.get("article_count")
        }
        
        return convert_numpy_types(summary)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Sentiment summary failed: {str(e)}") 