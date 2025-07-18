from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.database import get_db
from app.services.fundamental_analysis import FundamentalAnalysis
from app.utils.json_encoder import convert_numpy_types

router = APIRouter()
fundamental_analyzer = FundamentalAnalysis()

@router.get("/{ticker}")
async def get_fundamental_analysis(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get fundamental analysis for a stock
    """
    try:
        result = fundamental_analyzer.analyze(ticker)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Fundamental analysis failed: {str(e)}")

@router.get("/{ticker}/valuation")
async def get_valuation_metrics(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get valuation metrics for a stock
    """
    try:
        result = fundamental_analyzer.analyze(ticker)
        
        # Extract valuation metrics
        valuation_metrics = {
            "ticker": ticker,
            "pe_ratio": result.get("pe_ratio"),
            "pb_ratio": result.get("pb_ratio"),
            "ps_ratio": result.get("ps_ratio"),
            "peg_ratio": result.get("peg_ratio"),
            "dividend_yield": result.get("dividend_yield"),
            "valuation_score": result.get("valuation_score"),
            "company_name": result.get("company_name"),
            "sector": result.get("sector"),
            "industry": result.get("industry")
        }
        
        return convert_numpy_types(valuation_metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Valuation metrics failed: {str(e)}")

@router.get("/{ticker}/growth")
async def get_growth_metrics(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get growth metrics for a stock
    """
    try:
        result = fundamental_analyzer.analyze(ticker)
        
        # Extract growth metrics
        growth_metrics = {
            "ticker": ticker,
            "revenue_growth": result.get("revenue_growth"),
            "earnings_growth": result.get("earnings_growth"),
            "fcf_growth": result.get("fcf_growth"),
            "growth_score": result.get("growth_score"),
            "company_name": result.get("company_name"),
            "sector": result.get("sector"),
            "industry": result.get("industry")
        }
        
        return convert_numpy_types(growth_metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Growth metrics failed: {str(e)}")

@router.get("/{ticker}/health")
async def get_financial_health(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get financial health metrics for a stock
    """
    try:
        result = fundamental_analyzer.analyze(ticker)
        
        # Extract financial health metrics
        health_metrics = {
            "ticker": ticker,
            "debt_to_equity": result.get("debt_to_equity"),
            "current_ratio": result.get("current_ratio"),
            "roe": result.get("roe"),
            "financial_health_score": result.get("financial_health_score"),
            "company_name": result.get("company_name"),
            "sector": result.get("sector"),
            "industry": result.get("industry")
        }
        
        return convert_numpy_types(health_metrics)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Financial health metrics failed: {str(e)}")

@router.get("/{ticker}/summary")
async def get_fundamental_summary(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get a summary of fundamental analysis
    """
    try:
        result = fundamental_analyzer.analyze(ticker)
        
        # Extract summary
        summary = {
            "ticker": ticker,
            "company_name": result.get("company_name"),
            "sector": result.get("sector"),
            "industry": result.get("industry"),
            "summary": result.get("summary"),
            "valuation_score": result.get("valuation_score"),
            "growth_score": result.get("growth_score"),
            "financial_health_score": result.get("financial_health_score"),
            "business_summary": result.get("business_summary")
        }
        
        return convert_numpy_types(summary)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Fundamental summary failed: {str(e)}") 