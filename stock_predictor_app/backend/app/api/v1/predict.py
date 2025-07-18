from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import Dict, Any, List

from app.core.database import get_db
from app.services.predictor import Predictor
from app.services.ml_predictor import MLPredictor
from app.utils.json_encoder import convert_numpy_types

router = APIRouter()
predictor = Predictor()
ml_predictor = MLPredictor()

@router.get("/{ticker}")
async def get_prediction(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get stock price prediction
    """
    try:
        result = predictor.predict(ticker)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Prediction failed: {str(e)}")

@router.get("/{ticker}/recommendation")
async def get_recommendation(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get investment recommendation
    """
    try:
        result = predictor.predict(ticker)
        
        # Extract recommendation
        recommendation = {
            "ticker": ticker,
            "recommendation": result.get("prediction"),
            "recommendation_rating": result.get("predicted_return"),
            "confidence": result.get("confidence"),
            "risk_level": "medium",  # Default value
            "company_name": result.get("ticker")
        }
        
        return convert_numpy_types(recommendation)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Recommendation failed: {str(e)}")

@router.get("/{ticker}/factors")
async def get_prediction_factors(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get factors affecting prediction
    """
    try:
        result = predictor.predict(ticker)
        
        # Extract factors
        factors = {
            "ticker": ticker,
            "factors": {
                "fundamental": result.get("fundamental_factors", {}),
                "technical": result.get("technical_factors", {}),
                "sentiment": result.get("sentiment_factors", {})
            },
            "company_name": result.get("ticker")
        }
        
        return convert_numpy_types(factors)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Prediction factors failed: {str(e)}")

@router.get("/{ticker}/importance")
async def get_feature_importance(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get feature importance for prediction
    """
    try:
        result = predictor.predict(ticker)
        
        # Extract feature importance
        importance = {
            "ticker": ticker,
            "feature_importance": result.get("feature_importance", {}),
            "company_name": result.get("ticker")
        }
        
        return convert_numpy_types(importance)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Feature importance failed: {str(e)}")

@router.get("/{ticker}/summary")
async def get_prediction_summary(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get a summary of prediction
    """
    try:
        result = predictor.predict(ticker)
        
        # Extract summary
        summary = {
            "ticker": ticker,
            "current_price": 0,  # Would need to get from data
            "predicted_price": 0,  # Would need to calculate
            "predicted_change": 0,  # Would need to calculate
            "predicted_change_percent": result.get("predicted_return"),
            "recommendation": result.get("prediction"),
            "confidence": result.get("confidence"),
            "summary": result.get("explanation"),
            "company_name": result.get("ticker")
        }
        
        return convert_numpy_types(summary)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"Prediction summary failed: {str(e)}")

@router.get("/{ticker}/ml")
async def get_ml_prediction(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get advanced ML-based stock price prediction
    """
    try:
        result = ml_predictor.predict(ticker)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"ML prediction failed: {str(e)}")

@router.get("/{ticker}/ml/forecast")
async def get_ml_forecast(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Get detailed forecast data from ML model
    """
    try:
        result = ml_predictor.predict(ticker)
        
        # Extract forecast data
        forecast = {
            "ticker": ticker,
            "company_name": result.get("company_name", ticker),
            "current_price": result.get("current_price"),
            "forecast": result.get("forecast", []),
            "forecast_days": result.get("forecast_days"),
            "model_type": result.get("model_type")
        }
        
        return convert_numpy_types(forecast)
    except Exception as e:
        raise HTTPException(status_code=404, detail=f"ML forecast failed: {str(e)}")

@router.post("/{ticker}/ml/train")
async def train_ml_model(ticker: str, epochs: int = 50, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Train ML model for a specific stock
    """
    try:
        result = ml_predictor.train(ticker, epochs=epochs)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML model training failed: {str(e)}")

@router.get("/{ticker}/ml/evaluate")
async def evaluate_ml_model(ticker: str, db: Session = Depends(get_db)) -> Dict[str, Any]:
    """
    Evaluate ML model performance on historical data
    """
    try:
        result = ml_predictor.evaluate(ticker)
        return convert_numpy_types(result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"ML model evaluation failed: {str(e)}") 