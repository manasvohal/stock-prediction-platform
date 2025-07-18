import pandas as pd
import numpy as np
import pickle
import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import joblib
from datetime import datetime, timedelta

from app.services.data_fetcher import DataFetcher
from app.services.fundamental_analysis import FundamentalAnalysis
from app.services.technical_analysis import TechnicalAnalysis
from app.services.sentiment_analysis import SentimentAnalysis
from app.core.config import settings

logger = logging.getLogger(__name__)

class Predictor:
    """Service for predicting stock performance based on multiple factors"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.fundamental_analyzer = FundamentalAnalysis()
        self.technical_analyzer = TechnicalAnalysis()
        self.sentiment_analyzer = SentimentAnalysis()
        self.model = None
        self.model_path = settings.MODEL_PATH
    
    def _load_model(self):
        """Load the prediction model"""
        if self.model is None:
            try:
                if os.path.exists(self.model_path):
                    logger.info(f"Loading prediction model from {self.model_path}")
                    self.model = joblib.load(self.model_path)
                    logger.info("Prediction model loaded successfully")
                else:
                    logger.warning(f"Model file {self.model_path} not found. Using fallback model.")
                    self.model = self._create_fallback_model()
            except Exception as e:
                logger.error(f"Error loading prediction model: {e}")
                self.model = self._create_fallback_model()
    
    def _create_fallback_model(self):
        """Create a simple fallback model when the trained model is not available"""
        # This is a simple rule-based model that will be used if the trained model is not available
        logger.info("Creating fallback prediction model")
        return {"type": "fallback"}
    
    def predict(self, ticker: str) -> Dict[str, Any]:
        """Predict stock performance
        
        Args:
            ticker: Stock ticker symbol
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Load the model
            self._load_model()
            
            # Get fundamental analysis
            fundamental_data = self.fundamental_analyzer.analyze(ticker)
            
            # Get technical analysis
            technical_data = self.technical_analyzer.analyze(ticker)
            
            # Get sentiment analysis
            sentiment_data = self.sentiment_analyzer.analyze(ticker)
            
            # Prepare features for prediction
            features = self._prepare_features(fundamental_data, technical_data, sentiment_data)
            
            # Make prediction
            prediction, confidence, explanation = self._make_prediction(features)
            
            # Calculate predicted return
            predicted_return = self._calculate_predicted_return(prediction, confidence, features)
            
            # Generate detailed explanation
            detailed_explanation = self._generate_explanation(
                ticker, 
                prediction, 
                confidence, 
                predicted_return,
                fundamental_data,
                technical_data,
                sentiment_data
            )
            
            # Prepare result
            result = {
                "ticker": ticker,
                "prediction": prediction,
                "confidence": confidence,
                "predicted_return": predicted_return,
                "explanation": detailed_explanation,
                "prediction_date": datetime.now().isoformat(),
                "time_horizon": "3 months",
                "fundamental_factors": self._extract_key_factors(fundamental_data),
                "technical_factors": self._extract_key_factors(technical_data),
                "sentiment_factors": self._extract_key_factors(sentiment_data),
                "feature_importance": self._get_feature_importance(features)
            }
            
            return result
        except Exception as e:
            logger.error(f"Error in prediction for {ticker}: {e}")
            raise ValueError(f"Could not make prediction for {ticker}: {str(e)}")
    
    def _prepare_features(
        self,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Prepare features for the prediction model"""
        features = {}
        
        # Add fundamental features
        features["pe_ratio"] = fundamental_data.get("pe_ratio", 0) or 0
        features["pb_ratio"] = fundamental_data.get("pb_ratio", 0) or 0
        features["debt_to_equity"] = fundamental_data.get("debt_to_equity", 0) or 0
        features["current_ratio"] = fundamental_data.get("current_ratio", 0) or 0
        features["roe"] = fundamental_data.get("roe", 0) or 0
        features["revenue_growth"] = fundamental_data.get("revenue_growth", 0) or 0
        features["earnings_growth"] = fundamental_data.get("earnings_growth", 0) or 0
        features["dividend_yield"] = fundamental_data.get("dividend_yield", 0) or 0
        features["valuation_score"] = fundamental_data.get("valuation_score", 0.5)
        features["growth_score"] = fundamental_data.get("growth_score", 0.5)
        features["financial_health_score"] = fundamental_data.get("financial_health_score", 0.5)
        
        # Add technical features
        features["rsi"] = technical_data.get("indicators", {}).get("rsi", 50) or 50
        features["macd"] = technical_data.get("indicators", {}).get("macd", {}).get("macd", 0) or 0
        features["macd_signal"] = technical_data.get("indicators", {}).get("macd", {}).get("signal", 0) or 0
        features["price_to_ma20"] = self._calculate_price_to_ma(technical_data, "ma20")
        features["price_to_ma50"] = self._calculate_price_to_ma(technical_data, "ma50")
        features["price_to_ma200"] = self._calculate_price_to_ma(technical_data, "ma200")
        features["trend_strength"] = technical_data.get("trend_strength", 0.5)
        
        # Add sentiment features
        features["sentiment_score"] = sentiment_data.get("sentiment_score", 0)
        features["sentiment_magnitude"] = sentiment_data.get("sentiment_magnitude", 0)
        features["positive_news_ratio"] = self._calculate_positive_news_ratio(sentiment_data)
        
        # Add market context (simplified)
        features["market_condition"] = 0  # Neutral by default
        
        return features
    
    def _calculate_price_to_ma(self, technical_data: Dict[str, Any], ma_key: str) -> float:
        """Calculate ratio of current price to moving average"""
        try:
            price = technical_data.get("last_price", 0)
            ma = technical_data.get("indicators", {}).get("moving_averages", {}).get(ma_key, 0)
            
            if price and ma and ma > 0:
                return price / ma - 1  # Return as percentage difference
            return 0
        except:
            return 0
    
    def _calculate_positive_news_ratio(self, sentiment_data: Dict[str, Any]) -> float:
        """Calculate ratio of positive news articles"""
        try:
            positive = sentiment_data.get("positive_count", 0)
            total = sentiment_data.get("article_count", 0)
            
            if total > 0:
                return positive / total
            return 0.5  # Neutral if no articles
        except:
            return 0.5
    
    def _make_prediction(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """Make prediction using the loaded model
        
        Returns:
            Tuple of (prediction, confidence, explanation)
        """
        # If using a trained model
        if self.model and self.model.get("type") != "fallback":
            try:
                # Convert features to format expected by model
                X = pd.DataFrame([features])
                
                # Get prediction
                prediction_proba = self.model.predict_proba(X)[0]
                prediction_idx = np.argmax(prediction_proba)
                confidence = prediction_proba[prediction_idx]
                
                # Map index to label
                labels = ["sell", "hold", "buy"]
                prediction = labels[prediction_idx]
                
                # Get explanation from model if available
                explanation = "Based on the trained model analysis"
                
                return prediction, confidence, explanation
            except Exception as e:
                logger.error(f"Error using trained model: {e}")
                # Fall back to rule-based model
                return self._rule_based_prediction(features)
        else:
            # Use rule-based model
            return self._rule_based_prediction(features)
    
    def _rule_based_prediction(self, features: Dict[str, float]) -> Tuple[str, float, str]:
        """Make prediction using a rule-based approach"""
        # Calculate fundamental score (0-1, higher is better)
        fundamental_score = (
            (1 - min(1, max(0, features["valuation_score"]))) * 0.3 +  # Lower valuation score is better
            features["growth_score"] * 0.4 +
            features["financial_health_score"] * 0.3
        )
        
        # Calculate technical score (0-1, higher is better)
        technical_score = 0.5  # Neutral by default
        
        # Adjust based on trend
        technical_score = features["trend_strength"]
        
        # Adjust based on RSI
        rsi = features["rsi"]
        if rsi < 30:
            technical_score += 0.1  # Oversold, potentially bullish
        elif rsi > 70:
            technical_score -= 0.1  # Overbought, potentially bearish
        
        # Adjust based on MACD
        if features["macd"] > features["macd_signal"]:
            technical_score += 0.1  # Bullish signal
        else:
            technical_score -= 0.1  # Bearish signal
        
        # Ensure technical score is between 0 and 1
        technical_score = max(0, min(1, technical_score))
        
        # Calculate sentiment score (0-1, higher is better)
        sentiment_score = (features["sentiment_score"] + 1) / 2  # Convert from -1,1 to 0,1
        
        # Calculate final score (weighted average)
        final_score = (
            fundamental_score * 0.5 +
            technical_score * 0.3 +
            sentiment_score * 0.2
        )
        
        # Determine prediction and confidence
        if final_score > 0.65:
            prediction = "buy"
            confidence = min(1.0, (final_score - 0.65) * 2.85 + 0.5)  # Scale to 0.5-1.0
        elif final_score < 0.35:
            prediction = "sell"
            confidence = min(1.0, (0.35 - final_score) * 2.85 + 0.5)  # Scale to 0.5-1.0
        else:
            prediction = "hold"
            confidence = 1.0 - abs(final_score - 0.5) * 2  # Higher confidence near 0.5
        
        # Generate explanation
        explanation = "Based on a combination of fundamental, technical, and sentiment factors"
        
        return prediction, confidence, explanation
    
    def _calculate_predicted_return(self, prediction: str, confidence: float, features: Dict[str, float]) -> float:
        """Calculate predicted return percentage"""
        # Base expected returns for each prediction
        base_returns = {
            "buy": 15.0,  # 15% expected return for buy
            "hold": 5.0,  # 5% expected return for hold
            "sell": -10.0  # -10% expected return for sell
        }
        
        # Get base return for prediction
        base_return = base_returns.get(prediction, 0.0)
        
        # Adjust based on confidence
        adjusted_return = base_return * (0.5 + confidence / 2)
        
        # Adjust based on growth score
        growth_factor = features.get("growth_score", 0.5) * 2  # 0-2 range
        adjusted_return *= growth_factor
        
        # Adjust based on sentiment
        sentiment_score = features.get("sentiment_score", 0)
        sentiment_adjustment = 1 + (sentiment_score * 0.2)  # 0.8-1.2 range
        adjusted_return *= sentiment_adjustment
        
        return round(adjusted_return, 2)
    
    def _extract_key_factors(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key factors from analysis data"""
        # This is a simplified version that extracts a few key metrics
        # In a real implementation, this would be more comprehensive
        key_factors = {}
        
        # Extract fundamental factors
        if "pe_ratio" in data:
            key_factors["pe_ratio"] = data.get("pe_ratio")
            key_factors["growth_score"] = data.get("growth_score")
            key_factors["valuation_score"] = data.get("valuation_score")
            key_factors["financial_health_score"] = data.get("financial_health_score")
        
        # Extract technical factors
        if "indicators" in data:
            indicators = data.get("indicators", {})
            key_factors["rsi"] = indicators.get("rsi")
            key_factors["trend"] = data.get("trend")
            key_factors["trend_strength"] = data.get("trend_strength")
            
            macd = indicators.get("macd", {})
            key_factors["macd"] = macd.get("macd")
            key_factors["macd_signal"] = macd.get("signal")
        
        # Extract sentiment factors
        if "sentiment_score" in data:
            key_factors["sentiment_score"] = data.get("sentiment_score")
            key_factors["sentiment_label"] = data.get("sentiment_label")
            key_factors["sentiment_magnitude"] = data.get("sentiment_magnitude")
        
        return key_factors
    
    def _get_feature_importance(self, features: Dict[str, float]) -> Dict[str, float]:
        """Get feature importance for the prediction"""
        # If using a trained model with feature importance
        if self.model and hasattr(self.model, "feature_importances_"):
            try:
                feature_names = list(features.keys())
                importances = self.model.feature_importances_
                return {name: float(imp) for name, imp in zip(feature_names, importances)}
            except:
                # Fall back to predefined importances
                return self._get_predefined_feature_importance()
        else:
            # Use predefined feature importance
            return self._get_predefined_feature_importance()
    
    def _get_predefined_feature_importance(self) -> Dict[str, float]:
        """Get predefined feature importance"""
        return {
            "valuation_score": 0.15,
            "growth_score": 0.15,
            "financial_health_score": 0.10,
            "trend_strength": 0.10,
            "rsi": 0.05,
            "macd": 0.05,
            "price_to_ma50": 0.05,
            "price_to_ma200": 0.05,
            "sentiment_score": 0.10,
            "positive_news_ratio": 0.05,
            "pe_ratio": 0.05,
            "revenue_growth": 0.05,
            "earnings_growth": 0.05
        }
    
    def _generate_explanation(
        self,
        ticker: str,
        prediction: str,
        confidence: float,
        predicted_return: float,
        fundamental_data: Dict[str, Any],
        technical_data: Dict[str, Any],
        sentiment_data: Dict[str, Any]
    ) -> str:
        """Generate a detailed explanation for the prediction"""
        company_name = fundamental_data.get("company_name", ticker)
        
        # Start with prediction
        if prediction == "buy":
            explanation = f"{company_name} ({ticker}) appears to be a good investment opportunity "
        elif prediction == "sell":
            explanation = f"{company_name} ({ticker}) does not appear to be a good investment at this time "
        else:
            explanation = f"{company_name} ({ticker}) appears to be fairly valued at the moment "
        
        # Add confidence
        confidence_text = "high" if confidence > 0.8 else "moderate" if confidence > 0.6 else "low"
        explanation += f"with {confidence_text} confidence ({confidence:.0%}). "
        
        # Add predicted return
        explanation += f"The model predicts a potential {predicted_return:.1f}% return over the next 3 months. "
        
        # Add fundamental factors
        explanation += fundamental_data.get("summary", "")
        
        # Add technical factors
        explanation += " " + technical_data.get("summary", "")
        
        # Add sentiment factors
        explanation += " " + sentiment_data.get("summary", "")
        
        # Add recommendation
        if prediction == "buy":
            explanation += " Based on the combined analysis, this stock is recommended for purchase."
        elif prediction == "sell":
            explanation += " Based on the combined analysis, this stock is not recommended for purchase at this time."
        else:
            explanation += " Based on the combined analysis, investors should hold existing positions but may want to wait for a better entry point before making new purchases."
        
        return explanation 