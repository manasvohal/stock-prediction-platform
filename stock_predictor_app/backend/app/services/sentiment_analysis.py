import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from statistics import mean, stdev
import os

from app.services.data_fetcher import DataFetcher
from app.utils.text_preprocessing import preprocess_news_articles
from app.core.config import settings

logger = logging.getLogger(__name__)

class SentimentAnalysis:
    """Service for analyzing sentiment of news articles related to stocks"""
    
    def __init__(self):
        self.data_fetcher = DataFetcher()
        self.model_name = settings.SENTIMENT_MODEL
        self.model = None
        self.tokenizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _load_model(self):
        """Load the sentiment analysis model"""
        if self.model is None or self.tokenizer is None:
            try:
                logger.info(f"Loading sentiment model: {self.model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
                self.model.to(self.device)
                logger.info("Sentiment model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading sentiment model: {e}")
                raise ValueError(f"Could not load sentiment model: {str(e)}")
    
    def analyze(self, ticker: str, days: int = 7) -> Dict[str, Any]:
        """Analyze sentiment of news articles for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back for news
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Get news articles
            articles = self.data_fetcher.get_news(ticker, days=days)
            
            if not articles:
                return {
                    "ticker": ticker,
                    "sentiment_score": 0,
                    "sentiment_label": "neutral",
                    "sentiment_magnitude": 0,
                    "article_count": 0,
                    "positive_count": 0,
                    "negative_count": 0,
                    "neutral_count": 0,
                    "summary": f"No news articles found for {ticker} in the last {days} days",
                    "articles": []
                }
            
            # Preprocess articles
            processed_articles = preprocess_news_articles(articles)
            
            # Analyze sentiment
            analyzed_articles = self._analyze_articles(processed_articles)
            
            # Calculate aggregate sentiment
            sentiment_scores = [article["sentiment_score"] for article in analyzed_articles]
            positive_count = sum(1 for score in sentiment_scores if score > 0.1)
            negative_count = sum(1 for score in sentiment_scores if score < -0.1)
            neutral_count = len(sentiment_scores) - positive_count - negative_count
            
            # Calculate average sentiment score and magnitude
            avg_sentiment = mean(sentiment_scores) if sentiment_scores else 0
            sentiment_magnitude = stdev(sentiment_scores) if len(sentiment_scores) > 1 else 0
            
            # Determine sentiment label
            sentiment_label = self._get_sentiment_label(avg_sentiment)
            
            # Generate summary
            summary = self._generate_summary(ticker, avg_sentiment, sentiment_magnitude, len(analyzed_articles))
            
            return {
                "ticker": ticker,
                "sentiment_score": avg_sentiment,
                "sentiment_label": sentiment_label,
                "sentiment_magnitude": sentiment_magnitude,
                "article_count": len(analyzed_articles),
                "positive_count": positive_count,
                "negative_count": negative_count,
                "neutral_count": neutral_count,
                "summary": summary,
                "articles": analyzed_articles
            }
        except Exception as e:
            logger.error(f"Error in sentiment analysis for {ticker}: {e}")
            raise ValueError(f"Could not perform sentiment analysis for {ticker}: {str(e)}")
    
    def _analyze_articles(self, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze sentiment of individual articles
        
        Args:
            articles: List of preprocessed news articles
            
        Returns:
            List of articles with sentiment scores
        """
        # Load model if not already loaded
        self._load_model()
        
        analyzed_articles = []
        
        for article in articles:
            try:
                # Get text to analyze
                text = article.get("preprocessed_text", "")
                if not text:
                    text = f"{article.get('title', '')} {article.get('description', '')}"
                
                # Truncate text if too long
                max_length = self.tokenizer.model_max_length
                tokens = self.tokenizer(text, truncation=True, max_length=max_length)
                
                # Get sentiment
                with torch.no_grad():
                    inputs = {k: torch.tensor([v]).to(self.device) for k, v in tokens.items()}
                    outputs = self.model(**inputs)
                    scores = outputs.logits.squeeze().tolist()
                
                # For FinBERT, scores are [negative, neutral, positive]
                if len(scores) == 3:
                    # Convert to a single sentiment score between -1 and 1
                    sentiment_score = (scores[2] - scores[0]) / (scores[0] + scores[1] + scores[2])
                # For binary models, scores are [negative, positive]
                elif len(scores) == 2:
                    sentiment_score = (scores[1] - scores[0]) / (scores[0] + scores[1])
                else:
                    sentiment_score = 0
                
                # Add sentiment to article
                analyzed_article = article.copy()
                analyzed_article["sentiment_score"] = sentiment_score
                analyzed_article["sentiment_label"] = self._get_sentiment_label(sentiment_score)
                
                analyzed_articles.append(analyzed_article)
            except Exception as e:
                logger.error(f"Error analyzing article sentiment: {e}")
                # Add article with neutral sentiment
                analyzed_article = article.copy()
                analyzed_article["sentiment_score"] = 0
                analyzed_article["sentiment_label"] = "neutral"
                analyzed_articles.append(analyzed_article)
        
        return analyzed_articles
    
    def _get_sentiment_label(self, score: float) -> str:
        """Convert sentiment score to label
        
        Args:
            score: Sentiment score between -1 and 1
            
        Returns:
            Sentiment label (positive, neutral, negative)
        """
        if score > 0.1:
            return "positive"
        elif score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    def _generate_summary(self, ticker: str, sentiment_score: float, magnitude: float, article_count: int) -> str:
        """Generate a human-readable summary of sentiment analysis
        
        Args:
            ticker: Stock ticker symbol
            sentiment_score: Average sentiment score
            magnitude: Sentiment magnitude (standard deviation)
            article_count: Number of articles analyzed
            
        Returns:
            Summary string
        """
        # Determine sentiment strength
        if abs(sentiment_score) < 0.1:
            strength = "neutral"
        elif abs(sentiment_score) < 0.3:
            strength = "slightly " + ("positive" if sentiment_score > 0 else "negative")
        elif abs(sentiment_score) < 0.6:
            strength = "moderately " + ("positive" if sentiment_score > 0 else "negative")
        else:
            strength = "strongly " + ("positive" if sentiment_score > 0 else "negative")
        
        # Determine consistency
        if magnitude < 0.2:
            consistency = "consistent"
        elif magnitude < 0.4:
            consistency = "somewhat varied"
        else:
            consistency = "highly varied"
        
        # Generate summary
        summary = f"News sentiment for {ticker} is {strength} ({sentiment_score:.2f}) based on {article_count} recent articles. "
        
        if article_count > 1:
            summary += f"Sentiment is {consistency} across articles."
        
        return summary 