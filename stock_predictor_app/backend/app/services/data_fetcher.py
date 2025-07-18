import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import logging
from typing import Dict, List, Optional, Tuple, Any

from app.core.config import settings

logger = logging.getLogger(__name__)

class DataFetcher:
    """Service for fetching stock data from various sources"""
    
    def __init__(self):
        self.newsapi_key = settings.NEWS_API_KEY
    
    def get_stock_info(self, ticker: str) -> Dict[str, Any]:
        """Get basic stock information"""
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Extract relevant info
            return {
                "ticker": ticker,
                "company_name": info.get("longName", ""),
                "sector": info.get("sector", ""),
                "industry": info.get("industry", ""),
                "country": info.get("country", ""),
                "market_cap": info.get("marketCap", 0),
                "current_price": info.get("currentPrice", 0),
                "currency": info.get("currency", "USD"),
                "exchange": info.get("exchange", ""),
                "website": info.get("website", ""),
                "logo_url": info.get("logo_url", ""),
                "business_summary": info.get("longBusinessSummary", "")
            }
        except Exception as e:
            logger.error(f"Error fetching stock info for {ticker}: {e}")
            raise ValueError(f"Could not fetch data for ticker {ticker}")
    
    def get_historical_prices(self, ticker: str, period: str = "1y") -> pd.DataFrame:
        """Get historical price data
        
        Args:
            ticker: Stock ticker symbol
            period: Time period (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
        
        Returns:
            DataFrame with historical price data
        """
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            
            # Reset index to make Date a column
            df = df.reset_index()
            
            # Rename columns to lowercase
            df.columns = [col.lower() for col in df.columns]
            
            return df
        except Exception as e:
            logger.error(f"Error fetching historical prices for {ticker}: {e}")
            raise ValueError(f"Could not fetch historical data for ticker {ticker}")
    
    def get_financial_statements(self, ticker: str) -> Dict[str, pd.DataFrame]:
        """Get income statement, balance sheet, and cash flow statement"""
        try:
            stock = yf.Ticker(ticker)
            
            return {
                "income_statement": stock.income_stmt,
                "balance_sheet": stock.balance_sheet,
                "cash_flow": stock.cashflow
            }
        except Exception as e:
            logger.error(f"Error fetching financial statements for {ticker}: {e}")
            raise ValueError(f"Could not fetch financial statements for ticker {ticker}")
    
    def get_news(self, ticker: str, days: int = 7) -> List[Dict[str, Any]]:
        """Get news articles for a stock
        
        Args:
            ticker: Stock ticker symbol
            days: Number of days to look back
            
        Returns:
            List of news articles with title, description, url, etc.
        """
        if not self.newsapi_key:
            logger.warning("NewsAPI key not set. Using Yahoo Finance news as fallback.")
            return self._get_yahoo_news(ticker)
        
        try:
            # Calculate date range
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            # Format dates for NewsAPI
            from_date = start_date.strftime("%Y-%m-%d")
            to_date = end_date.strftime("%Y-%m-%d")
            
            # Make request to NewsAPI
            url = f"https://newsapi.org/v2/everything"
            params = {
                "q": f"{ticker} OR {self._get_company_name(ticker)}",
                "from": from_date,
                "to": to_date,
                "language": "en",
                "sortBy": "relevancy",
                "apiKey": self.newsapi_key
            }
            
            response = requests.get(url, params=params)
            data = response.json()
            
            if data.get("status") != "ok":
                logger.warning(f"NewsAPI error: {data.get('message')}")
                return self._get_yahoo_news(ticker)
            
            articles = data.get("articles", [])
            return articles
        except Exception as e:
            logger.error(f"Error fetching news for {ticker}: {e}")
            return self._get_yahoo_news(ticker)
    
    def _get_yahoo_news(self, ticker: str) -> List[Dict[str, Any]]:
        """Fallback method to get news from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            # Format news to match NewsAPI format
            articles = []
            for item in news:
                articles.append({
                    "title": item.get("title", ""),
                    "description": item.get("summary", ""),
                    "url": item.get("link", ""),
                    "publishedAt": datetime.fromtimestamp(item.get("providerPublishTime", 0)).isoformat(),
                    "source": {"name": item.get("publisher", "")}
                })
            
            return articles
        except Exception as e:
            logger.error(f"Error fetching Yahoo news for {ticker}: {e}")
            return []
    
    def _get_company_name(self, ticker: str) -> str:
        """Get company name from ticker for better news search"""
        try:
            stock = yf.Ticker(ticker)
            return stock.info.get("longName", ticker)
        except:
            return ticker 