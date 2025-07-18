#!/usr/bin/env python3
"""
Database initialization script for the Stock Predictor application.
This script creates the necessary database tables and populates them with sample data.
"""

import os
import sys
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add the parent directory to the path so we can import from the app
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()

# Get database URL from environment or use default
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/stockpredictor")

def create_tables(engine):
    """Create database tables if they don't exist."""
    print("Creating database tables...")
    
    # Create stocks table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS stocks (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) UNIQUE NOT NULL,
        name VARCHAR(255) NOT NULL,
        sector VARCHAR(100),
        industry VARCHAR(100),
        country VARCHAR(50),
        exchange VARCHAR(50),
        currency VARCHAR(10),
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW()
    )
    """)
    
    # Create historical_data table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS historical_data (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) REFERENCES stocks(ticker),
        date DATE NOT NULL,
        open FLOAT,
        high FLOAT,
        low FLOAT,
        close FLOAT,
        volume BIGINT,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(ticker, date)
    )
    """)
    
    # Create fundamental_data table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS fundamental_data (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) REFERENCES stocks(ticker),
        pe_ratio FLOAT,
        forward_pe FLOAT,
        pb_ratio FLOAT,
        ps_ratio FLOAT,
        peg_ratio FLOAT,
        dividend_yield FLOAT,
        market_cap BIGINT,
        revenue_growth_yoy FLOAT,
        earnings_growth_yoy FLOAT,
        debt_to_equity FLOAT,
        current_ratio FLOAT,
        quick_ratio FLOAT,
        roa FLOAT,
        roe FLOAT,
        profit_margin FLOAT,
        operating_margin FLOAT,
        created_at TIMESTAMP DEFAULT NOW(),
        updated_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(ticker)
    )
    """)
    
    # Create sentiment_data table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS sentiment_data (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) REFERENCES stocks(ticker),
        date DATE NOT NULL,
        sentiment_score FLOAT,
        news_count INTEGER,
        positive_count INTEGER,
        negative_count INTEGER,
        neutral_count INTEGER,
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(ticker, date)
    )
    """)
    
    # Create news_articles table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS news_articles (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) REFERENCES stocks(ticker),
        title TEXT NOT NULL,
        url TEXT,
        source VARCHAR(100),
        published_date TIMESTAMP,
        sentiment_score FLOAT,
        sentiment_label VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(url)
    )
    """)
    
    # Create predictions table
    engine.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id SERIAL PRIMARY KEY,
        ticker VARCHAR(10) REFERENCES stocks(ticker),
        prediction_date DATE NOT NULL,
        target_date DATE NOT NULL,
        predicted_price FLOAT NOT NULL,
        confidence FLOAT,
        recommendation VARCHAR(20),
        created_at TIMESTAMP DEFAULT NOW(),
        UNIQUE(ticker, prediction_date, target_date)
    )
    """)
    
    print("Database tables created successfully!")

def populate_sample_data(engine):
    """Populate database with sample data."""
    print("Populating database with sample data...")
    
    # Sample stock tickers
    tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'TSLA', 'NVDA', 'JPM', 'V', 'JNJ']
    
    # Fetch stock info
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Insert into stocks table
            engine.execute(text("""
            INSERT INTO stocks (ticker, name, sector, industry, country, exchange, currency)
            VALUES (:ticker, :name, :sector, :industry, :country, :exchange, :currency)
            ON CONFLICT (ticker) DO UPDATE SET
                name = EXCLUDED.name,
                sector = EXCLUDED.sector,
                industry = EXCLUDED.industry,
                country = EXCLUDED.country,
                exchange = EXCLUDED.exchange,
                currency = EXCLUDED.currency,
                updated_at = NOW()
            """), {
                'ticker': ticker,
                'name': info.get('shortName', ticker),
                'sector': info.get('sector', None),
                'industry': info.get('industry', None),
                'country': info.get('country', None),
                'exchange': info.get('exchange', None),
                'currency': info.get('currency', 'USD')
            })
            
            # Fetch historical data
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            hist = stock.history(start=start_date, end=end_date)
            
            # Insert historical data
            for date, row in hist.iterrows():
                engine.execute(text("""
                INSERT INTO historical_data (ticker, date, open, high, low, close, volume)
                VALUES (:ticker, :date, :open, :high, :low, :close, :volume)
                ON CONFLICT (ticker, date) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """), {
                    'ticker': ticker,
                    'date': date.date(),
                    'open': float(row['Open']),
                    'high': float(row['High']),
                    'low': float(row['Low']),
                    'close': float(row['Close']),
                    'volume': int(row['Volume'])
                })
            
            # Insert fundamental data
            engine.execute(text("""
            INSERT INTO fundamental_data (
                ticker, pe_ratio, forward_pe, pb_ratio, ps_ratio, peg_ratio, 
                dividend_yield, market_cap, revenue_growth_yoy, earnings_growth_yoy,
                debt_to_equity, current_ratio, quick_ratio, roa, roe, profit_margin, operating_margin
            )
            VALUES (
                :ticker, :pe_ratio, :forward_pe, :pb_ratio, :ps_ratio, :peg_ratio, 
                :dividend_yield, :market_cap, :revenue_growth_yoy, :earnings_growth_yoy,
                :debt_to_equity, :current_ratio, :quick_ratio, :roa, :roe, :profit_margin, :operating_margin
            )
            ON CONFLICT (ticker) DO UPDATE SET
                pe_ratio = EXCLUDED.pe_ratio,
                forward_pe = EXCLUDED.forward_pe,
                pb_ratio = EXCLUDED.pb_ratio,
                ps_ratio = EXCLUDED.ps_ratio,
                peg_ratio = EXCLUDED.peg_ratio,
                dividend_yield = EXCLUDED.dividend_yield,
                market_cap = EXCLUDED.market_cap,
                revenue_growth_yoy = EXCLUDED.revenue_growth_yoy,
                earnings_growth_yoy = EXCLUDED.earnings_growth_yoy,
                debt_to_equity = EXCLUDED.debt_to_equity,
                current_ratio = EXCLUDED.current_ratio,
                quick_ratio = EXCLUDED.quick_ratio,
                roa = EXCLUDED.roa,
                roe = EXCLUDED.roe,
                profit_margin = EXCLUDED.profit_margin,
                operating_margin = EXCLUDED.operating_margin,
                updated_at = NOW()
            """), {
                'ticker': ticker,
                'pe_ratio': info.get('trailingPE', None),
                'forward_pe': info.get('forwardPE', None),
                'pb_ratio': info.get('priceToBook', None),
                'ps_ratio': info.get('priceToSalesTrailing12Months', None),
                'peg_ratio': info.get('pegRatio', None),
                'dividend_yield': info.get('dividendYield', None),
                'market_cap': info.get('marketCap', None),
                'revenue_growth_yoy': info.get('revenueGrowth', None),
                'earnings_growth_yoy': info.get('earningsGrowth', None),
                'debt_to_equity': info.get('debtToEquity', None),
                'current_ratio': info.get('currentRatio', None),
                'quick_ratio': info.get('quickRatio', None),
                'roa': info.get('returnOnAssets', None),
                'roe': info.get('returnOnEquity', None),
                'profit_margin': info.get('profitMargins', None),
                'operating_margin': info.get('operatingMargins', None)
            })
            
            print(f"Added data for {ticker}")
            
        except Exception as e:
            print(f"Error processing {ticker}: {e}")
    
    print("Sample data populated successfully!")

def main():
    """Main function to initialize the database."""
    try:
        # Create database engine
        engine = create_engine(DATABASE_URL)
        
        # Create tables
        create_tables(engine)
        
        # Populate with sample data
        populate_sample_data(engine)
        
        print("Database initialization completed successfully!")
        
    except Exception as e:
        print(f"Error initializing database: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 