# Stock Prediction Platform

A full-stack application for stock prediction and analysis using fundamental analysis, technical analysis, sentiment analysis, and machine learning.

## Features

- **Stock Data**: Search and view historical stock data with interactive candlestick charts
- **Technical Analysis**: Interactive charts with technical indicators (Moving Averages, RSI, MACD, Support/Resistance)
- **Sentiment Analysis**: News sentiment analysis using FinBERT
- **Machine Learning Prediction**: LSTM neural network for time series forecasting
- **Investment Recommendations**: Get actionable investment recommendations

## Tech Stack

### Backend
- Python with FastAPI
- PostgreSQL database
- yfinance for stock data
- Pandas, NumPy for data processing
- PyTorch for LSTM neural networks
- Transformers (FinBERT) for sentiment analysis

### Frontend
- HTML/CSS/JavaScript
- ApexCharts for interactive stock charts
- Responsive design

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```
git clone https://github.com/manasvohal/stock-prediction-platform.git
cd stock-prediction-platform
```

2. Start the application with Docker Compose:
```
docker-compose up -d
```
