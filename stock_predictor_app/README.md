# Stock Predictor Application

A full-stack application for stock prediction and analysis using fundamental analysis, technical analysis, and sentiment analysis.

## Features

- **Stock Data**: Search and view historical stock data with interactive charts
- **Fundamental Analysis**: View key financial metrics, ratios, and company health indicators
- **Technical Analysis**: Interactive charts with technical indicators (Moving Averages, RSI, MACD, Support/Resistance)
- **Sentiment Analysis**: News sentiment analysis using FinBERT
- **Prediction Model**: ML-based stock prediction combining all analysis factors
- **Investment Recommendations**: Get actionable investment recommendations

## Tech Stack

### Backend
- Python with FastAPI
- PostgreSQL database
- yfinance for stock data
- Pandas, NumPy for data processing
- Transformers (FinBERT) for sentiment analysis
- Scikit-learn for ML models

### Frontend
- React with TypeScript
- Material UI components
- Lightweight Charts for interactive stock charts
- Recharts for data visualization
- React Router for navigation

## Setup Instructions

### Prerequisites
- Docker and Docker Compose
- Git

### Installation

1. Clone the repository:
```
git clone <repository-url>
cd stock_predictor_app
```

2. Start the application with Docker Compose:
```
docker-compose up -d
```

3. Access the application:
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

## API Endpoints

- `/api/v1/stocks` - Search and retrieve stock data
- `/api/v1/stocks/{ticker}/fundamentals` - Get fundamental analysis
- `/api/v1/stocks/{ticker}/technical` - Get technical analysis
- `/api/v1/stocks/{ticker}/sentiment` - Get sentiment analysis
- `/api/v1/stocks/{ticker}/predict` - Get stock predictions and recommendations

## Development

### Backend Development
```
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

### Frontend Development
```
cd frontend
npm install
npm start
```

## License

MIT 