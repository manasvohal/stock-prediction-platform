from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.api.v1 import stocks, fundamentals, technical, sentiment, predict
from app.core.config import settings
from app.core.database import engine, Base

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Create database tables
Base.metadata.create_all(bind=engine)

# Create FastAPI app
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="API for stock prediction and analysis",
    version="1.0.0",
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001", "http://localhost:3002", "http://localhost:3003"],  # Add frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    stocks.router,
    prefix=f"{settings.API_V1_STR}/stocks",
    tags=["stocks"],
)
app.include_router(
    fundamentals.router,
    prefix=f"{settings.API_V1_STR}/fundamentals",
    tags=["fundamentals"],
)
app.include_router(
    technical.router,
    prefix=f"{settings.API_V1_STR}/technical",
    tags=["technical"],
)
app.include_router(
    sentiment.router,
    prefix=f"{settings.API_V1_STR}/sentiment",
    tags=["sentiment"],
)
app.include_router(
    predict.router,
    prefix=f"{settings.API_V1_STR}/predict",
    tags=["predict"],
)

@app.get("/")
async def root():
    """
    Root endpoint
    """
    return {
        "message": "Welcome to the Stock Predictor API",
        "docs": "/docs",
        "version": "1.0.0",
    }

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "ok"} 