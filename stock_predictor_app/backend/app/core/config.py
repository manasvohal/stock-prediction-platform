from pydantic_settings import BaseSettings
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    """Application settings."""
    
    # API settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Stock Predictor API"
    
    # Database settings
    DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///./stock_predictor.db")
    SQLALCHEMY_DATABASE_URI: str = DATABASE_URL
    
    # External API keys
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")
    ALPHA_VANTAGE_API_KEY: str = os.getenv("ALPHA_VANTAGE_API_KEY", "")
    
    # Model settings
    MODEL_PATH: str = os.getenv("MODEL_PATH", "../data/models/predictor_model.pkl")
    MODEL_DIR: str = os.getenv("MODEL_DIR", "../data/models")
    SENTIMENT_MODEL: str = os.getenv("SENTIMENT_MODEL", "ProsusAI/finbert")

    class Config:
        case_sensitive = True

# Create settings instance
settings = Settings() 