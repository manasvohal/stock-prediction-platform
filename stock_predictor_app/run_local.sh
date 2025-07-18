#!/bin/bash

# Set environment variables
export DATABASE_URL="sqlite:///./stock_predictor.db"
export SECRET_KEY="your_secret_key_here"
export ENVIRONMENT="development"
export API_V1_STR="/api/v1"
export PROJECT_NAME="Stock Predictor API"
export ALPHA_VANTAGE_API_KEY="MA23H7ILWPCGWJVW"
export NEWS_API_KEY="1c4aca4cbbbd432f8fe51396554604d2"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "Python 3 is not installed. Please install Python 3 first."
    exit 1
fi

# Install Python dependencies if not already installed
echo "Installing Python dependencies..."
cd backend
pip install -r requirements.txt

# Start backend in background
echo "Starting backend server..."
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    echo "The backend is running at http://localhost:8000"
    echo "Press Ctrl+C to stop the backend server"
    wait $BACKEND_PID
    exit 1
fi

# Install Node.js dependencies if not already installed
echo "Installing Node.js dependencies..."
cd frontend
npm install

# Start frontend
echo "Starting frontend server..."
npm start &
FRONTEND_PID=$!
cd ..

# Display access information
echo "
=================================================
Stock Predictor Application is now running!
=================================================

Access the application:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs

Press Ctrl+C to stop the application
=================================================
"

# Wait for user to press Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT TERM
wait 