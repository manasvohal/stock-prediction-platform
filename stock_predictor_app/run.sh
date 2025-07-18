#!/bin/bash

# Set environment variables
export DATABASE_URL="postgresql://postgres:postgres@localhost:5432/stockpredictor"
export SECRET_KEY="your_secret_key_here"
export ENVIRONMENT="development"
export API_V1_STR="/api/v1"
export PROJECT_NAME="Stock Predictor API"
export ALPHA_VANTAGE_API_KEY="MA23H7ILWPCGWJVW"
export NEWS_API_KEY="1c4aca4cbbbd432f8fe51396554604d2"

# Check if PostgreSQL is running
echo "Checking PostgreSQL status..."
pg_isready -h localhost -p 5432 -U postgres > /dev/null 2>&1
if [ $? -ne 0 ]; then
    echo "PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Create database if it doesn't exist
echo "Creating database if it doesn't exist..."
psql -h localhost -p 5432 -U postgres -c "CREATE DATABASE stockpredictor;" > /dev/null 2>&1

# Start backend in background
echo "Starting backend server..."
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!
cd ..

# Wait for backend to start
echo "Waiting for backend to start..."
sleep 5

# Start frontend
echo "Starting frontend server..."
cd frontend
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
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT TERM
wait 