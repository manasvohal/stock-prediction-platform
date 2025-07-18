#!/bin/bash

# Get the absolute path to the project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Starting Stock Predictor Application..."
echo "Project root: $PROJECT_ROOT"

# Kill any processes running on ports 8000 and 3002
echo "Cleaning up any existing processes..."
lsof -ti:8000 | xargs kill -9 2>/dev/null || true
lsof -ti:3002 | xargs kill -9 2>/dev/null || true

# Start backend server
echo "Starting backend server on port 8000..."
cd "$PROJECT_ROOT/stock_predictor_app/backend" && python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait a moment for backend to initialize
sleep 2

# Start frontend server
echo "Starting frontend server on port 3002..."
cd "$PROJECT_ROOT/stock_predictor_app/frontend" && npm start -- --port 3002 &
FRONTEND_PID=$!

echo "Application started!"
echo "Backend running at http://localhost:8000"
echo "Frontend running at http://localhost:3002"
echo ""
echo "Press Ctrl+C to stop all servers"

# Wait for user to press Ctrl+C
trap "echo 'Stopping servers...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null || true; exit" INT
wait 