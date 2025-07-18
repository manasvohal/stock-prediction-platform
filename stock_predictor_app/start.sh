#!/bin/bash

# Start the backend server
echo "Starting backend server..."
cd backend
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start the frontend server
echo "Starting frontend server..."
cd ../frontend
npm run build
npm run dev -- --port 3002 &
FRONTEND_PID=$!

# Function to handle script termination
function cleanup() {
  echo "Shutting down servers..."
  kill $BACKEND_PID
  kill $FRONTEND_PID
  exit 0
}

# Register the cleanup function for when the script is terminated
trap cleanup SIGINT SIGTERM

# Keep the script running
echo "Servers are running. Press Ctrl+C to stop."
wait 