#!/bin/bash

# Medical RAG Chatbot - Complete Startup Script
# This script starts both the backend and frontend services in Docker environment

echo "🚀 Starting Medical RAG Chatbot..."

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "� Shutting down services..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    wait
    echo "✅ Services stopped"
    exit 0
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM

# Check environment
echo "🔧 Checking environment..."
python3 -c "
import os
if not os.getenv('QDRANT_URL'):
    print('❌ QDRANT_URL not set')
    exit(1)
print('✅ Environment OK')
"

if [ $? -ne 0 ]; then
    echo "❌ Environment check failed"
    exit 1
fi

# Start backend with optimized settings
echo "📡 Starting backend server (port 8000)..."
cd /app/app/server

# Set environment variables for better performance
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0"
export CUDA_VISIBLE_DEVICES=0

# Start backend server
python3 -u main.py &
BACKEND_PID=$!

# Wait for backend to start (model loading takes time)
echo "⏳ Waiting for backend to initialize (this may take 60-120 seconds for model loading)..."
echo "💡 First-time model loading from cache may take longer..."

for i in {1..90}; do
    sleep 2
    if curl -s -m 5 http://localhost:8000/healthz > /dev/null 2>&1; then
        echo "✅ Backend started successfully after $((i*2)) seconds"
        break
    elif [ $i -eq 90 ]; then
        echo "❌ Backend failed to start after 3 minutes - checking logs..."
        echo "🔍 Backend process status:"
        ps aux | grep "python3.*main.py" | grep -v grep || echo "No backend process found"
        echo "🔍 Last few log lines:"
        tail -20 /tmp/backend.log 2>/dev/null || echo "No log file found"
        kill $BACKEND_PID 2>/dev/null
        exit 1
    fi
    if [ $((i % 15)) -eq 0 ]; then
        echo "⏳ Still waiting... ($((i*2)) seconds elapsed)"
        echo "💭 Models are loading in background, please be patient..."
    fi
done

# Start frontend
echo "🌐 Starting frontend (port 3000)..."
cd /app/app/web

# Ensure dependencies are installed
if [ ! -d "node_modules" ] || [ ! -f "node_modules/.bin/next" ]; then
    echo "📦 Installing/updating frontend dependencies..."
    npm install
fi

# Start frontend using npx to ensure next is found
echo "🚀 Launching Next.js development server..."
HOST=0.0.0.0 npx next dev &
FRONTEND_PID=$!

# Wait a moment for frontend to start
sleep 5

# Display status
echo ""
echo "✅ Services started successfully!"
echo ""
echo "🌐 Access points:"
echo "  Frontend UI: http://localhost:3000 (main interface)"
echo "  Backend API: http://localhost:8000 (for API calls)"
echo "  API docs: http://localhost:8000/docs"
echo ""
echo "💡 Important: Open http://localhost:3000 for the chatbot interface!"
echo "💡 Test API directly:"
echo "  curl -X POST http://localhost:8000/api/ask -H 'Content-Type: application/json' -d '{\"question\":\"What is leishmaniasis?\"}'"
echo ""
echo "Press Ctrl+C to stop all services..."

# Wait for processes
wait