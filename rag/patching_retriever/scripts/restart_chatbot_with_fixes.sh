#!/bin/bash
# Quick script to restart the Docker container with the latest fixes

echo "🔄 Stopping current Docker container..."
docker stop medical-rag 2>/dev/null || true
docker rm medical-rag 2>/dev/null || true

echo "🚀 Starting Docker container with latest fixes..."
cd /home/students/Leishmania/app

# Run the container with the updated code
docker run -it --rm --gpus all --name medical-rag \
  --env-file .env.docker \
  -p 127.0.0.1:8000:8000 -p 127.0.0.1:3000:3000 \
  -v "/home/students/Leishmania:/app" \
  -v "/data4t/hf:/data4t/hf:ro" \
  -e RAG_ROOT=/app \
  -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e PYTHONPATH="/app" \
  -w /app/app \
  medical-rag-chatbot bash -lc "./script/start.sh" 

echo "🔄 Docker container starting in background..."
echo "⏳ Wait 60-90 seconds for model loading, then test at http://localhost:3000"
echo "🧪 Or test API directly: curl -X POST http://localhost:8000/api/ask -H 'Content-Type: application/json' -d '{\"question\":\"What is leishmaniasis?\"}'"