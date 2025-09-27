#!/bin/bash

# Quick DNS and network diagnostic script for Docker container
echo "🔍 Docker Network Diagnostics"

echo ""
echo "1. Testing DNS resolution from host..."
nslookup a66099be-6d6a-43c3-b001-6f6b045ca467.europe-west3-0.gcp.cloud.qdrant.io

echo ""
echo "2. Testing HTTPS connectivity from host..."
curl -I https://a66099be-6d6a-43c3-b001-6f6b045ca467.europe-west3-0.gcp.cloud.qdrant.io 2>/dev/null | head -5

echo ""
echo "3. Starting container with enhanced DNS settings..."

# Stop any existing container
docker stop medical-rag-chatbot 2>/dev/null || true

# Run with network host mode (for testing)
docker run -it --rm \
    --gpus all \
    --network host \
    --name medical-rag-chatbot-test \
    --env-file .env \
    -v "/home/students/Leishmania:/app" \
    -v "/data4t/hf:/data4t/hf:ro" \
    -e RAG_ROOT=/app \
    -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
    -e PYTHONPATH="/app" \
    -w /app/app \
    medical-rag-chatbot \
    bash -c "
        echo '🔍 Testing DNS inside container...'
        nslookup a66099be-6d6a-43c3-b001-6f6b045ca467.europe-west3-0.gcp.cloud.qdrant.io
        echo ''
        echo '🔍 Testing connectivity inside container...'
        curl -I https://a66099be-6d6a-43c3-b001-6f6b045ca467.europe-west3-0.gcp.cloud.qdrant.io 2>/dev/null | head -5
        echo ''
        echo '🚀 Starting services...'
        ./start.sh
    "