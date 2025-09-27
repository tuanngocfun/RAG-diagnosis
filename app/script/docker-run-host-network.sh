#!/bin/bash

# Alternative Docker run script using host network mode
# This bypasses Docker networking issues but requires different port handling

echo "🐳 Starting Medical RAG Chatbot with host network mode..."

cd "$(dirname "$0")/.."

# Build if needed
if [ ! "$(docker images -q medical-rag-chatbot 2>/dev/null)" ]; then
    echo "📦 Building Docker image..."
    docker build -f app/Dockerfile -t medical-rag-chatbot .
fi

# Check if .env.docker exists, otherwise use .env
ENV_FILE="/home/students/Leishmania/.env.docker"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="/home/students/Leishmania/.env"
fi

# Stop any existing container
docker stop medical-rag-chatbot 2>/dev/null || true

echo "🚀 Starting with host network mode (fixes DNS issues)..."

# Run with host network mode - this should resolve DNS issues
docker run -it --rm \
    --gpus all \
    --network host \
    --name medical-rag-chatbot \
    --env-file "$ENV_FILE" \
    -v "/home/students/Leishmania:/app" \
    -v "/data4t/hf:/data4t/hf:ro" \
    -e RAG_ROOT=/app \
    -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
    -e RAG_PDF_DIRS="/app/data/standard" \
    -e QDRANT_COLLECTION="leish_cases_pages" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONPATH='/app:${PYTHONPATH}' \
    -w /app/app \
    medical-rag-chatbot \
    bash -lc "./start.sh"

echo ""
echo "🌐 With host network mode:"
echo "  Frontend: http://localhost:3000"
echo "  Backend: http://localhost:8000"