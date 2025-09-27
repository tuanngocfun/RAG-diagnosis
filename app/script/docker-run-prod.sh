#!/bin/bash

# Production Docker Run Script for Medical RAG Chatbot
# Optimized for stability and performance

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "🚀 Starting Medical RAG Chatbot (Production Mode)..."

# Docker configuration
CONTAINER_NAME="medical-rag-prod"
IMAGE_NAME="medical-rag-chatbot"

# Environment variables
ENV_FILE="$SCRIPT_DIR/.env.docker"

# Check if environment file exists
if [ ! -f "$ENV_FILE" ]; then
    echo "❌ Environment file not found: $ENV_FILE"
    exit 1
fi

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🔄 Stopping container..."
    docker stop "$CONTAINER_NAME" 2>/dev/null || true
    docker rm "$CONTAINER_NAME" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup SIGINT SIGTERM EXIT

# Stop existing container if running
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

echo "🐳 Starting Docker container..."

# Run Docker container with optimized settings
docker run -it --rm \
    --gpus all \
    --name "$CONTAINER_NAME" \
    --env-file "$ENV_FILE" \
    -e ENVIRONMENT=production \
    -e PYTHONUNBUFFERED=1 \
    -e TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0" \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e NEXT_TELEMETRY_DISABLED=1 \
    -p 127.0.0.1:8000:8000 \
    -p 127.0.0.1:3000:3000 \
    -v "$PROJECT_ROOT:/app" \
    -v "/data4t/hf:/data4t/hf:ro" \
    -e RAG_ROOT=/app \
    -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
    -e PYTHONPATH="/app" \
    --shm-size=8g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -w /app/app \
    "$IMAGE_NAME" \
    bash -lc "./start.sh"