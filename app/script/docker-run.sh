#!/bin/bash

# Build and run the Medical RAG Chatbot in Docker
# This solves the npm/Node.js installation issues and provides GPU access

echo "🐳 Building Medical RAG Chatbot Docker environment..."

# Change to the parent directory to include the full project in build context
cd "$(dirname "$0")/.."

# Build the Docker image with the app/ subdirectory
docker build -f app/Dockerfile -t medical-rag-chatbot .

if [ $? -ne 0 ]; then
    echo "❌ Docker build failed"
    exit 1
fi

echo "✅ Docker image built successfully"

echo ""
echo "🚀 Starting container with GPU support..."

# Check if .env.docker exists, otherwise use .env
ENV_FILE="/home/students/Leishmania/.env.docker"
if [ ! -f "$ENV_FILE" ]; then
    ENV_FILE="/home/students/Leishmania/.env"
fi

# Run the container with GPU support and proper RAG system integration
docker run -it --rm \
    --gpus all \
    --user $(id -u):$(id -g) \
    --dns=8.8.8.8 \
    --dns=8.8.4.4 \
    --dns=1.1.1.1 \
    -p 8000:8000 \
    -p 3000:3000 \
    -v "/home/students/Leishmania:/app" \
    -v "/data4t/hf:/data4t/hf:ro" \
    -v "$ENV_FILE:/app/.env" \
    -e RAG_ROOT=/app \
    -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
    -e RAG_PDF_DIRS="/app/data/standard" \
    -e QDRANT_COLLECTION="leish_cases_pages" \
    -e HF_HUB_OFFLINE=1 \
    -e TRANSFORMERS_OFFLINE=1 \
    -e PYTHONPATH='/app:${PYTHONPATH}' \
    -w /app/app \
    --name medical-rag-chatbot \
    medical-rag-chatbot

echo ""
echo "📝 Inside the container, you can run:"
echo "1. Test the system: ./test.sh"
echo "2. Start backend: cd server && python3 main.py"
echo "3. Start frontend: cd web && npm run dev"
echo "4. Test RAG directly: python3 -m rag.retriever.run_batch_answers_medgemma4b_test_medcpt --help"