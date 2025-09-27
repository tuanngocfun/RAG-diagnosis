#!/bin/bash

# Test script for Medical RAG Chatbot fixes
# Tests connection stability and timeout handling

echo "🧪 Testing Medical RAG Chatbot fixes..."

# Function to test endpoint
test_endpoint() {
    local url=$1
    local name=$2
    local timeout=${3:-10}
    
    echo -n "Testing $name... "
    if curl -s -m $timeout "$url" > /dev/null 2>&1; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Function to test API with timeout
test_api_with_timeout() {
    local question=$1
    local timeout=${2:-30}
    
    echo -n "Testing API with question '$question' (timeout: ${timeout}s)... "
    
    local response=$(curl -s -m $timeout \
        -X POST \
        -H "Content-Type: application/json" \
        -d "{\"question\": \"$question\"}" \
        http://localhost:8000/api/ask 2>/dev/null)
    
    if [ $? -eq 0 ] && echo "$response" | grep -q "answer"; then
        echo "✅ OK"
        return 0
    else
        echo "❌ FAILED"
        return 1
    fi
}

# Wait for services to be available
echo "⏳ Waiting for services to start..."
sleep 5

# Test health endpoints
test_endpoint "http://localhost:8000/healthz" "Backend health" 5
test_endpoint "http://localhost:3000" "Frontend" 5

# Test API functionality
test_api_with_timeout "What is leishmaniasis?" 30

echo ""
echo "🏁 Test completed!"
echo ""
echo "💡 If tests pass, the timeout and connection issues should be resolved."
echo "💡 If tests fail, check:"
echo "   - Backend logs for model loading progress"
echo "   - Network connectivity"  
echo "   - Docker container status"