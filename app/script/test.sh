#!/bin/bash

# Medical RAG Chatbot - Real Integration Test Runner
# This script runs comprehensive tests to validate the chatbot functionality

echo "� Medical RAG Chatbot - Running Real Tests..."
echo ""
echo "This will test:"
echo "  ✓ Environment setup and database connectivity"
echo "  ✓ RAG pipeline (retrieval + generation)"
echo "  ✓ FastAPI backend server and API endpoints"
echo "  ✓ CLI tool with actual question processing"
echo ""
echo "⏳ Starting comprehensive tests (this may take a few minutes)..."
echo ""

# Run the real Python test script
python3 real_test.py

# Store the exit code
test_result=$?

echo ""
if [ $test_result -eq 0 ]; then
    echo "🎉 All tests passed! Your chatbot is ready to use."
    echo ""
    echo "📝 Quick start commands:"
    echo "1. All services: ./start.sh (recommended)"
    echo "2. Backend only: cd server && python3 main.py"
    echo "3. Frontend only: cd web && npm install && npm run dev"
    echo "4. CLI tool: cd tools && python3 ask.py --question 'What is leishmaniasis?'"
    echo ""
    echo "🌐 Access points:"
    echo "- 🎯 Main UI: http://localhost:3000 (USE THIS for chatbot)"
    echo "- 🔧 Backend API: http://localhost:8000 (for API calls)"
    echo "- 📚 API docs: http://localhost:8000/docs"
    echo ""
    echo "⚠️  Important: Use port 3000 for the web interface, not 8000!"
else
    echo "❌ Some tests failed. Check the output above for details."
    echo ""
    echo "🔧 Common fixes:"
    echo "- Ensure environment variables are set in .env"
    echo "- Check Qdrant database connection"
    echo "- Verify all Python dependencies are installed"
fi

exit $test_result