# Medical RAG Chatbot

A comprehensive chatbot application for medical case analysis using multimodal Retrieval-Augmented Generation (RAG). The system combines **ColQwen2** (retriever) + **MedGemma-4B-IT** (generator) to provide AI-powered analysis of Leishmania disease cases.

## 🏗️ Architecture

- **Backend**: FastAPI server with RAG pipeline integration
- **Frontend**: Next.js 14 + TypeScript + Tailwind CSS
- **CLI Tool**: Command-line interface for batch processing
- **RAG Pipeline**: ColQwen2 → FAISS/Qdrant → MedCPT cross-encoder → MedGemma-4B-IT

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Node.js 18+
- Access to Qdrant vector database
- HuggingFace token
- Google API key (optional, for fallback generation)

### 1. Environment Setup

Copy the main project's `.env` file or create a new one:

```bash
# Copy existing environment from parent directory
cp ../.env ./
```

The `.env` should contain:
```bash
QDRANT_URL=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token
GOOGLE_API_KEY=your_google_api_key  # Optional
```

### 2. Backend Setup

```bash
cd server/

# Install dependencies (most are already in main environment)
pip install -r requirements.txt

# Copy environment template
cp .env.example .env
# Edit .env with your actual values

# Start the backend server
python main.py
# Or with uvicorn:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The backend will be available at `http://localhost:8000`

### 3. Frontend Setup

```bash
cd web/

# Install dependencies
npm install

# Copy environment template
cp .env.sample .env.local
# Edit .env.local if needed (backend URL, etc.)

# Start the development server
npm run dev
```

The frontend will be available at `http://localhost:3000`

### 4. Test the System

#### Web Interface
1. Open `http://localhost:3000`
2. Ask a question like: "What are the diagnostic features on biopsy?"
3. Adjust settings in the sidebar for different case types, filters, etc.

#### CLI Tool
```bash
cd tools/

# Ask a single question
python ask.py --question "What are the diagnostic features on biopsy?" --case-type cutaneous

# Process a batch file
echo '{"question":"What is the likely diagnosis?","case_type":"cutaneous"}' > test.jsonl
python ask.py --input test.jsonl --output results.jsonl
```

#### API Testing
```bash
# Health check
curl http://localhost:8000/healthz

# Ask a question
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the diagnostic features on biopsy?",
    "case_type": "cutaneous",
    "top_k": 8,
    "images_per_answer": 2
  }'
```

## 📁 Project Structure

```
app/
├── server/                 # FastAPI backend
│   ├── main.py            # FastAPI app and server setup
│   ├── rag_api.py         # RAG service layer
│   ├── requirements.txt   # Python dependencies
│   ├── .env.example       # Environment template
│   └── README.md          # Backend documentation
├── web/                   # Next.js frontend
│   ├── src/
│   │   ├── app/           # Next.js app router
│   │   ├── components/    # React components
│   │   ├── lib/           # Utilities and API client
│   │   ├── types/         # TypeScript types
│   │   └── styles/        # CSS and Tailwind styles
│   ├── package.json       # Node.js dependencies
│   ├── tailwind.config.js # Tailwind configuration
│   ├── next.config.js     # Next.js configuration
│   └── .env.sample        # Environment template
├── tools/                 # CLI utilities
│   └── ask.py            # Command-line interface
└── README.md             # This file
```

## 🔧 Configuration

### Backend Configuration

Environment variables for the FastAPI server:

```bash
# Required
QDRANT_URL=https://your-qdrant-instance.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key
HF_TOKEN=your_huggingface_token

# Optional
GOOGLE_API_KEY=your_google_api_key
RAG_EXTRACT_ROOT=/path/to/extracted/documents
SERVER_HOST=0.0.0.0
SERVER_PORT=8000
LOG_LEVEL=info
```

### Frontend Configuration

Environment variables for the Next.js app:

```bash
# Backend API URL
BACKEND_URL=http://localhost:8000

# App metadata
NEXT_PUBLIC_APP_NAME="Medical RAG Chatbot"
NEXT_PUBLIC_APP_VERSION="1.0.0"
```

## 🎯 Features

### Web Interface
- **Chat Interface**: Intuitive conversation-style interaction
- **Settings Panel**: Configurable retrieval parameters
  - Top-K results (1-15)
  - Case type filtering (cutaneous, mucocutaneous, visceral, unknown)
  - Keyword filtering
  - Micrograph preferences
  - Images per answer (1-5)
- **Evidence Display**: Expandable evidence panel showing:
  - Retrieved document excerpts with scores
  - Source citations with document references
  - Medical image thumbnails
- **Responsive Design**: Works on desktop and mobile devices

### API Endpoints

#### `POST /api/ask`
Main RAG endpoint for processing questions.

**Request:**
```json
{
  "question": "What are the diagnostic features on biopsy?",
  "top_k": 8,
  "case_type": "cutaneous",
  "keyword": null,
  "any_keywords": null,
  "micrograph_only": false,
  "micrograph_strict": false,
  "images_per_answer": 2
}
```

**Response:**
```json
{
  "answer": "Based on the evidence provided, the diagnostic features...",
  "hits": [
    {
      "rank": 1,
      "score": 0.856,
      "doc_id": "case_001",
      "page_index": 3,
      "image_path": "case_001/pages/page_0004.png",
      "page_kind": "figure_or_micrograph",
      "micrograph_like": true,
      "keywords": ["amastigotes", "microscopy", "diagnosis"],
      "text_excerpt": "Microscopic examination revealed..."
    }
  ],
  "evidence": [
    ["microscopic examination revealed intracellular amastigotes", "case_001:p3"]
  ],
  "used_images": ["case_001/pages/page_0004.png"],
  "note": null
}
```

#### `GET /healthz`
Health check endpoint.

#### `GET /files/{file_path}`
Secure file serving for medical images.

### CLI Tool

The `tools/ask.py` script provides command-line access to the RAG system:

```bash
# Single question
python ask.py -q "What are the symptoms?" --case-type cutaneous

# Batch processing
python ask.py -i questions.jsonl -o answers.jsonl

# With specific parameters
python ask.py -q "Show me microscopy" --micrograph-only --top-k 10
```

**JSONL Input Format:**
```json
{"question": "What are the symptoms?", "top_k": 10, "case_type": "cutaneous"}
{"question": "Show me treatment options", "micrograph_only": true}
```

## 🧪 Testing

### Manual Testing

1. **Backend Health Check:**
   ```bash
   curl http://localhost:8000/healthz
   ```

2. **Simple Question:**
   ```bash
   curl -X POST http://localhost:8000/api/ask \
     -H "Content-Type: application/json" \
     -d '{"question":"What is cutaneous leishmaniasis?","top_k":5}'
   ```

3. **Web Interface:**
   - Open `http://localhost:3000`
   - Try various questions and settings
   - Check evidence display and image loading

4. **CLI Tool:**
   ```bash
   cd tools/
   python ask.py --question "What are the diagnostic features?" --case-type cutaneous
   ```

### Example Questions

- "What are the diagnostic features on dermal biopsy?"
- "Show me microscopy findings of amastigotes"
- "What treatments are recommended for cutaneous leishmaniasis?"
- "How do you differentiate between cutaneous and mucocutaneous forms?"
- "What are the key histopathological findings?"

### Expected Outputs

A successful query should return:
- ✅ Relevant answer text with medical information
- ✅ Evidence table with document scores and excerpts
- ✅ Source citations in the format `[1], [2], etc.`
- ✅ Medical image thumbnails (if available)
- ✅ Proper case type filtering and keyword matching

## 🔒 Security Features

- **Path Traversal Protection**: File serving restricted to extraction directory
- **CORS Configuration**: Limited to specific frontend origins
- **Input Validation**: Pydantic models validate all API inputs
- **Medical Disclaimer**: Clear warnings about educational use only
- **Error Handling**: Graceful error responses without sensitive information

## 🚨 Troubleshooting

### Common Issues

1. **Backend won't start:**
   - Check if ports 8000 is available
   - Verify environment variables are set correctly
   - Ensure RAG module can be imported

2. **Frontend build errors:**
   - Run `npm install` to ensure all dependencies are installed
   - Check Node.js version (requires 18+)
   - Verify TypeScript configuration

3. **No search results:**
   - Check Qdrant connection and API key
   - Verify the vector database has been populated
   - Check extraction root path configuration

4. **Images not loading:**
   - Verify `RAG_EXTRACT_ROOT` environment variable
   - Check file permissions on image directories
   - Ensure image paths are correctly mapped

5. **Model loading errors:**
   - Check HuggingFace token validity
   - Verify internet connectivity for model downloads
   - Check available disk space and memory

### Debug Mode

Enable debug logging:
```bash
# Backend
LOG_LEVEL=debug uvicorn main:app --reload

# Check logs for detailed error messages
```

### Performance Tips

- **GPU Usage**: Ensure CUDA is available for model inference
- **Memory Management**: Monitor RAM usage during model loading
- **Batch Size**: Adjust batch sizes based on available memory
- **Caching**: Models are cached after first load

## 🔄 Development

### Adding New Features

1. **Backend**: Add new endpoints in `server/main.py` and logic in `server/rag_api.py`
2. **Frontend**: Add components in `web/src/components/` and update types in `web/src/types/`
3. **CLI**: Extend `tools/ask.py` with new command-line options

### Code Style

- **Backend**: Follow PEP 8 with type hints
- **Frontend**: Use TypeScript with strict mode
- **CSS**: Use Tailwind utility classes with semantic component classes

### Testing

- **Unit Tests**: Add tests for new API endpoints and utility functions
- **Integration Tests**: Test full RAG pipeline with sample questions
- **E2E Tests**: Use Playwright for frontend testing

## 📄 License

This project follows the same license as the parent Leishmania RAG project.

## 🤝 Contributing

1. Follow the existing code style and patterns
2. Add appropriate error handling and logging
3. Update documentation for new features
4. Test thoroughly with various question types and parameters

## 📞 Support

For issues specific to this chatbot app, check:
1. Backend logs for API errors
2. Frontend console for client-side issues
3. CLI output for command-line problems
4. Parent project documentation for RAG pipeline issues