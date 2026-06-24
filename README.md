# Medical RAG Diagnosis Platform (rag branch)

## Current Curated Package

This branch adds a curated V12d RAG diagnosis defense and harness package:

- `rag/` contains the current pipeline, configs, CLI, and tests.
- `demo/flutter/` contains the local GPU assistant demo backend/UI source.
- `presentation/v12d/` contains the validated 31-slide defense deck package.
- `experiments/structured_cases_v4_2_2_rtx6000/` contains selected official
  Gemma 4 evidence used by V12d, not the full experiment archive.
- `data/whole_multicare_dataset/` is a lightweight dataset ledger, not raw
  corpus storage.
- `AGENTS.md` and `docs/harness-engineering/` define agent workflow,
  validation gates, and clinical-safety boundaries.

This repository is for research evaluation and thesis demonstration. It is not
a clinical deployment package and does not support diagnosis from image alone.

A multimodal Retrieval-Augmented Generation (RAG) system for medical case analysis (e.g. Leishmania) combining modern vision-language models with structured retrieval.  
Core pipeline (current architecture):  
Retriever (ColQwen2 / ColPali or similar) → Vector Store (FAISS / Qdrant) → (Optional) Cross-Encoder Re-Ranker (MedCPT) → Generator (MedGemma-4B-IT) → Answer + Supporting Images.

This README unifies usage instructions from:
- `app/README.md` (application orchestration: backend + frontend + CLI)
- `rag/prompting/README.md` (Standalone MedGemma prompting module)
- `rag/patching_retriever/README.md` (Fixes and tests for retriever)
- Inline module docstrings (PLACEHOLDER: integrate more when gathered)

> NOTE: Some referenced files/commands are inferred; verify paths in the rag branch.  
> Sections flagged with TODO need confirmation or enrichment from code-level docstrings.

---

## Table of Contents

1. Features  
2. System Architecture  
3. Components Overview  
4. Environment & Configuration  
5. Installation & Setup  
6. Running the System  
   - Backend API
   - Frontend Web App
   - CLI Tools
   - Standalone MedGemma (no RAG)
   - Batch Processing
   - Docker & GPU Execution
7. RAG Pipeline Details  
8. Data Preparation & Indexing (TODO)  
9. Retriever Patching & Testing  
10. API Endpoints  
11. Example Workflows  
12. Directory Structure (Observed / Expected)  
13. Troubleshooting  
14. Performance Tips  
15. Development & Testing  
16. Roadmap / TODO  
17. License (TODO)  

---

## 1. Features

- Multimodal medical question answering (images + text).
- Retrieval-Augmented Generation with optional re-ranking.
- Standalone generation mode (skip retrieval) using MedGemma-4B-IT.
- Batch answering for large manifests (JSONL / NDJSON).
- Vector store abstraction (Qdrant, FAISS).
- Modular retriever patching layer for rapid iterations.
- Web UI (Next.js 14 + Tailwind).
- FastAPI backend with clean API.
- CLI automation for experiments & evaluation.
- Docker- and GPU-ready.

---

## 2. System Architecture

```
+------------------+         +----------------------+
|   User (WebUI)   | <--->   |     FastAPI Backend  |
+------------------+         +----------+-----------+
                                      |
                                      v
                           +----------------------+
                           |   RAG Orchestrator  |
                           |  (rag/...)          |
                           +----+------+---------+
                                |      |
               +----------------+      +----------------+
               v                                    v
       +---------------+                   +--------------------+
       |  Retriever    | --> Candidates -->|  Re-Ranker (opt.)  |
       | (ColQwen2)    |                   | (MedCPT / cross)   |
       +-------+-------+                   +---------+----------+
               |                                   |
               v                                   v
        +-------------+                     +-------------+
        | Vector DB   |                     | Generator   |
        | (Qdrant/FAISS)                    | MedGemma4B  |
        +------+------+                     +------+------+  
               |                                   |
               +---------------+-------------------+
                               v
                         +-----------+
                         |  Answer   |
                         +-----------+
```

Standalone mode (no RAG):
```
Images/Text --> MedGemma-4B-IT --> Answer
```

---

## 3. Components Overview

| Component | Path (indicative) | Purpose |
|-----------|-------------------|---------|
| Backend API | `server/` | FastAPI app exposing /api endpoints |
| Frontend | `web/` | Next.js 14 UI |
| CLI Tools | `tools/` | Batch question answering & utilities |
| Prompting Standalone | `rag/prompting/` | Direct MedGemma inference |
| Retriever Patching | `rag/patching_retriever/` | Fixes & validation scripts |
| Vector Store Layer | `rag/...` (TODO) | Qdrant / FAISS integration |
| Batch Scripts | `rag/prompting/run_batch_medgemma4b_standalone.py` | Manifest-driven processing |

---

## 4. Environment & Configuration

### 4.1 Required Environment Variables

Create `.env` (for development) and `.env.docker` (for Docker) files:

```bash
# Essential variables
HF_TOKEN=hf_your_huggingface_token_here          # Required for model access
GOOGLE_API_KEY=your_google_api_key_here          # Required for Gemini evaluation
QDRANT_URL=https://your-qdrant-instance.com      # Required for RAG retrieval
QDRANT_API_KEY=your_qdrant_api_key               # Required for cloud Qdrant

# Model and cache paths
TRANSFORMERS_CACHE=/data4t/hf/transformers       # Model cache directory
HF_HOME=/data4t/hf                               # HuggingFace home
HF_HUB_OFFLINE=1                                 # Use cached models only
TRANSFORMERS_OFFLINE=1                           # Offline transformers

# Data paths
RAG_EXTRACT_ROOT=/app/kaggle/working2/extract    # Case documents root

# API configuration
MODEL_DEVICE=cuda                                 # GPU device
LOG_LEVEL=info                                   # Logging level
PORT=8000                                        # API server port
HOST=0.0.0.0                                    # API server host

# Frontend configuration (for web/.env.local)
NEXT_PUBLIC_API_URL=http://localhost:8000        # Backend URL for frontend
```

### 4.2 Environment Setup Steps

**1. Copy and configure environment files:**
```bash
# Copy example files if they exist
cp .env.example .env 2>/dev/null || touch .env
cp .env.example .env.docker 2>/dev/null || cp .env .env.docker

# Set up frontend environment
cd web
cp .env.sample .env.local 2>/dev/null || echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
cd ..
```

**2. Set your API tokens:**
```bash
# Get HuggingFace token from https://huggingface.co/settings/tokens
echo "HF_TOKEN=hf_your_token_here" >> .env
echo "HF_TOKEN=hf_your_token_here" >> .env.docker

# Get Google API key from https://console.cloud.google.com/
echo "GOOGLE_API_KEY=your_key_here" >> .env
echo "GOOGLE_API_KEY=your_key_here" >> .env.docker

# Set up Qdrant (if using cloud instance)
echo "QDRANT_URL=https://your-cluster.qdrant.io" >> .env
echo "QDRANT_API_KEY=your_api_key" >> .env
```

**3. Verify environment:**
```bash
# Check required variables are set
echo "Checking environment variables..."
for var in HF_TOKEN GOOGLE_API_KEY; do
  if grep -q "^${var}=" .env.docker; then
    echo "✅ $var is set"
  else
    echo "❌ $var is missing - please add to .env.docker"
  fi
done
```

### 4.3 Environment Variables Reference

**Complete Environment Variables Reference:**

| Variable | Description | Required |
|----------|-------------|----------|
| QDRANT_URL | Qdrant endpoint | Yes (if using Qdrant) |
| QDRANT_API_KEY | Qdrant API key | Yes (cloud Qdrant) |
| HF_TOKEN | Hugging Face token for gated models | Yes |
| GOOGLE_API_KEY | Fallback generation (optional) | No |
| TRANSFORMERS_CACHE | Cache directory for models | Recommended |
| HF_HOME | Base HF storage directory | Optional |
| RAG_EXTRACT_ROOT | Base path for extracted source docs/images | For indexing |
| MODEL_DEVICE | e.g. `cuda` or `cuda:0` | Optional |
| LOG_LEVEL | debug/info/warn | Optional |
| WEB_BACKEND_URL | Used by frontend to reach API | Optional |
| PORT / HOST | API server binding | Optional |

Create service-level `.env` in `server/` and `.env.local` in `web/`.

---

## 5. Installation & Setup

### 5.1 Dependency Files Overview

The project includes several dependency files for different installation scenarios:

| File | Purpose | Usage |
|------|---------|-------|
| `requirements_recovered.txt` | **Complete environment** (416 packages) | Full production setup with all dependencies |
| `requirements.txt` | **Minimal requirements** | Basic image processing (Pillow, pytesseract) |
| `constraints.txt` | **Version constraints** | Ensures compatible httpx versions |
| `server/requirements.txt` | **Backend API dependencies** | FastAPI server requirements |

### 5.2 Python Backend & Core Installation

**Option 1: Complete Environment (Recommended for full functionality):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install complete environment with all dependencies
pip install -r requirements_recovered.txt -c constraints.txt
```

**Option 2: Minimal Setup (Basic functionality only):**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip

# Install minimal requirements
pip install -r requirements.txt -c constraints.txt
# Then install additional packages as needed
pip install transformers torch torchvision torchaudio
pip install qdrant-client faiss-cpu
```

**Option 3: Backend API Only:**
```bash
cd server
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

### 5.3 Frontend Installation

```bash
cd web
npm install
cp .env.sample .env.local
# adjust WEB_BACKEND_URL=http://localhost:8000
```

### 5.4 Docker Installation (Recommended)

For the most reliable setup, use the provided Docker image which includes all dependencies:

```bash
# Pull the pre-built image (if available)
docker pull leish-gem25:latest

# Or build locally (if Dockerfile exists)
docker build -t leish-gem25:latest .

# Test Docker setup
docker run --rm --gpus all leish-gem25:latest python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 5.5 Development Extras

```bash
# Code quality tools
pip install black ruff mypy pytest pytest-cov

# Jupyter for experimentation
pip install jupyter ipykernel

# Additional ML tools
pip install wandb tensorboard matplotlib seaborn
```

### 5.6 Troubleshooting Installation

**Common Issues:**

```bash
# If pip install fails with conflicts
pip install --no-deps -r requirements_recovered.txt

# For CUDA compatibility issues
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# If transformers models fail to load
pip install transformers[torch] accelerate bitsandbytes

# For OCR functionality
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
```

**Verify Installation:**
```bash
# Test core dependencies
python -c "import torch, transformers, PIL, qdrant_client; print('✅ Core dependencies OK')"

# Test GPU access
python -c "import torch; print(f'GPU available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')"

# Test model loading
python -c "from transformers import AutoProcessor; print('✅ Transformers working')"
```

---

## 6. Running the System

### 6.1 Backend API

```bash
cd server
cp .env.example .env   # if template exists
python main.py
# or:
# uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://localhost:8000/healthz
```

### 6.2 Frontend Web App

```bash
cd web
npm run dev
# Visit http://localhost:3000
```

### 6.3 CLI Tools (RAG Mode)

**Single Question RAG Analysis:**
```bash
cd app/tools
python ask.py --question "What are the diagnostic features on biopsy?" --case-type cutaneous --top-k 8
```

**Batch RAG Processing:**
```bash
echo '{"question":"What is the likely diagnosis?","case_type":"cutaneous"}' > batch.jsonl
cd app/tools
python ask.py --input batch.jsonl --output answers.jsonl
```

**Available CLI options:**
- `--question` - Single question to ask
- `--top-k` - Number of retrieved documents/images (default: 8) 
- `--case-type` - Filter by case type (cutaneous, visceral, mucosal, etc.)
- `--keyword` - Additional keyword filtering
- `--any-keywords` - Match any of these keywords
- `--input` - Input JSONL file for batch processing
- `--output` - Output file for results

### 6.4 Full RAG Pipeline (Batch Processing)

**Using existing questions manifest:**
```bash
# Process questions from the manifest with full RAG pipeline
python -m rag.gen.run_batch_answers \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out kaggle/working2/rag_knowledge_base/answers/rag_answers.ndjson \
  --top_k 8 \
  --images_per_answer 3 \
  --resume
```

**Resume interrupted processing:**
```bash
# Continue from where you left off (skips already processed questions)
python -m rag.gen.run_batch_answers \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out kaggle/working2/rag_knowledge_base/answers/rag_answers.ndjson \
  --resume --retry_errors
```

**RAG Pipeline options:**
- `--manifest` - Input JSONL manifest file (required)
- `--out` - Output NDJSON file for answers (required)
- `--top_k` - Number of documents to retrieve (default: 8)
- `--images_per_answer` - Max images to include in answer (default: 3)
- `--resume` - Skip already processed questions
- `--retry_errors` - Retry questions that failed previously
- `--shuffle` - Randomize question order

### 6.5 Standalone MedGemma (No Retrieval)

**Python API Usage:**
```python
from rag.prompting.medgemma4b_standalone import StandaloneMedGemma4B

# Initialize the standalone analyzer
analyzer = StandaloneMedGemma4B()

# Analyze specific images
result = analyzer.analyze_images(
    image_paths=["/path/to/medical_image.png"],
    question="What diagnostic features are visible?"
)
print(result)

# Diagnose a case (finds images automatically)
diagnosis = analyzer.diagnose_case(
    case_id="1-case-Cutaneous Leishmaniasis",
    question="What is the most likely diagnosis?"
)
print(diagnosis)
```

**Batch Standalone Processing:**
```bash
# Process all questions without RAG retrieval - faster but less context
python -m rag.prompting.run_batch_medgemma4b_standalone \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out kaggle/working2/rag_knowledge_base/answers/standalone_answers.ndjson \
  --images_per_answer 3 \
  --resume
```

**Docker execution (recommended for GPU):**
```bash
docker run --rm -it --gpus all \
  --user $(id -u):$(id -g) \
  --env-file .env.docker \
  -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e HF_HOME=/data4t/hf \
  -e TRANSFORMERS_CACHE=/data4t/hf/transformers \
  -v $(pwd):/app \
  -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest \
  python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
    --out /app/kaggle/working2/rag_knowledge_base/answers/standalone_answers.ndjson \
    --images_per_answer 3 --resume
```

### 6.8 Docker + GPU

Example (adapt as needed):

```bash
docker run --rm -it --gpus all \
  --user $(id -u):$(id -g) \
  --env-file /path/to/.env.docker \
  -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e HF_HOME=/data/hf \
  -e TRANSFORMERS_CACHE=/data/hf/transformers \
  -v /host/project:/app \
  -v /data/hf:/data/hf \
  -w /app leish-gem25:latest \
  python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl
```

### 6.6 Evaluation & Testing

**Gemini-Only Evaluation (Recommended):**
```bash
# Evaluate answers using Gemini 2.5 Pro as judge
python -m rag.gen.evaluation_gemini_only \
  --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa \
  --answers_file kaggle/working2/rag_knowledge_base/answers/rag_answers.ndjson \
  --out_dir evaluation_results \
  --strategy missing \
  --gemini_rpm 5
```

**RAG System Evaluation:**
```bash
# Comprehensive RAG evaluation with retrieval metrics
python -m rag.gen.evaluation_qdrant_rag \
  --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa \
  --answers_file kaggle/working2/rag_knowledge_base/answers/rag_answers.ndjson \
  --out_dir rag_evaluation_results \
  --top_k 8
```

**Evaluation Options:**
- `--jsonl_dir` - Directory containing gold standard Q&A files
- `--answers_file` - Generated answers file to evaluate (NDJSON format)
- `--out_dir` - Output directory for evaluation results
- `--strategy` - `all` (re-evaluate everything) or `missing` (use cached successful results)
- `--gemini_rpm` - Gemini API rate limit (requests per minute)
- `--thinking_budget` - Token budget for Gemini thinking (128+ for Pro)
- `--stream_append` - Append to existing evaluation stream
- `--resume` - Continue interrupted evaluation

### 6.7 API Example

```bash
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the diagnostic features on biopsy?",
    "case_type": "cutaneous",
    "top_k": 8,
    "images_per_answer": 2
  }'
```

---

## 7. RAG Pipeline Details

1. Ingestion (TODO: confirm script names)
   - Parse medical case documents (text + associated images).
   - Normalize metadata (case type, lesion location, etc.).

2. Embedding / Indexing
   - Use ColQwen2 (or configured vision-language retriever) to produce embeddings.
   - Store in Qdrant (remote) or FAISS (local).
   - Maintain separate collections per modality if needed (text vs image).

3. Retrieval
   - Query embeddings generated from user question.
   - Top-k candidates (configurable).

4. Re-Ranking (Optional)
   - Cross-encode using MedCPT or a medical-specific cross encoder for improved ordering.

5. Generation
   - MedGemma-4B-IT consumes:
     - Reformulated prompt template (question + selected contexts).
     - Possibly attaches or re-encodes selected images.
   - Produces structured answer (text + maybe citations or image references).

6. (Fallback)
   - If MedGemma fails or times out: optional Google API key path (if implemented).

---

## 8. Data Structure & Locations

### 8.1 Key Data Paths

**Questions Manifest:** 
```
kaggle/working2/rag_knowledge_base/questions_manifest.jsonl
```
Contains 923 questions across all cases with metadata:
- `case_id` - Unique case identifier
- `question_id` - Unique question ID
- `question` - The medical question text
- `seed_image_paths` - Associated image paths (if any)
- `retrieve_mode` - Retrieval strategy ("seeded")

**Extracted Cases:**
```
kaggle/working2/extract/
```
Contains 450+ medical cases, each with:
- `pages/` - Page images (PNG format)
- Case documents and metadata

**Example case structure:**
```
kaggle/working2/extract/1-case-Cutaneous Leishmaniasis/
├── pages/
│   ├── page_0000.png
│   ├── page_0001.png
│   └── ...
├── metadata.json
└── extracted_text.txt
```

### 8.2 Question Manifest Format

```json
{
  "case_id": "1-case-Cutaneous Leishmaniasis Presenting to the Emergency Department",
  "doc_id": "1-case-Cutaneous Leishmaniasis Presenting to the Emergency Department", 
  "question_id": "2a3571aef0-q001",
  "question": "What were the lesion's key clinical features and course before referral?",
  "seed_image_paths": [],
  "retrieve_mode": "seeded"
}
```

### 8.3 Answer Output Format

**RAG Pipeline Output (NDJSON):**
```json
{
  "case_id": "1-case-Cutaneous Leishmaniasis",
  "question_id": "2a3571aef0-q001",
  "question": "What were the lesion's key clinical features?",
  "answer": "The lesion presented as...",
  "used_images": ["path/to/page_0001.png"],
  "retrieval_hits": [{"doc_id": "...", "score": 0.85, "page_index": 1}],
  "processing_time": 2.3
}
```

**Standalone Output (NDJSON):**
```json
{
  "case_id": "1-case-Cutaneous Leishmaniasis",
  "question": "What is the diagnosis?",
  "generated_answer": "Based on the clinical features...",
  "used_images": ["path/to/selected_image.png"],
  "confidence_score": 0.92
}
```

---

## 9. Retriever Patching & Testing

Directory: `rag/patching_retriever/`

- `fixes/`: Python modules applying monkey patches / strategy improvements (e.g. batching, caching, scoring normalization).
- `tests/`: Scripts or pytest suites to validate retriever quality.
- `scripts/`: Shell helpers (restart containers, run regression comparisons).

Example usage (inferred):
```bash
pytest rag/patching_retriever/tests -q
```
or
```bash
python rag/patching_retriever/tests/test_single_case.py --case-id case_001
```

---

## 10. API Endpoints (Baseline)

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/healthz` | Health check |
| POST | `/api/ask` | Submit question (RAG or standalone depending on params) |
| (TODO) | `/api/cases/{id}` | Retrieve case metadata |
| (TODO) | `/api/index/rebuild` | Trigger re-indexing |
| (TODO) | `/api/config` | Return active configuration |

Example request body (`/api/ask`):
```json
{
  "question": "What is the most likely diagnosis?",
  "case_type": "cutaneous",
  "top_k": 6,
  "images_per_answer": 2,
  "mode": "rag"   // or "standalone"
}
```

Example response (illustrative):
```json
{
  "answer": "The findings are consistent with cutaneous leishmaniasis...",
  "sources": [
    {"id": "case_045", "score": 0.89},
    {"id": "case_102", "score": 0.83}
  ],
  "images": ["case_045_img1.png","case_102_img2.png"],
  "latency_ms": 2750
}
```

---

## 11. Complete Workflow Examples

### 11.1 Quick Single Question Analysis
```bash
# Ask a single question using CLI
cd app/tools
python ask.py \
  --question "What are the key diagnostic features on histopathology?" \
  --case-type cutaneous \
  --top-k 5
```

### 11.2 Full RAG Research Pipeline
```bash
# 1. Process all questions with full RAG retrieval
python -m rag.gen.run_batch_answers \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out results/rag_answers_$(date +%Y%m%d).ndjson \
  --top_k 8 --images_per_answer 3 --resume

# 2. Evaluate using Gemini judge
python -m rag.gen.evaluation_gemini_only \
  --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa \
  --answers_file results/rag_answers_$(date +%Y%m%d).ndjson \
  --out_dir evaluation/rag_$(date +%Y%m%d) \
  --strategy missing

# 3. Check results
ls evaluation/rag_*/aggregate_eval_*.json
```

### 11.3 Standalone vs RAG Comparison
```bash
# Generate standalone answers (no retrieval)
python -m rag.prompting.run_batch_medgemma4b_standalone \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out results/standalone_answers_$(date +%Y%m%d).ndjson \
  --images_per_answer 3 --resume

# Generate RAG answers (with retrieval)
python -m rag.gen.run_batch_answers \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out results/rag_answers_$(date +%Y%m%d).ndjson \
  --top_k 8 --images_per_answer 3 --resume

# Evaluate both
for mode in standalone rag; do
  python -m rag.gen.evaluation_gemini_only \
    --answers_file results/${mode}_answers_$(date +%Y%m%d).ndjson \
    --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa \
    --out_dir evaluation/${mode}_$(date +%Y%m%d) \
    --strategy missing
done

# Compare results
diff -u evaluation/standalone_*/aggregate_eval_*.json evaluation/rag_*/aggregate_eval_*.json
```

### 11.4 Production Docker Workflow
```bash
# Set up environment
cp .env.example .env.docker
# Edit .env.docker with your API keys

# Run full pipeline in Docker
docker run --rm -it --gpus all \
  --user $(id -u):$(id -g) \
  --env-file .env.docker \
  -v $(pwd):/app \
  -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest \
  python -m rag.gen.run_batch_answers \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
    --out /app/results/docker_rag_$(date +%Y%m%d).ndjson \
    --top_k 8 --resume
```

### 11.5 Interactive Web Development
```bash
# Terminal 1: Start backend API
cd server
python main.py

# Terminal 2: Start frontend
cd web 
npm run dev

# Terminal 3: Test API
curl -X POST http://localhost:8000/api/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the most likely diagnosis?",
    "case_type": "cutaneous",
    "top_k": 5
  }' | jq .

# Open browser to http://localhost:3000
```

### 11.6 Evaluation Maintenance
```bash
# Compact evaluation cache (remove duplicates)
python -m rag.gen.evaluation_gemini_only \
  --out_dir evaluation/results \
  --compact_only

# Repair stream from cache (fix interrupted evaluations)
python -m rag.gen.evaluation_gemini_only \
  --out_dir evaluation/results \
  --stream_path evaluation/results/stream_latest.ndjson \
  --repair_stream_from_cache

# Prune old cache entries 
python -m rag.gen.evaluation_gemini_only \
  --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa \
  --out_dir evaluation/results \
  --prune_cache
```

---

## 12. Directory Structure (Observed / Inferred)

```
app/                    # (From existing README) umbrella or meta-level?
server/                 # FastAPI backend (main.py, routers, models) (verify)
web/                    # Next.js frontend
tools/                  # CLI scripts (ask.py, indexing utilities) (verify)
rag/
  prompting/
    medgemma4b_standalone.py
    run_batch_medgemma4b_standalone.py
    medgemma4b_standalone_demo.ipynb
    test_standalone.py
    README.md
  patching_retriever/
    fixes/
    tests/
    scripts/
    README.md
  (other RAG pipeline modules: embeddings, re_ranker, vector_store, utils) (TODO)
models/                 # (Potential model wrappers) (TODO)
data/ or kaggle/        # Input cases / manifests (TODO)
```

---

## 12. Command Quick Reference

### 12.1 Essential Commands

| Task | Command | Notes |
|------|---------|-------|
| **Single Question (CLI)** | `python app/tools/ask.py --question "What is the diagnosis?"` | Interactive analysis |
| **Batch RAG Processing** | `python -m rag.gen.run_batch_answers --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl --out results.ndjson --resume` | Full pipeline |
| **Standalone Processing** | `python -m rag.prompting.run_batch_medgemma4b_standalone --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl --out standalone.ndjson --resume` | No retrieval |
| **Gemini Evaluation** | `python -m rag.gen.evaluation_gemini_only --answers_file results.ndjson --jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa --out_dir eval_results` | AI judge |
| **Start Web Backend** | `cd server && python main.py` | API server |
| **Start Web Frontend** | `cd web && npm run dev` | React UI |
| **Docker RAG** | `docker run --gpus all --env-file .env.docker -v $(pwd):/app leish-gem25:latest python -m rag.gen.run_batch_answers --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl --out /app/results.ndjson` | GPU processing |

### 12.2 File Path Reference

| Resource | Path | Description |
|----------|------|-------------|
| **Questions** | `kaggle/working2/rag_knowledge_base/questions_manifest.jsonl` | 923 medical questions |
| **Cases** | `kaggle/working2/extract/` | 200+ case directories |
| **Gold Q&A** | `kaggle/working2/rag_knowledge_base/gold_qa/` | Evaluation ground truth |
| **CLI Tool** | `app/tools/ask.py` | Command-line interface |
| **RAG Pipeline** | `rag/gen/run_batch_answers.py` | Full RAG processing |
| **Standalone** | `rag/prompting/run_batch_medgemma4b_standalone.py` | Direct model inference |
| **Evaluation** | `rag/gen/evaluation_gemini_only.py` | Gemini judge evaluation |
| **Web Backend** | `server/main.py` | FastAPI application |
| **Web Frontend** | `web/` | Next.js application |
| **Dependencies** | `requirements_recovered.txt` | Complete environment (416 packages) |
| **Basic Deps** | `requirements.txt` | Minimal requirements (Pillow, pytesseract) |
| **Constraints** | `constraints.txt` | Version constraints (httpx compatibility) |

### 12.3 Common Parameter Patterns

**Standard RAG Parameters:**
```bash
--manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl  # Input questions
--out results/output_$(date +%Y%m%d).ndjson                             # Output file  
--top_k 8                                                                # Documents to retrieve
--images_per_answer 3                                                    # Max images in response
--resume                                                                 # Skip completed questions
--retry_errors                                                          # Retry failed questions
```

**Standard Evaluation Parameters:**
```bash
--answers_file results/answers.ndjson                                   # Generated answers
--jsonl_dir kaggle/working2/rag_knowledge_base/gold_qa                 # Ground truth
--out_dir evaluation/results_$(date +%Y%m%d)                           # Evaluation output
--strategy missing                                                       # Use cached successful results
--gemini_rpm 5                                                         # API rate limit
```

**Docker Environment:**
```bash
--gpus all                                                              # GPU access
--env-file .env.docker                                                  # Environment variables
-v $(pwd):/app                                                          # Mount project
-v /data4t/hf:/data4t/hf                                               # Mount model cache
leish-gem25:latest                                                      # Docker image
```

## 13. Troubleshooting

### 13.1 Common Errors

| Symptom | Cause | Fix |
|---------|-------|-----|
| **CUDA OOM during batch processing** | Model + batch too large for GPU memory | Reduce `--images_per_answer`, set `CUDA_VISIBLE_DEVICES=0`, use Docker with memory limits |
| **Empty retrieval results** | Vector database not initialized or wrong collection | Check Qdrant connection, verify `QDRANT_URL` and collection name |
| **Slow generation (>30s per question)** | CPU inference or large context | Ensure GPU available, reduce `--top_k`, check `nvidia-smi` |
| **401 from Qdrant** | Invalid API key or wrong region | Re-check `QDRANT_API_KEY` in `.env.docker`, verify endpoint URL |
| **HuggingFace auth errors** | Missing or invalid token | Set `HF_TOKEN` in environment, use `huggingface-cli login` |
| **Frontend cannot reach backend** | CORS or URL mismatch | Set `NEXT_PUBLIC_API_URL=http://localhost:8000` in `web/.env.local` |
| **Docker container exits immediately** | Missing environment file | Create `.env.docker` with required variables |
| **Questions manifest not found** | Wrong path or missing file | Verify `kaggle/working2/rag_knowledge_base/questions_manifest.jsonl` exists |
| **Gemini evaluation fails** | Missing API key or rate limits | Set `GOOGLE_API_KEY`, reduce `--gemini_rpm` to 3-5 |
| **Import errors in Python** | Wrong working directory | Run from project root, check `sys.path` |

### 13.2 Performance Issues

**Slow Processing:**
```bash
# Check GPU utilization
nvtop
# or
watch -n 1 nvidia-smi

# Reduce batch size for memory-constrained systems
python -m rag.gen.run_batch_answers \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out results.ndjson \
  --top_k 5 \
  --images_per_answer 2
```

**Memory Optimization:**
```bash
# Use standalone mode (lower memory)
python -m rag.prompting.run_batch_medgemma4b_standalone \
  --manifest kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
  --out results.ndjson \
  --images_per_answer 1

# Docker with memory limits
docker run --gpus all --memory=16g --env-file .env.docker \
  -v $(pwd):/app leish-gem25:latest \
  python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl
```

### 13.3 Data Issues

**Missing Files:**
```bash
# Check questions manifest
wc -l kaggle/working2/rag_knowledge_base/questions_manifest.jsonl
# Should show: 923 lines

# Verify case directories
ls kaggle/working2/extract/ | wc -l
# Should show: 200+ directories

# Check for empty case directories
find kaggle/working2/extract/ -type d -empty
```

**Corrupted Output:**
```bash
# Validate NDJSON output
python -c "
import json
with open('results.ndjson') as f:
    for i, line in enumerate(f, 1):
        try:
            json.loads(line)
        except json.JSONDecodeError as e:
            print(f'Line {i}: {e}')
"

# Remove incomplete last line
head -n -1 results.ndjson > results_fixed.ndjson
```

### 13.4 Dependency Issues

**Package Conflicts:**
```bash
# Check for conflicting packages
pip check

# Fix common conflicts
pip install --force-reinstall httpx>=0.28.1
pip install -r constraints.txt

# Clean install if needed
pip uninstall -r requirements_recovered.txt -y
pip install -r requirements_recovered.txt -c constraints.txt
```

**Missing Dependencies:**
```bash
# Install missing system packages
sudo apt-get update
sudo apt-get install tesseract-ocr tesseract-ocr-eng
sudo apt-get install ffmpeg libsm6 libxext6  # For OpenCV

# Verify core packages
python -c "import torch, transformers, PIL, pytesseract; print('✅ All core packages available')"

# Install GPU packages if needed
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### 13.5 Environment Issues

**Docker Problems:**
```bash
# Check Docker GPU access
docker run --rm --gpus all nvidia/cuda:11.8-base nvidia-smi

# Verify environment file
cat .env.docker | grep -E 'HF_TOKEN|GOOGLE_API_KEY|QDRANT'

# Check mount points
docker run --rm -v $(pwd):/app leish-gem25:latest ls -la /app/kaggle/working2/
```

**Model Loading Issues:**
```bash
# Test model access
python -c "
from transformers import AutoProcessor
processor = AutoProcessor.from_pretrained(
    'google/medgemma-4b-it',
    cache_dir='/data4t/hf/transformers',
    local_files_only=True
)
print('✅ Model loaded successfully')
"

# Check cache directory
ls -la /data4t/hf/transformers/models--google--medgemma-4b-it/
```

---

## 14. Performance Tips

- Use quantized MedGemma if available (bitsandbytes / GPTQ) for memory savings.
- Pre-warm retriever by running a dry query at startup.
- Consider approximate indexing parameters tuning in Qdrant (HNSW M, ef_search).
- Cache prompt templates & tokenized contexts to reduce latency.

---

## 15. Development & Testing

### 15.1 Testing Components

**Test Single Question Processing:**
```bash
# Test CLI tool
cd app/tools
python ask.py --question "What is cutaneous leishmaniasis?" --case-type cutaneous

# Test standalone processor
python -c "
from rag.prompting.medgemma4b_standalone import StandaloneMedGemma4B
analyzer = StandaloneMedGemma4B()
result = analyzer.diagnose_case(
    '1-case-Cutaneous Leishmaniasis', 
    'What are the key features?'
)
print(result)
"
```

**Test RAG Pipeline:**
```bash
# Process single question from manifest
head -1 kaggle/working2/rag_knowledge_base/questions_manifest.jsonl > test_single.jsonl
python -m rag.gen.run_batch_answers \
  --manifest test_single.jsonl \
  --out test_output.ndjson \
  --top_k 3
```

**Test Evaluation:**
```bash
# Create minimal test evaluation
echo '{"case_id":"test","questions":[{"question":"Test?","gold_answer":"Test answer"}]}' > test_gold.jsonl
echo '{"case_id":"test","question":"Test?","answer":"Test response"}' > test_answers.ndjson

mkdir -p test_eval
echo '{"case_id":"test","questions":[{"question":"Test?","gold_answer":"Test answer"}]}' > test_eval/test.jsonl

python -m rag.gen.evaluation_gemini_only \
  --jsonl_dir test_eval \
  --answers_file test_answers.ndjson \
  --out_dir test_results
```

### 15.2 Code Quality

**Linting & Formatting:**
```bash
# Check code style (if tools available)
ruff check rag/ || echo "Install ruff: pip install ruff"
black --check rag/ || echo "Install black: pip install black"  

# Type checking (if mypy available)
mypy rag/ || echo "Install mypy: pip install mypy"
```

**Performance Profiling:**
```bash
# Time single question processing
time python -m rag.gen.run_batch_answers \
  --manifest <(head -1 kaggle/working2/rag_knowledge_base/questions_manifest.jsonl) \
  --out /dev/null

# Monitor GPU usage during processing
nvtop &
python -m rag.prompting.run_batch_medgemma4b_standalone \
  --manifest <(head -10 kaggle/working2/rag_knowledge_base/questions_manifest.jsonl) \
  --out test_gpu.ndjson
```

### 15.3 Notebook Development

**Start Jupyter for Experimentation:**
```bash
# Install jupyter if needed
pip install jupyter ipykernel

# Start notebook server
jupyter lab
# or
jupyter notebook

# Navigate to experiments/ or create new notebooks
```

**Useful Notebook Snippets:**
```python
# Load questions manifest
import json
with open('kaggle/working2/rag_knowledge_base/questions_manifest.jsonl') as f:
    questions = [json.loads(line) for line in f]
print(f"Loaded {len(questions)} questions")

# Analyze case distribution
from collections import Counter
case_counts = Counter(q['case_id'] for q in questions)
print(f"Questions per case: {case_counts.most_common(5)}")

# Test model loading
from rag.prompting.medgemma4b_standalone import StandaloneMedGemma4B
analyzer = StandaloneMedGemma4B()  # This will load models
print("✅ Models loaded successfully")
```

### 15.4 Integration Testing

**End-to-End Pipeline Test:**
```bash
#!/bin/bash
set -e

echo "🧪 Running end-to-end pipeline test..."

# 1. Process 5 questions with standalone
head -5 kaggle/working2/rag_knowledge_base/questions_manifest.jsonl > test_5q.jsonl
python -m rag.prompting.run_batch_medgemma4b_standalone \
  --manifest test_5q.jsonl \
  --out test_standalone.ndjson

# 2. Process same 5 with RAG
python -m rag.gen.run_batch_answers \
  --manifest test_5q.jsonl \
  --out test_rag.ndjson \
  --top_k 3

# 3. Verify outputs
echo "Standalone results: $(wc -l < test_standalone.ndjson) lines"
echo "RAG results: $(wc -l < test_rag.ndjson) lines"

# 4. Clean up
rm test_5q.jsonl test_standalone.ndjson test_rag.ndjson

echo "✅ End-to-end test completed successfully"
```

---

## 16. Roadmap / TODO

- [ ] Verify and document indexing scripts.
- [ ] Add explicit re-ranker configuration section.
- [ ] Add evaluation metrics script usage.
- [ ] Integrate inline docstrings from retriever & generator modules.
- [ ] Provide real sample manifest file.
- [ ] Add security / PHI handling guidelines (if applicable).
- [ ] Dockerfile(s) and compose examples.
- [ ] License text.

---

## 17. License

MIT License (see LICENSE file).

---

## Acknowledgments

- MedGemma-4B-IT for medical multimodal inference.
- ColQwen / ColPali style vision-text retrievers.
- Qdrant & FAISS for vector similarity search.
- Hugging Face ecosystem.

---

## Appendix: Standalone vs RAG Comparison

| Aspect | Standalone MedGemma | Full RAG |
|--------|---------------------|----------|
| Latency | Lower | Higher (retrieval + ranking) |
| Context Source | Only user-provided images/text | Retrieved documents + images |
| Use Case | Focused case review | Broad knowledge augmentation |
| Setup Complexity | Minimal | Requires vector DB |
| Evidence Traceability | Limited | High (citations, source linking) |
