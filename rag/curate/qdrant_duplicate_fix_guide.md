# 🔧 Qdrant Indexing Duplicate Issue Analysis & Solution

## Problem Identified

You were experiencing duplicate answers in your NDJSON output, which indicates duplicate entries in your Qdrant vector database. After analyzing the code, here are the key issues:

### Root Cause
1. **Wrong CLI module**: You were running `qdrant-init` and `qdrant-index` commands against the wrong module
   - ❌ **Wrong**: `python -m rag.test.run_batch_answers_medgemma4b_test_medcpt qdrant-init`
   - ✅ **Correct**: `python -m rag.test.medgemma4b_qdrant_bge_medcpt qdrant-init`

2. **Multiple indexing runs**: Running `qdrant-index` multiple times without proper cleanup leads to duplicate entries

3. **Upsert behavior**: The indexing code uses `client.upsert()` which should prevent duplicates with the same ID, but if UIDs are generated inconsistently, duplicates can occur

## Technical Details

### UID Generation
The code uses UUID5 for generating unique IDs:
```python
def build_uid(doc_id: str, page_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}__p{page_idx:04d}"))
```

### Indexing Process
- Uses `client.upsert()` which should overwrite entries with the same ID
- Processes pages in batches for efficiency
- Generates both image and text embeddings

## Solution Steps

### 1. Diagnose Current State
```bash
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python rag/curate/diagnose_qdrant_duplicates.py
```

### 2. Clean Re-index (Recommended)
```bash
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python rag/curate/clean_reindex_qdrant.py
```

### 3. Manual Re-index (Alternative)
If you prefer to do it manually:

```bash
# Step 1: Initialize collection (drops existing)
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python -m rag.test.medgemma4b_qdrant_bge_medcpt qdrant-init

# Step 2: Index all pages
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python -m rag.test.medgemma4b_qdrant_bge_medcpt qdrant-index

# Step 3: Create payload indexes
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python -m rag.test.medgemma4b_qdrant_bge_medcpt qdrant-create-indexes
```

### 4. Verify Success
After re-indexing, run the diagnosis again:
```bash
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest python rag/curate/diagnose_qdrant_duplicates.py
```

### 5. Re-run Batch Answers
Once the collection is clean, re-run your batch processing:
```bash
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest bash -lc 'python -m rag.test.run_batch_answers_medgemma4b_test_medcpt --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl --out /app/kaggle/working2/rag_knowledge_base/answers/answers_clean_reindexed.ndjson --topk 8 --pool_mult 3 --images_per_answer 3 --use_reranker --resume'
```

## Prevention Tips

1. **Always use the correct module** for CLI commands:
   - Indexing: `python -m rag.test.medgemma4b_qdrant_bge_medcpt`
   - Batch processing: `python -m rag.test.run_batch_answers_medgemma4b_test_medcpt`

2. **Use `qdrant-init` before re-indexing** to ensure a clean slate

3. **Run diagnostic checks** after indexing to verify collection health

4. **Monitor collection size** - if it grows unexpectedly large, you may have duplicates

## Expected Results

After proper re-indexing:
- ✅ No duplicate UIDs
- ✅ No duplicate doc_id + page_index combinations
- ✅ One point per page per case
- ✅ Unique answers in batch processing output

The diagnostic script will provide detailed analysis and confirm the collection is healthy.