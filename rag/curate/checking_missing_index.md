# 🔍 Checking Qdrant Index Completeness

This guide shows how to verify that your Qdrant collection (`leish_cases_pages`) contains all expected documents and that each vector entry has non-empty `text_excerpt` fields.

---

## 1. Inspect Collection Info

Check whether the collection exists and confirm number of points.

```bash
docker run --rm -it --gpus all \
  --env-file /home/students/Leishmania/.env.docker \
  -v /home/students/Leishmania:/app \
  -w /app qdrant/qdrant curl -s http://localhost:6333/collections/leish_cases_pages
```

**What to look for:**

* `"points_count"` should roughly match your number of extracted page chunks.
* `"status": "green"` means the index is healthy.

---

## 2. Count All Indexed Points

```bash
curl -s -X POST "http://localhost:6333/collections/leish_cases_pages/points/count" \
  -H "Content-Type: application/json" \
  -d '{"exact": true}' | jq
```

Compare this number with the number of processed chunks in your `extract/` folder.

---

## 3. Sample Random Points

Query a few random points to check if `text_excerpt` exists.

```bash
curl -s -X POST "http://localhost:6333/collections/leish_cases_pages/points/scroll" \
  -H "Content-Type: application/json" \
  -d '{"limit": 3, "with_payload": true}' | jq '.result.points[].payload.text_excerpt | length'
```

* Should return lengths > 0 for all.
* If you see `null` or `0`, some entries are still empty.

---

## 4. Verify a Specific Case ID

For example, check if case `"1-case- Visceral Leishmaniasis with Renal Involvement_ A Case Report"` is fully indexed:

```bash
curl -s -X POST "http://localhost:6333/collections/leish_cases_pages/points/search" \
  -H "Content-Type: application/json" \
  -d '{
    "vector": [0.0, 0.0, 0.0, 0.0], 
    "top": 5,
    "filter": { "must": [ { "key": "doc_id", "match": { "value": "1-case- Visceral Leishmaniasis with Renal Involvement_ A Case Report" } } ] }
  }' | jq '.result[].payload.text_excerpt | length'
```

---

## 5. Run Integrity Check in Python

You can run a quick script inside Docker:

```bash
docker run --rm -it --gpus all \
  --env-file /home/students/Leishmania/.env.docker \
  -v /home/students/Leishmania:/app \
  -w /app leish-gem25:latest python3 - <<'PYCODE'
from qdrant_client import QdrantClient
client = QdrantClient("localhost", port=6333)

stats = client.get_collection("leish_cases_pages")
print("Total points:", stats.points_count)

empty = 0
for batch in client.scroll("leish_cases_pages", limit=100, with_payload=True)[0]:
    if not batch.payload.get("text_excerpt"):
        empty += 1
print("❌ Empty text_excerpts:", empty)
PYCODE
```

---

## 6. Cross-Check With Extracted Data

Compare:

* Number of `.json` or `.txt` chunks in `/app/kaggle/working2/extract/*/pages/`
* Versus the count in **Qdrant**.

Mismatch = some pages never indexed.

---

## ✅ Success Criteria

* `points_count` matches expected extracted chunks.
* No or very few empty `text_excerpt`.
* Queries on specific `doc_id` return non-empty text.

If something is missing, re-run your indexing step for those cases only.

BY running:

```bash
docker run --rm -it --gpus all --env-file /home/students/Leishmania/.env.docker -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf -w /app leish-gem25:latest python rag/curate/fix_text_extraction.py
```

Or to reindex everything cleanly, follow the steps:
1. Diagnose current issues:
```bash
docker run --rm -it --gpus all --user $(id -u):$(id -g) --env-file /home/students/Leishmania/.env.docker -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf -w /app leish-gem25:latest python rag/curate/diagnose_qdrant_duplicates.py
```
2. Clean re-index(recommended):
```bash
docker run --rm -it --gpus all --user $(id -u):$(id -g) --env-file /home/students/Leishmania/.env.docker -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf -w /app leish-gem25:latest python rag/curate/clean_reindex_qdrant.py
```
3. Verify success again:
```bash
docker run --rm -it --gpus all --user $(id -u):$(id -g) --env-file /home/students/Leishmania/.env.docker -e RAG_ROOT=/app -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract -e RAG_PDF_DIRS="/app/data/standard" -e QDRANT_COLLECTION="leish_cases_pages" -v /home/students/Leishmania:/app -v /data4t/hf:/data4t/hf -w /app leish-gem25:latest python rag/curate/diagnose_qdrant_duplicates.py
```