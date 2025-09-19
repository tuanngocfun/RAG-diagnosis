## 1. Make sure your environment is ready

* This script imports from your repo (`rag.reranking.med4b_qdrant_bge`), so it must be run inside the container where that module is available.
* It also uses `CFG.QDRANT_URL` and `CFG.QDRANT_API_KEY`, so you’ll need those set up in your `.env.docker` file.

---

## 2. Run inside Docker

```bash
docker run --rm -it --gpus all \
  --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app \
  -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" \
  -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app \
  -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest \
  python rag/validate_files/quick_check.py
```

---

## 3. What it does

* It builds an embedding for the fixed question:

  ```
  "Which therapy was used and what was the short-term outcome?"
  ```

* It **filters** results to one doc:

  ```
  "1-case- Case Report_ Simple Nodular Cutaneous Leishmaniasis..."
  ```

* Then it calls `_qdrant_search` with top-k=10 and prints tuples like:

  ```
  [(score, page_index), (score, page_index), ...]
  ```

So you’ll see a list of similarity scores (rounded to 4 decimals) and the page indices of the matching chunks.

---

## 4. Customizing

* Change `DOC` to test a different case.
* Change `q` to ask another question.
* Change `10` in `_qdrant_search(c, qv, 10, flt, None)` to adjust top-k.