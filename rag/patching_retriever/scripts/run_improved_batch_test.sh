#!/bin/bash

# Improved batch processing with better parameters for higher precision
echo "Starting improved batch processing with enhanced retrieval settings..."

docker run --rm -it --gpus all \
  --user $(id -u):$(id -g) \
  --env-file /home/students/Leishmania/.env.docker \
  -e RAG_ROOT=/app \
  -e RAG_EXTRACT_ROOT=/app/kaggle/working2/extract \
  -e RAG_PDF_DIRS="/app/data/standard" \
  -e QDRANT_COLLECTION="leish_cases_pages" \
  -v /home/students/Leishmania:/app \
  -v /data4t/hf:/data4t/hf \
  -w /app leish-gem25:latest \
  bash -lc 'python -m rag.test.run_batch_answers_gem25_test_better_retrieving \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
    --out /app/kaggle/working2/rag_knowledge_base/answers/answers_gem25_improved_retrieval.ndjson \
    --topk 8 \
    --pool_mult 5 \
    --images_per_answer 3 \
    --use_reranker \
    --score_threshold 0.0 \
    --resume'

echo "Batch processing completed. Running evaluation..."

# Run evaluation on improved results
python -m rag.retriever.offline_retrieval_eval_fixed \
  --answers "/home/students/Leishmania/kaggle/working2/rag_knowledge_base/answers/answers_gem25_improved_retrieval.ndjson" \
  --questions "/home/students/Leishmania/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl" \
  --qa_dir "/home/students/Leishmania/kaggle/working2/rag_knowledge_base/qa/jsonl" \
  --k 10 \
  --out "/home/students/Leishmania/kaggle/working2/rag_knowledge_base/eval/gem25_retrieval_improved"

echo "Evaluation completed. Check results in:"
echo "  - Answers: kaggle/working2/rag_knowledge_base/answers/answers_gem25_improved_retrieval.ndjson"
echo "  - Evaluation: kaggle/working2/rag_knowledge_base/eval/gem25_retrieval_improved/"