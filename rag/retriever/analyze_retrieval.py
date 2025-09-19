#!/usr/bin/env python3
import json

queries = []
with open('kaggle/working2/rag_knowledge_base/eval/retrieval_offline_gemini25_bge_reranker/retrieval_offline.per_query.jsonl', 'r') as f:
    for line in f:
        q = json.loads(line.strip())
        queries.append(q)

# Filter out queries with None precision
valid_queries = [q for q in queries if q.get('Precision@k') is not None]

# Sort by precision and show worst performing queries
worst_precision = sorted(valid_queries, key=lambda x: x.get('Precision@k', 0))[:10]
print('WORST PRECISION@10:')
for q in worst_precision:
    print(f"Question ID: {q['question_id']}, Precision@10: {q.get('Precision@k', 0):.3f}, Text Coverage: {q.get('proxy_textcov@k', 0):.3f}")

print('\nBEST PRECISION@10:')
best_precision = sorted(valid_queries, key=lambda x: x.get('Precision@k', 0), reverse=True)[:10]
for q in best_precision:
    print(f"Question ID: {q['question_id']}, Precision@10: {q.get('Precision@k', 0):.3f}, Text Coverage: {q.get('proxy_textcov@k', 0):.3f}")

print(f'\nOVERALL STATISTICS:')
print(f'Total queries: {len(queries)}')
print(f'Valid precision queries: {len(valid_queries)}')
print(f'Average Precision@10: {sum(q.get("Precision@k", 0) for q in valid_queries) / len(valid_queries):.3f}')
print(f'Average Text Coverage: {sum(q.get("proxy_textcov@k", 0) for q in queries) / len(queries):.3f}')
print(f'Queries with 0 precision: {sum(1 for q in valid_queries if q.get("Precision@k", 0) == 0)}')
print(f'Queries with None precision: {sum(1 for q in queries if q.get("Precision@k") is None)}')

# Analyze text coverage distribution
low_coverage = [q for q in queries if q.get('proxy_textcov@k', 0) < 0.3]
print(f'Queries with low text coverage (<30%): {len(low_coverage)}')

# Check modality distribution for low-performing queries
print(f'\nMODALITY ANALYSIS FOR LOW PRECISION QUERIES:')
for q in worst_precision[:5]:
    modalities = q.get('proxy_modality_counts@k', {})
    print(f"Question {q['question_id']}: {modalities}")