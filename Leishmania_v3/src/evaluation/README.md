# Evaluation Framework

Scripts for evaluating and comparing KG methods.

## Files

| File | Description |
|------|-------------|
| `metrics.py` | Core metrics (Precision, Recall, F1, MRR) |
| `compare_methods.py` | Compare your method vs baselines vs external |

## Metrics

### Entity Extraction
- Precision, Recall, F1 (by entity type)

### KG Quality
- Coverage (% cases with entities)
- Density (avg entities per case)

### Retrieval (RAG)
- Precision@K, Recall@K
- MRR (Mean Reciprocal Rank)
- MAP (Mean Average Precision)

## Usage

```bash
# Compare all methods
python compare_methods.py

# Use metrics in your code
from metrics import entity_extraction_metrics, kg_quality_metrics
```
