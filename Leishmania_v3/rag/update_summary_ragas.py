#!/usr/bin/env python3
"""Update summary.json with RAGAS metrics for a run."""
import json
from pathlib import Path

run_dir = Path('/home/students/Leishmania/Leishmania_v3/rag/runs/eval_bioclinical_bm25_20260120')
ragas_file = run_dir / 'ragas.jsonl'
summary_file = run_dir / 'summary.json'

# Read existing summary
with open(summary_file) as f:
    summary = json.load(f)

# Read RAGAS results
with open(ragas_file) as f:
    ragas_results = [json.loads(l) for l in f]

# Calculate all averages
ragas_metrics = {}
for metric in ['multimodal_faithfulness', 'multimodal_relevance', 'context_relevance',
               'diagnosis_accuracy', 'diagnosis_type_accuracy']:
    scores = [r.get(metric) for r in ragas_results if r.get(metric) is not None]
    ragas_metrics[metric] = sum(scores) / len(scores) if scores else None

# Grounded accuracy
grounded_scores = [r.get('traces', {}).get('grounded_accuracy') 
                   for r in ragas_results 
                   if r.get('traces', {}).get('grounded_accuracy') is not None]
if grounded_scores:
    ragas_metrics['grounded_accuracy'] = sum(grounded_scores) / len(grounded_scores)

# Update summary
summary['ragas_metrics'] = ragas_metrics
summary['metrics']['grounded_accuracy'] = ragas_metrics.get('grounded_accuracy')

# Save updated summary
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)

print('✓ Updated summary.json with RAGAS metrics')
print(json.dumps(ragas_metrics, indent=2))
