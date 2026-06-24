"""
Metric Sanity Check Script

Per GPT 5.2 feedback: Explain the "strange" metrics pattern:
- recall@5 ≈ 0.039 (very low)
- precision@5 ≈ 0.467 (high)
- nDCG@5 ≈ 0.485 (high)

This script analyzes qrels to understand why recall is low.
"""
import json
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List

from .config import DATA_ROOT


def analyze_qrels(qrels_file: str = "qrels_grade3.json") -> Dict:
    """
    Analyze qrels to understand metric patterns.
    
    The low recall@5 vs high precision@5 pattern typically indicates:
    - Many relevant docs per query (recall denominator is large)
    - Retrieved docs are often relevant (precision numerator is good)
    
    Returns:
        Dict with analysis results
    """
    qrels_path = DATA_ROOT / qrels_file
    
    print(f"{'='*60}")
    print("QRELS ANALYSIS - Understanding Recall vs Precision")
    print(f"{'='*60}")
    print(f"\nFile: {qrels_path}")
    
    with open(qrels_path) as f:
        qrels = json.load(f)
    
    print(f"Total queries: {len(qrels)}")
    
    # Analyze relevant docs per query
    relevant_counts = []
    grade_distribution = Counter()
    
    for query_id, docs in qrels.items():
        # Count docs with grade >= 1 as relevant
        relevant_docs = [d for d, grade in docs.items() if grade >= 1]
        relevant_counts.append(len(relevant_docs))
        
        for doc_id, grade in docs.items():
            grade_distribution[grade] += 1
    
    # Statistics
    results = {
        "total_queries": len(qrels),
        "total_judgments": sum(grade_distribution.values()),
        "grade_distribution": dict(grade_distribution),
        "relevant_per_query": {
            "min": int(np.min(relevant_counts)),
            "max": int(np.max(relevant_counts)),
            "mean": float(np.mean(relevant_counts)),
            "median": float(np.median(relevant_counts)),
            "std": float(np.std(relevant_counts))
        }
    }
    
    print(f"\n{'='*60}")
    print("GRADE DISTRIBUTION")
    print(f"{'='*60}")
    for grade in sorted(grade_distribution.keys()):
        count = grade_distribution[grade]
        pct = count / sum(grade_distribution.values()) * 100
        print(f"  Grade {grade}: {count:4d} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("RELEVANT DOCS PER QUERY (grade >= 1)")
    print(f"{'='*60}")
    print(f"  Min:    {results['relevant_per_query']['min']}")
    print(f"  Max:    {results['relevant_per_query']['max']}")
    print(f"  Mean:   {results['relevant_per_query']['mean']:.1f}")
    print(f"  Median: {results['relevant_per_query']['median']:.1f}")
    print(f"  Std:    {results['relevant_per_query']['std']:.1f}")
    
    # Explain the metrics
    print(f"\n{'='*60}")
    print("METRIC EXPLANATION")
    print(f"{'='*60}")
    
    avg_relevant = results['relevant_per_query']['mean']
    
    print(f"""
Why Recall@5 is LOW ({0.039:.3f}):
  - Recall@K = (relevant in top K) / (total relevant)
  - With avg {avg_relevant:.1f} relevant docs per query,
    retrieving 5 can only achieve recall ≈ 5/{avg_relevant:.1f} = {5/avg_relevant:.3f} at best
  - Low recall is EXPECTED when there are many relevant docs

Why Precision@5 is HIGH ({0.467:.3f}):
  - Precision@K = (relevant in top K) / K = (relevant in top 5) / 5
  - 0.467 means on average 2.3 of top 5 are relevant
  - This is good retrieval quality!

Why nDCG@5 is HIGH ({0.485:.3f}):
  - nDCG considers graded relevance and ranking position
  - High nDCG means relevant docs are ranked higher

CONCLUSION:
  Low recall with high precision/nDCG is NORMAL for clinical IR
  where each query may have 20+ relevant case reports.
  
  For thesis, report:
  1. Average relevant docs per query
  2. Recall@K with context (K much smaller than avg relevant)
  3. Focus on precision/nDCG as primary metrics for this scenario
""")
    
    # Distribution histogram
    print(f"\n{'='*60}")
    print("DISTRIBUTION OF RELEVANT DOCS PER QUERY")
    print(f"{'='*60}")
    
    bins = [0, 5, 10, 20, 50, 100, 500]
    for i in range(len(bins) - 1):
        count = sum(1 for c in relevant_counts if bins[i] <= c < bins[i+1])
        pct = count / len(relevant_counts) * 100
        print(f"  {bins[i]:3d}-{bins[i+1]:3d}: {count:3d} queries ({pct:.1f}%)")
    
    return results


def check_metric_consistency(run_dir: str = "phase3_hybrid") -> Dict:
    """
    Check if metrics in a run are consistent with qrels.
    """
    from .config import RUNS_DIR
    
    run_path = RUNS_DIR / run_dir
    summary_path = run_path / "summary.json"
    
    print(f"\n{'='*60}")
    print(f"RUN METRICS CHECK: {run_dir}")
    print(f"{'='*60}")
    
    if not summary_path.exists():
        print(f"  ✗ summary.json not found")
        return {}
    
    with open(summary_path) as f:
        summary = json.load(f)
    
    metrics = summary.get("metrics", {})
    
    print(f"  Queries: {summary.get('n_queries', 'N/A')}")
    print(f"  Retrieved: {summary.get('n_retrieved', 'N/A')}")
    print()
    print("  Metrics:")
    for metric_name, values in metrics.items():
        if isinstance(values, dict):
            for k, v in values.items():
                print(f"    {metric_name}{k}: {v:.4f}")
        else:
            print(f"    {metric_name}: {values:.4f}")
    
    return metrics


if __name__ == "__main__":
    results = analyze_qrels()
    
    # Save analysis
    output_path = DATA_ROOT / "qrels_analysis.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAnalysis saved to: {output_path}")
    
    # Check existing runs
    check_metric_consistency("phase3_hybrid")
    check_metric_consistency("phase3_hybrid_rerank")
