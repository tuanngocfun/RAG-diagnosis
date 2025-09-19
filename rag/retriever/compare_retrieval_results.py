#!/usr/bin/env python3
"""
Comparison script for retrieval evaluation results between different RAG systems.
Usage:
    python compare_retrieval_results.py --eval1 path/to/eval1_summary.json --eval2 path/to/eval2_summary.json
"""

import argparse
import json
from pathlib import Path
import pandas as pd

def load_summary(path: str) -> dict:
    """Load evaluation summary JSON"""
    with open(path, 'r') as f:
        return json.load(f)

def load_per_query(eval_dir: str) -> pd.DataFrame:
    """Load per-query results"""
    per_query_path = Path(eval_dir) / "retrieval_offline.per_query.jsonl"
    if not per_query_path.exists():
        return pd.DataFrame()
    
    rows = []
    with open(per_query_path, 'r') as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return pd.DataFrame(rows)

def compare_evaluations(eval1_path: str, eval2_path: str):
    """Compare two evaluation results"""
    
    # Load summaries
    eval1 = load_summary(eval1_path)
    eval2 = load_summary(eval2_path)
    
    # Extract system names from paths
    system1 = Path(eval1_path).parent.name
    system2 = Path(eval2_path).parent.name
    
    print("=" * 80)
    print(f"RETRIEVAL EVALUATION COMPARISON")
    print("=" * 80)
    print(f"System 1: {system1}")
    print(f"System 2: {system2}")
    print()
    
    # Basic stats comparison
    print("DATASET OVERVIEW:")
    print("-" * 40)
    metrics = ["queries_total", "queries_with_gold_pages_used", "gold_page_inference_success_rate"]
    for metric in metrics:
        val1 = eval1.get(metric, "N/A")
        val2 = eval2.get(metric, "N/A")
        print(f"{metric:35s}: {val1:>10} vs {val2:>10}")
    print()
    
    # Main retrieval metrics comparison
    print("PRIMARY RETRIEVAL METRICS:")
    print("-" * 40)
    main_metrics = ["MRR@k", "nDCG@k", "Recall@k", "Precision@k", "doc_level_recall@k"]
    for metric in main_metrics:
        val1 = eval1.get(metric, 0.0)
        val2 = eval2.get(metric, 0.0)
        diff = val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else "N/A"
        
        if isinstance(diff, (int, float)):
            change_str = f"({diff:+.4f})" if diff != 0 else "(same)"
        else:
            change_str = ""
            
        print(f"{metric:20s}: {val1:>8.4f} vs {val2:>8.4f} {change_str}")
    print()
    
    # Quality metrics
    print("RETRIEVAL QUALITY INDICATORS:")
    print("-" * 40)
    quality_metrics = ["avg_proxy_textcov@k", "avg_proxy_dup_ratio@k"]
    for metric in quality_metrics:
        val1 = eval1.get(metric, 0.0)
        val2 = eval2.get(metric, 0.0)
        diff = val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else "N/A"
        
        if isinstance(diff, (int, float)):
            change_str = f"({diff:+.4f})" if diff != 0 else "(same)"
        else:
            change_str = ""
            
        print(f"{metric:25s}: {val1:>8.4f} vs {val2:>8.4f} {change_str}")
    print()
    
    # Modality distribution
    print("MODALITY DISTRIBUTION:")
    print("-" * 40)
    mod1 = eval1.get("proxy_modality_share@k", {})
    mod2 = eval2.get("proxy_modality_share@k", {})
    all_mods = set(mod1.keys()) | set(mod2.keys())
    for mod in sorted(all_mods):
        val1 = mod1.get(mod, 0.0)
        val2 = mod2.get(mod, 0.0)
        diff = val2 - val1
        change_str = f"({diff:+.4f})" if diff != 0 else "(same)"
        print(f"{mod:15s}: {val1:>8.4f} vs {val2:>8.4f} {change_str}")
    print()
    
    # Try to load per-query data for more detailed analysis
    try:
        eval1_dir = str(Path(eval1_path).parent)
        eval2_dir = str(Path(eval2_path).parent)
        
        df1 = load_per_query(eval1_dir)
        df2 = load_per_query(eval2_dir)
        
        if not df1.empty and not df2.empty:
            print("PER-QUERY ANALYSIS:")
            print("-" * 40)
            
            # Merge on question_id
            df1_indexed = df1.set_index('question_id').add_suffix('_1')
            df2_indexed = df2.set_index('question_id').add_suffix('_2')
            merged = df1_indexed.join(df2_indexed)
            
            # Calculate improvements
            for metric in ["MRR@k", "nDCG@k", "Recall@k", "Precision@k"]:
                if f"{metric}_1" in merged.columns and f"{metric}_2" in merged.columns:
                    improved = (merged[f"{metric}_2"] > merged[f"{metric}_1"]).sum()
                    degraded = (merged[f"{metric}_2"] < merged[f"{metric}_1"]).sum()
                    same = (merged[f"{metric}_2"] == merged[f"{metric}_1"]).sum()
                    total = len(merged)
                    
                    print(f"{metric:15s}: +{improved:3d} improved, -{degraded:3d} degraded, ={same:3d} same (out of {total})")
            
            print()
            
            # Show biggest improvements/degradations
            if "MRR@k_1" in merged.columns and "MRR@k_2" in merged.columns:
                merged['mrr_diff'] = merged['MRR@k_2'] - merged['MRR@k_1']
                
                print("TOP 5 MOST IMPROVED QUERIES (by MRR@k):")
                top_improved = merged.nlargest(5, 'mrr_diff')
                for idx, row in top_improved.iterrows():
                    print(f"  {idx}: {row['mrr_diff']:+.4f} ({row['MRR@k_1']:.4f} → {row['MRR@k_2']:.4f})")
                
                print("\nTOP 5 MOST DEGRADED QUERIES (by MRR@k):")
                top_degraded = merged.nsmallest(5, 'mrr_diff')
                for idx, row in top_degraded.iterrows():
                    print(f"  {idx}: {row['mrr_diff']:+.4f} ({row['MRR@k_1']:.4f} → {row['MRR@k_2']:.4f})")
    
    except Exception as e:
        print(f"Could not load per-query data for detailed analysis: {e}")
    
    print("\n" + "=" * 80)

def main():
    parser = argparse.ArgumentParser(description="Compare retrieval evaluation results")
    parser.add_argument("--eval1", required=True, help="Path to first evaluation summary.json")
    parser.add_argument("--eval2", required=True, help="Path to second evaluation summary.json") 
    parser.add_argument("--output", help="Save comparison to file (optional)")
    
    args = parser.parse_args()
    
    if args.output:
        import sys
        with open(args.output, 'w') as f:
            # Redirect stdout to file
            old_stdout = sys.stdout
            sys.stdout = f
            compare_evaluations(args.eval1, args.eval2)
            sys.stdout = old_stdout
        print(f"Comparison saved to {args.output}")
    else:
        compare_evaluations(args.eval1, args.eval2)

if __name__ == "__main__":
    main()