#!/usr/bin/env python3
"""
Score Distribution Analysis for Adaptive RAG Threshold Selection

Per GPT 5.2, Gemini 3 Pro, Grok 4.1 recommendations:
- Analyze retrieval score distribution
- Correlate with correct/incorrect predictions
- Find data-driven threshold for confidence gating

Usage:
    python analyze_score_distribution.py <run_dir>
    
Example:
    python analyze_score_distribution.py rag/runs/medgemma_4b_topk3_extended_20260131
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

def load_retrieval_data(run_dir: Path) -> List[Dict]:
    """Load retrieval.jsonl with scores."""
    retrieval_file = run_dir / "retrieval.jsonl"
    if not retrieval_file.exists():
        raise FileNotFoundError(f"No retrieval.jsonl in {run_dir}")
    
    with open(retrieval_file) as f:
        return [json.loads(line) for line in f]

def load_ragas_data(run_dir: Path) -> Dict[str, Dict]:
    """Load RAGAS evaluation results with diagnosis accuracy."""
    ragas_file = run_dir / "ragas.jsonl"
    if not ragas_file.exists():
        raise FileNotFoundError(f"No ragas.jsonl in {run_dir}")
    
    results = {}
    with open(ragas_file) as f:
        for line in f:
            r = json.loads(line)
            results[r["qid"]] = r
    return results

def extract_scores(retrieval_data: List[Dict]) -> Dict[str, Dict]:
    """Extract max_score, margin, and other signals from retrieval results."""
    scores = {}
    
    for item in retrieval_data:
        qid = item.get("qid")
        if not qid:
            continue
        
        # Use 'contexts' key (not 'results') - this is the actual format
        contexts = item.get("contexts", [])
        if not contexts:
            scores[qid] = {"max_score": 0.0, "margin": 0.0, "num_results": 0}
            continue
        
        # Extract scores from contexts
        result_scores = [c.get("score", 0.0) for c in contexts]
        result_scores = sorted(result_scores, reverse=True)
        
        max_score = result_scores[0] if result_scores else 0.0
        top3_score = result_scores[2] if len(result_scores) > 2 else result_scores[-1] if result_scores else 0.0
        margin = max_score - top3_score  # Score margin (higher = more confident)
        
        scores[qid] = {
            "max_score": max_score,
            "margin": margin,
            "num_results": len(contexts),
            "all_scores": result_scores[:5]  # Top-5 scores
        }
    
    return scores

def analyze_by_bins(
    scores: Dict[str, Dict],
    ragas: Dict[str, Dict],
    bins: List[Tuple[float, float]] = None
) -> Dict[str, Dict]:
    """
    Analyze accuracy by score bins.
    Returns: {bin_label: {rag_accuracy, count, correct, incorrect}}
    """
    if bins is None:
        # Use bins appropriate for hybrid retrieval scores (typically 0.01-0.03 range)
        bins = [
            (0.000, 0.010), (0.010, 0.012), (0.012, 0.014), 
            (0.014, 0.016), (0.016, 0.018), (0.018, 0.020),
            (0.020, 0.025), (0.025, 0.030), (0.030, 1.000)
        ]
    
    results = {}
    
    for low, high in bins:
        bin_label = f"{low:.1f}-{high:.1f}"
        bin_data = {"count": 0, "correct": 0, "partial": 0, "incorrect": 0}
        
        for qid, score_info in scores.items():
            max_score = score_info["max_score"]
            
            if low <= max_score < high:
                bin_data["count"] += 1
                
                if qid in ragas:
                    diag_acc = ragas[qid].get("diagnosis_accuracy", 0.0)
                    if diag_acc is None:
                        diag_acc = 0.0
                    
                    if diag_acc >= 0.99:
                        bin_data["correct"] += 1
                    elif diag_acc >= 0.5:
                        bin_data["partial"] += 1
                    else:
                        bin_data["incorrect"] += 1
        
        if bin_data["count"] > 0:
            bin_data["accuracy"] = (bin_data["correct"] + 0.5 * bin_data["partial"]) / bin_data["count"]
        else:
            bin_data["accuracy"] = 0.0
        
        results[bin_label] = bin_data
    
    return results

def find_optimal_threshold(bin_analysis: Dict[str, Dict], norag_accuracy: float = 0.8227) -> float:
    """
    Find threshold where RAG accuracy exceeds No-RAG.
    Per GPT 5.2: Gate ON only in bins where RAG > NoRAG.
    """
    best_threshold = 0.6  # Default fallback
    
    for bin_label, data in sorted(bin_analysis.items()):
        if data["count"] >= 3:  # Need enough samples
            rag_acc = data["accuracy"]
            delta = rag_acc - norag_accuracy
            
            # Find first bin where RAG helps
            if delta > 0:
                low = float(bin_label.split("-")[0])
                best_threshold = low
                break
    
    return best_threshold

def print_analysis_report(
    bin_analysis: Dict[str, Dict],
    norag_accuracy: float,
    recommended_threshold: float
):
    """Print formatted analysis report."""
    print("\n" + "="*70)
    print("SCORE DISTRIBUTION ANALYSIS FOR ADAPTIVE RAG")
    print("="*70)
    
    print(f"\nNo-RAG Baseline Accuracy: {norag_accuracy:.2%}")
    print(f"\nAnalysis by Score Bins:")
    print("-"*70)
    print(f"{'Bin':<12} {'Count':>8} {'Correct':>8} {'Partial':>8} {'Wrong':>8} {'Accuracy':>10} {'Δ vs NoRAG':>12}")
    print("-"*70)
    
    for bin_label, data in sorted(bin_analysis.items()):
        if data["count"] > 0:
            delta = data["accuracy"] - norag_accuracy
            delta_str = f"{delta:+.2%}"
            rag_helps = "✓" if delta > 0 else "✗"
            print(f"{bin_label:<12} {data['count']:>8} {data['correct']:>8} {data['partial']:>8} {data['incorrect']:>8} {data['accuracy']:>10.2%} {delta_str:>10} {rag_helps}")
    
    print("-"*70)
    print(f"\n🎯 RECOMMENDED THRESHOLD: {recommended_threshold:.1f}")
    print(f"   Use RAG when max_score >= {recommended_threshold}")
    print(f"   Fall back to No-RAG when max_score < {recommended_threshold}")
    
    # Calculate expected improvement
    gate_on_correct = 0
    gate_on_total = 0
    gate_off_total = 0
    
    for bin_label, data in bin_analysis.items():
        low = float(bin_label.split("-")[0])
        if low >= recommended_threshold:
            gate_on_total += data["count"]
            gate_on_correct += data["correct"] + 0.5 * data["partial"]
        else:
            gate_off_total += data["count"]
    
    total = gate_on_total + gate_off_total
    if total > 0:
        # Expected accuracy with adaptive RAG
        gate_on_acc = gate_on_correct / gate_on_total if gate_on_total > 0 else 0
        expected_acc = (gate_on_correct + gate_off_total * norag_accuracy) / total
        
        print(f"\n📊 EXPECTED OUTCOME:")
        print(f"   Queries with RAG (gate ON):  {gate_on_total:>4} ({100*gate_on_total/total:.1f}%)")
        print(f"   Queries with No-RAG (gate OFF): {gate_off_total:>4} ({100*gate_off_total/total:.1f}%)")
        print(f"   Expected RAG-on accuracy: {gate_on_acc:.2%}")
        print(f"   Expected overall accuracy: {expected_acc:.2%}")
        print(f"   Expected improvement: {expected_acc - norag_accuracy:+.2%}")

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_score_distribution.py <run_dir>")
        print("Example: python analyze_score_distribution.py rag/runs/medgemma_4b_topk3_extended_20260131")
        sys.exit(1)
    
    run_dir = Path(sys.argv[1])
    if not run_dir.exists():
        print(f"Error: Run directory not found: {run_dir}")
        sys.exit(1)
    
    norag_accuracy = 0.8227  # Known No-RAG baseline for MedGemma
    
    try:
        # Load data
        print(f"Loading data from {run_dir}...")
        retrieval_data = load_retrieval_data(run_dir)
        ragas_data = load_ragas_data(run_dir)
        
        print(f"  Loaded {len(retrieval_data)} retrieval results")
        print(f"  Loaded {len(ragas_data)} RAGAS evaluations")
        
        # Extract scores
        scores = extract_scores(retrieval_data)
        
        # Analyze by bins
        bin_analysis = analyze_by_bins(scores, ragas_data)
        
        # Find optimal threshold
        recommended_threshold = find_optimal_threshold(bin_analysis, norag_accuracy)
        
        # Print report
        print_analysis_report(bin_analysis, norag_accuracy, recommended_threshold)
        
        # Save results for later use
        output_file = run_dir / "score_analysis.json"
        with open(output_file, "w") as f:
            json.dump({
                "bin_analysis": bin_analysis,
                "recommended_threshold": recommended_threshold,
                "norag_accuracy": norag_accuracy,
                "num_samples": len(scores)
            }, f, indent=2)
        print(f"\n✓ Analysis saved to {output_file}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
