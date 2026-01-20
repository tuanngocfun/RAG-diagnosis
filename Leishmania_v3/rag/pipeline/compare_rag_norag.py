#!/usr/bin/env python3
"""
Compare RAG vs No-RAG results with diagnosis accuracy.

Compares two runs:
1. RAG run: answers generated WITH retrieved contexts
2. No-RAG baseline: answers generated WITHOUT retrieval

Reports diagnosis accuracy for each using LLM judge.

Usage:
    python -m rag.pipeline.compare_rag_norag --rag multimodal_rag_full --norag baseline_norag_full
"""
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from collections import defaultdict

from .config import RUNS_DIR, SPLIT_DIR


@dataclass
class ComparisonResult:
    """Result for a single query comparison."""
    case_id: str
    query_type: str
    ground_truth_diagnosis: str
    ground_truth_type: str
    rag_predicted: str
    rag_score: float
    norag_predicted: str
    norag_score: float
    rag_correct: bool
    norag_correct: bool


def extract_diagnosis_from_answer(answer: str) -> str:
    """Extract predicted diagnosis from structured answer."""
    import re
    
    # Try to find "Primary Diagnosis:" pattern
    match = re.search(r'\*\*Primary Diagnosis:\*\*\s*(.+?)(?:\n|$)', answer)
    if match:
        return match.group(1).strip()
    
    # Try "Diagnosis Type:" 
    match = re.search(r'\*\*Diagnosis Type:\*\*\s*(.+?)(?:\n|$)', answer)
    if match:
        return match.group(1).strip()
    
    # Fallback: first 100 chars
    return answer[:100].strip()


def compare_rag_norag(
    rag_run_id: str,
    norag_run_id: str,
    use_llm_judge: bool = True,
    sample_size: Optional[int] = None
) -> Path:
    """
    Compare RAG vs No-RAG runs.
    
    Args:
        rag_run_id: Run ID for RAG evaluation
        norag_run_id: Run ID for No-RAG baseline
        use_llm_judge: Use LLM judge for diagnosis accuracy
        sample_size: Number of samples to evaluate (None = all)
    
    Returns:
        Path to comparison report
    """
    rag_dir = RUNS_DIR / rag_run_id
    norag_dir = RUNS_DIR / norag_run_id
    
    print(f"{'='*60}")
    print(f"RAG vs NO-RAG COMPARISON")
    print(f"{'='*60}")
    print(f"RAG run: {rag_run_id}")
    print(f"No-RAG run: {norag_run_id}")
    
    # Load RAG answers
    rag_file = rag_dir / "answers.jsonl"
    if not rag_file.exists():
        rag_file = rag_dir / "answers_gemini.jsonl"
    
    with open(rag_file) as f:
        rag_answers = {json.loads(l)['qid']: json.loads(l) for l in f}
    print(f"RAG answers loaded: {len(rag_answers)}")
    
    # Load No-RAG answers
    norag_file = norag_dir / "answers_norag.jsonl"
    if not norag_file.exists():
        norag_file = norag_dir / "answers_gemini.jsonl"
    
    with open(norag_file) as f:
        norag_answers = {json.loads(l)['qid']: json.loads(l) for l in f}
    print(f"No-RAG answers loaded: {len(norag_answers)}")
    
    # Load ground truth
    test_file = SPLIT_DIR / "test.jsonl"
    with open(test_file) as f:
        test_cases = {json.loads(l)['case_id']: json.loads(l) for l in f}
    print(f"Ground truth loaded: {len(test_cases)} cases")
    
    # Initialize evaluator if using LLM judge
    evaluator = None
    if use_llm_judge:
        from .ragas_evaluator import RAGAsLibraryEvaluator
        evaluator = RAGAsLibraryEvaluator()
        print("LLM Judge initialized")
    
    # Compare all matching queries
    results = []
    by_query_type = defaultdict(lambda: {"rag": [], "norag": []})
    
    common_qids = set(rag_answers.keys()) & set(norag_answers.keys())
    if sample_size:
        common_qids = list(common_qids)[:sample_size]
    
    print(f"\nEvaluating {len(common_qids)} common queries...")
    
    for i, qid in enumerate(common_qids):
        rag_ans = rag_answers[qid]
        norag_ans = norag_answers[qid]
        
        # Extract case_id and query_type from qid
        parts = qid.split("::")
        case_id = parts[0]
        query_type = parts[1] if len(parts) > 1 else "unknown"
        
        # Get ground truth
        if case_id not in test_cases:
            continue
        
        gt = test_cases[case_id]
        ground_truth = {
            "diagnosis": gt.get("diagnosis", ""),
            "diagnosis_type": gt.get("diagnosis_type", ""),
            "species": gt.get("species", "")  # Include species for LLM judge
        }
        
        # Extract predictions
        rag_pred = extract_diagnosis_from_answer(rag_ans.get("answer", ""))
        norag_pred = extract_diagnosis_from_answer(norag_ans.get("answer", ""))
        
        # Evaluate accuracy
        if evaluator and use_llm_judge:
            import asyncio
            
            async def eval_pair():
                try:
                    rag_res = await evaluator.evaluate_diagnosis_equivalence(
                        rag_ans.get("answer", ""), ground_truth
                    )
                    r_score = rag_res.diagnosis_score
                except Exception as e:
                    print(f"  Error evaluating RAG {qid}: {e}")
                    r_score = 0.0
                
                try:
                    norag_res = await evaluator.evaluate_diagnosis_equivalence(
                        norag_ans.get("answer", ""), ground_truth
                    )
                    n_score = norag_res.diagnosis_score
                except Exception as e:
                    print(f"  Error evaluating No-RAG {qid}: {e}")
                    n_score = 0.0
                
                return r_score, n_score
            
            rag_score, norag_score = asyncio.run(eval_pair())
            
            time.sleep(0.5)  # Rate limiting
        else:
            # Simple string matching fallback
            rag_score = 1.0 if ground_truth["diagnosis_type"].lower() in rag_pred.lower() else 0.0
            norag_score = 1.0 if ground_truth["diagnosis_type"].lower() in norag_pred.lower() else 0.0
        
        result = ComparisonResult(
            case_id=case_id,
            query_type=query_type,
            ground_truth_diagnosis=ground_truth["diagnosis"],
            ground_truth_type=ground_truth["diagnosis_type"],
            rag_predicted=rag_pred,
            rag_score=rag_score,
            norag_predicted=norag_pred,
            norag_score=norag_score,
            rag_correct=rag_score >= 0.5,
            norag_correct=norag_score >= 0.5
        )
        results.append(result)
        
        by_query_type[query_type]["rag"].append(rag_score)
        by_query_type[query_type]["norag"].append(norag_score)
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(common_qids)}")
    
    # Calculate summary statistics
    if not results:
        print("No results to compare!")
        return None
    
    rag_mean = sum(r.rag_score for r in results) / len(results)
    norag_mean = sum(r.norag_score for r in results) / len(results)
    rag_correct = sum(1 for r in results if r.rag_correct)
    norag_correct = sum(1 for r in results if r.norag_correct)
    
    # Print summary
    print(f"\n{'='*60}")
    print("COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"Total queries evaluated: {len(results)}")
    print()
    print(f"{'Metric':<30} {'RAG':>12} {'No-RAG':>12} {'Diff':>12}")
    print("-" * 66)
    print(f"{'Mean Accuracy Score':<30} {rag_mean:>11.2%} {norag_mean:>11.2%} {(rag_mean-norag_mean):>+11.2%}")
    print(f"{'Correct (score >= 0.5)':<30} {rag_correct:>12} {norag_correct:>12} {rag_correct-norag_correct:>+12}")
    print(f"{'Accuracy Rate':<30} {rag_correct/len(results):>11.2%} {norag_correct/len(results):>11.2%} {(rag_correct-norag_correct)/len(results):>+11.2%}")
    
    # By query type
    print(f"\n{'='*60}")
    print("BY QUERY TYPE")
    print(f"{'='*60}")
    print(f"{'Query Type':<35} {'RAG':>10} {'No-RAG':>10}")
    print("-" * 55)
    for qt, scores in by_query_type.items():
        rag_qt_mean = sum(scores["rag"]) / len(scores["rag"]) if scores["rag"] else 0
        norag_qt_mean = sum(scores["norag"]) / len(scores["norag"]) if scores["norag"] else 0
        print(f"{qt:<35} {rag_qt_mean:>9.2%} {norag_qt_mean:>9.2%}")
    
    # Save detailed results
    output_file = RUNS_DIR / f"comparison_{rag_run_id}_vs_{norag_run_id}.json"
    comparison_data = {
        "rag_run": rag_run_id,
        "norag_run": norag_run_id,
        "summary": {
            "total_evaluated": len(results),
            "rag_mean_accuracy": rag_mean,
            "norag_mean_accuracy": norag_mean,
            "rag_correct_count": rag_correct,
            "norag_correct_count": norag_correct,
            "rag_advantage": rag_mean - norag_mean
        },
        "by_query_type": {
            qt: {
                "rag_mean": sum(s["rag"]) / len(s["rag"]) if s["rag"] else 0,
                "norag_mean": sum(s["norag"]) / len(s["norag"]) if s["norag"] else 0,
                "count": len(s["rag"])
            }
            for qt, s in by_query_type.items()
        },
        "details": [
            {
                "case_id": r.case_id,
                "query_type": r.query_type,
                "ground_truth": r.ground_truth_diagnosis,  # Full diagnosis from test.jsonl
                "ground_truth_type": r.ground_truth_type,  # Abbreviated type
                "rag_prediction": r.rag_predicted[:100],
                "rag_score": r.rag_score,
                "norag_prediction": r.norag_predicted[:100],
                "norag_score": r.norag_score
            }
            for r in results
        ]
    }
    
    with open(output_file, "w") as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\n✓ Detailed results saved to: {output_file}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Compare RAG vs No-RAG results")
    parser.add_argument("--rag", required=True, help="RAG run ID")
    parser.add_argument("--norag", required=True, help="No-RAG run ID")
    parser.add_argument("--no-llm-judge", action="store_true", help="Use string matching instead of LLM")
    parser.add_argument("--sample", type=int, default=None, help="Number of samples to evaluate")
    
    args = parser.parse_args()
    
    compare_rag_norag(
        rag_run_id=args.rag,
        norag_run_id=args.norag,
        use_llm_judge=not args.no_llm_judge,
        sample_size=args.sample
    )
