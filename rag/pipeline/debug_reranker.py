"""
Reranker Debug Script - Analyze before/after rankings

Helps diagnose why MedCPT reranker decreases nDCG@5 but improves MRR.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple

from .config import DATA_ROOT, TRAIN_JSONL
from .retriever import Lane1Retriever
from .reranker import get_medcpt_reranker


def debug_rerank_samples(
    n_samples: int = 5,
    qrels_file: str = "qrels.json"
):
    """
    Debug reranker behavior on sample queries.
    
    Shows before/after rankings and highlights issues.
    """
    # Load data
    with open(DATA_ROOT / qrels_file) as f:
        qrels = json.load(f)
    
    with open(DATA_ROOT / "eval_queries.jsonl") as f:
        queries = [json.loads(l) for l in f if "Q1" in json.loads(l)["query_type"]]
    
    with open(TRAIN_JSONL) as f:
        train_cases = {}
        for line in f:
            obj = json.loads(line)
            train_cases[obj["case_id"]] = obj
    
    # Initialize
    lane1 = Lane1Retriever()
    reranker = get_medcpt_reranker()
    
    print("="*80)
    print("RERANKER DEBUG ANALYSIS")
    print("="*80)
    
    for i, q in enumerate(queries[:n_samples]):
        case_id = q["case_id"]
        query_text = q["text"]
        relevant = set(qrels.get(case_id, []))
        
        print(f"\n{'='*80}")
        print(f"Query {i+1}: {case_id}")
        print(f"Query text: {query_text[:100]}...")
        print(f"Relevant docs: {len(relevant)}")
        
        # Get hybrid results
        hybrid_results = lane1.retrieve_hybrid(query_text, top_k=10)
        
        # Prepare for reranking
        candidates = []
        for cid, score in hybrid_results:
            if cid in train_cases:
                doc_text = train_cases[cid].get("case_text", "")[:512]
                candidates.append((cid, doc_text, score))
        
        # Rerank
        reranked = reranker.rerank(query_text, candidates, top_k=10)
        
        # Compare rankings
        print(f"\n{'BEFORE (Hybrid)':<40} {'AFTER (Rerank)':<40}")
        print("-"*80)
        
        for rank in range(min(10, len(hybrid_results))):
            before_id = hybrid_results[rank][0] if rank < len(hybrid_results) else "-"
            before_rel = "✓" if before_id in relevant else "✗"
            before_score = hybrid_results[rank][1] if rank < len(hybrid_results) else 0
            
            after_id = reranked[rank][0] if rank < len(reranked) else "-"
            after_rel = "✓" if after_id in relevant else "✗"
            after_score = reranked[rank][2] if rank < len(reranked) else 0
            
            print(f"{rank+1}. {before_rel} {before_id[:15]:<15} ({before_score:.3f})  →  {after_rel} {after_id[:15]:<15} ({after_score:.3f})")
        
        # Calculate metrics change
        before_rel_at_5 = sum(1 for cid, _ in hybrid_results[:5] if cid in relevant)
        after_rel_at_5 = sum(1 for cid, _, _ in reranked[:5] if cid in relevant)
        
        print(f"\nRelevant@5: {before_rel_at_5} → {after_rel_at_5} ({'↑' if after_rel_at_5 > before_rel_at_5 else '↓' if after_rel_at_5 < before_rel_at_5 else '='})")
        
        # Check first position
        before_first_rel = hybrid_results[0][0] in relevant if hybrid_results else False
        after_first_rel = reranked[0][0] in relevant if reranked else False
        print(f"First position relevant: {before_first_rel} → {after_first_rel}")


def analyze_reranker_issue():
    """
    Hypothesis: MedCPT may have "domain shift" on Leishmaniasis data.
    
    MedCPT is trained on general medical Q&A, not case reports.
    """
    print("\n" + "="*80)
    print("HYPOTHESIS: Domain Shift Analysis")
    print("="*80)
    
    # Check reranker behavior
    reranker = get_medcpt_reranker()
    
    # Test with Leishmaniasis-specific queries
    test_cases = [
        {
            "query": "What causes visceral leishmaniasis with hepatosplenomegaly?",
            "docs": [
                "Patient presented with fever, weight loss, and hepatosplenomegaly consistent with VL.",
                "Leishmaniasis is caused by parasites transmitted by sandflies.",
                "The patient had malaria symptoms including fever and chills."
            ]
        },
        {
            "query": "Cutaneous leishmaniasis ulcer on the arm",
            "docs": [
                "Skin lesion with raised borders on the forearm, biopsy confirmed Leishmania.",
                "The patient had a chronic non-healing ulcer for 3 months.",
                "Treatment included intralesional antimonials."
            ]
        }
    ]
    
    for case in test_cases:
        print(f"\nQuery: {case['query']}")
        scores = reranker.score(case["query"], case["docs"])
        for doc, score in zip(case["docs"], scores):
            print(f"  {score:+.3f}: {doc[:60]}...")


if __name__ == "__main__":
    debug_rerank_samples(n_samples=5)
    analyze_reranker_issue()
