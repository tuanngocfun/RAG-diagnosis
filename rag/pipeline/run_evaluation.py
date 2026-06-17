"""
Evaluation Runner - Creates proper run artifacts

Outputs per ChatGPT 5.2 spec:
- runs/<run_id>/run_config.json
- runs/<run_id>/queries.json  
- runs/<run_id>/retrieval.jsonl
- runs/<run_id>/metrics_per_query.csv
- runs/<run_id>/summary.json
"""
import json
import csv
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .config import (
    DATA_ROOT, TRAIN_JSONL, RUNS_DIR,
    EVAL_CONFIG, CHUNK_CONFIG
)
from .evaluator import evaluate_retrieval, RetrievalResults
from .retriever import Lane1Retriever, Lane2Retriever, rrf_fusion
from .reranker import get_medcpt_reranker


@dataclass
class RunConfig:
    """Configuration for a single evaluation run."""
    run_id: str
    qrels_file: str
    query_type: str
    retriever_method: str  # "bm25" | "e5" | "hybrid" | "2lane"
    rerank: bool
    k_values: List[int]
    created_at: str


def create_run_id(prefix: str = "run") -> str:
    """Create unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def run_evaluation(
    qrels_file: str = "qrels_grade3.json",
    query_type: str = "Q1_symptom_only",
    method: str = "hybrid",
    rerank: bool = False,
    k_values: List[int] = [5, 10],
    run_id: Optional[str] = None
) -> Path:
    """
    Run full evaluation and save artifacts.
    
    Args:
        qrels_file: QRELs filename in DATA_ROOT
        query_type: Query type filter
        method: Retrieval method
        rerank: Whether to use MedCPT reranking
        k_values: K values for metrics
        run_id: Optional run ID (auto-generated if None)
    
    Returns:
        Path to run directory
    """
    # Setup
    if run_id is None:
        run_id = create_run_id(method)
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    with open(DATA_ROOT / qrels_file) as f:
        qrels = json.load(f)
    
    with open(DATA_ROOT / "eval_queries.jsonl") as f:
        all_queries = [json.loads(l) for l in f]
    
    queries = [q for q in all_queries 
               if q["query_type"] == query_type and q["case_id"] in qrels]
    
    with open(TRAIN_JSONL) as f:
        train_cases = {json.loads(l)["case_id"]: json.loads(l) for l in f}
    
    # Save config
    config = RunConfig(
        run_id=run_id,
        qrels_file=qrels_file,
        query_type=query_type,
        retriever_method=method,
        rerank=rerank,
        k_values=k_values,
        created_at=datetime.now().isoformat()
    )
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Save queries
    with open(run_dir / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    # Initialize retrievers
    lane1 = Lane1Retriever()
    reranker = get_medcpt_reranker() if rerank else None
    
    # Run retrieval
    all_retrieved = {}
    retrieval_records = []
    
    for q in queries:
        qid = q["case_id"]
        query_text = q["text"]
        
        if not query_text:
            continue
        
        # Get results based on method
        if method == "bm25":
            results = lane1.retrieve_bm25(query_text, top_k=20)
        elif method == "e5":
            results = lane1.retrieve_e5(query_text, top_k=20)
        elif method == "hybrid":
            results = lane1.retrieve_hybrid(query_text, top_k=20)
        else:
            results = lane1.retrieve_hybrid(query_text, top_k=20)
        
        stage = method
        
        # Rerank if requested
        if rerank and reranker:
            candidates = []
            for case_id, score in results:
                if case_id in train_cases:
                    doc_text = train_cases[case_id].get("case_text", "")[:512]
                    candidates.append((case_id, doc_text, score))
            
            reranked = reranker.rerank(query_text, candidates, top_k=20)
            results = [(c, s) for c, _, s in reranked]
            stage = f"{method}+rerank"
        
        # Store
        all_retrieved[qid] = [c for c, _ in results]
        
        # Build contexts for retrieval.jsonl
        contexts = []
        for case_id, score in results[:10]:
            if case_id in train_cases:
                contexts.append({
                    "doc_id": case_id,
                    "score": float(score),
                    "text": train_cases[case_id].get("case_text", "")[:2000],  # Increased from 500 for medical precision
                    "meta": {"has_images": len(train_cases[case_id].get("images", [])) > 0}
                })
        
        retrieval_records.append({
            "qid": qid,
            "query": query_text,
            "contexts": contexts,
            "stage": stage
        })
    
    # Save retrieval.jsonl
    with open(run_dir / "retrieval.jsonl", "w") as f:
        for rec in retrieval_records:
            f.write(json.dumps(rec) + "\n")
    
    # Evaluate
    results = evaluate_retrieval(all_retrieved, qrels, k_values, return_per_query=True)
    
    # Save per-query metrics
    with open(run_dir / "metrics_per_query.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["qid"] + [f"recall@{k}" for k in k_values] + \
                 [f"ndcg@{k}" for k in k_values] + \
                 [f"precision@{k}" for k in k_values] + ["mrr", "ap"]
        writer.writerow(header)
        
        for qid, metrics in (results.per_query_metrics or {}).items():
            row = [qid]
            for k in k_values:
                row.append(f"{metrics.get(f'recall@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'ndcg@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'precision@{k}', 0):.4f}")
            row.append(f"{metrics.get('mrr', 0):.4f}")
            row.append(f"{metrics.get('ap', 0):.4f}")
            writer.writerow(row)
    
    # Save summary
    summary = {
        "run_id": run_id,
        "n_queries": len(queries),
        "n_retrieved": len(all_retrieved),
        "metrics": {
            "recall": {f"@{k}": results.recall_at_k[k] for k in k_values},
            "ndcg": {f"@{k}": results.ndcg_at_k[k] for k in k_values},
            "precision": {f"@{k}": results.precision_at_k[k] for k in k_values},
            "mrr": results.mrr,
            "map": results.map_score
        }
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Run saved to {run_dir}")
    print(f"  Queries: {len(queries)}")
    print(f"  nDCG@5: {results.ndcg_at_k[5]:.4f}")
    print(f"  MRR: {results.mrr:.4f}")
    
    return run_dir


if __name__ == "__main__":
    # Run phase3 evaluation
    run_evaluation(
        qrels_file="qrels_grade3.json",
        query_type="Q1_symptom_only",
        method="hybrid",
        rerank=False,
        run_id="phase3_hybrid"
    )
    
    run_evaluation(
        qrels_file="qrels_grade3.json", 
        query_type="Q1_symptom_only",
        method="hybrid",
        rerank=True,
        run_id="phase3_hybrid_rerank"
    )
