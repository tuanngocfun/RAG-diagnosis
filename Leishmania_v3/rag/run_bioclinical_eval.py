"""
Run evaluation with BioClinical-ModernBERT retriever.

Compares:
1. BioClinical-ModernBERT + BM25 (new medical retriever)
2. E5-large-v2 + BM25 (old baseline)
3. No-RAG baseline

Creates new run folder: runs/bioclinical_medical_full
"""
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import (
    SPLIT_DIR, RUNS_DIR, TRAIN_JSONL, DATA_ROOT, DATASET_VERSION
)
from pipeline.medical_retriever import MedicalRetriever
from pipeline.generators.gemini import GeminiGenerator


def load_test_queries(query_types: List[str] = None) -> List[Dict]:
    """Load test queries from eval_queries_test.jsonl."""
    queries_file = SPLIT_DIR / "eval_queries_test.jsonl"
    
    queries = []
    with open(queries_file) as f:
        for line in f:
            q = json.loads(line)
            if query_types is None or q.get("query_type") in query_types:
                queries.append(q)
    
    return queries


def run_bioclinical_evaluation(
    run_id: str = "bioclinical_medical_full",
    query_types: List[str] = None,
    top_k: int = 10,
    method: str = "hybrid"
) -> Path:
    """
    Run full evaluation with BioClinical-ModernBERT retriever.
    
    Args:
        run_id: Identifier for this run
        query_types: Which query types to evaluate
        top_k: Number of retrieved contexts
        method: "dense" or "hybrid"
    
    Returns:
        Path to run directory
    """
    # Default query types
    if query_types is None:
        query_types = ["Q1_diagnosis", "Q1_Q3_multimodal_diagnosis"]
    
    # Create run directory
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"BIOCLINICAL RETRIEVER EVALUATION")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Query types: {query_types}")
    print(f"Method: {method}")
    
    # Initialize retriever and generator
    retriever = MedicalRetriever()
    generator = GeminiGenerator()
    print(f"✓ BioClinical Retriever initialized")
    print(f"✓ Generator: {generator.model_name}")
    
    # Load training cases for context enrichment
    with open(TRAIN_JSONL) as f:
        train_cases = {json.loads(l)["case_id"]: json.loads(l) for l in f}
    
    # Load queries
    queries = load_test_queries(query_types)
    print(f"Loaded {len(queries)} queries")
    
    # Run retrieval and generation
    retrieval_results = []
    answer_results = []
    
    for i, q in enumerate(queries):
        case_id = q["case_id"]
        query_type = q.get("query_type", "Q1_diagnosis")
        qid = f"{case_id}::{query_type}"
        query_text = q.get("formatted_query") or q.get("question", "")
        ground_truth = q.get("ground_truth", {})
        query_images = q.get("query_images", [])
        
        # Retrieve with BioClinical
        if method == "hybrid":
            results = retriever.retrieve_hybrid(query_text, top_k=top_k)
        else:
            results = retriever.retrieve_dense(query_text, top_k=top_k)
        
        # Build contexts
        contexts = []
        for case_id, score in results:
            if case_id in train_cases:
                case = train_cases[case_id]
                contexts.append({
                    "doc_id": case_id,
                    "text": case.get("case_text", "")[:2000],
                    "score": float(score),
                    "diagnosis": case.get("diagnosis", ""),
                    "diagnosis_type": case.get("diagnosis_type", "")
                })
        
        # Save retrieval result
        retrieval_results.append({
            "qid": qid,
            "query": query_text,
            "contexts": contexts,
            "query_images": query_images,
            "ground_truth": ground_truth
        })
        
        # Generate answer
        answer = generator.generate(
            query_text,
            contexts,
            image_paths=query_images[:5] if query_images else None
        )
        
        answer_results.append({
            "qid": qid,
            "query": query_text,
            "contexts": contexts,
            "answer": answer,
            "model_name": generator.model_name,
            "query_images": query_images[:5] if query_images else [],
            "ground_truth": ground_truth
        })
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(queries)}")
    
    # Save results
    retrieval_file = run_dir / "retrieval.jsonl"
    with open(retrieval_file, "w") as f:
        for r in retrieval_results:
            f.write(json.dumps(r) + "\n")
    
    answers_file = run_dir / "answers.jsonl"
    with open(answers_file, "w") as f:
        for a in answer_results:
            f.write(json.dumps(a) + "\n")
    
    # Save config
    config = {
        "run_id": run_id,
        "retriever": "BioClinical-ModernBERT",
        "collection": retriever.collection_name,
        "method": method,
        "top_k": top_k,
        "query_types": query_types,
        "n_queries": len(queries),
        "timestamp": datetime.now().isoformat()
    }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print(f"\n✓ Saved {len(queries)} results to {run_dir}")
    print(f"  - retrieval.jsonl: {len(retrieval_results)} queries")
    print(f"  - answers.jsonl: {len(answer_results)} answers")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="bioclinical_medical_full")
    parser.add_argument("--method", default="hybrid", choices=["dense", "hybrid"])
    parser.add_argument("--query-types", nargs="+", 
                       default=["Q1_diagnosis", "Q1_Q3_multimodal_diagnosis"])
    args = parser.parse_args()
    
    run_bioclinical_evaluation(
        run_id=args.run_id,
        method=args.method,
        query_types=args.query_types
    )
