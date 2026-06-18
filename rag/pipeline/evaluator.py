"""
Evaluator for RAG Retrieval Pipeline

Implements Q1 journal standard metrics:
- Recall@K (primary - safety)
- nDCG@K (primary - graded relevance)
- Precision@K (secondary - noise control)
- MRR (secondary - first relevant)
- MAP (secondary - multiple relevant)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class RetrievalResults:
    """Container for retrieval evaluation results."""
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    precision_at_k: Dict[int, float]
    mrr: float
    map_score: float
    per_query_metrics: Optional[Dict] = None


def dcg_at_k(relevances: List[int], k: int) -> float:
    """
    Calculate Discounted Cumulative Gain at K.
    
    DCG@K = sum_{i=1}^{K} (2^rel_i - 1) / log2(i + 1)
    """
    relevances = relevances[:k]
    if not relevances:
        return 0.0
    
    dcg = 0.0
    for i, rel in enumerate(relevances):
        dcg += (2 ** rel - 1) / np.log2(i + 2)  # i+2 because i is 0-indexed
    
    return dcg


def ndcg_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int
) -> float:
    """
    Calculate Normalized DCG at K.
    
    Args:
        retrieved: List of retrieved document IDs (ranked)
        qrels: Dict mapping doc_id to relevance grade
        k: Cutoff
    
    Returns:
        nDCG@K score (0-1)
    """
    # Get relevances for retrieved docs
    relevances = [qrels.get(doc_id, 0) for doc_id in retrieved[:k]]
    
    dcg = dcg_at_k(relevances, k)
    
    # Ideal DCG: sort all qrels by relevance
    ideal_relevances = sorted(qrels.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal_relevances, k)
    
    if idcg == 0:
        return 0.0
    
    return dcg / idcg


def recall_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int,
    min_grade: int = 1
) -> float:
    """
    Calculate Recall at K.
    
    Args:
        retrieved: List of retrieved document IDs
        qrels: Dict mapping doc_id to relevance grade
        k: Cutoff
        min_grade: Minimum grade to consider relevant
    
    Returns:
        Recall@K score (0-1)
    """
    relevant_docs = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    
    if not relevant_docs:
        return 0.0
    
    retrieved_relevant = set(retrieved[:k]) & relevant_docs
    
    return len(retrieved_relevant) / len(relevant_docs)


def precision_at_k(
    retrieved: List[str],
    qrels: Dict[str, int],
    k: int,
    min_grade: int = 1
) -> float:
    """
    Calculate Precision at K.
    
    Args:
        retrieved: List of retrieved document IDs
        qrels: Dict mapping doc_id to relevance grade
        k: Cutoff
        min_grade: Minimum grade to consider relevant
    
    Returns:
        Precision@K score (0-1)
    """
    relevant_docs = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    retrieved_k = retrieved[:k]
    
    if not retrieved_k:
        return 0.0
    
    retrieved_relevant = set(retrieved_k) & relevant_docs
    
    return len(retrieved_relevant) / len(retrieved_k)


def mrr(
    retrieved: List[str],
    qrels: Dict[str, int],
    min_grade: int = 1
) -> float:
    """
    Calculate Mean Reciprocal Rank.
    
    Args:
        retrieved: List of retrieved document IDs
        qrels: Dict mapping doc_id to relevance grade
        min_grade: Minimum grade to consider relevant
    
    Returns:
        MRR score (0-1)
    """
    relevant_docs = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_docs:
            return 1.0 / (i + 1)
    
    return 0.0


def average_precision(
    retrieved: List[str],
    qrels: Dict[str, int],
    min_grade: int = 1
) -> float:
    """
    Calculate Average Precision.
    
    Args:
        retrieved: List of retrieved document IDs
        qrels: Dict mapping doc_id to relevance grade
        min_grade: Minimum grade to consider relevant
    
    Returns:
        AP score (0-1)
    """
    relevant_docs = {doc_id for doc_id, grade in qrels.items() if grade >= min_grade}
    
    if not relevant_docs:
        return 0.0
    
    num_relevant_seen = 0
    precision_sum = 0.0
    
    for i, doc_id in enumerate(retrieved):
        if doc_id in relevant_docs:
            num_relevant_seen += 1
            precision_at_i = num_relevant_seen / (i + 1)
            precision_sum += precision_at_i
    
    return precision_sum / len(relevant_docs)


def evaluate_retrieval(
    all_retrieved: Dict[str, List[str]],
    all_qrels: Dict[str, Dict[str, int]],
    k_values: List[int] = [5, 10],
    min_grade: int = 1,
    return_per_query: bool = False
) -> RetrievalResults:
    """
    Evaluate retrieval results against qrels.
    
    Args:
        all_retrieved: Dict mapping query_id to list of retrieved doc_ids (ranked)
        all_qrels: Dict mapping query_id to {doc_id: grade}
        k_values: List of K values for metrics
        min_grade: Minimum grade for binary relevance
        return_per_query: Whether to return per-query metrics
    
    Returns:
        RetrievalResults with all metrics
    """
    # Initialize accumulators
    recall_scores = {k: [] for k in k_values}
    ndcg_scores = {k: [] for k in k_values}
    precision_scores = {k: [] for k in k_values}
    mrr_scores = []
    ap_scores = []
    
    per_query = {} if return_per_query else None
    
    for query_id, retrieved in all_retrieved.items():
        qrels = all_qrels.get(query_id, {})
        
        if not qrels:
            continue
        
        # Calculate metrics for this query
        query_mrr = mrr(retrieved, qrels, min_grade)
        query_ap = average_precision(retrieved, qrels, min_grade)
        
        mrr_scores.append(query_mrr)
        ap_scores.append(query_ap)
        
        query_metrics = {"mrr": query_mrr, "ap": query_ap} if return_per_query else None
        
        for k in k_values:
            r = recall_at_k(retrieved, qrels, k, min_grade)
            n = ndcg_at_k(retrieved, qrels, k)
            p = precision_at_k(retrieved, qrels, k, min_grade)
            
            recall_scores[k].append(r)
            ndcg_scores[k].append(n)
            precision_scores[k].append(p)
            
            if query_metrics:
                query_metrics[f"recall@{k}"] = r
                query_metrics[f"ndcg@{k}"] = n
                query_metrics[f"precision@{k}"] = p
        
        if per_query is not None:
            per_query[query_id] = query_metrics
    
    # Compute means
    results = RetrievalResults(
        recall_at_k={k: np.mean(scores) if scores else 0.0 for k, scores in recall_scores.items()},
        ndcg_at_k={k: np.mean(scores) if scores else 0.0 for k, scores in ndcg_scores.items()},
        precision_at_k={k: np.mean(scores) if scores else 0.0 for k, scores in precision_scores.items()},
        mrr=np.mean(mrr_scores) if mrr_scores else 0.0,
        map_score=np.mean(ap_scores) if ap_scores else 0.0,
        per_query_metrics=per_query
    )
    
    return results


def print_results(results: RetrievalResults, method_name: str = "") -> None:
    """Print retrieval results in formatted table."""
    print(f"\n{'='*50}")
    print(f"Retrieval Results: {method_name}")
    print(f"{'='*50}")
    
    print(f"\n{'Metric':<20} {'Value':>10}")
    print("-" * 32)
    
    for k, v in results.recall_at_k.items():
        print(f"Recall@{k:<13} {v:>10.4f}")
    
    for k, v in results.ndcg_at_k.items():
        print(f"nDCG@{k:<15} {v:>10.4f}")
    
    for k, v in results.precision_at_k.items():
        print(f"Precision@{k:<10} {v:>10.4f}")
    
    print(f"{'MRR':<20} {results.mrr:>10.4f}")
    print(f"{'MAP':<20} {results.map_score:>10.4f}")


if __name__ == "__main__":
    # Test evaluation
    qrels = {
        "q1": {"doc1": 3, "doc2": 2, "doc3": 1},
        "q2": {"doc4": 3, "doc5": 1},
    }
    
    retrieved = {
        "q1": ["doc1", "doc3", "doc5", "doc2", "doc6"],
        "q2": ["doc6", "doc4", "doc5", "doc7", "doc8"],
    }
    
    results = evaluate_retrieval(retrieved, qrels, k_values=[3, 5])
    print_results(results, "Test Run")
