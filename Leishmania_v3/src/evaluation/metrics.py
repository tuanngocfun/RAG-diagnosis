#!/usr/bin/env python3
"""
Evaluation Metrics for KG and RAG Systems

Provides metrics for:
1. Entity Extraction (Precision, Recall, F1)
2. KG Quality (Coverage, Density)
3. RAG Performance (MRR, Precision@K, Recall@K)
"""

import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np


def precision_recall_f1(predicted: Set, gold: Set) -> Tuple[float, float, float]:
    """Calculate precision, recall, F1."""
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    
    true_positives = len(predicted & gold)
    
    precision = true_positives / len(predicted) if predicted else 0
    recall = true_positives / len(gold) if gold else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1


def entity_extraction_metrics(
    predicted_entities: List[Dict],
    gold_entities: List[Dict],
    match_on: str = "name"
) -> Dict:
    """
    Evaluate entity extraction quality.
    
    predicted_entities: List of {"name": ..., "entity_type": ...}
    gold_entities: Same format
    """
    # Group by type
    pred_by_type = defaultdict(set)
    gold_by_type = defaultdict(set)
    
    for e in predicted_entities:
        pred_by_type[e.get("entity_type", "Other")].add(e.get(match_on, "").lower())
    
    for e in gold_entities:
        gold_by_type[e.get("entity_type", "Other")].add(e.get(match_on, "").lower())
    
    all_types = set(pred_by_type.keys()) | set(gold_by_type.keys())
    
    results = {"by_type": {}, "overall": {}}
    
    all_pred = set()
    all_gold = set()
    
    for entity_type in all_types:
        pred = pred_by_type[entity_type]
        gold = gold_by_type[entity_type]
        
        all_pred.update(pred)
        all_gold.update(gold)
        
        p, r, f = precision_recall_f1(pred, gold)
        results["by_type"][entity_type] = {
            "precision": p,
            "recall": r,
            "f1": f,
            "predicted_count": len(pred),
            "gold_count": len(gold)
        }
    
    # Overall
    p, r, f = precision_recall_f1(all_pred, all_gold)
    results["overall"] = {
        "precision": p,
        "recall": r,
        "f1": f,
        "predicted_count": len(all_pred),
        "gold_count": len(all_gold)
    }
    
    return results


def kg_quality_metrics(kg: Dict, case_entity_links: List[Dict]) -> Dict:
    """
    Evaluate Knowledge Graph quality.
    
    kg: KG with "entities" list
    case_entity_links: List of case-entity mappings
    """
    entities = kg.get("entities", [])
    
    # Entity distribution
    type_counts = defaultdict(int)
    for e in entities:
        type_counts[e.get("entity_type", "Other")] += 1
    
    # Coverage: how many unique cases have entities
    cases_with_entities = set(link["case_id"] for link in case_entity_links)
    
    # Density: avg entities per case
    entities_per_case = defaultdict(int)
    for link in case_entity_links:
        entities_per_case[link["case_id"]] += 1
    
    avg_density = np.mean(list(entities_per_case.values())) if entities_per_case else 0
    
    return {
        "total_entities": len(entities),
        "entity_type_distribution": dict(type_counts),
        "cases_with_entities": len(cases_with_entities),
        "avg_entities_per_case": avg_density,
        "min_entities_per_case": min(entities_per_case.values()) if entities_per_case else 0,
        "max_entities_per_case": max(entities_per_case.values()) if entities_per_case else 0
    }


def retrieval_metrics(
    results: List[Tuple[str, float]],
    relevant: Set[str],
    k: int = 5
) -> Dict:
    """
    Evaluate single retrieval.
    
    results: [(case_id, score), ...]
    relevant: Set of relevant case_ids
    k: Cutoff for metrics
    """
    retrieved = [r[0] for r in results[:k]]
    
    # Precision@K
    hits = len(set(retrieved) & relevant)
    precision_at_k = hits / k if k > 0 else 0
    
    # Recall@K
    recall_at_k = hits / len(relevant) if relevant else 0
    
    # MRR
    mrr = 0
    for rank, case_id in enumerate(retrieved, 1):
        if case_id in relevant:
            mrr = 1.0 / rank
            break
    
    # Average Precision
    precisions = []
    hits_so_far = 0
    for i, case_id in enumerate(retrieved, 1):
        if case_id in relevant:
            hits_so_far += 1
            precisions.append(hits_so_far / i)
    
    ap = np.mean(precisions) if precisions else 0
    
    return {
        "precision@k": precision_at_k,
        "recall@k": recall_at_k,
        "mrr": mrr,
        "average_precision": ap,
        "k": k
    }


def batch_retrieval_metrics(
    all_results: List[List[Tuple[str, float]]],
    all_relevant: List[Set[str]],
    k: int = 5
) -> Dict:
    """Aggregate retrieval metrics over multiple queries."""
    metrics = []
    
    for results, relevant in zip(all_results, all_relevant):
        m = retrieval_metrics(results, relevant, k)
        metrics.append(m)
    
    return {
        "mean_precision@k": np.mean([m["precision@k"] for m in metrics]),
        "mean_recall@k": np.mean([m["recall@k"] for m in metrics]),
        "mean_mrr": np.mean([m["mrr"] for m in metrics]),
        "mean_average_precision": np.mean([m["average_precision"] for m in metrics]),
        "num_queries": len(metrics),
        "k": k
    }


def compare_methods(method_results: Dict[str, Dict]) -> Dict:
    """
    Compare multiple methods.
    
    method_results: {"method_name": {"metric": value, ...}, ...}
    """
    comparison = {"methods": {}, "rankings": {}}
    
    all_metrics = set()
    for name, metrics in method_results.items():
        comparison["methods"][name] = metrics
        all_metrics.update(metrics.keys())
    
    # Rank methods per metric
    for metric in all_metrics:
        if metric in ["k", "num_queries"]:
            continue
        
        values = []
        for name, metrics in method_results.items():
            if metric in metrics:
                values.append((name, metrics[metric]))
        
        # Sort descending (higher is better for most metrics)
        sorted_methods = sorted(values, key=lambda x: -x[1])
        comparison["rankings"][metric] = [m[0] for m in sorted_methods]
    
    return comparison


if __name__ == "__main__":
    # Demo
    print("Evaluation Metrics Module")
    print("Import and use: from metrics import entity_extraction_metrics, kg_quality_metrics")
