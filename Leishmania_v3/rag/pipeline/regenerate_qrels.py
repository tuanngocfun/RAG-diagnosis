#!/usr/bin/env python3
"""
Regenerate QRELs from 230 Strict Dataset

Uses updated qrels_generator.py with:
- exclude_generic_cl: Excludes 'CL', 'VL' generic labels when specific diseases present
- top_k: Caps max relevant docs per query to prevent inflation

Usage:
    python -m rag.pipeline.regenerate_qrels
"""
import json
from pathlib import Path

from .config import TRAIN_JSONL, TEST_JSONL, ENTITY_LINKS, DATA_ROOT, SPLIT_DIR
from .qrels_generator import generate_qrels, load_entity_links


def regenerate_qrels(
    output_name: str = "qrels_230strict.json",
    top_k: int = 20,
    exclude_generic_cl: bool = True
) -> Path:
    """
    Regenerate QRELs using updated generator with fixes.
    
    Args:
        output_name: Output filename
        top_k: Max relevant docs per query
        exclude_generic_cl: Exclude generic CL/VL labels
    
    Returns:
        Path to generated qrels file
    """
    # Load train and test cases
    print(f"Loading train cases from: {TRAIN_JSONL}")
    with open(TRAIN_JSONL) as f:
        train_cases = [json.loads(l) for l in f]
    train_ids = [c["case_id"] for c in train_cases]
    
    print(f"Loading test cases from: {TEST_JSONL}")
    with open(TEST_JSONL) as f:
        test_cases = [json.loads(l) for l in f]
    test_ids = [c["case_id"] for c in test_cases]
    
    print(f"  Train: {len(train_ids)} cases")
    print(f"  Test: {len(test_ids)} cases")
    
    # Load entity links
    print(f"Loading entity links from: {ENTITY_LINKS}")
    all_entities = load_entity_links(ENTITY_LINKS)
    print(f"  Loaded entities for {len(all_entities)} cases")
    
    # Generate QRELs with fixes
    print(f"\nGenerating QRELs with:")
    print(f"  - exclude_generic_cl={exclude_generic_cl}")
    print(f"  - top_k={top_k}")
    
    qrels = generate_qrels(
        test_case_ids=test_ids,
        train_case_ids=train_ids,
        entity_links_path=ENTITY_LINKS,  # Pass path, not loaded dict
        top_k=top_k,
        exclude_generic_cl=exclude_generic_cl
    )
    
    # Save to SPLIT_DIR (verified dataset)
    output_path = SPLIT_DIR / output_name
    with open(output_path, "w") as f:
        json.dump(qrels, f, indent=2)
    
    # Statistics
    total_relevant = sum(len(docs) for docs in qrels.values())
    grades = {}
    for docs in qrels.values():
        for grade in docs.values():
            grades[grade] = grades.get(grade, 0) + 1
    
    print(f"\n{'='*60}")
    print(f"QRELS REGENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"Output: {output_path}")
    print(f"Queries: {len(qrels)}")
    print(f"Total relevant pairs: {total_relevant}")
    print(f"Avg relevant/query: {total_relevant/len(qrels):.1f}")
    print(f"Grade distribution: {grades}")
    print(f"{'='*60}")
    
    return output_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Regenerate QRELs with fixes")
    parser.add_argument("--output", default="qrels_230strict.json", help="Output filename")
    parser.add_argument("--top-k", type=int, default=20, help="Max relevant docs per query")
    parser.add_argument("--no-exclude-cl", action="store_true", help="Disable CL exclusion")
    
    args = parser.parse_args()
    
    regenerate_qrels(
        output_name=args.output,
        top_k=args.top_k,
        exclude_generic_cl=not args.no_exclude_cl
    )
