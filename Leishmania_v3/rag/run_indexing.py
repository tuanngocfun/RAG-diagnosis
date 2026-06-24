#!/usr/bin/env python3
"""
Run Indexing Pipeline

Usage:
    python run_indexing.py [--strategy fixed|section|semantic] [--skip-e5] [--skip-bm25]

Requirements:
    pip install torch sentence-transformers qdrant-client python-dotenv rank_bm25 tqdm
"""
import sys
import argparse
from pathlib import Path

# Add rag directory to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import TRAIN_JSONL, DATA_ROOT, COLLECTIONS
from pipeline.chunker import chunk_cases
from pipeline.encoders.bm25 import get_bm25_retriever
import json


def run_bm25_indexing():
    """Index train data with BM25."""
    print("="*50)
    print("BM25 Indexing")
    print("="*50)
    
    # Load train cases
    with open(TRAIN_JSONL) as f:
        cases = [json.loads(line) for line in f]
    print(f"Loaded {len(cases)} train cases")
    
    # Fit BM25
    bm25 = get_bm25_retriever()
    bm25.fit_from_cases(cases)
    
    # Save index
    bm25_path = DATA_ROOT / "bm25_index.json"
    bm25.save(bm25_path)
    print(f"✓ Saved BM25 index to {bm25_path}")
    
    return len(cases)


def run_e5_indexing(strategy: str = "fixed"):
    """Index train data with E5 to Qdrant."""
    print("="*50)
    print(f"E5 Indexing (strategy={strategy})")
    print("="*50)
    
    from pipeline.indexer import index_train_data
    
    stats = index_train_data(strategy=strategy)
    
    print("\nIndexing Complete!")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="RAG Indexing Pipeline")
    parser.add_argument(
        "--strategy",
        choices=["fixed", "section", "semantic"],
        default="fixed",
        help="Chunking strategy"
    )
    parser.add_argument("--skip-e5", action="store_true", help="Skip E5 indexing")
    parser.add_argument("--skip-bm25", action="store_true", help="Skip BM25 indexing")
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("RAG Indexing Pipeline - Train Data Only")
    print("="*60 + "\n")
    
    results = {}
    
    if not args.skip_bm25:
        results["bm25_cases"] = run_bm25_indexing()
    
    if not args.skip_e5:
        results["e5_stats"] = run_e5_indexing(args.strategy)
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
