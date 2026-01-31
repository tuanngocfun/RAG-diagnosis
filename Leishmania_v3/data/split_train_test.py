#!/usr/bin/env python3
"""
Train/Test Split for Leishmaniasis RAG Pipeline (Q1 Journal Quality)

Implements article-based splitting with case-text hash deduplication to prevent
data leakage. Uses 230 strict Leishmaniasis cases with proper 80/20 split.

Key features:
- Article-based grouping: Cases from same article stay together
- Hash-based deduplication: Removes near-duplicate case texts  
- Reproducible split: Fixed random seed
- Outputs: train.jsonl, test.jsonl + statistics

Usage:
    python split_train_test.py --strict-only --seed 42 --test-ratio 0.2
"""

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple


def compute_text_hash(text: str, n_chars: int = 500) -> str:
    """Compute hash of first n_chars for deduplication."""
    normalized = " ".join(text.lower().split())[:n_chars]
    return hashlib.sha256(normalized.encode()).hexdigest()[:16]


def load_jsonl(path: Path) -> List[Dict]:
    """Load JSONL file."""
    cases = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                cases.append(json.loads(line))
    return cases


def save_jsonl(cases: List[Dict], path: Path):
    """Save cases to JSONL."""
    with open(path, 'w', encoding='utf-8') as f:
        for case in cases:
            f.write(json.dumps(case, ensure_ascii=False) + '\n')


def deduplicate_by_text_hash(
    cases: List[Dict], 
    text_field: str = 'case_text'
) -> Tuple[List[Dict], int]:
    """
    Remove cases with duplicate text hashes.
    
    Returns:
        Tuple of (deduplicated cases, count of duplicates removed)
    """
    seen_hashes: Set[str] = set()
    unique_cases = []
    duplicates = 0
    
    for case in cases:
        text = case.get(text_field, '')
        if not text:
            continue
        
        text_hash = compute_text_hash(text)
        if text_hash not in seen_hashes:
            seen_hashes.add(text_hash)
            unique_cases.append(case)
        else:
            duplicates += 1
    
    return unique_cases, duplicates


def group_by_article(cases: List[Dict]) -> Dict[str, List[Dict]]:
    """Group cases by article_id."""
    article_groups: Dict[str, List[Dict]] = defaultdict(list)
    for case in cases:
        article_id = case.get('article_id', case['case_id'].rsplit('_', 1)[0])
        article_groups[article_id].append(case)
    return dict(article_groups)


def split_by_articles(
    article_groups: Dict[str, List[Dict]],
    test_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict]]:
    """
    Split cases by article_id to prevent leakage.
    
    All cases from the same article go to either train or test.
    """
    random.seed(seed)
    
    article_ids = list(article_groups.keys())
    random.shuffle(article_ids)
    
    total_cases = sum(len(cases) for cases in article_groups.values())
    target_test = int(total_cases * test_ratio)
    
    train_cases = []
    test_cases = []
    test_count = 0
    
    for article_id in article_ids:
        cases = article_groups[article_id]
        if test_count < target_test:
            test_cases.extend(cases)
            test_count += len(cases)
        else:
            train_cases.extend(cases)
    
    return train_cases, test_cases


def get_leish_type_distribution(cases: List[Dict]) -> Dict[str, int]:
    """Get distribution of leishmaniasis types."""
    dist: Dict[str, int] = defaultdict(int)
    for case in cases:
        leish_type = case.get('leish_type', 'Unknown')
        dist[leish_type] += 1
    return dict(dist)


def print_statistics(
    train_cases: List[Dict], 
    test_cases: List[Dict],
    duplicates_removed: int
):
    """Print split statistics."""
    total = len(train_cases) + len(test_cases)
    
    print("\n" + "="*60)
    print("TRAIN/TEST SPLIT STATISTICS")
    print("="*60)
    print(f"Total unique cases after deduplication: {total}")
    print(f"Duplicates removed: {duplicates_removed}")
    print(f"\nTrain set: {len(train_cases)} ({100*len(train_cases)/total:.1f}%)")
    print(f"Test set: {len(test_cases)} ({100*len(test_cases)/total:.1f}%)")
    
    # Article count
    train_articles = len(set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                            for c in train_cases))
    test_articles = len(set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                           for c in test_cases))
    print(f"\nTrain articles: {train_articles}")
    print(f"Test articles: {test_articles}")
    
    # Type distribution
    print("\n--- Leishmaniasis Type Distribution ---")
    print("\nTrain set:")
    for lt, count in sorted(get_leish_type_distribution(train_cases).items()):
        print(f"  {lt}: {count}")
    
    print("\nTest set:")
    for lt, count in sorted(get_leish_type_distribution(test_cases).items()):
        print(f"  {lt}: {count}")
    
    # Verify no article overlap
    train_article_set = set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                           for c in train_cases)
    test_article_set = set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                          for c in test_cases)
    overlap = train_article_set & test_article_set
    
    print(f"\n--- Anti-Leakage Verification ---")
    if overlap:
        print(f"⚠️  WARNING: {len(overlap)} overlapping articles found!")
        for a in list(overlap)[:5]:
            print(f"    - {a}")
    else:
        print("✅ No article overlap between train and test (anti-leakage verified)")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Split Leishmaniasis dataset for RAG evaluation'
    )
    parser.add_argument(
        '--input', '-i',
        type=Path,
        default=Path(__file__).parent / 'whole_multicare_dataset' / 'leishmaniasis_verified_strict.jsonl',
        help='Input JSONL file (default: 230 strict cases)'
    )
    parser.add_argument(
        '--output-dir', '-o',
        type=Path,
        default=Path(__file__).parent / 'leishmaniasis_split',
        help='Output directory for train/test files'
    )
    parser.add_argument(
        '--test-ratio', '-r',
        type=float,
        default=0.2,
        help='Fraction of data for test set (default: 0.2)'
    )
    parser.add_argument(
        '--seed', '-s',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '--no-dedup',
        action='store_true',
        help='Skip text hash deduplication'
    )
    
    args = parser.parse_args()
    
    # Load data
    print(f"Loading data from: {args.input}")
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return 1
    
    cases = load_jsonl(args.input)
    print(f"Loaded {len(cases)} cases")
    
    # Deduplication
    duplicates_removed = 0
    if not args.no_dedup:
        print("Deduplicating by case_text hash...")
        cases, duplicates_removed = deduplicate_by_text_hash(cases)
        print(f"After deduplication: {len(cases)} cases ({duplicates_removed} removed)")
    
    # Group by article
    article_groups = group_by_article(cases)
    print(f"Cases grouped into {len(article_groups)} articles")
    
    # Split
    train_cases, test_cases = split_by_articles(
        article_groups,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    
    # Print statistics
    print_statistics(train_cases, test_cases, duplicates_removed)
    
    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    train_path = args.output_dir / 'train.jsonl'
    test_path = args.output_dir / 'test.jsonl'
    stats_path = args.output_dir / 'split_stats.json'
    
    save_jsonl(train_cases, train_path)
    save_jsonl(test_cases, test_path)
    
    # Save statistics
    stats = {
        'input_file': str(args.input),
        'seed': args.seed,
        'test_ratio': args.test_ratio,
        'total_cases': len(train_cases) + len(test_cases),
        'duplicates_removed': duplicates_removed,
        'train_count': len(train_cases),
        'test_count': len(test_cases),
        'train_articles': len(set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                                  for c in train_cases)),
        'test_articles': len(set(c.get('article_id', c['case_id'].rsplit('_', 1)[0]) 
                                 for c in test_cases)),
        'train_leish_types': get_leish_type_distribution(train_cases),
        'test_leish_types': get_leish_type_distribution(test_cases),
    }
    
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"\n✅ Saved:")
    print(f"   - {train_path}")
    print(f"   - {test_path}")
    print(f"   - {stats_path}")
    
    return 0


if __name__ == '__main__':
    exit(main())
