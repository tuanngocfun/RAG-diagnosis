#!/usr/bin/env python3
"""
Stratified Train-Test Split for Leishmaniasis Multimodal Dataset

Based on Q1 journal standards (JAMIA, Journal of Biomedical Informatics):
1. Stratified splitting by entity type distribution
2. Patient/case-level splitting (no data leakage)
3. Ensure rare entities appear in training set
4. Statistics verification

References:
- Practical Considerations for Cross-Validation in Health Care (Tutorial)
- Impact of train/test sample regimen on performance estimate stability

Usage:
    python split_train_test.py --ratio 0.8
    python split_train_test.py --ratio 0.85 --seed 42
"""

import json
import sys
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Set, Tuple
import random

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import importlib.util
config_dir = Path(__file__).parent.parent / "00_config"

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config = load_module("config", config_dir / "config.py")

MULTIMODAL_JSONL = config.MULTIMODAL_JSONL
OUTPUT_ROOT = config.OUTPUT_ROOT
DATA_ROOT = config.DATA_ROOT


def load_kg_extended() -> dict:
    """Load the extended Knowledge Graph with entity-case links."""
    kg_path = OUTPUT_ROOT / "leishmaniasis_kg_extended.json"
    if not kg_path.exists():
        print(f"Error: {kg_path} not found. Run build_kg_entities.py first.")
        sys.exit(1)
    
    with open(kg_path) as f:
        return json.load(f)


def load_case_entity_links() -> List[dict]:
    """Load case-entity links."""
    links_path = OUTPUT_ROOT / "case_entity_links.jsonl"
    if not links_path.exists():
        print(f"Error: {links_path} not found. Run build_kg_entities.py first.")
        sys.exit(1)
    
    links = []
    with open(links_path) as f:
        for line in f:
            links.append(json.loads(line))
    return links


def load_multimodal_records() -> List[dict]:
    """Load all multimodal records."""
    records = []
    with open(MULTIMODAL_JSONL) as f:
        for line in f:
            records.append(json.loads(line))
    return records


def compute_case_entity_vectors(links: List[dict]) -> Dict[str, Set[str]]:
    """
    Create a mapping from case_id to set of entity types present.
    This is used for stratification.
    """
    case_entities = defaultdict(set)
    for link in links:
        case_id = link["case_id"]
        entity_type = link["entity_type"]
        case_entities[case_id].add(entity_type)
    return case_entities


def compute_entity_case_mapping(links: List[dict]) -> Dict[str, Set[str]]:
    """
    Create a mapping from entity_name to set of case_ids.
    Used to ensure rare entities appear in training.
    """
    entity_cases = defaultdict(set)
    for link in links:
        entity_name = link["entity_name"]
        case_id = link["case_id"]
        entity_cases[entity_name].add(case_id)
    return entity_cases


def identify_rare_entities(entity_cases: Dict[str, Set[str]], threshold: int = 3) -> Set[str]:
    """
    Identify entities that appear in fewer than threshold cases.
    These entities should preferentially appear in training set.
    """
    rare = set()
    for entity, cases in entity_cases.items():
        if len(cases) <= threshold:
            rare.add(entity)
    return rare


def stratified_split(
    records: List[dict],
    case_entity_vectors: Dict[str, Set[str]],
    entity_cases: Dict[str, Set[str]],
    train_ratio: float = 0.8,
    seed: int = 42
) -> Tuple[List[dict], List[dict]]:
    """
    Perform stratified train-test split.
    
    Strategy:
    1. First, assign cases with rare entities to training set
    2. Then, stratify remaining cases by entity type distribution
    3. Ensure test set only has entities that appear in training
    """
    random.seed(seed)
    
    # Get all case IDs
    all_case_ids = [r["case_id"] for r in records]
    case_lookup = {r["case_id"]: r for r in records}
    
    # Find rare entities and their cases
    rare_entities = identify_rare_entities(entity_cases, threshold=3)
    print(f"   Identified {len(rare_entities)} rare entities (≤3 cases)")
    
    # Cases that MUST be in training (have rare entities)
    must_train_cases = set()
    for entity in rare_entities:
        must_train_cases.update(entity_cases[entity])
    
    print(f"   {len(must_train_cases)} cases must be in training (contain rare entities)")
    
    # Remaining cases to split
    remaining_cases = [c for c in all_case_ids if c not in must_train_cases]
    
    # Group remaining cases by their entity type signature
    signature_groups = defaultdict(list)
    for case_id in remaining_cases:
        entity_types = case_entity_vectors.get(case_id, set())
        signature = tuple(sorted(entity_types))  # Make hashable
        signature_groups[signature].append(case_id)
    
    # Stratified split within each signature group
    train_cases = list(must_train_cases)
    test_cases = []
    
    for signature, cases in signature_groups.items():
        random.shuffle(cases)
        n_train = int(len(cases) * train_ratio)
        train_cases.extend(cases[:n_train])
        test_cases.extend(cases[n_train:])
    
    # Verify: check if any entity appears ONLY in test
    train_entity_set = set()
    for case_id in train_cases:
        for entity, cases in entity_cases.items():
            if case_id in cases:
                train_entity_set.add(entity)
    
    test_only_entities = set()
    for case_id in test_cases:
        for entity, cases in entity_cases.items():
            if case_id in cases and entity not in train_entity_set:
                test_only_entities.add(entity)
    
    # Move cases with test-only entities to training
    if test_only_entities:
        print(f"   ⚠ Found {len(test_only_entities)} entities only in test set, moving their cases to training")
        cases_to_move = set()
        for entity in test_only_entities:
            cases_to_move.update(entity_cases[entity] & set(test_cases))
        
        test_cases = [c for c in test_cases if c not in cases_to_move]
        train_cases.extend(cases_to_move)
    
    # Build final record lists
    train_records = [case_lookup[c] for c in train_cases if c in case_lookup]
    test_records = [case_lookup[c] for c in test_cases if c in case_lookup]
    
    return train_records, test_records


def compute_split_statistics(
    train_records: List[dict],
    test_records: List[dict],
    links: List[dict]
) -> dict:
    """Compute detailed statistics for the split."""
    train_ids = {r["case_id"] for r in train_records}
    test_ids = {r["case_id"] for r in test_records}
    
    # Entity distribution
    train_entities = defaultdict(lambda: defaultdict(int))
    test_entities = defaultdict(lambda: defaultdict(int))
    
    for link in links:
        case_id = link["case_id"]
        entity_type = link["entity_type"]
        entity_name = link["entity_name"]
        
        if case_id in train_ids:
            train_entities[entity_type][entity_name] += 1
        elif case_id in test_ids:
            test_entities[entity_type][entity_name] += 1
    
    # Image distribution
    train_images = sum(len(r.get("images", [])) for r in train_records)
    test_images = sum(len(r.get("images", [])) for r in test_records)
    
    # Compute overlap (entities in both sets)
    all_entity_types = set(train_entities.keys()) | set(test_entities.keys())
    overlap_stats = {}
    
    for etype in all_entity_types:
        train_names = set(train_entities[etype].keys())
        test_names = set(test_entities[etype].keys())
        overlap = train_names & test_names
        test_only = test_names - train_names
        
        overlap_stats[etype] = {
            "train_unique": len(train_names),
            "test_unique": len(test_names),
            "overlap": len(overlap),
            "test_only": len(test_only),
            "test_only_names": list(test_only)[:10]  # First 10 for inspection
        }
    
    return {
        "train_cases": len(train_records),
        "test_cases": len(test_records),
        "train_ratio": len(train_records) / (len(train_records) + len(test_records)),
        "train_images": train_images,
        "test_images": test_images,
        "entity_overlap": overlap_stats,
        "train_entity_counts": {k: len(v) for k, v in train_entities.items()},
        "test_entity_counts": {k: len(v) for k, v in test_entities.items()}
    }


def save_splits(
    train_records: List[dict],
    test_records: List[dict],
    stats: dict,
    output_dir: Path
):
    """Save train/test splits and statistics."""
    # Save train set
    train_path = output_dir / "train.jsonl"
    with open(train_path, 'w') as f:
        for r in train_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len(train_records)} train records to {train_path.name}")
    
    # Save test set
    test_path = output_dir / "test.jsonl"
    with open(test_path, 'w') as f:
        for r in test_records:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    print(f"   ✓ Saved {len(test_records)} test records to {test_path.name}")
    
    # Save statistics
    stats_path = output_dir / "split_stats.json"
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"   ✓ Saved split statistics to {stats_path.name}")


def print_summary(stats: dict):
    """Print a summary of the split."""
    print("\n" + "=" * 60)
    print("📊 TRAIN-TEST SPLIT SUMMARY")
    print("=" * 60)
    
    print(f"\n📌 Split Statistics:")
    print(f"   Training set: {stats['train_cases']} cases ({stats['train_ratio']*100:.1f}%)")
    print(f"   Test set: {stats['test_cases']} cases ({(1-stats['train_ratio'])*100:.1f}%)")
    print(f"   Training images: {stats['train_images']}")
    print(f"   Test images: {stats['test_images']}")
    
    print(f"\n📋 Entity Distribution by Type:")
    for etype in sorted(stats['entity_overlap'].keys()):
        overlap = stats['entity_overlap'][etype]
        train_n = overlap['train_unique']
        test_n = overlap['test_unique']
        test_only = overlap['test_only']
        status = "✅" if test_only == 0 else f"⚠ {test_only} test-only"
        print(f"   {etype:12s}: train={train_n:3d}, test={test_n:3d} → {status}")
    
    # Check for test-only entities
    total_test_only = sum(o['test_only'] for o in stats['entity_overlap'].values())
    if total_test_only == 0:
        print("\n✅ All entities in test set also appear in training set!")
    else:
        print(f"\n⚠ WARNING: {total_test_only} entities appear only in test set")


def main():
    parser = argparse.ArgumentParser(description="Stratified train-test split")
    parser.add_argument("--ratio", type=float, default=0.8,
                        help="Training set ratio (default: 0.8)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()
    
    print("=" * 60)
    print("STRATIFIED TRAIN-TEST SPLIT")
    print("=" * 60)
    print(f"   Target ratio: {args.ratio:.0%} train / {1-args.ratio:.0%} test")
    print(f"   Random seed: {args.seed}")
    
    # Load data
    print("\n📂 Loading data...")
    records = load_multimodal_records()
    print(f"   Loaded {len(records)} multimodal records")
    
    kg = load_kg_extended()
    print(f"   Loaded KG with {kg['metadata']['total_entities']} entities")
    
    links = load_case_entity_links()
    print(f"   Loaded {len(links)} case-entity links")
    
    # Compute mappings
    print("\n🔧 Computing entity mappings...")
    case_entity_vectors = compute_case_entity_vectors(links)
    entity_cases = compute_entity_case_mapping(links)
    print(f"   Cases with entities: {len(case_entity_vectors)}")
    print(f"   Unique entities: {len(entity_cases)}")
    
    # Perform split
    print("\n🔀 Performing stratified split...")
    train_records, test_records = stratified_split(
        records, case_entity_vectors, entity_cases,
        train_ratio=args.ratio, seed=args.seed
    )
    
    # Compute statistics
    print("\n📊 Computing statistics...")
    stats = compute_split_statistics(train_records, test_records, links)
    
    # Save results
    print("\n💾 Saving splits...")
    save_splits(train_records, test_records, stats, OUTPUT_ROOT)
    
    # Print summary
    print_summary(stats)
    
    print("\n✅ Split complete!")


if __name__ == "__main__":
    main()
