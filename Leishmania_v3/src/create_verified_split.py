#!/usr/bin/env python3
"""
Create Stratified Train/Test Split from Gemini 3 Pro Verified Cases

Filters high-confidence Leishmaniasis cases and creates an 80/20 split
stratified by diagnosis_type for balanced representation.

Output:
    data/leishmaniasis_verified/
    ├── train.jsonl          # ~114 cases (80%)
    ├── test.jsonl           # ~29 cases (20%)
    ├── split_stats.json     # Statistics
    └── all_verified.jsonl   # All 143 cases
"""

import json
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split
import random

# Paths
DATA_ROOT = Path(__file__).parent.parent / "data"
INPUT_FILE = DATA_ROOT / "leishmaniasis_split" / "cases_with_diagnosis_gem3pro_0_temperature_head_tail7k.jsonl"
OUTPUT_DIR = DATA_ROOT / "leishmaniasis_verified"

# Seed for reproducibility
RANDOM_SEED = 42


def load_original_cases(multimodal_file: Path) -> dict:
    """Load original multimodal cases indexed by case_id."""
    cases_by_id = {}
    with open(multimodal_file) as f:
        for line in f:
            case = json.loads(line)
            cases_by_id[case["case_id"]] = case
    return cases_by_id


def load_verified_cases(input_file: Path, multimodal_file: Path = None) -> list:
    """
    Load and filter high-confidence Leishmaniasis cases.
    
    Enriches with original multimodal data (case_text, images, etc.) if provided.
    """
    # Load original cases for enrichment
    original_cases = {}
    if multimodal_file and multimodal_file.exists():
        original_cases = load_original_cases(multimodal_file)
        print(f"Loaded {len(original_cases)} original cases for enrichment")
    
    cases = []
    with open(input_file) as f:
        for line in f:
            case = json.loads(line)
            # Filter: is_leishmaniasis AND (high OR medium confidence)
            if case.get("is_leishmaniasis", False) and case.get("confidence") in ["high", "medium"]:
                # Enrich with original multimodal data
                if case["case_id"] in original_cases:
                    orig = original_cases[case["case_id"]]
                    case["case_text"] = orig.get("case_text", "")
                    case["title"] = orig.get("title", "")
                    case["abstract"] = orig.get("abstract", "")
                    case["images"] = orig.get("images", [])
                    case["journal"] = orig.get("journal", "")
                    case["year"] = orig.get("year", "")
                    case["doi"] = orig.get("doi", "")
                    case["license"] = orig.get("license", "")
                cases.append(case)
    return cases


def stratified_split(cases: list, test_size: float = 0.2, seed: int = RANDOM_SEED):
    """
    Create stratified train/test split by diagnosis_type.
    
    Args:
        cases: List of case dictionaries
        test_size: Fraction for test set (default 0.2)
        seed: Random seed for reproducibility
    
    Returns:
        Tuple of (train_cases, test_cases)
    """
    # Extract stratification labels
    labels = [c.get("diagnosis_type", "Unknown") for c in cases]
    
    # Handle small strata: if any stratum has < 2 samples, use random split
    label_counts = Counter(labels)
    min_count = min(label_counts.values())
    
    if min_count < 2:
        print(f"Warning: Some strata have <2 samples, using random split")
        random.seed(seed)
        shuffled = cases.copy()
        random.shuffle(shuffled)
        split_idx = int(len(shuffled) * (1 - test_size))
        return shuffled[:split_idx], shuffled[split_idx:]
    
    # Stratified split
    train_cases, test_cases = train_test_split(
        cases,
        test_size=test_size,
        stratify=labels,
        random_state=seed
    )
    
    return train_cases, test_cases


def compute_stats(cases: list, name: str) -> dict:
    """Compute statistics for a set of cases."""
    stats = {
        "name": name,
        "total_cases": len(cases),
        "diagnosis_type_distribution": dict(Counter(
            c.get("diagnosis_type", "Unknown") for c in cases
        )),
        "confidence_distribution": dict(Counter(
            c.get("confidence", "unknown") for c in cases
        )),
        "species_distribution": dict(Counter(
            c.get("species") or "Not specified" for c in cases
        )),
        "cases_with_species": sum(1 for c in cases if c.get("species")),
    }
    return stats


def main():
    """Main entry point."""
    print("=" * 60)
    print("Creating Verified Leishmaniasis Dataset Split")
    print("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load verified cases with multimodal enrichment
    MULTIMODAL_FILE = DATA_ROOT / "leishmaniasis_multimodal" / "leishmaniasis_multimodal.jsonl"
    print(f"\nLoading from: {INPUT_FILE}")
    print(f"Enriching with: {MULTIMODAL_FILE}")
    cases = load_verified_cases(INPUT_FILE, MULTIMODAL_FILE)
    print(f"Loaded {len(cases)} verified Leishmaniasis cases")
    
    # Show image statistics
    cases_with_images = sum(1 for c in cases if c.get("images") and len(c["images"]) > 0)
    total_images = sum(len(c.get("images", [])) for c in cases)
    print(f"\n--- Multimodal Statistics ---")
    print(f"  Cases with images: {cases_with_images}/{len(cases)} ({100*cases_with_images/len(cases):.1f}%)")
    print(f"  Total images: {total_images}")
    
    # Show distribution before split
    print("\n--- Diagnosis Type Distribution ---")
    type_dist = Counter(c.get("diagnosis_type", "Unknown") for c in cases)
    for dtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} ({100*count/len(cases):.1f}%)")
    
    # Create stratified split
    print("\n--- Creating 80/20 Stratified Split ---")
    train_cases, test_cases = stratified_split(cases, test_size=0.2)
    
    print(f"Train: {len(train_cases)} cases")
    print(f"Test:  {len(test_cases)} cases")
    
    # Verify stratification
    print("\n--- Train Set Distribution ---")
    train_dist = Counter(c.get("diagnosis_type", "Unknown") for c in train_cases)
    for dtype, count in sorted(train_dist.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} ({100*count/len(train_cases):.1f}%)")
    
    print("\n--- Test Set Distribution ---")
    test_dist = Counter(c.get("diagnosis_type", "Unknown") for c in test_cases)
    for dtype, count in sorted(test_dist.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} ({100*count/len(test_cases):.1f}%)")
    
    # Save outputs
    print("\n--- Saving Files ---")
    
    # All verified cases
    all_file = OUTPUT_DIR / "all_verified.jsonl"
    with open(all_file, "w") as f:
        for case in cases:
            f.write(json.dumps(case) + "\n")
    print(f"Saved: {all_file}")
    
    # Train split
    train_file = OUTPUT_DIR / "train.jsonl"
    with open(train_file, "w") as f:
        for case in train_cases:
            f.write(json.dumps(case) + "\n")
    print(f"Saved: {train_file}")
    
    # Test split
    test_file = OUTPUT_DIR / "test.jsonl"
    with open(test_file, "w") as f:
        for case in test_cases:
            f.write(json.dumps(case) + "\n")
    print(f"Saved: {test_file}")
    
    # Statistics
    stats = {
        "source_file": str(INPUT_FILE),
        "random_seed": RANDOM_SEED,
        "all": compute_stats(cases, "all_verified"),
        "train": compute_stats(train_cases, "train"),
        "test": compute_stats(test_cases, "test"),
    }
    
    stats_file = OUTPUT_DIR / "split_stats.json"
    with open(stats_file, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"Saved: {stats_file}")
    
    print("\n" + "=" * 60)
    print("Split complete!")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
