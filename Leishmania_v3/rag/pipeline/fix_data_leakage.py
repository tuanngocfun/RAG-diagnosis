"""
Fix Data Leakage in Train/Test Split

Per Gemini 3 Pro feedback: Move ALL cases belonging to overlapping PMC IDs 
from Test set to Train set, ensuring evaluation is on strictly "unseen" papers.

Creates:
- train_clean.jsonl
- test_clean.jsonl
"""
import json
from pathlib import Path
from typing import Set, List, Dict


def extract_pmc_id(case: Dict) -> str:
    """Extract PMC ID from case."""
    # Try article_id field first
    if case.get("article_id"):
        return case["article_id"]
    # Extract from case_id (format: PMC123456_01)
    case_id = case.get("case_id", "")
    if case_id.startswith("PMC"):
        return case_id.split("_")[0]
    return ""


def fix_data_leakage(data_root: Path) -> Dict:
    """
    Fix train/test overlap by moving overlapping cases to train.
    
    Returns:
        Dict with stats
    """
    train_path = data_root / "train.jsonl"
    test_path = data_root / "test.jsonl"
    
    # Load data
    print("Loading data...")
    with open(train_path) as f:
        train_cases = [json.loads(line) for line in f]
    with open(test_path) as f:
        test_cases = [json.loads(line) for line in f]
    
    print(f"  Original train: {len(train_cases)} cases")
    print(f"  Original test:  {len(test_cases)} cases")
    
    # Extract PMC IDs
    train_pmc_ids = {extract_pmc_id(c) for c in train_cases if extract_pmc_id(c)}
    test_pmc_ids = {extract_pmc_id(c) for c in test_cases if extract_pmc_id(c)}
    
    # Find overlap
    overlap_pmc_ids = train_pmc_ids & test_pmc_ids
    
    print(f"\n  Train unique PMC IDs: {len(train_pmc_ids)}")
    print(f"  Test unique PMC IDs:  {len(test_pmc_ids)}")
    print(f"  Overlapping PMC IDs:  {len(overlap_pmc_ids)}")
    
    if overlap_pmc_ids:
        print("\n  Overlapping IDs:")
        for pid in sorted(overlap_pmc_ids)[:10]:
            print(f"    - {pid}")
        if len(overlap_pmc_ids) > 10:
            print(f"    ... and {len(overlap_pmc_ids) - 10} more")
    
    # Split test into clean and overlapping
    test_clean = []
    test_to_move = []
    
    for case in test_cases:
        pmc_id = extract_pmc_id(case)
        if pmc_id in overlap_pmc_ids:
            test_to_move.append(case)
        else:
            test_clean.append(case)
    
    print(f"\n  Test cases to move to train: {len(test_to_move)}")
    print(f"  Test cases remaining (clean): {len(test_clean)}")
    
    # Merge moved cases into train
    train_clean = train_cases + test_to_move
    
    # Verify disjointness
    train_clean_pmc = {extract_pmc_id(c) for c in train_clean if extract_pmc_id(c)}
    test_clean_pmc = {extract_pmc_id(c) for c in test_clean if extract_pmc_id(c)}
    final_overlap = train_clean_pmc & test_clean_pmc
    
    print(f"\n  Final train: {len(train_clean)} cases ({len(train_clean_pmc)} unique PMC IDs)")
    print(f"  Final test:  {len(test_clean)} cases ({len(test_clean_pmc)} unique PMC IDs)")
    print(f"  Final overlap: {len(final_overlap)} PMC IDs")
    
    if final_overlap:
        print("  ✗ ERROR: Still have overlap!")
        return {"success": False, "reason": "remaining overlap"}
    
    print("\n  ✓ SUCCESS: No overlap in clean datasets")
    
    # Save clean versions
    train_clean_path = data_root / "train_clean.jsonl"
    test_clean_path = data_root / "test_clean.jsonl"
    
    with open(train_clean_path, "w") as f:
        for case in train_clean:
            f.write(json.dumps(case) + "\n")
    
    with open(test_clean_path, "w") as f:
        for case in test_clean:
            f.write(json.dumps(case) + "\n")
    
    print(f"\n  Saved: {train_clean_path}")
    print(f"  Saved: {test_clean_path}")
    
    return {
        "success": True,
        "original_train": len(train_cases),
        "original_test": len(test_cases),
        "overlapping_pmc_ids": len(overlap_pmc_ids),
        "cases_moved": len(test_to_move),
        "final_train": len(train_clean),
        "final_test": len(test_clean)
    }


if __name__ == "__main__":
    from rag.pipeline.config import DATA_ROOT
    
    print("=" * 60)
    print("FIX DATA LEAKAGE")
    print("=" * 60)
    
    stats = fix_data_leakage(DATA_ROOT)
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for k, v in stats.items():
        print(f"  {k}: {v}")
