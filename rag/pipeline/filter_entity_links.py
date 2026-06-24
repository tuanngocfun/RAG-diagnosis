#!/usr/bin/env python3
"""
Filter Entity Links to Verified Cases Only

Filters case_entity_links.jsonl from leishmaniasis_multimodal (406 cases)
to only include cases present in leishmaniasis_verified_v2 (163 cases).

Usage:
    python -m rag.pipeline.filter_entity_links
"""
import json
from pathlib import Path

from .config import DATA_ROOT, SPLIT_DIR, TRAIN_JSONL, TEST_JSONL


def filter_entity_links(
    source_path: Path = None,
    output_path: Path = None,
) -> Path:
    """
    Filter entity links to only include verified cases.
    
    Args:
        source_path: Source entity links (default: DATA_ROOT/case_entity_links.jsonl)
        output_path: Output path (default: SPLIT_DIR/case_entity_links.jsonl)
    
    Returns:
        Path to filtered entity links file
    """
    # Default paths
    if source_path is None:
        source_path = DATA_ROOT / "case_entity_links.jsonl"
    if output_path is None:
        output_path = SPLIT_DIR / "case_entity_links.jsonl"
    
    print(f"{'='*60}")
    print("FILTER ENTITY LINKS TO VERIFIED CASES")
    print(f"{'='*60}")
    print(f"Source: {source_path}")
    print(f"Output: {output_path}")
    
    # Load verified case IDs
    verified_cases = set()
    
    with open(TRAIN_JSONL) as f:
        for line in f:
            verified_cases.add(json.loads(line)["case_id"])
    
    with open(TEST_JSONL) as f:
        for line in f:
            verified_cases.add(json.loads(line)["case_id"])
    
    print(f"\nVerified cases: {len(verified_cases)}")
    
    # Filter entity links
    filtered_count = 0
    total_count = 0
    filtered_case_ids = set()
    
    with open(source_path) as src, open(output_path, "w") as dst:
        for line in src:
            total_count += 1
            entity = json.loads(line)
            if entity["case_id"] in verified_cases:
                dst.write(line)
                filtered_count += 1
                filtered_case_ids.add(entity["case_id"])
    
    print(f"Total entity links: {total_count}")
    print(f"Filtered entity links: {filtered_count}")
    print(f"Unique cases in output: {len(filtered_case_ids)}")
    
    # Verify coverage
    missing = verified_cases - filtered_case_ids
    if missing:
        print(f"\nWARNING: {len(missing)} verified cases have no entity links:")
        for case_id in list(missing)[:5]:
            print(f"  - {case_id}")
        if len(missing) > 5:
            print(f"  ... and {len(missing) - 5} more")
    else:
        print(f"\n✓ All {len(verified_cases)} verified cases have entity links")
    
    print(f"\n✓ Saved filtered entity links to {output_path}")
    
    return output_path


if __name__ == "__main__":
    filter_entity_links()
