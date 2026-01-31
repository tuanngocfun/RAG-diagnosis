#!/usr/bin/env python3
"""
Integration Script for Articles and Guidelines

Merges processed article data with the existing train.jsonl to create
train_extended.jsonl for RAG training set augmentation.

Features:
- Deduplication based on case_id and content hash
- Validates JSON schema before merge
- Preserves original train.jsonl entries
- Adds source_type metadata for routing
"""

import json
import hashlib
from pathlib import Path
from typing import Dict, List, Set

# Configuration
DATA_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/leishmaniasis_verified")
TRAIN_FILE = DATA_DIR / "train.jsonl"
ARTICLES_FILE = DATA_DIR / "articles_processed.jsonl"
OUTPUT_FILE = DATA_DIR / "train_extended.jsonl"

# Required fields for validation
REQUIRED_FIELDS = ["case_id", "case_text", "diagnosis", "is_leishmaniasis"]


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load entries from a JSONL file."""
    entries = []
    if not filepath.exists():
        print(f"Warning: File not found: {filepath}")
        return entries
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON at line {line_num}: {e}")
    
    print(f"Loaded {len(entries)} entries from {filepath.name}")
    return entries


def validate_entry(entry: Dict, index: int) -> bool:
    """Validate that entry has all required fields."""
    missing = [f for f in REQUIRED_FIELDS if f not in entry]
    if missing:
        print(f"  Warning: Entry {index} missing fields: {missing}")
        return False
    return True


def get_content_hash(entry: Dict) -> str:
    """Generate hash of entry content for deduplication."""
    # Use case_text for content-based dedup
    content = entry.get("case_text", "")[:1000]
    return hashlib.sha256(content.encode()).hexdigest()[:16]


def add_source_type(entries: List[Dict], source_type: str) -> List[Dict]:
    """Add source_type field to entries that don't have it."""
    for entry in entries:
        if "source_type" not in entry:
            entry["source_type"] = source_type
    return entries


def merge_entries(train_entries: List[Dict], article_entries: List[Dict]) -> List[Dict]:
    """
    Merge article entries into train entries, handling deduplication.
    """
    # Track existing case_ids and content hashes
    existing_ids: Set[str] = {e.get("case_id", "") for e in train_entries}
    existing_hashes: Set[str] = {get_content_hash(e) for e in train_entries}
    
    # Validate and add source_type to existing entries
    train_entries = add_source_type(train_entries, "case_report")
    
    # Process new entries
    merged = list(train_entries)
    added = 0
    skipped_id = 0
    skipped_hash = 0
    
    for i, entry in enumerate(article_entries):
        case_id = entry.get("case_id", "")
        content_hash = get_content_hash(entry)
        
        # Skip if duplicate case_id
        if case_id in existing_ids:
            print(f"  Skipping duplicate ID: {case_id}")
            skipped_id += 1
            continue
        
        # Skip if duplicate content
        if content_hash in existing_hashes:
            print(f"  Skipping duplicate content: {case_id}")
            skipped_hash += 1
            continue
        
        # Validate entry
        if not validate_entry(entry, i):
            continue
        
        # Add entry
        merged.append(entry)
        existing_ids.add(case_id)
        existing_hashes.add(content_hash)
        added += 1
    
    print(f"\nMerge Summary:")
    print(f"  Original entries: {len(train_entries)}")
    print(f"  New entries added: {added}")
    print(f"  Skipped (duplicate ID): {skipped_id}")
    print(f"  Skipped (duplicate content): {skipped_hash}")
    print(f"  Total entries: {len(merged)}")
    
    return merged


def save_jsonl(entries: List[Dict], filepath: Path):
    """Save entries to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(entries)} entries to {filepath}")


def print_statistics(entries: List[Dict]):
    """Print statistics about the merged dataset."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    # Count by source type
    source_counts: Dict[str, int] = {}
    for entry in entries:
        src = entry.get("source_type", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print("\nBy Source Type:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")
    
    # Count by diagnosis type
    diag_counts: Dict[str, int] = {}
    for entry in entries:
        diag = entry.get("diagnosis_type", "unknown")
        diag_counts[diag] = diag_counts.get(diag, 0) + 1
    
    print("\nBy Diagnosis Type:")
    for diag, count in sorted(diag_counts.items()):
        print(f"  {diag}: {count}")
    
    # Image statistics
    total_images = sum(len(entry.get("images", [])) for entry in entries)
    entries_with_images = sum(1 for e in entries if e.get("images"))
    
    print(f"\nImages:")
    print(f"  Total images: {total_images}")
    print(f"  Entries with images: {entries_with_images}")
    print(f"  Entries without images: {len(entries) - entries_with_images}")
    
    # Text statistics
    total_chars = sum(len(e.get("case_text", "")) for e in entries)
    print(f"\nText:")
    print(f"  Total characters: {total_chars:,}")
    print(f"  Avg chars per entry: {total_chars // len(entries):,}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Article & Guidelines Integration Script")
    print("=" * 60)
    
    # Load existing training data
    print("\n1. Loading existing training data...")
    train_entries = load_jsonl(TRAIN_FILE)
    
    # Load processed articles
    print("\n2. Loading processed articles...")
    article_entries = load_jsonl(ARTICLES_FILE)
    
    if not article_entries:
        print("No article entries to merge. Run article_preprocessor.py first.")
        return
    
    # Merge entries
    print("\n3. Merging entries...")
    merged = merge_entries(train_entries, article_entries)
    
    # Save merged output
    print("\n4. Saving merged dataset...")
    save_jsonl(merged, OUTPUT_FILE)
    
    # Print statistics
    print_statistics(merged)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS")
    print("=" * 60)
    print("1. Verify merged data: head -c 2000 " + str(OUTPUT_FILE))
    print("2. Update config.py to use train_extended.jsonl")
    print("3. Re-index with Qdrant: python3 indexer.py")
    print("4. Run evaluation pipeline")


if __name__ == "__main__":
    main()
