#!/usr/bin/env python3
"""
Integrate Guideline Entries into Training Set

This script:
1. Validates guideline_entries.jsonl format
2. Creates train_extended.jsonl with original + guideline entries
3. Optionally updates the original train.jsonl

Usage:
    python3 integrate_guidelines.py [--merge-to-original]
"""

import json
import shutil
import sys
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).parent
GUIDELINE_ENTRIES = BASE_DIR / "guideline_entries.jsonl"
TRAIN_JSONL = BASE_DIR.parent / "leishmaniasis_verified" / "train.jsonl"
TRAIN_EXTENDED = BASE_DIR.parent / "leishmaniasis_verified" / "train_extended.jsonl"
BACKUP_DIR = BASE_DIR / "backups"

# Required fields in train.jsonl format
REQUIRED_FIELDS = ["case_id", "diagnosis", "case_text"]
OPTIONAL_FIELDS = ["diagnosis_type", "species", "confirmation_method", "evidence_span", 
                   "confidence", "is_leishmaniasis", "title", "abstract", "images",
                   "source_type", "source", "license"]


def load_jsonl(path: Path) -> list:
    """Load JSONL file."""
    entries = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"ERROR: Invalid JSON on line {i}: {e}")
                    return []
    return entries


def validate_entry(entry: dict, idx: int) -> list:
    """Validate a single entry, return list of issues."""
    issues = []
    
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in entry:
            issues.append(f"Entry {idx}: Missing required field '{field}'")
        elif not entry[field]:
            issues.append(f"Entry {idx}: Empty required field '{field}'")
    
    # Check case_id uniqueness (caller handles this)
    
    # Validate images structure if present
    if "images" in entry and entry["images"]:
        for i, img in enumerate(entry["images"]):
            if "file" not in img:
                issues.append(f"Entry {idx}, image {i}: Missing 'file' field")
            if "caption" not in img:
                issues.append(f"Entry {idx}, image {i}: Missing 'caption' field")
    
    return issues


def main():
    """Main integration pipeline."""
    print("=" * 60)
    print("Integrate Guidelines into Training Set")
    print("=" * 60)
    
    merge_to_original = "--merge-to-original" in sys.argv
    
    # Check files exist
    if not GUIDELINE_ENTRIES.exists():
        print(f"ERROR: Guideline entries not found: {GUIDELINE_ENTRIES}")
        return 1
    if not TRAIN_JSONL.exists():
        print(f"ERROR: Train JSONL not found: {TRAIN_JSONL}")
        return 1
    
    # Load entries
    print(f"\n[1] Loading data...")
    
    train_entries = load_jsonl(TRAIN_JSONL)
    guideline_entries = load_jsonl(GUIDELINE_ENTRIES)
    
    if not train_entries or not guideline_entries:
        return 1
    
    print(f"    Train entries: {len(train_entries)}")
    print(f"    Guideline entries: {len(guideline_entries)}")
    
    # Validate guideline entries
    print(f"\n[2] Validating guideline entries...")
    
    all_issues = []
    existing_ids = {e["case_id"] for e in train_entries}
    
    for idx, entry in enumerate(guideline_entries):
        issues = validate_entry(entry, idx)
        all_issues.extend(issues)
        
        # Check for ID collision
        if entry.get("case_id") in existing_ids:
            all_issues.append(f"Entry {idx}: case_id '{entry['case_id']}' already exists in train.jsonl")
        existing_ids.add(entry.get("case_id"))
    
    if all_issues:
        print(f"    VALIDATION FAILED:")
        for issue in all_issues[:10]:  # Show first 10
            print(f"      - {issue}")
        if len(all_issues) > 10:
            print(f"      ... and {len(all_issues) - 10} more issues")
        return 1
    
    print(f"    All entries valid!")
    
    # Create backup
    BACKUP_DIR.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Write extended training file
    print(f"\n[3] Creating extended training file...")
    
    extended_entries = train_entries + guideline_entries
    
    with open(TRAIN_EXTENDED, 'w', encoding='utf-8') as f:
        for entry in extended_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"    Created: {TRAIN_EXTENDED}")
    print(f"    Total entries: {len(extended_entries)}")
    
    # Optionally merge to original
    if merge_to_original:
        print(f"\n[4] Merging to original train.jsonl...")
        
        # Backup original
        backup_path = BACKUP_DIR / f"train_backup_{timestamp}.jsonl"
        shutil.copy(TRAIN_JSONL, backup_path)
        print(f"    Backup created: {backup_path}")
        
        # Overwrite original
        shutil.copy(TRAIN_EXTENDED, TRAIN_JSONL)
        print(f"    Updated: {TRAIN_JSONL}")
    
    # Summary
    print("\n" + "=" * 60)
    print("INTEGRATION COMPLETE")
    print("=" * 60)
    print(f"\nOriginal train entries: {len(train_entries)}")
    print(f"Guideline entries added: {len(guideline_entries)}")
    print(f"Total extended entries: {len(extended_entries)}")
    
    print(f"\nFiles created:")
    print(f"  - {TRAIN_EXTENDED}")
    if merge_to_original:
        print(f"  - {TRAIN_JSONL} (updated)")
        print(f"  - {backup_path} (backup)")
    
    print(f"\nNext steps:")
    print(f"  1. Review train_extended.jsonl")
    print(f"  2. Re-index with: python3 -m rag.run_indexing")
    print(f"  3. Regenerate qrels if needed")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
