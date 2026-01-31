#!/usr/bin/env python3
"""
Complete Merge Script for All Data Sources

Combines:
1. PubMed case reports (train.jsonl) - 114 entries
2. Processed articles (articles_processed.jsonl) - 4 entries
3. Guidelines/textbooks (guideline_entries.jsonl) - 8 entries

Output: train_extended.jsonl with proper source_type and image paths normalized
"""

import json
import shutil
import hashlib
from pathlib import Path
from typing import Dict, List, Set

# Paths
DATA_DIR = Path("/home/students/Leishmania/Leishmania_v3/data")
VERIFIED_DIR = DATA_DIR / "leishmaniasis_verified"
GUIDELINE_DIR = DATA_DIR / "guideline_textbook"
IMAGES_OUTPUT_DIR = DATA_DIR / "leishmaniasis_images" / "images"

# Input files
TRAIN_FILE = VERIFIED_DIR / "train.jsonl"  # 114 PubMed cases
ARTICLES_FILE = VERIFIED_DIR / "articles_processed.jsonl"  # 4 articles
GUIDELINES_FILE = GUIDELINE_DIR / "guideline_entries.jsonl"  # 8 guideline sections

# Output
OUTPUT_FILE = VERIFIED_DIR / "train_extended.jsonl"

REQUIRED_FIELDS = ["case_id", "case_text", "diagnosis", "is_leishmaniasis"]


def load_jsonl(path: Path) -> List[Dict]:
    """Load entries from JSONL file."""
    entries = []
    if not path.exists():
        print(f"  Warning: File not found: {path}")
        return entries
    
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"  Warning: Invalid JSON: {e}")
    
    return entries


def normalize_image_paths(entry: Dict, source_type: str) -> Dict:
    """
    Copy images to canonical location and update paths.
    Returns updated entry with normalized image paths.
    """
    if not entry.get("images"):
        return entry
    
    case_id = entry["case_id"]
    normalized_images = []
    
    for i, img in enumerate(entry["images"]):
        old_path = img.get("file", "")
        
        # Check if path needs normalization (contains guideline_textbook)
        if "guideline_textbook" in old_path or "gpt52extendedthinking" in old_path:
            # Full source path
            src_path = DATA_DIR / old_path
            if not src_path.exists():
                # Try with guideline_textbook prefix
                src_path = GUIDELINE_DIR / old_path
            
            if src_path.exists():
                # Create destination directory
                dest_dir = IMAGES_OUTPUT_DIR / case_id
                dest_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate canonical filename
                new_filename = f"{case_id}_fig{i+1:02d}.webp"
                dest_path = dest_dir / new_filename
                
                # Copy image
                if not dest_path.exists():
                    shutil.copy2(src_path, dest_path)
                    print(f"    Copied: {src_path.name} -> {dest_path}")
                
                # Update path in entry
                img = dict(img)  # Copy to avoid modifying original
                img["file"] = f"{case_id}/{new_filename}"
            else:
                print(f"    Warning: Image not found: {old_path}")
        
        normalized_images.append(img)
    
    entry = dict(entry)
    entry["images"] = normalized_images
    return entry


def add_source_type(entry: Dict, source_type: str) -> Dict:
    """Add source_type field if missing."""
    if "source_type" not in entry:
        entry = dict(entry)
        entry["source_type"] = source_type
    return entry


def content_hash(entry: Dict) -> str:
    """Generate hash for deduplication."""
    text = entry.get("case_text", "")[:500]
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def merge_all_sources() -> List[Dict]:
    """Merge all data sources with deduplication."""
    
    # Track for deduplication
    seen_ids: Set[str] = set()
    seen_hashes: Set[str] = set()
    all_entries: List[Dict] = []
    
    # 1. Load PubMed case reports (primary source)
    print("\n1. Loading PubMed case reports...")
    train_entries = load_jsonl(TRAIN_FILE)
    print(f"   Loaded {len(train_entries)} entries")
    
    for entry in train_entries:
        entry = add_source_type(entry, "case_report")
        seen_ids.add(entry.get("case_id", ""))
        seen_hashes.add(content_hash(entry))
        all_entries.append(entry)
    
    # 2. Load processed articles
    print("\n2. Loading processed articles...")
    article_entries = load_jsonl(ARTICLES_FILE)
    print(f"   Loaded {len(article_entries)} entries")
    
    added = 0
    for entry in article_entries:
        case_id = entry.get("case_id", "")
        if case_id in seen_ids or content_hash(entry) in seen_hashes:
            print(f"   Skipping duplicate: {case_id}")
            continue
        entry = add_source_type(entry, "review_article")
        seen_ids.add(case_id)
        seen_hashes.add(content_hash(entry))
        all_entries.append(entry)
        added += 1
    print(f"   Added {added} entries")
    
    # 3. Load guidelines/textbooks
    print("\n3. Loading guidelines/textbooks...")
    guideline_entries = load_jsonl(GUIDELINES_FILE)
    print(f"   Loaded {len(guideline_entries)} entries")
    
    added = 0
    for entry in guideline_entries:
        case_id = entry.get("case_id", "")
        if case_id in seen_ids or content_hash(entry) in seen_hashes:
            print(f"   Skipping duplicate: {case_id}")
            continue
        
        # Normalize image paths (copy to canonical location)
        entry = normalize_image_paths(entry, "guideline")
        entry = add_source_type(entry, "guideline")
        
        seen_ids.add(case_id)
        seen_hashes.add(content_hash(entry))
        all_entries.append(entry)
        added += 1
    print(f"   Added {added} entries")
    
    return all_entries


def print_statistics(entries: List[Dict]):
    """Print dataset statistics."""
    print("\n" + "=" * 60)
    print("DATASET STATISTICS")
    print("=" * 60)
    
    # By source type
    sources: Dict[str, int] = {}
    for e in entries:
        src = e.get("source_type", "unknown")
        sources[src] = sources.get(src, 0) + 1
    
    print("\nBy Source Type:")
    for src, count in sorted(sources.items()):
        print(f"  {src}: {count}")
    
    # By diagnosis type
    diag_types: Dict[str, int] = {}
    for e in entries:
        dtype = e.get("diagnosis_type", "Unknown")
        diag_types[dtype] = diag_types.get(dtype, 0) + 1
    
    print("\nBy Diagnosis Type:")
    for dtype, count in sorted(diag_types.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count}")
    
    # Images
    total_images = sum(len(e.get("images", [])) for e in entries)
    entries_with_images = sum(1 for e in entries if e.get("images"))
    
    print(f"\nImages:")
    print(f"  Total images: {total_images}")
    print(f"  Entries with images: {entries_with_images}")
    
    # Text
    total_chars = sum(len(e.get("case_text", "")) for e in entries)
    print(f"\nText:")
    print(f"  Total characters: {total_chars:,}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Complete Data Merge: PubMed + Articles + Guidelines")
    print("=" * 60)
    
    # Merge all sources
    entries = merge_all_sources()
    
    print(f"\nTotal entries: {len(entries)}")
    
    # Save to output file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nSaved to: {OUTPUT_FILE}")
    
    # Print statistics
    print_statistics(entries)
    
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print("\nNext steps:")
    print("  1. Re-index with Qdrant")
    print("  2. Run RAG evaluation pipeline")


if __name__ == "__main__":
    main()
