#!/usr/bin/env python3
"""
Guideline & Textbook Preprocessor with Image Integration

This script creates training entries from guideline/textbook PDFs with:
1. Text extraction from PDFs (using existing guideline_entries or re-extracting)
2. GPT5.2 image annotations integration
3. Proper image copying to destination directory
4. Schema matching train.jsonl format

This complements article_preprocessor.py for articles.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

# Configuration
GUIDELINE_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/guideline_textbook")
GPT52_DIR = GUIDELINE_DIR / "gpt52extendedthinking_images_metadata"
OUTPUT_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/leishmaniasis_verified")
IMAGES_OUTPUT_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/leishmaniasis_images/images")

# Mapping of PDFs to their GPT5.2 annotations and image folders
GUIDELINE_CONFIG = {
    "CDC - DPDx - Leishmaniasis.pdf": {
        "annotations": "CDC - DPDx - Leishmaniasis.jsonl",
        "images_dir": "cdc_dpdx_figures_only_images",
        "doc_id": "guideline_cdc_dpdx",
        "title": "CDC DPDx - Leishmaniasis Diagnostic Reference",
        "diagnosis": "Leishmaniasis (DPDx Reference)",
        "journal": "CDC DPDx",
        "license": "Public Domain (CDC)",
    },
    "WHO GUIDELINE for the treatment of visceral leishmaniasis in HIV co-infected patients in East Africa and South-East Asia.pdf": {
        "annotations": "WHO GUIDELINE for the treatment of visceral leishmaniasis in HIV co-infected patients in East Africa and South-East Asia.jsonl",
        "images_dir": "who_vl_hiv_figures_only_images",
        "doc_id": "guideline_who_vl_hiv",
        "title": "WHO Guideline for VL-HIV Coinfection Treatment",
        "diagnosis": "Visceral Leishmaniasis HIV Coinfection",
        "journal": "WHO Guidelines",
        "license": "CC BY-NC-SA 3.0 IGO",
    },
}


def load_existing_guideline_entries() -> Dict[str, Dict]:
    """Load existing guideline entries (text extracted)."""
    entries_file = GUIDELINE_DIR / "guideline_entries.jsonl"
    entries = {}
    
    if entries_file.exists():
        with open(entries_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = json.loads(line)
                    entries[entry.get("case_id", "")] = entry
    
    print(f"Loaded {len(entries)} existing guideline entries")
    return entries


def load_gpt52_annotations(jsonl_name: str) -> List[Dict]:
    """Load GPT5.2 image annotations."""
    path = GPT52_DIR / jsonl_name
    annotations = []
    
    if path.exists():
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(json.loads(line))
    
    return annotations


def copy_images(src_dir_name: str, doc_id: str, annotations: List[Dict]) -> List[Dict]:
    """Copy images to destination and return formatted image list."""
    src_dir = GPT52_DIR / src_dir_name
    dest_dir = IMAGES_OUTPUT_DIR / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    formatted_images = []
    
    for i, ann in enumerate(annotations):
        src_file = src_dir / ann.get("file", "")
        if not src_file.exists():
            print(f"  Warning: Image not found: {src_file}")
            continue
        
        # Use canonical naming
        new_filename = f"{doc_id}_fig{i+1:02d}.webp"
        dest_file = dest_dir / new_filename
        
        # Copy image
        shutil.copy2(src_file, dest_file)
        print(f"  Copied: {src_file.name} -> {dest_file.name}")
        
        formatted_images.append({
            "file": f"{doc_id}/{new_filename}",
            "file_id": ann.get("file_id", f"file_{doc_id}_{i+1:04d}"),
            "caption": ann.get("caption", ""),
            "image_type": ann.get("image_type", "diagram"),
            "image_subtype": ann.get("image_subtype", ""),
            "radiology_region": None,
            "radiology_view": None,
            "labels_supervised": ann.get("labels_supervised", "[]"),
            "labels_semisupervised": ann.get("labels_semisupervised", "[]"),
        })
    
    return formatted_images


def process_guideline(pdf_name: str, config: Dict, existing_entries: Dict[str, Dict]) -> Optional[Dict]:
    """Process a single guideline/textbook PDF."""
    print(f"\nProcessing: {pdf_name}")
    
    doc_id = config["doc_id"]
    
    # Load GPT5.2 annotations
    annotations = load_gpt52_annotations(config["annotations"])
    print(f"  Found {len(annotations)} image annotations")
    
    # Copy images and get formatted list
    images = copy_images(config["images_dir"], doc_id, annotations)
    print(f"  Copied {len(images)} images")
    
    # Find existing text entry (by matching case_id patterns)
    case_text = ""
    for eid, entry in existing_entries.items():
        # Match by partial case_id or title
        if config["title"][:20].lower() in eid.lower() or \
           "CDC" in eid and "CDC" in config["doc_id"] or \
           "WHO" in eid and "WHO" in config["doc_id"]:
            case_text = entry.get("case_text", "")
            print(f"  Found existing text entry: {eid}")
            break
    
    if not case_text:
        # Use a placeholder or extract from PDF
        case_text = f"[MEDICAL GUIDELINE]\nTitle: {config['title']}\n\nNo extracted text available. Please run PDF extraction."
        print("  Warning: No existing text entry found")
    
    # Add document type prefix if not present
    if not case_text.startswith("[MEDICAL GUIDELINE]"):
        case_text = f"[MEDICAL GUIDELINE]\nTitle: {config['title']}\n\n" + case_text
    
    # Build entry
    entry = {
        "case_id": doc_id,
        "diagnosis": config["diagnosis"],
        "diagnosis_type": "Guideline",
        "species": "",
        "confirmation_method": "Reference document",
        "evidence_span": "",
        "confidence": "high",
        "is_leishmaniasis": True,
        "raw_llm_response": None,
        "error": None,
        "case_text": case_text,
        "title": config["title"],
        "abstract": "",
        "images": images,
        "journal": config["journal"],
        "year": "2024",
        "doi": "",
        "license": config["license"],
        "source_type": "guideline",
        "license_source": "",
    }
    
    return entry


def main():
    """Main entry point."""
    print("=" * 60)
    print("Guideline & Textbook Preprocessor with Image Integration")
    print("=" * 60)
    
    # Load existing text entries
    existing_entries = load_existing_guideline_entries()
    
    # Process each guideline
    entries = []
    for pdf_name, config in GUIDELINE_CONFIG.items():
        entry = process_guideline(pdf_name, config, existing_entries)
        if entry:
            entries.append(entry)
    
    # Save to file
    output_file = OUTPUT_DIR / "guidelines_processed.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    
    print(f"\nSaved {len(entries)} entries to {output_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for entry in entries:
        print(f"  {entry['case_id']}: {len(entry['images'])} images")
    
    total_images = sum(len(e['images']) for e in entries)
    print(f"\nTotal: {len(entries)} guidelines, {total_images} images")
    print("\nNext: Run merge script to combine all data")


if __name__ == "__main__":
    main()
