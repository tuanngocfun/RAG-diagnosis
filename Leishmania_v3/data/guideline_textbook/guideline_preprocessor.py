#!/usr/bin/env python3
"""
Guideline Preprocessor for RAG Training Data

Preprocesses CDC DPDx and WHO VL-HIV guideline data to match train.jsonl format.
Uses pdfplumber to extract text from PDF files.

Key decisions (per LLM reviewer consensus):
- GPT 5.2 annotations as primary source
- CDC: Document-level (single entry)
- WHO: Section-level (coarse sections based on TOC)
- Output to separate guideline_entries.jsonl

Usage:
    python3 guideline_preprocessor.py
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

try:
    import pdfplumber
except ImportError:
    print("ERROR: pdfplumber not installed. Run: pip install pdfplumber")
    exit(1)

# Base paths
BASE_DIR = Path(__file__).parent
GPT52_DIR = BASE_DIR / "gpt52extendedthinking_images_metadata"
OUTPUT_FILE = BASE_DIR / "guideline_entries.jsonl"

# PDF files
CDC_PDF = BASE_DIR / "CDC - DPDx - Leishmaniasis.pdf"
WHO_PDF = BASE_DIR / "WHO GUIDELINE for the treatment of visceral leishmaniasis in HIV co-infected patients in East Africa and South-East Asia.pdf"

# Image directories
CDC_IMAGES_SUBDIR = "gpt52extendedthinking_images_metadata/cdc_dpdx_figures_only_images"
WHO_IMAGES_SUBDIR = "gpt52extendedthinking_images_metadata/who_vl_hiv_figures_only_images"


def parse_labels_string(label_str: str) -> List[str]:
    """Parse string like \"['a', 'b']\" to actual list."""
    if not label_str or label_str == "[]":
        return []
    return re.findall(r"'([^']*)'", label_str)


def load_jsonl(file_path: Path) -> List[Dict]:
    """Load JSONL file and return list of dicts."""
    entries = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def normalize_image_metadata(images: List[Dict], images_subdir: str) -> List[Dict]:
    """
    Normalize image metadata to match train.jsonl format:
    - Add relative path prefix
    - Convert label strings to arrays
    - Add missing fields with null values
    """
    normalized = []
    for img in images:
        normalized_img = {
            "file": f"{images_subdir}/{img['file']}",
            "file_id": img.get("file_id", ""),
            "caption": img.get("caption", ""),
            "image_type": img.get("image_type", ""),
            "image_subtype": img.get("image_subtype", ""),
            "radiology_region": None,
            "radiology_view": None,
            "labels_supervised": parse_labels_string(img.get("labels_supervised", "[]")),
            "labels_semisupervised": parse_labels_string(img.get("labels_semisupervised", "[]")),
        }
        normalized.append(normalized_img)
    return normalized


def extract_pdf_text(pdf_path: Path, start_page: int = 0, end_page: Optional[int] = None) -> str:
    """
    Extract text from PDF using pdfplumber.
    
    Args:
        pdf_path: Path to PDF file
        start_page: 0-indexed start page
        end_page: 0-indexed end page (exclusive), None for all pages
    
    Returns:
        Extracted text as string
    """
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        if end_page is None:
            end_page = total_pages
        
        for page_num in range(start_page, min(end_page, total_pages)):
            page = pdf.pages[page_num]
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    
    return "\n\n".join(text_parts)


def extract_pdf_full_text(pdf_path: Path) -> Tuple[str, int]:
    """Extract all text from PDF, return text and page count."""
    with pdfplumber.open(pdf_path) as pdf:
        text_parts = []
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n\n".join(text_parts), len(pdf.pages)


def clean_text(text: str) -> str:
    """Clean extracted text - remove excessive whitespace, fix common OCR issues."""
    # Replace multiple newlines with double newline
    text = re.sub(r'\n{3,}', '\n\n', text)
    # Replace multiple spaces with single space
    text = re.sub(r' {2,}', ' ', text)
    # Strip lines
    lines = [line.strip() for line in text.split('\n')]
    return '\n'.join(lines)


def create_cdc_entry(images: List[Dict], pdf_path: Path) -> Dict[str, Any]:
    """
    Create single document-level entry for CDC DPDx.
    CDC is a short, single-topic reference - extract all text.
    """
    print(f"    Extracting text from CDC PDF: {pdf_path.name}")
    
    full_text, page_count = extract_pdf_full_text(pdf_path)
    full_text = clean_text(full_text)
    
    print(f"    Extracted {len(full_text)} characters from {page_count} pages")
    
    return {
        "case_id": "CDC_DPDx_Leishmaniasis",
        "diagnosis": "Leishmaniasis (DPDx reference)",
        "diagnosis_type": "Educational",
        "species": "Leishmania spp.",
        "confirmation_method": "Reference document",
        "evidence_span": "CDC DPDx diagnostic reference for leishmaniasis covering all clinical forms.",
        "confidence": "high",
        "is_leishmaniasis": True,
        "case_text": full_text,
        "title": "CDC DPDx - Leishmaniasis",
        "abstract": "Reference document covering parasitology, clinical features, geographic distribution, and laboratory diagnosis of leishmaniasis.",
        "images": images,
        "source_type": "guideline",
        "source": "CDC DPDx",
        "license": "Public Domain (US Government)"
    }


# WHO document structure based on Table of Contents
# Logical page numbers (in document) vs absolute page numbers (0-indexed)
# Front matter: ~12 pages (cover, TOC, acknowledgements, abbreviations, executive summary)
WHO_SECTIONS = [
    {
        "id": "Introduction",
        "title": "Introduction",
        "pages": (12, 16),  # Absolute pages 0-indexed: Introduction starts after front matter
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Introduction",
        "diagnosis_type": "Visceral",
    },
    {
        "id": "Methods",
        "title": "Methods",
        "pages": (16, 19),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Methods",
        "diagnosis_type": "Reference",
    },
    {
        "id": "Epidemiology",
        "title": "Background and Epidemiology",
        "pages": (19, 24),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Epidemiology",
        "diagnosis_type": "Visceral",
    },
    {
        "id": "Diagnosis",
        "title": "Case Definition and Diagnosis",
        "pages": (24, 30),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Case Definition",
        "diagnosis_type": "Visceral",
    },
    {
        "id": "Treatment_EastAfrica",
        "title": "Treatment Recommendations - East Africa",
        "pages": (30, 40),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Treatment East Africa",
        "diagnosis_type": "Visceral",
    },
    {
        "id": "Treatment_SEAsia",
        "title": "Treatment Recommendations - South-East Asia",
        "pages": (40, 50),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Treatment South-East Asia",
        "diagnosis_type": "Visceral",
    },
    {
        "id": "SecondaryProphylaxis",
        "title": "Secondary Prophylaxis",
        "pages": (50, 60),
        "diagnosis": "Visceral leishmaniasis in HIV coinfection (WHO guideline) - Secondary Prophylaxis",
        "diagnosis_type": "Visceral",
    },
]


def create_who_entries(images: List[Dict], pdf_path: Path) -> List[Dict[str, Any]]:
    """
    Create section-level entries for WHO VL-HIV guideline.
    Extract text from each section based on page ranges.
    """
    print(f"    Extracting text from WHO PDF: {pdf_path.name}")
    
    # Get total pages
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
    print(f"    Total pages: {total_pages}")
    
    # Map images to sections based on content
    patient_photo = [img for img in images if "patient" in img.get("image_subtype", "").lower()]
    epidemiology_map = [img for img in images if "map" in img.get("image_subtype", "").lower() or "infographic" in img.get("image_subtype", "").lower()]
    
    entries = []
    
    for section in WHO_SECTIONS:
        start_page, end_page = section["pages"]
        
        # Clamp to actual page count
        if start_page >= total_pages:
            print(f"    Skipping {section['id']}: start page {start_page} >= total {total_pages}")
            continue
        end_page = min(end_page, total_pages)
        
        # Extract text for this section
        section_text = extract_pdf_text(pdf_path, start_page, end_page)
        section_text = clean_text(section_text)
        
        if not section_text.strip():
            print(f"    Warning: No text extracted for section {section['id']}")
            continue
        
        print(f"    {section['id']}: pages {start_page}-{end_page}, {len(section_text)} chars")
        
        # Assign images to relevant sections
        section_images = []
        if section["id"] == "Introduction":
            section_images = patient_photo
        elif section["id"] == "Epidemiology":
            section_images = epidemiology_map
        
        entry = {
            "case_id": f"WHO_VL_HIV_{section['id']}",
            "diagnosis": section["diagnosis"],
            "diagnosis_type": section["diagnosis_type"],
            "species": "Leishmania donovani",
            "confirmation_method": "Reference document",
            "evidence_span": f"WHO guideline - {section['title']}",
            "confidence": "high",
            "is_leishmaniasis": True,
            "case_text": section_text,
            "title": f"WHO VL-HIV Guideline - {section['title']}",
            "abstract": f"WHO guideline section on {section['title'].lower()} for VL in HIV-positive patients.",
            "images": section_images,
            "source_type": "guideline",
            "source": "WHO",
            "license": "CC BY-NC-SA 3.0 IGO"
        }
        entries.append(entry)
    
    return entries


def main():
    """Main preprocessing pipeline."""
    print("=" * 60)
    print("Guideline Preprocessor for RAG Training Data")
    print("Using pdfplumber for text extraction")
    print("=" * 60)
    
    # Check PDFs exist
    if not CDC_PDF.exists():
        print(f"ERROR: CDC PDF not found: {CDC_PDF}")
        return
    if not WHO_PDF.exists():
        print(f"ERROR: WHO PDF not found: {WHO_PDF}")
        return
    
    # Load GPT 5.2 image metadata
    cdc_jsonl = GPT52_DIR / "CDC - DPDx - Leishmaniasis.jsonl"
    who_jsonl = GPT52_DIR / "WHO GUIDELINE for the treatment of visceral leishmaniasis in HIV co-infected patients in East Africa and South-East Asia.jsonl"
    
    print(f"\n[1] Loading image metadata...")
    
    cdc_images_raw = load_jsonl(cdc_jsonl)
    who_images_raw = load_jsonl(who_jsonl)
    
    print(f"    CDC: {len(cdc_images_raw)} images")
    print(f"    WHO: {len(who_images_raw)} images")
    
    # Normalize image metadata
    print(f"\n[2] Normalizing image metadata...")
    
    cdc_images = normalize_image_metadata(cdc_images_raw, CDC_IMAGES_SUBDIR)
    who_images = normalize_image_metadata(who_images_raw, WHO_IMAGES_SUBDIR)
    
    print(f"    Labels converted from strings to arrays")
    print(f"    Relative paths added")
    
    # Generate entries with PDF text extraction
    print(f"\n[3] Extracting text from PDFs and generating entries...")
    
    entries = []
    
    # CDC: single document-level entry
    cdc_entry = create_cdc_entry(cdc_images, CDC_PDF)
    entries.append(cdc_entry)
    print(f"    CDC: 1 document-level entry created")
    
    # WHO: section-level entries
    who_entries = create_who_entries(who_images, WHO_PDF)
    entries.extend(who_entries)
    print(f"    WHO: {len(who_entries)} section-level entries created")
    
    # Write output
    print(f"\n[4] Writing output to {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')
    
    print(f"    Total entries: {len(entries)}")
    
    # Summary
    print("\n" + "=" * 60)
    print("PREPROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput: {OUTPUT_FILE}")
    print(f"Entries: {len(entries)} total")
    print(f"  - CDC DPDx: 1 entry ({len(cdc_images)} images)")
    print(f"  - WHO VL-HIV: {len(who_entries)} entries ({len(who_images)} images)")
    
    # Text statistics
    total_chars = sum(len(e.get("case_text", "")) for e in entries)
    avg_chars = total_chars // len(entries) if entries else 0
    print(f"\nText Statistics:")
    print(f"  - Total characters: {total_chars:,}")
    print(f"  - Average per entry: {avg_chars:,}")


if __name__ == "__main__":
    main()
