#!/usr/bin/env python3
"""
Article Preprocessor for RAG Training Set Augmentation

This script preprocesses article PDFs to create entries compatible with
the Leishmaniasis RAG training set. It:
1. Extracts text from PDFs using pdfplumber with section-based chunking
2. Loads GPT5.2 annotations for image metadata
3. Generates JSON entries matching train.jsonl schema
4. Copies images to the appropriate images directory

Based on LLM consensus (GPT5.2, Gemini3Pro, Grok4.1):
- Use GPT5.2 exclusively for images/captions
- Section-based chunking (not full-text blobs)
- Add source_type field to distinguish from case reports
- Remove References, headers/footers
"""

import json
import hashlib
import re
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import pdfplumber
except ImportError:
    print("Installing pdfplumber...")
    import subprocess
    subprocess.run(["pip3", "install", "--user", "pdfplumber"], check=True)
    import pdfplumber


# Configuration
ARTICLES_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/articles")
GPT52_METADATA_DIR = ARTICLES_DIR / "gpt52extendedthinking_images_metadata"
OUTPUT_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/leishmaniasis_verified")
IMAGES_OUTPUT_DIR = Path("/home/students/Leishmania/Leishmania_v3/data/leishmaniasis_images/images")

# Sections to remove (noise for RAG)
SECTIONS_TO_REMOVE = [
    "references", "acknowledgements", "acknowledgments", "author contributions",
    "competing interests", "conflict of interest", "funding", "data availability",
    "supplementary", "supporting information", "author information"
]

# Header/footer patterns to remove
NOISE_PATTERNS = [
    r"PLOS\s+Neglected\s+Tropical\s+Diseases",
    r"https?://dx\.doi\.org/\S+",
    r"https?://journals\.plos\.org/\S+",
    r"Page\s+\d+\s+of\s+\d+",
    r"^\d+\s*$",  # Page numbers
    r"www\.\S+",
    r"Copyright.*?\d{4}",
    r"This is an open access article.*",
    r"Published:.*\d{4}",
]


def generate_doc_id(doi: str) -> str:
    """Generate a canonical document ID from DOI."""
    # Clean DOI and create hash
    clean_doi = doi.replace("/", "_").replace(".", "_")
    hash_part = hashlib.sha256(doi.encode()).hexdigest()[:8]
    return f"article_{hash_part}"


def extract_title_from_filename(filename: str) -> str:
    """Extract article title from PDF filename."""
    # Remove .pdf extension and clean up
    title = Path(filename).stem
    # Replace underscores/hyphens with spaces
    title = re.sub(r"[-_]+", " ", title)
    return title.strip()


def clean_text(text: str) -> str:
    """Remove headers, footers, and noise patterns from text."""
    lines = text.split("\n")
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines matching noise patterns
        skip = False
        for pattern in NOISE_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                skip = True
                break
        
        if not skip:
            cleaned_lines.append(line)
    
    return "\n".join(cleaned_lines)


def identify_section(text: str) -> Optional[str]:
    """Identify common section headers."""
    section_keywords = {
        "abstract": ["abstract"],
        "introduction": ["introduction", "background"],
        "methods": ["methods", "methodology", "materials and methods", "study design"],
        "results": ["results", "findings"],
        "discussion": ["discussion"],
        "conclusion": ["conclusion", "conclusions", "summary"],
    }
    
    text_lower = text.lower().strip()
    for section, keywords in section_keywords.items():
        for keyword in keywords:
            if text_lower == keyword or text_lower.startswith(keyword + ":"):
                return section
    return None


def should_skip_section(text: str) -> bool:
    """Check if this section should be skipped (References, etc.)."""
    text_lower = text.lower().strip()
    for skip_section in SECTIONS_TO_REMOVE:
        if text_lower == skip_section or text_lower.startswith(skip_section):
            return True
    return False


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, str]:
    """
    Extract text from PDF with section-based segmentation.
    Returns dict with 'full_text', 'abstract', 'sections'.
    """
    result = {
        "full_text": "",
        "abstract": "",
        "sections": {},
        "title": ""
    }
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            all_text = []
            current_section = "introduction"  # Default section
            section_text = {current_section: []}
            in_skip_section = False
            
            for page_num, page in enumerate(pdf.pages):
                page_text = page.extract_text() or ""
                page_text = clean_text(page_text)
                
                # Process line by line to identify sections
                for line in page_text.split("\n"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Check if this is a section header
                    section = identify_section(line)
                    if section:
                        current_section = section
                        in_skip_section = False
                        if current_section not in section_text:
                            section_text[current_section] = []
                        continue
                    
                    # Check if entering a section to skip
                    if should_skip_section(line):
                        in_skip_section = True
                        continue
                    
                    if not in_skip_section:
                        all_text.append(line)
                        if current_section not in section_text:
                            section_text[current_section] = []
                        section_text[current_section].append(line)
            
            result["full_text"] = "\n".join(all_text)
            result["sections"] = {k: "\n".join(v) for k, v in section_text.items() if v}
            
            # Extract abstract if found
            if "abstract" in section_text and section_text["abstract"]:
                result["abstract"] = "\n".join(section_text["abstract"])
            
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {e}")
        result["full_text"] = f"[Error extracting PDF: {e}]"
    
    return result


def load_gpt52_annotations(jsonl_path: Path) -> List[Dict]:
    """Load GPT5.2 image annotations from JSONL file."""
    annotations = []
    try:
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    annotations.append(json.loads(line))
    except Exception as e:
        print(f"Error loading annotations from {jsonl_path}: {e}")
    return annotations


def find_matching_annotations(pdf_stem: str) -> tuple[List[Dict], Path]:
    """Find GPT5.2 annotations matching a PDF file."""
    # GPT5.2 uses truncated and normalized names
    # Try to find matching JSONL file
    for jsonl_file in GPT52_METADATA_DIR.glob("*.jsonl"):
        # Normalize both names for comparison
        jsonl_stem = jsonl_file.stem.lower().replace("_figures_only", "")
        pdf_normalized = pdf_stem.lower().replace(" ", "_").replace("-", "_")
        
        # Check for partial match
        if pdf_normalized[:30] in jsonl_stem or jsonl_stem[:30] in pdf_normalized:
            annotations = load_gpt52_annotations(jsonl_file)
            images_dir = GPT52_METADATA_DIR / f"{jsonl_file.stem}_images"
            return annotations, images_dir
    
    return [], Path()


def format_images_for_schema(annotations: List[Dict], images_src_dir: Path, 
                             doc_id: str, images_dest_dir: Path) -> List[Dict]:
    """
    Format annotations to match train.jsonl image schema.
    Also copies images to destination directory.
    """
    formatted_images = []
    
    # Create destination directory
    dest_dir = images_dest_dir / doc_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for i, ann in enumerate(annotations):
        src_file = images_src_dir / ann.get("file", "")
        
        # Generate canonical filename
        new_filename = f"{doc_id}_fig{i+1:02d}.webp"
        dest_file = dest_dir / new_filename
        
        # Copy image if source exists
        if src_file.exists():
            shutil.copy2(src_file, dest_file)
            print(f"  Copied: {src_file.name} -> {dest_file}")
        else:
            print(f"  Warning: Source image not found: {src_file}")
        
        formatted_images.append({
            "file": f"{doc_id}/{new_filename}",
            "file_id": ann.get("file_id", f"file_{doc_id}_{i+1:04d}"),
            "caption": ann.get("caption", ""),
            "image_type": ann.get("image_type", "medical_photograph"),
            "image_subtype": ann.get("image_subtype", ""),
            "radiology_region": None,
            "radiology_view": None,
            "labels_supervised": ann.get("labels_supervised", "[]"),
            "labels_semisupervised": ann.get("labels_semisupervised", "[]")
        })
    
    return formatted_images


def determine_diagnosis_from_title(title: str) -> tuple[str, str]:
    """Determine diagnosis type from article title."""
    title_lower = title.lower()
    
    if "pkdl" in title_lower or "post-kala-azar" in title_lower or "post kala-azar" in title_lower:
        return "Post-kala-azar Dermal Leishmaniasis (PKDL)", "Post-PKDL"
    elif "mucocutaneous" in title_lower or "mcl" in title_lower:
        return "Mucocutaneous Leishmaniasis", "Mucocutaneous"
    elif "visceral" in title_lower or "kala-azar" in title_lower:
        return "Visceral Leishmaniasis", "Visceral"
    elif "cutaneous" in title_lower:
        return "Cutaneous Leishmaniasis", "Cutaneous"
    else:
        return "Leishmaniasis", "Reference"


def process_article(pdf_path: Path) -> Optional[Dict]:
    """
    Process a single article PDF and return a train.jsonl compatible entry.
    """
    print(f"\nProcessing: {pdf_path.name}")
    
    # Extract title from filename
    title = extract_title_from_filename(pdf_path.name)
    print(f"  Title: {title}")
    
    # Find GPT5.2 annotations
    annotations, images_src_dir = find_matching_annotations(pdf_path.stem)
    print(f"  Found {len(annotations)} image annotations")
    
    # Get DOI from annotations if available
    doi = annotations[0].get("patient_id", "") if annotations else ""
    if not doi:
        # Generate from filename hash
        doi = hashlib.sha256(pdf_path.name.encode()).hexdigest()[:16]
    
    doc_id = generate_doc_id(doi)
    print(f"  Document ID: {doc_id}")
    
    # Extract text from PDF
    extracted = extract_text_from_pdf(pdf_path)
    print(f"  Extracted {len(extracted['full_text'])} chars, {len(extracted['sections'])} sections")
    
    # Determine diagnosis from title
    diagnosis, diagnosis_type = determine_diagnosis_from_title(title)
    
    # Format case_text with document type prefix (per LLM suggestions)
    case_text = f"[MEDICAL REVIEW ARTICLE]\n"
    case_text += f"Title: {title}\n\n"
    
    # Add sections with headers
    for section_name, section_content in extracted["sections"].items():
        if section_content.strip():
            case_text += f"## {section_name.upper()}\n{section_content}\n\n"
    
    # Format images
    formatted_images = []
    if annotations and images_src_dir.exists():
        formatted_images = format_images_for_schema(
            annotations, images_src_dir, doc_id, IMAGES_OUTPUT_DIR
        )
    
    # Determine license (from annotations or default)
    license_type = annotations[0].get("license", "CC BY") if annotations else "CC BY"
    
    # Build entry matching train.jsonl schema
    entry = {
        "case_id": doc_id,
        "diagnosis": diagnosis,
        "diagnosis_type": diagnosis_type,
        "species": "",
        "confirmation_method": "",
        "evidence_span": "",
        "confidence": "high",
        "is_leishmaniasis": True,
        "raw_llm_response": None,
        "error": None,
        "case_text": case_text.strip(),
        "title": title,
        "abstract": extracted.get("abstract", ""),
        "images": formatted_images,
        "journal": "PLoS NTD / Review Article",
        "year": "",
        "doi": doi,
        "license": license_type,
        # New fields added per LLM suggestions
        "source_type": "review_article",
        "license_source": f"https://doi.org/{doi}" if doi.startswith("10.") else ""
    }
    
    return entry


def process_all_articles() -> List[Dict]:
    """Process all PDF articles in the articles directory."""
    entries = []
    
    pdf_files = list(ARTICLES_DIR.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files to process")
    
    for pdf_path in pdf_files:
        try:
            entry = process_article(pdf_path)
            if entry:
                entries.append(entry)
        except Exception as e:
            print(f"Error processing {pdf_path.name}: {e}")
            import traceback
            traceback.print_exc()
    
    return entries


def save_entries(entries: List[Dict], output_path: Path):
    """Save entries as JSONL file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"\nSaved {len(entries)} entries to {output_path}")


def main():
    """Main entry point."""
    print("=" * 60)
    print("Article Preprocessor for RAG Training Set")
    print("=" * 60)
    
    # Ensure output directories exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMAGES_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Process all articles
    entries = process_all_articles()
    
    if not entries:
        print("\nNo entries generated. Check the PDF files and annotations.")
        return
    
    # Save to separate file first (for review)
    articles_output = OUTPUT_DIR / "articles_processed.jsonl"
    save_entries(entries, articles_output)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Processed: {len(entries)} articles")
    total_images = sum(len(e.get("images", [])) for e in entries)
    print(f"Total images: {total_images}")
    print(f"Output file: {articles_output}")
    print("\nNext step: Run integrate_articles_guidelines.py to merge with train.jsonl")


if __name__ == "__main__":
    main()
