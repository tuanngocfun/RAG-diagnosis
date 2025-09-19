#!/usr/bin/env python3
"""
Fix text extraction for Qdrant records with empty text_excerpt fields.
This script will:
1. Find all points with empty text_excerpt
2. Extract text using improved OCR methods 
3. Update Qdrant with the extracted text
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import json

# Import from our RAG system
sys.path.append('/app')
from rag.test.medgemma4b_qdrant_bge_medcpt import (
    qdrant, find_case_pdf, read_pdf_page_text, ocr_png_tesseract,
    find_case_dir, CFG
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def improved_text_extraction(case_dir: Path, page_index: int, pdf_path: Optional[Path] = None) -> str:
    """
    Improved text extraction with multiple fallbacks.
    """
    # Try PDF text extraction first if PDF is available
    if pdf_path and pdf_path.exists():
        try:
            text = read_pdf_page_text(pdf_path, page_index)
            if text and len(text.strip()) > 50 and "OCR extraction not available" not in text:
                logger.info(f"✅ PDF text extracted for page {page_index}: {len(text)} chars")
                return text.strip()
        except Exception as e:
            logger.warning(f"PDF extraction failed for page {page_index}: {e}")
    
    # Fallback to OCR on the PNG file
    try:
        text = ocr_png_tesseract(case_dir, page_index)
        if text and len(text.strip()) > 20:
            logger.info(f"✅ OCR text extracted for page {page_index}: {len(text)} chars")
            return text.strip()
    except Exception as e:
        logger.warning(f"OCR extraction failed for page {page_index}: {e}")
    
    return ""

def fix_empty_text_excerpts(batch_size: int = 100):
    """
    Find and fix all Qdrant points with empty text_excerpt fields.
    """
    client = qdrant()
    
    # Get all points with empty text_excerpt
    logger.info("🔍 Scanning for points with empty text_excerpt...")
    
    points_to_fix = []
    offset = None
    total_scanned = 0
    
    while True:
        result = client.scroll(
            collection_name=CFG.COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,  # We don't need vectors
        )
        
        points, next_offset = result
        
        if not points:
            break
        
        for point in points:
            total_scanned += 1
            payload = point.payload
            text_excerpt = payload.get("text_excerpt", "")
            
            # Check if text_excerpt is empty or just placeholder
            if not text_excerpt or "OCR extraction not available" in text_excerpt:
                points_to_fix.append({
                    "id": point.id,
                    "doc_id": payload.get("doc_id"),
                    "page_index": payload.get("page_index"),
                    "payload": payload
                })
        
        offset = next_offset
        if offset is None:
            break
    
    logger.info(f"📊 Scanned {total_scanned} points, found {len(points_to_fix)} needing text extraction")
    
    # Fix each point
    fixed_count = 0
    for i, point_info in enumerate(points_to_fix):
        doc_id = point_info["doc_id"]
        page_index = point_info["page_index"]
        point_id = point_info["id"]
        
        if i % 10 == 0:
            logger.info(f"🔄 Processing {i+1}/{len(points_to_fix)}: {doc_id} page {page_index}")
        
        # Find case directory and PDF
        case_dir = find_case_dir(doc_id, CFG.EXTRACT_ROOT)
        if not case_dir:
            logger.warning(f"❌ Case directory not found for {doc_id}")
            continue
        
        pdf_path = find_case_pdf(case_dir)
        
        # Extract text
        text = improved_text_extraction(case_dir, page_index, pdf_path)
        
        if text:
            # Update the payload
            new_payload = point_info["payload"].copy()
            new_payload["text_excerpt"] = text
            
            # Update in Qdrant
            try:
                client.set_payload(
                    collection_name=CFG.COLLECTION,
                    payload=new_payload,
                    points=[point_id]
                )
                fixed_count += 1
                if fixed_count % 10 == 0:
                    logger.info(f"✅ Fixed {fixed_count} points so far")
            except Exception as e:
                logger.error(f"❌ Failed to update point {point_id}: {e}")
        else:
            logger.warning(f"⚠️ No text extracted for {doc_id} page {page_index}")
    
    logger.info(f"🎉 Completed! Fixed {fixed_count}/{len(points_to_fix)} points")
    return fixed_count

def main():
    """Main function"""
    logger.info("🚀 Starting text extraction fix...")
    
    try:
        fixed_count = fix_empty_text_excerpts(batch_size=50)
        logger.info(f"✅ Text extraction fix completed! Fixed {fixed_count} points.")
    except Exception as e:
        logger.error(f"❌ Text extraction fix failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()