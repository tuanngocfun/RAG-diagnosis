#!/usr/bin/env python3
"""
Re-index script with improved text extraction for pages with poor text coverage.
Run this to update the Qdrant collection with better text excerpts.
"""

import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

from rag.test.gemini25pro_qdrant_bge import (
    CFG, qdrant, CQ2, is_meaningful_text, 
    discover_cases, list_pages_full, build_payload,
    ocr_png_tesseract, ocr_png_fallback
)
from PIL import Image
import logging

logging.basicConfig(level=logging.INFO)

def update_poor_text_pages():
    """Update pages in Qdrant that have poor text extraction."""
    client = qdrant()
    encoder = CQ2(CFG.RET_MODEL_ID)
    
    # Get all points with poor text coverage
    poor_text_filter = {
        "must": [
            {
                "key": "text_excerpt",
                "match": {"value": ""}
            }
        ]
    }
    
    # Scroll through all points to find ones with poor text
    updated_count = 0
    
    for case_dir in discover_cases(CFG.EXTRACT_ROOT)[:5]:  # Start with first 5 cases
        print(f"Processing case: {case_dir.name}")
        
        for page_path in list_pages_full(case_dir):
            page_idx = int(page_path.stem.split('_')[-1]) - 1  # Convert from 1-based to 0-based
            
            try:
                img = Image.open(page_path)
                
                # Build new payload with improved text extraction
                payload = build_payload(case_dir, page_path, page_idx, img)
                
                # Only update if we have meaningful text now
                if is_meaningful_text(payload.get("text_excerpt", "")):
                    # Encode the improved text
                    text_for_embedding = f"{payload['case_title']} {payload['text_excerpt']}"
                    tv = encoder.embed_texts([text_for_embedding])[0]
                    iv = encoder.embed_images([img])[0]
                    
                    # Update the point in Qdrant
                    uid = payload["uid"]
                    client.upsert(
                        collection_name=CFG.COLLECTION,
                        points=[{
                            "id": uid,
                            "vector": {
                                "text": tv.tolist(),
                                "image": iv.tolist()
                            },
                            "payload": payload
                        }]
                    )
                    
                    updated_count += 1
                    if updated_count % 10 == 0:
                        print(f"Updated {updated_count} pages...")
                        
            except Exception as e:
                print(f"Error processing {page_path}: {e}")
                continue
    
    print(f"Updated {updated_count} pages with improved text extraction")

if __name__ == "__main__":
    update_poor_text_pages()