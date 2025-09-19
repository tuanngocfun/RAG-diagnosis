#!/usr/bin/env python3
"""
Clean and re-index Qdrant collection properly.
This script will:
1. Drop the existing collection 
2. Create a fresh collection
3. Index all pages with proper duplicate prevention
4. Verify the indexing was successful
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
import time

# Import from our RAG system
sys.path.append('/app')
from rag.test.medgemma4b_qdrant_bge_medcpt import (
    qdrant, CFG, qdrant_init, qdrant_index, create_payload_indexes
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clean_and_reindex():
    """
    Perform a complete clean re-index of the Qdrant collection.
    """
    logger.info("🚀 Starting clean re-index process...")
    
    try:
        # Step 1: Initialize (recreate) collection
        logger.info("🗑️  Step 1: Recreating Qdrant collection...")
        qdrant_init()
        logger.info("✅ Collection recreated successfully")
        
        # Wait a moment for Qdrant to settle
        time.sleep(2)
        
        # Step 2: Index all pages
        logger.info("📥 Step 2: Indexing all pages...")
        qdrant_index()
        logger.info("✅ Indexing completed successfully")
        
        # Step 3: Create payload indexes for performance
        logger.info("🔍 Step 3: Creating payload indexes...")
        client = qdrant()
        create_payload_indexes(client)
        logger.info("✅ Payload indexes created successfully")
        
        # Step 4: Verify the result
        logger.info("🔍 Step 4: Verifying collection...")
        collection_info = client.get_collection(CFG.COLLECTION)
        points_count = collection_info.points_count
        logger.info(f"📊 Collection now contains {points_count} points")
        
        # Quick duplicate check
        logger.info("🔍 Quick duplicate check...")
        sample_result = client.scroll(
            collection_name=CFG.COLLECTION,
            limit=1000,
            with_payload=True,
            with_vectors=False,
        )
        
        sample_points = sample_result[0]
        uids = set()
        doc_page_combos = set()
        duplicates_found = False
        
        for point in sample_points:
            payload = point.payload
            uid = payload.get("uid", "")
            doc_id = payload.get("doc_id", "")
            page_index = payload.get("page_index", -1)
            combo = f"{doc_id}__p{page_index:04d}"
            
            if uid in uids:
                logger.error(f"❌ Duplicate UID found in sample: {uid}")
                duplicates_found = True
            if combo in doc_page_combos:
                logger.error(f"❌ Duplicate doc+page found in sample: {combo}")
                duplicates_found = True
            
            uids.add(uid)
            doc_page_combos.add(combo)
        
        if not duplicates_found:
            logger.info("✅ No duplicates found in sample - indexing appears successful!")
        else:
            logger.error("❌ Duplicates still found - there may be an issue with the indexing logic")
        
        logger.info("🎉 Clean re-index process completed!")
        return not duplicates_found
        
    except Exception as e:
        logger.error(f"❌ Clean re-index failed: {e}")
        return False

def main():
    """Main function"""
    logger.info("🚀 Starting clean re-index...")
    
    success = clean_and_reindex()
    
    if success:
        logger.info("✅ Clean re-index completed successfully!")
        logger.info("🔍 Run the diagnostic script to verify: python rag/curate/diagnose_qdrant_duplicates.py")
    else:
        logger.error("❌ Clean re-index failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()