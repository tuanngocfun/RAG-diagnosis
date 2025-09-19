#!/usr/bin/env python3
"""
Diagnose Qdrant collection for duplicate entries and indexing issues.
This script will:
1. Check for duplicate UIDs (should never happen with proper UUID5)
2. Check for duplicate doc_id + page_index combinations
3. Check for duplicate image paths
4. Count points per case to identify over-indexing
5. Show statistics about the collection
"""

import sys
import logging
from pathlib import Path
from typing import List, Dict, Any, Set
from collections import defaultdict, Counter
import json

# Import from our RAG system
sys.path.append('/app')
from rag.test.medgemma4b_qdrant_medcpt import (
    qdrant, CFG, build_uid
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def diagnose_qdrant_duplicates():
    """
    Comprehensive diagnosis of Qdrant collection for duplicate entries.
    """
    client = qdrant()
    
    logger.info("🔍 Starting Qdrant duplicate diagnosis...")
    
    # Collect all points
    all_points = []
    offset = None
    batch_size = 1000
    
    logger.info("📥 Collecting all points from Qdrant...")
    while True:
        result = client.scroll(
            collection_name=CFG.COLLECTION,
            limit=batch_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        
        points, next_offset = result
        
        if not points:
            break
            
        all_points.extend(points)
        logger.info(f"  📊 Collected {len(all_points)} points so far...")
        
        offset = next_offset
        if offset is None:
            break
    
    logger.info(f"📊 Total points collected: {len(all_points)}")
    
    # Analysis containers
    uid_counts = Counter()
    doc_page_combinations = Counter()
    image_path_counts = Counter()
    doc_id_counts = Counter()
    
    # Track for detailed analysis
    duplicate_uids = defaultdict(list)
    duplicate_doc_pages = defaultdict(list)
    duplicate_image_paths = defaultdict(list)
    
    logger.info("🔍 Analyzing points for duplicates...")
    
    for point in all_points:
        payload = point.payload
        point_id = point.id
        
        # Extract key fields
        uid = payload.get("uid", "")
        doc_id = payload.get("doc_id", "")
        page_index = payload.get("page_index", -1)
        image_path = payload.get("image_path", "")
        
        # Count occurrences
        uid_counts[uid] += 1
        doc_page_combo = f"{doc_id}__p{page_index:04d}"
        doc_page_combinations[doc_page_combo] += 1
        image_path_counts[image_path] += 1
        doc_id_counts[doc_id] += 1
        
        # Track duplicates
        if uid_counts[uid] > 1:
            duplicate_uids[uid].append({
                "point_id": point_id,
                "doc_id": doc_id,
                "page_index": page_index,
                "image_path": image_path
            })
        
        if doc_page_combinations[doc_page_combo] > 1:
            duplicate_doc_pages[doc_page_combo].append({
                "point_id": point_id,
                "uid": uid,
                "doc_id": doc_id,
                "page_index": page_index,
                "image_path": image_path
            })
        
        if image_path_counts[image_path] > 1:
            duplicate_image_paths[image_path].append({
                "point_id": point_id,
                "uid": uid,
                "doc_id": doc_id,
                "page_index": page_index
            })
    
    # Generate report
    logger.info("📋 DUPLICATE ANALYSIS REPORT")
    logger.info("=" * 50)
    
    # 1. UID duplicates (should NEVER happen with UUID5)
    uid_duplicates = {uid: count for uid, count in uid_counts.items() if count > 1}
    logger.info(f"❌ Duplicate UIDs found: {len(uid_duplicates)}")
    if uid_duplicates:
        logger.error("🚨 CRITICAL: Found duplicate UIDs - this should never happen with UUID5!")
        for uid, count in sorted(uid_duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.error(f"  UID {uid}: {count} duplicates")
            for dup in duplicate_uids[uid][:3]:
                logger.error(f"    - Point {dup['point_id']}: {dup['doc_id']} page {dup['page_index']}")
    
    # 2. Doc+Page duplicates
    doc_page_duplicates = {combo: count for combo, count in doc_page_combinations.items() if count > 1}
    logger.info(f"❌ Duplicate doc_id+page_index combinations: {len(doc_page_duplicates)}")
    if doc_page_duplicates:
        logger.warning("🔥 Found duplicate doc_id+page_index combinations!")
        for combo, count in sorted(doc_page_duplicates.items(), key=lambda x: x[1], reverse=True)[:10]:
            logger.warning(f"  {combo}: {count} duplicates")
            for dup in duplicate_doc_pages[combo][:3]:
                logger.warning(f"    - Point {dup['point_id']}: UID {dup['uid']}")
    
    # 3. Image path duplicates
    image_duplicates = {path: count for path, count in image_path_counts.items() if count > 1}
    logger.info(f"❌ Duplicate image paths: {len(image_duplicates)}")
    if image_duplicates:
        for path, count in sorted(image_duplicates.items(), key=lambda x: x[1], reverse=True)[:5]:
            logger.warning(f"  {path}: {count} duplicates")
    
    # 4. Per-case statistics
    logger.info("📊 CASE STATISTICS")
    logger.info("=" * 30)
    logger.info(f"Total unique cases: {len(doc_id_counts)}")
    
    # Show cases with most pages
    top_cases = sorted(doc_id_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    logger.info("📄 Cases with most pages:")
    for doc_id, count in top_cases:
        logger.info(f"  {doc_id}: {count} pages")
    
    # Check for suspiciously high page counts
    suspicious_cases = [(doc_id, count) for doc_id, count in doc_id_counts.items() if count > 50]
    if suspicious_cases:
        logger.warning(f"⚠️  {len(suspicious_cases)} cases with >50 pages (possibly over-indexed):")
        for doc_id, count in suspicious_cases[:5]:
            logger.warning(f"  {doc_id}: {count} pages")
    
    # 5. Generate expected UIDs for verification
    logger.info("🔍 EXPECTED UID VERIFICATION")
    logger.info("=" * 35)
    
    # Sample a few points and verify their UIDs
    verification_errors = []
    sample_points = all_points[:min(100, len(all_points))]
    
    for point in sample_points:
        payload = point.payload
        uid = payload.get("uid", "")
        doc_id = payload.get("doc_id", "")
        page_index = payload.get("page_index", -1)
        
        if doc_id and page_index >= 0:
            expected_uid = build_uid(doc_id, page_index)
            if uid != expected_uid:
                verification_errors.append({
                    "point_id": point.id,
                    "actual_uid": uid,
                    "expected_uid": expected_uid,
                    "doc_id": doc_id,
                    "page_index": page_index
                })
    
    if verification_errors:
        logger.error(f"🚨 Found {len(verification_errors)} UID verification errors!")
        for error in verification_errors[:5]:
            logger.error(f"  Point {error['point_id']}: got {error['actual_uid']}, expected {error['expected_uid']}")
    else:
        logger.info("✅ UID verification passed for sample points")
    
    # Summary and recommendations
    logger.info("🎯 SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 40)
    
    total_issues = len(uid_duplicates) + len(doc_page_duplicates) + len(image_duplicates) + len(verification_errors)
    
    if total_issues == 0:
        logger.info("✅ No duplicate issues found! Collection looks healthy.")
    else:
        logger.warning(f"❌ Found {total_issues} types of duplicate issues")
        
        if uid_duplicates or verification_errors:
            logger.error("🚨 CRITICAL: UID generation issues detected!")
            logger.error("   Recommendation: Full re-index required")
        
        if doc_page_duplicates:
            logger.warning("🔥 Duplicate doc+page combinations found")
            logger.warning("   Recommendation: Clean duplicates or re-index")
        
        if image_duplicates:
            logger.warning("📷 Duplicate image paths found")
            logger.warning("   Recommendation: Check indexing logic")
    
    # Save detailed report
    report_path = Path("/app/kaggle/working2/rag_knowledge_base/duplicate_analysis_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = {
        "total_points": len(all_points),
        "unique_cases": len(doc_id_counts),
        "duplicate_uids": len(uid_duplicates),
        "duplicate_doc_pages": len(doc_page_duplicates),
        "duplicate_image_paths": len(image_duplicates),
        "verification_errors": len(verification_errors),
        "top_cases": top_cases[:20],
        "suspicious_cases": suspicious_cases,
        "sample_duplicates": {
            "uids": dict(list(uid_duplicates.items())[:5]),
            "doc_pages": dict(list(doc_page_duplicates.items())[:5]),
            "image_paths": dict(list(image_duplicates.items())[:5])
        }
    }
    
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    logger.info(f"📄 Detailed report saved to: {report_path}")
    
    return total_issues == 0

def main():
    """Main function"""
    logger.info("🚀 Starting Qdrant duplicate diagnosis...")
    
    try:
        is_healthy = diagnose_qdrant_duplicates()
        if is_healthy:
            logger.info("✅ Qdrant collection is healthy!")
            sys.exit(0)
        else:
            logger.error("❌ Qdrant collection has duplicate issues!")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Diagnosis failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()