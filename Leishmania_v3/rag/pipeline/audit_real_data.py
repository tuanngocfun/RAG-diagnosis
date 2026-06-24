"""
Data Audit Script - Verify real dataset is being used.

Per GPT 5.2 & Gemini 3 Pro feedback: MANDATORY audit before any evaluation runs.

Checks:
1. Existence of DATA_ROOT and IMAGES_DIR
2. Counts for train.jsonl and test.jsonl
3. Train/Test DISJOINTNESS (critical for scientific validity)
4. Image file existence on disk (sample verification)
5. Caption quality (empty/null rate)
6. Environment configuration summary
"""
import json
import os
import random
from pathlib import Path
from collections import Counter
from typing import Dict, List, Set, Tuple

# Import from package to ensure HF cache is set
from . import DATA_ROOT, TRAIN_JSONL, TEST_JSONL, QDRANT_URL, QDRANT_API_KEY


# Derive paths
IMAGES_DIR = DATA_ROOT / "images"


def log_env_summary() -> None:
    """Print environment configuration for reproducibility."""
    print("=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"  HF_HOME:                  {os.environ.get('HF_HOME', 'NOT SET')}")
    print(f"  TRANSFORMERS_CACHE:       {os.environ.get('TRANSFORMERS_CACHE', 'NOT SET')}")
    print(f"  SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'NOT SET')}")
    print(f"  DATA_ROOT:                {DATA_ROOT}")
    print(f"  QDRANT_URL:               {QDRANT_URL}")
    print("=" * 60)


def load_cases(jsonl_path: Path) -> List[Dict]:
    """Load cases from JSONL file."""
    if not jsonl_path.exists():
        return []
    with open(jsonl_path) as f:
        return [json.loads(line) for line in f]


def check_disjointness(
    train_cases: List[Dict],
    test_cases: List[Dict]
) -> Tuple[bool, Set[str], Set[str]]:
    """
    Check train/test disjointness by case_id and pmc_id/article_id.
    
    Returns:
        (is_disjoint, overlapping_case_ids, overlapping_pmc_ids)
    """
    train_case_ids = {c.get("case_id") for c in train_cases if c.get("case_id")}
    test_case_ids = {c.get("case_id") for c in test_cases if c.get("case_id")}
    
    # Extract PMC IDs (may be in case_id like PMC123456_01 or separate field)
    def extract_pmc_id(case: Dict) -> str:
        # Try pmc_id field first
        if case.get("pmc_id"):
            return case["pmc_id"]
        # Try to extract from case_id (format: PMC123456_01)
        case_id = case.get("case_id", "")
        if case_id.startswith("PMC"):
            return case_id.split("_")[0]
        return ""
    
    train_pmc_ids = {extract_pmc_id(c) for c in train_cases if extract_pmc_id(c)}
    test_pmc_ids = {extract_pmc_id(c) for c in test_cases if extract_pmc_id(c)}
    
    overlap_case_ids = train_case_ids & test_case_ids
    overlap_pmc_ids = train_pmc_ids & test_pmc_ids
    
    is_disjoint = len(overlap_case_ids) == 0 and len(overlap_pmc_ids) == 0
    
    return is_disjoint, overlap_case_ids, overlap_pmc_ids


def check_image_existence(
    cases: List[Dict],
    sample_size: int = 10
) -> Tuple[int, int, List[str]]:
    """
    Check if images exist on disk.
    
    Images are stored in: IMAGES_DIR/{case_id}/{filename}
    Image info uses 'file' key (not 'file_name')
    
    Returns:
        (total_images, existing_images, sample_missing_paths)
    """
    all_image_paths = []
    
    for case in cases:
        case_id = case.get("case_id", "")
        for img in case.get("images", []):
            # Try 'file' first, then 'file_name' for compatibility
            file_name = img.get("file") or img.get("file_name", "")
            if file_name:
                # Images are in subdirectory by case_id
                all_image_paths.append(IMAGES_DIR / case_id / file_name)
    
    existing = 0
    missing_paths = []
    
    for img_path in all_image_paths:
        if img_path.exists():
            existing += 1
        else:
            missing_paths.append(str(img_path))
    
    # Sample missing paths for display
    sample_missing = missing_paths[:sample_size] if missing_paths else []
    
    return len(all_image_paths), existing, sample_missing


def check_caption_quality(cases: List[Dict]) -> Dict[str, float]:
    """
    Check caption quality metrics.
    
    Returns:
        Dict with empty_rate, avg_length, etc.
    """
    all_captions = []
    
    for case in cases:
        for img in case.get("images", []):
            caption = img.get("caption", "")
            all_captions.append(caption)
    
    if not all_captions:
        return {"total": 0, "empty_rate": 0, "avg_length": 0}
    
    empty_count = sum(1 for c in all_captions if not c or c.strip() == "")
    total_length = sum(len(c) for c in all_captions)
    
    return {
        "total": len(all_captions),
        "empty_rate": empty_count / len(all_captions),
        "avg_length": total_length / len(all_captions) if all_captions else 0
    }


def check_qdrant_collections() -> Dict[str, int]:
    """Check Qdrant collection point counts."""
    try:
        from qdrant_client import QdrantClient
        client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        
        collections = {}
        for col_name in ["cases_text_e5", "captions_biomedclip", "images_biomedclip",
                         "cases_text_e5_1024", "captions_biomedclip_512", "images_biomedclip_512"]:
            try:
                info = client.get_collection(col_name)
                collections[col_name] = info.points_count
            except:
                collections[col_name] = -1  # Not found
        
        return collections
    except Exception as e:
        return {"error": str(e)}


def run_audit() -> Dict:
    """
    Run full data audit.
    
    Returns:
        Dict with all audit results
    """
    print("=" * 70)
    print("DATA AUDIT - Verifying Real Dataset Integrity")
    print("=" * 70)
    
    results = {
        "passed": True,
        "checks": {}
    }
    
    # ========================================
    # 1. PATH EXISTENCE
    # ========================================
    print(f"\n{'='*60}")
    print("[1] PATH EXISTENCE")
    print(f"{'='*60}")
    
    data_root_exists = DATA_ROOT.exists()
    images_dir_exists = IMAGES_DIR.exists()
    train_exists = TRAIN_JSONL.exists()
    test_exists = TEST_JSONL.exists()
    
    print(f"  DATA_ROOT:   {DATA_ROOT.resolve()}")
    print(f"               {'✓ EXISTS' if data_root_exists else '✗ MISSING'}")
    print(f"  IMAGES_DIR:  {IMAGES_DIR}")
    print(f"               {'✓ EXISTS' if images_dir_exists else '✗ MISSING'}")
    print(f"  TRAIN_JSONL: {TRAIN_JSONL}")
    print(f"               {'✓ EXISTS' if train_exists else '✗ MISSING'}")
    print(f"  TEST_JSONL:  {TEST_JSONL}")
    print(f"               {'✓ EXISTS' if test_exists else '✗ MISSING'}")
    
    results["checks"]["paths"] = {
        "data_root": data_root_exists,
        "images_dir": images_dir_exists,
        "train_jsonl": train_exists,
        "test_jsonl": test_exists
    }
    
    if not all([data_root_exists, train_exists]):
        results["passed"] = False
        print("\n  ✗ CRITICAL: Required paths missing!")
        return results
    
    # ========================================
    # 2. CASE COUNTS
    # ========================================
    print(f"\n{'='*60}")
    print("[2] CASE COUNTS")
    print(f"{'='*60}")
    
    train_cases = load_cases(TRAIN_JSONL)
    test_cases = load_cases(TEST_JSONL) if test_exists else []
    
    train_with_images = sum(1 for c in train_cases if c.get("images"))
    test_with_images = sum(1 for c in test_cases if c.get("images"))
    
    print(f"  Train cases:       {len(train_cases)}")
    print(f"    with images:     {train_with_images}")
    print(f"  Test cases:        {len(test_cases)}")
    print(f"    with images:     {test_with_images}")
    
    results["checks"]["counts"] = {
        "train_total": len(train_cases),
        "train_with_images": train_with_images,
        "test_total": len(test_cases),
        "test_with_images": test_with_images
    }
    
    # ========================================
    # 3. TRAIN/TEST DISJOINTNESS (CRITICAL)
    # ========================================
    print(f"\n{'='*60}")
    print("[3] TRAIN/TEST DISJOINTNESS (Scientific Requirement)")
    print(f"{'='*60}")
    
    if test_cases:
        is_disjoint, overlap_case_ids, overlap_pmc_ids = check_disjointness(
            train_cases, test_cases
        )
        
        if is_disjoint:
            print("  ✓ PASSED: Train and Test sets are DISJOINT")
            print(f"    - No overlapping case_ids")
            print(f"    - No overlapping PMC/article IDs")
        else:
            results["passed"] = False
            print("  ✗ FAILED: DATA LEAKAGE DETECTED!")
            if overlap_case_ids:
                print(f"    Overlapping case_ids ({len(overlap_case_ids)}):")
                for cid in list(overlap_case_ids)[:5]:
                    print(f"      - {cid}")
            if overlap_pmc_ids:
                print(f"    Overlapping PMC IDs ({len(overlap_pmc_ids)}):")
                for pid in list(overlap_pmc_ids)[:5]:
                    print(f"      - {pid}")
        
        results["checks"]["disjointness"] = {
            "is_disjoint": is_disjoint,
            "overlapping_case_ids": list(overlap_case_ids),
            "overlapping_pmc_ids": list(overlap_pmc_ids)
        }
    else:
        print("  ⚠ SKIPPED: No test.jsonl found")
        results["checks"]["disjointness"] = {"is_disjoint": None, "reason": "no test set"}
    
    # ========================================
    # 4. IMAGE FILE VERIFICATION
    # ========================================
    print(f"\n{'='*60}")
    print("[4] IMAGE FILE VERIFICATION")
    print(f"{'='*60}")
    
    if images_dir_exists:
        # Count actual files in images dir (including webp)
        image_files = list(IMAGES_DIR.glob("*.*"))
        image_extensions = Counter(f.suffix.lower() for f in image_files)
        
        print(f"  Files in IMAGES_DIR: {len(image_files)}")
        print(f"  By extension:")
        for ext, count in image_extensions.most_common():
            print(f"    {ext}: {count}")
        
        # Check referenced images from JSONL
        all_cases = train_cases + test_cases
        total_referenced, existing, missing_sample = check_image_existence(all_cases)
        
        existence_rate = existing / total_referenced if total_referenced > 0 else 0
        
        print(f"\n  Referenced in JSONL:")
        print(f"    Total:    {total_referenced}")
        print(f"    Existing: {existing} ({existence_rate*100:.1f}%)")
        print(f"    Missing:  {total_referenced - existing}")
        
        if missing_sample:
            print(f"\n  Sample missing paths:")
            for mp in missing_sample[:5]:
                print(f"    - {mp}")
        
        if existence_rate < 0.9:
            print("\n  ⚠ WARNING: >10% images missing from disk!")
            results["passed"] = False
        
        results["checks"]["images"] = {
            "files_in_dir": len(image_files),
            "extensions": dict(image_extensions),
            "referenced": total_referenced,
            "existing": existing,
            "existence_rate": existence_rate
        }
    else:
        print("  ✗ IMAGES_DIR does not exist!")
        results["checks"]["images"] = {"error": "directory missing"}
        results["passed"] = False
    
    # ========================================
    # 5. CAPTION QUALITY
    # ========================================
    print(f"\n{'='*60}")
    print("[5] CAPTION QUALITY")
    print(f"{'='*60}")
    
    all_cases = train_cases + test_cases
    caption_stats = check_caption_quality(all_cases)
    
    print(f"  Total captions: {caption_stats['total']}")
    print(f"  Empty rate:     {caption_stats['empty_rate']*100:.1f}%")
    print(f"  Avg length:     {caption_stats['avg_length']:.0f} chars")
    
    if caption_stats['empty_rate'] > 0.3:
        print("\n  ⚠ WARNING: >30% captions are empty!")
    
    results["checks"]["captions"] = caption_stats
    
    # ========================================
    # 6. QDRANT COLLECTIONS
    # ========================================
    print(f"\n{'='*60}")
    print("[6] QDRANT COLLECTIONS")
    print(f"{'='*60}")
    
    qdrant_info = check_qdrant_collections()
    
    if "error" in qdrant_info:
        print(f"  ⚠ Connection failed: {qdrant_info['error']}")
    else:
        for col_name, count in qdrant_info.items():
            status = f"{count:,} points" if count >= 0 else "NOT FOUND"
            print(f"  {col_name}: {status}")
    
    results["checks"]["qdrant"] = qdrant_info
    
    # ========================================
    # 7. ENVIRONMENT SUMMARY
    # ========================================
    print()
    log_env_summary()
    
    # ========================================
    # FINAL VERDICT
    # ========================================
    print(f"\n{'='*70}")
    if results["passed"]:
        print("✓ AUDIT PASSED - Dataset is valid for scientific experiments")
    else:
        print("✗ AUDIT FAILED - Fix issues before proceeding!")
    print(f"{'='*70}\n")
    
    return results


if __name__ == "__main__":
    results = run_audit()
    
    # Save results to JSON for reference
    output_path = DATA_ROOT / "audit_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Audit results saved to: {output_path}")
