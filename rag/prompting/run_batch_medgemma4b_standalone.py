#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch processing script for standalone MedGemma-4B-IT analysis.
Similar to run_batch_answers_medgemma4b_test_medcpt.py but without RAG pipeline.

Usage:
python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest /path/to/questions_manifest.jsonl \
    --out /path/to/answers_standalone.ndjson \
    --images_per_answer 3 \
    --resume
"""

import argparse
import json
import os
import signal
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional
import time

from rag.prompting.medgemma4b_standalone import (
    StandaloneMedGemma4B, StandaloneCFG, 
    find_case_dir, select_images
)

# -------------------- Utilities --------------------

def _stable_key(row: Dict[str, Any]) -> str:
    """Prefer question_id; else derive a stable key from (case_id, question)."""
    qid = row.get("question_id")
    if isinstance(qid, str) and qid.strip():
        return f"qid:{qid.strip()}"
    case_id = (row.get("case_id") or row.get("doc_id") or row.get("caseId") or "").strip()
    qtxt = (row.get("question") or "").strip()
    h = hashlib.sha1((case_id + "\n" + qtxt).encode("utf-8")).hexdigest()
    return f"hk:{h}"

def _read_existing_out(path: Path) -> tuple:
    """Returns (done_keys, error_keys) from an NDJSON output file."""
    done, err = set(), set()
    if not path.exists():
        return done, err
    
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            k = _stable_key(rec)
            ans = (rec.get("answer") or "").strip()
            if "error" in rec:
                err.add(k)
            elif ans:
                done.add(k)
    return done, err

def derive_doc_id_from_row(row_in: Dict[str, Any]) -> Optional[str]:
    """
    Derive doc_id from row data. Prefer explicit doc_id, then case_id.
    """
    doc_id = (row_in.get("doc_id") or "").strip()
    if doc_id:
        return doc_id

    case_id = (row_in.get("case_id") or "").strip()
    if case_id:
        return case_id
        
    return None

# -------------------- Batch Processor --------------------

class StandaloneBatchProcessor:
    """Batch processor for standalone MedGemma analysis."""
    
    def __init__(self):
        print("[INFO] Loading MedGemma model (one-time initialization)...")
        self.analyzer = StandaloneMedGemma4B()
        print("[INFO] MedGemma model loaded successfully!")
    
    def process_single_question(self, row: Dict[str, Any], images_per_answer: int = 3) -> Dict[str, Any]:
        """
        Process a single question from the manifest.
        
        Args:
            row: Question data from manifest
            images_per_answer: Maximum images to use per answer
            
        Returns:
            Result dictionary with answer and metadata
        """
        question = (row.get("question") or "").strip()
        doc_id = derive_doc_id_from_row(row)
        
        if not question:
            return {"error": "empty_question", **row}
            
        if not doc_id:
            return {"error": "doc_id_unresolvable", **row}
        
        try:
            # Get seed images if provided
            seed_images = row.get("seed_image_paths") or []
            
            # Find case directory
            case_dir = find_case_dir(doc_id, StandaloneCFG.EXTRACT_ROOT)
            if not case_dir:
                return {"error": "case_dir_not_found", "doc_id": doc_id, **row}
            
            # Select images (prioritize seed images, then first few pages)
            target_pages = [0, 1, 2] if not seed_images else []
            image_paths = select_images(seed_images, case_dir, target_pages, images_per_answer)
            
            if not image_paths:
                return {"error": "no_images_found", "doc_id": doc_id, **row}
            
            # Determine analysis type based on question
            if any(term in question.lower() for term in [
                "diagnosis", "diagnose", "identify", "what is", "condition"
            ]):
                # Use diagnostic analysis
                answer = self.analyzer.diagnose_case(doc_id, question, seed_images)
            else:
                # Use general image analysis with question
                answer = self.analyzer.analyze_images(image_paths, question)
            
            # Prepare result
            result = {
                "question_id": row.get("question_id"),
                "case_id": row.get("case_id"),
                "doc_id": doc_id,
                "question": question,
                "answer": answer,
                "used_images": image_paths[:images_per_answer],
                "model_type": "medgemma4b_standalone",
                "model_id": self.analyzer.model_id,
                "processing_mode": "standalone"
            }
            
            return result
            
        except Exception as e:
            return {"error": f"processing_failed: {str(e)}", "doc_id": doc_id, **row}

# -------------------- Main Script --------------------

def main():
    parser = argparse.ArgumentParser(
        description="Batch process questions with standalone MedGemma-4B-IT (no RAG)"
    )
    parser.add_argument("--manifest", required=True, help="Input questions manifest (JSONL)")
    parser.add_argument("--out", required=True, help="Output answers file (NDJSON)")
    parser.add_argument("--images_per_answer", type=int, default=3, 
                       help="Maximum images per answer")
    parser.add_argument("--resume", action="store_true", 
                       help="Resume from existing output file")
    parser.add_argument("--retry_errors", action="store_true",
                       help="Retry items that previously had errors")
    parser.add_argument("--fsync_interval", type=int, default=25,
                       help="Fsync after this many processed items")

    args = parser.parse_args()

    # Prepare output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Check existing progress
    done_keys, error_keys = _read_existing_out(out_path)
    print(f"[INFO] Found {len(done_keys)} completed, {len(error_keys)} errors in existing output")

    # Open output file
    mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(mode, encoding="utf-8")

    # Signal handling for graceful shutdown
    exiting = {"flag": False}
    def graceful_exit(signum, frame):
        exiting["flag"] = True
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)

    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, graceful_exit)

    # Initialize processor
    processor = StandaloneBatchProcessor()

    # Process manifest
    total = processed = skipped = retried = written_since_fsync = 0

    def write_result(result_dict: Dict[str, Any]) -> None:
        nonlocal written_since_fsync
        fout.write(json.dumps(result_dict, ensure_ascii=False) + "\n")
        fout.flush()
        written_since_fsync += 1
        if written_since_fsync >= args.fsync_interval:
            os.fsync(fout.fileno())
            written_since_fsync = 0

    print(f"[INFO] Starting batch processing...")
    start_time = time.time()

    with open(args.manifest, "r", encoding="utf-8") as fin:
        for line_num, line in enumerate(fin, 1):
            if exiting["flag"]:
                break
                
            total += 1
            
            try:
                row = json.loads(line.strip())
            except Exception as e:
                write_result({
                    "error": "invalid_json_line",
                    "line_number": line_num,
                    "detail": str(e),
                    "line": line[:200]
                })
                continue

            # Generate processing key
            key = _stable_key(row)

            # Resume logic
            if args.resume:
                if key in done_keys:
                    skipped += 1
                    continue
                if key in error_keys and not args.retry_errors:
                    skipped += 1
                    continue
                if key in error_keys and args.retry_errors:
                    retried += 1

            # Process the question
            try:
                result = processor.process_single_question(row, args.images_per_answer)
                write_result(result)
                processed += 1

                # Progress update
                if processed % 10 == 0:
                    elapsed = time.time() - start_time
                    rate = processed / elapsed if elapsed > 0 else 0
                    print(f"[PROGRESS] Processed {processed}/{total} questions "
                          f"(skipped: {skipped}, rate: {rate:.1f}/s)")

            except Exception as e:
                error_result = {
                    "error": f"unexpected_error: {str(e)}",
                    **row
                }
                write_result(error_result)
                print(f"[ERROR] Unexpected error processing line {line_num}: {e}")

    # Final cleanup
    try:
        fout.flush()
        os.fsync(fout.fileno())
    except Exception:
        pass
    fout.close()

    # Summary
    elapsed = time.time() - start_time
    print(f"\n[SUMMARY]")
    print(f"Total questions: {total}")
    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")
    print(f"Retried: {retried}")
    print(f"Processing time: {elapsed:.1f}s")
    print(f"Average rate: {processed/elapsed:.2f} questions/second")
    print(f"Output file: {out_path}")

if __name__ == "__main__":
    main()