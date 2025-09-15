#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch answer questions with doc_id-constrained retrieval + robust resume/dedupe.

Optimized version that loads models once and reuses them for all questions.
"""

import argparse
import json
import os
import signal
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Import from the correct module path based on your structure
from rag.reranking.med4b_qdrant_bge import (
    CFG, qdrant, CQ2, MedGemma, TextCrossReranker, 
    rerank_with_text, find_case_dir, page_indices_to_paths
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# -------------------- small utils --------------------

def _stable_key(row: Dict[str, Any]) -> str:
    """Prefer question_id; else derive a stable key from (case_id, question)."""
    qid = row.get("question_id")
    if isinstance(qid, str) and qid.strip():
        return f"qid:{qid.strip()}"
    case_id = (row.get("case_id") or "").strip()
    qtxt = (row.get("question") or "").strip()
    h = hashlib.sha1((case_id + "\n" + qtxt).encode("utf-8")).hexdigest()
    return f"hk:{h}"

def _read_existing_out(path: Path) -> Tuple[Set[str], Set[str]]:
    """
    Returns (done_keys, error_keys).
    done_keys = keys for rows that contain an 'answer' (successful)
    error_keys = keys for rows that contain an 'error' (failed previously)
    Ignores malformed/truncated lines.
    """
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
                continue  # skip broken tail lines
            k = _stable_key(rec)
            if "answer" in rec and isinstance(rec["answer"], str) and rec["answer"].strip():
                done.add(k)
            elif "error" in rec:
                err.add(k)
    return done, err

def repair_image_path(hit: Dict[str, Any]) -> Optional[Path]:
    """Repair image paths using the same logic as qdrant_bge.py"""
    raw = hit.get("image_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    doc_id = hit.get("doc_id")
    pi = hit.get("page_index")
    if not (doc_id and isinstance(pi, int)):
        return None

    # Use the find_case_dir function from qdrant_bge.py
    case_dir = find_case_dir(doc_id, CFG.EXTRACT_ROOT)
    if not case_dir:
        return None

    # Try using page_indices_to_paths function
    paths = page_indices_to_paths(case_dir, [pi])
    if paths and paths[0].exists():
        return paths[0]

    # Fallback to manual path construction
    p1 = case_dir / "pages" / f"page_{pi+1:04d}.png"  # 1-based filename
    if p1.exists():
        return p1

    p2 = case_dir / "pages" / f"page_{pi:04d}.png"  # 0-based filename  
    if p2.exists():
        return p2

    return None

def select_images(seed: List[str], hits: List[Dict[str, Any]], take: int) -> List[str]:
    out, seen = [], set()
    # seed paths first
    for p in seed or []:
        if p and Path(p).exists() and p not in seen:
            out.append(p); seen.add(p)
            if len(out) >= take: return out
    # then repaired hit paths
    for h in hits:
        rp = repair_image_path(h)
        if rp and rp.exists() and str(rp) not in seen:
            out.append(str(rp)); seen.add(str(rp))
            if len(out) >= take: return out
    return out

class BatchProcessor:
    """Optimized batch processor that loads models once and reuses them."""
    
    def __init__(self, use_reranker: bool = True):
        print("[INFO] Loading models (one-time initialization)...")
        
        # Load models once
        self.client = qdrant()
        self.encoder = CQ2(CFG.RET_MODEL_ID)
        self.mdg = MedGemma(CFG.GEN_MODEL_ID)
        
        # Optional reranker
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = TextCrossReranker(CFG.RERANKER_MODEL_ID)
                print(f"[INFO] Reranker loaded: {CFG.RERANKER_MODEL_ID}")
            except Exception as e:
                print(f"[WARN] Reranker unavailable: {e}")
                self.reranker = None
        
        print("[INFO] All models loaded successfully!")
    
    def enhanced_qdrant_ask_text(self, question: str, doc_id: Optional[str] = None, 
                               top_k: int = CFG.TOP_K,
                               case_type: Optional[str] = None,
                               keyword: Optional[str] = None,
                               micrograph_only: bool = False,
                               micrograph_strict: bool = False,
                               pool_multiplier: int = 3) -> Dict[str, Any]:
        """Enhanced version of qdrant_ask_text with doc_id filtering"""
        qv = self.encoder.embed_texts([question])[0]

        # Build filter including doc_id constraint
        musts = []
        if doc_id:
            musts.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if case_type:
            musts.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
        if keyword:
            musts.append(FieldCondition(key="keywords", match=MatchAny(any=[keyword])))
        
        base_filter = Filter(must=musts) if musts else None

        # Adjust retrieval size for reranking pool
        pool_size = top_k * pool_multiplier if self.reranker else top_k

        # Handle strict micrograph filtering
        if micrograph_only and micrograph_strict:
            strict_filter = Filter(must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))])
            from rag.reranking.med4b_qdrant_bge import _qdrant_search
            hits = _qdrant_search(self.client, qv, pool_size, strict_filter, CFG.SCORE_THRESHOLD)
        else:
            from rag.reranking.med4b_qdrant_bge import _qdrant_search
            hits = _qdrant_search(self.client, qv, pool_size, base_filter, CFG.SCORE_THRESHOLD)

        # Convert to dict format
        raw_items = [{
            "rank": i+1,
            "score": float(h.score),
            "doc_id": h.payload.get("doc_id"),
            "page_index": h.payload.get("page_index"),
            "image_path": h.payload.get("image_path"),
            "case_type": h.payload.get("case_type"),
            "page_kind": h.payload.get("page_kind"),
            "micrograph_like": h.payload.get("micrograph_like"),
            "keywords": h.payload.get("keywords", [])[:6],
            "text_excerpt": h.payload.get("text_excerpt")
        } for i, h in enumerate(hits)]

        # Apply reranking if enabled
        if self.reranker and raw_items:
            try:
                raw_items = rerank_with_text(question, raw_items, self.reranker)
            except Exception as e:
                print(f"[WARN] Reranking failed, falling back to original scores: {e}")

        # Apply micrograph preference if requested (after reranking)
        if micrograph_only and not micrograph_strict and raw_items:
            prefer = [r for r in raw_items if r.get("micrograph_like") is True]
            others = [r for r in raw_items if not r.get("micrograph_like")]
            selected = (prefer + others)[:top_k] if prefer else raw_items[:top_k]
        else:
            selected = raw_items[:top_k]

        return {"mode": "text", "question": question, "hits": selected}
    
    def answer_with_images(self, question: str, image_paths: List[str]) -> str:
        """Answer using MedGemma with provided image paths."""
        paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not paths:
            return "No valid images provided."
        return self.mdg.answer(question, paths)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Batch answer questions with case-constrained retrieval (resume-safe).")
    ap.add_argument("--manifest", required=True, help="questions_manifest.jsonl")
    ap.add_argument("--out", required=True, help="Output NDJSON path (append-safe with --resume)")
    ap.add_argument("--topk", type=int, default=CFG.TOP_K, help="Top-K retrieval")
    ap.add_argument("--images_per_answer", type=int, default=4, help="Images to feed to MedGemma")
    ap.add_argument("--micrograph_only", action="store_true", help="Soft preference for micrograph-like pages")
    ap.add_argument("--micrograph_strict", action="store_true", help="Hard filter to micrograph-like pages")
    ap.add_argument("--score_threshold", type=float, default=None, help="Override CFG.SCORE_THRESHOLD (None => no threshold)")
    ap.add_argument("--case_type", choices=["cutaneous", "mucocutaneous", "visceral", "unknown"])
    ap.add_argument("--keyword", help="Optional keyword payload filter")
    
    # Reranking arguments
    ap.add_argument("--use_reranker", action="store_true",
                    help="Enable BGE cross-encoder reranking")
    ap.add_argument("--pool_mult", type=int, default=3,
                    help="Multiplier for retrieval pool before reranking")

    # Resume/durability arguments
    ap.add_argument("--resume", action="store_true",
                    help="Scan existing --out and skip already answered items")
    ap.add_argument("--retry_errors", action="store_true",
                    help="When used with --resume, reattempt items that previously had 'error'")
    ap.add_argument("--fsync_interval", type=int, default=25,
                    help="Call os.fsync after this many lines (durability vs throughput)")

    args = ap.parse_args()

    if args.score_threshold is not None:
        CFG.SCORE_THRESHOLD = args.score_threshold

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing progress (if any)
    done_keys, err_keys = _read_existing_out(out_path)
    want_retry_errors = args.resume and args.retry_errors

    # Open OUT in append mode if resuming, else write mode.
    mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(mode, encoding="utf-8")
    need_fsync_every = max(1, int(args.fsync_interval))

    # Signal handling: ensure flush on SIGINT/SIGTERM
    exiting = {"flag": False}
    def _graceful(signum, frame):
        exiting["flag"] = True
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass
        # Second signal => hard exit
        signal.signal(signum, signal.SIG_DFL)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _graceful)

    # Initialize the batch processor (loads all models once)
    processor = BatchProcessor(use_reranker=args.use_reranker)

    total = 0
    ok = 0
    skipped = 0
    retried = 0
    written_since_fsync = 0

    def write_row(row: Dict[str, Any]) -> None:
        nonlocal written_since_fsync
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        written_since_fsync += 1
        if written_since_fsync >= need_fsync_every:
            os.fsync(fout.fileno())
            written_since_fsync = 0

    print(f"[INFO] Starting batch processing...")

    with open(args.manifest, "r", encoding="utf-8") as fin:
        for ln in fin:
            if exiting["flag"]:
                break
            total += 1
            try:
                row_in = json.loads(ln)
            except Exception as e:
                write_row({"error": "bad_manifest_line", "detail": str(e), "line": ln[:200]})
                continue

            key = _stable_key(row_in)

            # Resume logic
            if args.resume:
                if key in done_keys:
                    skipped += 1
                    continue
                if (not want_retry_errors) and key in err_keys:
                    skipped += 1
                    continue
                if want_retry_errors and key in err_keys:
                    retried += 1

            qtext = (row_in.get("question") or "").strip()
            if not qtext:
                write_row({"error": "empty_question", **row_in})
                continue

            try:
                doc_id = row_in.get("doc_id")
                seed_images: List[str] = row_in.get("seed_image_paths") or []

                # Use the processor's method (no model reloading)
                result = processor.enhanced_qdrant_ask_text(
                    question=qtext,
                    doc_id=doc_id,
                    top_k=args.topk,
                    case_type=args.case_type,
                    keyword=args.keyword,
                    micrograph_only=args.micrograph_only,
                    micrograph_strict=args.micrograph_strict,
                    pool_multiplier=args.pool_mult
                )
                
                hits = result["hits"]

                # Verify doc_id constraint was respected
                if doc_id and any(h.get("doc_id") != doc_id for h in hits):
                    write_row({"error": "retriever_leak_other_doc",
                               "expected_doc_id": doc_id,
                               "got_doc_ids": sorted({h.get('doc_id') for h in hits if h.get('doc_id')}),
                               **row_in})
                    continue

                if not hits:
                    write_row({"error": "no_hits_in_doc", **row_in})
                    continue

                used_images = select_images(seed_images, hits, args.images_per_answer)
                if not used_images:
                    write_row({"error": "no_images_on_disk", **row_in, "hits": hits})
                    continue

                # Use the processor's answer method (no model reloading)
                answer_text = processor.answer_with_images(qtext, used_images)

                out_rec = {
                    "question_id": row_in.get("question_id"),
                    "case_id": row_in.get("case_id"),
                    "doc_id": doc_id,
                    "question": qtext,
                    "retrieve_mode": row_in.get("retrieve_mode", "pure"),
                    "used_images": used_images,
                    "answer": answer_text,
                    "retrieval_hits": hits,
                }
                write_row(out_rec)
                ok += 1

                # Progress indicator
                if total % 10 == 0:
                    print(f"[PROGRESS] Processed {total} questions, answered {ok}")

            except Exception as e:
                # Log & continue (keep going)
                sys.stderr.write(f"[WARN] failed row {total}: {e}\n")
                try:
                    write_row({"error": f"exception:{type(e).__name__}", "detail": str(e), **row_in})
                except Exception:
                    pass

    # final flush/fsync on exit
    try:
        fout.flush()
        os.fsync(fout.fileno())
    except Exception:
        pass
    fout.close()

    print(f"[FINAL] Answered {ok}/{total} (skipped={skipped}, retried={retried}) -> {out_path}")

if __name__ == "__main__":
    main()