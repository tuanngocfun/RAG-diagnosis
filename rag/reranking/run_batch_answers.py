#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch answer questions with doc_id-constrained retrieval + robust resume/dedupe.

Adds:
- --resume: skip already-answered rows found in --out
- --retry_errors: reattempt rows that previously logged {"error": ...}
- Safe appends with per-line flush and periodic fsync
- Signal handling for graceful flush on SIGINT/SIGTERM
"""

import argparse
import json
import os
import signal
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .med4b_qdrant_bge import CFG, qdrant, CQ2, MedGemma, TextCrossReranker, rerank_with_text
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
    raw = hit.get("image_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    doc_id = hit.get("doc_id")
    pi = hit.get("page_index")
    if not (doc_id and isinstance(pi, int)):
        return None

    # Try 1-based first (common in filenames)
    p1 = CFG.EXTRACT_ROOT / doc_id / "pages" / f"page_{pi:04d}.png"
    if p1.exists():
        return p1

    # Then try +1 (if payload stored 0-based for PDF)
    p2 = CFG.EXTRACT_ROOT / doc_id / "pages" / f"page_{pi+1:04d}.png"
    if p2.exists():
        return p2

    return None

def embed_query_text(encoder: CQ2, question: str) -> np.ndarray:
    return encoder.embed_texts([question])[0]

def build_filter(doc_id_exact: Optional[str],
                 case_type: Optional[str],
                 keyword: Optional[str],
                 micrograph_strict: bool) -> Optional[Filter]:
    must = []
    if doc_id_exact:
        must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id_exact)))
    if case_type:
        must.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
    if keyword:
        must.append(FieldCondition(key="keywords", match=MatchAny(any=[keyword])))
    if micrograph_strict:
        must.append(FieldCondition(key="micrograph_like", match=MatchValue(value=True)))
    return Filter(must=must) if must else None

def query_points(client, qv: np.ndarray, top_k: int,
                 base_filter: Optional[Filter],
                 score_threshold: Optional[float]):
    kwargs = dict(
        collection_name=CFG.COLLECTION,
        query=qv.tolist(),
        using="image",
        limit=top_k,
        query_filter=base_filter,
        with_payload=True,
    )
    if score_threshold is not None:
        kwargs["score_threshold"] = score_threshold
    return client.query_points(**kwargs)

def to_hit_dict(points_result) -> List[Dict[str, Any]]:
    items = getattr(points_result, "points", []) or []
    out = []
    for i, h in enumerate(items):
        pl = getattr(h, "payload", {}) or {}
        out.append({
            "rank": i + 1,
            "score": float(getattr(h, "score", 0.0)),
            "doc_id": pl.get("doc_id"),
            "page_index": pl.get("page_index"),
            "image_path": pl.get("image_path"),
            "case_type": pl.get("case_type"),
            "page_kind": pl.get("page_kind"),
            "micrograph_like": pl.get("micrograph_like"),
            "keywords": (pl.get("keywords") or [])[:6],
            "text_excerpt": pl.get("text_excerpt"),
        })
    return out

def prefer_micrographs(hits: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
    if not hits:
        return []
    prefer = [r for r in hits if r.get("micrograph_like") is True]
    others = [r for r in hits if not r.get("micrograph_like")]
    return (prefer + others)[:top_k] if prefer else hits[:top_k]

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
    # Add these new arguments
    ap.add_argument("--use_reranker", action="store_true",
                    help="Enable BGE cross-encoder reranking")
    ap.add_argument("--rerank_alpha", type=float, default=0.6,
                    help="Weight for text reranker vs ColQwen2 (0-1)")
    ap.add_argument("--pool_mult", type=int, default=3,
                    help="Multiplier for retrieval pool before reranking")
    ap.add_argument("--rerank_min_excerpt_chars", type=int, default=CFG.RERANK_MIN_EXCERPT_CHARS,
                help="Min excerpt length to trust text reranker")
    ap.add_argument("--rerank_fallback_alpha", type=float, default=CFG.RERANK_FALLBACK_ALPHA,
                help="Fallback weight when excerpt is short/missing")

    # New flags for durability/resume
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

    client = qdrant()
    encoder = CQ2(CFG.RET_MODEL_ID)
    mdg = MedGemma(CFG.GEN_MODEL_ID)
    reranker = None
    if args.use_reranker:
        try:
            reranker = TextCrossReranker(CFG.RERANKER_MODEL_ID)
            print(f"[INFO] Reranker loaded: {CFG.RERANKER_MODEL_ID}")
        except Exception as e:
            print(f"[WARN] Reranker unavailable: {e}")
            reranker = None

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
                    # Keep prior error; don't retry unless asked
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

                qv = embed_query_text(encoder, qtext)
                base_filter = build_filter(doc_id, args.case_type, args.keyword, args.micrograph_strict)

                limit = (args.topk * args.pool_mult) if (args.micrograph_only and not args.micrograph_strict) else (args.topk * args.pool_mult if args.use_reranker else args.topk)
                res = query_points(client, qv, limit, base_filter, CFG.SCORE_THRESHOLD)
                hits = to_hit_dict(res)

                # Optional rerank
                if reranker is not None and hits:
                    hits = rerank_with_text(
                        qtext, hits, reranker,
                        alpha=args.rerank_alpha,
                        min_excerpt_chars=args.rerank_min_excerpt_chars,
                        fallback_alpha=args.rerank_fallback_alpha
                    )

                # Soft preference for micrographs after reranking (if requested)
                if args.micrograph_only and not args.micrograph_strict:
                    hits = prefer_micrographs(hits, args.topk)
                else:
                    hits = hits[:args.topk]


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

                answer_text = mdg.answer(qtext, [Path(p) for p in used_images])

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

            except Exception as e:
                # Log & continue (keep going)
                sys.stderr.write(f"[WARN] failed row: {e}\n")
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

    print(f"[OK] Answered {ok}/{total} (skipped={skipped}, retried={retried}) -> {out_path}")

if __name__ == "__main__":
    main()
