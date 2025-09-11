#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prepare a questions-only manifest for batch answering.

Reads per-case QA JSONL files (each file contains a single JSON object):
{
  "case_id": "...",
  "questions": [
    {
      "question": "...",
      "gold_answer": "...",
      "gold_evidence": [
        {"page_id": "44-45", "evidence_type": "text|image", "text_span"|"bbox_i": "..."},
        ...
      ]
    }
  ]
}

Outputs NDJSON at:
  /home/students/Leishmania/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl

Each line:
{
  "case_id": "<string>",
  "doc_id": "<folder name under extract/>",
  "question_id": "<stable id>",
  "question": "<string>",
  "seed_image_paths": [".../pages/page_0007.png", ...],   # only in --mode seeded
  "retrieve_mode": "pure" | "seeded"
}
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# --- Paths (adapt if needed) ---
QA_DIR = Path("kaggle/working2/rag_knowledge_base/qa/jsonl")
EXTRACT_ROOT = Path("kaggle/working2/extract")
OUT_MANIFEST = Path("kaggle/working2/rag_knowledge_base/questions_manifest.jsonl")
MAX_SEED_IMAGES_PER_Q = 6

# Import helpers from your indexing module
# Run this script as a module:  python -m rag.gen.prepare_questions_manifest --mode pure
from .qdrant_rag import (
    find_case_dir, load_page_map, map_page_id_to_indices, page_indices_to_paths
)


def _hash_case_id(case_id: str) -> str:
    return hashlib.sha256(case_id.encode("utf-8")).hexdigest()[:10]


def iter_qa_files():
    for f in sorted(QA_DIR.glob("*.jsonl")):
        yield f


def collect_seed_images(case_id: str, evidence: List[Dict[str, Any]]) -> List[str]:
    """Map gold_evidence.page_id -> actual page PNG paths using _page_map.json if present."""
    case_dir = find_case_dir(case_id, EXTRACT_ROOT)
    if not case_dir:
        return []
    pm = load_page_map(case_dir)

    paths: List[str] = []
    for ev in evidence:
        pid = str(ev.get("page_id", "")).strip()
        if not pid:
            continue
        idxs = map_page_id_to_indices(pid, pm)
        for p in page_indices_to_paths(case_dir, idxs):
            if p.exists():
                paths.append(str(p))

    # dedupe while preserving order, then cap
    seen, uniq = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq[:MAX_SEED_IMAGES_PER_Q]


def main():
    ap = argparse.ArgumentParser(description="Build questions-only manifest from QA JSONL files.")
    ap.add_argument("--mode", choices=["pure", "seeded"], default="pure",
                    help="pure: no seed images; seeded: attach images from gold_evidence page_ids")
    args = ap.parse_args()

    out_rows: List[Dict[str, Any]] = []

    for fp in iter_qa_files():
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"[WARN] Skip {fp.name}: cannot parse JSON ({e})")
            continue

        case_id = data.get("case_id") or fp.stem
        questions = data.get("questions") or []

        # Resolve doc_id = exact folder name under extract root
        case_dir = find_case_dir(case_id, EXTRACT_ROOT)
        doc_id = case_dir.name if case_dir else None

        case_hash = _hash_case_id(case_id)
        qn = 0
        for q in questions:
            qtext = (q.get("question") or "").strip()
            if not qtext:
                continue
            qn += 1
            qid = f"{case_hash}-q{qn:03d}"

            seed_imgs: List[str] = []
            if args.mode == "seeded":
                ev = q.get("gold_evidence") or []
                seed_imgs = collect_seed_images(case_id, ev)

            out_rows.append({
                "case_id": case_id,
                "doc_id": doc_id,
                "question_id": qid,
                "question": qtext,
                "seed_image_paths": seed_imgs,
                "retrieve_mode": args.mode,
            })

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with OUT_MANIFEST.open("w", encoding="utf-8") as f:
        for row in out_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[OK] Wrote {len(out_rows)} entries -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
