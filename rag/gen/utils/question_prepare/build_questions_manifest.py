#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Scan gold QA JSONL files and build:
  - tasks/questions_manifest.jsonl  (one record per question)
Optionally adds "oracle_images": [paths] by mapping gold_evidence.page_id to PNGs.

Input gold folder:
  /media/pc1/Ubuntu/Extend_Data/ngoc/kaggle/working2/rag_knowledge_base/qa/jsonl

Extracted PNG structure (already in your project):
  /media/pc1/Ubuntu/Extend_Data/ngoc/kaggle/working2/extract/<CASE>/pages/page_0000.png ...

This reuses your helper mapping logic by importing qdrant_rag.py (find_case_dir, map_page_id_to_indices, page_indices_to_paths).
"""
import json, sys
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path("/media/pc1/Ubuntu/Extend_Data/ngoc")
GOLD_DIR = ROOT / "kaggle" / "working2" / "rag_knowledge_base" / "qa" / "jsonl"
EXTRACT_ROOT = ROOT / "kaggle" / "working2" / "extract"
OUT_DIR = ROOT / "rag" / "tasks"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "questions_manifest.jsonl"

# import helpers from your indexer (must be importable as module)
sys.path.append(str((ROOT / "rag" / "gen").resolve()))
from qdrant_rag import find_case_dir, load_page_map, map_page_id_to_indices, page_indices_to_paths

def _oracle_images_for_question(case_id: str, q: Dict[str, Any]) -> List[str]:
    """Collect image PNGs referenced by gold_evidence (if any) for oracle mode."""
    evs = q.get("gold_evidence") or []
    case_dir = find_case_dir(case_id, EXTRACT_ROOT)
    if not case_dir:
        return []
    pm = load_page_map(case_dir)
    paths: List[str] = []
    for ev in evs:
        if (ev.get("evidence_type") or "").lower() != "image":
            continue
        pid = str(ev.get("page_id", "")).strip()
        if not pid:
            continue
        idxs = map_page_id_to_indices(pid, pm)
        for p in page_indices_to_paths(case_dir, idxs):
            paths.append(str(p))
    # de-dup
    return sorted(list(dict.fromkeys(paths)))

def main():
    n_files = 0
    n_q = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for jf in sorted(GOLD_DIR.glob("*.jsonl")):
            data = json.loads(jf.read_text(encoding="utf-8"))
            case_id = data.get("case_id")
            qs = data.get("questions") or []
            for qi, q in enumerate(qs):
                rec = {
                    "case_id": case_id,
                    "qid": qi,
                    "question": q.get("question", "").strip(),
                    # knobs you can change later:
                    "micrograph_only": False,
                    "micrograph_strict": False,
                    "topk": 8,
                    "score_threshold": 0.25
                }
                # Add oracle paths (useful for A/B tests)
                oracle = _oracle_images_for_question(case_id, q)
                if oracle:
                    rec["oracle_images"] = oracle
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_q += 1
            n_files += 1
    print(f"[OK] Wrote {n_q} questions from {n_files} files -> {OUT_PATH}")

if __name__ == "__main__":
    main()
