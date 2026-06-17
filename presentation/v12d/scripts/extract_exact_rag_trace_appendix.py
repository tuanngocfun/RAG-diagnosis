#!/usr/bin/env python3
"""Extract exact official RAG trace and fresh GPU audit rows for V12d."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "exact_rag_trace_appendix"
OUTPUT_PATH = OUTPUT_DIR / "trace_summary.json"
FRESH_PATTERN = "revalidation_real_model_v12d_*"

RAG_RUN_DIR = Path(
    "/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000/runs/"
    "p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935"
)
RAG_RETRIEVAL = RAG_RUN_DIR / "retrieval.jsonl"
RAG_ANSWERS = RAG_RUN_DIR / "answers_latency.jsonl"
RAG_CONFIG = RAG_RUN_DIR / "run_config.json"
RAG_SUMMARY = RAG_RUN_DIR / "summary.json"
QUERY_TYPE = "Q1_Q3_multimodal_diagnosis"

SELECTED_CASES = (
    "PMC7516301_01",
    "PMC7456484_01",
    "PMC10026180_04",
)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl_by_qid(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = str(row.get("qid", ""))
            if row.get("query_type") == QUERY_TYPE:
                rows[qid] = row
    return rows


def latest_fresh_dir() -> Path:
    candidates = sorted(
        path for path in (PROJECT_ROOT / "data").glob(FRESH_PATTERN) if path.is_dir()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No fresh V12d revalidation directory matching {FRESH_PATTERN}"
        )
    return candidates[-1]


def exact_prefix(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text[:limit].strip()


def rank1_field(markdown: str, label: str) -> str:
    for line in markdown.splitlines():
        if label in line:
            return line.split(label, 1)[1].strip()
    return ""


def extract_answer_fields(row: dict[str, Any]) -> dict[str, str]:
    answer = row.get("answer") or row.get("assistant_markdown") or ""
    return {
        "rank1": (
            rank1_field(answer, "**Rank 1 supportive consideration:**")
            or rank1_field(answer, "**Rank 1 (Most Likely):**")
        ),
        "rank1_type": rank1_field(answer, "**Rank 1 Diagnosis Type:**"),
        "rank1_confidence": rank1_field(answer, "**Rank 1 Confidence:**"),
    }


def extract_fresh_fields(result: dict[str, Any]) -> dict[str, Any]:
    markdown = result.get("assistant_markdown", "")
    return {
        "request_id": result.get("metadata", {}).get("request_id", ""),
        "model_name": result.get("metadata", {}).get("model_name", ""),
        "provider_mode": result.get("metadata", {}).get("provider_mode", ""),
        "elapsed_seconds": result.get("metadata", {}).get("elapsed_seconds", ""),
        "query_image_tensor_count": result.get("metadata", {}).get(
            "query_image_tensor_count", ""
        ),
        "safety_state": result.get("safety_state", ""),
        "rank1": (
            rank1_field(markdown, "**Rank 1 supportive consideration:**")
            or rank1_field(markdown, "**Rank 1 (Most Likely):**")
        ),
        "rank1_type": rank1_field(markdown, "**Rank 1 Diagnosis Type:**"),
        "rank1_confidence": rank1_field(markdown, "**Rank 1 Confidence:**"),
    }


def context_for_slide(context: dict[str, Any], rank: int) -> dict[str, Any]:
    text = str(context.get("text", ""))
    return {
        "rank": rank,
        "doc_id": str(context.get("doc_id", "")),
        "score": context.get("score"),
        "diagnosis_type": str(context.get("diagnosis_type", "")),
        "label_source": str(context.get("label_source", "")),
        "text_prefix_260": exact_prefix(text, 260),
        "text_char_count": len(text),
    }


def evidence_for_slide(evidence: dict[str, Any], rank: int) -> dict[str, Any]:
    excerpt = str(evidence.get("excerpt", ""))
    return {
        "rank": rank,
        "chunk_id": str(evidence.get("chunk_id", "")),
        "source_case_id": str(evidence.get("source_case_id", "")),
        "score": evidence.get("score"),
        "title": str(evidence.get("title", "")),
        "diagnosis_label": str(evidence.get("diagnosis_label", "")),
        "confirmatory": bool(evidence.get("confirmatory", False)),
        "excerpt": excerpt,
    }


def main() -> int:
    for path in (RAG_RETRIEVAL, RAG_ANSWERS, RAG_CONFIG, RAG_SUMMARY):
        if not path.exists():
            raise FileNotFoundError(path)

    run_config = read_json(RAG_CONFIG)
    run_summary = read_json(RAG_SUMMARY)
    fresh_dir = latest_fresh_dir()
    retrieval_rows = read_jsonl_by_qid(RAG_RETRIEVAL)
    answer_rows = read_jsonl_by_qid(RAG_ANSWERS)

    cases = []
    for case_id in SELECTED_CASES:
        qid = f"{case_id}::{QUERY_TYPE}"
        if qid not in retrieval_rows:
            raise RuntimeError(f"Missing retrieval row for {qid}")
        if qid not in answer_rows:
            raise RuntimeError(f"Missing answer row for {qid}")
        fresh_path = fresh_dir / f"{case_id}_result.json"
        if not fresh_path.exists():
            raise FileNotFoundError(fresh_path)

        retrieval_row = retrieval_rows[qid]
        answer_row = answer_rows[qid]
        fresh_result = read_json(fresh_path)
        contexts = retrieval_row.get("contexts") or []
        evidence = fresh_result.get("evidence") or []

        case = {
            "case_id": case_id,
            "qid": qid,
            "query_type": QUERY_TYPE,
            "official_rag_trace": {
                "retrieval_source": str(RAG_RETRIEVAL),
                "answer_source": str(RAG_ANSWERS),
                "retrieval_row": retrieval_row,
                "answer_row": answer_row,
                "answer_fields": extract_answer_fields(answer_row),
                "context_count": len(contexts),
                "top_contexts_for_slide": [
                    context_for_slide(context, rank)
                    for rank, context in enumerate(contexts[:3], start=1)
                ],
            },
            "fresh_gpu_audit": {
                "source_dir": str(fresh_dir),
                "source_path": str(fresh_path),
                "result": fresh_result,
                "fields": extract_fresh_fields(fresh_result),
                "evidence_count": len(evidence),
                "evidence_for_slide": [
                    evidence_for_slide(item, rank)
                    for rank, item in enumerate(evidence[:3], start=1)
                ],
            },
        }
        cases.append(case)

    summary = {
        "version": "V12d",
        "purpose": "Q&A-only exact retriever/rerank and fresh GPU output appendix",
        "official_rag_run": {
            "run_dir": str(RAG_RUN_DIR),
            "retriever_method": run_config.get("retriever_method"),
            "rerank": bool(run_config.get("rerank")),
            "retrieval_top_k": run_config.get("retrieval_top_k"),
            "summary_rerank": bool(
                run_summary.get("rerank")
                or run_summary.get("retrieval_contract", {}).get("rerank")
            ),
            "summary_retrieval_top_k": (
                run_summary.get("retrieval_top_k")
                or run_summary.get("retrieval_contract", {}).get("retrieval_top_k")
            ),
        },
        "fresh_gpu_revalidation_dir": str(fresh_dir),
        "source_boundary": (
            "Official pipeline retrieval rows are rerank-enabled final context "
            "lists. The live backend exposes lexical retrieval scores but no "
            "separate re-ranker contract."
        ),
        "cases": cases,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    for case in cases:
        (OUTPUT_DIR / f"{case['case_id']}_trace.json").write_text(
            json.dumps(case, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"Wrote {OUTPUT_PATH}")
    print(f"Fresh GPU source: {fresh_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
