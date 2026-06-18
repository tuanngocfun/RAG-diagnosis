#!/usr/bin/env python3
"""Normalize official Gemma 4 RAG/no-RAG experiment outputs for V12d slides."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "data" / "gemma4_rag_norag_comparison"
OUTPUT_PATH = OUTPUT_DIR / "comparison_summary.json"

EXPERIMENT_ROOT = Path("/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000/runs")
RAG_RUN_DIR = (
    EXPERIMENT_ROOT
    / "p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935"
)
NORAG_RUN_DIR = (
    EXPERIMENT_ROOT
    / "p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411"
)
RAG_ANSWERS = RAG_RUN_DIR / "answers_latency.jsonl"
NORAG_ANSWERS = NORAG_RUN_DIR / "answers_norag.jsonl"
QUERY_TYPE = "Q1_Q3_multimodal_diagnosis"

CASES = [
    {
        "case_id": "PMC7516301_01",
        "silver_reference_type": "MCL",
        "role": "retrieval-sensitive MCL example",
        "interpretation": (
            "Retrieval changed this selected backup example from Non-Leishmaniasis "
            "to MCL in the official Gemma 4 experiment-pipeline comparison."
        ),
    },
    {
        "case_id": "PMC7456484_01",
        "silver_reference_type": "PKDL",
        "role": "subtype challenge",
        "interpretation": (
            "Both conditions remain inside the leishmaniasis family but miss the "
            "PKDL subtype, so this remains a subtype-resolution challenge."
        ),
    },
    {
        "case_id": "PMC10026180_04",
        "silver_reference_type": "verified Non-Leish / pseudolabel CL",
        "role": "label-conflict stress test",
        "interpretation": (
            "The official comparison returns Non-Leishmaniasis in both conditions, "
            "but the full-text live demo and LLM council reviews make this a "
            "label-conflict/evidence-attribution stress test, not specificity proof."
        ),
    },
]


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_rank1(answer: str) -> dict[str, str]:
    parsed = {
        "rank1": "Unknown",
        "diagnosis_type": "Unknown",
        "confidence": "Unknown",
    }
    for line in answer.splitlines():
        if "**Rank 1 supportive consideration:**" in line:
            parsed["rank1"] = line.split("**Rank 1 supportive consideration:**", 1)[1].strip()
        elif "**Rank 1 (Most Likely):**" in line:
            parsed["rank1"] = line.split("**Rank 1 (Most Likely):**", 1)[1].strip()
        elif "**Rank 1 Diagnosis Type:**" in line:
            parsed["diagnosis_type"] = line.split("**Rank 1 Diagnosis Type:**", 1)[1].strip()
        elif "**Rank 1 Confidence:**" in line:
            parsed["confidence"] = line.split("**Rank 1 Confidence:**", 1)[1].strip()
    return parsed


def read_case_rows(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            qid = row.get("qid", "")
            case_id = qid.split("::", 1)[0]
            if row.get("query_type") == QUERY_TYPE:
                rows[case_id] = row
    return rows


def run_summary(run_dir: Path, is_rag: bool) -> dict[str, Any]:
    config = load_json(run_dir / "run_config.json")
    summary = load_json(run_dir / "summary.json")
    contract_path = run_dir / "answer_generation_contract.json"
    contract = load_json(contract_path) if contract_path.exists() else {}
    return {
        "run_dir": str(run_dir),
        "run_id": summary.get("run_id") or config.get("run_id"),
        "is_rag": is_rag,
        "generator_model": (
            contract.get("generator_model")
            or summary.get("generator")
            or config.get("generator_model")
            or "google/gemma-4-E4B-it"
        ),
        "query_file": config.get("queries_file", ""),
        "query_type": QUERY_TYPE,
        "context_k": contract.get("context_k") or config.get("context_k"),
    }


def normalized_case(case: dict[str, str], rag_rows: dict[str, dict[str, Any]],
                    norag_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    case_id = case["case_id"]
    if case_id not in rag_rows:
        raise RuntimeError(f"Missing RAG row for {case_id}")
    if case_id not in norag_rows:
        raise RuntimeError(f"Missing no-RAG row for {case_id}")

    rag_row = rag_rows[case_id]
    norag_row = norag_rows[case_id]
    rag_answer = rag_row.get("answer") or rag_row.get("assistant_markdown") or ""
    norag_answer = norag_row.get("answer") or ""

    rag = parse_rank1(rag_answer)
    norag = parse_rank1(norag_answer)
    rag["retrieved_context_count"] = str(len(rag_row.get("contexts") or []))
    norag["retrieved_context_count"] = str(norag_row.get("retrieved_context_count_used", 0))
    rag["answer_source"] = str(RAG_ANSWERS)
    norag["answer_source"] = str(NORAG_ANSWERS)

    return {
        **case,
        "rag": rag,
        "no_rag": norag,
        "boundary": (
            "Selected Q&A backup example only; aggregate thesis metrics remain "
            "the benchmark."
        ),
    }


def main() -> int:
    for path in (
        RAG_ANSWERS,
        NORAG_ANSWERS,
        RAG_RUN_DIR / "run_config.json",
        NORAG_RUN_DIR / "run_config.json",
    ):
        if not path.exists():
            raise FileNotFoundError(path)

    rag_rows = read_case_rows(RAG_ANSWERS)
    norag_rows = read_case_rows(NORAG_ANSWERS)
    cases = [normalized_case(case, rag_rows, norag_rows) for case in CASES]

    summary = {
        "version": "V12d",
        "comparison_label": "official Gemma 4 experiment-pipeline comparison",
        "purpose": "Q&A-only backup evidence for retrieval effect inspection",
        "rag_run": run_summary(RAG_RUN_DIR, is_rag=True),
        "no_rag_run": run_summary(NORAG_RUN_DIR, is_rag=False),
        "cases": cases,
        "active_claim_boundary": (
            "This file supports selected case-level discussion only. It must not "
            "be used as a three-case aggregate accuracy estimate."
        ),
        "supersedes": (
            "The earlier demo-backend no-RAG attempt is not used because the "
            "backend still returned retrieved support chunks."
        ),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)

    for case in cases:
        with (OUTPUT_DIR / f"{case['case_id']}_comparison.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(case, handle, indent=2, ensure_ascii=False)

    print(f"Wrote {OUTPUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
