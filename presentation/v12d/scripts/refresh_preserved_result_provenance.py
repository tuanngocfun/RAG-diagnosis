#!/usr/bin/env python3
"""Refresh preserved V11a result metadata without rerunning inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from run_heldout_case_evaluation import (
    EVAL_QUERY_JSONL,
    OUTPUT_DIR,
    SELECTED_CASES,
    TEST_JSONL,
    TRAIN_JSONL,
    by_case_id,
    load_jsonl,
)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    for path in (TEST_JSONL, EVAL_QUERY_JSONL, TRAIN_JSONL):
        if not path.exists():
            raise FileNotFoundError(path)

    test_by_id = by_case_id(load_jsonl(TEST_JSONL))
    eval_by_id = by_case_id(load_jsonl(EVAL_QUERY_JSONL))
    train_by_id = by_case_id(load_jsonl(TRAIN_JSONL))
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    health = summary.get("backend_health", {})
    runtime_kb = health.get("kb_path", "unknown")

    refreshed_results: list[dict[str, Any]] = []
    for config in SELECTED_CASES:
        case_id = config["case_id"]
        if len(test_by_id.get(case_id, [])) != 1:
            raise RuntimeError(f"{case_id}: held-out row count is not one")
        if len(eval_by_id.get(case_id, [])) != 1:
            raise RuntimeError(f"{case_id}: eval-query row count is not one")
        if train_by_id.get(case_id):
            raise RuntimeError(
                f"{case_id}: present in official experimental retrieval corpus"
            )

        path = OUTPUT_DIR / f"{case_id}_result.json"
        result = json.loads(path.read_text(encoding="utf-8"))
        reference_label = result.get(
            "silver_reference_label", result.get("expected_diagnosis", "Unknown")
        )
        result["silver_reference_label"] = reference_label
        result["split_provenance"] = {
            "held_out_source": str(TEST_JSONL),
            "eval_query_source": str(EVAL_QUERY_JSONL),
            "clinical_retrieval_corpus_source": str(TRAIN_JSONL),
            "runtime_retrieval_kb_source": runtime_kb,
            "test_count": 1,
            "eval_query_count": 1,
            "train_count": 0,
            "source_terms": {
                "official_experimental_retrieval_corpus": (
                    "121-case phase1b Tier A+B train artifact; used for "
                    "split-exclusion verification"
                ),
                "saved_demo_runtime_kb": (
                    "small local defense demo KB used by the preserved backend "
                    "output"
                ),
                "held_out_evaluation_set": "56 non-indexed evaluation cases",
            },
            "label_status": result.get("label_contract", {}).get(
                "ground_truth_status", "silver_reference_only"
            ),
        }
        write_json(path, result)
        refreshed_results.append(result)

    summary["case_source"] = "56-case held-out evaluation set"
    summary["official_experimental_retrieval_corpus"] = {
        "description": "121-case phase1b Tier A+B train artifact",
        "path": str(TRAIN_JSONL),
        "use": "split-exclusion verification",
    }
    summary["runtime_retrieval_kb"] = {
        "description": "small local defense demo KB",
        "path": runtime_kb,
        "use": "retrieval support for the preserved backend outputs",
    }
    summary["preservation_note"] = (
        "Metadata refreshed for V11a; no new model inference was run."
    )
    summary.pop("clinical_retrieval_corpus", None)
    summary["results"] = refreshed_results
    write_json(summary_path, summary)

    print(
        "Refreshed preserved result provenance: 3 held-out cases, "
        "121-case split exclusion, local demo-KB runtime disclosed."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
