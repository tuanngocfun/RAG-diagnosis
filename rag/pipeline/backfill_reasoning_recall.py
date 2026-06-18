#!/usr/bin/env python3
"""Backfill reasoning-recall fields into existing phased RAGAS artifacts."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

try:
    from .ragas_evaluator import RAGAsLibraryEvaluator, _sanitize_record_for_json
    from .ragas_summary import update_summary_with_ragas_metrics
    from .run_ragas_phased import (
        FINAL_RAGAS_FILE,
        PHASE_A,
        _load_answers,
        _load_unique_jsonl,
        _write_jsonl,
    )
except ImportError:  # pragma: no cover - direct script execution fallback
    from pipeline.ragas_evaluator import RAGAsLibraryEvaluator, _sanitize_record_for_json
    from pipeline.ragas_summary import update_summary_with_ragas_metrics
    from pipeline.run_ragas_phased import (
        FINAL_RAGAS_FILE,
        PHASE_A,
        _load_answers,
        _load_unique_jsonl,
        _write_jsonl,
    )


_ANSWER_FILE_CANDIDATES = (
    "answers.jsonl",
    "answers_rag_std.jsonl",
    "answers_rag.jsonl",
    "answers_norag.jsonl",
    "answers_gemini.jsonl",
)


def _resolve_answers_file(run_dir: Path, requested: Optional[str]) -> str:
    if requested:
        candidate = run_dir / requested
        if not candidate.exists():
            raise FileNotFoundError(f"Missing answers file: {candidate}")
        return requested

    manifest_path = run_dir / "eval_manifest.json"
    if manifest_path.exists():
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        manifest_answers = str(manifest.get("answers_file") or "").strip()
        if manifest_answers and (run_dir / manifest_answers).exists():
            return manifest_answers

    for candidate in _ANSWER_FILE_CANDIDATES:
        if (run_dir / candidate).exists():
            return candidate
    raise FileNotFoundError(f"Could not auto-detect answers file in {run_dir}")


def _row_has_reasoning_recall_payload(row: Dict[str, Any]) -> bool:
    if "reasoning_recall_method" not in row:
        return False
    if "reasoning_recall_source_id" not in row:
        return False
    if "reasoning_recall_source_path" not in row and "reasoning_recall_source" not in row:
        return False
    diagnostics = (row.get("traces") or {}).get("reasoning_recall_diagnostics")
    if not isinstance(diagnostics, dict):
        return False
    required_diagnostics = {
        "method",
        "groundtruth_count",
        "matched_count",
        "predicted_step_count",
        "judge_model",
        "requested_judge_model",
        "source_id",
        "source_path",
    }
    return required_diagnostics.issubset(diagnostics.keys())


def _apply_reasoning_recall_payload(row: Dict[str, Any], payload: Dict[str, Any]) -> Dict[str, Any]:
    enriched = dict(row)
    enriched["reasoning_recall"] = payload.get("recall")
    enriched["reasoning_recall_method"] = str(payload.get("method") or "")
    enriched["reasoning_recall_groundtruth_count"] = payload.get("groundtruth_count")
    enriched["reasoning_recall_matched_count"] = payload.get("matched_groundtruth_count")
    enriched["reasoning_recall_explanation"] = payload.get("explanation")
    enriched["reasoning_recall_matching_dict"] = {
        "matched_groundtruth_indices": payload.get("matched_groundtruth_indices", []),
        "matched_groundtruth_points": payload.get("matched_groundtruth_points", []),
        "unmatched_groundtruth_points": payload.get("unmatched_groundtruth_points", []),
    }
    enriched["reasoning_recall_source"] = str(payload.get("source_path") or "")
    enriched["reasoning_recall_source_id"] = str(payload.get("source_id") or "")
    enriched["reasoning_recall_source_path"] = str(payload.get("source_path") or "")
    enriched["reasoning_recall_judge_model"] = str(payload.get("judge_model") or "")
    enriched["reasoning_trace_source"] = "answer_proxy"
    traces = dict(enriched.get("traces") or {})
    traces["reasoning_recall_diagnostics"] = {
        "method": enriched["reasoning_recall_method"],
        "groundtruth_count": enriched["reasoning_recall_groundtruth_count"],
        "matched_count": enriched["reasoning_recall_matched_count"],
        "predicted_step_count": len(payload.get("predicted_reasoning_steps") or []),
        "judge_model": enriched["reasoning_recall_judge_model"],
        "requested_judge_model": str(payload.get("requested_judge_model") or ""),
        "source_id": enriched["reasoning_recall_source_id"],
        "source_path": enriched["reasoning_recall_source"],
    }
    enriched["traces"] = traces
    return _sanitize_record_for_json(enriched)


async def _backfill_rows(
    evaluator: Any,
    ordered_qids: Sequence[str],
    samples_by_qid: Dict[str, Dict[str, Any]],
    rows_by_qid: Dict[str, Dict[str, Any]],
    overwrite: bool,
) -> Tuple[Dict[str, Dict[str, Any]], int, int]:
    updated_rows: Dict[str, Dict[str, Any]] = dict(rows_by_qid)
    evaluated = 0
    skipped = 0

    for qid in ordered_qids:
        row = rows_by_qid.get(qid)
        if row is None:
            raise RuntimeError(f"Missing row for qid={qid}")
        if not overwrite and _row_has_reasoning_recall_payload(row):
            skipped += 1
            continue
        sample = samples_by_qid[qid]
        payload = await evaluator.evaluate_reasoning_recall_for_sample(qid=qid, answer=sample["answer"])
        updated_rows[qid] = _apply_reasoning_recall_payload(row, payload)
        evaluated += 1

    return updated_rows, evaluated, skipped


def backfill_run_dir(
    run_dir: Path,
    answers_file: Optional[str] = None,
    judge_model: Optional[str] = None,
    evaluator: Optional[Any] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    run_dir = Path(run_dir).expanduser().resolve()
    resolved_answers_file = _resolve_answers_file(run_dir, answers_file)
    ordered_qids, samples_by_qid, _, _ = _load_answers(run_dir, resolved_answers_file)

    if evaluator is None:
        evaluator = RAGAsLibraryEvaluator(model=judge_model)

    stats: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "answers_file": resolved_answers_file,
        "phase_a_updated": 0,
        "phase_a_skipped": 0,
        "ragas_updated": 0,
        "ragas_skipped": 0,
        "summary_updated": False,
    }

    touched_any = False

    phase_a_path = run_dir / "phase_a_metrics.jsonl"
    if phase_a_path.exists():
        _, phase_a_rows = _load_unique_jsonl(phase_a_path, PHASE_A)
        updated_phase_a_rows, evaluated, skipped = asyncio.run(
            _backfill_rows(
                evaluator=evaluator,
                ordered_qids=ordered_qids,
                samples_by_qid=samples_by_qid,
                rows_by_qid=phase_a_rows,
                overwrite=overwrite,
            )
        )
        _write_jsonl(phase_a_path, ordered_qids, updated_phase_a_rows)
        stats["phase_a_updated"] = evaluated
        stats["phase_a_skipped"] = skipped
        touched_any = True

    ragas_path = run_dir / FINAL_RAGAS_FILE
    if ragas_path.exists():
        _, ragas_rows = _load_unique_jsonl(ragas_path, FINAL_RAGAS_FILE)
        updated_ragas_rows, evaluated, skipped = asyncio.run(
            _backfill_rows(
                evaluator=evaluator,
                ordered_qids=ordered_qids,
                samples_by_qid=samples_by_qid,
                rows_by_qid=ragas_rows,
                overwrite=overwrite,
            )
        )
        _write_jsonl(ragas_path, ordered_qids, updated_ragas_rows)
        stats["ragas_updated"] = evaluated
        stats["ragas_skipped"] = skipped
        touched_any = True
    else:
        raise FileNotFoundError(f"Missing final ragas file: {ragas_path}")

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        update_summary_with_ragas_metrics(run_dir)
        stats["summary_updated"] = True

    if not touched_any:
        raise FileNotFoundError(f"No phased or final evaluation artifacts found in {run_dir}")

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill reasoning-recall fields into existing RAGAS artifacts")
    parser.add_argument("--run-dir", required=True, help="Run directory containing phase_a_metrics.jsonl or ragas.jsonl")
    parser.add_argument("--answers-file", default=None, help="Optional answers JSONL filename inside the run dir")
    parser.add_argument("--judge-model", default=None, help="Optional judge model override")
    parser.add_argument("--overwrite", action="store_true", help="Recompute reasoning recall even when rows already contain it")
    args = parser.parse_args()

    stats = backfill_run_dir(
        run_dir=Path(args.run_dir),
        answers_file=args.answers_file,
        judge_model=args.judge_model,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
