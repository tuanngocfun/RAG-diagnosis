"""Helpers for persisting RAGAS aggregates back into summary.json."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


_COMMON_METRIC_MAP = {
    "multimodal_faithfulness": "multimodal_faithfulness",
    "multimodal_relevance": "multimodal_relevance",
    "context_relevance": "context_relevance",
    "grounded_accuracy": "traces.grounded_accuracy",
    "reasoning_recall": "reasoning_recall",
}

_VERIFIED_METRIC_MAP = {
    "l3_top1_correct": "l3_top1_correct",
    "top3_hit": "top3_hit",
    "diagnosis_accuracy": "diagnosis_accuracy",
    "diagnosis_type_accuracy": "diagnosis_type_accuracy",
    "diagnosis_family_accuracy": "diagnosis_family_accuracy",
}

_PSEUDOLABEL_METRIC_MAP = {
    "l3_top1_correct": "l3_top1_correct_pseudolabel",
    "top3_hit": "top3_hit_pseudolabel",
    "diagnosis_accuracy": "diagnosis_accuracy_pseudolabel",
    "diagnosis_type_accuracy": "diagnosis_type_accuracy_pseudolabel",
}


def _load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _as_number(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return None


def _extract_value(row: Dict[str, Any], key_path: str) -> Any:
    current: Any = row
    for part in key_path.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def _mean_metric(rows: Iterable[Dict[str, Any]], key_path: str) -> Optional[float]:
    values = []
    for row in rows:
        value = _as_number(_extract_value(row, key_path))
        if value is not None:
            values.append(value)
    if not values:
        return None
    return sum(values) / len(values)


def _non_empty_values(rows: Iterable[Dict[str, Any]], key: str) -> List[str]:
    values = []
    for row in rows:
        value = str(row.get(key) or "").strip()
        if value:
            values.append(value)
    return values


def _reasoning_recall_summary(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    available_count = 0
    missing_groundtruth = 0
    missing_trace = 0
    judge_errors = 0

    for row in rows:
        method = str(row.get("reasoning_recall_method") or "")
        if _as_number(row.get("reasoning_recall")) is not None:
            available_count += 1
        if method == "skipped_missing_groundtruth_reasoning":
            missing_groundtruth += 1
        elif method == "skipped_unparseable_predicted_reasoning":
            missing_trace += 1
        elif method == "judge_error":
            judge_errors += 1

    method_counts = Counter(value or "MISSING" for value in _non_empty_values(rows, "reasoning_recall_method"))
    if not method_counts and total:
        method_counts["MISSING"] = total

    return {
        "reasoning_recall_coverage": (available_count / total) if total else None,
        "reasoning_recall_available_count": available_count,
        "reasoning_recall_missing_groundtruth_count": missing_groundtruth,
        "reasoning_recall_missing_trace_count": missing_trace,
        "reasoning_recall_judge_error_count": judge_errors,
        "reasoning_recall_method_counts": dict(method_counts),
        "reasoning_recall_judge_models": sorted(set(_non_empty_values(rows, "reasoning_recall_judge_model"))),
        "reasoning_recall_source_ids": sorted(set(_non_empty_values(rows, "reasoning_recall_source_id"))),
        "reasoning_recall_source_paths": sorted(set(_non_empty_values(rows, "reasoning_recall_source_path"))),
    }


def _track_metric_map(track: str) -> Dict[str, str]:
    if track == "pseudolabel":
        return {**_COMMON_METRIC_MAP, **_PSEUDOLABEL_METRIC_MAP}
    return {**_COMMON_METRIC_MAP, **_VERIFIED_METRIC_MAP}


def build_ragas_metrics_block(rows: List[Dict[str, Any]], track: str = "verified") -> Dict[str, Any]:
    metric_map = _track_metric_map(track)
    block: Dict[str, Any] = {"n_rows": len(rows)}
    for output_key, source_key in metric_map.items():
        block[output_key] = _mean_metric(rows, source_key)
    block.update(_reasoning_recall_summary(rows))
    return block


def has_pseudolabel_track(rows: List[Dict[str, Any]]) -> bool:
    for row in rows:
        if any(
            _as_number(row.get(key)) is not None
            for key in (
                "diagnosis_accuracy_pseudolabel",
                "diagnosis_type_accuracy_pseudolabel",
                "top3_hit_pseudolabel",
                "l3_top1_correct_pseudolabel",
            )
        ):
            return True
    return False


def update_summary_with_ragas_metrics(
    run_dir: Path,
    summary_file_name: str = "summary.json",
    ragas_file_name: str = "ragas.jsonl",
) -> Dict[str, Any]:
    summary_path = run_dir / summary_file_name
    ragas_path = run_dir / ragas_file_name
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")
    if not ragas_path.exists():
        raise FileNotFoundError(f"Missing ragas file: {ragas_path}")

    summary = _load_json(summary_path)
    rows = _load_jsonl(ragas_path)

    verified_block = build_ragas_metrics_block(rows, track="verified")
    summary["ragas_metrics_verified"] = verified_block
    summary["ragas_metrics"] = verified_block

    if has_pseudolabel_track(rows):
        summary["ragas_metrics_pseudolabel"] = build_ragas_metrics_block(rows, track="pseudolabel")
    else:
        summary.pop("ragas_metrics_pseudolabel", None)

    metrics = summary.get("metrics")
    if isinstance(metrics, dict):
        metrics["grounded_accuracy"] = verified_block.get("grounded_accuracy")

    metrics_verified = summary.get("metrics_verified")
    if isinstance(metrics_verified, dict):
        metrics_verified["grounded_accuracy"] = verified_block.get("grounded_accuracy")

    _write_json(summary_path, summary)
    return summary
