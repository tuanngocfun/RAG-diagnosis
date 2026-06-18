#!/usr/bin/env python3
"""Phased, resume-safe RAGAS evaluation workflow."""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None

from .config import JUDGE_MODEL, JUDGE_MODEL_FALLBACK, get_runtime_metadata
from .diagnosis_output_parser import compute_family_metric_details, evaluate_ranked_diagnosis_contract
from .ragas_evaluator import (
    RAGAsLibraryEvaluator,
    _sanitize_record_for_json,
)
from .ragas_summary import update_summary_with_ragas_metrics

PHASE_A = "phase_a_metrics"
PHASE_B = "phase_b_verified"
PHASE_C = "phase_c_pseudolabel"
MERGE_PHASE = "merge"
PHASE_SEQUENCE = [PHASE_A, PHASE_B, PHASE_C, MERGE_PHASE]
PHASE_FILE_NAMES = {
    PHASE_A: "phase_a_metrics.jsonl",
    PHASE_B: "phase_b_verified.jsonl",
    PHASE_C: "phase_c_pseudolabel.jsonl",
}
PHASE_LABEL_TRACK = {
    PHASE_A: "retrieval_metrics",
    PHASE_B: "verified",
    PHASE_C: "pseudolabel",
}
START_PHASE_ALIASES = {
    "all": PHASE_A,
    "phase_a": PHASE_A,
    "phase_a_metrics": PHASE_A,
    "phase_b": PHASE_B,
    "phase_b_verified": PHASE_B,
    "phase_c": PHASE_C,
    "phase_c_pseudolabel": PHASE_C,
    "merge": MERGE_PHASE,
}
MANIFEST_FILE = "eval_manifest.json"
MANIFEST_LOCK_FILE = "eval_manifest.lock"
FINAL_RAGAS_FILE = "ragas.jsonl"
ACTIVE_BATCH_STATE_FLAGS = ("PENDING", "RUNNING")
DIAGNOSIS_PHASE_REQUIRED_FIELDS = {
    PHASE_B: (
        "qid",
        "diagnosis_accuracy",
        "diagnosis_type_accuracy",
        "gt_rank",
        "top3_hit",
        "l3_top1_correct",
        "fallback_level",
        "rank_source",
        "judge_gt_rank",
        "parser_gt_rank",
        "judge_parser_disagreement",
        "diagnosis_reasoning",
        "diagnosis_method",
        "phase",
        "label_track",
        "answers_sha256",
        "eval_scope_id",
    ),
    PHASE_C: (
        "qid",
        "diagnosis_accuracy_pseudolabel",
        "diagnosis_type_accuracy_pseudolabel",
        "gt_rank_pseudolabel",
        "top3_hit_pseudolabel",
        "l3_top1_correct_pseudolabel",
        "fallback_level_pseudolabel",
        "rank_source_pseudolabel",
        "judge_gt_rank_pseudolabel",
        "parser_gt_rank_pseudolabel",
        "judge_parser_disagreement_pseudolabel",
        "diagnosis_reasoning_pseudolabel",
        "diagnosis_method_pseudolabel",
        "phase",
        "label_track",
        "answers_sha256",
        "eval_scope_id",
    ),
}


def _coerce_rank(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return int(value)
    try:
        text = str(value).strip().lower()
        if text in {"", "none", "null", "na", "n/a"}:
            return None
        return int(float(text))
    except Exception:
        return None


def _coerce_binary_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        val = float(value)
        if math.isnan(val):
            return None
        return 0.0 if val <= 0.0 else 1.0
    try:
        text = str(value).strip().lower()
        if text in {"", "none", "null", "na", "n/a"}:
            return None
        val = float(text)
        if math.isnan(val):
            return None
        return 0.0 if val <= 0.0 else 1.0
    except Exception:
        return None


def _resolve_rank_contract_result(
    judge_result: Dict[str, Any],
    parser_contract: Dict[str, Any],
) -> Dict[str, Any]:
    judge_gt_rank = _coerce_rank(judge_result.get("gt_rank"))
    parser_gt_rank = _coerce_rank(parser_contract.get("gt_rank"))

    judge_top3 = _coerce_binary_score(judge_result.get("top3_hit"))
    judge_l3 = _coerce_binary_score(judge_result.get("l3_top1_correct"))
    parser_top3 = _coerce_binary_score(parser_contract.get("top3_hit"))
    parser_l3 = _coerce_binary_score(parser_contract.get("l3_top1_correct"))

    if judge_top3 is None and judge_gt_rank is not None:
        judge_top3 = 1.0 if judge_gt_rank in {1, 2, 3} else 0.0
    if judge_l3 is None and judge_gt_rank is not None:
        judge_l3 = 1.0 if judge_gt_rank == 1 else 0.0

    if parser_top3 is None and parser_gt_rank is not None:
        parser_top3 = 1.0 if parser_gt_rank in {1, 2, 3} else 0.0
    if parser_l3 is None and parser_gt_rank is not None:
        parser_l3 = 1.0 if parser_gt_rank == 1 else 0.0

    if judge_gt_rank is not None:
        gt_rank = judge_gt_rank
        top3_hit = judge_top3
        l3_top1_correct = judge_l3
        rank_source = "judge"
        fallback_level = str(judge_result.get("fallback_level") or parser_contract.get("fallback_level") or "judge")
    else:
        gt_rank = parser_gt_rank
        top3_hit = parser_top3
        l3_top1_correct = parser_l3
        rank_source = "parser"
        fallback_level = str(parser_contract.get("fallback_level") or "parser_missing")

    disagreement: Optional[bool] = None
    if judge_gt_rank is not None and parser_gt_rank is not None:
        disagreement = bool(judge_gt_rank != parser_gt_rank)
    elif judge_l3 is not None and parser_l3 is not None:
        disagreement = bool(judge_l3 != parser_l3)

    return {
        "gt_rank": gt_rank,
        "top3_hit": top3_hit,
        "l3_top1_correct": l3_top1_correct,
        "fallback_level": fallback_level[:80],
        "rank_source": rank_source,
        "judge_gt_rank": judge_gt_rank,
        "parser_gt_rank": parser_gt_rank,
        "judge_parser_disagreement": disagreement,
    }


class ActiveDiagnosisBatch(RuntimeError):
    """Raised when a remote Gemini batch is still active and should be reattached later."""

    def __init__(
        self,
        job_id: str,
        state: str,
        timeout_seconds: int,
        phase: str = "",
    ):
        super().__init__(f"Diagnosis batch still active after {timeout_seconds}s: {state}")
        self.job_id = job_id
        self.state = state
        self.timeout_seconds = timeout_seconds
        self.phase = phase


def _now_ts() -> float:
    return time.time()


def _phase_output_path(run_dir: Path, phase: str) -> Path:
    if phase == MERGE_PHASE:
        return run_dir / FINAL_RAGAS_FILE
    return run_dir / PHASE_FILE_NAMES[phase]


def _normalize_start_phase(start_phase: str) -> str:
    key = (start_phase or "all").strip().lower()
    if key not in START_PHASE_ALIASES:
        raise ValueError(f"Unknown start_phase={start_phase!r}. Choose from: {sorted(START_PHASE_ALIASES)}")
    return START_PHASE_ALIASES[key]


def _stable_json(value: Any) -> str:
    return json.dumps(_sanitize_record_for_json(value), sort_keys=True, ensure_ascii=False)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_lines(values: Sequence[str]) -> str:
    h = hashlib.sha256()
    for value in values:
        h.update(str(value).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _bucket_from_ground_truth(ground_truth: Optional[Dict[str, Any]]) -> str:
    gt_type = str((ground_truth or {}).get("diagnosis_type", "") or "").strip()
    if gt_type == "Non-Leishmaniasis":
        return "nonleish"
    if gt_type:
        return "leish"
    return "unknown"


def _build_eval_scope_id(
    run_id: str,
    answers_sha256: str,
    phase: str,
    label_track: str,
    qid: str,
) -> str:
    return "|".join([run_id, answers_sha256, phase, label_track, qid])


def _load_unique_jsonl(path: Path, label: str) -> Tuple[List[str], Dict[str, Dict[str, Any]]]:
    ordered_qids: List[str] = []
    rows_by_qid: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return ordered_qids, rows_by_qid

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            row = _sanitize_record_for_json(json.loads(line))
            qid = str(row.get("qid", "") or "").strip()
            if not qid:
                raise RuntimeError(f"{label} has a row without qid at line {line_no}: {path}")
            if qid in rows_by_qid:
                if _stable_json(rows_by_qid[qid]) != _stable_json(row):
                    raise RuntimeError(f"Conflicting duplicate qid={qid} in {label}: {path}")
                continue
            ordered_qids.append(qid)
            rows_by_qid[qid] = row
    return ordered_qids, rows_by_qid


def _write_jsonl(path: Path, ordered_qids: Sequence[str], rows_by_qid: Dict[str, Dict[str, Any]]) -> None:
    extra_qids = sorted(set(rows_by_qid) - set(ordered_qids))
    if extra_qids:
        raise RuntimeError(f"Cannot write {path}: found rows with unexpected qids: {extra_qids[:5]}")
    with open(path, "w", encoding="utf-8") as f:
        for qid in ordered_qids:
            row = rows_by_qid.get(qid)
            if row is None:
                continue
            f.write(json.dumps(_sanitize_record_for_json(row), ensure_ascii=False) + "\n")


def _load_answers(
    run_dir: Path,
    answers_file: str,
    max_samples: Optional[int] = None,
) -> Tuple[List[str], Dict[str, Dict[str, Any]], str, str]:
    answers_path = run_dir / answers_file
    if not answers_path.exists():
        raise FileNotFoundError(f"No {answers_file} in {run_dir}")

    answers_sha256 = _sha256_file(answers_path)
    ordered_qids, samples_by_qid = _load_unique_jsonl(answers_path, "answers")
    if max_samples is not None:
        ordered_qids = ordered_qids[:max_samples]
        samples_by_qid = {qid: samples_by_qid[qid] for qid in ordered_qids}
    selected_qids_sha256 = _sha256_lines(ordered_qids)
    return ordered_qids, samples_by_qid, answers_sha256, selected_qids_sha256


def _load_queries_ground_truth_maps(run_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    queries_gt_map: Dict[str, Dict[str, Any]] = {}
    queries_gt_pseudo_map: Dict[str, Dict[str, Any]] = {}
    queries_path = run_dir / "queries.json"
    if not queries_path.exists():
        return queries_gt_map, queries_gt_pseudo_map

    try:
        with open(queries_path, "r", encoding="utf-8") as f:
            queries_payload = json.load(f)
        if isinstance(queries_payload, list):
            for query in queries_payload:
                case_id = query.get("case_id")
                query_type = query.get("query_type")
                ground_truth = query.get("ground_truth")
                ground_truth_pseudolabel = query.get("ground_truth_pseudolabel")
                if case_id and query_type and isinstance(ground_truth, dict):
                    queries_gt_map[f"{case_id}::{query_type}"] = ground_truth
                if case_id and query_type and isinstance(ground_truth_pseudolabel, dict):
                    queries_gt_pseudo_map[f"{case_id}::{query_type}"] = ground_truth_pseudolabel
    except Exception as exc:  # pragma: no cover - defensive fallback
        print(f"Warning: Could not load queries.json for GT backfill: {exc}")
    return queries_gt_map, queries_gt_pseudo_map


def _backfill_ground_truth(
    sample: Dict[str, Any],
    qid: str,
    queries_gt_map: Dict[str, Dict[str, Any]],
    queries_gt_pseudo_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    enriched = dict(sample)
    ground_truth = enriched.get("ground_truth")
    ground_truth_pseudolabel = enriched.get("ground_truth_pseudolabel")

    if not ground_truth and qid in queries_gt_map:
        ground_truth = queries_gt_map[qid]
    elif isinstance(ground_truth, dict):
        if not any(ground_truth.get(key) for key in ["diagnosis", "diagnosis_type", "species"]) and qid in queries_gt_map:
            ground_truth = queries_gt_map[qid]

    if not ground_truth_pseudolabel and qid in queries_gt_pseudo_map:
        ground_truth_pseudolabel = queries_gt_pseudo_map[qid]
    elif isinstance(ground_truth_pseudolabel, dict):
        if not any(ground_truth_pseudolabel.get(key) for key in ["diagnosis", "diagnosis_type", "species"]) and qid in queries_gt_pseudo_map:
            ground_truth_pseudolabel = queries_gt_pseudo_map[qid]

    if not enriched.get("query_images"):
        enriched["query_images"] = enriched.get("image_paths", [])

    enriched["ground_truth"] = ground_truth
    enriched["ground_truth_pseudolabel"] = ground_truth_pseudolabel
    return enriched


def _new_manifest(
    run_dir: Path,
    answers_file: str,
    answers_sha256: str,
    selected_qids_sha256: str,
    ordered_qids: Sequence[str],
    judge_model: Optional[str],
) -> Dict[str, Any]:
    manifest: Dict[str, Any] = {
        "version": 2,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "answers_file": answers_file,
        "answers_sha256": answers_sha256,
        "selected_qids_sha256": selected_qids_sha256,
        "selected_qid_count": len(ordered_qids),
        "judge_model": judge_model or JUDGE_MODEL,
        "runtime_metadata": get_runtime_metadata(),
        "updated_at": _now_ts(),
        "phases": {},
    }
    for phase in PHASE_SEQUENCE:
        phase_entry: Dict[str, Any] = {
            "status": "pending",
            "file": FINAL_RAGAS_FILE if phase == MERGE_PHASE else PHASE_FILE_NAMES[phase],
            "label_track": PHASE_LABEL_TRACK.get(phase, ""),
            "n_completed": 0,
            "updated_at": None,
        }
        if phase in {PHASE_B, PHASE_C}:
            phase_entry["batch_job_id"] = None
            phase_entry["batch_judge_model"] = None
            phase_entry["last_observed_batch_state"] = None
        manifest["phases"][phase] = phase_entry
    return manifest


def _manifest_matches(
    manifest: Dict[str, Any],
    run_dir: Path,
    answers_file: str,
    answers_sha256: str,
    selected_qids_sha256: str,
    ordered_qids: Sequence[str],
) -> bool:
    return (
        manifest.get("run_id") == run_dir.name
        and manifest.get("answers_file") == answers_file
        and manifest.get("answers_sha256") == answers_sha256
        and manifest.get("selected_qids_sha256") == selected_qids_sha256
        and int(manifest.get("selected_qid_count", -1)) == len(ordered_qids)
    )


def _load_manifest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_manifest(path: Path, manifest: Dict[str, Any]) -> None:
    manifest["updated_at"] = _now_ts()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)


def _clear_phase_outputs(run_dir: Path) -> None:
    for name in [MANIFEST_FILE, *PHASE_FILE_NAMES.values(), FINAL_RAGAS_FILE]:
        path = run_dir / name
        if path.exists():
            path.unlink()


def _refresh_manifest_from_files(
    manifest: Dict[str, Any],
    run_dir: Path,
    ordered_qids: Sequence[str],
) -> None:
    expected_qids = set(ordered_qids)
    for phase in PHASE_SEQUENCE:
        phase_state = manifest["phases"][phase]
        phase_path = _phase_output_path(run_dir, phase)
        if not phase_path.exists():
            phase_state["n_completed"] = 0
            if phase_state.get("status") == "completed":
                phase_state["status"] = "pending"
            continue
        _, rows_by_qid = _load_unique_jsonl(phase_path, phase)
        if set(rows_by_qid) - expected_qids:
            raise RuntimeError(f"{phase_path} contains qids outside current answer scope")
        phase_state["n_completed"] = len(rows_by_qid)
        if len(rows_by_qid) == len(ordered_qids):
            phase_state["status"] = "completed"
        elif len(rows_by_qid) > 0:
            phase_state["status"] = "partial"
        else:
            phase_state["status"] = "pending"


def _prepare_manifest(
    run_dir: Path,
    answers_file: str,
    answers_sha256: str,
    selected_qids_sha256: str,
    ordered_qids: Sequence[str],
    judge_model: Optional[str],
    resume: bool,
) -> Tuple[Path, Dict[str, Any]]:
    manifest_path = run_dir / MANIFEST_FILE
    manifest = _load_manifest(manifest_path)
    if resume and manifest and _manifest_matches(manifest, run_dir, answers_file, answers_sha256, selected_qids_sha256, ordered_qids):
        pass
    else:
        if manifest is not None:
            print("  Manifest invalidated: answers file, hash, or selected qids changed; starting phased evaluation fresh")
        elif not resume:
            print("  Resume disabled: starting phased evaluation fresh")
        _clear_phase_outputs(run_dir)
        manifest = _new_manifest(run_dir, answers_file, answers_sha256, selected_qids_sha256, ordered_qids, judge_model)
    _refresh_manifest_from_files(manifest, run_dir, ordered_qids)
    _save_manifest(manifest_path, manifest)
    return manifest_path, manifest


def _acquire_manifest_lock(run_dir: Path):
    lock_path = run_dir / MANIFEST_LOCK_FILE
    lock_fh = open(lock_path, "a+")
    if fcntl is not None:
        try:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            lock_fh.seek(0)
            holder = lock_fh.read().strip()
            raise RuntimeError(
                "Another phased evaluation is already running for this run_dir. "
                f"Lock file: {lock_path}. Holder info: {holder or 'unknown'}"
            ) from exc
    lock_fh.seek(0)
    lock_fh.truncate(0)
    lock_fh.write(json.dumps({"pid": os.getpid(), "started_at": _now_ts()}))
    lock_fh.flush()
    return lock_fh


def _release_manifest_lock(lock_fh) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(lock_fh.fileno(), fcntl.LOCK_UN)
    finally:
        lock_fh.close()


def _phase_index(phase: str) -> int:
    return PHASE_SEQUENCE.index(phase)


def _ensure_phase_prerequisites(manifest: Dict[str, Any], start_phase: str) -> None:
    target_idx = _phase_index(start_phase)
    for phase in PHASE_SEQUENCE[:target_idx]:
        if manifest["phases"][phase].get("status") != "completed":
            raise RuntimeError(
                f"Cannot start at {start_phase}: prerequisite {phase} is not completed in {MANIFEST_FILE}"
            )


def _annotate_phase_row(
    row: Dict[str, Any],
    run_id: str,
    answers_sha256: str,
    phase: str,
    label_track: str,
) -> Dict[str, Any]:
    qid = row["qid"]
    enriched = dict(row)
    enriched["phase"] = phase
    enriched["label_track"] = label_track
    enriched["answers_sha256"] = answers_sha256
    enriched["eval_scope_id"] = _build_eval_scope_id(run_id, answers_sha256, phase, label_track, qid)
    return _sanitize_record_for_json(enriched)


def _validate_phase_output(
    run_dir: Path,
    phase: str,
    ordered_qids: Sequence[str],
    answers_sha256: str,
    run_id: str,
) -> Dict[str, Dict[str, Any]]:
    phase_path = _phase_output_path(run_dir, phase)
    if not phase_path.exists():
        raise RuntimeError(f"Cannot validate {phase}: missing file {phase_path}")

    _, rows_by_qid = _load_unique_jsonl(phase_path, phase)
    expected_qids = list(ordered_qids)
    missing = [qid for qid in expected_qids if qid not in rows_by_qid]
    unexpected = sorted(set(rows_by_qid) - set(expected_qids))
    if missing:
        raise RuntimeError(f"{phase_path} is missing qids, e.g. {missing[:5]}")
    if unexpected:
        raise RuntimeError(f"{phase_path} has qids outside the current answer scope, e.g. {unexpected[:5]}")
    if len(rows_by_qid) != len(expected_qids):
        raise RuntimeError(
            f"{phase_path} row count mismatch: expected {len(expected_qids)}, found {len(rows_by_qid)}"
        )

    label_track = PHASE_LABEL_TRACK.get(phase, "")
    required_fields = DIAGNOSIS_PHASE_REQUIRED_FIELDS.get(phase, ())
    for qid in expected_qids:
        row = rows_by_qid[qid]
        missing_fields = [field for field in required_fields if field not in row]
        if missing_fields:
            raise RuntimeError(f"{phase_path} row for {qid} is missing required fields: {missing_fields}")
        if row.get("phase") != phase:
            raise RuntimeError(f"{phase_path} row for {qid} has phase={row.get('phase')!r}, expected {phase!r}")
        if row.get("label_track") != label_track:
            raise RuntimeError(
                f"{phase_path} row for {qid} has label_track={row.get('label_track')!r}, expected {label_track!r}"
            )
        if row.get("answers_sha256") != answers_sha256:
            raise RuntimeError(
                f"{phase_path} row for {qid} has answers_sha256={row.get('answers_sha256')!r}, "
                f"expected {answers_sha256!r}"
            )
        expected_scope = _build_eval_scope_id(run_id, answers_sha256, phase, label_track, qid)
        if row.get("eval_scope_id") != expected_scope:
            raise RuntimeError(
                f"{phase_path} row for {qid} has eval_scope_id={row.get('eval_scope_id')!r}, "
                f"expected {expected_scope!r}"
            )
    return rows_by_qid


def _compute_aggregates(rows: Sequence[Dict[str, Any]]) -> Dict[str, Optional[float]]:
    metrics = [
        "l3_top1_correct",
        "top3_hit",
        "multimodal_faithfulness",
        "multimodal_relevance",
        "context_relevance",
        "reasoning_recall",
        "diagnosis_accuracy",
        "diagnosis_type_accuracy",
        "diagnosis_family_accuracy",
        "l3_top1_correct_pseudolabel",
        "top3_hit_pseudolabel",
        "diagnosis_accuracy_pseudolabel",
        "diagnosis_type_accuracy_pseudolabel",
    ]
    aggregates: Dict[str, Optional[float]] = {}
    for metric in metrics:
        values: List[float] = []
        for row in rows:
            value = row.get(metric)
            if isinstance(value, (int, float)) and not (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
                values.append(float(value))
        aggregates[metric] = (sum(values) / len(values)) if values else None
    return aggregates


def _print_aggregates(output_path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    agg = _compute_aggregates(rows)
    print(f"\n✓ RAGAS evaluation saved to {output_path}")
    print("  Using: Official RAGAS library (Collections API)")
    print("\n  --- Generation Metrics ---")
    mf = f"{agg['multimodal_faithfulness']:.4f}" if agg["multimodal_faithfulness"] is not None else "NA"
    mr = f"{agg['multimodal_relevance']:.4f}" if agg["multimodal_relevance"] is not None else "NA"
    print(f"  Multimodal Faithfulness: {mf}")
    print(f"  Multimodal Relevance: {mr}")
    print("\n  --- Retrieval Metrics ---")
    cr = f"{agg['context_relevance']:.4f}" if agg["context_relevance"] is not None else "NA"
    print(f"  Context Relevance: {cr}")
    rr = f"{agg['reasoning_recall']:.4f}" if agg["reasoning_recall"] is not None else "NA"
    print(f"  Reasoning Recall: {rr}")
    print("\n  --- Diagnosis Accuracy (MAIN METRIC, VERIFIED TRACK) ---")
    l3 = f"{agg['l3_top1_correct']:.4f}" if agg["l3_top1_correct"] is not None else "NA"
    top3 = f"{agg['top3_hit']:.4f}" if agg["top3_hit"] is not None else "NA"
    da = f"{agg['diagnosis_accuracy']:.4f}" if agg["diagnosis_accuracy"] is not None else "NA"
    dta = f"{agg['diagnosis_type_accuracy']:.4f}" if agg["diagnosis_type_accuracy"] is not None else "NA"
    dfa = f"{agg['diagnosis_family_accuracy']:.4f}" if agg["diagnosis_family_accuracy"] is not None else "NA"
    print(f"  L3 Top-1 Accuracy: {l3}")
    print(f"  Top-3 Hit: {top3}")
    print(f"  Diagnosis Accuracy: {da}")
    print(f"  Diagnosis Type Accuracy: {dta}")
    print(f"  Diagnosis Family Accuracy: {dfa}")
    print("\n  --- Diagnosis Accuracy (PSEUDOLABEL TRACK) ---")
    l3_p = f"{agg['l3_top1_correct_pseudolabel']:.4f}" if agg["l3_top1_correct_pseudolabel"] is not None else "NA"
    top3_p = f"{agg['top3_hit_pseudolabel']:.4f}" if agg["top3_hit_pseudolabel"] is not None else "NA"
    da_p = f"{agg['diagnosis_accuracy_pseudolabel']:.4f}" if agg["diagnosis_accuracy_pseudolabel"] is not None else "NA"
    dta_p = f"{agg['diagnosis_type_accuracy_pseudolabel']:.4f}" if agg["diagnosis_type_accuracy_pseudolabel"] is not None else "NA"
    print(f"  L3 Top-1 Accuracy (Pseudo): {l3_p}")
    print(f"  Top-3 Hit (Pseudo): {top3_p}")
    print(f"  Diagnosis Accuracy (Pseudo): {da_p}")
    print(f"  Diagnosis Type Accuracy (Pseudo): {dta_p}")


def _candidate_diagnosis_models(evaluator: RAGAsLibraryEvaluator) -> List[str]:
    judge_models: List[str] = []
    for candidate in [evaluator.model_name, JUDGE_MODEL, JUDGE_MODEL_FALLBACK]:
        if candidate and candidate not in judge_models:
            judge_models.append(candidate)
    return judge_models


def _submit_diagnosis_batch(
    evaluator: RAGAsLibraryEvaluator,
    requests: Sequence[Dict[str, Any]],
    phase: str,
) -> Dict[str, str]:
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=evaluator.api_key)
    last_error: Optional[Exception] = None
    for judge_model in _candidate_diagnosis_models(evaluator):
        try:
            inlined_requests = []
            for req in requests:
                prompt = evaluator._build_diagnosis_prompt(
                    prediction=req["prediction"],
                    ground_truth=req["ground_truth"],
                    clinical_context=req.get("clinical_context", ""),
                    query_images=req.get("query_images", []),
                )
                contents = evaluator._build_multimodal_contents(prompt, req.get("query_images", []))
                inlined_requests.append(
                    types.InlinedRequest(
                        model=judge_model,
                        contents=contents,
                        metadata={"qid": req["qid"]},
                        config=types.GenerateContentConfig(
                            temperature=0.0,
                            response_mime_type="application/json",
                        ),
                    )
                )
            display_name = f"diag-judge-batch-{PHASE_LABEL_TRACK[phase]}-{int(time.time())}"
            job = client.batches.create(
                model=judge_model,
                src=types.BatchJobSource(inlined_requests=inlined_requests),
                config=types.CreateBatchJobConfig(display_name=display_name),
            )
            print(f"Diagnosis judge batch started: {job.name} ({judge_model})")
            return {"job_id": job.name, "judge_model": judge_model}
        except Exception as exc:
            last_error = exc
            continue
    raise RuntimeError(f"All diagnosis judge batch model attempts failed during submit: {last_error}")


def _poll_diagnosis_batch(
    evaluator: RAGAsLibraryEvaluator,
    requests: Sequence[Dict[str, Any]],
    job_id: str,
    judge_model: Optional[str],
    poll_seconds: float,
    timeout_seconds: int,
) -> Dict[str, Dict[str, Any]]:
    from google import genai

    client = genai.Client(api_key=evaluator.api_key)
    print(f"  Polling diagnosis batch: {job_id} ({judge_model or 'unknown_model'})")
    deadline = time.time() + timeout_seconds
    while True:
        job = client.batches.get(name=job_id)
        state = str(getattr(job, "state", "UNKNOWN"))
        if "SUCCEEDED" in state:
            break
        if any(flag in state for flag in ["FAILED", "CANCELLED", "EXPIRED"]):
            raise RuntimeError(f"Diagnosis batch ended in error state: {state}")
        if time.time() > deadline:
            if any(flag in state for flag in ACTIVE_BATCH_STATE_FLAGS):
                raise ActiveDiagnosisBatch(job_id=job_id, state=state, timeout_seconds=timeout_seconds)
            raise TimeoutError(f"Diagnosis batch timeout after {timeout_seconds}s: {state}")
        time.sleep(poll_seconds)

    dest = getattr(job, "dest", None)
    inlined_responses = getattr(dest, "inlined_responses", None) if dest else None
    if not inlined_responses:
        raise RuntimeError("Diagnosis batch succeeded but returned no inlined responses")

    by_qid: Dict[str, Dict[str, Any]] = {}
    for response_item in inlined_responses:
        metadata = getattr(response_item, "metadata", None) or {}
        qid = str(metadata.get("qid", ""))
        error = getattr(response_item, "error", None)
        if error is not None:
            by_qid[qid] = {
                "diagnosis_score": 0.0,
                "diagnosis_type_score": 0.0,
                "reasoning": f"[Batch judge error: {error}]",
                "method": "llm_judge_batch_error",
                "gt_rank": None,
                "top3_hit": None,
                "l3_top1_correct": None,
                "fallback_level": "judge_error",
            }
            continue

        response_obj = getattr(response_item, "response", None)
        result_text = ""
        try:
            result_text = (response_obj.text or "").strip()
        except Exception:
            result_text = ""
        parsed = evaluator._parse_diagnosis_json(result_text)
        by_qid[qid] = {
            "diagnosis_score": float(parsed.get("diagnosis_score", 0.0)),
            "diagnosis_type_score": float(parsed.get("diagnosis_type_score", 0.0)),
            "reasoning": parsed.get("reasoning", "No reasoning provided"),
            "method": "llm_judge_batch",
            "gt_rank": parsed.get("gt_rank"),
            "top3_hit": parsed.get("top3_hit"),
            "l3_top1_correct": parsed.get("l3_top1_correct"),
            "fallback_level": parsed.get("fallback_level", ""),
        }

    for req in requests:
        qid = req["qid"]
        if qid not in by_qid:
            by_qid[qid] = {
                "diagnosis_score": 0.0,
                "diagnosis_type_score": 0.0,
                "reasoning": "[Batch judge missing response]",
                "method": "llm_judge_batch_missing",
                "gt_rank": None,
                "top3_hit": None,
                "l3_top1_correct": None,
                "fallback_level": "judge_missing",
            }
    return by_qid


def _create_phase_b_row(
    qid: str,
    answers_sha256: str,
    run_id: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "qid": qid,
        "diagnosis_accuracy": result.get("diagnosis_score"),
        "diagnosis_type_accuracy": result.get("diagnosis_type_score"),
        "gt_rank": result.get("gt_rank"),
        "top3_hit": result.get("top3_hit"),
        "l3_top1_correct": result.get("l3_top1_correct"),
        "fallback_level": result.get("fallback_level"),
        "rank_source": result.get("rank_source"),
        "judge_gt_rank": result.get("judge_gt_rank"),
        "parser_gt_rank": result.get("parser_gt_rank"),
        "judge_parser_disagreement": result.get("judge_parser_disagreement"),
        "diagnosis_reasoning": result.get("reasoning"),
        "diagnosis_method": result.get("method"),
    }
    return _annotate_phase_row(row, run_id, answers_sha256, PHASE_B, PHASE_LABEL_TRACK[PHASE_B])


def _create_phase_c_row(
    qid: str,
    answers_sha256: str,
    run_id: str,
    result: Dict[str, Any],
) -> Dict[str, Any]:
    row = {
        "qid": qid,
        "diagnosis_accuracy_pseudolabel": result.get("diagnosis_score"),
        "diagnosis_type_accuracy_pseudolabel": result.get("diagnosis_type_score"),
        "gt_rank_pseudolabel": result.get("gt_rank"),
        "top3_hit_pseudolabel": result.get("top3_hit"),
        "l3_top1_correct_pseudolabel": result.get("l3_top1_correct"),
        "fallback_level_pseudolabel": result.get("fallback_level"),
        "rank_source_pseudolabel": result.get("rank_source"),
        "judge_gt_rank_pseudolabel": result.get("judge_gt_rank"),
        "parser_gt_rank_pseudolabel": result.get("parser_gt_rank"),
        "judge_parser_disagreement_pseudolabel": result.get("judge_parser_disagreement"),
        "diagnosis_reasoning_pseudolabel": result.get("reasoning"),
        "diagnosis_method_pseudolabel": result.get("method"),
    }
    return _annotate_phase_row(row, run_id, answers_sha256, PHASE_C, PHASE_LABEL_TRACK[PHASE_C])


def _create_skipped_diagnosis_row(
    phase: str,
    qid: str,
    answers_sha256: str,
    run_id: str,
    reason: str,
) -> Dict[str, Any]:
    if phase == PHASE_B:
        row = {
            "qid": qid,
            "diagnosis_accuracy": None,
            "diagnosis_type_accuracy": None,
            "gt_rank": None,
            "top3_hit": None,
            "l3_top1_correct": None,
            "fallback_level": "skipped_missing_ground_truth",
            "rank_source": "skipped",
            "judge_gt_rank": None,
            "parser_gt_rank": None,
            "judge_parser_disagreement": None,
            "diagnosis_reasoning": reason,
            "diagnosis_method": "skipped_no_ground_truth",
        }
    else:
        row = {
            "qid": qid,
            "diagnosis_accuracy_pseudolabel": None,
            "diagnosis_type_accuracy_pseudolabel": None,
            "gt_rank_pseudolabel": None,
            "top3_hit_pseudolabel": None,
            "l3_top1_correct_pseudolabel": None,
            "fallback_level_pseudolabel": "skipped_missing_ground_truth",
            "rank_source_pseudolabel": "skipped",
            "judge_gt_rank_pseudolabel": None,
            "parser_gt_rank_pseudolabel": None,
            "judge_parser_disagreement_pseudolabel": None,
            "diagnosis_reasoning_pseudolabel": reason,
            "diagnosis_method_pseudolabel": "skipped_no_ground_truth",
        }
    return _annotate_phase_row(row, run_id, answers_sha256, phase, PHASE_LABEL_TRACK[phase])


def _run_phase_a_metrics(
    evaluator: RAGAsLibraryEvaluator,
    run_dir: Path,
    ordered_qids: Sequence[str],
    samples_by_qid: Dict[str, Dict[str, Any]],
    manifest_path: Path,
    manifest: Dict[str, Any],
    answers_sha256: str,
    delay_seconds: float,
    evaluate_retrieval_metrics: bool,
    resume: bool,
) -> Dict[str, Dict[str, Any]]:
    phase_path = _phase_output_path(run_dir, PHASE_A)
    _, existing_rows = _load_unique_jsonl(phase_path, PHASE_A) if (resume and phase_path.exists()) else ([], {})
    phase_rows = dict(existing_rows)

    async def _evaluate() -> None:
        completed_before = len(phase_rows)
        if completed_before:
            print(f"  Phase A resume: skipping {completed_before} completed qids")
        for idx, qid in enumerate(ordered_qids, start=1):
            if qid in phase_rows:
                continue
            sample = samples_by_qid[qid]
            contexts = [context.get("text", "") for context in sample.get("contexts", [])]
            result = await evaluator.evaluate_sample(
                qid=qid,
                query=sample["query"],
                answer=sample["answer"],
                contexts=contexts,
                query_images=sample.get("query_images", []),
                context_images=sample.get("context_images", []),
                ground_truth=None,
                ground_truth_pseudolabel=None,
                evaluate_retrieval_metrics=evaluate_retrieval_metrics,
                generation_mode=sample.get("generation_mode", ""),
                retrieval_support_status=sample.get("retrieval_support_status", ""),
            )
            row = _sanitize_record_for_json(asdict(result))
            row["run_id"] = run_dir.name
            row["ground_truth_bucket"] = _bucket_from_ground_truth(sample.get("ground_truth"))
            traces = row.setdefault("traces", {})
            traces["ground_truth_type"] = (sample.get("ground_truth") or {}).get("diagnosis_type", "")
            traces["ground_truth_bucket"] = row["ground_truth_bucket"]
            row = _annotate_phase_row(row, run_dir.name, answers_sha256, PHASE_A, PHASE_LABEL_TRACK[PHASE_A])
            phase_rows[qid] = row
            _write_jsonl(phase_path, ordered_qids, phase_rows)
            manifest["phases"][PHASE_A]["status"] = "running"
            manifest["phases"][PHASE_A]["n_completed"] = len(phase_rows)
            manifest["phases"][PHASE_A]["updated_at"] = _now_ts()
            _save_manifest(manifest_path, manifest)
            print(
                f"  [Phase A {idx}/{len(ordered_qids)}] {qid}: "
                f"f={row.get('multimodal_faithfulness')}, "
                f"r={row.get('multimodal_relevance')}, "
                f"c={row.get('context_relevance')}"
            )
            if delay_seconds > 0 and idx < len(ordered_qids):
                await asyncio.sleep(delay_seconds)

    asyncio.run(_evaluate())
    manifest["phases"][PHASE_A]["status"] = "completed"
    manifest["phases"][PHASE_A]["n_completed"] = len(phase_rows)
    manifest["phases"][PHASE_A]["updated_at"] = _now_ts()
    _save_manifest(manifest_path, manifest)
    return phase_rows


def _run_diagnosis_phase(
    phase: str,
    evaluator: RAGAsLibraryEvaluator,
    run_dir: Path,
    ordered_qids: Sequence[str],
    samples_by_qid: Dict[str, Dict[str, Any]],
    manifest_path: Path,
    manifest: Dict[str, Any],
    answers_sha256: str,
    poll_seconds: float,
    timeout_seconds: int,
    diagnosis_batch_api: bool,
    resume: bool,
) -> Dict[str, Dict[str, Any]]:
    phase_path = _phase_output_path(run_dir, phase)
    _, existing_rows = _load_unique_jsonl(phase_path, phase) if (resume and phase_path.exists()) else ([], {})
    phase_rows = dict(existing_rows)
    phase_state = manifest["phases"][phase]
    phase_state["status"] = "running"
    _save_manifest(manifest_path, manifest)

    if phase == PHASE_B:
        gt_key = "ground_truth"
        row_builder = _create_phase_b_row
    else:
        gt_key = "ground_truth_pseudolabel"
        row_builder = _create_phase_c_row

    pending_requests: List[Dict[str, Any]] = []
    for qid in ordered_qids:
        if qid in phase_rows:
            continue
        sample = samples_by_qid[qid]
        ground_truth = sample.get(gt_key)
        if not ground_truth:
            phase_rows[qid] = _create_skipped_diagnosis_row(
                phase=phase,
                qid=qid,
                answers_sha256=answers_sha256,
                run_id=run_dir.name,
                reason="Missing ground truth for this track",
            )
            continue
        parser_contract = evaluate_ranked_diagnosis_contract(
            answer=sample.get("answer", ""),
            ground_truth=ground_truth,
            max_rank=3,
        )
        pending_requests.append(
            {
                "qid": qid,
                "prediction": sample["answer"],
                "ground_truth": ground_truth,
                "clinical_context": sample.get("query", ""),
                "query_images": sample.get("query_images", []),
                "parser_contract": parser_contract,
            }
        )

    results_by_qid: Dict[str, Dict[str, Any]] = {}

    def _mark_active_batch_waiting(exc: ActiveDiagnosisBatch) -> None:
        phase_state["status"] = "running"
        phase_state["batch_job_id"] = exc.job_id
        phase_state["last_observed_batch_state"] = exc.state
        phase_state["updated_at"] = _now_ts()
        _save_manifest(manifest_path, manifest)

    if pending_requests:
        if diagnosis_batch_api:
            batch_job_id = phase_state.get("batch_job_id")
            batch_judge_model = phase_state.get("batch_judge_model")
            if not (resume and batch_job_id):
                submitted = _submit_diagnosis_batch(evaluator, pending_requests, phase)
                batch_job_id = submitted["job_id"]
                batch_judge_model = submitted["judge_model"]
                phase_state["batch_job_id"] = batch_job_id
                phase_state["batch_judge_model"] = batch_judge_model
                phase_state["updated_at"] = _now_ts()
                _save_manifest(manifest_path, manifest)
            try:
                results_by_qid = _poll_diagnosis_batch(
                    evaluator=evaluator,
                    requests=pending_requests,
                    job_id=batch_job_id,
                    judge_model=batch_judge_model,
                    poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds,
                )
            except ActiveDiagnosisBatch as exc:
                _mark_active_batch_waiting(exc)
                raise ActiveDiagnosisBatch(
                    job_id=exc.job_id,
                    state=exc.state,
                    timeout_seconds=exc.timeout_seconds,
                    phase=phase,
                ) from exc
            except Exception as exc:
                print(f"  Existing diagnosis batch unusable for {phase}, resubmitting: {exc}")
                submitted = _submit_diagnosis_batch(evaluator, pending_requests, phase)
                phase_state["batch_job_id"] = submitted["job_id"]
                phase_state["batch_judge_model"] = submitted["judge_model"]
                phase_state["updated_at"] = _now_ts()
                _save_manifest(manifest_path, manifest)
                try:
                    results_by_qid = _poll_diagnosis_batch(
                        evaluator=evaluator,
                        requests=pending_requests,
                        job_id=submitted["job_id"],
                        judge_model=submitted["judge_model"],
                        poll_seconds=poll_seconds,
                        timeout_seconds=timeout_seconds,
                    )
                except ActiveDiagnosisBatch as active_exc:
                    _mark_active_batch_waiting(active_exc)
                    raise ActiveDiagnosisBatch(
                        job_id=active_exc.job_id,
                        state=active_exc.state,
                        timeout_seconds=active_exc.timeout_seconds,
                        phase=phase,
                    ) from active_exc
        else:
            async def _run_sequential() -> Dict[str, Dict[str, Any]]:
                out: Dict[str, Dict[str, Any]] = {}
                for req in pending_requests:
                    diagnosis_result = await evaluator.evaluate_diagnosis_equivalence(
                        prediction=req["prediction"],
                        ground_truth=req["ground_truth"],
                        clinical_context=req.get("clinical_context", ""),
                        query_images=req.get("query_images", []),
                        use_llm_judge=True,
                    )
                    out[req["qid"]] = {
                        "diagnosis_score": diagnosis_result.diagnosis_score,
                        "diagnosis_type_score": diagnosis_result.diagnosis_type_score,
                        "reasoning": diagnosis_result.reasoning,
                        "method": diagnosis_result.method,
                        "gt_rank": diagnosis_result.gt_rank,
                        "top3_hit": diagnosis_result.top3_hit,
                        "l3_top1_correct": diagnosis_result.l3_top1_correct,
                        "fallback_level": diagnosis_result.fallback_level,
                    }
                return out
            results_by_qid = asyncio.run(_run_sequential())

    for req in pending_requests:
        qid = req["qid"]
        judge_result = results_by_qid.get(qid, {
            "diagnosis_score": 0.0,
            "diagnosis_type_score": 0.0,
            "reasoning": "[Missing diagnosis result]",
            "method": "missing_result",
            "gt_rank": None,
            "top3_hit": None,
            "l3_top1_correct": None,
            "fallback_level": "missing_result",
        })
        rank_contract = _resolve_rank_contract_result(
            judge_result=judge_result,
            parser_contract=req.get("parser_contract", {}),
        )
        merged_result = dict(judge_result)
        merged_result.update(rank_contract)
        results_by_qid[qid] = merged_result

    for qid, result in results_by_qid.items():
        phase_rows[qid] = row_builder(qid=qid, answers_sha256=answers_sha256, run_id=run_dir.name, result=result)

    _write_jsonl(phase_path, ordered_qids, phase_rows)
    phase_state["status"] = "completed"
    phase_state["n_completed"] = len(phase_rows)
    phase_state["last_observed_batch_state"] = None
    phase_state["updated_at"] = _now_ts()
    _save_manifest(manifest_path, manifest)
    return phase_rows


def _merge_phase_outputs(
    run_dir: Path,
    ordered_qids: Sequence[str],
    samples_by_qid: Dict[str, Dict[str, Any]],
    manifest_path: Path,
    manifest: Dict[str, Any],
) -> Path:
    rows_by_phase: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for phase in [PHASE_A, PHASE_B, PHASE_C]:
        _, rows_by_qid = _load_unique_jsonl(_phase_output_path(run_dir, phase), phase)
        missing = [qid for qid in ordered_qids if qid not in rows_by_qid]
        if missing:
            raise RuntimeError(f"Cannot merge: {phase} is missing qids, e.g. {missing[:5]}")
        rows_by_phase[phase] = rows_by_qid

    final_rows: List[Dict[str, Any]] = []
    for qid in ordered_qids:
        sample = samples_by_qid[qid]
        merged = dict(rows_by_phase[PHASE_A][qid])
        merged.pop("phase", None)
        merged.pop("label_track", None)
        merged.pop("eval_scope_id", None)
        merged["run_id"] = run_dir.name

        for source_row in [rows_by_phase[PHASE_B][qid], rows_by_phase[PHASE_C][qid]]:
            for key, value in source_row.items():
                if key in {"qid", "phase", "label_track", "eval_scope_id"}:
                    continue
                merged[key] = value

        family_details = compute_family_metric_details(
            sample.get("answer", ""),
            sample.get("ground_truth"),
        )
        prompt_context_doc_ids = sample.get("prompt_context_doc_ids") or [
            context.get("doc_id")
            for context in sample.get("contexts", [])
            if context.get("doc_id")
        ]
        merged["diagnosis_family"] = family_details["diagnosis_family"]
        merged["diagnosis_family_accuracy"] = family_details["diagnosis_family_accuracy"]
        merged["diagnosis_family_source"] = family_details["diagnosis_family_source"]
        merged["answer_format_valid"] = sample.get("answer_format_valid", family_details["answer_format_valid"])
        merged["answer_format_error"] = sample.get("answer_format_error") or family_details["answer_format_error"]
        merged["prompt_context_doc_ids"] = prompt_context_doc_ids
        merged["prompt_context_count"] = sample.get("prompt_context_count", len(prompt_context_doc_ids))
        merged["format_retry_count"] = sample.get("format_retry_count", 0)
        merged["ground_truth_bucket"] = _bucket_from_ground_truth(sample.get("ground_truth"))
        traces = merged.setdefault("traces", {})
        traces["ground_truth_type"] = (sample.get("ground_truth") or {}).get("diagnosis_type", "")
        traces["ground_truth_bucket"] = merged["ground_truth_bucket"]
        traces["rank1_diagnosis_text"] = family_details.get("rank1_diagnosis_text")
        traces["rank1_diagnosis_type"] = family_details.get("rank1_diagnosis_type")
        traces["gt_rank"] = merged.get("gt_rank")
        traces["top3_hit"] = merged.get("top3_hit")
        traces["l3_top1_correct"] = merged.get("l3_top1_correct")
        traces["fallback_level"] = merged.get("fallback_level")
        traces["rank_source"] = merged.get("rank_source")
        traces["judge_parser_disagreement"] = merged.get("judge_parser_disagreement")
        traces["gt_rank_pseudolabel"] = merged.get("gt_rank_pseudolabel")
        traces["top3_hit_pseudolabel"] = merged.get("top3_hit_pseudolabel")
        traces["l3_top1_correct_pseudolabel"] = merged.get("l3_top1_correct_pseudolabel")
        traces["fallback_level_pseudolabel"] = merged.get("fallback_level_pseudolabel")
        traces["rank_source_pseudolabel"] = merged.get("rank_source_pseudolabel")
        traces["judge_parser_disagreement_pseudolabel"] = merged.get("judge_parser_disagreement_pseudolabel")
        final_rows.append(_sanitize_record_for_json(merged))

    output_path = run_dir / FINAL_RAGAS_FILE
    with open(output_path, "w", encoding="utf-8") as f:
        for row in final_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest["phases"][MERGE_PHASE]["status"] = "completed"
    manifest["phases"][MERGE_PHASE]["n_completed"] = len(final_rows)
    manifest["phases"][MERGE_PHASE]["updated_at"] = _now_ts()
    _save_manifest(manifest_path, manifest)
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        update_summary_with_ragas_metrics(run_dir)
    _print_aggregates(output_path, final_rows)
    return output_path


def run_ragas_evaluation_phased(
    run_dir: Path,
    answers_file: str = "answers.jsonl",
    judge_model: Optional[str] = None,
    max_samples: Optional[int] = None,
    delay_seconds: float = 1.0,
    resume: bool = True,
    diagnosis_batch_api: bool = False,
    diagnosis_batch_poll_seconds: float = 10.0,
    diagnosis_batch_timeout_seconds: int = 7200,
    evaluate_retrieval_metrics: bool = True,
    start_phase: str = "all",
) -> Path:
    start_phase_name = _normalize_start_phase(start_phase)
    lock_fh = _acquire_manifest_lock(run_dir)
    try:
        ordered_qids, raw_samples_by_qid, answers_sha256, selected_qids_sha256 = _load_answers(
            run_dir=run_dir,
            answers_file=answers_file,
            max_samples=max_samples,
        )
        if not ordered_qids:
            raise RuntimeError(f"No answer rows found in {run_dir / answers_file}")

        queries_gt_map, queries_gt_pseudo_map = _load_queries_ground_truth_maps(run_dir)
        samples_by_qid = {
            qid: _backfill_ground_truth(raw_samples_by_qid[qid], qid, queries_gt_map, queries_gt_pseudo_map)
            for qid in ordered_qids
        }

        manifest_path, manifest = _prepare_manifest(
            run_dir=run_dir,
            answers_file=answers_file,
            answers_sha256=answers_sha256,
            selected_qids_sha256=selected_qids_sha256,
            ordered_qids=ordered_qids,
            judge_model=judge_model,
            resume=resume,
        )
        _ensure_phase_prerequisites(manifest, start_phase_name)

        evaluator = RAGAsLibraryEvaluator(model=judge_model)
        effective_judge = judge_model or JUDGE_MODEL
        print(f"RAGAS judge model: {effective_judge} (fallback: {JUDGE_MODEL_FALLBACK})")

        start_idx = _phase_index(start_phase_name)
        for phase in PHASE_SEQUENCE[start_idx:]:
            if phase == PHASE_A:
                if manifest["phases"][PHASE_A].get("status") != "completed":
                    _run_phase_a_metrics(
                        evaluator=evaluator,
                        run_dir=run_dir,
                        ordered_qids=ordered_qids,
                        samples_by_qid=samples_by_qid,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        answers_sha256=answers_sha256,
                        delay_seconds=delay_seconds,
                        evaluate_retrieval_metrics=evaluate_retrieval_metrics,
                        resume=resume,
                    )
            elif phase == PHASE_B:
                if manifest["phases"][PHASE_B].get("status") != "completed":
                    if diagnosis_batch_api:
                        print("\n  Running diagnosis judge via Batch API for verified-label samples...")
                    else:
                        print("\n  Running diagnosis judge synchronously for verified-label samples...")
                    _run_diagnosis_phase(
                        phase=PHASE_B,
                        evaluator=evaluator,
                        run_dir=run_dir,
                        ordered_qids=ordered_qids,
                        samples_by_qid=samples_by_qid,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        answers_sha256=answers_sha256,
                        poll_seconds=diagnosis_batch_poll_seconds,
                        timeout_seconds=diagnosis_batch_timeout_seconds,
                        diagnosis_batch_api=diagnosis_batch_api,
                        resume=resume,
                    )
            elif phase == PHASE_C:
                _validate_phase_output(
                    run_dir=run_dir,
                    phase=PHASE_B,
                    ordered_qids=ordered_qids,
                    answers_sha256=answers_sha256,
                    run_id=run_dir.name,
                )
                if manifest["phases"][PHASE_C].get("status") != "completed":
                    if diagnosis_batch_api:
                        print("\n  Running diagnosis judge via Batch API for pseudolabel samples...")
                    else:
                        print("\n  Running diagnosis judge synchronously for pseudolabel samples...")
                    _run_diagnosis_phase(
                        phase=PHASE_C,
                        evaluator=evaluator,
                        run_dir=run_dir,
                        ordered_qids=ordered_qids,
                        samples_by_qid=samples_by_qid,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        answers_sha256=answers_sha256,
                        poll_seconds=diagnosis_batch_poll_seconds,
                        timeout_seconds=diagnosis_batch_timeout_seconds,
                        diagnosis_batch_api=diagnosis_batch_api,
                        resume=resume,
                    )
            elif phase == MERGE_PHASE:
                _validate_phase_output(
                    run_dir=run_dir,
                    phase=PHASE_B,
                    ordered_qids=ordered_qids,
                    answers_sha256=answers_sha256,
                    run_id=run_dir.name,
                )
                _validate_phase_output(
                    run_dir=run_dir,
                    phase=PHASE_C,
                    ordered_qids=ordered_qids,
                    answers_sha256=answers_sha256,
                    run_id=run_dir.name,
                )
                return _merge_phase_outputs(
                    run_dir=run_dir,
                    ordered_qids=ordered_qids,
                    samples_by_qid=samples_by_qid,
                    manifest_path=manifest_path,
                    manifest=manifest,
                )

        return run_dir / FINAL_RAGAS_FILE
    except ActiveDiagnosisBatch as exc:
        phase_label = exc.phase or "unknown_phase"
        print(
            f"\nRemote Gemini batch is still active for {phase_label}: {exc.job_id} "
            f"({exc.state}). Re-run the same command later to reattach without resubmitting."
        )
        return _phase_output_path(run_dir, phase_label) if exc.phase else (run_dir / MANIFEST_FILE)
    finally:
        _release_manifest_lock(lock_fh)


def parse_args() -> Any:
    import argparse

    parser = argparse.ArgumentParser(description="Run phased, resume-safe RAGAS evaluation")
    parser.add_argument("--run-dir", required=True, help="Run directory containing answers and retrieval artifacts")
    parser.add_argument("--answers-file", default="answers.jsonl", help="Answers JSONL filename inside run-dir")
    parser.add_argument("--judge-model", default=None, help="Optional judge model override")
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap on samples for debugging")
    parser.add_argument("--delay-seconds", type=float, default=1.0, help="Delay between Phase A samples")
    parser.add_argument("--no-resume", action="store_true", help="Ignore existing phase artifacts and start fresh")
    parser.add_argument("--start-phase", default="all", choices=sorted(START_PHASE_ALIASES), help="Phase to start or resume from")
    parser.add_argument("--repair-merge", action="store_true", help="Recompute final ragas.jsonl from saved phase artifacts without rerunning earlier phases")
    parser.add_argument("--judge-batch-api", action="store_true", help="Use Gemini Batch API for diagnosis phases")
    parser.add_argument("--judge-batch-poll-seconds", type=float, default=10.0, help="Polling interval for diagnosis batches")
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=7200, help="Timeout for diagnosis batches")
    parser.add_argument("--diagnosis-only-eval", action="store_true", help="Disable retrieval-grounded metrics in Phase A")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    start_phase = "merge" if args.repair_merge else args.start_phase
    resume = True if args.repair_merge else not args.no_resume
    run_ragas_evaluation_phased(
        run_dir=Path(args.run_dir),
        answers_file=args.answers_file,
        judge_model=args.judge_model,
        max_samples=args.max_samples,
        delay_seconds=args.delay_seconds,
        resume=resume,
        diagnosis_batch_api=args.judge_batch_api,
        diagnosis_batch_poll_seconds=args.judge_batch_poll_seconds,
        diagnosis_batch_timeout_seconds=args.judge_batch_timeout_seconds,
        evaluate_retrieval_metrics=not args.diagnosis_only_eval,
        start_phase=start_phase,
    )
