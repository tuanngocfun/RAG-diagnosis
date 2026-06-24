#!/usr/bin/env python3
"""Artifact-driven Phase1a v2 pre-gate and diagnostic analyzer."""
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import RUNS_DIR, TRAIN_JSONL
from .diagnosis_output_parser import analyze_answer_format


def _resolve_run_dir(run_id_or_path: str) -> Path:
    candidate = Path(run_id_or_path).expanduser()
    if candidate.exists():
        return candidate.resolve()
    return (RUNS_DIR / run_id_or_path).resolve()


def _load_json(path: Path) -> Dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_jsonl(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _numeric_mean(values: Iterable[object]) -> Optional[float]:
    clean: List[float] = []
    for value in values:
        if isinstance(value, bool):
            clean.append(float(value))
        elif isinstance(value, (int, float)):
            clean.append(float(value))
    if not clean:
        return None
    return sum(clean) / len(clean)


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _normalize_diagnosis_type(value: object) -> str:
    return _normalize_text(value).upper()


def _bucket_from_ground_truth(ground_truth: object) -> Optional[str]:
    if not isinstance(ground_truth, dict):
        return None
    diagnosis_type = _normalize_diagnosis_type(ground_truth.get("diagnosis_type"))
    diagnosis = _normalize_text(ground_truth.get("diagnosis")).lower()
    if diagnosis_type == "NON-LEISHMANIASIS" or "non-leish" in diagnosis:
        return "nonleish"
    if diagnosis_type or diagnosis:
        return "leish"
    return None


def _bucket_for_row(row: Dict) -> str:
    bucket = row.get("ground_truth_bucket")
    if bucket:
        return str(bucket)
    traces = row.get("traces") or {}
    traced_bucket = traces.get("ground_truth_bucket")
    if traced_bucket:
        return str(traced_bucket)
    inferred = _bucket_from_ground_truth(row.get("ground_truth")) or _bucket_from_ground_truth(
        row.get("ground_truth_pseudolabel")
    )
    return inferred or "unknown"


def _find_answer_file(run_dir: Path) -> Optional[Path]:
    candidates = sorted(run_dir.glob("answers*.jsonl"))
    if not candidates:
        return None
    preferred = sorted(
        candidates,
        key=lambda path: (
            "norag" not in path.name.lower(),
            "rag" not in path.name.lower(),
            path.name,
        ),
    )
    return preferred[0]


def _load_retrieval_rows(run_dir: Path) -> Dict[str, Dict]:
    rows = {}
    for row in _load_jsonl(run_dir / "retrieval.jsonl"):
        qid = row.get("qid")
        if qid:
            rows[str(qid)] = row
    return rows


def _load_answer_rows(run_dir: Path) -> Dict[str, Dict]:
    rows: Dict[str, Dict] = {}

    answer_file = _find_answer_file(run_dir)
    if answer_file and answer_file.exists():
        for row in _load_jsonl(answer_file):
            qid = row.get("qid")
            if not qid:
                continue
            analysis = analyze_answer_format(row.get("answer", ""))
            merged = dict(row)
            merged.setdefault("predicted_rank1_diagnosis_text", analysis.rank1_diagnosis_text)
            merged.setdefault("predicted_rank1_diagnosis_type", analysis.rank1_diagnosis_type)
            rows[str(qid)] = merged

    ragas_path = run_dir / "ragas.jsonl"
    if ragas_path.exists():
        for row in _load_jsonl(ragas_path):
            qid = row.get("qid")
            if not qid:
                continue
            merged = dict(rows.get(str(qid), {}))
            merged.update(row)
            rows[str(qid)] = merged

    for qid, row in rows.items():
        prompt_context_doc_ids = row.get("prompt_context_doc_ids")
        if not prompt_context_doc_ids:
            prompt_context_doc_ids = [
                ctx.get("doc_id")
                for ctx in (row.get("contexts") or [])
                if ctx.get("doc_id")
            ]
        row["prompt_context_doc_ids"] = [doc_id for doc_id in (prompt_context_doc_ids or []) if doc_id]
        row["ground_truth_bucket"] = row.get("ground_truth_bucket") or _bucket_for_row(row)
        rows[qid] = row

    return rows


def _load_run_snapshot(run_dir_or_id: str) -> Dict[str, object]:
    run_dir = _resolve_run_dir(run_dir_or_id)
    summary = _load_json(run_dir / "summary.json")
    return {
        "run_dir": run_dir,
        "run_config": _load_json(run_dir / "run_config.json"),
        "summary": summary,
        "retrieval_rows": _load_retrieval_rows(run_dir),
        "answer_rows": _load_answer_rows(run_dir),
    }


def _extract_doc_id(item: Dict) -> Optional[str]:
    doc_id = item.get("doc_id") or item.get("case_id")
    if not doc_id:
        return None
    return str(doc_id)


def _build_usage_summary(run_snapshot: Dict[str, object], augmented_case_ids: set[str]) -> Dict[str, object]:
    retrieval_rows: Dict[str, Dict] = run_snapshot["retrieval_rows"]  # type: ignore[assignment]
    answer_rows: Dict[str, Dict] = run_snapshot["answer_rows"]  # type: ignore[assignment]
    retrieval_hits: List[Dict[str, object]] = []
    prompt_hits: List[Dict[str, object]] = []
    distinct_retrieval_docs = set()
    distinct_prompt_docs = set()
    nonleish_prompt_qids = []

    for qid, row in retrieval_rows.items():
        for rank, ctx in enumerate(row.get("contexts") or [], start=1):
            doc_id = _extract_doc_id(ctx)
            if doc_id not in augmented_case_ids:
                continue
            retrieval_hits.append(
                {
                    "qid": qid,
                    "doc_id": doc_id,
                    "rank": rank,
                    "score": ctx.get("score"),
                }
            )
            distinct_retrieval_docs.add(doc_id)

    for qid, row in answer_rows.items():
        matched = [doc_id for doc_id in row.get("prompt_context_doc_ids", []) if doc_id in augmented_case_ids]
        if not matched:
            continue
        prompt_hits.append(
            {
                "qid": qid,
                "doc_ids": matched,
                "ground_truth_bucket": row.get("ground_truth_bucket"),
            }
        )
        distinct_prompt_docs.update(matched)
        if row.get("ground_truth_bucket") == "nonleish":
            nonleish_prompt_qids.append(qid)

    summary = run_snapshot["summary"]  # type: ignore[assignment]
    return {
        "run_dir": str(run_snapshot["run_dir"]),
        "retrieval_metrics": {
            "ndcg@10": summary.get("metrics", {}).get("ndcg", {}).get("@10"),
            "mrr": summary.get("metrics", {}).get("mrr"),
            "map": summary.get("metrics", {}).get("map"),
            "precision@5": summary.get("metrics", {}).get("precision", {}).get("@5"),
            "recall@5": summary.get("metrics", {}).get("recall", {}).get("@5"),
        },
        "augmented_retrieval_hit_count": len(retrieval_hits),
        "distinct_augmented_docs_retrieved": len(distinct_retrieval_docs),
        "augmented_prompt_context_hit_count": len(prompt_hits),
        "distinct_augmented_docs_in_prompt": len(distinct_prompt_docs),
        "nonleish_prompt_context_hit_count": len(nonleish_prompt_qids),
        "retrieval_hit_examples": retrieval_hits[:10],
        "prompt_hit_examples": prompt_hits[:10],
    }


def _load_corpus_index(train_jsonl: Path) -> Dict[str, Dict[str, object]]:
    rows: Dict[str, Dict[str, object]] = {}
    for row in _load_jsonl(train_jsonl):
        case_id = _normalize_text(row.get("case_id"))
        if not case_id:
            continue
        rows[case_id] = {
            "diagnosis": row.get("diagnosis"),
            "diagnosis_type": row.get("diagnosis_type"),
            "is_leishmaniasis": row.get("is_leishmaniasis"),
        }
    return rows


def _train_jsonl_from_snapshot(run_snapshot: Dict[str, object]) -> Path:
    run_config = run_snapshot.get("run_config") or {}
    runtime_metadata = run_config.get("runtime_metadata") if isinstance(run_config, dict) else {}
    candidate = None
    if isinstance(runtime_metadata, dict):
        candidate = runtime_metadata.get("train_jsonl")
    if candidate:
        return Path(str(candidate)).expanduser()
    return TRAIN_JSONL


def _prompt_context_profile(row: Dict, corpus_index: Dict[str, Dict[str, object]]) -> Dict[str, object]:
    doc_ids = [str(doc_id) for doc_id in (row.get("prompt_context_doc_ids") or []) if doc_id]
    leish_count = 0
    nonleish_count = 0
    unknown_count = 0

    for doc_id in doc_ids:
        meta = corpus_index.get(doc_id)
        if not meta:
            unknown_count += 1
            continue
        if meta.get("is_leishmaniasis") is True:
            leish_count += 1
        elif meta.get("is_leishmaniasis") is False:
            nonleish_count += 1
        else:
            unknown_count += 1

    if not doc_ids:
        dominance = "empty"
    elif leish_count > nonleish_count and leish_count > 0:
        dominance = "leish_dominant"
    elif nonleish_count > leish_count and nonleish_count > 0:
        dominance = "nonleish_dominant"
    elif leish_count and nonleish_count:
        dominance = "mixed"
    else:
        dominance = "unknown"

    return {
        "prompt_context_doc_ids": doc_ids,
        "prompt_context_count": len(doc_ids),
        "leish_count": leish_count,
        "nonleish_count": nonleish_count,
        "unknown_count": unknown_count,
        "dominance": dominance,
    }


def _mode_bucket_counts(answer_rows: Dict[str, Dict]) -> Dict[str, Dict[str, int]]:
    buckets: Dict[str, Counter] = {
        "overall": Counter(),
        "leish": Counter(),
        "nonleish": Counter(),
    }
    for row in answer_rows.values():
        mode = _normalize_text(row.get("generation_mode")) or "unknown"
        bucket = _normalize_text(row.get("ground_truth_bucket")) or "unknown"
        buckets["overall"][mode] += 1
        if bucket in {"leish", "nonleish"}:
            buckets[bucket][mode] += 1
    return {label: dict(counter) for label, counter in buckets.items()}


def _gating_info_counts(answer_rows: Dict[str, Dict]) -> Dict[str, int]:
    counter = Counter()
    for row in answer_rows.values():
        key = _normalize_text(row.get("gating_info")) or "<empty>"
        counter[key] += 1
    return dict(counter)


def _threshold_summary(answer_rows: Dict[str, Dict]) -> Dict[str, object]:
    counter = Counter()
    numeric: List[float] = []
    for row in answer_rows.values():
        value = row.get("threshold_used")
        if isinstance(value, bool):
            numeric.append(float(value))
            counter[f"{float(value):.4f}"] += 1
        elif isinstance(value, (int, float)):
            numeric.append(float(value))
            counter[f"{float(value):.4f}"] += 1
    return {
        "counts": dict(counter),
        "min": min(numeric) if numeric else None,
        "max": max(numeric) if numeric else None,
        "mean": _numeric_mean(numeric),
    }


def _metric_value(row: Dict, metric_name: str) -> Optional[float]:
    value = row.get(metric_name)
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _build_pure_rag_loser_sets(
    pure_rag_snapshot: Dict[str, object],
    norag_snapshot: Dict[str, object],
) -> Dict[str, List[str]]:
    pure_rows: Dict[str, Dict] = pure_rag_snapshot["answer_rows"]  # type: ignore[assignment]
    norag_rows: Dict[str, Dict] = norag_snapshot["answer_rows"]  # type: ignore[assignment]
    common_qids = sorted(set(pure_rows) & set(norag_rows))

    diagnosis_losers: List[str] = []
    diagnosis_type_losers: List[str] = []
    diagnosis_type_only_losers: List[str] = []
    nonleish_losers: List[str] = []
    nonleish_diagnosis_flip_qids: List[str] = []

    for qid in common_qids:
        pure_row = pure_rows[qid]
        norag_row = norag_rows[qid]
        diag_delta = None
        type_delta = None
        pure_diag = _metric_value(pure_row, "diagnosis_accuracy")
        norag_diag = _metric_value(norag_row, "diagnosis_accuracy")
        pure_type = _metric_value(pure_row, "diagnosis_type_accuracy")
        norag_type = _metric_value(norag_row, "diagnosis_type_accuracy")
        if pure_diag is not None and norag_diag is not None:
            diag_delta = pure_diag - norag_diag
        if pure_type is not None and norag_type is not None:
            type_delta = pure_type - norag_type

        bucket = _normalize_text(pure_row.get("ground_truth_bucket")) or _normalize_text(norag_row.get("ground_truth_bucket"))

        if diag_delta is not None and diag_delta < 0:
            diagnosis_losers.append(qid)
            if bucket == "nonleish":
                nonleish_diagnosis_flip_qids.append(qid)
        if type_delta is not None and type_delta < 0:
            diagnosis_type_losers.append(qid)
            if not (diag_delta is not None and diag_delta < 0):
                diagnosis_type_only_losers.append(qid)
        if bucket == "nonleish" and (
            (diag_delta is not None and diag_delta < 0) or (type_delta is not None and type_delta < 0)
        ):
            nonleish_losers.append(qid)

    return {
        "diagnosis_loser_qids": diagnosis_losers,
        "diagnosis_type_loser_qids": diagnosis_type_losers,
        "diagnosis_type_only_loser_qids": diagnosis_type_only_losers,
        "nonleish_loser_qids": nonleish_losers,
        "nonleish_diagnosis_flip_qids": nonleish_diagnosis_flip_qids,
    }


def _build_loser_delta_rows(
    qids: List[str],
    pure_rag_snapshot: Dict[str, object],
    revised_snapshot: Dict[str, object],
    corpus_index: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    pure_rows: Dict[str, Dict] = pure_rag_snapshot["answer_rows"]  # type: ignore[assignment]
    revised_rows: Dict[str, Dict] = revised_snapshot["answer_rows"]  # type: ignore[assignment]
    rows: List[Dict[str, object]] = []

    switched_from_rag_to_norag = 0
    generation_mode_changed = 0
    pure_leish_dominant = 0
    revised_leish_dominant = 0
    reduced_or_empty = 0
    mixed_or_nonleish = 0

    for qid in qids:
        pure_row = pure_rows.get(qid, {})
        revised_row = revised_rows.get(qid, {})
        pure_profile = _prompt_context_profile(pure_row, corpus_index)
        revised_profile = _prompt_context_profile(revised_row, corpus_index)

        pure_mode = _normalize_text(pure_row.get("generation_mode")) or "unknown"
        revised_mode = _normalize_text(revised_row.get("generation_mode")) or "unknown"
        mode_changed = pure_mode != revised_mode
        switched = pure_mode == "rag_prompt" and revised_mode == "norag_prompt"

        if mode_changed:
            generation_mode_changed += 1
        if switched:
            switched_from_rag_to_norag += 1
        if pure_profile["dominance"] == "leish_dominant":
            pure_leish_dominant += 1
        if revised_profile["dominance"] == "leish_dominant":
            revised_leish_dominant += 1
        if revised_mode == "norag_prompt" or revised_profile["dominance"] == "empty" or (
            isinstance(revised_profile["prompt_context_count"], int)
            and isinstance(pure_profile["prompt_context_count"], int)
            and revised_profile["prompt_context_count"] < pure_profile["prompt_context_count"]
        ):
            reduced_or_empty += 1
        if revised_profile["dominance"] in {"mixed", "nonleish_dominant"}:
            mixed_or_nonleish += 1

        rows.append(
            {
                "qid": qid,
                "bucket": revised_row.get("ground_truth_bucket") or pure_row.get("ground_truth_bucket"),
                "pure_rag": {
                    "generation_mode": pure_mode,
                    "gating_info": pure_row.get("gating_info"),
                    "threshold_used": pure_row.get("threshold_used"),
                    "retrieval_support_status": pure_row.get("retrieval_support_status"),
                    "predicted_rank1_diagnosis_text": pure_row.get("predicted_rank1_diagnosis_text"),
                    "predicted_rank1_diagnosis_type": pure_row.get("predicted_rank1_diagnosis_type"),
                    "prompt_profile": pure_profile,
                },
                "adaptive": {
                    "generation_mode": revised_mode,
                    "gating_info": revised_row.get("gating_info"),
                    "threshold_used": revised_row.get("threshold_used"),
                    "retrieval_support_status": revised_row.get("retrieval_support_status"),
                    "predicted_rank1_diagnosis_text": revised_row.get("predicted_rank1_diagnosis_text"),
                    "predicted_rank1_diagnosis_type": revised_row.get("predicted_rank1_diagnosis_type"),
                    "prompt_profile": revised_profile,
                },
                "generation_mode_changed": mode_changed,
                "switched_from_rag_to_norag_prompt": switched,
            }
        )

    return {
        "qid_count": len(qids),
        "generation_mode_changed_count": generation_mode_changed,
        "switched_from_rag_to_norag_count": switched_from_rag_to_norag,
        "pure_rag_leish_dominant_count": pure_leish_dominant,
        "adaptive_leish_dominant_count": revised_leish_dominant,
        "adaptive_reduced_or_empty_count": reduced_or_empty,
        "adaptive_mixed_or_nonleish_count": mixed_or_nonleish,
        "rows": rows,
    }


def _build_watchlist_delta_vs_pure_rag(
    watchlist_items: List[Dict[str, object]],
    pure_rag_snapshot: Dict[str, object],
    revised_snapshot: Dict[str, object],
) -> Dict[str, object]:
    pure_rows: Dict[str, Dict] = pure_rag_snapshot["answer_rows"]  # type: ignore[assignment]
    revised_rows: Dict[str, Dict] = revised_snapshot["answer_rows"]  # type: ignore[assignment]
    rows: List[Dict[str, object]] = []
    no_new_regression = True

    for item in watchlist_items:
        qid = str(item["qid"])
        anchor_doc_ids = [str(doc_id) for doc_id in item.get("anchor_doc_ids") or []]
        pure_row = pure_rows.get(qid, {})
        revised_row = revised_rows.get(qid, {})
        pure_hits = [doc_id for doc_id in pure_row.get("prompt_context_doc_ids", []) if doc_id in anchor_doc_ids]
        revised_hits = [doc_id for doc_id in revised_row.get("prompt_context_doc_ids", []) if doc_id in anchor_doc_ids]
        regression = set(pure_hits) - set(revised_hits)
        pure_mode = _normalize_text(pure_row.get("generation_mode")) or "unknown"
        revised_mode = _normalize_text(revised_row.get("generation_mode")) or "unknown"
        prompt_anchor_regressed = bool(regression)
        no_new_regression &= not prompt_anchor_regressed
        rows.append(
            {
                "qid": qid,
                "anchor_doc_ids": anchor_doc_ids,
                "pure_rag": {
                    "generation_mode": pure_mode,
                    "prompt_anchor_hits": pure_hits,
                    "prompt_context_doc_ids": pure_row.get("prompt_context_doc_ids", []),
                },
                "adaptive": {
                    "generation_mode": revised_mode,
                    "prompt_anchor_hits": revised_hits,
                    "prompt_context_doc_ids": revised_row.get("prompt_context_doc_ids", []),
                },
                "prompt_anchor_regressed_vs_pure_rag": prompt_anchor_regressed,
            }
        )

    return {
        "watchlist": rows,
        "summary": {
            "no_new_watchlist_regression_vs_pure_rag": no_new_regression,
        },
    }


def _build_adaptive_vs_pure_rag_delta(
    pure_rag_snapshot: Dict[str, object],
    revised_snapshot: Dict[str, object],
    norag_snapshot: Dict[str, object],
    watchlist_items: List[Dict[str, object]],
    corpus_index: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    loser_sets = _build_pure_rag_loser_sets(pure_rag_snapshot, norag_snapshot)
    mode_counts = {
        "pure_rag": _mode_bucket_counts(pure_rag_snapshot["answer_rows"]),  # type: ignore[arg-type]
        "adaptive": _mode_bucket_counts(revised_snapshot["answer_rows"]),  # type: ignore[arg-type]
    }
    gating_counts = {
        "pure_rag": _gating_info_counts(pure_rag_snapshot["answer_rows"]),  # type: ignore[arg-type]
        "adaptive": _gating_info_counts(revised_snapshot["answer_rows"]),  # type: ignore[arg-type]
    }
    threshold_counts = {
        "pure_rag": _threshold_summary(pure_rag_snapshot["answer_rows"]),  # type: ignore[arg-type]
        "adaptive": _threshold_summary(revised_snapshot["answer_rows"]),  # type: ignore[arg-type]
    }

    diagnosis_delta = _build_loser_delta_rows(
        loser_sets["diagnosis_loser_qids"], pure_rag_snapshot, revised_snapshot, corpus_index
    )
    diagnosis_type_delta = _build_loser_delta_rows(
        loser_sets["diagnosis_type_loser_qids"], pure_rag_snapshot, revised_snapshot, corpus_index
    )
    nonleish_delta = _build_loser_delta_rows(
        loser_sets["nonleish_loser_qids"], pure_rag_snapshot, revised_snapshot, corpus_index
    )
    nonleish_flip_delta = _build_loser_delta_rows(
        loser_sets["nonleish_diagnosis_flip_qids"], pure_rag_snapshot, revised_snapshot, corpus_index
    )
    watchlist_delta = _build_watchlist_delta_vs_pure_rag(watchlist_items, pure_rag_snapshot, revised_snapshot)

    path_usage_changed = (
        mode_counts["adaptive"]["overall"].get("norag_prompt", 0) > 0
        and nonleish_delta["generation_mode_changed_count"] > 0
    )
    contamination_reduced = (
        nonleish_delta["adaptive_leish_dominant_count"] < nonleish_delta["pure_rag_leish_dominant_count"]
    )
    continue_to_judge = (
        path_usage_changed
        and contamination_reduced
        and bool(watchlist_delta["summary"].get("no_new_watchlist_regression_vs_pure_rag"))
    )

    return {
        "pure_rag_run_dir": str(pure_rag_snapshot["run_dir"]),
        "adaptive_run_dir": str(revised_snapshot["run_dir"]),
        "matched_norag_run_dir": str(norag_snapshot["run_dir"]),
        "loser_sets": loser_sets,
        "generation_mode_counts": mode_counts,
        "gating_info_counts": gating_counts,
        "threshold_used_summary": threshold_counts,
        "diagnosis_loser_delta": diagnosis_delta,
        "diagnosis_type_loser_delta": diagnosis_type_delta,
        "nonleish_loser_delta": nonleish_delta,
        "nonleish_diagnosis_flip_delta": nonleish_flip_delta,
        "watchlist_delta_vs_pure_rag": watchlist_delta,
        "decision": {
            "path_usage_changed_on_loser_sets": path_usage_changed,
            "nonleish_contamination_materially_reduced": contamination_reduced,
            "no_new_watchlist_regression_vs_pure_rag": bool(
                watchlist_delta["summary"].get("no_new_watchlist_regression_vs_pure_rag")
            ),
            "strong_indicator_two_of_four_nonleish_flip_qids_switched": (
                nonleish_flip_delta["switched_from_rag_to_norag_count"] >= 2
            ),
            "continue_to_judge": continue_to_judge,
        },
    }


def _build_adaptive_vs_pure_rag_markdown(delta: Dict[str, object]) -> str:
    decision = delta.get("decision", {})
    mode_counts = delta.get("generation_mode_counts", {})
    gating_counts = delta.get("gating_info_counts", {})
    threshold_summary = delta.get("threshold_used_summary", {})
    diagnosis_delta = delta.get("diagnosis_loser_delta", {})
    diagnosis_type_delta = delta.get("diagnosis_type_loser_delta", {})
    nonleish_delta = delta.get("nonleish_loser_delta", {})
    nonleish_flip_delta = delta.get("nonleish_diagnosis_flip_delta", {})
    watchlist_summary = delta.get("watchlist_delta_vs_pure_rag", {}).get("summary", {})

    lines = [
        "# Adaptive vs Pure-RAG Loser Delta",
        "",
        "## Path Usage",
        f"- Adaptive overall generation modes: `{mode_counts.get('adaptive', {}).get('overall', {})}`",
        f"- Adaptive Leish generation modes: `{mode_counts.get('adaptive', {}).get('leish', {})}`",
        f"- Adaptive Non-Leish generation modes: `{mode_counts.get('adaptive', {}).get('nonleish', {})}`",
        f"- Pure-RAG overall generation modes: `{mode_counts.get('pure_rag', {}).get('overall', {})}`",
        f"- Pure-RAG Leish generation modes: `{mode_counts.get('pure_rag', {}).get('leish', {})}`",
        f"- Pure-RAG Non-Leish generation modes: `{mode_counts.get('pure_rag', {}).get('nonleish', {})}`",
        f"- Adaptive gating_info counts: `{gating_counts.get('adaptive', {})}`",
        f"- Adaptive threshold_used summary: `{threshold_summary.get('adaptive', {})}`",
        "",
        "## Loser Sets",
        f"- Diagnosis losers: `{diagnosis_delta.get('qid_count')}` qids, `{diagnosis_delta.get('generation_mode_changed_count')}` mode changes",
        f"- Diagnosis-type losers: `{diagnosis_type_delta.get('qid_count')}` qids, `{diagnosis_type_delta.get('generation_mode_changed_count')}` mode changes",
        f"- Non-Leish losers: `{nonleish_delta.get('qid_count')}` qids, `{nonleish_delta.get('switched_from_rag_to_norag_count')}` switched to `norag_prompt`",
        f"- Non-Leish diagnosis-flip qids: `{nonleish_flip_delta.get('qid_count')}` qids, `{nonleish_flip_delta.get('switched_from_rag_to_norag_count')}` switched to `norag_prompt`",
        f"- Non-Leish Leish-dominant prompts: pure-RAG `{nonleish_delta.get('pure_rag_leish_dominant_count')}` -> adaptive `{nonleish_delta.get('adaptive_leish_dominant_count')}`",
        f"- Non-Leish loser prompts reduced/empty under adaptive: `{nonleish_delta.get('adaptive_reduced_or_empty_count')}`",
        "",
        "## Decision Flags",
        f"- Path usage changed on loser sets: `{decision.get('path_usage_changed_on_loser_sets')}`",
        f"- Non-Leish contamination materially reduced: `{decision.get('nonleish_contamination_materially_reduced')}`",
        f"- No new watchlist regression vs pure-RAG: `{watchlist_summary.get('no_new_watchlist_regression_vs_pure_rag')}`",
        f"- Strong indicator (>=2/4 Non-Leish diagnosis-flip qids switched): `{decision.get('strong_indicator_two_of_four_nonleish_flip_qids_switched')}`",
        f"- Continue to judge: `{decision.get('continue_to_judge')}`",
    ]
    return "\n".join(lines) + "\n"


def _load_watchlist(path: Path) -> List[Dict[str, object]]:
    data = _load_json(path)
    items = data.get("watchlist") or []
    return [item for item in items if item.get("qid") and item.get("anchor_doc_ids")]


def _rank_for_anchor(contexts: List[Dict], anchor_doc_id: str) -> Optional[int]:
    for rank, ctx in enumerate(contexts or [], start=1):
        doc_id = _extract_doc_id(ctx)
        if doc_id == anchor_doc_id:
            return rank
    return None


def _rank_no_worse(revised_rank: Optional[int], failed_rank: Optional[int]) -> bool:
    if failed_rank is None:
        return True
    if revised_rank is None:
        return False
    return revised_rank <= failed_rank


def _rank_improved(revised_rank: Optional[int], failed_rank: Optional[int]) -> bool:
    if failed_rank is None:
        return revised_rank is not None
    if revised_rank is None:
        return False
    return revised_rank < failed_rank


def _build_watchlist_comparison(
    watchlist_items: List[Dict[str, object]],
    run_snapshots: Dict[str, Dict[str, object]],
) -> Dict[str, object]:
    comparisons: List[Dict[str, object]] = []
    no_worse_retrieval = True
    no_worse_prompt = True
    any_anchor_improved = False

    for item in watchlist_items:
        qid = str(item["qid"])
        anchor_doc_ids = [str(doc_id) for doc_id in item.get("anchor_doc_ids") or []]
        per_run: Dict[str, Dict[str, object]] = {}

        for label, snapshot in run_snapshots.items():
            retrieval_row = snapshot["retrieval_rows"].get(qid, {})  # type: ignore[index]
            answer_row = snapshot["answer_rows"].get(qid, {})  # type: ignore[index]
            contexts = retrieval_row.get("contexts") or []
            prompt_doc_ids = answer_row.get("prompt_context_doc_ids") or []
            per_run[label] = {
                "top_doc_ids": [_extract_doc_id(ctx) for ctx in contexts[:10] if _extract_doc_id(ctx)],
                "anchor_ranks": {doc_id: _rank_for_anchor(contexts, doc_id) for doc_id in anchor_doc_ids},
                "prompt_context_doc_ids": prompt_doc_ids,
                "prompt_anchor_hits": [doc_id for doc_id in prompt_doc_ids if doc_id in anchor_doc_ids],
                "retrieval_support_status": answer_row.get("retrieval_support_status"),
                "generation_mode": answer_row.get("generation_mode"),
                "predicted_rank1_diagnosis_text": answer_row.get("predicted_rank1_diagnosis_text"),
                "predicted_rank1_diagnosis_type": answer_row.get("predicted_rank1_diagnosis_type"),
                "diagnosis_accuracy": answer_row.get("diagnosis_accuracy"),
                "diagnosis_type_accuracy": answer_row.get("diagnosis_type_accuracy"),
            }

        failed = per_run.get("failed_phase1a", {})
        revised = per_run.get("revised_phase1a", {})
        failed_anchor_ranks = failed.get("anchor_ranks", {})
        revised_anchor_ranks = revised.get("anchor_ranks", {})

        retrieval_ok = all(
            _rank_no_worse(revised_anchor_ranks.get(doc_id), failed_anchor_ranks.get(doc_id))
            for doc_id in anchor_doc_ids
        )
        prompt_ok = set(failed.get("prompt_anchor_hits", [])) <= set(revised.get("prompt_anchor_hits", []))
        improved = any(
            _rank_improved(revised_anchor_ranks.get(doc_id), failed_anchor_ranks.get(doc_id))
            for doc_id in anchor_doc_ids
        )

        no_worse_retrieval &= retrieval_ok
        no_worse_prompt &= prompt_ok
        any_anchor_improved |= improved

        comparisons.append(
            {
                "qid": qid,
                "anchor_doc_ids": anchor_doc_ids,
                "runs": per_run,
                "retrieval_no_worse_than_failed": retrieval_ok,
                "prompt_anchor_support_no_worse_than_failed": prompt_ok,
                "anchor_rank_improved_vs_failed": improved,
            }
        )

    return {
        "watchlist": comparisons,
        "summary": {
            "retrieval_no_worse_than_failed": no_worse_retrieval,
            "prompt_anchor_support_no_worse_than_failed": no_worse_prompt,
            "any_anchor_rank_improved_vs_failed": any_anchor_improved,
        },
    }


def _bucket_metrics(answer_rows: Dict[str, Dict]) -> Dict[str, Dict[str, Optional[float]]]:
    rows = list(answer_rows.values())
    report: Dict[str, Dict[str, Optional[float]]] = {}
    for bucket_name in ["overall", "leish", "nonleish"]:
        subset = rows if bucket_name == "overall" else [row for row in rows if row.get("ground_truth_bucket") == bucket_name]
        report[bucket_name] = {
            "n": float(len(subset)),
            "diagnosis_accuracy": _numeric_mean(row.get("diagnosis_accuracy") for row in subset),
            "diagnosis_type_accuracy": _numeric_mean(row.get("diagnosis_type_accuracy") for row in subset),
        }
    return report


def _build_rag_vs_norag_report(
    revised_snapshot: Dict[str, object],
    norag_snapshot: Dict[str, object],
    watchlist_comparison: Dict[str, object],
) -> Dict[str, object]:
    rag_metrics = _bucket_metrics(revised_snapshot["answer_rows"])  # type: ignore[arg-type]
    norag_metrics = _bucket_metrics(norag_snapshot["answer_rows"])  # type: ignore[arg-type]

    delta: Dict[str, Dict[str, Optional[float]]] = {}
    for bucket_name in rag_metrics:
        delta[bucket_name] = {}
        for metric_name in ["diagnosis_accuracy", "diagnosis_type_accuracy"]:
            rag_value = rag_metrics[bucket_name].get(metric_name)
            norag_value = norag_metrics[bucket_name].get(metric_name)
            delta[bucket_name][metric_name] = None if rag_value is None or norag_value is None else rag_value - norag_value

    watchlist_rows = []
    watchlist_index = {row["qid"]: row for row in watchlist_comparison.get("watchlist", [])}
    for qid, row in watchlist_index.items():
        norag_row = norag_snapshot["answer_rows"].get(qid, {})  # type: ignore[index]
        watchlist_rows.append(
            {
                "qid": qid,
                "rag": row["runs"].get("revised_phase1a", {}),
                "norag": {
                    "predicted_rank1_diagnosis_text": norag_row.get("predicted_rank1_diagnosis_text"),
                    "predicted_rank1_diagnosis_type": norag_row.get("predicted_rank1_diagnosis_type"),
                    "diagnosis_accuracy": norag_row.get("diagnosis_accuracy"),
                    "diagnosis_type_accuracy": norag_row.get("diagnosis_type_accuracy"),
                },
            }
        )

    return {
        "rag_run_dir": str(revised_snapshot["run_dir"]),
        "norag_run_dir": str(norag_snapshot["run_dir"]),
        "bucket_metrics": {
            "rag": rag_metrics,
            "norag": norag_metrics,
            "delta_rag_minus_norag": delta,
        },
        "watchlist": watchlist_rows,
    }


def _build_pre_gate_summary(
    usage_by_label: Dict[str, Dict[str, object]],
    watchlist_comparison: Dict[str, object],
    adaptive_vs_pure_rag_delta: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    failed = usage_by_label["failed_phase1a"]
    revised = usage_by_label["revised_phase1a"]
    watchlist_summary = watchlist_comparison.get("summary", {})

    conditions = {
        "augmented_retrieval_hit_count_gt_failed": revised["augmented_retrieval_hit_count"] > failed["augmented_retrieval_hit_count"],
        "distinct_augmented_docs_retrieved_gte_6": revised["distinct_augmented_docs_retrieved"] >= 6,
        "augmented_prompt_context_hit_count_gte_3": revised["augmented_prompt_context_hit_count"] >= 3,
        "distinct_augmented_docs_in_prompt_gte_2": revised["distinct_augmented_docs_in_prompt"] >= 2,
        "nonleish_prompt_context_hit_count_gte_1": revised["nonleish_prompt_context_hit_count"] >= 1,
        "watchlist_retrieval_no_worse_than_failed": bool(watchlist_summary.get("retrieval_no_worse_than_failed")),
        "watchlist_prompt_anchor_support_no_worse_than_failed": bool(
            watchlist_summary.get("prompt_anchor_support_no_worse_than_failed")
        ),
        "watchlist_any_anchor_rank_improved_vs_failed": bool(watchlist_summary.get("any_anchor_rank_improved_vs_failed")),
    }
    failed_conditions = [name for name, passed in conditions.items() if not passed]
    summary = {
        "runs": usage_by_label,
        "conditions": conditions,
        "passed": not failed_conditions,
        "failed_conditions": failed_conditions,
    }
    if adaptive_vs_pure_rag_delta:
        summary["adaptive_vs_pure_rag"] = {
            "generation_mode_counts": adaptive_vs_pure_rag_delta.get("generation_mode_counts", {}),
            "gating_info_counts": adaptive_vs_pure_rag_delta.get("gating_info_counts", {}),
            "threshold_used_summary": adaptive_vs_pure_rag_delta.get("threshold_used_summary", {}),
            "loser_sets": adaptive_vs_pure_rag_delta.get("loser_sets", {}),
            "diagnosis_loser_delta_summary": {
                k: v
                for k, v in adaptive_vs_pure_rag_delta.get("diagnosis_loser_delta", {}).items()
                if k != "rows"
            },
            "diagnosis_type_loser_delta_summary": {
                k: v
                for k, v in adaptive_vs_pure_rag_delta.get("diagnosis_type_loser_delta", {}).items()
                if k != "rows"
            },
            "nonleish_loser_delta_summary": {
                k: v
                for k, v in adaptive_vs_pure_rag_delta.get("nonleish_loser_delta", {}).items()
                if k != "rows"
            },
            "nonleish_diagnosis_flip_delta_summary": {
                k: v
                for k, v in adaptive_vs_pure_rag_delta.get("nonleish_diagnosis_flip_delta", {}).items()
                if k != "rows"
            },
            "watchlist_delta_vs_pure_rag": adaptive_vs_pure_rag_delta.get("watchlist_delta_vs_pure_rag", {}),
            "decision": adaptive_vs_pure_rag_delta.get("decision", {}),
        }
    return summary


def _build_markdown_memo(
    pre_gate_summary: Dict[str, object],
    watchlist_comparison: Dict[str, object],
    adaptive_vs_pure_rag_delta: Optional[Dict[str, object]] = None,
) -> str:
    failed_conditions = pre_gate_summary.get("failed_conditions") or []
    status = "PASS" if pre_gate_summary.get("passed") else "FAIL"
    runs = pre_gate_summary["runs"]
    failed = runs["failed_phase1a"]
    revised = runs["revised_phase1a"]
    watchlist_summary = watchlist_comparison.get("summary", {})

    lines = [
        "# Phase1a v2 Pre-Gate Memo",
        "",
        f"- Status: **{status}**",
        f"- Failed Phase1a augmented retrieval hits: `{failed['augmented_retrieval_hit_count']}` across `{failed['distinct_augmented_docs_retrieved']}` docs",
        f"- Revised Phase1a augmented retrieval hits: `{revised['augmented_retrieval_hit_count']}` across `{revised['distinct_augmented_docs_retrieved']}` docs",
        f"- Failed Phase1a prompt-context hits: `{failed['augmented_prompt_context_hit_count']}` across `{failed['distinct_augmented_docs_in_prompt']}` docs",
        f"- Revised Phase1a prompt-context hits: `{revised['augmented_prompt_context_hit_count']}` across `{revised['distinct_augmented_docs_in_prompt']}` docs",
        f"- Revised Non-Leish prompt-context hits: `{revised['nonleish_prompt_context_hit_count']}`",
        f"- Watchlist retrieval no worse than failed: `{watchlist_summary.get('retrieval_no_worse_than_failed')}`",
        f"- Watchlist prompt support no worse than failed: `{watchlist_summary.get('prompt_anchor_support_no_worse_than_failed')}`",
        f"- Any watchlist anchor-rank improvement: `{watchlist_summary.get('any_anchor_rank_improved_vs_failed')}`",
    ]
    if adaptive_vs_pure_rag_delta:
        decision = adaptive_vs_pure_rag_delta.get("decision", {})
        mode_counts = adaptive_vs_pure_rag_delta.get("generation_mode_counts", {})
        gating_counts = adaptive_vs_pure_rag_delta.get("gating_info_counts", {})
        threshold_summary = adaptive_vs_pure_rag_delta.get("threshold_used_summary", {})
        nonleish_delta = adaptive_vs_pure_rag_delta.get("nonleish_loser_delta", {})
        nonleish_flip_delta = adaptive_vs_pure_rag_delta.get("nonleish_diagnosis_flip_delta", {})
        diagnosis_delta = adaptive_vs_pure_rag_delta.get("diagnosis_loser_delta", {})
        diagnosis_type_delta = adaptive_vs_pure_rag_delta.get("diagnosis_type_loser_delta", {})
        watchlist_pure_summary = adaptive_vs_pure_rag_delta.get("watchlist_delta_vs_pure_rag", {}).get("summary", {})
        lines.extend(
            [
                "",
                "## Adaptive vs Frozen Pure-RAG",
                f"- Adaptive overall generation modes: `{mode_counts.get('adaptive', {}).get('overall', {})}`",
                f"- Adaptive Leish generation modes: `{mode_counts.get('adaptive', {}).get('leish', {})}`",
                f"- Adaptive Non-Leish generation modes: `{mode_counts.get('adaptive', {}).get('nonleish', {})}`",
                f"- Adaptive gating_info counts: `{gating_counts.get('adaptive', {})}`",
                f"- Adaptive threshold_used summary: `{threshold_summary.get('adaptive', {})}`",
                f"- Diagnosis loser qids with path changes: `{diagnosis_delta.get('generation_mode_changed_count')}` / `{diagnosis_delta.get('qid_count')}`",
                f"- Diagnosis-type loser qids with path changes: `{diagnosis_type_delta.get('generation_mode_changed_count')}` / `{diagnosis_type_delta.get('qid_count')}`",
                f"- Non-Leish loser qids switched from `rag_prompt` to `norag_prompt`: `{nonleish_delta.get('switched_from_rag_to_norag_count')}` / `{nonleish_delta.get('qid_count')}`",
                f"- Non-Leish diagnosis-flip qids switched to `norag_prompt`: `{nonleish_flip_delta.get('switched_from_rag_to_norag_count')}` / `{nonleish_flip_delta.get('qid_count')}`",
                f"- Non-Leish Leish-dominant prompts: pure-RAG `{nonleish_delta.get('pure_rag_leish_dominant_count')}` -> adaptive `{nonleish_delta.get('adaptive_leish_dominant_count')}`",
                f"- Non-Leish loser prompts reduced or empty under adaptive: `{nonleish_delta.get('adaptive_reduced_or_empty_count')}`",
                f"- No new watchlist regression vs frozen pure-RAG: `{watchlist_pure_summary.get('no_new_watchlist_regression_vs_pure_rag')}`",
                f"- Path usage changed on loser sets: `{decision.get('path_usage_changed_on_loser_sets')}`",
                f"- Non-Leish contamination materially reduced: `{decision.get('nonleish_contamination_materially_reduced')}`",
                f"- Strong indicator (>=2/4 diagnosis-flip qids switched): `{decision.get('strong_indicator_two_of_four_nonleish_flip_qids_switched')}`",
                f"- Continue to Gemini diagnosis evaluation: `{decision.get('continue_to_judge')}`",
            ]
        )
    if failed_conditions:
        lines.extend(["", "## Failed Conditions"])
        lines.extend([f"- `{condition}`" for condition in failed_conditions])
    return "\n".join(lines) + "\n"


def _write_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase1a v2 pre-gate and watchlist analyzer")
    parser.add_argument("--baseline-run", required=True, help="Baseline run dir or run id")
    parser.add_argument("--failed-run", required=True, help="Failed Phase1a run dir or run id")
    parser.add_argument("--revised-run", required=True, help="Revised Phase1a run dir or run id")
    parser.add_argument("--augmented-case-list", required=True, type=Path, help="Selected Phase1a Tier A case list")
    parser.add_argument(
        "--watchlist-json",
        type=Path,
        default=Path("/home/ngocnt/experiments/structured_cases_v4/runbooks/phase1a_watchlist.json"),
        help="Watchlist config JSON",
    )
    parser.add_argument("--output-dir", type=Path, help="Directory for pre-gate outputs")
    parser.add_argument("--norag-run", help="Optional matched no-RAG diagnostic run")
    parser.add_argument("--pure-rag-run", help="Optional frozen pure-RAG run for adaptive-vs-pure comparison")
    args = parser.parse_args()

    augmented_case_ids = {
        line.strip()
        for line in args.augmented_case_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    run_snapshots = {
        "baseline": _load_run_snapshot(args.baseline_run),
        "failed_phase1a": _load_run_snapshot(args.failed_run),
        "revised_phase1a": _load_run_snapshot(args.revised_run),
    }
    if args.norag_run:
        run_snapshots["norag"] = _load_run_snapshot(args.norag_run)
    pure_rag_run = args.pure_rag_run or os.environ.get("FROZEN_PURE_RAG_RUN")
    if pure_rag_run:
        run_snapshots["pure_rag"] = _load_run_snapshot(pure_rag_run)

    output_dir = args.output_dir or (Path(run_snapshots["revised_phase1a"]["run_dir"]) / "phase1a_pre_gate")
    output_dir.mkdir(parents=True, exist_ok=True)

    usage_by_label = {
        label: _build_usage_summary(snapshot, augmented_case_ids)
        for label, snapshot in run_snapshots.items()
        if label != "norag"
    }
    watchlist_items = _load_watchlist(args.watchlist_json)
    watchlist_comparison = _build_watchlist_comparison(
        watchlist_items,
        {
            label: snapshot
            for label, snapshot in run_snapshots.items()
            if label in {"baseline", "failed_phase1a", "revised_phase1a"}
        },
    )
    adaptive_vs_pure_rag_delta = None
    if "pure_rag" in run_snapshots and "norag" in run_snapshots:
        corpus_index = _load_corpus_index(_train_jsonl_from_snapshot(run_snapshots["revised_phase1a"]))
        adaptive_vs_pure_rag_delta = _build_adaptive_vs_pure_rag_delta(
            run_snapshots["pure_rag"],
            run_snapshots["revised_phase1a"],
            run_snapshots["norag"],
            watchlist_items,
            corpus_index,
        )

    pre_gate_summary = _build_pre_gate_summary(usage_by_label, watchlist_comparison, adaptive_vs_pure_rag_delta)
    pre_gate_memo = _build_markdown_memo(pre_gate_summary, watchlist_comparison, adaptive_vs_pure_rag_delta)

    _write_json(output_dir / "pre_gate_summary.json", pre_gate_summary)
    _write_json(output_dir / "watchlist_comparison.json", watchlist_comparison)
    _write_text(output_dir / "pre_gate_memo.md", pre_gate_memo)
    if adaptive_vs_pure_rag_delta:
        _write_json(output_dir / "adaptive_vs_pure_rag_loser_delta.json", adaptive_vs_pure_rag_delta)
        _write_text(
            output_dir / "adaptive_vs_pure_rag_loser_delta.md",
            _build_adaptive_vs_pure_rag_markdown(adaptive_vs_pure_rag_delta),
        )

    if "norag" in run_snapshots:
        rag_vs_norag = _build_rag_vs_norag_report(
            run_snapshots["revised_phase1a"],
            run_snapshots["norag"],
            watchlist_comparison,
        )
        _write_json(output_dir / "rag_vs_norag_watchlist.json", rag_vs_norag)

    print(pre_gate_memo, end="")
    print(f"pre_gate_summary={output_dir / 'pre_gate_summary.json'}")
    print(f"watchlist_comparison={output_dir / 'watchlist_comparison.json'}")
    if adaptive_vs_pure_rag_delta:
        print(f"adaptive_vs_pure_rag_loser_delta={output_dir / 'adaptive_vs_pure_rag_loser_delta.json'}")
    if "norag" in run_snapshots:
        print(f"rag_vs_norag_watchlist={output_dir / 'rag_vs_norag_watchlist.json'}")


if __name__ == "__main__":
    main()
