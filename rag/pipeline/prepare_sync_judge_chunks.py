#!/usr/bin/env python3
"""Prepare fresh chunked sync-judge replacement runs from completed generator artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


FORBIDDEN_COPY_FILES = {
    "eval_manifest.json",
    "eval_manifest.lock",
    "phase_a_metrics.jsonl",
    "phase_b_verified.jsonl",
    "phase_c_pseudolabel.jsonl",
    "ragas.jsonl",
}


@dataclass(frozen=True)
class ChunkSpec:
    chunk_id: str
    qids: List[str]


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: Dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_jsonl_rows(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl_rows(path: Path, rows: Sequence[Dict]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _index_by_qid(rows: Iterable[Dict], label: str, path: Path) -> Dict[str, Dict]:
    indexed: Dict[str, Dict] = {}
    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            raise RuntimeError(f"{label} row without qid in {path}")
        if qid in indexed:
            raise RuntimeError(f"Duplicate qid={qid} in {label}: {path}")
        indexed[qid] = row
    return indexed


def _load_chunk_specs(plan_path: Path) -> List[ChunkSpec]:
    payload = _load_json(plan_path)
    specs: List[ChunkSpec] = []
    for raw_chunk in payload.get("chunks", []):
        chunk_id = str(raw_chunk.get("chunk_id", "")).strip()
        qids = [str(qid).strip() for qid in raw_chunk.get("qids", []) if str(qid).strip()]
        if not chunk_id:
            raise RuntimeError(f"{plan_path} contains a chunk without chunk_id")
        if not qids:
            raise RuntimeError(f"{plan_path} chunk {chunk_id!r} has no qids")
        specs.append(ChunkSpec(chunk_id=chunk_id, qids=qids))
    if not specs:
        raise RuntimeError(f"{plan_path} contains no chunks")
    return specs


def _validate_chunk_specs(specs: Sequence[ChunkSpec]) -> List[str]:
    all_qids: List[str] = []
    seen = set()
    for spec in specs:
        for qid in spec.qids:
            if qid in seen:
                raise RuntimeError(f"Duplicate qid across chunk plan: {qid}")
            seen.add(qid)
            all_qids.append(qid)
    return all_qids


def _queries_to_qid(query: Dict) -> str:
    return f"{query['case_id']}::{query['query_type']}"


def _copy_snapshot_dir(snapshot_dir: Optional[Path], run_dir: Path) -> Optional[str]:
    if snapshot_dir is None:
        return None
    if not snapshot_dir.exists():
        raise RuntimeError(f"Snapshot dir does not exist: {snapshot_dir}")
    target = run_dir / "batch_attempt_snapshot"
    shutil.copytree(snapshot_dir, target)
    return str(target)


def _filter_jsonl_file(source_path: Path, ordered_qids: Sequence[str]) -> List[Dict]:
    indexed = _index_by_qid(_load_jsonl_rows(source_path), source_path.name, source_path)
    missing = [qid for qid in ordered_qids if qid not in indexed]
    if missing:
        raise RuntimeError(f"{source_path} missing qids: {missing[:5]}")
    return [indexed[qid] for qid in ordered_qids]


def prepare_sync_judge_chunks(
    *,
    source_run: Path,
    plan_path: Path,
    output_root: Path,
    run_prefix: str,
    answers_file: str,
    copy_files: Sequence[str],
    source_batch_judge_run: Path,
    source_batch_job_id: str,
    snapshot_dir: Optional[Path],
    sync_replacement_of: str,
    execution_mode: str,
    chunk_plan_label: str,
    phase_a_lineage: str,
    raw_answer_file: Optional[str] = None,
    evaluation_protocol_note: Optional[str] = None,
) -> List[Path]:
    source_run = source_run.resolve()
    source_batch_judge_run = source_batch_judge_run.resolve()
    output_root = output_root.resolve()
    plan_path = plan_path.resolve()
    snapshot_dir = snapshot_dir.resolve() if snapshot_dir is not None else None
    specs = _load_chunk_specs(plan_path)
    plan_qids = _validate_chunk_specs(specs)

    answers_path = source_run / answers_file
    answers_rows = _load_jsonl_rows(answers_path)
    full_qid_order = [row["qid"] for row in answers_rows]
    if set(full_qid_order) != set(plan_qids):
        missing = sorted(set(full_qid_order) - set(plan_qids))
        extra = sorted(set(plan_qids) - set(full_qid_order))
        raise RuntimeError(
            "Chunk plan must cover the source answers exactly. "
            f"Missing from plan: {missing[:5]} Extra in plan: {extra[:5]}"
        )

    with open(source_run / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    queries_by_qid = {_queries_to_qid(query): query for query in queries}

    copy_files = list(dict.fromkeys(copy_files))
    for name in copy_files:
        if name in FORBIDDEN_COPY_FILES:
            raise RuntimeError(f"Refusing to prepare chunks with phase/judge artifact copy file: {name}")
        if not (source_run / name).exists():
            raise RuntimeError(f"Requested copy file missing from source run: {source_run / name}")
    if answers_file not in copy_files:
        raise RuntimeError(f"answers_file must also be included in copy_files: {answers_file}")
    if raw_answer_file and raw_answer_file not in copy_files:
        raise RuntimeError(f"raw_answer_file must also be included in copy_files: {raw_answer_file}")

    source_run_config = _load_json(source_run / "run_config.json") if (source_run / "run_config.json").exists() else {}
    source_summary = _load_json(source_run / "summary.json") if (source_run / "summary.json").exists() else {}

    created: List[Path] = []
    qid_set_cache = {spec.chunk_id: set(spec.qids) for spec in specs}
    for index, spec in enumerate(specs, start=1):
        run_dir = output_root / f"{run_prefix}_{spec.chunk_id}"
        if run_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing chunk run dir: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        ordered_qids = [qid for qid in full_qid_order if qid in qid_set_cache[spec.chunk_id]]

        for name in copy_files:
            source_path = source_run / name
            target_path = run_dir / name
            if name == "queries.json":
                filtered_queries = [queries_by_qid[qid] for qid in ordered_qids]
                with open(target_path, "w", encoding="utf-8") as f:
                    json.dump(filtered_queries, f, ensure_ascii=False, indent=2)
            elif source_path.suffix == ".jsonl":
                _write_jsonl_rows(target_path, _filter_jsonl_file(source_path, ordered_qids))
            else:
                shutil.copy2(source_path, target_path)

        copied_snapshot_dir = _copy_snapshot_dir(snapshot_dir, run_dir)

        run_config = dict(source_run_config)
        run_config.update(
            {
                "run_id": run_dir.name,
                "source_generator_run_dir": str(source_run),
                "source_batch_judge_run_dir": str(source_batch_judge_run),
                "source_batch_job_ids": [source_batch_job_id],
                "sync_replacement_of": sync_replacement_of,
                "execution_mode": execution_mode,
                "chunk_plan": chunk_plan_label,
                "shared_chunk_plan_file": str(plan_path),
                "batch_attempt_snapshot_dir": copied_snapshot_dir,
                "phase_a_lineage": phase_a_lineage,
                "answers_file_for_eval": answers_file,
                "chunk_index": index,
                "chunk_id": spec.chunk_id,
                "chunk_qids": ordered_qids,
            }
        )
        if raw_answer_file:
            run_config["raw_norag_answers_file"] = raw_answer_file
        if evaluation_protocol_note:
            run_config["evaluation_protocol_note"] = evaluation_protocol_note
        _write_json(run_dir / "run_config.json", run_config)

        summary = dict(source_summary)
        summary.update(
            {
                "run_id": run_dir.name,
                "n_queries": len(ordered_qids),
                "source_generator_run_dir": str(source_run),
                "source_batch_judge_run_dir": str(source_batch_judge_run),
                "source_batch_job_ids": [source_batch_job_id],
                "sync_replacement_of": sync_replacement_of,
                "execution_mode": execution_mode,
                "chunk_plan": chunk_plan_label,
                "shared_chunk_plan_file": str(plan_path),
                "batch_attempt_snapshot_dir": copied_snapshot_dir,
                "phase_a_lineage": phase_a_lineage,
                "answers_file_for_eval": answers_file,
                "chunk_index": index,
                "chunk_id": spec.chunk_id,
                "chunk_qids": ordered_qids,
                "note": (
                    "Chunked synchronous judge replacement built from completed Gemini generator artifacts. "
                    "Phase A will be recomputed locally for this chunk."
                ),
            }
        )
        if raw_answer_file:
            summary["raw_norag_answers_file"] = raw_answer_file
        if evaluation_protocol_note:
            summary["evaluation_protocol_note"] = evaluation_protocol_note
        _write_json(run_dir / "summary.json", summary)

        created.append(run_dir)

    return created


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare fresh chunked sync-judge replacement runs")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--plan-json", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--run-prefix", required=True)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--copy-file", action="append", required=True, help="Generator-side file to copy/filter into each chunk")
    parser.add_argument("--source-batch-judge-run", required=True, type=Path)
    parser.add_argument("--source-batch-job-id", required=True)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--sync-replacement-of", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--chunk-plan-label", required=True)
    parser.add_argument("--phase-a-lineage", required=True)
    parser.add_argument("--raw-answer-file", default=None)
    parser.add_argument("--evaluation-protocol-note", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = prepare_sync_judge_chunks(
        source_run=args.source_run,
        plan_path=args.plan_json,
        output_root=args.output_root,
        run_prefix=args.run_prefix,
        answers_file=args.answers_file,
        copy_files=args.copy_file,
        source_batch_judge_run=args.source_batch_judge_run,
        source_batch_job_id=args.source_batch_job_id,
        snapshot_dir=args.snapshot_dir,
        sync_replacement_of=args.sync_replacement_of,
        execution_mode=args.execution_mode,
        chunk_plan_label=args.chunk_plan_label,
        phase_a_lineage=args.phase_a_lineage,
        raw_answer_file=args.raw_answer_file,
        evaluation_protocol_note=args.evaluation_protocol_note,
    )
    for path in run_dirs:
        print(path)


if __name__ == "__main__":
    main()
