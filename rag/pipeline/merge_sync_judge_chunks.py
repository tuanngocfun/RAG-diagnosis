#!/usr/bin/env python3
"""Merge completed sync-judge chunks into one synthetic full judged run."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from .config import get_runtime_metadata, JUDGE_MODEL


PHASE_FILE_NAMES = {
    "phase_a_metrics": "phase_a_metrics.jsonl",
    "phase_b_verified": "phase_b_verified.jsonl",
    "phase_c_pseudolabel": "phase_c_pseudolabel.jsonl",
}
FINAL_RAGAS_FILE = "ragas.jsonl"
FORBIDDEN_COPY_FILES = {
    "eval_manifest.json",
    "eval_manifest.lock",
    "phase_a_metrics.jsonl",
    "phase_b_verified.jsonl",
    "phase_c_pseudolabel.jsonl",
    "ragas.jsonl",
}


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


def _copy_snapshot_dir(snapshot_dir: Optional[Path], output_run: Path) -> Optional[str]:
    if snapshot_dir is None:
        return None
    if not snapshot_dir.exists():
        raise RuntimeError(f"Snapshot dir does not exist: {snapshot_dir}")
    target = output_run / "batch_attempt_snapshot"
    shutil.copytree(snapshot_dir, target)
    return str(target)


def _build_completed_manifest(
    *,
    run_dir: Path,
    answers_file: str,
    ordered_qids: Sequence[str],
    judge_model: Optional[str],
) -> Dict:
    answers_path = run_dir / answers_file
    answers_sha256 = _sha256_file(answers_path)
    selected_qids_sha256 = _sha256_lines(ordered_qids)
    now = time.time()
    n = len(ordered_qids)
    return {
        "version": 2,
        "run_id": run_dir.name,
        "run_dir": str(run_dir.resolve()),
        "answers_file": answers_file,
        "answers_sha256": answers_sha256,
        "selected_qids_sha256": selected_qids_sha256,
        "selected_qid_count": n,
        "judge_model": judge_model or JUDGE_MODEL,
        "runtime_metadata": get_runtime_metadata(),
        "updated_at": now,
        "phases": {
            "phase_a_metrics": {
                "status": "completed",
                "file": PHASE_FILE_NAMES["phase_a_metrics"],
                "label_track": "retrieval_metrics",
                "n_completed": n,
                "updated_at": now,
            },
            "phase_b_verified": {
                "status": "completed",
                "file": PHASE_FILE_NAMES["phase_b_verified"],
                "label_track": "verified",
                "n_completed": n,
                "updated_at": now,
                "batch_job_id": None,
                "batch_judge_model": None,
                "last_observed_batch_state": None,
            },
            "phase_c_pseudolabel": {
                "status": "completed",
                "file": PHASE_FILE_NAMES["phase_c_pseudolabel"],
                "label_track": "pseudolabel",
                "n_completed": n,
                "updated_at": now,
                "batch_job_id": None,
                "batch_judge_model": None,
                "last_observed_batch_state": None,
            },
            "merge": {
                "status": "completed",
                "file": FINAL_RAGAS_FILE,
                "label_track": "",
                "n_completed": n,
                "updated_at": now,
            },
        },
    }


def merge_sync_judge_chunks(
    *,
    source_run: Path,
    output_run: Path,
    answers_file: str,
    copy_files: Sequence[str],
    chunk_runs: Sequence[Path],
    source_batch_judge_run: Path,
    source_batch_job_id: str,
    snapshot_dir: Optional[Path],
    sync_replacement_of: str,
    execution_mode: str,
    chunk_plan_label: str,
    phase_a_lineage: str,
    judge_model: Optional[str] = None,
    raw_answer_file: Optional[str] = None,
    evaluation_protocol_note: Optional[str] = None,
    shared_chunk_plan_file: Optional[str] = None,
) -> Path:
    source_run = source_run.resolve()
    source_batch_judge_run = source_batch_judge_run.resolve()
    output_run = output_run.resolve()
    snapshot_dir = snapshot_dir.resolve() if snapshot_dir is not None else None
    chunk_runs = [chunk_run.resolve() for chunk_run in chunk_runs]

    if output_run.exists():
        raise RuntimeError(f"Refusing to overwrite existing merged run dir: {output_run}")
    output_run.mkdir(parents=True, exist_ok=False)

    copy_files = list(dict.fromkeys(copy_files))
    for name in copy_files:
        if name in FORBIDDEN_COPY_FILES:
            raise RuntimeError(f"Refusing to merge with phase/judge artifact copy file: {name}")
        if not (source_run / name).exists():
            raise RuntimeError(f"Requested copy file missing from source run: {source_run / name}")
    if answers_file not in copy_files:
        raise RuntimeError(f"answers_file must also be included in copy_files: {answers_file}")
    if raw_answer_file and raw_answer_file not in copy_files:
        raise RuntimeError(f"raw_answer_file must also be included in copy_files: {raw_answer_file}")

    source_answers = _load_jsonl_rows(source_run / answers_file)
    ordered_qids = [row["qid"] for row in source_answers]

    phase_a_by_qid: Dict[str, Dict] = {}
    phase_b_by_qid: Dict[str, Dict] = {}
    phase_c_by_qid: Dict[str, Dict] = {}
    ragas_by_qid: Dict[str, Dict] = {}
    chunk_run_dirs: List[str] = []

    for chunk_run in chunk_runs:
        chunk_run_dirs.append(str(chunk_run))
        for qid, row in _index_by_qid(
            _load_jsonl_rows(chunk_run / "phase_a_metrics.jsonl"),
            "phase_a_metrics",
            chunk_run / "phase_a_metrics.jsonl",
        ).items():
            if qid in phase_a_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk phase_a outputs: {qid}")
            phase_a_by_qid[qid] = row
        for qid, row in _index_by_qid(
            _load_jsonl_rows(chunk_run / "phase_b_verified.jsonl"),
            "phase_b_verified",
            chunk_run / "phase_b_verified.jsonl",
        ).items():
            if qid in phase_b_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk phase_b outputs: {qid}")
            phase_b_by_qid[qid] = row
        for qid, row in _index_by_qid(
            _load_jsonl_rows(chunk_run / "phase_c_pseudolabel.jsonl"),
            "phase_c_pseudolabel",
            chunk_run / "phase_c_pseudolabel.jsonl",
        ).items():
            if qid in phase_c_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk phase_c outputs: {qid}")
            phase_c_by_qid[qid] = row
        for qid, row in _index_by_qid(
            _load_jsonl_rows(chunk_run / "ragas.jsonl"),
            "ragas",
            chunk_run / "ragas.jsonl",
        ).items():
            if qid in ragas_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk ragas outputs: {qid}")
            ragas_by_qid[qid] = row

    for label, index in [
        ("phase_a_metrics", phase_a_by_qid),
        ("phase_b_verified", phase_b_by_qid),
        ("phase_c_pseudolabel", phase_c_by_qid),
        ("ragas", ragas_by_qid),
    ]:
        missing = [qid for qid in ordered_qids if qid not in index]
        extra = sorted(set(index) - set(ordered_qids))
        if missing or extra:
            raise RuntimeError(f"Merged chunk coverage mismatch for {label}. Missing={missing[:5]} Extra={extra[:5]}")

    for name in copy_files:
        shutil.copy2(source_run / name, output_run / name)

    copied_snapshot_dir = _copy_snapshot_dir(snapshot_dir, output_run)

    _write_jsonl_rows(output_run / "phase_a_metrics.jsonl", [phase_a_by_qid[qid] for qid in ordered_qids])
    _write_jsonl_rows(output_run / "phase_b_verified.jsonl", [phase_b_by_qid[qid] for qid in ordered_qids])
    _write_jsonl_rows(output_run / "phase_c_pseudolabel.jsonl", [phase_c_by_qid[qid] for qid in ordered_qids])
    _write_jsonl_rows(output_run / "ragas.jsonl", [ragas_by_qid[qid] for qid in ordered_qids])

    run_config = _load_json(output_run / "run_config.json") if (output_run / "run_config.json").exists() else {}
    run_config.update(
        {
            "run_id": output_run.name,
            "source_generator_run_dir": str(source_run),
            "source_batch_judge_run_dir": str(source_batch_judge_run),
            "source_batch_job_ids": [source_batch_job_id],
            "sync_replacement_of": sync_replacement_of,
            "execution_mode": execution_mode,
            "chunk_plan": chunk_plan_label,
            "shared_chunk_plan_file": shared_chunk_plan_file,
            "batch_attempt_snapshot_dir": copied_snapshot_dir,
            "phase_a_lineage": phase_a_lineage,
            "answers_file_for_eval": answers_file,
            "merged_from_chunk_runs": chunk_run_dirs,
        }
    )
    if raw_answer_file:
        run_config["raw_norag_answers_file"] = raw_answer_file
    if evaluation_protocol_note:
        run_config["evaluation_protocol_note"] = evaluation_protocol_note
    _write_json(output_run / "run_config.json", run_config)

    summary = _load_json(output_run / "summary.json") if (output_run / "summary.json").exists() else {}
    summary.update(
        {
            "run_id": output_run.name,
            "n_queries": len(ordered_qids),
            "source_generator_run_dir": str(source_run),
            "source_batch_judge_run_dir": str(source_batch_judge_run),
            "source_batch_job_ids": [source_batch_job_id],
            "sync_replacement_of": sync_replacement_of,
            "execution_mode": execution_mode,
            "chunk_plan": chunk_plan_label,
            "shared_chunk_plan_file": shared_chunk_plan_file,
            "batch_attempt_snapshot_dir": copied_snapshot_dir,
            "phase_a_lineage": phase_a_lineage,
            "answers_file_for_eval": answers_file,
            "merged_from_chunk_runs": chunk_run_dirs,
            "note": (
                "Synthetic merged full judged run created from four synchronous sync-replacement chunks. "
                "Generator outputs are reused from the frozen Gemini source run."
            ),
        }
    )
    if raw_answer_file:
        summary["raw_norag_answers_file"] = raw_answer_file
    if evaluation_protocol_note:
        summary["evaluation_protocol_note"] = evaluation_protocol_note
    _write_json(output_run / "summary.json", summary)

    manifest = _build_completed_manifest(
        run_dir=output_run,
        answers_file=answers_file,
        ordered_qids=ordered_qids,
        judge_model=judge_model,
    )
    _write_json(output_run / "eval_manifest.json", manifest)
    return output_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge completed sync-judge chunks into one synthetic full judged run")
    parser.add_argument("--source-run", required=True, type=Path)
    parser.add_argument("--output-run", required=True, type=Path)
    parser.add_argument("--answers-file", required=True)
    parser.add_argument("--copy-file", action="append", required=True, help="Generator-side source file to copy unchanged")
    parser.add_argument("--merged-from-chunk-run", action="append", required=True, type=Path)
    parser.add_argument("--source-batch-judge-run", required=True, type=Path)
    parser.add_argument("--source-batch-job-id", required=True)
    parser.add_argument("--snapshot-dir", type=Path, default=None)
    parser.add_argument("--sync-replacement-of", required=True)
    parser.add_argument("--execution-mode", required=True)
    parser.add_argument("--chunk-plan-label", required=True)
    parser.add_argument("--phase-a-lineage", required=True)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--raw-answer-file", default=None)
    parser.add_argument("--evaluation-protocol-note", default=None)
    parser.add_argument("--shared-chunk-plan-file", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_sync_judge_chunks(
        source_run=args.source_run,
        output_run=args.output_run,
        answers_file=args.answers_file,
        copy_files=args.copy_file,
        chunk_runs=args.merged_from_chunk_run,
        source_batch_judge_run=args.source_batch_judge_run,
        source_batch_job_id=args.source_batch_job_id,
        snapshot_dir=args.snapshot_dir,
        sync_replacement_of=args.sync_replacement_of,
        execution_mode=args.execution_mode,
        chunk_plan_label=args.chunk_plan_label,
        phase_a_lineage=args.phase_a_lineage,
        judge_model=args.judge_model,
        raw_answer_file=args.raw_answer_file,
        evaluation_protocol_note=args.evaluation_protocol_note,
        shared_chunk_plan_file=args.shared_chunk_plan_file,
    )
    print(merged)


if __name__ == "__main__":
    main()
