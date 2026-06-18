#!/usr/bin/env python3
"""Merge completed chunked no-RAG control outputs into a single synthetic 56-case control run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


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


def merge_chunked_control(
    source_run: Path,
    chunk_runs: Sequence[Path],
    output_run: Path,
    source_batch_job_id: str,
) -> Path:
    source_run = source_run.resolve()
    output_run = output_run.resolve()
    if output_run.exists():
        raise RuntimeError(f"Refusing to overwrite existing merged run dir: {output_run}")
    output_run.mkdir(parents=True, exist_ok=False)

    source_answers = _load_jsonl_rows(source_run / "answers_norag.jsonl")
    ordered_qids = [row["qid"] for row in source_answers]
    ragas_by_qid: Dict[str, Dict] = {}
    phase_b_by_qid: Dict[str, Dict] = {}
    phase_c_by_qid: Dict[str, Dict] = {}
    chunk_run_dirs: List[str] = []

    for chunk_run in chunk_runs:
        chunk_run = chunk_run.resolve()
        chunk_run_dirs.append(str(chunk_run))
        for qid, row in _index_by_qid(_load_jsonl_rows(chunk_run / "ragas.jsonl"), "ragas", chunk_run / "ragas.jsonl").items():
            if qid in ragas_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk ragas outputs: {qid}")
            ragas_by_qid[qid] = row
        for qid, row in _index_by_qid(_load_jsonl_rows(chunk_run / "phase_b_verified.jsonl"), "phase_b_verified", chunk_run / "phase_b_verified.jsonl").items():
            if qid in phase_b_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk phase_b outputs: {qid}")
            phase_b_by_qid[qid] = row
        for qid, row in _index_by_qid(_load_jsonl_rows(chunk_run / "phase_c_pseudolabel.jsonl"), "phase_c_pseudolabel", chunk_run / "phase_c_pseudolabel.jsonl").items():
            if qid in phase_c_by_qid:
                raise RuntimeError(f"Duplicate qid across chunk phase_c outputs: {qid}")
            phase_c_by_qid[qid] = row

    missing = [qid for qid in ordered_qids if qid not in ragas_by_qid]
    extra = sorted(set(ragas_by_qid) - set(ordered_qids))
    if missing or extra:
        raise RuntimeError(f"Merged chunk coverage mismatch. Missing={missing[:5]} Extra={extra[:5]}")

    for filename in ("answers_norag.jsonl", "answers_gemini.jsonl", "retrieval.jsonl"):
        _write_jsonl_rows(output_run / filename, _load_jsonl_rows(source_run / filename))
    with open(source_run / "queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    with open(output_run / "queries.json", "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)

    _write_jsonl_rows(output_run / "ragas.jsonl", [ragas_by_qid[qid] for qid in ordered_qids])
    _write_jsonl_rows(output_run / "phase_b_verified.jsonl", [phase_b_by_qid[qid] for qid in ordered_qids])
    _write_jsonl_rows(output_run / "phase_c_pseudolabel.jsonl", [phase_c_by_qid[qid] for qid in ordered_qids])

    run_config = _load_json(source_run / "run_config.json")
    run_config["run_id"] = output_run.name
    run_config["control_type"] = "matched_norag_chunked_sync"
    run_config["chunk_plan"] = "4x14_sync"
    run_config["source_full_run_dir"] = str(source_run)
    run_config["source_full_batch_job_id"] = source_batch_job_id
    run_config["merged_from_chunk_runs"] = chunk_run_dirs
    run_config["phase_a_lineage"] = "reused_from_source_full_run"
    _write_json(output_run / "run_config.json", run_config)

    summary = _load_json(source_run / "summary.json")
    summary["run_id"] = output_run.name
    summary["n_queries"] = len(ordered_qids)
    summary["control_type"] = "matched_norag_chunked_sync"
    summary["chunk_plan"] = "4x14_sync"
    summary["source_full_run_dir"] = str(source_run)
    summary["source_full_batch_job_id"] = source_batch_job_id
    summary["merged_from_chunk_runs"] = chunk_run_dirs
    summary["phase_a_lineage"] = "reused_from_source_full_run"
    summary["note"] = (
        "Synthetic merged full no-RAG control created from four synchronous chunked judge runs. "
        "Use this merged run for the final 56-case RAG-vs-no-RAG comparison. "
        "Phase A is reused from the source full run because generation answers and retrieved-context absence are unchanged."
    )
    _write_json(output_run / "summary.json", summary)
    return output_run


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge completed chunked no-RAG control runs into a synthetic full control")
    parser.add_argument("--source-run", required=True, type=Path, help="Original full matched no-RAG v2 generation run")
    parser.add_argument("--chunk-run", required=True, action="append", type=Path, help="Completed chunk run dir (repeat four times)")
    parser.add_argument("--output-run", required=True, type=Path, help="Output merged run directory")
    parser.add_argument("--source-batch-job-id", required=True, help="Cancelled source batch job id to record in merged metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    merged = merge_chunked_control(
        source_run=args.source_run,
        chunk_runs=args.chunk_run,
        output_run=args.output_run,
        source_batch_job_id=args.source_batch_job_id,
    )
    print(merged)


if __name__ == "__main__":
    main()
