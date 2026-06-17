#!/usr/bin/env python3
"""Create balanced chunk run directories from an existing full matched no-RAG run."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


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


def _filter_rows_in_order(indexed: Dict[str, Dict], ordered_qids: Sequence[str], label: str, path: Path) -> List[Dict]:
    missing = [qid for qid in ordered_qids if qid not in indexed]
    if missing:
        raise RuntimeError(f"{label} missing qids in {path}: {missing[:5]}")
    return [indexed[qid] for qid in ordered_qids]


def prepare_chunked_control(
    source_run: Path,
    plan_path: Path,
    output_root: Path,
    run_prefix: str,
    source_batch_job_id: str,
) -> List[Path]:
    source_run = source_run.resolve()
    output_root = output_root.resolve()
    specs = _load_chunk_specs(plan_path)
    plan_qids = _validate_chunk_specs(specs)

    answers_rows = _load_jsonl_rows(source_run / "answers_norag.jsonl")
    full_qid_order = [row["qid"] for row in answers_rows]
    if set(full_qid_order) != set(plan_qids):
        missing = sorted(set(full_qid_order) - set(plan_qids))
        extra = sorted(set(plan_qids) - set(full_qid_order))
        raise RuntimeError(
            "Chunk plan must cover the source answers exactly. "
            f"Missing from plan: {missing[:5]} Extra in plan: {extra[:5]}"
        )

    answers_by_qid = _index_by_qid(answers_rows, "answers_norag", source_run / "answers_norag.jsonl")
    answers_gemini_by_qid = _index_by_qid(
        _load_jsonl_rows(source_run / "answers_gemini.jsonl"),
        "answers_gemini",
        source_run / "answers_gemini.jsonl",
    )
    retrieval_by_qid = _index_by_qid(
        _load_jsonl_rows(source_run / "retrieval.jsonl"),
        "retrieval",
        source_run / "retrieval.jsonl",
    )
    phase_a_by_qid = _index_by_qid(
        _load_jsonl_rows(source_run / "phase_a_metrics.jsonl"),
        "phase_a_metrics",
        source_run / "phase_a_metrics.jsonl",
    )
    with open(source_run / "queries.json", "r", encoding="utf-8") as f:
        raw_queries = json.load(f)
    queries_by_qid = {_queries_to_qid(query): query for query in raw_queries}

    run_config = _load_json(source_run / "run_config.json")
    summary = _load_json(source_run / "summary.json")
    created_run_dirs: List[Path] = []

    for index, spec in enumerate(specs, start=1):
        run_dir = output_root / f"{run_prefix}_{spec.chunk_id}"
        if run_dir.exists():
            raise RuntimeError(f"Refusing to overwrite existing chunk run dir: {run_dir}")
        run_dir.mkdir(parents=True, exist_ok=False)
        ordered_qids = [qid for qid in full_qid_order if qid in set(spec.qids)]

        _write_jsonl_rows(run_dir / "answers_norag.jsonl", _filter_rows_in_order(answers_by_qid, ordered_qids, "answers_norag", source_run / "answers_norag.jsonl"))
        _write_jsonl_rows(run_dir / "answers_gemini.jsonl", _filter_rows_in_order(answers_gemini_by_qid, ordered_qids, "answers_gemini", source_run / "answers_gemini.jsonl"))
        _write_jsonl_rows(run_dir / "retrieval.jsonl", _filter_rows_in_order(retrieval_by_qid, ordered_qids, "retrieval", source_run / "retrieval.jsonl"))
        _write_jsonl_rows(run_dir / "phase_a_metrics.jsonl", _filter_rows_in_order(phase_a_by_qid, ordered_qids, "phase_a_metrics", source_run / "phase_a_metrics.jsonl"))
        with open(run_dir / "queries.json", "w", encoding="utf-8") as f:
            json.dump([queries_by_qid[qid] for qid in ordered_qids], f, ensure_ascii=False, indent=2)

        chunk_run_config = dict(run_config)
        chunk_run_config["run_id"] = run_dir.name
        chunk_run_config["control_type"] = "matched_norag_chunk"
        chunk_run_config["chunk_plan"] = "4x14_sync"
        chunk_run_config["chunk_index"] = index
        chunk_run_config["chunk_id"] = spec.chunk_id
        chunk_run_config["chunk_qids"] = ordered_qids
        chunk_run_config["source_full_run_dir"] = str(source_run)
        chunk_run_config["source_full_batch_job_id"] = source_batch_job_id
        chunk_run_config["phase_a_lineage"] = "sliced_from_source_full_run"
        _write_json(run_dir / "run_config.json", chunk_run_config)

        chunk_summary = dict(summary)
        chunk_summary["run_id"] = run_dir.name
        chunk_summary["n_queries"] = len(ordered_qids)
        chunk_summary["control_type"] = "matched_norag_chunk"
        chunk_summary["chunk_plan"] = "4x14_sync"
        chunk_summary["chunk_index"] = index
        chunk_summary["chunk_id"] = spec.chunk_id
        chunk_summary["chunk_qids"] = ordered_qids
        chunk_summary["source_full_run_dir"] = str(source_run)
        chunk_summary["source_full_batch_job_id"] = source_batch_job_id
        chunk_summary["phase_a_lineage"] = "sliced_from_source_full_run"
        chunk_summary["note"] = (
            "Chunked synchronous no-RAG control built from the full matched no-RAG v2 generation artifacts. "
            "This chunk contributes to the merged 56-case control and can also be used as an early directional read. "
            "Phase A is reused by slicing the already-computed source full-run phase_a_metrics rows for the chunk qids."
        )
        _write_json(run_dir / "summary.json", chunk_summary)
        created_run_dirs.append(run_dir)

    return created_run_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare chunked no-RAG control runs from an existing full matched no-RAG run")
    parser.add_argument("--source-run", required=True, type=Path, help="Full matched no-RAG v2 run directory")
    parser.add_argument("--plan-json", required=True, type=Path, help="Chunk plan JSON file")
    parser.add_argument("--output-root", required=True, type=Path, help="Runs directory where chunk runs will be created")
    parser.add_argument("--run-prefix", required=True, help="Shared prefix for the created chunk run directories")
    parser.add_argument("--source-batch-job-id", required=True, help="Cancelled source batch job id to record in chunk metadata")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = prepare_chunked_control(
        source_run=args.source_run,
        plan_path=args.plan_json,
        output_root=args.output_root,
        run_prefix=args.run_prefix,
        source_batch_job_id=args.source_batch_job_id,
    )
    for path in run_dirs:
        print(path)


if __name__ == "__main__":
    main()
