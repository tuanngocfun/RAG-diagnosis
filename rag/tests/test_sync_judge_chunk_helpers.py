import json

from pipeline.merge_sync_judge_chunks import merge_sync_judge_chunks
from pipeline.prepare_sync_judge_chunks import prepare_sync_judge_chunks


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _make_source_run(tmp_path):
    source = tmp_path / "source_run"
    source.mkdir()
    qids = [f"case_{i}::Q1" for i in range(1, 5)]
    answers = [{"qid": qid, "answer": f"answer-{qid}", "query": f"query-{qid}", "ground_truth": {"diagnosis_type": "VL"}} for qid in qids]
    retrieval = [{"qid": qid, "contexts": [{"doc_id": f"PMC{i}", "text": "ctx"}], "ground_truth": {"diagnosis_type": "VL"}} for i, qid in enumerate(qids, start=1)]
    queries = [{"case_id": f"case_{i}", "query_type": "Q1", "question": "q"} for i in range(1, 5)]

    _write_jsonl(source / "answers_rag_std.jsonl", answers)
    _write_jsonl(source / "answers_gemini.jsonl", answers)
    _write_jsonl(source / "answers_norag.jsonl", answers)
    _write_jsonl(source / "retrieval.jsonl", retrieval)
    _write_json(source / "queries.json", queries)
    _write_json(source / "run_config.json", {"run_id": source.name, "generator_model": "gemini-2.5-pro"})
    _write_json(source / "summary.json", {"run_id": source.name, "n_queries": 4})
    _write_json(source / "seed_sweep_config.json", {"run_id": source.name, "generator": "gemini"})
    snapshot_dir = tmp_path / "snapshot"
    snapshot_dir.mkdir()
    _write_json(snapshot_dir / "source_eval_manifest_snapshot.json", {"status": "running"})
    _write_json(snapshot_dir / "source_batch_job_snapshot.json", {"state": "PENDING"})
    return source, qids, snapshot_dir


def test_prepare_sync_judge_chunks_filters_rows_and_copies_snapshot(tmp_path):
    source, qids, snapshot_dir = _make_source_run(tmp_path)
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "chunks": [
                {"chunk_id": "chunk_a", "qids": qids[:2]},
                {"chunk_id": "chunk_b", "qids": qids[2:]},
            ]
        },
    )

    run_dirs = prepare_sync_judge_chunks(
        source_run=source,
        plan_path=plan,
        output_root=tmp_path,
        run_prefix="sync_chunks",
        answers_file="answers_rag_std.jsonl",
        copy_files=["answers_rag_std.jsonl", "retrieval.jsonl", "queries.json", "run_config.json", "summary.json", "seed_sweep_config.json"],
        source_batch_judge_run=source,
        source_batch_job_id="batches/original",
        snapshot_dir=snapshot_dir,
        sync_replacement_of=str(source),
        execution_mode="synchronous_judge_replacement",
        chunk_plan_label="4x14_sync",
        phase_a_lineage="recomputed_in_sync_replacement",
    )

    assert [path.name for path in run_dirs] == ["sync_chunks_chunk_a", "sync_chunks_chunk_b"]
    chunk_a_answers = [json.loads(line) for line in (run_dirs[0] / "answers_rag_std.jsonl").open()]
    assert [row["qid"] for row in chunk_a_answers] == qids[:2]
    assert not (run_dirs[0] / "phase_a_metrics.jsonl").exists()
    assert (run_dirs[0] / "batch_attempt_snapshot" / "source_batch_job_snapshot.json").exists()

    chunk_a_config = json.load((run_dirs[0] / "run_config.json").open())
    assert chunk_a_config["execution_mode"] == "synchronous_judge_replacement"
    assert chunk_a_config["chunk_plan"] == "4x14_sync"
    assert chunk_a_config["shared_chunk_plan_file"] == str(plan.resolve())
    assert chunk_a_config["phase_a_lineage"] == "recomputed_in_sync_replacement"


def test_merge_sync_judge_chunks_merges_phase_outputs_and_writes_manifest(tmp_path):
    source, qids, snapshot_dir = _make_source_run(tmp_path)

    chunk_a = tmp_path / "chunk_a"
    chunk_b = tmp_path / "chunk_b"
    chunk_a.mkdir()
    chunk_b.mkdir()
    _write_jsonl(chunk_a / "phase_a_metrics.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "phase_a_metrics.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])
    _write_jsonl(chunk_a / "phase_b_verified.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "phase_b_verified.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])
    _write_jsonl(chunk_a / "phase_c_pseudolabel.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "phase_c_pseudolabel.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])
    _write_jsonl(chunk_a / "ragas.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "ragas.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])

    merged = merge_sync_judge_chunks(
        source_run=source,
        output_run=tmp_path / "merged",
        answers_file="answers_rag_std.jsonl",
        copy_files=["answers_rag_std.jsonl", "retrieval.jsonl", "queries.json", "run_config.json", "summary.json", "seed_sweep_config.json"],
        chunk_runs=[chunk_a, chunk_b],
        source_batch_judge_run=source,
        source_batch_job_id="batches/original",
        snapshot_dir=snapshot_dir,
        sync_replacement_of=str(source),
        execution_mode="synchronous_judge_replacement",
        chunk_plan_label="4x14_sync",
        phase_a_lineage="merged_from_recomputed_chunk_phase_a",
        judge_model="gemini-2.5-pro",
        shared_chunk_plan_file="/tmp/plan.json",
    )

    merged_rows = [json.loads(line) for line in (merged / "ragas.jsonl").open()]
    assert [row["qid"] for row in merged_rows] == qids
    manifest = json.load((merged / "eval_manifest.json").open())
    assert manifest["judge_model"] == "gemini-2.5-pro"
    assert manifest["phases"]["phase_b_verified"]["status"] == "completed"
    merged_config = json.load((merged / "run_config.json").open())
    assert merged_config["merged_from_chunk_runs"] == [str(chunk_a.resolve()), str(chunk_b.resolve())]
    assert merged_config["phase_a_lineage"] == "merged_from_recomputed_chunk_phase_a"
    assert (merged / "batch_attempt_snapshot" / "source_eval_manifest_snapshot.json").exists()
