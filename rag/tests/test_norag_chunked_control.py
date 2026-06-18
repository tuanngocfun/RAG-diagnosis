import json

from pipeline.merge_norag_chunked_control import merge_chunked_control
from pipeline.prepare_norag_chunked_control import prepare_chunked_control


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
    answers = [{"qid": qid, "answer": f"answer-{qid}", "query": f"query-{qid}"} for qid in qids]
    answers_gemini = [{"qid": qid, "answer": f"answer-{qid}", "generation_mode": "norag_prompt"} for qid in qids]
    retrieval = [{"qid": qid, "contexts": [], "control_type": "matched_norag"} for qid in qids]
    phase_a = [{"qid": qid, "phase": "phase_a_metrics"} for qid in qids]
    queries = [{"case_id": f"case_{i}", "query_type": "Q1", "question": "q"} for i in range(1, 5)]

    _write_jsonl(source / "answers_norag.jsonl", answers)
    _write_jsonl(source / "answers_gemini.jsonl", answers_gemini)
    _write_jsonl(source / "retrieval.jsonl", retrieval)
    _write_jsonl(source / "phase_a_metrics.jsonl", phase_a)
    _write_json(source / "queries.json", queries)
    _write_json(
        source / "run_config.json",
        {
            "run_id": source.name,
            "control_type": "matched_norag",
            "prompt_contract_version": "norag_matched_v2_query_guardrail",
        },
    )
    _write_json(source / "summary.json", {"run_id": source.name, "n_queries": 4})
    return source, qids


def test_prepare_chunked_control_filters_rows_and_records_metadata(tmp_path):
    source, qids = _make_source_run(tmp_path)
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

    run_dirs = prepare_chunked_control(
        source_run=source,
        plan_path=plan,
        output_root=tmp_path,
        run_prefix="chunked_sync",
        source_batch_job_id="batches/original",
    )

    assert [path.name for path in run_dirs] == ["chunked_sync_chunk_a", "chunked_sync_chunk_b"]
    chunk_a_answers = [json.loads(line) for line in (run_dirs[0] / "answers_norag.jsonl").open()]
    assert [row["qid"] for row in chunk_a_answers] == qids[:2]

    chunk_a_config = json.load((run_dirs[0] / "run_config.json").open())
    assert chunk_a_config["control_type"] == "matched_norag_chunk"
    assert chunk_a_config["chunk_plan"] == "4x14_sync"
    assert chunk_a_config["chunk_index"] == 1
    assert chunk_a_config["chunk_id"] == "chunk_a"
    assert chunk_a_config["source_full_batch_job_id"] == "batches/original"
    assert chunk_a_config["phase_a_lineage"] == "sliced_from_source_full_run"


def test_merge_chunked_control_rebuilds_full_qid_order(tmp_path):
    source, qids = _make_source_run(tmp_path)

    chunk_a = tmp_path / "chunk_a"
    chunk_b = tmp_path / "chunk_b"
    chunk_a.mkdir()
    chunk_b.mkdir()
    _write_jsonl(chunk_a / "ragas.jsonl", [{"qid": qids[0], "score": 1}, {"qid": qids[2], "score": 3}])
    _write_jsonl(chunk_b / "ragas.jsonl", [{"qid": qids[1], "score": 2}, {"qid": qids[3], "score": 4}])
    _write_jsonl(chunk_a / "phase_b_verified.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "phase_b_verified.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])
    _write_jsonl(chunk_a / "phase_c_pseudolabel.jsonl", [{"qid": qids[0]}, {"qid": qids[2]}])
    _write_jsonl(chunk_b / "phase_c_pseudolabel.jsonl", [{"qid": qids[1]}, {"qid": qids[3]}])

    merged = merge_chunked_control(
        source_run=source,
        chunk_runs=[chunk_a, chunk_b],
        output_run=tmp_path / "merged",
        source_batch_job_id="batches/original",
    )

    merged_rows = [json.loads(line) for line in (merged / "ragas.jsonl").open()]
    assert [row["qid"] for row in merged_rows] == qids
    merged_config = json.load((merged / "run_config.json").open())
    assert merged_config["control_type"] == "matched_norag_chunked_sync"
    assert merged_config["chunk_plan"] == "4x14_sync"
    assert merged_config["source_full_batch_job_id"] == "batches/original"
    assert merged_config["phase_a_lineage"] == "reused_from_source_full_run"
    assert merged_config["merged_from_chunk_runs"] == [str(chunk_a.resolve()), str(chunk_b.resolve())]
