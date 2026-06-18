import json

from configs.prompt_mode import PromptMode
from pipeline.run_seed_sweep import run_seed_sweep


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_run_seed_sweep_records_explicit_model_pin(tmp_path, monkeypatch):
    base_run = tmp_path / "base_run"
    base_run.mkdir()
    _write_jsonl(base_run / "retrieval.jsonl", [{"qid": "q1", "query": "query", "contexts": []}])
    _write_json(base_run / "queries.json", [{"qid": "q1"}])
    _write_json(
        base_run / "run_config.json",
        {
            "run_id": "base",
            "retrieval_top_k": 12,
            "experiment_controls": {"retrieval_top_k": 12},
        },
    )
    _write_json(base_run / "summary.json", {"run_id": "base"})

    def fake_generate_answers(run_dir, **kwargs):
        with open(run_dir / kwargs["output_file"], "w", encoding="utf-8") as f:
            f.write("")
        return run_dir / kwargs["output_file"]

    monkeypatch.setattr("pipeline.run_seed_sweep.generate_answers", fake_generate_answers)

    out_dirs = run_seed_sweep(
        base_run_dir=base_run,
        seeds=[42],
        generator="gemini",
        model="gemini-2.5-pro",
        variant="4b",
        prompt_mode=PromptMode.BALANCED,
        force_rag=True,
        output_file="answers_rag_std.jsonl",
        delay=0.0,
        evaluate=False,
        judge_model=None,
        use_batch_api=False,
        batch_poll_seconds=10.0,
        batch_timeout_seconds=3600,
        judge_batch_api=False,
        judge_batch_poll_seconds=10.0,
        judge_batch_timeout_seconds=7200,
        eval_resume=False,
        evaluate_retrieval_metrics=True,
        strip_query_images=False,
        ablation_scope="",
        context_k=5,
        ordering_mode="image_first",
        use_context_image_tensors=False,
        support_image_tensor_budget=0,
    )

    config = json.load((out_dirs[0] / "seed_sweep_config.json").open())
    assert config["generator"] == "gemini"
    assert config["model"] == "gemini-2.5-pro"
    assert config["force_rag"] is True
    assert config["context_k"] == 5
    assert config["ordering_mode"] == "image_first"
    assert config["retrieval_top_k_inherited"] == 12
    assert config["retrieval_top_k_inherited_from_base_run"] is True
