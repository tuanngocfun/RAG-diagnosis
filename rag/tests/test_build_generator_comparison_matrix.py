import json

from pipeline.build_generator_comparison_matrix import build_generator_comparison_matrix


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _seed_sweep_generator_fields(generator_model):
    if "gemini" in generator_model:
        return {"generator": "gemini", "model": generator_model, "variant": "4b"}
    if "medgemma" in generator_model:
        return {"generator": "medgemma", "model": None, "variant": "4b"}
    if "gemma-3" in generator_model:
        return {"generator": "gemma3", "model": generator_model, "variant": "4b"}
    raise AssertionError(f"Unsupported test generator_model: {generator_model}")


def _make_run(tmp_path, name, *, is_rag, generator_model, prompt_mode, prompt_contract_version, control_type=None):
    run_dir = tmp_path / name
    run_dir.mkdir()
    ragas_rows = [
        {
            "qid": "leish::Q1",
            "diagnosis_accuracy": 1.0,
            "diagnosis_type_accuracy": 1.0,
            "ground_truth": {"diagnosis_type": "VL"},
        },
        {
            "qid": "nonleish::Q1",
            "diagnosis_accuracy": 0.0,
            "diagnosis_type_accuracy": 0.0,
            "ground_truth": {"diagnosis_type": "Non-Leishmaniasis"},
        },
    ]
    answer_rows = [
        {
            "qid": "leish::Q1",
            "answer": "answer",
            "ground_truth": {"diagnosis_type": "VL"},
            "generation_mode": "rag_prompt" if is_rag else "norag_prompt",
            "contexts": [{"doc_id": "PMC1"}] if is_rag else [],
        },
        {
            "qid": "nonleish::Q1",
            "answer": "answer",
            "ground_truth": {"diagnosis_type": "Non-Leishmaniasis"},
            "generation_mode": "rag_prompt" if is_rag else "norag_prompt",
            "contexts": [{"doc_id": "PMC2"}] if is_rag else [],
        },
    ]
    _write_jsonl(run_dir / "ragas.jsonl", ragas_rows)
    if is_rag:
        _write_jsonl(run_dir / "answers_rag_std.jsonl", answer_rows)
        seed_payload = _seed_sweep_generator_fields(generator_model)
        _write_json(
            run_dir / "seed_sweep_config.json",
            {
                **seed_payload,
                "prompt_mode": prompt_mode,
                "force_rag": True,
            },
        )
    else:
        _write_jsonl(run_dir / "answers_norag.jsonl", answer_rows)
        _write_jsonl(run_dir / "answers_gemini.jsonl", answer_rows)
        _write_json(
            run_dir / "run_config.json",
            {
                "generator_model": generator_model,
                "prompt_mode": prompt_mode,
                "prompt_contract_version": prompt_contract_version,
                "control_type": control_type,
                "is_rag": False,
            },
        )
    _write_json(
        run_dir / "eval_manifest.json",
        {
            "judge_model": "gemini-2.5-pro",
            "answers_file": "answers_rag_std.jsonl" if is_rag else "answers_gemini.jsonl",
        },
    )
    return run_dir


def test_build_generator_comparison_matrix_writes_six_arm_report(tmp_path, monkeypatch):
    monkeypatch.setattr("pipeline.build_generator_comparison_matrix.RUNS_DIR", tmp_path)

    med_rag = _make_run(
        tmp_path,
        "med_rag",
        is_rag=True,
        generator_model="google/medgemma-4b-it",
        prompt_mode="PromptMode.BALANCED",
        prompt_contract_version=None,
    )
    med_norag = _make_run(
        tmp_path,
        "med_norag",
        is_rag=False,
        generator_model="google/medgemma-4b-it",
        prompt_mode="no_context",
        prompt_contract_version="norag_matched_v2_query_guardrail",
        control_type="matched_norag_chunked_sync",
    )
    gem_rag = _make_run(
        tmp_path,
        "gem_rag",
        is_rag=True,
        generator_model="gemini-2.5-pro",
        prompt_mode="PromptMode.BALANCED",
        prompt_contract_version=None,
    )
    gem_norag = _make_run(
        tmp_path,
        "gem_norag",
        is_rag=False,
        generator_model="gemini-2.5-pro",
        prompt_mode="no_context",
        prompt_contract_version="norag_matched_v2_query_guardrail",
        control_type="matched_norag",
    )
    gemma_rag = _make_run(
        tmp_path,
        "gemma_rag",
        is_rag=True,
        generator_model="google/gemma-3-4b-it",
        prompt_mode="PromptMode.BALANCED",
        prompt_contract_version=None,
    )
    gemma_norag = _make_run(
        tmp_path,
        "gemma_norag",
        is_rag=False,
        generator_model="google/gemma-3-4b-it",
        prompt_mode="no_context",
        prompt_contract_version="norag_matched_v2_query_guardrail",
        control_type="matched_norag",
    )

    metric_deltas = {
        "all": {"diagnosis_accuracy": 0.1},
        "leish": {"diagnosis_accuracy": 0.2},
        "nonleish": {"diagnosis_accuracy": -0.1},
    }
    med_compare = tmp_path / "med_compare.json"
    gem_compare = tmp_path / "gem_compare.json"
    gemma_compare = tmp_path / "gemma_compare.json"
    _write_json(med_compare, {"metric_deltas": metric_deltas})
    _write_json(gem_compare, {"metric_deltas": metric_deltas})
    _write_json(gemma_compare, {"metric_deltas": metric_deltas})

    output = build_generator_comparison_matrix(
        medgemma_rag=str(med_rag),
        medgemma_norag=str(med_norag),
        medgemma_compare=str(med_compare),
        gemini_rag=str(gem_rag),
        gemini_norag=str(gem_norag),
        gemini_compare=str(gem_compare),
        gemma3_rag=str(gemma_rag),
        gemma3_norag=str(gemma_norag),
        gemma3_compare=str(gemma_compare),
    )

    report = json.load(output.open())
    assert len(report["arms"]) == 6
    gemini_rag_arm = next(arm for arm in report["arms"] if arm["label"] == "Gemini RAG")
    gemma_rag_arm = next(arm for arm in report["arms"] if arm["label"] == "Gemma 3 4B RAG")
    assert gemini_rag_arm["generator_model"] == "gemini-2.5-pro"
    assert gemini_rag_arm["same_family_generator_judge"] is True
    assert gemma_rag_arm["generator_model"] == "google/gemma-3-4b-it"
    assert gemma_rag_arm["same_family_generator_judge"] is False
    assert "gemma3" in report["within_model_deltas"]
    assert output.with_suffix(".md").exists()
