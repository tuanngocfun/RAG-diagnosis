import json

from pipeline import compare_rag_norag


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_compare_rag_norag_prefers_generation_sidecar_and_separates_capability_caveats(tmp_path, monkeypatch):
    monkeypatch.setattr(compare_rag_norag, "RUNS_DIR", tmp_path)

    base_run = tmp_path / "base_run"
    rag_dir = tmp_path / "rag_run"
    norag_dir = tmp_path / "norag_run"
    base_run.mkdir()
    rag_dir.mkdir()
    norag_dir.mkdir()

    train_jsonl = tmp_path / "train_phase1b.jsonl"
    _write_jsonl(
        train_jsonl,
        [
            {"case_id": "PMC1", "diagnosis_type": "VL", "is_leishmaniasis": True},
            {"case_id": "PMC2", "diagnosis_type": "Non-Leishmaniasis", "is_leishmaniasis": False},
        ],
    )
    _write_json(
        base_run / "run_config.json",
        {
            "run_id": "base_run",
            "runtime_metadata": {"train_jsonl": str(train_jsonl)},
        },
    )
    _write_jsonl(
        base_run / "retrieval.jsonl",
        [
            {
                "qid": "case-1::Q3_image_diagnosis",
                "contexts": [
                    {"doc_id": "PMC1", "text": "Biopsy confirmed amastigotes in tissue."},
                    {"doc_id": "PMC2", "text": "Non-leish differential remained possible."},
                ],
            }
        ],
    )

    rag_rows = [
        {
            "qid": "case-1::Q3_image_diagnosis",
            "diagnosis_accuracy": 0.0,
            "diagnosis_type_accuracy": 0.0,
            "top3_hit": 0.0,
            "l3_top1_correct": 0.0,
            "reasoning_recall": 0.25,
            "reasoning_recall_method": "llm_judge_fallback_model",
            "reasoning_recall_judge_model": "gemini-2.5-flash",
            "reasoning_recall_source_id": "p14_v7_reaudit_shared56",
            "reasoning_recall_source_path": "/tmp/v7.jsonl",
            "ground_truth": {"diagnosis_type": "VL"},
        }
    ]
    norag_rows = [
        {
            "qid": "case-1::Q3_image_diagnosis",
            "diagnosis_accuracy": 1.0,
            "diagnosis_type_accuracy": 1.0,
            "top3_hit": 1.0,
            "l3_top1_correct": 1.0,
            "reasoning_recall": 0.5,
            "reasoning_recall_method": "llm_judge",
            "reasoning_recall_judge_model": "gemini-2.5-pro",
            "reasoning_recall_source_id": "p14_v7_reaudit_shared56",
            "reasoning_recall_source_path": "/tmp/v7.jsonl",
            "ground_truth": {"diagnosis_type": "VL"},
        }
    ]
    rag_answers = [
        {
            "qid": "case-1::Q3_image_diagnosis",
            "query_type": "Q3_image_diagnosis",
            "answer": "Rank 1: Cutaneous leishmaniasis.",
            "contexts": [{"doc_id": "PMC1"}],
            "generation_mode": "rag_prompt",
            "retrieval_support_status": "supported",
            "ground_truth": {"diagnosis_type": "VL"},
            "prompt_context_count": 1,
            "prompt_context_doc_ids": ["PMC1"],
            "query_images": ["query.jpg"],
            "context_images": ["support.jpg"],
            "query_image_tensor_attempt_count": 1,
            "support_image_tensor_attempt_count": 0,
            "query_image_tensor_count": 1,
            "support_image_tensor_count": 0,
            "image_tensor_fallback_used": False,
            "image_tensor_fallback_reason": "",
            "use_context_image_tensors": False,
            "answer_format_valid": True,
            "answer_format_error": "",
            "diagnosis_family": "leish_family",
        }
    ]
    norag_answers = [
        {
            "qid": "case-1::Q3_image_diagnosis",
            "query_type": "Q3_image_diagnosis",
            "answer": "Rank 1: Cutaneous leishmaniasis.",
            "contexts": [],
            "generation_mode": "norag_prompt",
            "ground_truth": {"diagnosis_type": "VL"},
            "prompt_context_count": 0,
            "query_images": ["query.jpg"],
            "context_images": [],
            "query_image_tensor_attempt_count": 1,
            "support_image_tensor_attempt_count": 0,
            "query_image_tensor_count": 1,
            "support_image_tensor_count": 0,
            "image_tensor_fallback_used": False,
            "image_tensor_fallback_reason": "",
            "answer_format_valid": True,
            "answer_format_error": "",
            "diagnosis_family": "leish_family",
        }
    ]

    _write_jsonl(rag_dir / "ragas.jsonl", rag_rows)
    _write_jsonl(norag_dir / "ragas.jsonl", norag_rows)
    _write_jsonl(rag_dir / "answers_rag_std.jsonl", rag_answers)
    _write_jsonl(norag_dir / "answers_norag.jsonl", norag_answers)
    _write_jsonl(norag_dir / "answers_gemini.jsonl", norag_answers)
    _write_json(rag_dir / "seed_sweep_config.json", {"base_run": str(base_run)})

    _write_json(
        rag_dir / "answer_generation_contract.json",
        {
            "is_rag": True,
            "generator_model": "google/gemma-4-E4B-it",
            "prompt_mode_requested": "balanced",
            "prompt_contract_version": "mixed",
            "prompt_contract_notes": "multiple_prompt_contracts",
            "prompt_contract_summary": {
                "prompt_contract_version": "mixed",
                "prompt_contract_notes": "multiple_prompt_contracts",
                "mixed_contracts": True,
                "by_version": {
                    "rag_balanced_image_grounding_v1": {"count": 1, "notes": "balanced", "image_grounding_required_count": 1}
                },
            },
            "ordering_mode": "image_first",
            "use_context_image_tensors": False,
            "support_image_tensor_budget": 0,
            "multimodal_usage_summary": {
                "rows_with_query_images": 1,
                "rows_with_context_images_available": 1,
                "rows_with_query_image_tensors": 1,
                "rows_with_support_image_tensors": 0,
                "mean_query_image_tensor_count": 1.0,
                "mean_support_image_tensor_count": 0.0,
                "image_tensor_fallback_count": 0,
                "true_multimodal_support_active": False,
            },
        },
    )
    _write_json(
        norag_dir / "run_config.json",
        {
            "is_rag": False,
            "control_type": "matched_norag",
            "prompt_mode": "no_context",
            "prompt_contract_version": "norag_matched_v2_query_guardrail",
            "prompt_contract_notes": "matched_norag",
        },
    )

    output_path = compare_rag_norag.compare_rag_norag(str(rag_dir), str(norag_dir))
    report = json.load(output_path.open("r", encoding="utf-8"))

    assert report["rag_prompt_contract"]["prompt_contract_version"] == "mixed"
    assert report["rag_prompt_contract"]["answer_generation_contract_source"].endswith("answer_generation_contract.json")
    assert "failure_taxonomy_summary" in report
    assert "capability_caveats" in report
    assert "contamination_summary" in report
    assert "modality_use_summary" in report
    assert "confirmatory_evidence_summary" in report
    assert report["capability_caveats"]["by_bucket"]["all"]["label_counts"]["support_images_used_as_prompt_references_only"] == 1
    assert report["details"][0]["query_type"] == "Q3_image_diagnosis"
    assert report["details"][0]["rag_prompt_context_count"] == 1
    assert report["details"][0]["rag_query_image_count"] == 1
    assert report["details"][0]["rag_query_image_tensor_count"] == 1
    assert report["details"][0]["rag_support_image_tensor_count"] == 0
    assert report["details"][0]["rag_use_context_image_tensors"] is False
    assert report["details"][0]["rag_reasoning_recall_method"] == "llm_judge_fallback_model"
    assert report["details"][0]["norag_reasoning_recall_judge_model"] == "gemini-2.5-pro"
    assert report["details"][0]["rag_reasoning_recall_source_id"] == "p14_v7_reaudit_shared56"
    assert report["details"][0]["retrieved_leish_context_count"] == 1
    assert report["details"][0]["retrieved_nonleish_context_count"] == 1
    assert report["details"][0]["confirmatory_in_topk"] is True
    assert report["details"][0]["confirmatory_in_prompt_context"] is True
    assert report["details"][0]["confirmatory_dropped_by_pruning"] is False
    assert report["metric_deltas"]["all"]["reasoning_recall"] == -0.25
    assert report["rag_summary"]["all"]["reasoning_recall_method_counts"]["llm_judge_fallback_model"] == 1
    assert report["norag_summary"]["all"]["reasoning_recall_source_id_counts"]["p14_v7_reaudit_shared56"] == 1
    assert report["contamination_summary"]["by_bucket"]["all"]["mean_retrieved_leish_context_count"] == 1.0
    assert report["contamination_summary"]["by_bucket"]["all"]["mean_retrieved_nonleish_context_count"] == 1.0
    assert report["modality_use_summary"]["by_bucket"]["all"]["ignored_query_image_count"] == 1
    assert report["confirmatory_evidence_summary"]["by_bucket"]["all"]["confirmatory_in_topk_count"] == 1
    assert report["rag_modality_status"]["true_multimodal_support_active"] is False
    assert report["rag_modality_status"]["rows_with_support_image_tensors"] == 0
    assert report["details"][0]["rag_query_image_tensor_attempt_count"] == 1
    assert report["details"][0]["rag_support_image_tensor_attempt_count"] == 0
    assert report["details"][0]["rag_image_tensor_fallback_used"] is False
    markdown_path = output_path.with_suffix(".md")
    assert markdown_path.exists()
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "reasoning_recall: rag=0.2500 norag=0.5000 delta=-0.2500" in markdown
