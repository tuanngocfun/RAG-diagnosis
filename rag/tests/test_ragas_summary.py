import json

from pipeline.ragas_summary import update_summary_with_ragas_metrics


def test_update_summary_with_ragas_metrics_persists_reasoning_recall_metadata(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    summary = {
        "metrics": {"grounded_accuracy": None},
        "metrics_verified": {"mrr": 0.5},
        "metrics_pseudolabel": {"mrr": 0.4},
    }
    with open(run_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f)

    rows = [
        {
            "qid": "PMC1::Q1_diagnosis",
            "multimodal_faithfulness": 1.0,
            "multimodal_relevance": 0.5,
            "context_relevance": 0.25,
            "diagnosis_accuracy": 1.0,
            "diagnosis_type_accuracy": 0.5,
            "diagnosis_family_accuracy": 1.0,
            "l3_top1_correct": 1.0,
            "top3_hit": 1.0,
            "diagnosis_accuracy_pseudolabel": 0.5,
            "diagnosis_type_accuracy_pseudolabel": 0.5,
            "l3_top1_correct_pseudolabel": 1.0,
            "top3_hit_pseudolabel": 1.0,
            "reasoning_recall": 0.5,
            "reasoning_recall_method": "llm_judge",
            "reasoning_recall_judge_model": "gemini-2.5-pro",
            "reasoning_recall_source_id": "p14_v7_reaudit_shared56",
            "reasoning_recall_source_path": "/tmp/v7.jsonl",
            "traces": {"grounded_accuracy": 0.75},
        },
        {
            "qid": "PMC2::Q1_diagnosis",
            "multimodal_faithfulness": 0.0,
            "multimodal_relevance": 0.25,
            "context_relevance": 0.5,
            "diagnosis_accuracy": 0.0,
            "diagnosis_type_accuracy": 0.0,
            "diagnosis_family_accuracy": 0.0,
            "l3_top1_correct": 0.0,
            "top3_hit": 0.0,
            "diagnosis_accuracy_pseudolabel": 0.0,
            "diagnosis_type_accuracy_pseudolabel": 0.0,
            "l3_top1_correct_pseudolabel": 0.0,
            "top3_hit_pseudolabel": 0.0,
            "reasoning_recall": None,
            "reasoning_recall_method": "skipped_missing_groundtruth_reasoning",
            "reasoning_recall_judge_model": "",
            "reasoning_recall_source_id": "",
            "reasoning_recall_source_path": "",
            "traces": {"grounded_accuracy": 0.0},
        },
    ]
    with open(run_dir / "ragas.jsonl", "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    updated = update_summary_with_ragas_metrics(run_dir)

    assert updated["ragas_metrics_verified"]["reasoning_recall"] == 0.5
    assert updated["ragas_metrics_verified"]["reasoning_recall_coverage"] == 0.5
    assert updated["ragas_metrics_verified"]["reasoning_recall_missing_groundtruth_count"] == 1
    assert updated["ragas_metrics_verified"]["reasoning_recall_judge_models"] == ["gemini-2.5-pro"]
    assert updated["ragas_metrics_verified"]["reasoning_recall_source_ids"] == ["p14_v7_reaudit_shared56"]
    assert updated["ragas_metrics_pseudolabel"]["diagnosis_accuracy"] == 0.25
    assert updated["metrics"]["grounded_accuracy"] == 0.375
    assert updated["metrics_verified"]["grounded_accuracy"] == 0.375
