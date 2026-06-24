import json
from pathlib import Path

import pytest

from pipeline.pseudolabel_adapter import build_pseudolabel_artifacts, get_pseudolabel_artifact_paths


P14_TRAIN_RESULTS = Path(
    "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/"
    "p14_train_structure_only_main_v7_batch_single_repair/results.jsonl"
)
P14_TEST_RESULTS = Path(
    "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/"
    "p14_test_held_out_structure_only_v7_local_reaudit/results.jsonl"
)


def test_get_pseudolabel_artifact_paths_default_names():
    paths = get_pseudolabel_artifact_paths("", dataset_version="v163_pseudolabel_v2")
    assert Path(paths["train"]).name == "train_pseudolabel_v2_normalized.jsonl"
    assert Path(paths["test"]).name == "test_pseudolabel_v2_normalized.jsonl"
    assert Path(paths["query"]).name == "eval_queries_v163_pseudolabel_v2.jsonl"
    assert Path(paths["query_mixed56"]).name == "eval_queries_v163_mixed56.jsonl"
    assert Path(paths["qrels_verified"]).name == "qrels_pseudolabel_verified.json"
    assert Path(paths["qrels_pseudolabel"]).name == "qrels_pseudolabel_label.json"


def test_get_pseudolabel_artifact_paths_with_suffix():
    suffix = "v7_process14"
    paths = get_pseudolabel_artifact_paths(suffix, dataset_version="v163_pseudolabel_v2")
    assert Path(paths["train"]).name == "train_pseudolabel_v2_normalized_v7_process14.jsonl"
    assert Path(paths["test"]).name == "test_pseudolabel_v2_normalized_v7_process14.jsonl"
    assert Path(paths["query"]).name == "eval_queries_v163_pseudolabel_v2_v7_process14.jsonl"
    assert Path(paths["query_mixed56"]).name == "eval_queries_v163_mixed56_v7_process14.jsonl"
    assert Path(paths["qrels_verified"]).name == "qrels_pseudolabel_verified_v7_process14.json"
    assert Path(paths["qrels_pseudolabel"]).name == "qrels_pseudolabel_label_v7_process14.json"


def test_get_pseudolabel_artifact_paths_p14_v7_names():
    paths = get_pseudolabel_artifact_paths(dataset_version="p14_v7")
    assert Path(paths["train"]).name == "train_p14_v7_normalized.jsonl"
    assert Path(paths["test"]).name == "test_p14_v7_normalized.jsonl"
    assert Path(paths["query"]).name == "eval_queries_p14_v7.jsonl"
    assert Path(paths["query_mixed56"]).name == "eval_queries_p14_v7_mixed56.jsonl"
    assert Path(paths["qrels_verified"]).name == "qrels_p14_v7_verified.json"
    assert Path(paths["qrels_pseudolabel"]).name == "qrels_p14_v7_pseudolabel.json"


def test_build_p14_v7_artifacts_counts(tmp_path):
    if not P14_TRAIN_RESULTS.exists() or not P14_TEST_RESULTS.exists():
        pytest.skip("external P14 v7 structure-only fixture files are not available")

    stats = build_pseudolabel_artifacts(
        force=True,
        train_results_path=P14_TRAIN_RESULTS,
        test_results_path=P14_TEST_RESULTS,
        dataset_version="p14_v7",
        output_dir=tmp_path,
    )

    assert stats.dataset_version == "p14_v7"
    assert stats.train_rows == 106
    assert stats.test_rows == 56
    assert stats.query_rows == 136
    assert stats.query_mixed56_rows == 56

    query_counts = {}
    with open(stats.query_path, "r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            query_type = row["query_type"]
            query_counts[query_type] = query_counts.get(query_type, 0) + 1
            assert row["label_contract"]["dataset_version"] == "p14_v7"

    assert query_counts == {
        "Q1_diagnosis": 56,
        "Q3_image_diagnosis": 24,
        "Q1_Q3_multimodal_diagnosis": 56,
    }
