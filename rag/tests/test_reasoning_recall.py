import json

from pipeline.reasoning_recall import (
    extract_case_id_from_qid,
    load_reasoning_source_map,
    normalize_reasoning_recall_result,
    parse_numbered_reasoning_points,
    parse_predicted_reasoning_steps,
    resolve_groundtruth_payload,
    resolve_groundtruth_points,
)


def test_parse_numbered_reasoning_points_extracts_lines():
    raw = """1. Fever and splenomegaly support VL\n2) Cytopenia supports systemic infection\n3: Travel exposure increases likelihood"""
    points = parse_numbered_reasoning_points(raw)
    assert points == [
        "Fever and splenomegaly support VL",
        "Cytopenia supports systemic infection",
        "Travel exposure increases likelihood",
    ]


def test_parse_predicted_reasoning_steps_filters_headers():
    answer = """## DIAGNOSIS PREDICTION\n**Primary Diagnosis:** Visceral Leishmaniasis\n\n## SUPPORTING EVIDENCE\n- Prolonged fever and splenomegaly\n- Pancytopenia and weight loss\n\n## DIFFERENTIAL CONSIDERATIONS\n1. Hematologic malignancy\n2. Visceral Leishmaniasis"""
    steps = parse_predicted_reasoning_steps(answer)
    assert "Prolonged fever and splenomegaly" in steps
    assert "Pancytopenia and weight loss" in steps
    assert all("Primary Diagnosis" not in step for step in steps)


def test_load_reasoning_source_map_and_resolve(tmp_path):
    source = tmp_path / "source.jsonl"
    row = {
        "case_id": "PMC123_01",
        "prompt1_converted_case": {
            "diagnostic_reasoning": "1. Point A\n2. Point B"
        },
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    source_map = load_reasoning_source_map([source])
    points, path = resolve_groundtruth_points("PMC123_01::Q1_diagnosis", source_map)

    assert extract_case_id_from_qid("PMC123_01::Q1_diagnosis") == "PMC123_01"
    assert points == ["Point A", "Point B"]
    assert path.endswith("source.jsonl")


def test_resolve_groundtruth_payload_includes_source_id(tmp_path):
    source = tmp_path / "legacy_process14_reasoning_source_scaffold.jsonl"
    row = {
        "case_id": "PMC123_01",
        "prompt1_converted_case": {
            "diagnostic_reasoning": "1. Point A\n2. Point B"
        },
    }
    source.write_text(json.dumps(row) + "\n", encoding="utf-8")

    source_map = load_reasoning_source_map([source])
    payload = resolve_groundtruth_payload("PMC123_01::Q1_diagnosis", source_map)

    assert payload["groundtruth_points"] == ["Point A", "Point B"]
    assert payload["source_id"] == "legacy_process14_reasoning_source_scaffold"
    assert payload["source_path"].endswith("legacy_process14_reasoning_source_scaffold.jsonl")


def test_load_reasoning_source_map_respects_source_path_order(tmp_path):
    first = tmp_path / "first_source.jsonl"
    second = tmp_path / "second_source.jsonl"
    first.write_text(
        json.dumps(
            {
                "case_id": "PMC999_01",
                "prompt1_converted_case": {"diagnostic_reasoning": "1. First source"},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "case_id": "PMC999_01",
                "prompt1_converted_case": {"diagnostic_reasoning": "1. Second source"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    source_map = load_reasoning_source_map([first, second])
    payload = resolve_groundtruth_payload("PMC999_01::Q1_diagnosis", source_map)

    assert payload["groundtruth_points"] == ["First source"]
    assert payload["source_path"].endswith("first_source.jsonl")


def test_normalize_reasoning_recall_result_uses_groundtruth_length():
    gt_points = ["A", "B", "C"]
    parsed = {
        "matched_groundtruth_indices": [1, 3, 99],
        "explanation": "Matched 1 and 3",
    }
    normalized = normalize_reasoning_recall_result(parsed, gt_points)

    assert normalized["matched_groundtruth_indices"] == [1, 3]
    assert normalized["matched_groundtruth_count"] == 2
    assert normalized["groundtruth_count"] == 3
    assert normalized["recall"] == 2 / 3
    assert normalized["unmatched_groundtruth_points"] == ["B"]
