from pipeline.failure_taxonomy import (
    CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY,
    build_failure_taxonomy,
    summarize_capability_caveats,
    summarize_failure_taxonomy,
)


def test_failure_taxonomy_labels_image_and_nonleish_failures():
    rag_rows = {
        "case-1::Q3_image_diagnosis": {
            "query_type": "Q3_image_diagnosis",
            "ground_truth_bucket": "leish",
            "query_image_count": 1,
            "prompt_context_count": 2,
            "context_image_count": 1,
            "support_image_tensor_count": 0,
            "generation_mode": "rag_prompt",
            "retrieval_support_status": "supported",
            "answer_format_valid": False,
            "diagnosis_accuracy": 0.0,
            "multimodal_faithfulness": 0.3,
            "answer_text": "The image shows cutaneous plaques with crusting.",
            "ordering_mode": "image_first",
            "diagnosis_family": "leish_family",
        },
        "case-2::Q1_diagnosis": {
            "query_type": "Q1_diagnosis",
            "ground_truth_bucket": "nonleish",
            "query_image_count": 0,
            "prompt_context_count": 3,
            "context_image_count": 0,
            "support_image_tensor_count": 0,
            "generation_mode": "rag_prompt",
            "retrieval_support_status": "weak_support",
            "answer_format_valid": True,
            "diagnosis_accuracy": 0.0,
            "multimodal_faithfulness": None,
            "answer_text": "Most likely cutaneous leishmaniasis.",
            "ordering_mode": "image_first",
            "diagnosis_family": "leish_family",
        },
    }
    norag_rows = {
        "case-1::Q3_image_diagnosis": {
            "query_type": "Q3_image_diagnosis",
            "ground_truth_bucket": "leish",
            "diagnosis_accuracy": 1.0,
            "diagnosis_family": "leish_family",
        },
        "case-2::Q1_diagnosis": {
            "query_type": "Q1_diagnosis",
            "ground_truth_bucket": "nonleish",
            "diagnosis_accuracy": 1.0,
            "diagnosis_family": "nonleish_family",
        },
    }

    records = build_failure_taxonomy(sorted(rag_rows), rag_rows, norag_rows)
    by_qid = {row["qid"]: row for row in records}

    assert "format_contract_failure" in by_qid["case-1::Q3_image_diagnosis"]["labels"]
    assert "unsupported_visual_claim" in by_qid["case-1::Q3_image_diagnosis"]["labels"]
    assert "retrieved_evidence_conflict" in by_qid["case-1::Q3_image_diagnosis"]["labels"]
    assert CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY not in by_qid["case-1::Q3_image_diagnosis"]["labels"]
    assert "nonleish_confusion_from_leish_context" in by_qid["case-2::Q1_diagnosis"]["labels"]

    summary = summarize_failure_taxonomy(records)
    assert summary["all"]["label_counts"]["retrieved_evidence_conflict"] == 1
    assert summary["nonleish"]["label_counts"]["nonleish_confusion_from_leish_context"] == 1

    caveats = summarize_capability_caveats(rag_rows.values())
    assert caveats["by_bucket"]["all"]["label_counts"][CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY] == 1
    assert caveats["by_bucket"]["leish"]["label_counts"][CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY] == 1


def test_failure_taxonomy_marks_no_issue_when_no_heuristic_triggers():
    rag_rows = {
        "case-3::Q1_diagnosis": {
            "query_type": "Q1_diagnosis",
            "ground_truth_bucket": "leish",
            "query_image_count": 0,
            "prompt_context_count": 2,
            "context_image_count": 0,
            "support_image_tensor_count": 0,
            "generation_mode": "rag_prompt",
            "retrieval_support_status": "supported",
            "answer_format_valid": True,
            "diagnosis_accuracy": 1.0,
            "multimodal_faithfulness": None,
            "answer_text": "Rank 1: Visceral leishmaniasis.",
            "ordering_mode": "image_first",
            "diagnosis_family": "leish_family",
        }
    }
    norag_rows = {
        "case-3::Q1_diagnosis": {
            "query_type": "Q1_diagnosis",
            "ground_truth_bucket": "leish",
            "diagnosis_accuracy": 1.0,
            "diagnosis_family": "leish_family",
        }
    }

    records = build_failure_taxonomy(["case-3::Q1_diagnosis"], rag_rows, norag_rows)
    assert records[0]["labels"] == ["no_issue_detected"]
