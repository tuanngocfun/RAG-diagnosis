from pipeline.phase1a_pre_gate import (
    _build_adaptive_vs_pure_rag_delta,
    _build_usage_summary,
    _build_watchlist_comparison,
)


def test_usage_summary_counts_augmented_hits_from_contexts():
    run_snapshot = {
        "run_dir": "/tmp/revised",
        "summary": {"metrics": {"mrr": 0.5, "ndcg": {"@10": 0.6}, "map": 0.4, "precision": {"@5": 0.2}, "recall": {"@5": 0.3}}},
        "retrieval_rows": {
            "Q_NONLEISH": {
                "qid": "Q_NONLEISH",
                "contexts": [
                    {"doc_id": "AUG1", "score": 0.12},
                    {"doc_id": "BASE1", "score": 0.11},
                ],
                "results": [
                    {"doc_id": "BASE1", "score": 0.99},
                ],
            }
        },
        "answer_rows": {
            "Q_NONLEISH": {
                "qid": "Q_NONLEISH",
                "prompt_context_doc_ids": ["AUG1"],
                "ground_truth_bucket": "nonleish",
            }
        },
    }

    summary = _build_usage_summary(run_snapshot, {"AUG1"})

    assert summary["augmented_retrieval_hit_count"] == 1
    assert summary["distinct_augmented_docs_retrieved"] == 1
    assert summary["augmented_prompt_context_hit_count"] == 1
    assert summary["distinct_augmented_docs_in_prompt"] == 1
    assert summary["nonleish_prompt_context_hit_count"] == 1


def test_watchlist_comparison_requires_no_worse_than_failed_and_detects_improvement():
    watchlist = [
        {
            "qid": "WATCH_QID",
            "anchor_doc_ids": ["ANCHOR1"],
        }
    ]
    run_snapshots = {
        "baseline": {
            "retrieval_rows": {
                "WATCH_QID": {"contexts": [{"doc_id": "ANCHOR1", "score": 0.9}]}
            },
            "answer_rows": {
                "WATCH_QID": {
                    "prompt_context_doc_ids": ["ANCHOR1"],
                    "predicted_rank1_diagnosis_text": "VL",
                }
            },
        },
        "failed_phase1a": {
            "retrieval_rows": {
                "WATCH_QID": {"contexts": [{"doc_id": "BASEX", "score": 0.9}, {"doc_id": "ANCHOR1", "score": 0.5}]}
            },
            "answer_rows": {
                "WATCH_QID": {
                    "prompt_context_doc_ids": ["ANCHOR1"],
                    "predicted_rank1_diagnosis_text": "Non-Leishmaniasis",
                }
            },
        },
        "revised_phase1a": {
            "retrieval_rows": {
                "WATCH_QID": {"contexts": [{"doc_id": "ANCHOR1", "score": 0.95}, {"doc_id": "AUG1", "score": 0.4}]}
            },
            "answer_rows": {
                "WATCH_QID": {
                    "prompt_context_doc_ids": ["ANCHOR1", "AUG1"],
                    "predicted_rank1_diagnosis_text": "VL",
                }
            },
        },
    }

    comparison = _build_watchlist_comparison(watchlist, run_snapshots)

    assert comparison["summary"]["retrieval_no_worse_than_failed"] is True
    assert comparison["summary"]["prompt_anchor_support_no_worse_than_failed"] is True
    assert comparison["summary"]["any_anchor_rank_improved_vs_failed"] is True


def test_adaptive_delta_reports_mode_switches_and_contamination_reduction():
    pure_rag_snapshot = {
        "run_dir": "/tmp/pure",
        "answer_rows": {
            "Q_NON_1": {
                "qid": "Q_NON_1",
                "ground_truth_bucket": "nonleish",
                "generation_mode": "rag_prompt",
                "gating_info": "",
                "threshold_used": 0.0,
                "retrieval_support_status": "weak_support",
                "prompt_context_doc_ids": ["LEISH_A", "LEISH_B", "NON_A"],
                "diagnosis_accuracy": 0.0,
                "diagnosis_type_accuracy": 0.0,
            },
            "Q_NON_2": {
                "qid": "Q_NON_2",
                "ground_truth_bucket": "nonleish",
                "generation_mode": "rag_prompt",
                "gating_info": "",
                "threshold_used": 0.0,
                "retrieval_support_status": "weak_support",
                "prompt_context_doc_ids": ["LEISH_A", "LEISH_B", "LEISH_C", "LEISH_D"],
                "diagnosis_accuracy": 0.0,
                "diagnosis_type_accuracy": 0.0,
            },
        },
    }
    adaptive_snapshot = {
        "run_dir": "/tmp/adaptive",
        "answer_rows": {
            "Q_NON_1": {
                "qid": "Q_NON_1",
                "ground_truth_bucket": "nonleish",
                "generation_mode": "norag_prompt",
                "gating_info": "[GATE OFF] weak_support",
                "threshold_used": 0.025,
                "retrieval_support_status": "empty_contexts",
                "prompt_context_doc_ids": [],
            },
            "Q_NON_2": {
                "qid": "Q_NON_2",
                "ground_truth_bucket": "nonleish",
                "generation_mode": "rag_prompt",
                "gating_info": "[SOFT GATE] score=0.0200, using top-1",
                "threshold_used": 0.025,
                "retrieval_support_status": "weak_support",
                "prompt_context_doc_ids": ["NON_A"],
            },
        },
    }
    norag_snapshot = {
        "run_dir": "/tmp/norag",
        "answer_rows": {
            "Q_NON_1": {
                "qid": "Q_NON_1",
                "ground_truth_bucket": "nonleish",
                "diagnosis_accuracy": 1.0,
                "diagnosis_type_accuracy": 1.0,
            },
            "Q_NON_2": {
                "qid": "Q_NON_2",
                "ground_truth_bucket": "nonleish",
                "diagnosis_accuracy": 1.0,
                "diagnosis_type_accuracy": 1.0,
            },
        },
    }
    watchlist = [
        {
            "qid": "WATCH_QID",
            "anchor_doc_ids": ["ANCHOR_1"],
        }
    ]
    pure_rag_snapshot["answer_rows"]["WATCH_QID"] = {
        "qid": "WATCH_QID",
        "ground_truth_bucket": "leish",
        "generation_mode": "rag_prompt",
        "prompt_context_doc_ids": ["ANCHOR_1"],
        "diagnosis_accuracy": 1.0,
        "diagnosis_type_accuracy": 1.0,
    }
    adaptive_snapshot["answer_rows"]["WATCH_QID"] = {
        "qid": "WATCH_QID",
        "ground_truth_bucket": "leish",
        "generation_mode": "rag_prompt",
        "prompt_context_doc_ids": ["ANCHOR_1"],
    }
    norag_snapshot["answer_rows"]["WATCH_QID"] = {
        "qid": "WATCH_QID",
        "ground_truth_bucket": "leish",
        "diagnosis_accuracy": 1.0,
        "diagnosis_type_accuracy": 1.0,
    }
    corpus_index = {
        "LEISH_A": {"is_leishmaniasis": True},
        "LEISH_B": {"is_leishmaniasis": True},
        "LEISH_C": {"is_leishmaniasis": True},
        "LEISH_D": {"is_leishmaniasis": True},
        "NON_A": {"is_leishmaniasis": False},
        "ANCHOR_1": {"is_leishmaniasis": True},
    }

    delta = _build_adaptive_vs_pure_rag_delta(
        pure_rag_snapshot,
        adaptive_snapshot,
        norag_snapshot,
        watchlist,
        corpus_index,
    )

    assert delta["loser_sets"]["nonleish_loser_qids"] == ["Q_NON_1", "Q_NON_2"]
    assert delta["nonleish_loser_delta"]["switched_from_rag_to_norag_count"] == 1
    assert delta["nonleish_loser_delta"]["pure_rag_leish_dominant_count"] == 2
    assert delta["nonleish_loser_delta"]["adaptive_leish_dominant_count"] == 0
    assert delta["decision"]["path_usage_changed_on_loser_sets"] is True
    assert delta["decision"]["nonleish_contamination_materially_reduced"] is True
    assert delta["decision"]["continue_to_judge"] is True
