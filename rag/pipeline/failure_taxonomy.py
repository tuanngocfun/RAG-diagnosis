"""Heuristic failure taxonomy for multimodal RAG vs no-RAG analysis.

These labels are analysis-time silver-label heuristics, not clinician-verified truth.
They are intended to explain why RAG may help or hurt across buckets.
"""

from __future__ import annotations

from collections import Counter
from typing import Dict, Iterable, List


VISUAL_GROUNDING_MARKERS = (
    "image shows",
    "image demonstrates",
    "image reveals",
    "in the image",
    "on the image",
    "visually",
    "visible",
    "appearance",
    "photograph",
    "photo",
    "shown here",
)
CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY = "support_images_used_as_prompt_references_only"


def _normalize_text(value: object) -> str:
    return str(value or "").strip()


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _answer_mentions_visual_grounding(text: object) -> bool:
    lowered = _normalize_text(text).lower()
    return any(marker in lowered for marker in VISUAL_GROUNDING_MARKERS)


def _is_image_query(row: Dict[str, object]) -> bool:
    query_type = _normalize_text(row.get("query_type")).lower()
    return "q3" in query_type or "multimodal" in query_type


def build_failure_taxonomy(
    common_qids: Iterable[str],
    rag_rows: Dict[str, Dict],
    norag_rows: Dict[str, Dict],
) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []

    for qid in common_qids:
        rag_row = rag_rows[qid]
        norag_row = norag_rows[qid]
        labels: List[str] = []
        reasons: List[str] = []

        query_image_count = int(rag_row.get("query_image_count") or 0)
        rag_context_count = int(rag_row.get("prompt_context_count") or 0)
        rag_support_images = int(rag_row.get("context_image_count") or 0)
        rag_support_tensor_count = int(rag_row.get("support_image_tensor_count") or 0)
        rag_diag = _as_float(rag_row.get("diagnosis_accuracy"))
        norag_diag = _as_float(norag_row.get("diagnosis_accuracy"))
        rag_faithfulness = _as_float(rag_row.get("multimodal_faithfulness"))
        bucket = str(rag_row.get("ground_truth_bucket") or "unknown")
        rag_family = _normalize_text(rag_row.get("diagnosis_family")).lower()
        rag_answer = rag_row.get("answer_text")

        if rag_row.get("answer_format_valid") is False:
            labels.append("format_contract_failure")
            reasons.append("rag_answer_failed_structured_output_contract")

        if _is_image_query(rag_row) and query_image_count > 0 and not _answer_mentions_visual_grounding(rag_answer):
            labels.append("ignored_query_image")
            reasons.append("image_query_present_but_answer_lacks_explicit_visual_grounding_language")

        if (
            _is_image_query(rag_row)
            and query_image_count > 0
            and _answer_mentions_visual_grounding(rag_answer)
            and rag_faithfulness is not None
            and rag_faithfulness < 0.5
        ):
            labels.append("unsupported_visual_claim")
            reasons.append("answer_makes_visual_claims_but_multimodal_faithfulness_is_low")

        if (
            rag_context_count > 0
            and _normalize_text(rag_row.get("generation_mode")) == "rag_prompt"
            and _normalize_text(rag_row.get("retrieval_support_status")) not in {"empty_contexts", "not_applicable_norag"}
            and bucket != "nonleish"
            and rag_diag == 0.0
            and norag_diag == 1.0
        ):
            labels.append("retrieved_evidence_conflict")
            reasons.append("rag_used_contexts_and_lost_against_matched_norag_on_exact_diagnosis")

        if (
            bucket == "nonleish"
            and rag_context_count > 0
            and rag_diag is not None
            and norag_diag is not None
            and rag_diag < norag_diag
            and "nonleish" not in rag_family
        ):
            labels.append("nonleish_confusion_from_leish_context")
            reasons.append("nonleish_case_degraded_under_rag_while_rag_rank1_family_is_not_nonleish")

        if not labels:
            labels = ["no_issue_detected"]
            reasons.append("no_failure_heuristic_triggered")

        records.append(
            {
                "qid": qid,
                "bucket": bucket,
                "query_type": rag_row.get("query_type"),
                "labels": labels,
                "reasons": reasons,
                "query_image_count": query_image_count,
                "rag_prompt_context_count": rag_context_count,
                "rag_context_image_count": rag_support_images,
                "rag_support_image_tensor_count": rag_support_tensor_count,
                "rag_generation_mode": rag_row.get("generation_mode"),
                "rag_diagnosis_accuracy": rag_diag,
                "norag_diagnosis_accuracy": norag_diag,
            }
        )

    return records


def summarize_failure_taxonomy(records: Iterable[Dict[str, object]]) -> Dict[str, Dict[str, object]]:
    materialized = list(records)
    by_bucket: Dict[str, Dict[str, object]] = {}

    for bucket in ("all", "leish", "nonleish", "unknown"):
        bucket_rows = materialized if bucket == "all" else [row for row in materialized if row.get("bucket") == bucket]
        counter = Counter()
        for row in bucket_rows:
            for label in row.get("labels") or []:
                counter[str(label)] += 1
        by_bucket[bucket] = {
            "n_rows": len(bucket_rows),
            "label_counts": dict(counter),
        }

    return by_bucket


def summarize_capability_caveats(rows: Iterable[Dict[str, object]]) -> Dict[str, object]:
    materialized = list(rows)
    by_bucket: Dict[str, Dict[str, object]] = {}

    for bucket in ("all", "leish", "nonleish", "unknown"):
        bucket_rows = materialized if bucket == "all" else [row for row in materialized if row.get("ground_truth_bucket") == bucket]
        support_prompt_only_count = sum(
            1
            for row in bucket_rows
            if int(row.get("context_image_count") or 0) > 0
            and int(row.get("support_image_tensor_count") or 0) == 0
        )
        label_counts = {}
        if support_prompt_only_count:
            label_counts[CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY] = support_prompt_only_count
        by_bucket[bucket] = {
            "n_rows": len(bucket_rows),
            "label_counts": label_counts,
        }

    return {
        "analysis_type": "capability_caveat_summary",
        "descriptions": {
            CAPABILITY_CAVEAT_SUPPORT_IMAGES_PROMPT_ONLY: (
                "Retrieved support images were available as prompt references but were not attached as true multimodal image tensors."
            )
        },
        "by_bucket": by_bucket,
    }
