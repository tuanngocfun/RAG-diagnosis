"""
Multimodal Evaluation Runner - Handles Q3 Image-Only Queries

This module extends run_evaluation.py to support TRUE multimodal evaluation:
- Q1/Q2: Text queries → text retrieval (existing behavior)
- Q3: Image queries → image/caption retrieval (NEW)

Aligns with MMed-RAG (2024/2025) evaluation protocol for medical VQA.

BUG FIXES per GPT 5.2 review:
- qid now uses case_id::query_type to avoid collision
- Expanded qrels mapping for unique qids
- Added image path existence check with skip reason tracking
- Fixed mutable default args antipattern
- Fixed double JSON parse
- Added auto catalog update
"""
import json
import csv
import re
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .config import (
    CHUNK_CONFIG,
    DATA_ROOT,
    DATASET_VERSION,
    EVAL_CONFIG,
    RUNS_DIR,
    SILVER_LABEL_DISCLAIMER,
    SPLIT_DIR,
    TRAIN_JSONL,
    TEST_JSONL,
    get_dataset_artifact_filenames,
    get_dataset_support_snapshot,
    get_path_fingerprint,
    get_runtime_metadata,
    is_strict_retrieval_mode,
)
from .query_templates import DIAGNOSIS_QUESTION_WITH_TYPE
from .evaluator import evaluate_retrieval, RetrievalResults
from .retriever import (
    Lane1Retriever,
    Lane2Retriever,
    get_collection_snapshot,
    get_expected_retrieval_resources,
    rrf_fusion,
)
from .reranker import get_medcpt_reranker
from .image_resolver import normalize_query_image_paths, resolve_case_image_paths
from .pseudolabel_adapter import (
    TEST_NORMALIZED,
    TRAIN_NORMALIZED,
    build_pseudolabel_artifacts,
    get_pseudolabel_artifact_paths,
)


_ACTIVE_DATASET_ARTIFACTS = get_dataset_artifact_filenames(DATASET_VERSION)
DEFAULT_MULTIMODAL_QRELS = _ACTIVE_DATASET_ARTIFACTS["qrels_verified"]
DEFAULT_MULTIMODAL_IMAGE_SEARCH = "images"
DEFAULT_MULTIMODAL_QUERY_FALLBACK = _ACTIVE_DATASET_ARTIFACTS["query"]
CANONICAL_P14_FULL_QUERY_TYPES = (
    "Q1_diagnosis",
    "Q3_image_diagnosis",
    "Q1_Q3_multimodal_diagnosis",
)
CAPTION_COLLECTION_ABSENT_ZERO_COVERAGE_REASON = (
    "caption_collection_absent_due_to_zero_caption_coverage"
)


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _query_type_set(query_types: List[str]) -> set[str]:
    return {str(qt).strip() for qt in query_types or [] if str(qt).strip()}


def _is_canonical_p14_full_pilot(
    query_types: List[str],
    *,
    strip_query_images: bool,
    ablation_scope: str,
) -> bool:
    return (
        DATASET_VERSION == "p14_v7"
        and _query_type_set(query_types) == set(CANONICAL_P14_FULL_QUERY_TYPES)
        and not strip_query_images
        and not str(ablation_scope or "").strip()
    )


def _expected_runtime_use(
    *,
    query_types: List[str],
    method: str,
    image_search_mode: str,
    rerank: bool,
) -> Dict[str, bool]:
    qset = _query_type_set(query_types)
    has_text_queries = bool(qset & {"Q1_symptom_only", "Q2_symptom_exposure", "Q1_diagnosis", "Q2_diagnosis_exposure"})
    has_image_only_queries = bool(qset & {"Q3_image_only", "Q3_image_diagnosis"})
    has_multimodal_queries = "Q1_Q3_multimodal_diagnosis" in qset

    uses_dense = method in {"e5", "hybrid", "2lane"} and (has_text_queries or has_multimodal_queries)
    uses_bm25 = method in {"bm25", "hybrid", "2lane"} and (has_text_queries or has_multimodal_queries)
    uses_caption = (
        (method == "2lane" and has_text_queries)
        or (image_search_mode == "captions" and (has_image_only_queries or has_multimodal_queries))
    )
    uses_image = image_search_mode == "images" and (has_image_only_queries or has_multimodal_queries)
    uses_rerank = bool(rerank) and (has_text_queries or has_multimodal_queries)

    return {
        "dense_lane": uses_dense,
        "bm25_lane": uses_bm25,
        "caption_lane": uses_caption,
        "image_lane": uses_image,
        "rerank": uses_rerank,
    }


def _scan_caption_support(train_jsonl: Path) -> Dict[str, object]:
    """
    Match the caption bootstrap rule exactly: a caption counts only if img["caption"] is truthy.
    """
    total_image_entries = 0
    caption_entries = 0

    if train_jsonl.exists():
        with open(train_jsonl, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                for img in row.get("images") or []:
                    if not isinstance(img, dict):
                        continue
                    total_image_entries += 1
                    caption = img.get("caption", "")
                    if caption:
                        caption_entries += 1

    coverage_ratio = (
        float(caption_entries) / float(total_image_entries)
        if total_image_entries
        else 0.0
    )
    return {
        "total_image_entries": total_image_entries,
        "caption_entries": caption_entries,
        "caption_coverage_ratio": coverage_ratio,
    }


def _build_required_resource_matrix(
    *,
    query_types: List[str],
    method: str,
    image_search_mode: str,
    rerank: bool,
    strict_mode: bool,
    strip_query_images: bool,
    ablation_scope: str,
    caption_support: Optional[Dict[str, object]] = None,
) -> Dict[str, Dict[str, object]]:
    expected_resources = get_expected_retrieval_resources(DATASET_VERSION)
    canonical_full_pilot = _is_canonical_p14_full_pilot(
        query_types,
        strip_query_images=strip_query_images,
        ablation_scope=ablation_scope,
    )
    expected_use = _expected_runtime_use(
        query_types=query_types,
        method=method,
        image_search_mode=image_search_mode,
        rerank=rerank,
    )
    caption_support = dict(caption_support or _scan_caption_support(TRAIN_JSONL))
    caption_entries = int(caption_support.get("caption_entries") or 0)

    required_at_bootstrap = {
        "dense_collection": canonical_full_pilot or expected_use["dense_lane"],
        "caption_collection": expected_use["caption_lane"] or caption_entries > 0,
        "image_collection": canonical_full_pilot or expected_use["image_lane"],
        "bm25_index": canonical_full_pilot or expected_use["bm25_lane"],
    }

    return {
        "dense_collection": {
            "name": expected_resources["dense_collection"],
            "required_at_bootstrap": required_at_bootstrap["dense_collection"],
            "required_for_runtime_use": expected_use["dense_lane"],
            "expected_vector_size": 1024,
        },
        "caption_collection": {
            "name": expected_resources["caption_collection"],
            "required_at_bootstrap": required_at_bootstrap["caption_collection"],
            "required_for_runtime_use": expected_use["caption_lane"],
            "expected_vector_size": 512,
        },
        "image_collection": {
            "name": expected_resources["image_collection"],
            "required_at_bootstrap": required_at_bootstrap["image_collection"],
            "required_for_runtime_use": expected_use["image_lane"],
            "expected_vector_size": 512,
        },
        "bm25_index": {
            "name": str(expected_resources["bm25_index_path"]),
            "required_at_bootstrap": required_at_bootstrap["bm25_index"],
            "required_for_runtime_use": expected_use["bm25_lane"],
        },
        "expected_runtime_use": expected_use,
        "caption_coverage": caption_support,
        "canonical_full_pilot": canonical_full_pilot,
        "strict_mode": strict_mode,
    }


def _build_caption_support_contract(
    *,
    required_resources: Dict[str, object],
    collection_snapshots: Dict[str, Dict[str, object]],
    image_search_mode: str,
) -> Dict[str, object]:
    coverage = dict(required_resources.get("caption_coverage") or {})
    caption_requirement = required_resources["caption_collection"]
    collection_snapshot = collection_snapshots["caption_collection"]
    collection_exists = bool(collection_snapshot.get("exists"))
    lane_expected = bool(caption_requirement["required_for_runtime_use"])
    required_at_bootstrap = bool(caption_requirement["required_at_bootstrap"])
    caption_entries = int(coverage.get("caption_entries") or 0)

    absence_reason = None
    if (
        is_strict_retrieval_mode(DATASET_VERSION)
        and DATASET_VERSION == "p14_v7"
        and image_search_mode == "images"
        and not lane_expected
        and caption_entries == 0
        and not collection_exists
    ):
        absence_reason = CAPTION_COLLECTION_ABSENT_ZERO_COVERAGE_REASON

    return {
        "coverage": {
            "total_image_entries": int(coverage.get("total_image_entries") or 0),
            "caption_entries": caption_entries,
            "caption_coverage_ratio": float(coverage.get("caption_coverage_ratio") or 0.0),
        },
        "collection_name": caption_requirement["name"],
        "collection_exists": collection_exists,
        "lane_expected": lane_expected,
        "lane_exercised": False,
        "required_at_bootstrap": required_at_bootstrap,
        "absence_reason": absence_reason,
    }


def _validate_canonical_pilot_contract(
    *,
    query_types: List[str],
    method: str,
    rerank: bool,
    image_search_mode: str,
    strip_query_images: bool,
    ablation_scope: str,
) -> None:
    if not _is_canonical_p14_full_pilot(
        query_types,
        strip_query_images=strip_query_images,
        ablation_scope=ablation_scope,
    ):
        return

    if method != "hybrid":
        raise ValueError(
            "Canonical p14_v7 full-pack pilot requires --method hybrid. "
            f"Received method={method!r}."
        )
    if not rerank:
        raise ValueError("Canonical p14_v7 full-pack pilot requires --rerank.")
    if image_search_mode != "images":
        raise ValueError(
            "Canonical p14_v7 full-pack pilot requires --image-search images. "
            f"Received image_search_mode={image_search_mode!r}."
        )


def _build_retrieval_contract(
    *,
    lane1: Lane1Retriever,
    lane2: Lane2Retriever,
    qpath: Path,
    qrels_path: Path,
    query_types: List[str],
    method: str,
    rerank: bool,
    image_search_mode: str,
    strip_query_images: bool,
    ablation_scope: str,
) -> Dict[str, object]:
    strict_mode = is_strict_retrieval_mode(DATASET_VERSION)
    required_resources = _build_required_resource_matrix(
        query_types=query_types,
        method=method,
        image_search_mode=image_search_mode,
        rerank=rerank,
        strict_mode=strict_mode,
        strip_query_images=strip_query_images,
        ablation_scope=ablation_scope,
    )
    lane1_contract = lane1.get_resource_contract()
    lane2_contract = lane2.get_resource_contract()

    collection_snapshots = {
        "dense_collection": get_collection_snapshot(lane1.client, lane1.collection_name),
        "caption_collection": get_collection_snapshot(lane2.client, lane2.caption_collection),
        "image_collection": get_collection_snapshot(lane2.client, lane2.image_collection),
    }
    caption_support = _build_caption_support_contract(
        required_resources=required_resources,
        collection_snapshots=collection_snapshots,
        image_search_mode=image_search_mode,
    )

    return {
        "dataset_version": DATASET_VERSION,
        "strict_mode": strict_mode,
        "canonical_full_pilot": required_resources["canonical_full_pilot"],
        "method": method,
        "rerank": rerank,
        "image_search_mode": image_search_mode,
        "query_types": list(query_types),
        "required_resources": required_resources,
        "artifacts": {
            "train_jsonl": get_path_fingerprint(TRAIN_JSONL),
            "test_jsonl": get_path_fingerprint(TEST_JSONL),
            "queries_jsonl": get_path_fingerprint(qpath),
            "qrels_json": get_path_fingerprint(qrels_path),
        },
        "resolved_resources": {
            "dense_collection": lane1.collection_name,
            "caption_collection": lane2.caption_collection,
            "image_collection": lane2.image_collection,
            "bm25_index": lane1_contract["bm25_index"],
        },
        "collections_at_start": collection_snapshots,
        "caption_support": caption_support,
        "resource_events": list(lane1_contract.get("resource_events", [])),
        "usage": {},
        "usage_expectations": required_resources["expected_runtime_use"],
        "strip_query_images": strip_query_images,
        "ablation_scope": ablation_scope,
    }


def _validate_retrieval_contract(contract: Dict[str, object]) -> None:
    required_resources = contract["required_resources"]
    bm25_info = contract["resolved_resources"]["bm25_index"]
    collection_snapshots = contract["collections_at_start"]
    caption_support = contract.get("caption_support") or {}

    if required_resources["canonical_full_pilot"]:
        _validate_canonical_pilot_contract(
            query_types=contract["query_types"],
            method=contract["method"],
            rerank=bool(contract["rerank"]),
            image_search_mode=str(contract["image_search_mode"]),
            strip_query_images=bool(contract["strip_query_images"]),
            ablation_scope=str(contract["ablation_scope"]),
        )

    if required_resources["bm25_index"]["required_at_bootstrap"] and not bm25_info.get("exists"):
        raise FileNotFoundError(
            "Required versioned BM25 index is missing for this run: "
            f"{required_resources['bm25_index']['name']}"
        )
    if contract["strict_mode"] and bm25_info.get("fallback_used"):
        raise RuntimeError(
            "Strict retrieval mode forbids legacy BM25 fallback, but a fallback was resolved: "
            f"{bm25_info.get('resolved_path')}"
        )

    for resource_key in ("dense_collection", "caption_collection", "image_collection"):
        requirement = required_resources[resource_key]
        snapshot = collection_snapshots[resource_key]
        if (
            resource_key == "caption_collection"
            and not requirement["required_at_bootstrap"]
            and not requirement["required_for_runtime_use"]
        ):
            if not snapshot.get("exists"):
                caption_entries = int(
                    ((caption_support.get("coverage") or {}).get("caption_entries") or 0)
                )
                if caption_entries > 0:
                    raise RuntimeError(
                        "Caption collection is missing even though caption coverage is non-zero."
                    )
            continue
        if requirement["required_at_bootstrap"] and not snapshot.get("exists"):
            raise FileNotFoundError(
                f"Required versioned collection is missing: {requirement['name']}"
            )
        vector_size = snapshot.get("vector_size")
        expected_size = requirement.get("expected_vector_size")
        if snapshot.get("exists") and expected_size and vector_size not in {None, expected_size}:
            raise RuntimeError(
                f"Collection {requirement['name']} has vector_size={vector_size}, "
                f"expected {expected_size}."
            )
        if requirement["required_at_bootstrap"] and snapshot.get("points_count") == 0:
            raise RuntimeError(
                f"Required collection exists but has 0 points: {requirement['name']}"
            )


def _build_retrieval_usage_summary(
    *,
    lane1: Lane1Retriever,
    lane2: Lane2Retriever,
    rerank_query_count: int,
    rerank_pass_count: int,
) -> Dict[str, object]:
    lane1_usage = lane1.get_usage_snapshot()
    lane2_usage = lane2.get_usage_snapshot()
    resource_events = list(lane1.resource_events)

    usage = {
        "dense_lane_query_count": int(lane1_usage.get("dense_lane_query_count", 0)),
        "bm25_lane_query_count": int(lane1_usage.get("bm25_lane_query_count", 0)),
        "hybrid_lane_query_count": int(lane1_usage.get("hybrid_lane_query_count", 0)),
        "caption_query_count": int(
            lane2_usage.get("caption_query_count", 0) + lane2_usage.get("caption_image_query_count", 0)
        ),
        "image_query_count": int(
            lane2_usage.get("image_query_count", 0) + lane2_usage.get("image_path_query_count", 0)
        ),
        "rerank_query_count": int(rerank_query_count),
        "rerank_pass_count": int(rerank_pass_count),
        "fallback_event_count": len(resource_events),
        "fallback_events": resource_events,
        "collection_queries": {
            key.split("::", 1)[1]: int(value)
            for key, value in {**lane1_usage, **lane2_usage}.items()
            if key.startswith("collection_queries::")
        },
    }
    usage["resources_exercised"] = {
        "dense_lane": usage["dense_lane_query_count"] > 0,
        "bm25_lane": usage["bm25_lane_query_count"] > 0,
        "caption_lane": usage["caption_query_count"] > 0,
        "image_lane": usage["image_query_count"] > 0,
        "rerank": usage["rerank_query_count"] > 0,
    }
    return usage


def _resolve_queries_path(queries_file: Optional[str], dataset_pack: str) -> Path:
    artifact_names = get_dataset_artifact_filenames(DATASET_VERSION)
    if queries_file:
        qpath = Path(queries_file)
        if not qpath.is_absolute():
            qpath = SPLIT_DIR / queries_file
        return qpath

    if dataset_pack == "mixed56":
        return SPLIT_DIR / artifact_names["query_mixed56"]
    if dataset_pack == "test":
        return SPLIT_DIR / "eval_queries_v163.jsonl"

    qpath = SPLIT_DIR / artifact_names["query"]
    if qpath.exists():
        return qpath

    pseudo_fallback = SPLIT_DIR / DEFAULT_MULTIMODAL_QUERY_FALLBACK
    if pseudo_fallback.exists():
        print(f"Warning: Using pseudolabel queries file: {pseudo_fallback}")
        return pseudo_fallback

    legacy_path = SPLIT_DIR / "eval_queries.jsonl"
    print(f"Warning: Using legacy queries file: {legacy_path}")
    return legacy_path


def _is_baseline_query_pack(qpath: Path) -> bool:
    return qpath.name in {
        f"eval_queries_{DATASET_VERSION}.jsonl",
        DEFAULT_MULTIMODAL_QUERY_FALLBACK,
    }


def _build_evaluation_contract(
    *,
    dataset_pack: str,
    qrels_file: str,
    qpath: Path,
    image_search_mode: str,
) -> Dict[str, object]:
    warnings: List[str] = []

    if qrels_file != DEFAULT_MULTIMODAL_QRELS:
        warnings.append(
            f"qrels_file={qrels_file} is not baseline-equivalent; expected {DEFAULT_MULTIMODAL_QRELS}."
        )
    if image_search_mode != DEFAULT_MULTIMODAL_IMAGE_SEARCH:
        warnings.append(
            f"image_search_mode={image_search_mode} is not baseline-equivalent; expected {DEFAULT_MULTIMODAL_IMAGE_SEARCH}."
        )
    if not _is_baseline_query_pack(qpath):
        warnings.append(
            f"queries_file={qpath.name} is not the baseline-equivalent 56-query pseudolabel pack."
        )
    if dataset_pack == "mixed56":
        warnings.append(
            "dataset_pack=mixed56 is a multimodal-only subset and should not be compared directly to the full dataset pack."
        )

    return {
        "dataset_pack": dataset_pack,
        "qrels_file": qrels_file,
        "queries_file": str(qpath),
        "queries_file_name": qpath.name,
        "image_search_mode": image_search_mode,
        "baseline_equivalent": len(warnings) == 0,
        "warnings": warnings,
    }


def _resolve_secondary_pseudolabel_qrels_path(pseudolabel_stats=None) -> Path:
    if pseudolabel_stats is not None:
        return Path(pseudolabel_stats.qrels_pseudolabel_path)
    return get_pseudolabel_artifact_paths(dataset_version=DATASET_VERSION)["qrels_pseudolabel"]


def _label_payload(row: Dict, label_key: str) -> Dict[str, object]:
    labels = row.get("labels") or {}
    if isinstance(labels, dict):
        label = labels.get(label_key)
        if isinstance(label, dict):
            return label
    return {
        "diagnosis": row.get("diagnosis", ""),
        "diagnosis_type": row.get("diagnosis_type", ""),
        "species": row.get("species", ""),
    }


def _build_case_qrels(
    train_cases: Dict[str, Dict],
    test_cases: Dict[str, Dict],
    label_key: str,
) -> Dict[str, Dict[str, int]]:
    train_by_type: Dict[str, List[str]] = {}
    for case_id, row in train_cases.items():
        diagnosis_type = str((_label_payload(row, label_key).get("diagnosis_type") or "")).strip()
        if not diagnosis_type:
            continue
        train_by_type.setdefault(diagnosis_type, []).append(case_id)

    qrels: Dict[str, Dict[str, int]] = {}
    for case_id, row in test_cases.items():
        diagnosis_type = str((_label_payload(row, label_key).get("diagnosis_type") or "")).strip()
        qrels[case_id] = {doc_id: 3 for doc_id in train_by_type.get(diagnosis_type, [])}
    return qrels


def _expand_qrels_for_queries(
    case_qrels: Dict[str, Dict[str, int]],
    queries: List[Dict],
) -> Dict[str, Dict[str, int]]:
    expanded: Dict[str, Dict[str, int]] = {}
    for q in queries:
        case_id = q["case_id"]
        query_type = q["query_type"]
        expanded[f"{case_id}::{query_type}"] = dict(case_qrels.get(case_id, {}))
    return expanded


def _format_metrics(eval_results: RetrievalResults, k_values: List[int]) -> Dict[str, object]:
    return {
        "recall": {f"@{k}": eval_results.recall_at_k.get(k, 0) for k in k_values},
        "ndcg": {f"@{k}": eval_results.ndcg_at_k.get(k, 0) for k in k_values},
        "precision": {f"@{k}": eval_results.precision_at_k.get(k, 0) for k in k_values},
        "mrr": eval_results.mrr,
        "map": eval_results.map_score,
    }


def _primary_qrels_audit(
    *,
    train_cases: Dict[str, Dict],
    primary_qrels: Dict[str, Dict[str, int]],
    baseline_train_case_ids: Optional[set[str]] = None,
) -> Dict[str, object]:
    referenced_doc_ids = {
        doc_id
        for docs in primary_qrels.values()
        if isinstance(docs, dict)
        for doc_id in docs.keys()
    }
    train_doc_ids = set(train_cases)
    missing_train_doc_ids = sorted(train_doc_ids - referenced_doc_ids)
    nonleish_train_doc_ids = sorted(
        case_id
        for case_id, row in train_cases.items()
        if str((_label_payload(row, "verified").get("diagnosis_type") or "")).strip() == "Non-Leishmaniasis"
    )
    missing_nonleish_doc_ids = [
        case_id for case_id in nonleish_train_doc_ids if case_id not in referenced_doc_ids
    ]

    missing_by_type = Counter(
        str((_label_payload(train_cases[case_id], "verified").get("diagnosis_type") or "UNKNOWN")).strip() or "UNKNOWN"
        for case_id in missing_train_doc_ids
    )

    audit: Dict[str, object] = {
        "primary_qrels_query_count": len(primary_qrels),
        "primary_qrels_referenced_train_doc_count": len(train_doc_ids & referenced_doc_ids),
        "train_doc_count": len(train_doc_ids),
        "train_docs_missing_from_primary_qrels_count": len(missing_train_doc_ids),
        "train_docs_missing_from_primary_qrels_examples": missing_train_doc_ids[:20],
        "missing_train_docs_by_diagnosis_type": dict(sorted(missing_by_type.items())),
        "nonleish_train_doc_count": len(nonleish_train_doc_ids),
        "nonleish_train_docs_missing_from_primary_qrels_count": len(missing_nonleish_doc_ids),
        "nonleish_train_docs_missing_from_primary_qrels_examples": missing_nonleish_doc_ids[:20],
    }

    if baseline_train_case_ids is not None:
        augmented_doc_ids = sorted(train_doc_ids - baseline_train_case_ids)
        augmented_nonleish_doc_ids = [
            case_id for case_id in augmented_doc_ids if case_id in set(nonleish_train_doc_ids)
        ]
        missing_augmented_doc_ids = [
            case_id for case_id in augmented_doc_ids if case_id not in referenced_doc_ids
        ]
        missing_augmented_nonleish_doc_ids = [
            case_id for case_id in augmented_nonleish_doc_ids if case_id not in referenced_doc_ids
        ]
        audit.update(
            {
                "augmented_train_doc_count": len(augmented_doc_ids),
                "augmented_train_docs_missing_from_primary_qrels_count": len(missing_augmented_doc_ids),
                "augmented_train_docs_missing_from_primary_qrels_examples": missing_augmented_doc_ids[:20],
                "augmented_nonleish_train_doc_count": len(augmented_nonleish_doc_ids),
                "augmented_nonleish_train_docs_missing_from_primary_qrels_count": len(missing_augmented_nonleish_doc_ids),
                "augmented_nonleish_train_docs_missing_from_primary_qrels_examples": missing_augmented_nonleish_doc_ids[:20],
            }
        )

    return audit


def extract_query_focused_snippet(case_text: str, query: str, max_chars: int = 1800) -> str:
    """Extract diagnosis-bearing text while penalizing early differential noise."""
    if not case_text:
        return ""

    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', case_text) if s.strip()]
    if not sentences:
        return case_text[:max_chars]

    query_terms = set(re.findall(r'\b\w{3,}\b', (query or '').lower()))
    positive_patterns = [
        r'final diagnosis',
        r'diagnos(?:ed|is) with',
        r'confirmed',
        r'amastigot',
        r'leishman[- ]donovan',
        r'biopsy',
        r'histopath',
        r'bone\s*marrow',
        r'pcr.{0,20}positive',
        r'smear.{0,20}positive',
        r'rk39.{0,20}positive',
        r'responded to',
        r'treated with amphotericin',
        r'treated with antimonial',
    ]
    negative_patterns = [
        r'suspected',
        r'considered',
        r'differential',
        r'ruled out',
        r'negative',
        r'inadequate response',
        r'no evidence of',
        r'not diagnostic',
    ]

    windows = []
    for idx in range(len(sentences)):
        window_sentences = sentences[idx:idx + 2]
        window = ' '.join(window_sentences).strip()
        if not window:
            continue
        lowered = window.lower()
        overlap = len(query_terms & set(re.findall(r'\b\w{3,}\b', lowered)))
        pos = sum(1 for pat in positive_patterns if re.search(pat, lowered))
        neg = sum(1 for pat in negative_patterns if re.search(pat, lowered))
        tail_bonus = 0.75 if idx >= max(0, len(sentences) - 4) else 0.0
        score = (pos * 4.0) + (overlap * 0.35) + tail_bonus - (neg * 1.5)
        windows.append((score, idx, window_sentences, pos))

    chosen_sentence_ids = set()
    remaining_chars = max_chars

    confirmatory = sorted(
        [w for w in windows if w[3] > 0],
        key=lambda item: (item[3], item[0], -item[1]),
        reverse=True,
    )
    for _, _, window_sentences, _ in confirmatory:
        for sent in window_sentences:
            sent_idx = sentences.index(sent)
            if sent_idx in chosen_sentence_ids:
                continue
            if remaining_chars <= len(sent) + 1:
                break
            chosen_sentence_ids.add(sent_idx)
            remaining_chars -= len(sent) + 1
        if remaining_chars <= 0:
            break

    if remaining_chars > 0:
        for _, idx, window_sentences, _ in sorted(windows, key=lambda item: item[0], reverse=True):
            for offset, sent in enumerate(window_sentences):
                sent_idx = idx + offset
                if sent_idx in chosen_sentence_ids:
                    continue
                if remaining_chars <= len(sent) + 1:
                    break
                chosen_sentence_ids.add(sent_idx)
                remaining_chars -= len(sent) + 1
            if remaining_chars <= 0:
                break

    if not chosen_sentence_ids:
        return case_text[:max_chars]

    result = ' '.join(sentences[idx] for idx in sorted(chosen_sentence_ids))
    return result[:max_chars] if len(result) > max_chars else result


# ==============================================================================
# TYPE-AWARE SOFT RERANK (per GPT 5.2 Level B)
# Purpose: Reduce harm from wrong-subtype contexts
# WARNING: Uses query keywords to infer subtype, NOT ground truth (data leakage)
# ==============================================================================

def infer_query_subtype(query_text: str) -> tuple:
    """
    Infer Leishmaniasis subtype from query symptoms.
    
    Per Gemini 3 Pro: Do NOT use ground truth labels - that's data leakage.
    Instead, use high-precision keyword patterns from the query text.
    
    Returns:
        (subtype: str, confidence: float)
        subtype is one of: 'Visceral', 'Cutaneous', 'Mucocutaneous', 'PKDL', 'Unknown'
    """
    if not query_text:
        return ('Unknown', 0.0)
    
    q = query_text.lower()
    
    # High-confidence patterns (per Gemini 3 Pro recommendation)
    # These are explicit mentions or pathognomonic symptoms
    
    # PKDL: Very specific patterns
    if any(k in q for k in ['post-kala', 'pkdl', 'post kala-azar', 'dermal leishmaniasis']):
        return ('PKDL', 0.95)
    
    # Mucocutaneous: Mucosal involvement is distinctive
    if any(k in q for k in ['mucosal', 'mucocutaneous', 'mcl', 'nasal septum', 
                             'nasal mucosa', 'palate ulcer', 'oronasal', 'espundia']):
        return ('Mucocutaneous', 0.9)
    
    # Visceral: Systemic symptoms with organ involvement
    visceral_patterns = ['splenomegaly', 'hepatomegaly', 'pancytopenia', 'kala-azar',
                        'visceral leishmaniasis', 'bone marrow aspirate', 'ld bodies',
                        'leishman-donovan', 'fever.*weight loss.*spleen']
    if any(k in q for k in visceral_patterns[:7]):  # Explicit mentions
        return ('Visceral', 0.85)
    if re.search(r'fever.{0,50}(spleen|hepato)', q):  # Pattern-based
        return ('Visceral', 0.7)
    
    # Cutaneous: Skin lesions without mucosal involvement
    cutaneous_patterns = ['cutaneous leishmaniasis', 'skin ulcer', 'skin nodule',
                          'papule', 'oriental sore', 'skin lesion', 'delhi boil']
    if any(k in q for k in cutaneous_patterns):
        # Check NOT mucosal (to differentiate from MCL)
        if not any(m in q for m in ['mucosa', 'nasal', 'palate']):
            return ('Cutaneous', 0.7)
    
    # Low confidence: Generic skin mention (could be CL or MCL)
    if 'ulcer' in q or 'nodule' in q or 'lesion' in q:
        return ('Cutaneous', 0.4)  # Low confidence - no filtering
    
    # Unknown: No clear signal - don't filter
    return ('Unknown', 0.0)


def infer_doc_subtype(doc_text: str) -> str:
    """
    Infer subtype from document/context text.
    Uses same patterns but returns just the type (for corpus annotation).
    """
    subtype, conf = infer_query_subtype(doc_text)
    return subtype if conf > 0.5 else 'Unknown'


def soft_rerank_by_subtype(
    contexts: List[Dict], 
    query_text: str,
    train_cases: Dict = None
) -> List[Dict]:
    """
    Soft rerank contexts by subtype match/mismatch.
    
    Per GPT 5.2 Level B approach:
    - If query subtype inferred with high confidence:
      - Boost matching subtype docs (+0.005)
      - Penalize mismatching docs (-0.003 * confidence)
    - If low confidence: return unchanged (don't risk filtering correct docs)
    
    Args:
        contexts: List of context dicts with 'doc_id', 'score', 'text'
        query_text: Query text to infer subtype from
        train_cases: Optional dict of train cases for additional metadata
    
    Returns:
        Reranked contexts list
    """
    query_subtype, query_conf = infer_query_subtype(query_text)
    
    # Don't rerank if confidence too low
    if query_subtype == 'Unknown' or query_conf < 0.5:
        return contexts
    
    reranked = []
    for ctx in contexts:
        original_score = ctx.get('score', 0.0)
        adjustment = 0.0
        
        # Infer doc subtype from context text
        doc_subtype = infer_doc_subtype(ctx.get('text', ''))
        
        # Also check train_cases metadata if available
        if train_cases and ctx.get('doc_id') in train_cases:
            case_meta = train_cases[ctx['doc_id']]
            # Use explicit label if available
            if case_meta.get('diagnosis_type'):
                doc_subtype = case_meta['diagnosis_type']
        
        # Apply adjustment
        if doc_subtype != 'Unknown':
            if query_subtype.lower() in doc_subtype.lower() or doc_subtype.lower() in query_subtype.lower():
                adjustment = +0.005  # Small boost for match
            else:
                adjustment = -0.003 * query_conf  # Penalty scales with confidence
        
        reranked.append({
            **ctx,
            'score': original_score + adjustment,
            '_subtype_match': f'{query_subtype}({query_conf:.2f}) vs {doc_subtype}'
        })
    
    # Sort by adjusted score
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    return reranked


@dataclass
class MultimodalRunConfig:
    """Configuration for multimodal evaluation run."""
    run_id: str
    qrels_file: str
    query_types: List[str]
    retriever_method: str
    rerank: bool
    k_values: List[int]
    created_at: str
    image_search_mode: str
    agentic_lite: bool = False
    queries_file: str = ""
    dataset_pack: str = "auto"
    context_k: Optional[int] = None
    retrieval_top_k: int = 20
    ordering_mode: str = "image_first"


def _score_stats(results: List[tuple]) -> tuple[float, float]:
    if not results:
        return 0.0, 0.0
    scores = [float(s) for _, s in results]
    top = scores[0]
    third = scores[2] if len(scores) >= 3 else scores[-1]
    margin = top - third
    return top, margin


def _should_refine_agentic(query_type: str, results: List[tuple]) -> tuple[bool, str]:
    if not results:
        return True, "no_results"
    top, margin = _score_stats(results)
    if query_type in ["Q1_diagnosis", "Q2_diagnosis_exposure"]:
        min_top = 0.016
        min_margin = 0.002
    elif query_type == "Q1_Q3_multimodal_diagnosis":
        min_top = 0.014
        min_margin = 0.0015
    else:
        min_top = 0.016
        min_margin = 0.002

    if top < min_top:
        return True, f"low_top_score({top:.4f}<{min_top:.4f})"
    if margin < min_margin:
        return True, f"low_margin({margin:.4f}<{min_margin:.4f})"
    return False, "sufficient"


def _rewrite_query_agentic_lite(query_text: str) -> str:
    q = (query_text or "").strip()
    if not q:
        return q

    ql = q.lower()
    hints = []
    if any(k in ql for k in ["splenomegaly", "hepatomegaly", "pancytopenia", "kala-azar", "bone marrow"]):
        hints.append("prioritize visceral leishmaniasis evidence")
    if any(k in ql for k in ["ulcer", "plaque", "nodule", "cutaneous", "skin lesion"]):
        hints.append("prioritize cutaneous leishmaniasis and subtype differentiation")
    if any(k in ql for k in ["mucosal", "nasal", "palate", "mucocutaneous"]):
        hints.append("prioritize mucocutaneous involvement evidence")
    if any(k in ql for k in ["immunosupp", "hiv", "transplant", "steroid"]):
        hints.append("include immunosuppression-associated atypical presentation evidence")
    if any(k in ql for k in ["travel", "endemic", "geograph", "region"]):
        hints.append("include epidemiology and species-region links")

    base_focus = "focus on confirmatory findings, subtype cues, species clues, and high-yield differential diagnosis"
    if hints:
        focus = base_focus + "; " + "; ".join(hints[:4])
    else:
        focus = base_focus

    return f"{q}\n\nRefinement instruction: {focus}."


def run_multimodal_evaluation(
    qrels_file: str = DEFAULT_MULTIMODAL_QRELS,
    query_types: Optional[List[str]] = None,
    method: str = "hybrid",
    rerank: bool = False,
    k_values: Optional[List[int]] = None,
    run_id: Optional[str] = None,
    image_search_mode: str = DEFAULT_MULTIMODAL_IMAGE_SEARCH,
    agentic_lite: bool = False,
    queries_file: Optional[str] = None,
    dataset_pack: str = "auto",
    pseudolabel_train_results: Optional[str] = None,
    pseudolabel_test_results: Optional[str] = None,
    pseudolabel_suffix: str = "",
    pseudolabel_force: bool = False,
    strip_query_images: bool = False,
    ablation_scope: str = "",
    context_k: Optional[int] = None,
    retrieval_top_k: int = 20,
    ordering_mode: str = "image_first",
) -> Path:
    """
    Run multimodal evaluation supporting both text and image queries.
    
    Args:
        qrels_file: QRELs filename in SPLIT_DIR
        query_types: List of query types to evaluate (default: Q1_Q3 multimodal only)
        method: Retrieval method for text queries
        rerank: Whether to use MedCPT reranking (text only)
        k_values: K values for metrics (default: [5, 10])
        run_id: Optional run ID (auto-generated if None)
        image_search_mode: "captions" or "images" for Q3 retrieval
    
    Returns:
        Path to run directory
    """
    # Handle mutable default args properly (per GPT 5.2)
    # Default to multimodal-only queries to reduce generation/evaluation API cost.
    if query_types is None:
        query_types = ["Q1_Q3_multimodal_diagnosis"]
    if k_values is None:
        k_values = [5, 10]
    if retrieval_top_k < 1:
        raise ValueError("retrieval_top_k must be >= 1")
    if context_k is not None and int(context_k) < 1:
        raise ValueError("context_k must be >= 1 when provided")
    if context_k is not None and retrieval_top_k < int(context_k):
        raise ValueError(
            f"retrieval_top_k must be >= context_k when both are set "
            f"(got retrieval_top_k={retrieval_top_k}, context_k={context_k})"
        )
    
    pseudolabel_stats = None

    # Setup
    try:
        pseudolabel_stats = build_pseudolabel_artifacts(
            force=pseudolabel_force,
            train_results_path=pseudolabel_train_results,
            test_results_path=pseudolabel_test_results,
            output_suffix=pseudolabel_suffix,
            dataset_version=DATASET_VERSION,
        )
        active_train_rows = _count_jsonl_rows(TRAIN_JSONL)
        active_test_rows = _count_jsonl_rows(TEST_JSONL)

        train_override_active = Path(TRAIN_JSONL).resolve() != Path(TRAIN_NORMALIZED).resolve()
        test_override_active = Path(TEST_JSONL).resolve() != Path(TEST_NORMALIZED).resolve()

        if not (train_override_active or test_override_active):
            print(
                "Prepared pseudolabel artifacts: "
                f"train={pseudolabel_stats.train_rows}, "
                f"test={pseudolabel_stats.test_rows}, "
                f"queries={pseudolabel_stats.query_rows}, "
                f"mixed56_queries={pseudolabel_stats.query_mixed56_rows}"
            )
        else:
            print(
                "Prepared pseudolabel artifacts for baseline files. "
                "Active overrides are enabled for this run."
            )
        print(
            "Active dataset files: "
            f"train={TRAIN_JSONL} (rows={active_train_rows}), "
            f"test={TEST_JSONL} (rows={active_test_rows})"
        )
        if pseudolabel_stats.train_rows != active_train_rows or pseudolabel_stats.test_rows != active_test_rows:
            print(
                "Note: active train/test overrides differ from baseline pseudolabel artifacts; "
                "retrieval/indexing uses the active dataset files above."
            )
    except Exception as e:
        print(f"Warning: Could not refresh pseudolabel artifacts: {e}")

    if strip_query_images and not ablation_scope:
        ablation_scope = "retrieval_query_image_strip"

    if pseudolabel_stats and pseudolabel_suffix and not queries_file:
        if dataset_pack == "mixed56":
            queries_file = pseudolabel_stats.query_mixed56_path
        elif dataset_pack == "auto":
            queries_file = pseudolabel_stats.query_path

    if pseudolabel_stats and pseudolabel_suffix and qrels_file == DEFAULT_MULTIMODAL_QRELS:
        qrels_file = Path(pseudolabel_stats.qrels_verified_path).name

    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"multimodal_{timestamp}"
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load qrels (keyed by case_id) from SPLIT_DIR.
    qrels_path = SPLIT_DIR / qrels_file
    with open(qrels_path) as f:
        original_qrels = json.load(f)

    qpath = _resolve_queries_path(queries_file, dataset_pack)

    if not qpath.exists():
        raise FileNotFoundError(f"Queries file not found: {qpath}")

    evaluation_contract = _build_evaluation_contract(
        dataset_pack=dataset_pack,
        qrels_file=qrels_file,
        qpath=qpath,
        image_search_mode=image_search_mode,
    )
    contract_label = "baseline-equivalent" if evaluation_contract["baseline_equivalent"] else "non-equivalent"
    print(
        "Evaluation contract: "
        f"{contract_label} "
        f"(qrels={qrels_file}, queries={qpath.name}, image_search={image_search_mode})"
    )
    for warning in evaluation_contract["warnings"]:
        print(f"Warning: {warning}")

    strict_resources = is_strict_retrieval_mode(DATASET_VERSION)
    lane1 = Lane1Retriever(strict_resources=strict_resources)
    lane2 = Lane2Retriever(strict_resources=strict_resources)
    retrieval_contract = _build_retrieval_contract(
        lane1=lane1,
        lane2=lane2,
        qpath=qpath,
        qrels_path=qrels_path,
        query_types=query_types,
        method=method,
        rerank=rerank,
        image_search_mode=image_search_mode,
        strip_query_images=strip_query_images,
        ablation_scope=ablation_scope,
    )
    _validate_retrieval_contract(retrieval_contract)
    
    with open(qpath) as f:
        all_queries = [json.loads(l) for l in f]
    
    # Filter queries by type AND existence in qrels

    queries = [
        q for q in all_queries 
        if q["query_type"] in query_types and q["case_id"] in original_qrels
    ]
    
    # CRITICAL FIX: Expand qrels to use unique qid (case_id::query_type)
    # This prevents Q1/Q3 from overwriting each other (per GPT 5.2)
    expanded_qrels = _expand_qrels_for_queries(original_qrels, queries)

    # Optional secondary qrels track for pseudolabel analysis.
    # This enables dual reporting without changing the primary metrics contract.
    expanded_qrels_pseudolabel = None
    pseudo_qrels_path = _resolve_secondary_pseudolabel_qrels_path(pseudolabel_stats)
    if pseudo_qrels_path.exists() and qrels_file != pseudo_qrels_path.name:
        with open(pseudo_qrels_path) as f:
            pseudo_qrels = json.load(f)
        expanded_qrels_pseudolabel = _expand_qrels_for_queries(pseudo_qrels, queries)
    
    # Load train cases (fixed: single JSON parse per line)
    with open(TRAIN_JSONL) as f:
        train_cases = {}
        for line in f:
            obj = json.loads(line)
            train_cases[obj["case_id"]] = obj

    baseline_train_case_ids: Optional[set[str]] = None
    train_override_active = Path(TRAIN_JSONL).resolve() != Path(TRAIN_NORMALIZED).resolve()
    if train_override_active and Path(TRAIN_NORMALIZED).exists():
        with open(TRAIN_NORMALIZED) as f:
            baseline_train_case_ids = {
                json.loads(line)["case_id"]
                for line in f
                if line.strip()
            }
    
    # Load test cases for ground truth and query images (per GPT 5.2)
    with open(TEST_JSONL) as f:
        test_cases = {}
        for line in f:
            obj = json.loads(line)
            test_cases[obj["case_id"]] = obj
    
    normalized_queries = []
    for q in queries:
        case_id = q["case_id"]
        test_case = test_cases.get(case_id)
        query_images = resolve_case_image_paths(test_case, max_images=5)
        if not query_images:
            fallback_inputs = list(q.get("query_images") or [])
            if q.get("image_path"):
                fallback_inputs.append(q["image_path"])
            query_images = normalize_query_image_paths(
                case_id=case_id,
                query_images=fallback_inputs,
                max_images=5,
            )
        normalized_q = dict(q)
        if strip_query_images:
            normalized_q["query_images"] = []
            normalized_q["image_path"] = None
            normalized_q["query_images_stripped"] = bool(query_images)
        else:
            normalized_q["query_images"] = query_images
            normalized_q["image_path"] = query_images[0] if query_images else None
            normalized_q["query_images_stripped"] = False
        normalized_queries.append(normalized_q)
    queries = normalized_queries

    # Save config
    runtime_metadata = get_runtime_metadata()
    dataset_support = get_dataset_support_snapshot()
    config = MultimodalRunConfig(
        run_id=run_id,
        qrels_file=qrels_file,
        query_types=query_types,
        retriever_method=method,
        rerank=rerank,
        k_values=k_values,
        created_at=datetime.now().isoformat(),
        image_search_mode=image_search_mode,
        agentic_lite=agentic_lite,
        queries_file=str(qpath),
        dataset_pack=dataset_pack,
        context_k=context_k,
        retrieval_top_k=retrieval_top_k,
        ordering_mode=ordering_mode,
    )
    run_config_payload = asdict(config)
    run_config_payload["context_k"] = context_k
    run_config_payload["retrieval_top_k"] = retrieval_top_k
    run_config_payload["ordering_mode"] = ordering_mode
    run_config_payload["experiment_controls"] = {
        "context_k": context_k,
        "retrieval_top_k": retrieval_top_k,
        "ordering_mode": ordering_mode,
    }
    run_config_payload["runtime_metadata"] = runtime_metadata
    run_config_payload["corpus_support"] = dataset_support
    run_config_payload["evaluation_contract"] = evaluation_contract
    run_config_payload["retrieval_contract"] = retrieval_contract
    run_config_payload["ablation_scope"] = ablation_scope
    run_config_payload["strip_query_images"] = strip_query_images
    if pseudolabel_stats is not None:
        run_config_payload["pseudolabel_artifacts"] = {
            "dataset_version": pseudolabel_stats.dataset_version,
            "train_source": pseudolabel_stats.train_source,
            "test_source": pseudolabel_stats.test_source,
            "suffix": pseudolabel_stats.suffix,
            "output_dir": pseudolabel_stats.output_dir,
            "train_path": pseudolabel_stats.train_path,
            "test_path": pseudolabel_stats.test_path,
            "query_path": pseudolabel_stats.query_path,
            "query_mixed56_path": pseudolabel_stats.query_mixed56_path,
            "qrels_verified_path": pseudolabel_stats.qrels_verified_path,
            "qrels_pseudolabel_path": pseudolabel_stats.qrels_pseudolabel_path,
        }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config_payload, f, indent=2)
    
    # Save queries
    with open(run_dir / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)

    reranker_model = get_medcpt_reranker() if rerank else None
    
    # Run retrieval
    all_retrieved = {}
    retrieval_records = []
    rerank_query_count = 0
    rerank_pass_count = 0
    
    # Enhanced statistics with skip reasons (per GPT 5.2)
    stats = {
        "Q1": {"attempted": 0, "success": 0},
        "Q2": {"attempted": 0, "success": 0},
        "Q3": {"attempted": 0, "success": 0},
        "skip_reasons": {
            "no_text": 0,
            "no_image_path": 0,
            "image_not_found": 0,
            "image_search_disabled": 0,
            "no_results": 0
        }
    }
    
    for q in queries:
        case_id = q["case_id"]
        query_type = q["query_type"]
        # Support both legacy (text) and new (clinical_context) field naming
        query_text = q.get("text") or q.get("clinical_context") or q.get("formatted_query", "")
        query_images_stripped = bool(q.get("query_images_stripped", False))
        # Support both legacy (image_path) and new (query_images) field naming
        image_path = q.get("image_path")
        if not image_path:
            query_images_list = q.get("query_images", [])
            image_path = query_images_list[0] if query_images_list else None
        
        # CRITICAL FIX: Use unique qid to prevent collision (per GPT 5.2)
        qid = f"{case_id}::{query_type}"
        
        results = []
        stage = "unknown"
        skip_reason = None
        used_dense_lane = False
        used_bm25_lane = False
        used_caption_lane = False
        used_image_lane = False
        rerank_applied = False
        agentic_trace = {
            "enabled": agentic_lite,
            "triggered": False,
            "reason": "disabled",
            "refined_query": "",
        }
        
        # === Q3: Image-only query (TRUE MULTIMODAL) ===
        # Support both legacy (Q3_image_only) and new (Q3_image_diagnosis) naming
        if query_type in ["Q3_image_only", "Q3_image_diagnosis"]:
            stats["Q3"]["attempted"] += 1

            if image_search_mode == "none":
                stats["skip_reasons"]["image_search_disabled"] += 1
                skip_reason = "image_search_disabled"
            elif not image_path:
                stats["skip_reasons"]["no_image_path"] += 1
                skip_reason = "no_image_path"
            elif not Path(image_path).exists():
                stats["skip_reasons"]["image_not_found"] += 1
                skip_reason = f"image_not_found: {image_path}"
            else:
                # Use image encoder to retrieve
                search_captions = (image_search_mode == "captions")
                raw_results = lane2.retrieve_by_image_path(
                    image_path, 
                    top_k=retrieval_top_k,
                    search_captions=search_captions
                )
                if search_captions:
                    used_caption_lane = True
                else:
                    used_image_lane = True
                results = [(cid, score) for cid, score, _ in raw_results]
                stage = f"biomedclip_image→{image_search_mode}"
                
                if results:
                    stats["Q3"]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # === Q1/Q2: Text query ===
        # Support both legacy (Q1_symptom_only) and new (Q1_diagnosis) naming
        elif query_type in ["Q1_symptom_only", "Q2_symptom_exposure", "Q1_diagnosis", "Q2_diagnosis_exposure"]:
            stat_key = "Q1" if "Q1" in query_type else "Q2"
            stats[stat_key]["attempted"] += 1
            
            if not query_text:
                stats["skip_reasons"]["no_text"] += 1
                skip_reason = "no_text"
            else:
                if method == "bm25":
                    results = lane1.retrieve_bm25(query_text, top_k=retrieval_top_k)
                    used_bm25_lane = True
                elif method == "e5":
                    results = lane1.retrieve_e5(query_text, top_k=retrieval_top_k)
                    used_dense_lane = True
                elif method == "hybrid":
                    results = lane1.retrieve_hybrid(query_text, top_k=retrieval_top_k)
                    used_dense_lane = True
                    used_bm25_lane = True
                elif method == "2lane":
                    l1_results = lane1.retrieve_hybrid(query_text, top_k=retrieval_top_k)
                    used_dense_lane = True
                    used_bm25_lane = True
                    l2_results = lane2.retrieve_by_caption(query_text, top_k=retrieval_top_k)
                    used_caption_lane = True
                    l2_results = [(c, s) for c, s, _ in l2_results]
                    results = rrf_fusion(l1_results, l2_results)[:retrieval_top_k]
                else:
                    results = lane1.retrieve_hybrid(query_text, top_k=retrieval_top_k)
                    used_dense_lane = True
                    used_bm25_lane = True
                
                stage = method
                
                # Rerank if requested (text queries only)
                if rerank and reranker_model and results:
                    candidates = []
                    for cid, score in results:
                        if cid in train_cases:
                            doc_text = train_cases[cid].get("case_text", "")[:512]
                            candidates.append((cid, doc_text, score))
                    
                    if candidates:
                        reranked = reranker_model.rerank(query_text, candidates, top_k=retrieval_top_k)
                        results = [(c, s) for c, _, s in reranked]
                        stage = f"{method}+rerank"
                        rerank_applied = True
                        rerank_query_count += 1
                        rerank_pass_count += 1

                if agentic_lite and query_text and results:
                    refine, reason = _should_refine_agentic(query_type, results)
                    if refine:
                        refined_query = _rewrite_query_agentic_lite(query_text)
                        refined_results = lane1.retrieve_hybrid(refined_query, top_k=retrieval_top_k)
                        used_dense_lane = True
                        used_bm25_lane = True
                        if rerank and reranker_model and refined_results:
                            candidates = []
                            for cid, score in refined_results:
                                if cid in train_cases:
                                    doc_text = train_cases[cid].get("case_text", "")[:512]
                                    candidates.append((cid, doc_text, score))
                            if candidates:
                                reranked = reranker_model.rerank(refined_query, candidates, top_k=retrieval_top_k)
                                refined_results = [(c, s) for c, _, s in reranked]
                                rerank_applied = True
                                rerank_pass_count += 1

                        if refined_results:
                            results = rrf_fusion(results, refined_results)[:retrieval_top_k]
                            stage = f"{stage}+agentic_refine"
                            agentic_trace = {
                                "enabled": True,
                                "triggered": True,
                                "reason": reason,
                                "refined_query": refined_query[:300],
                            }
                        else:
                            agentic_trace = {
                                "enabled": True,
                                "triggered": True,
                                "reason": f"{reason}; refinement_no_results",
                                "refined_query": refined_query[:300],
                            }
                    else:
                        agentic_trace = {
                            "enabled": True,
                            "triggered": False,
                            "reason": reason,
                            "refined_query": "",
                        }
                
                if results:
                    stats[stat_key]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # === Q1_Q3: Combined Multimodal Query (per Claude 4.5 + Grok 4.1) ===
        elif query_type == "Q1_Q3_multimodal_diagnosis":
            # Initialize multimodal stats if not present
            if "MULTIMODAL" not in stats:
                stats["MULTIMODAL"] = {"attempted": 0, "success": 0}
            stats["MULTIMODAL"]["attempted"] += 1
            
            # Get image path from query if available
            query_images_list = q.get("query_images", [])
            first_image = query_images_list[0] if query_images_list else None
            
            if not query_text and not first_image:
                stats["skip_reasons"]["no_text"] += 1
                skip_reason = "no_text_or_image"
            else:
                # Combine text and image retrieval using 2-lane fusion
                text_results = []
                image_results = []
                
                # Lane 1: Text retrieval
                if query_text:
                    text_results = lane1.retrieve_hybrid(query_text, top_k=retrieval_top_k)
                    used_dense_lane = True
                    used_bm25_lane = True
                    if rerank and reranker_model and text_results:
                        candidates = []
                        for cid, score in text_results:
                            if cid in train_cases:
                                doc_text = train_cases[cid].get("case_text", "")[:512]
                                candidates.append((cid, doc_text, score))
                        if candidates:
                            reranked = reranker_model.rerank(query_text, candidates, top_k=retrieval_top_k)
                            text_results = [(c, s) for c, _, s in reranked]
                            rerank_applied = True
                            rerank_query_count += 1
                            rerank_pass_count += 1
                
                # Lane 2: Image retrieval (if image available)
                if image_search_mode != "none" and first_image and Path(first_image).exists():
                    raw_img_results = lane2.retrieve_by_image_path(
                        first_image, 
                        top_k=retrieval_top_k,
                        search_captions=(image_search_mode == "captions")
                    )
                    if image_search_mode == "captions":
                        used_caption_lane = True
                    else:
                        used_image_lane = True
                    image_results = [(cid, score) for cid, score, _ in raw_img_results]
                
                # Combine with RRF fusion
                if text_results and image_results:
                    results = rrf_fusion(text_results, image_results)[:retrieval_top_k]
                    stage = "hybrid+biomedclip_fusion"
                elif text_results:
                    results = text_results[:retrieval_top_k]
                    stage = "hybrid_text_only"
                elif image_results:
                    results = image_results[:retrieval_top_k]
                    stage = "biomedclip_image_only"
                else:
                    results = []

                if agentic_lite and query_text and results:
                    refine, reason = _should_refine_agentic(query_type, results)
                    if refine:
                        refined_query = _rewrite_query_agentic_lite(query_text)
                        refined_text_results = lane1.retrieve_hybrid(refined_query, top_k=retrieval_top_k)
                        used_dense_lane = True
                        used_bm25_lane = True
                        if rerank and reranker_model and refined_text_results:
                            candidates = []
                            for cid, score in refined_text_results:
                                if cid in train_cases:
                                    doc_text = train_cases[cid].get("case_text", "")[:512]
                                    candidates.append((cid, doc_text, score))
                            if candidates:
                                reranked = reranker_model.rerank(refined_query, candidates, top_k=retrieval_top_k)
                                refined_text_results = [(c, s) for c, _, s in reranked]
                                rerank_applied = True
                                rerank_pass_count += 1
                        if refined_text_results:
                            results = rrf_fusion(results, refined_text_results)[:retrieval_top_k]
                            stage = f"{stage}+agentic_refine"
                        agentic_trace = {
                            "enabled": True,
                            "triggered": True,
                            "reason": reason,
                            "refined_query": refined_query[:300],
                        }
                    else:
                        agentic_trace = {
                            "enabled": True,
                            "triggered": False,
                            "reason": reason,
                            "refined_query": "",
                        }
                
                if results:
                    stats["MULTIMODAL"]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # Skip if no results or error
        if not results:
            if skip_reason:
                resource_trace = {
                    "dense_collection_used": lane1.collection_name if used_dense_lane else None,
                    "bm25_index_path_used": (
                        lane1.get_resource_contract()["bm25_index"].get("resolved_path")
                        if used_bm25_lane else None
                    ),
                    "caption_collection_used": lane2.caption_collection if used_caption_lane else None,
                    "image_collection_used": lane2.image_collection if used_image_lane else None,
                    "rerank_applied": rerank_applied,
                    "resource_events": list(lane1.resource_events),
                }
                retrieval_records.append({
                    "qid": qid,
                    "query_type": query_type,
                    "query": (query_text[:100] if query_text else f"[IMAGE]"),
                    "skip_reason": skip_reason,
                    "contexts": [],
                    "stage": "skipped",
                    "resource_trace": resource_trace,
                    "ablation_scope": ablation_scope,
                    "query_images_stripped": query_images_stripped,
                })
            continue
        
        # Store results with unique qid
        all_retrieved[qid] = [c for c, _ in results]
        
        # Build retrieval record with SEPARATE query vs context images (per GPT 5.2)
        contexts = []
        context_images = []
        for cid, score in results[:10]:
            if cid in train_cases:
                ctx_case = train_cases[cid]
                # Use query-focused snippet extraction (per GPT 5.2)
                snippet = extract_query_focused_snippet(
                    ctx_case.get("case_text", ""),
                    query_text or "",
                    max_chars=1500  # Increased from 500
                )
                contexts.append({
                    "doc_id": cid,
                    "score": float(score),
                    "text": snippet,
                    "meta": {"has_images": len(ctx_case.get("images", [])) > 0}
                })
                context_images.extend(resolve_case_image_paths(ctx_case, max_images=5))
        
        # APPLY TYPE-AWARE SOFT RERANK (per GPT 5.2 Level B)
        # Reduces harm from wrong-subtype contexts without hard filtering
        # DISABLED: Investigation shows this may hurt more than help
        # TODO: Fix subtype inference before re-enabling
        # if contexts and query_text:
        #     contexts = soft_rerank_by_subtype(contexts, query_text, train_cases)
        
        # Extract query images and ground truth from TEST case (per GPT 5.2)
        query_images = list(q.get("query_images") or [])
        ground_truth = None
        ground_truth_pseudolabel = None
        if case_id in test_cases:
            test_case = test_cases[case_id]
            # Ground truth
            ground_truth = {
                "diagnosis": test_case.get("diagnosis", ""),
                "diagnosis_type": test_case.get("diagnosis_type", ""),
                "species": test_case.get("species", "")
            }

            labels = test_case.get("labels", {})
            if isinstance(labels, dict):
                pseudo_label = labels.get("pseudolabel")
                if isinstance(pseudo_label, dict):
                    ground_truth_pseudolabel = {
                        "diagnosis": pseudo_label.get("diagnosis", ""),
                        "diagnosis_type": pseudo_label.get("diagnosis_type", ""),
                        "species": pseudo_label.get("species", ""),
                    }

        # Fallback for non-TEST packs (e.g., mixed56) where ground truth is present in query JSON.
        if ground_truth is None:
            q_gt = q.get("ground_truth")
            if isinstance(q_gt, dict):
                ground_truth = {
                    "diagnosis": q_gt.get("diagnosis", ""),
                    "diagnosis_type": q_gt.get("diagnosis_type", ""),
                    "species": q_gt.get("species", ""),
                }

        if ground_truth_pseudolabel is None:
            q_gt_pseudo = q.get("ground_truth_pseudolabel")
            if isinstance(q_gt_pseudo, dict):
                ground_truth_pseudolabel = {
                    "diagnosis": q_gt_pseudo.get("diagnosis", ""),
                    "diagnosis_type": q_gt_pseudo.get("diagnosis_type", ""),
                    "species": q_gt_pseudo.get("species", ""),
                }
        
        # Build proper query for RAGAS: canonical question + clinical context
        # Use canonical full taxonomy wording to avoid legacy 4-type drift from stale eval files.
        raw_question = q.get("question", "What is the diagnosis?")
        if query_type in ["Q1_diagnosis", "Q2_diagnosis_exposure", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"]:
            question = DIAGNOSIS_QUESTION_WITH_TYPE
        else:
            question = raw_question
        full_query_for_ragas = f"{question}\n\nClinical Context: {query_text[:300]}" if query_text else question
        
        retrieval_records.append({
            "qid": qid,
            "query_type": query_type,
            "query": full_query_for_ragas,  # FIXED: Now includes question for RAGAS
            "clinical_context": query_text[:200] if query_text else "",  # Preserved for reference
            "raw_question": raw_question,
            "resolved_question": question,
            "contexts": contexts,
            "query_images": query_images,        # NEW: from TEST case
            "context_images": context_images[:5], # NEW: from TRAIN cases
            "ground_truth": ground_truth,         # NEW: for diagnosis accuracy
            "ground_truth_pseudolabel": ground_truth_pseudolabel,
            "stage": stage,
            "resource_trace": {
                "dense_collection_used": lane1.collection_name if used_dense_lane else None,
                "bm25_index_path_used": (
                    lane1.get_resource_contract()["bm25_index"].get("resolved_path")
                    if used_bm25_lane else None
                ),
                "caption_collection_used": lane2.caption_collection if used_caption_lane else None,
                "image_collection_used": lane2.image_collection if used_image_lane else None,
                "rerank_applied": rerank_applied,
                "resource_events": list(lane1.resource_events),
            },
            "agentic_trace": agentic_trace,
            "ablation_scope": ablation_scope,
            "query_images_stripped": query_images_stripped,
        })
    
    # Save retrieval.jsonl
    with open(run_dir / "retrieval.jsonl", "w") as f:
        for rec in retrieval_records:
            f.write(json.dumps(rec) + "\n")
    
    # Evaluate using expanded qrels (with unique qids)
    eval_results = evaluate_retrieval(all_retrieved, expanded_qrels, k_values, return_per_query=True)
    eval_results_pseudolabel = None
    if expanded_qrels_pseudolabel is not None:
        eval_results_pseudolabel = evaluate_retrieval(
            all_retrieved,
            expanded_qrels_pseudolabel,
            k_values,
            return_per_query=False,
        )

    active_verified_case_qrels = _build_case_qrels(train_cases, test_cases, label_key="verified")
    active_verified_qrels_audit = _primary_qrels_audit(
        train_cases=train_cases,
        primary_qrels=original_qrels,
        baseline_train_case_ids=baseline_train_case_ids,
    )

    eval_results_active_verified = None
    if active_verified_case_qrels != original_qrels:
        eval_results_active_verified = evaluate_retrieval(
            all_retrieved,
            _expand_qrels_for_queries(active_verified_case_qrels, queries),
            k_values,
            return_per_query=False,
        )
    
    # Save per-query metrics
    with open(run_dir / "metrics_per_query.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["qid", "query_type"] + [f"recall@{k}" for k in k_values] + \
                 [f"ndcg@{k}" for k in k_values] + \
                 [f"precision@{k}" for k in k_values] + ["mrr", "ap"]
        writer.writerow(header)
        
        for qid_full, metrics in (eval_results.per_query_metrics or {}).items():
            # Extract query_type from qid
            parts = qid_full.split("::")
            qtype = parts[1] if len(parts) > 1 else "unknown"
            row = [qid_full, qtype]
            for k in k_values:
                row.append(f"{metrics.get(f'recall@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'ndcg@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'precision@{k}', 0):.4f}")
            row.append(f"{metrics.get('mrr', 0):.4f}")
            row.append(f"{metrics.get('ap', 0):.4f}")
            writer.writerow(row)
    
    # Save summary (grounded_accuracy added per Grok 4.1 recommendation)
    summary = {
        "run_id": run_id,
        "query_types": query_types,
        "query_stats": stats,
        "n_queries": len(queries),
        "n_retrieved": len(all_retrieved),
        "runtime_metadata": runtime_metadata,
        "corpus_support": dataset_support,
        "evaluation_contract": evaluation_contract,
        "ablation_scope": ablation_scope,
        "strip_query_images": strip_query_images,
        "context_k": context_k,
        "retrieval_top_k": retrieval_top_k,
        "ordering_mode": ordering_mode,
        "experiment_controls": {
            "context_k": context_k,
            "retrieval_top_k": retrieval_top_k,
            "ordering_mode": ordering_mode,
        },
        "silver_label_disclaimer": SILVER_LABEL_DISCLAIMER,
        "qrels_audit": active_verified_qrels_audit,
        "metrics": {
            **_format_metrics(eval_results, k_values),
            "grounded_accuracy": None,
        },
        "metrics_verified": _format_metrics(eval_results, k_values),
    }

    if pseudolabel_stats is not None:
        summary["pseudolabel_artifacts"] = {
            "dataset_version": pseudolabel_stats.dataset_version,
            "train_source": pseudolabel_stats.train_source,
            "test_source": pseudolabel_stats.test_source,
            "suffix": pseudolabel_stats.suffix,
            "output_dir": pseudolabel_stats.output_dir,
            "query_path": pseudolabel_stats.query_path,
            "query_mixed56_path": pseudolabel_stats.query_mixed56_path,
            "qrels_verified_path": pseudolabel_stats.qrels_verified_path,
            "qrels_pseudolabel_path": pseudolabel_stats.qrels_pseudolabel_path,
        }

    if eval_results_pseudolabel is not None:
        summary["metrics_pseudolabel"] = _format_metrics(eval_results_pseudolabel, k_values)

    if eval_results_active_verified is not None:
        summary["metrics_active_corpus_verified"] = _format_metrics(
            eval_results_active_verified,
            k_values,
        )

    retrieval_usage = _build_retrieval_usage_summary(
        lane1=lane1,
        lane2=lane2,
        rerank_query_count=rerank_query_count,
        rerank_pass_count=rerank_pass_count,
    )
    caption_support = dict(retrieval_contract.get("caption_support") or {})
    caption_support["lane_exercised"] = bool(
        retrieval_usage["resources_exercised"].get("caption_lane")
    )
    caption_support["collection_exists"] = bool(
        (retrieval_contract.get("collections_at_start") or {})
        .get("caption_collection", {})
        .get("exists")
    )
    if caption_support["collection_exists"]:
        caption_support["absence_reason"] = None
    retrieval_contract["caption_support"] = caption_support
    retrieval_contract["usage"] = retrieval_usage
    retrieval_contract["resource_events"] = list(lane1.resource_events)
    summary["retrieval_contract"] = retrieval_contract
    summary["retrieval_usage"] = retrieval_usage
    summary["caption_support"] = caption_support
    run_config_payload["retrieval_contract"] = retrieval_contract
    run_config_payload["caption_support"] = caption_support

    for usage_key, expected in retrieval_contract["usage_expectations"].items():
        if expected and not retrieval_usage["resources_exercised"].get(usage_key, False):
            raise RuntimeError(
                f"Retrieval run did not actually exercise required resource path: {usage_key}"
            )
    
    # NOTE: RAGAS metrics are added separately via rag/update_summary_ragas.py
    # This keeps the pipeline modular - run retrieval first, then RAGAS evaluation separately

    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config_payload, f, indent=2)
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results (with safe key access)
    print(f"\n{'='*60}")
    print(f"MULTIMODAL EVALUATION: {run_id}")
    print(f"{'='*60}")
    print(f"Query Types: {query_types}")
    print(f"Query Stats:")
    for qt in ["Q1", "Q2", "Q3", "MULTIMODAL"]:
        if qt in stats:
            print(f"  {qt}: {stats[qt]['success']}/{stats[qt]['attempted']} successful")
    print(f"Skip Reasons: {stats['skip_reasons']}")
    print(f"Method: {method} (rerank={rerank})")
    print(
        "Resource usage: "
        f"dense={retrieval_usage['dense_lane_query_count']} "
        f"bm25={retrieval_usage['bm25_lane_query_count']} "
        f"caption={retrieval_usage['caption_query_count']} "
        f"image={retrieval_usage['image_query_count']} "
        f"rerank={retrieval_usage['rerank_query_count']}"
    )
    print(f"Primary qrels coverage gap: {active_verified_qrels_audit['train_docs_missing_from_primary_qrels_count']} train docs not referenced")
    print(f"\nMetrics:")
    for k in k_values:
        print(f"  nDCG@{k}:     {eval_results.ndcg_at_k.get(k, 0):.4f}")
    print(f"  MRR:        {eval_results.mrr:.4f}")
    if eval_results_active_verified is not None:
        print("  Active-corpus verified metrics also saved to summary.json")
    print(f"\n✓ Run saved to {run_dir}")
    
    # Auto-update catalog (per GPT 5.2)
    try:
        from .run_catalog import update_catalog
        update_catalog()
    except Exception as e:
        print(f"Note: Could not update catalog: {e}")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal RAG Evaluation")
    parser.add_argument(
        "--qrels",
        default=DEFAULT_MULTIMODAL_QRELS,
        help=f"QRELs file (default: {DEFAULT_MULTIMODAL_QRELS}, baseline-equivalent)",
    )
    parser.add_argument("--query-types", nargs="+", 
                        default=None,
                        help="Query types to evaluate (default: Q1_Q3 multimodal only)")
    parser.add_argument("--method", default="hybrid", 
                        choices=["bm25", "e5", "hybrid", "2lane"],
                        help="Retrieval method for text queries")
    parser.add_argument("--rerank", action="store_true", help="Use MedCPT reranking")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    parser.add_argument("--queries-file", default=None,
                        help="Explicit eval queries JSONL path (overrides dataset-pack)")
    parser.add_argument("--dataset-pack", choices=["auto", "test", "mixed56"], default="auto",
                        help="Predefined eval query pack; auto resolves to the versioned pseudolabel pack when present")
    parser.add_argument("--pseudolabel-train-results", default=None,
                        help="Optional override train pseudolabel results.jsonl source path")
    parser.add_argument("--pseudolabel-test-results", default=None,
                        help="Optional override test pseudolabel results.jsonl source path")
    parser.add_argument("--pseudolabel-suffix", default="",
                        help="Optional suffix for versioned pseudolabel artifacts")
    parser.add_argument("--pseudolabel-force", action="store_true",
                        help="Force pseudolabel artifact rebuild even when outputs are up to date")
    parser.add_argument("--image-search", default=DEFAULT_MULTIMODAL_IMAGE_SEARCH, 
                        choices=["captions", "images", "none"],
                        help=f"Collection to search for Q3 image queries (default: {DEFAULT_MULTIMODAL_IMAGE_SEARCH}, baseline-equivalent)")
    parser.add_argument("--strip-query-images", action="store_true",
                        help="Retrieval-stage image-off ablation: strip query images before retrieval")
    parser.add_argument("--ablation-scope", default="",
                        help="Optional ablation scope label stored in retrieval artifacts")
    parser.add_argument("--context-k", type=int, default=None,
                        help="Optional explicit prompt context budget to stamp into run artifacts")
    parser.add_argument("--retrieval-top-k", type=int, default=20,
                        help="Top-k candidate pool size used during retrieval and rerank stages")
    parser.add_argument("--ordering-mode", choices=["image_first", "text_first", "interleaved"], default="image_first",
                        help="Intended multimodal ordering mode to stamp for downstream generation experiments")
    parser.add_argument(
        "--agentic-lite",
        action="store_true",
        help="Enable one-step retrieve-evaluate-refine retrieval loop",
    )
    
    args = parser.parse_args()
    
    run_multimodal_evaluation(
        qrels_file=args.qrels,
        query_types=args.query_types,
        method=args.method,
        rerank=args.rerank,
        run_id=args.run_id,
        image_search_mode=args.image_search,
        agentic_lite=args.agentic_lite,
        queries_file=args.queries_file,
        dataset_pack=args.dataset_pack,
        pseudolabel_train_results=args.pseudolabel_train_results,
        pseudolabel_test_results=args.pseudolabel_test_results,
        pseudolabel_suffix=args.pseudolabel_suffix,
        pseudolabel_force=args.pseudolabel_force,
        strip_query_images=args.strip_query_images,
        ablation_scope=(args.ablation_scope or ("retrieval_query_image_strip" if args.strip_query_images else "")),
        context_k=args.context_k,
        retrieval_top_k=args.retrieval_top_k,
        ordering_mode=args.ordering_mode,
    )
