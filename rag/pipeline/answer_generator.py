"""
Answer Generator - Main Module.

Orchestrates answer generation for RAG evaluation.
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from configs.prompt_mode import (
    PromptMode,
    RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES,
    RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION,
    RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES,
    RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION,
    build_rag_prompt as build_shared_rag_prompt,
)

from . import ADAPTIVE_RAG, IMAGES_DIR, RUNS_DIR, TRAIN_JSONL, get_dataset_support_snapshot
from .confirmatory_signals import (
    context_has_confirmatory_signal as _context_has_confirmatory_signal,
    context_support_score as _context_support_score,
    query_has_confirmatory_signal as _query_has_confirmatory_signal,
    query_has_leish_signal as _query_has_leish_signal,
)
from .diagnosis_output_parser import analyze_answer_format
from .generators import GeminiGenerator, Gemma3Generator, Gemma4Generator, MedGemmaGenerator, QwenVLGenerator
RAW_QUERY_PROMPT_CONTRACT_VERSION = "raw_query_passthrough_v1"
RAW_QUERY_PROMPT_CONTRACT_NOTES = "no_shared_prompt_contract_applied"
STRICT_CONTEXT_PROMPT_CONTRACT_VERSION = "strict_context_shared_prompt_v1"
STRICT_CONTEXT_PROMPT_CONTRACT_NOTES = "shared_strict_context_prompt_contract"
NO_CONTEXT_SHARED_PROMPT_CONTRACT_VERSION = "no_context_shared_prompt_v1"
NO_CONTEXT_SHARED_PROMPT_CONTRACT_NOTES = "shared_no_context_prompt_contract"


@dataclass
class AnswerRecord:
    """Single answer record for answers.jsonl."""

    qid: str
    query: str
    contexts: List[Dict]
    answer: str
    model_name: str
    citations: List[str] = field(default_factory=list)
    query_images: List[str] = field(default_factory=list)
    context_images: List[str] = field(default_factory=list)
    ground_truth: Optional[Dict] = None
    ground_truth_pseudolabel: Optional[Dict] = None
    image_paths: List[str] = field(default_factory=list)
    decoding_params: Dict = field(default_factory=dict)
    gating_info: str = ""
    query_type: str = ""
    top_score: float = 0.0
    threshold_used: float = 0.0
    generation_mode: str = "rag_prompt"
    retrieval_support_status: str = "supported"
    prompt_context_doc_ids: List[str] = field(default_factory=list)
    prompt_context_count: int = 0
    format_retry_count: int = 0
    answer_format_valid: Optional[bool] = None
    answer_format_error: str = ""
    ablation_scope: str = ""
    query_images_stripped: bool = False
    prompt_contract_version: str = ""
    prompt_contract_notes: str = ""
    prompt_image_grounding_required: bool = False
    context_k: Optional[int] = None
    ordering_mode: str = ""
    use_context_image_tensors: bool = False
    support_image_tensor_budget: int = 0
    query_image_tensor_attempt_count: int = 0
    support_image_tensor_attempt_count: int = 0
    query_image_tensor_count: int = 0
    support_image_tensor_count: int = 0
    image_tensor_fallback_used: bool = False
    image_tensor_fallback_reason: str = ""


def enrich_contexts_with_images(contexts: List[Dict], train_cases: Dict) -> Tuple[List[Dict], List[str]]:
    """Add train-case image paths to already-selected contexts."""
    all_image_paths: List[str] = []
    enriched: List[Dict] = []

    for ctx in contexts:
        new_ctx = dict(ctx)
        doc_id = new_ctx.get("doc_id")
        ctx_images: List[str] = []
        if doc_id in train_cases:
            case = train_cases[doc_id]
            for img in case.get("images", []):
                filename = img.get("file") or img.get("file_name", "")
                if filename:
                    ctx_images.append(str(IMAGES_DIR / doc_id / filename))
        new_ctx["image_paths"] = ctx_images
        all_image_paths.extend(ctx_images)
        enriched.append(new_ctx)

    return enriched, all_image_paths


def extract_citations(answer: str, contexts: List[Dict]) -> List[str]:
    return [ctx["doc_id"] for ctx in contexts if ctx.get("doc_id") and ctx["doc_id"] in answer]


def _sample_has_query_images(sample: Dict) -> bool:
    query_images = sample.get("query_images")
    if isinstance(query_images, list) and len(query_images) > 0:
        return True
    if isinstance(query_images, str) and bool(query_images.strip()):
        return True

    image_paths = sample.get("image_paths")
    if isinstance(image_paths, list) and len(image_paths) > 0:
        return True
    if isinstance(image_paths, str) and bool(image_paths.strip()):
        return True

    return False


def _build_q3_medical_no_context_prompt(query: str, query_images: List[str]) -> str:
    """Build a medical no-context fallback only for RAG-arm Q3 image cases."""
    prompt = build_shared_rag_prompt(
        query=query,
        contexts=[],
        mode=PromptMode.NO_CONTEXT,
        query_images=query_images or None,
        context_images=None,
        max_chars_per_context=1800,
        include_context_images=True,
        is_text_only_model=False,
    )
    if "## VISUAL GROUNDING REQUIREMENT" in prompt:
        return prompt
    visual_guardrail = (
        "\n\n## VISUAL GROUNDING REQUIREMENT\n"
        "- Because patient images are attached, explicitly state at least one concrete patient-image finding that supports or limits your diagnosis\n"
        "- If the image is ambiguous, low-quality, or nondiagnostic, say so explicitly rather than inventing a visual feature\n"
        "- Do not claim support from retrieved literature or unseen evidence in this fallback mode\n"
    )
    anchor = "\nTASK:\n"
    if anchor in prompt:
        return prompt.replace(anchor, visual_guardrail + anchor, 1)
    return prompt + visual_guardrail


def _shared_prompt_contract_metadata(prompt_mode: Optional[PromptMode]) -> Tuple[str, str, bool]:
    if prompt_mode == PromptMode.BALANCED:
        return (
            RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION,
            RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES,
            True,
        )
    if prompt_mode == PromptMode.STRICT_CONTEXT:
        return (
            STRICT_CONTEXT_PROMPT_CONTRACT_VERSION,
            STRICT_CONTEXT_PROMPT_CONTRACT_NOTES,
            False,
        )
    if prompt_mode == PromptMode.NO_CONTEXT:
        return (
            NO_CONTEXT_SHARED_PROMPT_CONTRACT_VERSION,
            NO_CONTEXT_SHARED_PROMPT_CONTRACT_NOTES,
            False,
        )
    return RAW_QUERY_PROMPT_CONTRACT_VERSION, RAW_QUERY_PROMPT_CONTRACT_NOTES, False


def _resolve_prompt_contract_for_sample(
    requested_prompt_mode: Optional[PromptMode],
    query_type: str,
    query_images: List[str],
    generation_mode: str,
    generation_query: str,
    original_query: str,
) -> Dict[str, object]:
    if generation_mode == "rag_prompt":
        version, notes, image_grounding_required = _shared_prompt_contract_metadata(requested_prompt_mode)
        return {
            "version": version,
            "notes": notes,
            "image_grounding_required": image_grounding_required,
        }

    if (
        query_type == "Q3_image_diagnosis"
        and bool(query_images)
        and generation_query != original_query
    ):
        return {
            "version": RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION,
            "notes": RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES,
            "image_grounding_required": True,
        }

    return {
        "version": RAW_QUERY_PROMPT_CONTRACT_VERSION,
        "notes": RAW_QUERY_PROMPT_CONTRACT_NOTES,
        "image_grounding_required": False,
    }


def _build_prompt_contract_summary(records: List[Dict]) -> Dict[str, object]:
    if not records:
        return {
            "prompt_contract_version": "none_generated",
            "prompt_contract_notes": "no_answers_generated",
            "mixed_contracts": False,
            "by_version": {},
        }

    version_counts = Counter()
    by_version: Dict[str, Dict[str, object]] = {}
    for record in records:
        version = str(record.get("prompt_contract_version") or RAW_QUERY_PROMPT_CONTRACT_VERSION)
        version_counts[version] += 1
        entry = by_version.setdefault(
            version,
            {
                "count": 0,
                "notes": record.get("prompt_contract_notes") or "",
                "image_grounding_required_count": 0,
            },
        )
        entry["count"] += 1
        if record.get("prompt_image_grounding_required"):
            entry["image_grounding_required_count"] += 1

    versions = list(version_counts.keys())
    top_level_version = versions[0] if len(versions) == 1 else "mixed"
    top_level_notes = (
        by_version[versions[0]]["notes"]
        if len(versions) == 1
        else "multiple_prompt_contracts"
    )
    return {
        "prompt_contract_version": top_level_version,
        "prompt_contract_notes": top_level_notes,
        "mixed_contracts": len(versions) > 1,
        "by_version": by_version,
    }


def _build_multimodal_usage_summary(records: List[Dict]) -> Dict[str, object]:
    if not records:
        return {
            "rows_with_query_images": 0,
            "rows_with_context_images_available": 0,
            "rows_with_query_image_tensors": 0,
            "rows_with_support_image_tensors": 0,
            "mean_query_image_tensor_count": 0.0,
            "mean_support_image_tensor_count": 0.0,
            "image_tensor_fallback_count": 0,
            "true_multimodal_support_active": False,
        }

    rows_with_query_images = sum(1 for rec in records if rec.get("query_images"))
    rows_with_context_images = sum(1 for rec in records if rec.get("context_images"))
    rows_with_query_image_tensors = sum(1 for rec in records if int(rec.get("query_image_tensor_count") or 0) > 0)
    rows_with_support_image_tensors = sum(1 for rec in records if int(rec.get("support_image_tensor_count") or 0) > 0)
    query_tensor_total = sum(int(rec.get("query_image_tensor_count") or 0) for rec in records)
    support_tensor_total = sum(int(rec.get("support_image_tensor_count") or 0) for rec in records)
    image_tensor_fallback_count = sum(1 for rec in records if rec.get("image_tensor_fallback_used"))
    return {
        "rows_with_query_images": rows_with_query_images,
        "rows_with_context_images_available": rows_with_context_images,
        "rows_with_query_image_tensors": rows_with_query_image_tensors,
        "rows_with_support_image_tensors": rows_with_support_image_tensors,
        "mean_query_image_tensor_count": query_tensor_total / len(records),
        "mean_support_image_tensor_count": support_tensor_total / len(records),
        "image_tensor_fallback_count": image_tensor_fallback_count,
        "true_multimodal_support_active": rows_with_support_image_tensors > 0,
    }


def _resolve_generation_setting(
    run_dir: Path,
    explicit_value,
    key: str,
    default=None,
):
    if explicit_value is not None:
        return explicit_value
    config_path = run_dir / "run_config.json"
    if not config_path.exists():
        return default
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return default
    if key in payload and payload[key] is not None:
        return payload[key]
    experiment_controls = payload.get("experiment_controls") or {}
    if key in experiment_controls and experiment_controls[key] is not None:
        return experiment_controls[key]
    return default


def _load_json_dict(path: Path) -> Dict[str, object]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_inherited_retrieval_top_k(run_dir: Path) -> Optional[int]:
    candidate_paths = [run_dir / "run_config.json", run_dir / "summary.json"]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        try:
            payload = _load_json_dict(candidate_path)
        except Exception:
            continue
        direct_value = payload.get("retrieval_top_k")
        if direct_value is not None:
            return int(direct_value)
        experiment_controls = payload.get("experiment_controls") or {}
        inherited_value = experiment_controls.get("retrieval_top_k")
        if inherited_value is not None:
            return int(inherited_value)
    return None


def _prune_contexts_for_generation(
    contexts: List[Dict],
    generator_type: str,
    context_k: Optional[int] = None,
) -> List[Dict]:
    if not contexts:
        return []

    if context_k is not None:
        max_contexts = max(1, int(context_k))
    else:
        max_contexts = 4 if generator_type == "medgemma" else 8
    scored = []
    for idx, ctx in enumerate(contexts):
        text = ctx.get("text", "")
        score = float(ctx.get("score", 0.0))
        support = _context_support_score(text)
        priority = (support * 10.0) + score - (idx * 0.05)
        scored.append((priority, idx, ctx))

    scored.sort(key=lambda item: item[0], reverse=True)
    selected = [ctx for _, _, ctx in scored[:max_contexts]]
    if generator_type == "medgemma" and selected and not any(
        _context_has_confirmatory_signal(ctx.get("text", "")) for ctx in selected
    ):
        confirmatory_candidates = [
            ctx for ctx in contexts if _context_has_confirmatory_signal(ctx.get("text", ""))
        ]
        if confirmatory_candidates:
            best_confirmatory = max(
                confirmatory_candidates,
                key=lambda ctx: (_context_support_score(ctx.get("text", "")), float(ctx.get("score", 0.0))),
            )
            if best_confirmatory not in selected:
                selected = selected[: max_contexts - 1] + [best_confirmatory]

    selected_ids = {ctx.get("doc_id") for ctx in selected if ctx.get("doc_id")}
    ordered = [ctx for ctx in contexts if ctx.get("doc_id") in selected_ids]
    if generator_type == "medgemma" and ordered:
        return ordered[:max_contexts]
    return ordered


def _classify_retrieval_support(
    query: str,
    query_type: str,
    contexts: List[Dict],
    dataset_support: Dict[str, object],
) -> str:
    if not contexts:
        return "empty_contexts"

    top_score = float(contexts[0].get("score", 0.0)) if contexts else 0.0
    support_scores = [_context_support_score(ctx.get("text", "")) for ctx in contexts[:4]]
    strong_positive_count = sum(score >= 1.0 for score in support_scores)
    has_confirmatory_context = any(_context_has_confirmatory_signal(ctx.get("text", "")) for ctx in contexts[:4])
    leish_signal = _query_has_leish_signal(query)
    confirmatory_query = _query_has_confirmatory_signal(query)
    missing_nonleish_support = (
        (dataset_support.get("train_diagnosis_type_counts") or {}).get("Non-Leishmaniasis", 0) == 0
    )

    if confirmatory_query and not has_confirmatory_context:
        return "low_support_query_has_confirmatory_signal"
    if not leish_signal and missing_nonleish_support and (top_score < 0.03 or strong_positive_count == 0):
        return "unsupported_nonleish_corpus_gap"
    if not has_confirmatory_context and strong_positive_count == 0 and top_score < 0.02:
        return "low_support_no_confirmatory_context"
    if has_confirmatory_context and top_score >= 0.02:
        return "supported"
    if strong_positive_count >= 2 and top_score >= 0.03:
        return "supported"
    return "weak_support"


def generate_answers(
    run_dir: Path,
    retrieval_file: str = "retrieval.jsonl",
    generator_type: str = "gemini",
    model_variant: str = "12b",
    output_file: str = None,
    prompt_mode=None,
    force_rag: bool = False,
    skip_empty_contexts: bool = False,
    strip_query_images: bool = False,
    ablation_scope: str = "",
    context_k: Optional[int] = None,
    ordering_mode: Optional[str] = None,
    use_context_image_tensors: Optional[bool] = None,
    support_image_tensor_budget: Optional[int] = None,
    **generator_kwargs,
) -> Path:
    """Generate answers for all queries in a run."""
    run_dir = Path(run_dir)
    retrieval_path = run_dir / retrieval_file
    if not retrieval_path.exists():
        raise FileNotFoundError(f"No {retrieval_file} in {run_dir}")

    with open(retrieval_path) as f:
        samples = [json.loads(line) for line in f if line.strip()]

    train_cases: Dict[str, Dict] = {}
    with open(TRAIN_JSONL) as f:
        for line in f:
            case = json.loads(line)
            train_cases[case["case_id"]] = case

    dataset_support = get_dataset_support_snapshot()

    model_name = generator_kwargs.pop("model", None)
    model_path = generator_kwargs.pop("model_path", None)
    if model_path is None and model_name and generator_type in {"gemma3", "gemma4", "medgemma"}:
        model_path = model_name
    use_batch_api = bool(generator_kwargs.pop("use_batch_api", False))
    batch_poll_seconds = float(generator_kwargs.pop("batch_poll_seconds", 10.0))
    batch_timeout_seconds = int(generator_kwargs.pop("batch_timeout_seconds", 3600))
    random_seed = generator_kwargs.pop("random_seed", None)
    resolved_context_k = _resolve_generation_setting(run_dir, context_k, "context_k", None)
    resolved_ordering_mode = _resolve_generation_setting(run_dir, ordering_mode, "ordering_mode", "image_first")
    resolved_use_context_image_tensors = bool(
        _resolve_generation_setting(run_dir, use_context_image_tensors, "use_context_image_tensors", False)
    )
    resolved_support_image_tensor_budget = int(
        _resolve_generation_setting(run_dir, support_image_tensor_budget, "support_image_tensor_budget", 0) or 0
    )

    if random_seed is not None:
        import random

        random.seed(int(random_seed))
        try:
            import numpy as np

            np.random.seed(int(random_seed))
        except Exception:
            pass

    q3_like_count = 0
    image_payload_count = 0
    for sample in samples:
        query_type = str(sample.get("query_type", "") or "")
        if "q3" in query_type.lower() or "multimodal" in query_type.lower():
            q3_like_count += 1
        if _sample_has_query_images(sample):
            image_payload_count += 1

    has_image_queries = (q3_like_count > 0) or (image_payload_count > 0)
    if strip_query_images:
        has_image_queries = False
        if not ablation_scope:
            ablation_scope = "generator_only_image_strip"

    if generator_type == "qwen_vl":
        generator = QwenVLGenerator(variant=model_variant, **generator_kwargs)
    elif generator_type == "gemma3":
        resolved_model_path = model_path
        generator = Gemma3Generator(
            variant=model_variant,
            use_vision=has_image_queries,
            random_seed=random_seed,
            prompt_mode=prompt_mode,
            model_path=resolved_model_path,
            **generator_kwargs,
        )
    elif generator_type == "gemma4":
        resolved_model_path = model_path or "google/gemma-4-E4B-it"
        generator = Gemma4Generator(
            use_vision=has_image_queries,
            random_seed=random_seed,
            prompt_mode=prompt_mode,
            model_path=resolved_model_path,
            ordering_mode=resolved_ordering_mode,
            use_context_image_tensors=resolved_use_context_image_tensors,
            support_image_tensor_budget=resolved_support_image_tensor_budget,
            **generator_kwargs,
        )
    elif generator_type == "medgemma":
        generator = MedGemmaGenerator(
            model_path=model_path,
            use_vision=has_image_queries,
            prompt_mode=prompt_mode,
            **generator_kwargs,
        )
    else:
        generator = GeminiGenerator(
            model=model_name,
            prompt_mode=prompt_mode,
            **generator_kwargs,
        )

    if prompt_mode:
        print(f"Using prompt mode: {prompt_mode}")

    if strip_query_images:
        print(
            "Ablation mode enabled: generator-side query image stripping only; "
            f"ablation_scope={ablation_scope or 'generator_only_image_strip'}"
        )

    if generator_type in {"gemma3", "gemma4", "medgemma"}:
        print(
            "Vision detection: "
            f"q3_like={q3_like_count}/{len(samples)}, "
            f"image_payload={image_payload_count}/{len(samples)}, "
            f"use_vision={has_image_queries}"
        )

    print(f"Generating answers for {len(samples)} queries with {generator.model_name}...")

    prepared: List[Dict] = []
    skipped_samples = 0
    for i, sample in enumerate(samples):
        qid = sample["qid"]
        query = sample.get("query", "")
        raw_query_images = sample.get("query_images", [])[:5]
        query_images = [] if strip_query_images else raw_query_images
        query_images_stripped = bool(strip_query_images and raw_query_images)
        query_type = sample.get("query_type", "default")
        ground_truth = sample.get("ground_truth")
        ground_truth_pseudolabel = sample.get("ground_truth_pseudolabel")
        raw_contexts = sample.get("contexts", [])
        contexts, context_images = enrich_contexts_with_images(raw_contexts, train_cases)
        skip_reason = sample.get("skip_reason")

        if query_images:
            valid_count = sum(1 for p in query_images if Path(p).exists())
            print(f"  [{qid}] query_images: {valid_count}/{len(query_images)} valid")

        use_norag_prompt = False
        gating_info = ""
        threshold_used = 0.0
        top_score = float(contexts[0].get("score", 0.0)) if contexts else 0.0
        initial_retrieval_support_status = _classify_retrieval_support(query, query_type, contexts, dataset_support)
        retrieval_support_status = initial_retrieval_support_status

        if skip_reason:
            if query or query_images:
                contexts = []
                use_norag_prompt = True
                retrieval_support_status = f"skip_reason:{skip_reason}"
                gating_info = f"[RETRIEVAL {skip_reason}] NO-RAG fallback"
                print(f"  [{qid}] {gating_info}")
            else:
                skipped_samples += 1
                print(f"  [{qid}] skip generation: {skip_reason} with no usable query input")
                continue
        elif force_rag:
            gating_info = "[PURE-RAG] Forced RAG (ablation mode)"
            print(f"  [{qid}] {gating_info}")
        elif query_type == "Q3_image_diagnosis":
            if contexts:
                gating_info = f"[ROUTER] Q3 -> IMAGE-RAG ({len(contexts)} contexts)"
            elif query_images:
                use_norag_prompt = True
                gating_info = "[ROUTER] Q3 -> NO-RAG (no image-RAG contexts)"
            else:
                use_norag_prompt = True
                gating_info = "[ROUTER] Q3 -> NO-RAG (no usable query images or contexts)"
            print(f"  [{qid}] {gating_info}")
        elif ADAPTIVE_RAG.get("enabled", False) and contexts:
            threshold = ADAPTIVE_RAG.get("thresholds", {}).get(
                query_type, ADAPTIVE_RAG.get("thresholds", {}).get("default", 0.015)
            )
            threshold_used = threshold
            margin_threshold = ADAPTIVE_RAG.get("margin_threshold", 0.002)
            ctx_scores = [float(c.get("score", 0.0)) for c in contexts]
            ctx_scores_sorted = sorted(ctx_scores, reverse=True)
            top_score = ctx_scores_sorted[0] if ctx_scores_sorted else 0.0
            top3_score = ctx_scores_sorted[2] if len(ctx_scores_sorted) > 2 else ctx_scores_sorted[-1]
            margin = top_score - top3_score
            is_confident = top_score >= threshold and margin >= margin_threshold

            if initial_retrieval_support_status in {
                "unsupported_nonleish_corpus_gap",
                "low_support_query_has_confirmatory_signal",
                "low_support_no_confirmatory_context",
            }:
                contexts = []
                use_norag_prompt = True
                gating_info = f"[GATE OFF] {initial_retrieval_support_status}"
            elif is_confident:
                gating_info = f"[GATE ON] score={top_score:.4f}>={threshold:.4f}"
            elif ADAPTIVE_RAG.get("soft_gating", False):
                soft_k = ADAPTIVE_RAG.get("low_confidence_k", 1)
                contexts = contexts[:soft_k]
                gating_info = f"[SOFT GATE] score={top_score:.4f}, using top-{soft_k}"
            else:
                contexts = []
                use_norag_prompt = ADAPTIVE_RAG.get("use_norag_prompt_on_fallback", True)
                gating_info = f"[GATE OFF] score={top_score:.4f}<{threshold:.4f}"

            if gating_info:
                print(f"  [{qid}] {gating_info}")

        contexts = _prune_contexts_for_generation(contexts, generator_type, context_k=resolved_context_k)
        retrieval_support_status = _classify_retrieval_support(query, query_type, contexts, dataset_support)
        selected_context_images = [img for ctx in contexts for img in ctx.get("image_paths", [])][:5]
        generation_query = query
        if not contexts:
            use_norag_prompt = True
            selected_context_images = []
            if not gating_info:
                gating_info = "[NO-RAG] Empty or unsupported contexts"
            if query_type == "Q3_image_diagnosis" and query_images:
                generation_query = _build_q3_medical_no_context_prompt(query, query_images)

        generation_mode = "norag_prompt" if use_norag_prompt else "rag_prompt"
        prompt_contract = _resolve_prompt_contract_for_sample(
            requested_prompt_mode=prompt_mode,
            query_type=query_type,
            query_images=query_images,
            generation_mode=generation_mode,
            generation_query=generation_query,
            original_query=query,
        )

        if skip_empty_contexts and not contexts and not (query or query_images):
            skipped_samples += 1
            print(f"  [{qid}] skip generation: empty contexts and no usable query input")
            continue

        prepared.append(
            {
                "qid": qid,
                "query": query,
                "generation_query": generation_query,
                "contexts": contexts,
                "query_images": query_images,
                "context_images": selected_context_images,
                "ground_truth": ground_truth,
                "ground_truth_pseudolabel": ground_truth_pseudolabel,
                "gating_info": gating_info,
                "query_type": query_type,
                "top_score": top_score,
                "threshold_used": threshold_used,
                "generation_mode": generation_mode,
                "retrieval_support_status": retrieval_support_status,
                "ablation_scope": ablation_scope,
                "query_images_stripped": query_images_stripped,
                "context_k": resolved_context_k,
                "prompt_contract": prompt_contract,
            }
        )

        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

    if skipped_samples:
        print(f"  Skipped {skipped_samples} retrieval rows before generation")

    answers_by_qid: Dict[str, str] = {}
    generation_metadata_by_qid: Dict[str, Dict] = {}
    if generator_type == "gemini" and use_batch_api and hasattr(generator, "generate_batch"):
        print("Using Gemini Batch API for answer generation")
        batch_samples = [
            {
                "qid": p["qid"],
                "query": p["generation_query"],
                "contexts": p["contexts"],
                "query_images": p["query_images"],
                "context_images": p["context_images"],
                "image_paths": p["query_images"],
                "use_rag_prompt": p["generation_mode"] == "rag_prompt",
            }
            for p in prepared
        ]
        batch_results = generator.generate_batch(
            batch_samples,
            progress=True,
            use_batch_api=True,
            poll_seconds=batch_poll_seconds,
            timeout_seconds=batch_timeout_seconds,
        )
        answers_by_qid = {r["qid"]: r.get("answer", "") for r in batch_results}
        generation_metadata_by_qid = {r["qid"]: {} for r in batch_results}
    else:
        for p in prepared:
            answer = generator.generate(
                p["generation_query"],
                p["contexts"],
                image_paths=p["query_images"],
                query_images=p["query_images"],
                context_images=p["context_images"],
                use_rag_prompt=p["generation_mode"] == "rag_prompt",
            )
            answers_by_qid[p["qid"]] = answer
            generation_metadata_by_qid[p["qid"]] = dict(getattr(generator, "last_generation_metadata", {}) or {})

    records: List[Dict] = []
    for p in prepared:
        answer = answers_by_qid.get(p["qid"], "")
        citations = extract_citations(answer, p["contexts"])
        generation_metadata = generation_metadata_by_qid.get(p["qid"], {}) or {}
        format_analysis = analyze_answer_format(answer)
        prompt_contract = p.get("prompt_contract") or {}
        prompt_context_doc_ids = generation_metadata.get("prompt_context_doc_ids") or [
            ctx.get("doc_id") for ctx in p["contexts"] if ctx.get("doc_id")
        ]
        record = AnswerRecord(
            qid=p["qid"],
            query=p["query"],
            contexts=p["contexts"],
            answer=answer,
            model_name=generator.model_name,
            citations=citations,
            query_images=p["query_images"],
            context_images=p["context_images"],
            ground_truth=p["ground_truth"],
            ground_truth_pseudolabel=p.get("ground_truth_pseudolabel"),
            image_paths=p["query_images"],
            decoding_params=generator.decoding_params,
            gating_info=p["gating_info"],
            query_type=p["query_type"],
            top_score=p["top_score"],
            threshold_used=p["threshold_used"],
            generation_mode=p["generation_mode"],
            retrieval_support_status=p["retrieval_support_status"],
            prompt_context_doc_ids=prompt_context_doc_ids,
            prompt_context_count=int(generation_metadata.get("prompt_context_count", len(prompt_context_doc_ids))),
            format_retry_count=int(generation_metadata.get("format_retry_count", 0) or 0),
            answer_format_valid=generation_metadata.get("answer_format_valid", format_analysis.answer_format_valid),
            answer_format_error=generation_metadata.get("answer_format_error") or format_analysis.answer_format_error,
            ablation_scope=p.get("ablation_scope", ""),
            query_images_stripped=bool(p.get("query_images_stripped", False)),
            prompt_contract_version=str(prompt_contract.get("version") or RAW_QUERY_PROMPT_CONTRACT_VERSION),
            prompt_contract_notes=str(prompt_contract.get("notes") or RAW_QUERY_PROMPT_CONTRACT_NOTES),
            prompt_image_grounding_required=bool(prompt_contract.get("image_grounding_required", False)),
            context_k=p.get("context_k"),
            ordering_mode=str(generation_metadata.get("ordering_mode") or resolved_ordering_mode or ""),
            use_context_image_tensors=bool(
                generation_metadata.get("use_context_image_tensors", resolved_use_context_image_tensors)
            ),
            support_image_tensor_budget=int(
                generation_metadata.get(
                    "support_image_tensor_budget",
                    resolved_support_image_tensor_budget,
                )
                or 0
            ),
            query_image_tensor_attempt_count=int(generation_metadata.get("query_image_tensor_attempt_count", 0) or 0),
            support_image_tensor_attempt_count=int(
                generation_metadata.get("support_image_tensor_attempt_count", 0) or 0
            ),
            query_image_tensor_count=int(generation_metadata.get("query_image_tensor_count", 0) or 0),
            support_image_tensor_count=int(generation_metadata.get("support_image_tensor_count", 0) or 0),
            image_tensor_fallback_used=bool(generation_metadata.get("image_tensor_fallback_used", False)),
            image_tensor_fallback_reason=str(generation_metadata.get("image_tensor_fallback_reason") or ""),
        )
        records.append(asdict(record))

    output_name = output_file or "answers.jsonl"
    output_path = run_dir / output_name
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    generation_mode_counts = Counter(rec["generation_mode"] for rec in records)
    prompt_contract_summary = _build_prompt_contract_summary(records)
    prompt_context_counts = [int(rec.get("prompt_context_count") or 0) for rec in records]
    prompt_context_avg = (
        sum(prompt_context_counts) / len(prompt_context_counts)
        if prompt_context_counts
        else 0.0
    )
    prompt_image_grounding_required_count = sum(
        1 for rec in records if rec.get("prompt_image_grounding_required")
    )
    multimodal_usage_summary = _build_multimodal_usage_summary(records)
    prompt_mode_text = str(prompt_mode.value if isinstance(prompt_mode, PromptMode) else prompt_mode or "")
    inherited_retrieval_top_k = _resolve_inherited_retrieval_top_k(run_dir)
    generation_contract = {
        "is_rag": True,
        "generator_type": generator_type,
        "generator_model": generator.model_name,
        "prompt_mode": prompt_mode_text,
        "prompt_mode_requested": prompt_mode_text,
        "prompt_contract_version": prompt_contract_summary["prompt_contract_version"],
        "prompt_contract_notes": prompt_contract_summary["prompt_contract_notes"],
        "prompt_contract_summary": prompt_contract_summary,
        "prompt_image_grounding_required_any": prompt_image_grounding_required_count > 0,
        "prompt_image_grounding_required_count": prompt_image_grounding_required_count,
        "context_k": resolved_context_k,
        "ordering_mode": resolved_ordering_mode,
        "use_context_image_tensors": resolved_use_context_image_tensors,
        "support_image_tensor_budget": resolved_support_image_tensor_budget,
        "answer_file": output_name,
        "record_count": len(records),
        "generation_mode_counts": dict(generation_mode_counts),
        "prompt_context_count_avg": prompt_context_avg,
        "retrieval_top_k_inherited": inherited_retrieval_top_k,
        "multimodal_usage_summary": multimodal_usage_summary,
    }

    generation_contract_path = run_dir / "answer_generation_contract.json"
    generation_contract_path.write_text(
        json.dumps(generation_contract, indent=2),
        encoding="utf-8",
    )

    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        try:
            summary_payload = _load_json_dict(summary_path)
        except Exception:
            summary_payload = {}
        summary_payload["answer_generation_contract"] = generation_contract
        summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"✓ Generated {len(records)} answers to {output_path}")
    return output_path


if __name__ == "__main__":
    run_dir = RUNS_DIR / "phase3_hybrid"
    if run_dir.exists():
        generate_answers(run_dir, generator_type="gemini")
