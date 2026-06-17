"""
RAGAs Evaluator using Official RAGAS Library

Integrates RAGAS metrics directly from Leishmania_v3/ragas/src for Q1 journal quality.
Following claude45opus_guide.md for correct Collections API usage.

Metrics:
- Generation: MultiModalFaithfulness, MultiModalRelevance
- Retrieval: ContextRelevance

Uses Gemini 2.5 Pro as the vision-capable LLM judge.
"""
import os
import sys
import json
import asyncio
import time
import math
import inspect
import base64
import uuid
from collections import Counter
try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field

# Add RAGAS source to path
def _has_ragas_version_file(candidate: Path) -> bool:
    return (candidate / "ragas" / "_version.py").exists()


def _select_ragas_src(candidates: List[Path]) -> Path:
    with_version = [candidate for candidate in candidates if _has_ragas_version_file(candidate)]
    if with_version:
        return with_version[0]
    existing = [candidate for candidate in candidates if candidate.exists()]
    if existing:
        return existing[0]
    return candidates[0]


_legacy_root_override = os.getenv("LEGACY_ROOT", "").strip()
RAGAS_SRC_CANDIDATES = [Path(__file__).resolve().parents[2] / "ragas" / "src"]
if _legacy_root_override:
    RAGAS_SRC_CANDIDATES.append(Path(_legacy_root_override) / "ragas" / "src")
RAGAS_SRC_CANDIDATES.extend(
    [
        Path("/home/ngocnt/Leishmania_v3/ragas/src"),
        Path("/home/ngocnt/github/ragas/src"),
        Path("/data1t/lab/ngocnt/Leishmania_v3/ragas/src"),
    ]
)
RAGAS_SRC = _select_ragas_src(RAGAS_SRC_CANDIDATES)
if RAGAS_SRC.exists() and str(RAGAS_SRC) not in sys.path:
    sys.path.insert(0, str(RAGAS_SRC))

# Ensure HF cache is set before imports
os.environ.setdefault("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")

from .config import GOOGLE_API_KEY, JUDGE_MODEL, JUDGE_MODEL_FALLBACK, get_runtime_metadata


def _patch_instructor_genai_image_safety() -> None:
    """Force instructor's Gemini adapter to use text safety categories for multimodal prompts."""
    try:
        from google.genai.types import HarmBlockThreshold, HarmCategory
        from instructor.providers.gemini import utils as gemini_utils
    except Exception:
        return

    if getattr(gemini_utils, "_structured_cases_text_safety_patch", False):
        return

    original_update_genai_kwargs = gemini_utils.update_genai_kwargs
    text_categories = [
        HarmCategory.HARM_CATEGORY_HARASSMENT,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
    ]

    def patched_update_genai_kwargs(kwargs: Dict[str, Any], base_config: Dict[str, Any]) -> Dict[str, Any]:
        config = original_update_genai_kwargs(kwargs, base_config)
        safety_settings = config.get("safety_settings")
        if isinstance(safety_settings, list):
            has_image_category = False
            for setting in safety_settings:
                category = setting.get("category") if isinstance(setting, dict) else getattr(setting, "category", None)
                name = getattr(category, "name", str(category or ""))
                if "HARM_CATEGORY_IMAGE_" in name:
                    has_image_category = True
                    break
            if has_image_category:
                config["safety_settings"] = [
                    {"category": category, "threshold": HarmBlockThreshold.OFF}
                    for category in text_categories
                ]
        return config

    gemini_utils.update_genai_kwargs = patched_update_genai_kwargs
    gemini_utils._structured_cases_text_safety_patch = True


@dataclass
class RAGAsResult:
    """Result from RAGAs evaluation using official RAGAS library."""
    qid: str
    multimodal_faithfulness: Optional[float] = None
    multimodal_relevance: Optional[float] = None
    context_relevance: Optional[float] = None
    diagnosis_accuracy: Optional[float] = None
    diagnosis_type_accuracy: Optional[float] = None
    diagnosis_family_accuracy: Optional[float] = None
    diagnosis_family: Optional[str] = None
    diagnosis_reasoning: Optional[str] = None
    diagnosis_method: str = "llm_judge"
    gt_rank: Optional[int] = None
    top3_hit: Optional[float] = None
    l3_top1_correct: Optional[float] = None
    fallback_level: str = ""
    diagnosis_accuracy_pseudolabel: Optional[float] = None
    diagnosis_type_accuracy_pseudolabel: Optional[float] = None
    diagnosis_reasoning_pseudolabel: Optional[str] = None
    diagnosis_method_pseudolabel: str = ""
    gt_rank_pseudolabel: Optional[int] = None
    top3_hit_pseudolabel: Optional[float] = None
    l3_top1_correct_pseudolabel: Optional[float] = None
    fallback_level_pseudolabel: str = ""
    reasoning_recall: Optional[float] = None
    reasoning_recall_method: str = ""
    reasoning_recall_groundtruth_count: Optional[int] = None
    reasoning_recall_matched_count: Optional[int] = None
    reasoning_recall_explanation: Optional[str] = None
    reasoning_recall_matching_dict: Optional[Dict[str, Any]] = None
    reasoning_recall_source: str = ""
    reasoning_recall_source_id: str = ""
    reasoning_recall_source_path: str = ""
    reasoning_recall_judge_model: str = ""
    reasoning_trace_source: str = ""
    judge_model: str = ""
    generation_mode: str = ""
    retrieval_support_status: str = ""
    traces: Dict = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# DIAGNOSIS ACCURACY (per GPT 5.2: main metric for diagnostic RAG)
# =============================================================================

import re

from .diagnosis_output_parser import (
    DIAGNOSIS_FAMILY_MAP,
    SUPPORTED_DIAGNOSIS_TYPES,
    analyze_answer_format,
    canonicalize_diagnosis_type,
    compute_family_metric,
    compute_family_metric_details,
    diagnosis_family_from_type,
    evaluate_ranked_diagnosis_contract,
    extract_rank1_diagnosis_type,
    normalize_diagnosis,
)
from .reasoning_recall import (
    REASONING_RECALL_PROMPT,
    build_reasoning_recall_user_payload,
    load_reasoning_source_map,
    normalize_reasoning_recall_result,
    parse_predicted_reasoning_steps,
    resolve_groundtruth_payload,
)


def _coerce_optional_rank(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return None
        return int(value)
    try:
        text = str(value).strip().lower()
        if text in {"", "none", "null", "na", "n/a"}:
            return None
        return int(float(text))
    except Exception:
        return None


def _coerce_optional_binary_score(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        val = float(value)
        if math.isnan(val):
            return None
        if val <= 0.0:
            return 0.0
        return 1.0
    try:
        text = str(value).strip().lower()
        if text in {"", "none", "null", "na", "n/a"}:
            return None
        val = float(text)
        if math.isnan(val):
            return None
        return 0.0 if val <= 0.0 else 1.0
    except Exception:
        return None


def normalize_diagnosis_legacy(diagnosis: str) -> str:
    """
    Normalize diagnosis string for comparison.
    
    - Lowercase
    - Remove punctuation
    - Normalize common variants (leishmaniasis -> leishmania)
    """
    if not diagnosis:
        return ""
    
    normalized = diagnosis.lower().strip()
    # Remove punctuation except hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


# Legacy function - kept for backward compatibility but deprecated
def calculate_diagnosis_accuracy(
    predicted_answer: str,
    ground_truth: Optional[Dict]
) -> Dict[str, float]:
    """
    DEPRECATED: Use evaluate_diagnosis_equivalence_llm() for Q1-standard evaluation.
    
    This string-matching approach is NOT clinically acceptable per:
    - Dinc et al. (2025): LLM judges achieve Kappa=0.852 with specialists
    - ASTRID framework: requires semantic understanding
    
    Kept for backward compatibility only.
    """
    result = {
        "diagnosis_accuracy": 0.0,
        "diagnosis_type_accuracy": 0.0
    }
    
    if not ground_truth:
        return result
    
    gt_diagnosis = ground_truth.get("diagnosis", "")
    gt_type = ground_truth.get("diagnosis_type", "")
    
    if not gt_diagnosis and not gt_type:
        return result
    
    # Normalize predicted answer
    pred_normalized = normalize_diagnosis_legacy(predicted_answer)
    gt_diagnosis_normalized = normalize_diagnosis_legacy(gt_diagnosis)
    gt_type_normalized = normalize_diagnosis_legacy(gt_type)
    
    # Check if ground truth diagnosis appears in prediction
    if gt_diagnosis_normalized and gt_diagnosis_normalized in pred_normalized:
        result["diagnosis_accuracy"] = 1.0
    # Also check for leishmaniasis mentions
    elif "leishmania" in pred_normalized and "leishmania" in gt_diagnosis_normalized:
        result["diagnosis_accuracy"] = 0.5  # Partial match
    
    # Check diagnosis type using full dataset taxonomy
    type_keywords = {k.lower(): v for k, v in SUPPORTED_DIAGNOSIS_TYPES.items()}
    
    if gt_type_normalized:
        gt_type_key = gt_type_normalized.lower()
        if gt_type_key in type_keywords:
            for keyword in type_keywords[gt_type_key]:
                if keyword in pred_normalized:
                    result["diagnosis_type_accuracy"] = 1.0
                    break
        # Direct match
        elif gt_type_normalized in pred_normalized:
            result["diagnosis_type_accuracy"] = 1.0
    
    return result


# =============================================================================
# LLM-BASED DIAGNOSIS EQUIVALENCE (Q1 Standard - Dinc et al., 2025)
# =============================================================================

# Diagnosis equivalence prompt following Q1 paper methodology
# Updated per Claude 4.5 + Grok 4.1 review: stricter differential scoring, hedge penalty, species matching
DIAGNOSIS_EQUIVALENCE_PROMPT = """You are a specialist medical evaluator assessing diagnostic accuracy.

## Ground Truth Diagnosis
- **Primary Diagnosis**: {gt_diagnosis}
- **Diagnosis Type**: {gt_type}
- **Species** (if applicable): {gt_species}

## Predicted Answer
{prediction}

## Clinical Context Summary
{clinical_context}

## Query Image Inputs
- Number of query images: {query_image_count}
- Query image files: {query_image_list}

## Evaluation Criteria (per Dinc et al., 2025 + Grok 4.1 review)

Assess if the predicted answer contains the CORRECT diagnosis based on:

1. **EXACT MATCH (1.0)**: The prediction explicitly states the ground truth diagnosis as PRIMARY
   - Example: GT="Visceral Leishmaniasis", Pred="Primary Diagnosis: Visceral Leishmaniasis" → 1.0

2. **SYNONYM/CLINICALLY EQUIVALENT (1.0)**: Different terms for the same condition as PRIMARY
   - "Kala-azar" = "Visceral Leishmaniasis" → 1.0
   - "Cutaneous Leishmaniasis" = "Oriental Sore" = "Baghdad Boil" → 1.0
   - "L. donovani infection" = "Visceral Leishmaniasis" → 1.0

3. **CORRECT IN DIFFERENTIAL (ranked score)**: If prediction lists differentials:
   - GT is #1 in differential list (but not stated as primary) → 0.75
   - GT is #2 or #3 in differential list → 0.5
   - GT is #4 or lower in differential list → 0.25

4. **HEDGE WITH CORRECT MENTION (0.5)**: Model hedges but mentions correct answer
   - Prediction says "insufficient evidence" or "cannot determine" or "unclear"
   - BUT mentions the correct diagnosis somewhere in the answer
   - This indicates possible parametric knowledge usage

5. **PARTIAL CREDIT (0.5)**: Correct disease family but wrong subtype
   - GT="Cutaneous Leishmaniasis", Pred="Leishmaniasis (type unspecified)" → 0.5
   - GT="Visceral Leishmaniasis", Pred="Mucocutaneous Leishmaniasis" → 0.5

6. **INCORRECT (0.0)**: Wrong diagnosis or unrelated condition
   - GT="Leishmaniasis", Pred="Malaria" → 0.0
   - GT="Cutaneous Leishmaniasis", Pred="Psoriasis" → 0.0

## Special Rule for Non-Leishmaniasis Ground Truth
If Ground Truth Diagnosis is "Non-Leishmaniasis":
- diagnosis_score = 1.0 when prediction explicitly rules out leishmaniasis OR gives a clear non-leish primary diagnosis
- diagnosis_score = 0.5 when prediction is uncertain but includes non-leish differential and does not commit to leishmaniasis
- diagnosis_score = 0.0 when prediction states any leishmaniasis form as primary

For diagnosis_type_score in this case:
- 1.0 if type is NON_LEISH/Other and consistent with non-leish primary diagnosis
- 0.0 if type implies any leishmaniasis form

## Type-Specific Matching
- CL = Cutaneous Leishmaniasis
- VL = Visceral Leishmaniasis = Kala-azar
- MCL = Mucocutaneous Leishmaniasis = Espundia
- PKDL = Post-Kala-azar Dermal Leishmaniasis
- DCL = Diffuse Cutaneous Leishmaniasis
- DsCL = Disseminated Cutaneous Leishmaniasis
- LCL = Localized Cutaneous Leishmaniasis
- LR = Leishmaniasis Recidivans
- Ocular = Ocular Leishmaniasis
- Veterinary = Veterinary Leishmaniasis
- Non-Leishmaniasis = clearly non-leish diagnosis

## Image-aware evaluation rule
- If query images are provided, use image evidence as an additional consistency signal.
- If images are non-diagnostic, do not over-penalize based on absent visual findings.
- Penalize only when prediction strongly contradicts obvious image morphology.

## Species Matching (for diagnosis_type_score adjustment)
- Exact species match: no penalty
- Same accepted species-complex/group but different species string: small penalty (e.g., multiply by 0.9)
- Species not mentioned but type correct: multiply type_score by 0.9
- Wrong species mentioned: multiply type_score by 0.5

## Output (JSON only)
Respond with ONLY valid JSON:
{{
    "diagnosis_score": <0.0 | 0.25 | 0.5 | 0.75 | 1.0>,
    "diagnosis_type_score": <0.0 | 0.25 | 0.5 | 0.75 | 1.0>,
    "gt_rank": <1 | 2 | 3 | 0 | null>,
    "top3_hit": <0.0 | 1.0 | null>,
    "l3_top1_correct": <0.0 | 1.0 | null>,
    "fallback_level": "<type_exact | family_fallback | diagnosis_text_fallback | unscorable>",
    "reasoning": "<brief clinical explanation including species assessment and image consistency>"
}}"""


@dataclass
class DiagnosisEquivalenceResult:
    """Result from LLM-based diagnosis equivalence evaluation."""
    diagnosis_score: float
    diagnosis_type_score: float
    reasoning: str
    method: str = "llm_judge"  # "llm_judge" or "string_match" (fallback)
    gt_rank: Optional[int] = None
    top3_hit: Optional[float] = None
    l3_top1_correct: Optional[float] = None
    fallback_level: str = ""


class RAGAsLibraryEvaluator:
    """
    RAGAs evaluator using official RAGAS library from Leishmania_v3/ragas/src.
    Following claude45opus_guide.md for Collections API usage.
    
    Generation Metrics:
    - MultiModalFaithfulness: Binary (0/1) - Is response grounded in multimodal context?
    - MultiModalRelevance: Binary (0/1) - Is response relevant to query and context?
    
    Retrieval Metrics:
    - ContextRelevance: Continuous (0-1) - Are contexts relevant to query?
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize RAGAS evaluator with Gemini LLM.
        
        Args:
            model: Model name (default: from config)
            api_key: Google API key (default: from config)
        """
        self.model_name = model or JUDGE_MODEL
        self.api_key = api_key or GOOGLE_API_KEY
        
        # Lazy initialization of RAGAS components
        self._llm = None
        self._metrics_initialized = False
        
        # Metric instances (created on first use)
        self._multimodal_faithfulness = None
        self._multimodal_relevance = None
        self._context_relevance = None

        # Last metric execution status for per-sample diagnostics
        self._last_metric_status: Dict[str, Dict[str, Any]] = {}

        # Lightweight input guards to avoid oversized multimodal metric prompts
        self._metric_max_query_chars = 5000
        self._metric_max_answer_chars = 6000
        self._metric_max_context_chars_each = 3000
        self._metric_max_context_chars_total = 18000
        self._metric_max_text_contexts = 8
        self._metric_max_query_images = 3

        # Reasoning-recall sources are loaded lazily on first use.
        self._reasoning_source_map: Optional[Dict[str, Dict[str, Any]]] = None
        self._reasoning_source_paths = self._build_reasoning_source_paths()

    def _build_reasoning_source_paths(self) -> List[Path]:
        env_value = os.getenv("REASONING_RECALL_SOURCE_PATHS", "").strip()
        if env_value:
            return [Path(item.strip()) for item in env_value.split(":") if item.strip()]

        return [
            Path(
                "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/"
                "p14_test_held_out_structure_only_v7_local_reaudit/results.jsonl"
            ),
            Path(
                "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/outputs/"
                "test_pseudolabel_v2_strict/results.jsonl"
            ),
        ]

    def _ensure_reasoning_source_map(self) -> Dict[str, Dict[str, Any]]:
        if self._reasoning_source_map is None:
            self._reasoning_source_map = load_reasoning_source_map(self._reasoning_source_paths)
        return self._reasoning_source_map

    def _truncate_text(self, text: str, max_chars: int) -> str:
        """Truncate long text deterministically to keep metric prompts bounded."""
        if text is None:
            return ""
        s = str(text)
        if max_chars <= 0 or len(s) <= max_chars:
            return s
        return s[:max_chars] + "\n...[truncated]"

    def _prepare_metric_inputs(
        self,
        query: str,
        answer: str,
        contexts: List[str],
        query_images: Optional[List[str]] = None,
    ) -> Tuple[str, str, List[str], List[str], Dict[str, Any]]:
        """Prepare bounded metric inputs and return preprocessing stats for traces."""
        safe_query = self._truncate_text(query, self._metric_max_query_chars)
        safe_answer = self._truncate_text(answer, self._metric_max_answer_chars)

        raw_contexts = [str(c).strip() for c in (contexts or []) if str(c).strip()]
        clipped_contexts: List[str] = []
        total_chars = 0
        dropped_by_total = 0

        for raw_ctx in raw_contexts[: self._metric_max_text_contexts]:
            clipped_ctx = self._truncate_text(raw_ctx, self._metric_max_context_chars_each)
            if not clipped_ctx.strip():
                continue
            if total_chars + len(clipped_ctx) > self._metric_max_context_chars_total:
                dropped_by_total += 1
                continue
            clipped_contexts.append(clipped_ctx)
            total_chars += len(clipped_ctx)

        dropped_by_count = max(0, len(raw_contexts) - self._metric_max_text_contexts)

        safe_query_images: List[str] = []
        skipped_missing_images = 0
        for image_path in (query_images or [])[: self._metric_max_query_images]:
            if Path(image_path).exists():
                safe_query_images.append(image_path)
            else:
                skipped_missing_images += 1

        retrieved_contexts = clipped_contexts.copy()
        if safe_query_images:
            retrieved_contexts.extend(safe_query_images)

        stats: Dict[str, Any] = {
            "query_chars_original": len(str(query or "")),
            "query_chars_used": len(safe_query),
            "query_truncated": len(safe_query) < len(str(query or "")),
            "answer_chars_original": len(str(answer or "")),
            "answer_chars_used": len(safe_answer),
            "answer_truncated": len(safe_answer) < len(str(answer or "")),
            "contexts_original_nonempty": len(raw_contexts),
            "contexts_used": len(clipped_contexts),
            "contexts_truncated_by_count": dropped_by_count,
            "contexts_truncated_by_total_chars": dropped_by_total,
            "context_chars_used": total_chars,
            "query_images_original": len(query_images or []),
            "query_images_used": len(safe_query_images),
            "query_images_missing": skipped_missing_images,
            "retrieved_contexts_total": len(retrieved_contexts),
        }

        return safe_query, safe_answer, clipped_contexts, retrieved_contexts, stats

    def _classify_metric_error(self, error_text: str) -> str:
        """Classify metric failure causes for downstream debugging and reporting."""
        msg = (error_text or "").lower()
        if any(m in msg for m in ["429", "rate limit", "quota", "resource exhausted"]):
            return "rate_limit_or_quota"
        if any(m in msg for m in ["503", "temporar", "timeout", "deadline", "unavailable"]):
            return "transient_backend"
        if any(m in msg for m in ["token", "context length", "too long", "max output", "prompt"]):
            return "input_too_large"
        if any(m in msg for m in ["json", "parse", "schema", "validation", "response_model"]):
            return "response_parse_or_schema"
        if any(m in msg for m in ["missing", "required", "invalid", "valueerror"]):
            return "invalid_input"
        return "other_error"

    def _build_diagnosis_prompt(
        self,
        prediction: str,
        ground_truth: Dict,
        clinical_context: str = "",
        query_images: Optional[List[str]] = None,
    ) -> str:
        return DIAGNOSIS_EQUIVALENCE_PROMPT.format(
            gt_diagnosis=ground_truth.get("diagnosis", "Unknown"),
            gt_type=ground_truth.get("diagnosis_type", "Unknown"),
            gt_species=ground_truth.get("species", "Not specified"),
            prediction=prediction[:2000],
            clinical_context=(clinical_context or "")[:1500],
            query_image_count=len(query_images or []),
            query_image_list=", ".join(Path(p).name for p in (query_images or [])[:5]) or "None"
        )

    def _build_multimodal_contents(
        self,
        prompt: str,
        query_images: Optional[List[str]] = None,
    ) -> List[Any]:
        contents: List[Any] = [prompt]
        if not query_images:
            return contents

        for image_path in query_images[:3]:
            image_file = Path(image_path)
            if not image_file.exists():
                continue
            try:
                suffix = image_file.suffix.lower()
                mime_map = {
                    ".jpg": "image/jpeg",
                    ".jpeg": "image/jpeg",
                    ".png": "image/png",
                    ".gif": "image/gif",
                    ".webp": "image/webp",
                }
                mime_type = mime_map.get(suffix, "image/jpeg")
                with open(image_file, "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                contents.append({
                    "inline_data": {
                        "mime_type": mime_type,
                        "data": data,
                    }
                })
            except Exception:
                continue
        return contents

    def _parse_diagnosis_json(self, result_text: str) -> Dict[str, Any]:
        try:
            return json.loads(result_text)
        except json.JSONDecodeError:
            import re
            json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            raise ValueError(f"Could not parse JSON: {result_text[:200]}")

    async def evaluate_reasoning_recall(
        self,
        groundtruth_points: List[str],
        predicted_steps: List[str],
    ) -> Dict[str, Any]:
        """Evaluate reasoning recall using an LLM-as-judge contract."""
        judge_models: List[str] = []
        for candidate in [self.model_name, JUDGE_MODEL, JUDGE_MODEL_FALLBACK]:
            if candidate and candidate not in judge_models:
                judge_models.append(candidate)

        if not groundtruth_points:
            return {
                "matched_groundtruth_indices": [],
                "matched_groundtruth_points": [],
                "unmatched_groundtruth_points": [],
                "matched_groundtruth_count": 0,
                "groundtruth_count": 0,
                "recall": 0.0,
                "explanation": "No groundtruth reasoning points",
                "method": "skipped_missing_groundtruth_reasoning",
                "judge_model": "",
                "requested_judge_model": judge_models[0] if judge_models else "",
            }

        if not predicted_steps:
            return {
                "matched_groundtruth_indices": [],
                "matched_groundtruth_points": [],
                "unmatched_groundtruth_points": list(groundtruth_points),
                "matched_groundtruth_count": 0,
                "groundtruth_count": len(groundtruth_points),
                "recall": 0.0,
                "explanation": "Predicted reasoning trace not parseable",
                "method": "skipped_unparseable_predicted_reasoning",
                "judge_model": "",
                "requested_judge_model": judge_models[0] if judge_models else "",
            }

        prompt = (
            REASONING_RECALL_PROMPT
            + "\n\nInput JSON:\n"
            + build_reasoning_recall_user_payload(
                groundtruth_points=groundtruth_points,
                predicted_steps=predicted_steps,
            )
        )

        try:
            from google import genai

            client = genai.Client(api_key=self.api_key)
            response = None
            last_error = None
            used_judge_model = ""
            for judge_model in judge_models:
                try:
                    response = client.models.generate_content(
                        model=judge_model,
                        contents=[prompt],
                        config={
                            "temperature": 0.0,
                            "response_mime_type": "application/json",
                        },
                    )
                    used_judge_model = judge_model
                    break
                except Exception as model_error:
                    last_error = model_error
                    continue

            if response is None and last_error is not None:
                raise last_error

            result_text = (response.text or "").strip()
            parsed_payload = self._parse_diagnosis_json(result_text)
            normalized = normalize_reasoning_recall_result(parsed_payload, groundtruth_points)
            normalized["method"] = (
                "llm_judge"
                if not judge_models or used_judge_model == judge_models[0]
                else "llm_judge_fallback_model"
            )
            normalized["judge_model"] = used_judge_model
            normalized["requested_judge_model"] = judge_models[0] if judge_models else ""
            return normalized
        except Exception as exc:
            return {
                "matched_groundtruth_indices": [],
                "matched_groundtruth_points": [],
                "unmatched_groundtruth_points": list(groundtruth_points),
                "matched_groundtruth_count": 0,
                "groundtruth_count": len(groundtruth_points),
                "recall": None,
                "explanation": f"[Reasoning recall judge error: {str(exc)[:200]}]",
                "method": "judge_error",
                "judge_model": "",
                "requested_judge_model": judge_models[0] if judge_models else "",
            }

    async def evaluate_reasoning_recall_for_sample(self, qid: str, answer: str) -> Dict[str, Any]:
        """Resolve source reasoning + parse predicted reasoning and score recall."""
        source_map = self._ensure_reasoning_source_map()
        source_payload = resolve_groundtruth_payload(qid, source_map)
        groundtruth_points = source_payload["groundtruth_points"]
        source_path = source_payload["source_path"]
        source_id = source_payload["source_id"]

        if not groundtruth_points:
            return {
                "recall": None,
                "method": "skipped_missing_groundtruth_reasoning",
                "groundtruth_count": 0,
                "matched_groundtruth_count": 0,
                "matched_groundtruth_indices": [],
                "matched_groundtruth_points": [],
                "unmatched_groundtruth_points": [],
                "explanation": "No groundtruth reasoning points found for case_id",
                "source_path": source_path,
                "source_id": source_id,
                "predicted_reasoning_steps": [],
            }

        predicted_steps = parse_predicted_reasoning_steps(answer)
        if not predicted_steps:
            return {
                "recall": 0.0,
                "method": "skipped_unparseable_predicted_reasoning",
                "groundtruth_count": len(groundtruth_points),
                "matched_groundtruth_count": 0,
                "matched_groundtruth_indices": [],
                "matched_groundtruth_points": [],
                "unmatched_groundtruth_points": list(groundtruth_points),
                "explanation": "Could not parse reasoning steps from answer",
                "source_path": source_path,
                "source_id": source_id,
                "predicted_reasoning_steps": [],
            }

        judged = await self.evaluate_reasoning_recall(groundtruth_points, predicted_steps)
        judged["source_path"] = source_path
        judged["source_id"] = source_id
        judged["predicted_reasoning_steps"] = predicted_steps
        return judged

    def evaluate_diagnosis_equivalence_batch(
        self,
        requests: List[Dict[str, Any]],
        poll_seconds: float = 10.0,
        timeout_seconds: int = 7200,
    ) -> Dict[str, DiagnosisEquivalenceResult]:
        """Batch-evaluate diagnosis equivalence using Gemini Batch API."""
        from google import genai
        from google.genai import types

        if not requests:
            return {}

        client = genai.Client(api_key=self.api_key)
        judge_models: List[str] = []
        for candidate in [self.model_name, JUDGE_MODEL, JUDGE_MODEL_FALLBACK]:
            if candidate and candidate not in judge_models:
                judge_models.append(candidate)

        last_error: Optional[Exception] = None
        for judge_model in judge_models:
            try:
                inlined_requests = []
                for req in requests:
                    qid = req["qid"]
                    prompt = self._build_diagnosis_prompt(
                        prediction=req["prediction"],
                        ground_truth=req["ground_truth"],
                        clinical_context=req.get("clinical_context", ""),
                        query_images=req.get("query_images", []),
                    )
                    contents = self._build_multimodal_contents(prompt, req.get("query_images", []))
                    inlined_requests.append(
                        types.InlinedRequest(
                            model=judge_model,
                            contents=contents,
                            metadata={"qid": qid},
                            config=types.GenerateContentConfig(
                                temperature=0.0,
                                response_mime_type="application/json",
                            ),
                        )
                    )

                display_name = f"diag-judge-batch-{uuid.uuid4().hex[:8]}"
                job = client.batches.create(
                    model=judge_model,
                    src=types.BatchJobSource(inlined_requests=inlined_requests),
                    config=types.CreateBatchJobConfig(display_name=display_name),
                )
                print(f"Diagnosis judge batch started: {job.name} ({judge_model})")

                deadline = time.time() + timeout_seconds
                while True:
                    job = client.batches.get(name=job.name)
                    state = str(getattr(job, "state", "UNKNOWN"))
                    if "SUCCEEDED" in state:
                        break
                    if any(x in state for x in ["FAILED", "CANCELLED", "EXPIRED"]):
                        raise RuntimeError(f"Diagnosis batch ended in error state: {state}")
                    if time.time() > deadline:
                        raise TimeoutError(f"Diagnosis batch timeout after {timeout_seconds}s: {state}")
                    time.sleep(poll_seconds)

                dest = getattr(job, "dest", None)
                inlined_responses = getattr(dest, "inlined_responses", None) if dest else None
                if not inlined_responses:
                    raise RuntimeError("Diagnosis batch succeeded but returned no inlined responses")

                by_qid: Dict[str, DiagnosisEquivalenceResult] = {}
                for resp in inlined_responses:
                    md = getattr(resp, "metadata", None) or {}
                    qid = str(md.get("qid", ""))
                    err = getattr(resp, "error", None)
                    if err is not None:
                        by_qid[qid] = DiagnosisEquivalenceResult(
                            diagnosis_score=0.0,
                            diagnosis_type_score=0.0,
                            reasoning=f"[Batch judge error: {err}]",
                            method="llm_judge_batch_error",
                            gt_rank=None,
                            top3_hit=None,
                            l3_top1_correct=None,
                            fallback_level="judge_error",
                        )
                        continue

                    response_obj = getattr(resp, "response", None)
                    result_text = ""
                    try:
                        result_text = (response_obj.text or "").strip()
                    except Exception:
                        pass

                    parsed = self._parse_diagnosis_json(result_text)
                    gt_rank = _coerce_optional_rank(parsed.get("gt_rank"))
                    top3_hit = _coerce_optional_binary_score(parsed.get("top3_hit"))
                    l3_top1 = _coerce_optional_binary_score(parsed.get("l3_top1_correct"))
                    if top3_hit is None and gt_rank is not None:
                        top3_hit = 1.0 if gt_rank in {1, 2, 3} else 0.0
                    if l3_top1 is None and gt_rank is not None:
                        l3_top1 = 1.0 if gt_rank == 1 else 0.0
                    by_qid[qid] = DiagnosisEquivalenceResult(
                        diagnosis_score=float(parsed.get("diagnosis_score", 0.0)),
                        diagnosis_type_score=float(parsed.get("diagnosis_type_score", 0.0)),
                        reasoning=parsed.get("reasoning", "No reasoning provided"),
                        method="llm_judge_batch",
                        gt_rank=gt_rank,
                        top3_hit=top3_hit,
                        l3_top1_correct=l3_top1,
                        fallback_level=str(parsed.get("fallback_level", ""))[:80],
                    )

                # Fill any missing qids defensively
                for req in requests:
                    qid = req["qid"]
                    if qid not in by_qid:
                        by_qid[qid] = DiagnosisEquivalenceResult(
                            diagnosis_score=0.0,
                            diagnosis_type_score=0.0,
                            reasoning="[Batch judge missing response]",
                            method="llm_judge_batch_missing",
                            gt_rank=None,
                            top3_hit=None,
                            l3_top1_correct=None,
                            fallback_level="judge_missing",
                        )
                return by_qid
            except Exception as e:
                last_error = e
                continue

        raise RuntimeError(f"All diagnosis judge batch model attempts failed: {last_error}")
    
    def _init_llm(self):
        """Initialize RAGAS LLM wrapper for Gemini."""
        if self._llm is not None:
            return

        _patch_instructor_genai_image_safety()
        
        try:
            from google import genai
            from ragas.llms import llm_factory
            
            # Create Gemini client using new google-genai SDK
            client = genai.Client(api_key=self.api_key)
            
            # Create RAGAS LLM wrapper using llm_factory as per guide
            self._llm = llm_factory(
                model=self.model_name,
                provider="google",
                client=client,
                temperature=0.0,  # Deterministic for evaluation
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import RAGAS or google-genai. "
                f"Ensure RAGAS is available at {RAGAS_SRC}.\n"
                f"Error: {e}"
            )
    
    def _init_metrics(self):
        """Initialize RAGAS metric instances using Collections API."""
        if self._metrics_initialized:
            return
        
        self._init_llm()
        
        try:
            import ragas as ragas_pkg
            print(f"RAGAS source in use: {getattr(ragas_pkg, '__file__', 'unknown')} (candidate={RAGAS_SRC})")

            # Import from Collections API as per claude45opus_guide.md
            from ragas.metrics.collections import (
                MultiModalFaithfulness,
                MultiModalRelevance,
                ContextRelevance,
            )
            
            # Initialize metrics with the LLM
            self._multimodal_faithfulness = MultiModalFaithfulness(llm=self._llm)
            self._multimodal_relevance = MultiModalRelevance(llm=self._llm)
            self._context_relevance = ContextRelevance(llm=self._llm)
            
            self._metrics_initialized = True
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import RAGAS metrics. Check RAGAS installation.\n"
                f"Error: {e}"
            )
    
    async def evaluate_multimodal_faithfulness(
        self,
        response: str,
        retrieved_contexts: List[str]
    ) -> Optional[float]:
        """
        Evaluate multimodal faithfulness using RAGAS Collections API.
        
        Args:
            response: The generated response to evaluate
            retrieved_contexts: List of text contexts or image paths
        
        Returns: 1.0 if faithful, 0.0 if not (Binary)
        """
        self._init_metrics()

        return await self._score_metric_with_retries(
            metric_name="MultiModalFaithfulness",
            metric=self._multimodal_faithfulness,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
    
    async def evaluate_multimodal_relevance(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str]
    ) -> Optional[float]:
        """
        Evaluate multimodal relevance using RAGAS Collections API.
        
        Args:
            user_input: The user's question/query
            response: The generated response to evaluate
            retrieved_contexts: List of text contexts or image paths
        
        Returns: 1.0 if relevant, 0.0 if not (Binary)
        """
        self._init_metrics()

        return await self._score_metric_with_retries(
            metric_name="MultiModalRelevance",
            metric=self._multimodal_relevance,
            user_input=user_input,
            response=response,
            retrieved_contexts=retrieved_contexts,
        )
    
    async def evaluate_context_relevance(
        self,
        user_input: str,
        retrieved_contexts: List[str]
    ) -> Optional[float]:
        """
        Evaluate context relevance using RAGAS Collections API.
        
        Args:
            user_input: The user's question/query
            retrieved_contexts: List of retrieved text contexts
        
        Returns: Continuous score 0.0-1.0
        """
        self._init_metrics()

        return await self._score_metric_with_retries(
            metric_name="ContextRelevance",
            metric=self._context_relevance,
            user_input=user_input,
            retrieved_contexts=retrieved_contexts,
        )

    async def _score_metric_with_retries(
        self,
        metric_name: str,
        metric: Any,
        **kwargs,
    ) -> Optional[float]:
        """Score one metric with transient-error retries to reduce null outputs."""
        max_attempts = 3
        transient_markers = [
            "429",
            "503",
            "rate limit",
            "quota",
            "temporar",
            "timeout",
            "deadline",
            "unavailable",
            "resource exhausted",
        ]

        last_error = ""
        last_error_type = ""

        for attempt in range(1, max_attempts + 1):
            try:
                score_call = metric.ascore(**kwargs)
                result = await score_call if inspect.isawaitable(score_call) else score_call
                value_raw = getattr(result, "value", result)
                value = float(value_raw)
                if math.isnan(value) or math.isinf(value):
                    self._last_metric_status[metric_name] = {
                        "status": "error",
                        "attempts": attempt,
                        "error_type": "nan_or_inf_value",
                        "error_message": "Metric returned NaN/Inf",
                    }
                    return None

                self._last_metric_status[metric_name] = {
                    "status": "ok",
                    "attempts": attempt,
                    "value": value,
                }
                return value
            except Exception as e:
                err_text = str(e)
                msg = err_text.lower()
                is_transient = any(marker in msg for marker in transient_markers)
                last_error = err_text
                last_error_type = self._classify_metric_error(err_text)
                if is_transient and attempt < max_attempts:
                    wait_s = float(min(2 ** (attempt - 1), 8))
                    print(
                        f"{metric_name} transient error (attempt {attempt}/{max_attempts}): {e}; "
                        f"retrying in {wait_s:.1f}s"
                    )
                    await asyncio.sleep(wait_s)
                    continue

                print(f"{metric_name} error: {e}")
                self._last_metric_status[metric_name] = {
                    "status": "error",
                    "attempts": attempt,
                    "error_type": last_error_type or "other_error",
                    "error_message": last_error[:400],
                }
                return None

        self._last_metric_status[metric_name] = {
            "status": "error",
            "attempts": max_attempts,
            "error_type": last_error_type or "unknown",
            "error_message": (last_error or "unknown metric failure")[:400],
        }
        return None
    
    async def evaluate_diagnosis_equivalence(
        self,
        prediction: str,
        ground_truth: Dict,
        clinical_context: str = "",
        query_images: Optional[List[str]] = None,
        use_llm_judge: bool = True
    ) -> DiagnosisEquivalenceResult:
        """
        Evaluate diagnosis accuracy using LLM-as-Judge (Q1 Standard).
        
        This method follows the methodology from:
        - Dinc et al. (2025) "Comparative Analysis of LLMs" - Kappa=0.852 with specialists
        - ASTRID framework - semantic understanding for clinical terms
        
        The LLM judge can recognize:
        - Synonyms (Kala-azar = Visceral Leishmaniasis)
        - Clinical equivalence (L. donovani infection = VL)
        - Differential diagnosis (GT in top-3 list)
        - Partial credit (correct family, wrong subtype)
        
        Args:
            prediction: The model's generated answer
            ground_truth: Dict with 'diagnosis', 'diagnosis_type', 'species'
            use_llm_judge: If True, use LLM; if False, fall back to string matching
        
        Returns:
            DiagnosisEquivalenceResult with scores and reasoning
        """
        # Fallback to legacy string matching if requested
        if not use_llm_judge:
            legacy_result = calculate_diagnosis_accuracy(prediction, ground_truth)
            parser_contract = evaluate_ranked_diagnosis_contract(prediction, ground_truth)
            return DiagnosisEquivalenceResult(
                diagnosis_score=legacy_result["diagnosis_accuracy"],
                diagnosis_type_score=legacy_result["diagnosis_type_accuracy"],
                reasoning="[Legacy string matching]",
                method="string_match",
                gt_rank=_coerce_optional_rank(parser_contract.get("gt_rank")),
                top3_hit=_coerce_optional_binary_score(parser_contract.get("top3_hit")),
                l3_top1_correct=_coerce_optional_binary_score(parser_contract.get("l3_top1_correct")),
                fallback_level=str(parser_contract.get("fallback_level", ""))[:80],
            )
        
        self._init_llm()
        
        # Format the prompt
        prompt = self._build_diagnosis_prompt(
            prediction=prediction,
            ground_truth=ground_truth,
            clinical_context=clinical_context,
            query_images=query_images,
        )
        
        try:
            # Use the google genai client directly for this call
            from google import genai
            
            client = genai.Client(api_key=self.api_key)
            
            multimodal_contents = self._build_multimodal_contents(prompt, query_images)

            judge_models: List[str] = []
            for candidate in [self.model_name, JUDGE_MODEL, JUDGE_MODEL_FALLBACK]:
                if candidate and candidate not in judge_models:
                    judge_models.append(candidate)

            response = None
            last_error = None
            for judge_model in judge_models:
                try:
                    response = client.models.generate_content(
                        model=judge_model,
                        contents=multimodal_contents,
                        config={
                            "temperature": 0.0,  # Deterministic
                            "response_mime_type": "application/json",
                        }
                    )
                    if judge_model != self.model_name:
                        print(f"Diagnosis judge fallback model used: {judge_model}")
                    break
                except Exception as model_error:
                    last_error = model_error
                    continue

            if response is None and last_error is not None:
                raise last_error
            
            result_text = response.text.strip()
            result = self._parse_diagnosis_json(result_text)
            gt_rank = _coerce_optional_rank(result.get("gt_rank"))
            top3_hit = _coerce_optional_binary_score(result.get("top3_hit"))
            l3_top1 = _coerce_optional_binary_score(result.get("l3_top1_correct"))
            if top3_hit is None and gt_rank is not None:
                top3_hit = 1.0 if gt_rank in {1, 2, 3} else 0.0
            if l3_top1 is None and gt_rank is not None:
                l3_top1 = 1.0 if gt_rank == 1 else 0.0
            
            return DiagnosisEquivalenceResult(
                diagnosis_score=float(result.get("diagnosis_score", 0.0)),
                diagnosis_type_score=float(result.get("diagnosis_type_score", 0.0)),
                reasoning=result.get("reasoning", "No reasoning provided"),
                method="llm_judge",
                gt_rank=gt_rank,
                top3_hit=top3_hit,
                l3_top1_correct=l3_top1,
                fallback_level=str(result.get("fallback_level", ""))[:80],
            )
            
        except Exception as e:
            print(f"LLM diagnosis evaluation error: {e}")
            # Fallback to string matching
            legacy_result = calculate_diagnosis_accuracy(prediction, ground_truth)
            parser_contract = evaluate_ranked_diagnosis_contract(prediction, ground_truth)
            return DiagnosisEquivalenceResult(
                diagnosis_score=legacy_result["diagnosis_accuracy"],
                diagnosis_type_score=legacy_result["diagnosis_type_accuracy"],
                reasoning=f"[Fallback to string matching due to: {str(e)[:100]}]",
                method="string_match_fallback",
                gt_rank=_coerce_optional_rank(parser_contract.get("gt_rank")),
                top3_hit=_coerce_optional_binary_score(parser_contract.get("top3_hit")),
                l3_top1_correct=_coerce_optional_binary_score(parser_contract.get("l3_top1_correct")),
                fallback_level=str(parser_contract.get("fallback_level", ""))[:80],
            )
    
    async def evaluate_sample(
        self,
        qid: str,
        query: str,
        answer: str,
        contexts: List[str],
        query_images: Optional[List[str]] = None,
        context_images: Optional[List[str]] = None,
        ground_truth: Optional[Dict] = None,
        ground_truth_pseudolabel: Optional[Dict] = None,
        image_paths: Optional[List[str]] = None,
        evaluate_retrieval_metrics: bool = True,
        generation_mode: str = "",
        retrieval_support_status: str = "",
    ) -> RAGAsResult:
        """
        Evaluate a single sample with all RAGAS metrics.
        
        CRITICAL FIX (per GPT 5.2):
        - query_images: Images from TEST case (for visual query understanding)
        - context_images: Images from TRAIN cases (retrieved contexts)
        - For multimodal metrics, use query_images (the actual test case images)
        
        NO-RAG BASELINE HANDLING:
        - If contexts is empty, skip retrieval-grounded metrics (faithfulness, relevance, context_relevance)
        - These metrics are conceptually undefined without retrieved contexts
        - Only diagnosis accuracy via LLM judge will be evaluated
        
        Args:
            qid: Query ID
            query: User query (user_input)
            answer: Generated answer (response)
            contexts: Retrieved text contexts
            query_images: Images from TEST case (NEW - use this)
            context_images: Images from retrieved TRAIN cases
            ground_truth: Dict with diagnosis, diagnosis_type, species
            image_paths: DEPRECATED - use query_images
        
        Returns:
            RAGAsResult with all metric scores including diagnosis accuracy
        """
        # Handle legacy image_paths parameter
        if query_images is None and image_paths is not None:
            query_images = image_paths
        
        # Detect no-RAG baseline (empty contexts)
        is_norag = not contexts or len(contexts) == 0
        
        result = RAGAsResult(
            qid=qid,
            judge_model=self.model_name,
            generation_mode=generation_mode,
            retrieval_support_status=retrieval_support_status,
            traces={
                "has_query_images": bool(query_images),
                "has_context_images": bool(context_images),
                "has_ground_truth": bool(ground_truth),
                "has_ground_truth_pseudolabel": bool(ground_truth_pseudolabel),
                "is_norag_baseline": is_norag,
                "generation_mode": generation_mode,
                "retrieval_support_status": retrieval_support_status,
                "ground_truth_type": (ground_truth or {}).get("diagnosis_type", ""),
            }
        )
        
        # Skip retrieval-grounded metrics for no-RAG baseline or diagnosis-only mode
        if is_norag or not evaluate_retrieval_metrics:
            # These metrics require retrieved contexts - skip for no-RAG
            result.multimodal_faithfulness = None
            result.multimodal_relevance = None
            result.context_relevance = None
            skip_reason = "No contexts (no-RAG baseline)" if is_norag else "Retrieval metrics disabled (diagnosis-only evaluation)"
            result.traces["ragas_metric_diagnostics"] = {
                "multimodal_faithfulness": {"status": "skipped", "reason": skip_reason},
                "multimodal_relevance": {"status": "skipped", "reason": skip_reason},
                "context_relevance": {"status": "skipped", "reason": skip_reason},
            }
            if is_norag:
                result.traces["skipped_ragas"] = skip_reason
            else:
                result.traces["skipped_ragas"] = skip_reason
        else:
            # For multimodal metrics: use bounded text contexts + query images from TEST case.
            safe_query, safe_answer, safe_contexts, retrieved_contexts, prep_stats = self._prepare_metric_inputs(
                query=query,
                answer=answer,
                contexts=contexts,
                query_images=query_images,
            )
            result.traces["metric_input_stats"] = prep_stats
            metric_diagnostics = result.traces.setdefault("ragas_metric_diagnostics", {})

            if not retrieved_contexts:
                # Guard against unusable contexts after preprocessing.
                result.multimodal_faithfulness = None
                result.multimodal_relevance = None
                result.context_relevance = None
                result.traces["skipped_ragas"] = "No usable contexts after preprocessing"
                metric_diagnostics["multimodal_faithfulness"] = {
                    "status": "skipped",
                    "reason": "No usable contexts after preprocessing",
                }
                metric_diagnostics["multimodal_relevance"] = {
                    "status": "skipped",
                    "reason": "No usable contexts after preprocessing",
                }
                metric_diagnostics["context_relevance"] = {
                    "status": "skipped",
                    "reason": "No usable contexts after preprocessing",
                }
            else:
                # Run metrics SEQUENTIALLY to avoid rate limiting (GPT 5.2 advice)
                try:
                    result.multimodal_faithfulness = await self.evaluate_multimodal_faithfulness(
                        response=safe_answer,
                        retrieved_contexts=retrieved_contexts
                    )
                except Exception as e:
                    print(f"MultiModalFaithfulness error: {e}")
                    self._last_metric_status["MultiModalFaithfulness"] = {
                        "status": "error",
                        "attempts": 1,
                        "error_type": self._classify_metric_error(str(e)),
                        "error_message": str(e)[:400],
                    }
                metric_diagnostics["multimodal_faithfulness"] = self._last_metric_status.get(
                    "MultiModalFaithfulness", {"status": "unknown"}
                )
            
                try:
                    result.multimodal_relevance = await self.evaluate_multimodal_relevance(
                        user_input=safe_query,
                        response=safe_answer,
                        retrieved_contexts=retrieved_contexts
                    )
                except Exception as e:
                    print(f"MultiModalRelevance error: {e}")
                    self._last_metric_status["MultiModalRelevance"] = {
                        "status": "error",
                        "attempts": 1,
                        "error_type": self._classify_metric_error(str(e)),
                        "error_message": str(e)[:400],
                    }
                metric_diagnostics["multimodal_relevance"] = self._last_metric_status.get(
                    "MultiModalRelevance", {"status": "unknown"}
                )

                if safe_contexts:
                    try:
                        result.context_relevance = await self.evaluate_context_relevance(
                            user_input=safe_query,
                            retrieved_contexts=safe_contexts  # Text contexts only
                        )
                    except Exception as e:
                        print(f"ContextRelevance error: {e}")
                        self._last_metric_status["ContextRelevance"] = {
                            "status": "error",
                            "attempts": 1,
                            "error_type": self._classify_metric_error(str(e)),
                            "error_message": str(e)[:400],
                        }
                    metric_diagnostics["context_relevance"] = self._last_metric_status.get(
                        "ContextRelevance", {"status": "unknown"}
                    )
                else:
                    result.context_relevance = None
                    metric_diagnostics["context_relevance"] = {
                        "status": "skipped",
                        "reason": "No text contexts after preprocessing",
                    }
        
        # Calculate diagnosis accuracy using LLM-as-Judge (Q1 Standard)
        # Per Dinc et al. (2025): LLM judges achieve Kappa=0.852 with specialists
        # NOTE: This works for BOTH RAG and no-RAG baselines
        if ground_truth:
            try:
                diag_result = await self.evaluate_diagnosis_equivalence(
                    prediction=answer,
                    ground_truth=ground_truth,
                    clinical_context=query,
                    query_images=query_images,
                    use_llm_judge=True
                )
                result.diagnosis_accuracy = diag_result.diagnosis_score
                result.diagnosis_type_accuracy = diag_result.diagnosis_type_score
                result.diagnosis_reasoning = diag_result.reasoning
                result.diagnosis_method = diag_result.method
                result.gt_rank = diag_result.gt_rank
                result.top3_hit = diag_result.top3_hit
                result.l3_top1_correct = diag_result.l3_top1_correct
                result.fallback_level = diag_result.fallback_level
            except Exception as e:
                print(f"Diagnosis equivalence error: {e}")
                # Fallback to legacy string matching
                legacy_scores = calculate_diagnosis_accuracy(answer, ground_truth)
                parser_contract = evaluate_ranked_diagnosis_contract(answer, ground_truth)
                result.diagnosis_accuracy = legacy_scores["diagnosis_accuracy"]
                result.diagnosis_type_accuracy = legacy_scores["diagnosis_type_accuracy"]
                result.diagnosis_method = "string_match_fallback"
                result.gt_rank = _coerce_optional_rank(parser_contract.get("gt_rank"))
                result.top3_hit = _coerce_optional_binary_score(parser_contract.get("top3_hit"))
                result.l3_top1_correct = _coerce_optional_binary_score(parser_contract.get("l3_top1_correct"))
                result.fallback_level = str(parser_contract.get("fallback_level", ""))[:80]

        if ground_truth_pseudolabel:
            try:
                diag_result_pseudo = await self.evaluate_diagnosis_equivalence(
                    prediction=answer,
                    ground_truth=ground_truth_pseudolabel,
                    clinical_context=query,
                    query_images=query_images,
                    use_llm_judge=True
                )
                result.diagnosis_accuracy_pseudolabel = diag_result_pseudo.diagnosis_score
                result.diagnosis_type_accuracy_pseudolabel = diag_result_pseudo.diagnosis_type_score
                result.diagnosis_reasoning_pseudolabel = diag_result_pseudo.reasoning
                result.diagnosis_method_pseudolabel = diag_result_pseudo.method
                result.gt_rank_pseudolabel = diag_result_pseudo.gt_rank
                result.top3_hit_pseudolabel = diag_result_pseudo.top3_hit
                result.l3_top1_correct_pseudolabel = diag_result_pseudo.l3_top1_correct
                result.fallback_level_pseudolabel = diag_result_pseudo.fallback_level
            except Exception as e:
                print(f"Diagnosis equivalence error (pseudolabel): {e}")
                legacy_scores = calculate_diagnosis_accuracy(answer, ground_truth_pseudolabel)
                parser_contract = evaluate_ranked_diagnosis_contract(answer, ground_truth_pseudolabel)
                result.diagnosis_accuracy_pseudolabel = legacy_scores["diagnosis_accuracy"]
                result.diagnosis_type_accuracy_pseudolabel = legacy_scores["diagnosis_type_accuracy"]
                result.diagnosis_method_pseudolabel = "string_match_fallback"
                result.gt_rank_pseudolabel = _coerce_optional_rank(parser_contract.get("gt_rank"))
                result.top3_hit_pseudolabel = _coerce_optional_binary_score(parser_contract.get("top3_hit"))
                result.l3_top1_correct_pseudolabel = _coerce_optional_binary_score(
                    parser_contract.get("l3_top1_correct")
                )
                result.fallback_level_pseudolabel = str(parser_contract.get("fallback_level", ""))[:80]

        reasoning_recall = await self.evaluate_reasoning_recall_for_sample(
            qid=qid,
            answer=answer,
        )
        result.reasoning_recall = reasoning_recall.get("recall")
        result.reasoning_recall_method = str(reasoning_recall.get("method") or "")
        result.reasoning_recall_groundtruth_count = reasoning_recall.get("groundtruth_count")
        result.reasoning_recall_matched_count = reasoning_recall.get("matched_groundtruth_count")
        result.reasoning_recall_explanation = reasoning_recall.get("explanation")
        result.reasoning_recall_matching_dict = {
            "matched_groundtruth_indices": reasoning_recall.get("matched_groundtruth_indices", []),
            "matched_groundtruth_points": reasoning_recall.get("matched_groundtruth_points", []),
            "unmatched_groundtruth_points": reasoning_recall.get("unmatched_groundtruth_points", []),
        }
        result.reasoning_recall_source = str(reasoning_recall.get("source_path") or "")
        result.reasoning_recall_source_id = str(reasoning_recall.get("source_id") or "")
        result.reasoning_recall_source_path = str(reasoning_recall.get("source_path") or "")
        result.reasoning_recall_judge_model = str(reasoning_recall.get("judge_model") or "")
        result.reasoning_trace_source = "answer_proxy"
        result.traces["reasoning_recall_diagnostics"] = {
            "method": result.reasoning_recall_method,
            "groundtruth_count": result.reasoning_recall_groundtruth_count,
            "matched_count": result.reasoning_recall_matched_count,
            "predicted_step_count": len(reasoning_recall.get("predicted_reasoning_steps") or []),
            "judge_model": result.reasoning_recall_judge_model,
            "requested_judge_model": str(reasoning_recall.get("requested_judge_model") or ""),
            "source_id": result.reasoning_recall_source_id,
            "source_path": result.reasoning_recall_source,
        }

        result.diagnosis_family, result.diagnosis_family_accuracy = compute_family_metric(answer, ground_truth)
        
        # Flag parametric knowledge usage (per Claude 4.5 + Grok 4.1 analysis)
        # For no-RAG baseline, all correct answers are by definition from parametric knowledge
        if is_norag:
            if result.diagnosis_accuracy is not None and result.diagnosis_accuracy >= 0.5:
                result.traces["parametric_knowledge_suspected"] = True
                result.traces["grounded_accuracy"] = 0.0  # No retrieval = no grounding
            else:
                result.traces["parametric_knowledge_suspected"] = False
                result.traces["grounded_accuracy"] = 0.0
        elif result.context_relevance is not None and result.diagnosis_accuracy is not None:
            # RAG case: If context is irrelevant but diagnosis is correct, LLM is using pre-trained knowledge
            if result.context_relevance < 0.2 and result.diagnosis_accuracy >= 0.8:
                result.traces["parametric_knowledge_suspected"] = True
                result.traces["grounded_accuracy"] = 0.0  # Not grounded in retrieval
            else:
                result.traces["parametric_knowledge_suspected"] = False
                result.traces["grounded_accuracy"] = result.diagnosis_accuracy
        
        return result


def _is_result_complete(record: Dict[str, Any]) -> bool:
    """
    Determine whether an existing RAGAS record is complete for resume purposes.

    RAG runs require all retrieval-grounded metrics plus diagnosis metrics.
    No-RAG runs intentionally store retrieval-grounded metrics as None, so
    diagnosis metrics are sufficient to consider the sample complete.
    """
    def _is_valid_number(v: Any) -> bool:
        return isinstance(v, (int, float)) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))

    diag_ready = (
        _is_valid_number(record.get("diagnosis_accuracy"))
        and _is_valid_number(record.get("diagnosis_type_accuracy"))
        and bool(record.get("diagnosis_method"))
    )

    # no-RAG records intentionally skip retrieval-grounded metrics
    is_norag = bool(record.get("traces", {}).get("is_norag_baseline", False))
    if is_norag:
        return diag_ready

    retrieval_ready = (
        _is_valid_number(record.get("multimodal_faithfulness"))
        and _is_valid_number(record.get("multimodal_relevance"))
        and _is_valid_number(record.get("context_relevance"))
    )
    return retrieval_ready and diag_ready


def _sanitize_record_for_json(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace NaN/Inf float values with None for JSON portability and clean resume logic."""

    def _clean(value: Any) -> Any:
        if isinstance(value, float):
            if math.isnan(value) or math.isinf(value):
                return None
            return value
        if isinstance(value, dict):
            return {k: _clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [_clean(v) for v in value]
        return value

    return _clean(record)


def _dedupe_rows_keep_last(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate result rows by qid, keeping the last occurrence."""
    latest_by_qid: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        qid = row.get("qid")
        if qid:
            latest_by_qid[qid] = row

    deduped_rows: List[Dict[str, Any]] = []
    seen: Set[str] = set()
    for row in reversed(rows):
        qid = row.get("qid")
        if not qid or qid in seen:
            continue
        seen.add(qid)
        deduped_rows.append(latest_by_qid[qid])

    deduped_rows.reverse()
    return deduped_rows


def run_ragas_evaluation(
    run_dir: Path,
    answers_file: str = "answers.jsonl",
    judge_model: str = None,
    max_samples: int = None,
    delay_seconds: float = 1.0,
    resume: bool = True,
    diagnosis_batch_api: bool = False,
    diagnosis_batch_poll_seconds: float = 10.0,
    diagnosis_batch_timeout_seconds: int = 7200,
    evaluate_retrieval_metrics: bool = True,
    start_phase: str = "all",
) -> Path:
    """Run phased, resume-safe RAGAS evaluation on generated answers."""
    from .run_ragas_phased import run_ragas_evaluation_phased

    return run_ragas_evaluation_phased(
        run_dir=run_dir,
        answers_file=answers_file,
        judge_model=judge_model,
        max_samples=max_samples,
        delay_seconds=delay_seconds,
        resume=resume,
        diagnosis_batch_api=diagnosis_batch_api,
        diagnosis_batch_poll_seconds=diagnosis_batch_poll_seconds,
        diagnosis_batch_timeout_seconds=diagnosis_batch_timeout_seconds,
        evaluate_retrieval_metrics=evaluate_retrieval_metrics,
        start_phase=start_phase,
    )


if __name__ == "__main__":
    # Test initialization
    print("Testing RAGAS Library Evaluator...")
    evaluator = RAGAsLibraryEvaluator()
    print(f"✓ Evaluator created with model: {evaluator.model_name}")
    print(f"✓ RAGAS source path: {RAGAS_SRC}")
    
    # Test metric initialization
    try:
        evaluator._init_metrics()
        print("✓ All RAGAS metrics initialized successfully")
        print("  - MultiModalFaithfulness (Binary 0/1)")
        print("  - MultiModalRelevance (Binary 0/1)")
        print("  - ContextRelevance (Continuous 0-1)")
    except Exception as e:
        print(f"✗ Failed to initialize metrics: {e}")
