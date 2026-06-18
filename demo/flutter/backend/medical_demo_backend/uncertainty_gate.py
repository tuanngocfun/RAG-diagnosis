"""Two-stage uncertainty gate for the medical demo."""

from __future__ import annotations

import io
from collections import Counter
from typing import List

from PIL import Image

from .kb import tokenize
from .types import GateEvaluation, GeneratorOutput, RetrievedEvidence

OUT_OF_SCOPE_TOKENS = {
    "eczema",
    "psoriasis",
    "carcinoma",
    "melanoma",
    "bacterial",
    "cellulitis",
}


def image_is_usable(image_bytes: bytes | None) -> bool:
    """Check whether the optional image payload is readable."""
    if not image_bytes:
        return False
    try:
        image = Image.open(io.BytesIO(image_bytes))
        width, height = image.size
        return width >= 32 and height >= 32
    except Exception:
        return False


def _missing_required_inputs(patient_text: str) -> List[str]:
    cleaned = patient_text.strip()
    if not cleaned:
        return ["clinical description"]
    if len(cleaned) < 24:
        return ["more detailed symptom history"]
    return []


def _requires_image_reasoning(patient_text: str) -> bool:
    lowered = patient_text.lower()
    return any(token in lowered for token in ("image only", "photo only", "see attached", "from the image"))


def _query_has_confirmatory_signal(patient_text: str) -> bool:
    lowered = patient_text.lower()
    return any(token in lowered for token in ("smear", "biopsy", "pcr", "amastigotes", "histopathology"))


def classify_retrieval_support(patient_text: str, contexts: List[RetrievedEvidence]) -> str:
    """Classify retrieval support in a product-facing way."""
    if not contexts:
        return "empty_contexts"

    top_score = contexts[0].score
    confirmatory_count = sum(1 for context in contexts[:3] if context.confirmatory)
    query_confirmatory = _query_has_confirmatory_signal(patient_text)
    if top_score >= 1.05 and confirmatory_count >= 1:
        return "supported"
    if top_score >= 0.75 and query_confirmatory and confirmatory_count >= 1:
        return "supported"
    if top_score >= 0.24 and len(contexts) >= 2:
        return "weak_support"
    if top_score > 0.0:
        return "low_support_no_confirmatory_context"
    return "empty_contexts"


def evidence_conflict_flag(contexts: List[RetrievedEvidence]) -> bool:
    """Detect materially conflicting evidence in the top results."""
    if len(contexts) < 2:
        return False
    labels = Counter(context.diagnosis_label for context in contexts[:3])
    if len(labels) < 2:
        return False
    first = contexts[0]
    second = contexts[1]
    if first.diagnosis_label != second.diagnosis_label and second.score >= (first.score * 0.45):
        return True
    return first.diagnosis_label != second.diagnosis_label and abs(first.score - second.score) <= 0.08


def corpus_gap_flag(patient_text: str, contexts: List[RetrievedEvidence]) -> bool:
    """Detect obvious out-of-scope requests for the leish-only KB."""
    query_tokens = set(tokenize(patient_text))
    if query_tokens & OUT_OF_SCOPE_TOKENS:
        return True
    if not contexts:
        return False
    return contexts[0].score < 0.15 and len(query_tokens) >= 6


def evaluate_evidence_gate(patient_text: str, image_bytes: bytes | None, contexts: List[RetrievedEvidence]) -> GateEvaluation:
    """Run stage 1 gate."""
    missing_inputs = _missing_required_inputs(patient_text)
    image_usable = image_is_usable(image_bytes)
    trigger_codes: List[str] = []
    retrieval_support = classify_retrieval_support(patient_text, contexts)
    top_score = contexts[0].score if contexts else 0.0
    conflict = evidence_conflict_flag(contexts)
    corpus_gap = corpus_gap_flag(patient_text, contexts)

    if missing_inputs:
        trigger_codes.append("missing_required_inputs")
    if _requires_image_reasoning(patient_text) and not image_usable:
        trigger_codes.append("image_required_but_unusable")
    if retrieval_support == "empty_contexts":
        trigger_codes.append("empty_contexts")
    if retrieval_support == "low_support_no_confirmatory_context":
        trigger_codes.append("low_support_no_confirmatory_context")
    if conflict:
        trigger_codes.append("evidence_conflict")
    if corpus_gap:
        trigger_codes.append("corpus_gap")

    if missing_inputs or "image_required_but_unusable" in trigger_codes or conflict or corpus_gap:
        outcome = "hard_abstain"
    elif retrieval_support == "supported":
        outcome = "rag_supported"
    else:
        outcome = "parametric_fallback_allowed"

    return GateEvaluation(
        outcome=outcome,
        stage="evidence",
        retrieval_support_status=retrieval_support,
        top_score=round(top_score, 4),
        evidence_conflict_flag=conflict,
        image_usable=image_usable,
        missing_required_inputs=missing_inputs,
        gate_trigger_codes=trigger_codes,
        escalation_required=True,
    )


def evaluate_parametric_gate(evidence_gate: GateEvaluation, generator_output: GeneratorOutput) -> GateEvaluation:
    """Run stage 2 gate on model-only fallback output."""
    trigger_codes = list(evidence_gate.gate_trigger_codes)
    confidence = generator_output.confidence.lower().strip()
    diagnosis = generator_output.diagnosis.strip()
    malformed = not diagnosis
    if malformed:
        trigger_codes.append("malformed_model_output")
    if diagnosis.upper() == "INSUFFICIENT":
        trigger_codes.append("model_returned_insufficient")
    if confidence == "low":
        trigger_codes.append("low_model_confidence")
    if generator_output.insufficient_reason:
        trigger_codes.append("insufficient_reason_present")

    outcome = "provisional_parametric"
    if malformed or diagnosis.upper() == "INSUFFICIENT" or confidence == "low":
        outcome = "hard_abstain"

    return GateEvaluation(
        outcome=outcome,
        stage="parametric_fallback",
        retrieval_support_status=evidence_gate.retrieval_support_status,
        top_score=evidence_gate.top_score,
        evidence_conflict_flag=evidence_gate.evidence_conflict_flag,
        image_usable=evidence_gate.image_usable,
        missing_required_inputs=list(
            dict.fromkeys(evidence_gate.missing_required_inputs + generator_output.needed_next_inputs)
        ),
        gate_trigger_codes=trigger_codes,
        escalation_required=True,
    )
