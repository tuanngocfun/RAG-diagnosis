"""Core dataclasses for the medical demo backend."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class ChatMessage:
    """Single message in the GPU assistant conversation."""

    role: str
    content: str


@dataclass(frozen=True)
class ConsultationRequest:
    """Normalized consultation request."""

    patient_text: str
    image_bytes: Optional[bytes] = None
    image_filename: str = ""
    client_request_id: str = ""
    device_platform: str = ""


@dataclass(frozen=True)
class ChatRequest:
    """Normalized real-GPU chat request."""

    messages: List[ChatMessage]
    image_bytes: Optional[bytes] = None
    image_filename: str = ""
    client_request_id: str = ""
    device_platform: str = ""
    response_mode: str = "live_gpu"


@dataclass(frozen=True)
class KnowledgeChunk:
    """Single KB chunk."""

    chunk_id: str
    source_case_id: str
    title: str
    diagnosis_label: str
    text: str
    tags: List[str]
    confirmatory: bool = False


@dataclass(frozen=True)
class RetrievedEvidence:
    """Retrieved chunk exposed to the response contract."""

    chunk_id: str
    source_case_id: str
    title: str
    diagnosis_label: str
    text: str
    score: float
    confirmatory: bool

    def to_response_dict(self) -> Dict[str, Any]:
        excerpt = self.text[:220].strip()
        return {
            "chunk_id": self.chunk_id,
            "source_case_id": self.source_case_id,
            "title": self.title,
            "diagnosis_label": self.diagnosis_label,
            "excerpt": excerpt,
            "score": self.score,
            "confirmatory": self.confirmatory,
        }


@dataclass(frozen=True)
class DiagnosisRank:
    """Diagnosis item returned to the app."""

    rank: int
    label: str
    confidence_band: str
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GeneratorOutput:
    """Structured model-like output."""

    diagnosis: str
    confidence: str
    ranked_differential: List[str]
    reasoning: str
    insufficient_reason: Optional[str]
    needed_next_inputs: List[str]
    key_evidence: List[str] = field(default_factory=list)
    answer_markdown: str = ""
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GateEvaluation:
    """Intermediate gate decision."""

    outcome: str
    stage: str
    retrieval_support_status: str
    top_score: float
    evidence_conflict_flag: bool
    image_usable: bool
    missing_required_inputs: List[str]
    gate_trigger_codes: List[str]
    escalation_required: bool


@dataclass(frozen=True)
class ConsultationResponse:
    """Public response contract."""

    request_id: str
    model_name: str
    decision_state: str
    top_diagnoses: List[DiagnosisRank]
    answer_markdown: str
    evidence: List[Dict[str, Any]]
    disclaimer: str
    timing_ms: int
    uncertainty_gate: Dict[str, Any]
    needed_next_inputs: List[str]
    safe_to_show_ranked_differential: bool
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "decision_state": self.decision_state,
            "top_diagnoses": [item.to_dict() for item in self.top_diagnoses],
            "answer_markdown": self.answer_markdown,
            "evidence": list(self.evidence),
            "disclaimer": self.disclaimer,
            "timing_ms": self.timing_ms,
            "uncertainty_gate": dict(self.uncertainty_gate),
            "needed_next_inputs": list(self.needed_next_inputs),
            "safe_to_show_ranked_differential": self.safe_to_show_ranked_differential,
            "runtime_metadata": dict(self.runtime_metadata),
        }


@dataclass(frozen=True)
class ChatResponse:
    """Public response contract for the real-GPU assistant."""

    request_id: str
    model_name: str
    provider_mode: str
    assistant_markdown: str
    evidence: List[Dict[str, Any]]
    disclaimer: str
    timing_ms: int
    safety_state: str
    needed_next_inputs: List[str]
    runtime_metadata: Dict[str, Any] = field(default_factory=dict)
    retrieval_audit: Dict[str, Any] = field(default_factory=dict)
    response_source_mode: str = "live_gpu"
    source_label: str = "fresh local Gemma 4 GPU generation"
    source_path: str = ""
    fresh_generation_executed: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "model_name": self.model_name,
            "provider_mode": self.provider_mode,
            "assistant_markdown": self.assistant_markdown,
            "evidence": list(self.evidence),
            "disclaimer": self.disclaimer,
            "timing_ms": self.timing_ms,
            "safety_state": self.safety_state,
            "needed_next_inputs": list(self.needed_next_inputs),
            "runtime_metadata": dict(self.runtime_metadata),
            "retrieval_audit": dict(self.retrieval_audit),
            "response_source_mode": self.response_source_mode,
            "source_label": self.source_label,
            "source_path": self.source_path,
            "fresh_generation_executed": self.fresh_generation_executed,
        }
