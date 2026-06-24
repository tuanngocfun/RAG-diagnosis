"""Consultation service that orchestrates retrieval, gating, and response shaping."""

from __future__ import annotations

import importlib
import json
import time
import uuid
import os
from pathlib import Path
from typing import List

from .generator import DemoGemma4Generator, GeneratorClient, RealGemma4Generator
from .kb import KnowledgeBase
from .types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ConsultationRequest,
    ConsultationResponse,
    DiagnosisRank,
    GateEvaluation,
    GeneratorOutput,
)
from .uncertainty_gate import evaluate_evidence_gate, evaluate_parametric_gate

DISCLAIMER = (
    "Decision support only. This demo is not a diagnosis system. "
    "The live provider is deterministic demo logic, not real clinical inference. "
    "A licensed clinician should review the case and confirm any suspected diagnosis."
)
REAL_GPU_DISCLAIMER = (
    "Decision support only. This demo uses local Gemma 4 GPU generation with a small local demo KB. "
    "It is not clinical deployment, has not been clinician-validated, and must not be used as a diagnosis system."
)
RERANK_BOUNDARY = (
    "Live backend exposes lexical retrieval scores but no separate re-ranker contract. "
    "Official rerank evidence, when shown, is reference evidence from the Gemma 4 experiment pipeline."
)
ESCALATION_TEXT = (
    "Please seek clinician review or in-person evaluation before acting on this result."
)
CHAT_REQUIRED_INPUTS = [
    "patient age and sex",
    "lesion or systemic symptom timeline",
    "travel or exposure history",
    "exam findings and lesion distribution",
    "confirmatory test status if available",
]
KNOWN_V12D_CASE_IDS = (
    "PMC7516301_01",
    "PMC7456484_01",
    "PMC10026180_04",
)
OFFICIAL_REPLAY_MODE = "official_v12d_replay"
LIVE_GPU_MODE = "live_gpu"
OFFICIAL_REPLAY_LABEL = "official V12d experiment-pipeline replay"
LIVE_GPU_LABEL = "fresh local Gemma 4 GPU generation"


def build_default_service() -> "ConsultService":
    """Build the default service from the local KB pack."""
    base_dir = Path(__file__).resolve().parents[2]
    kb_path = base_dir / "kb" / "leishmaniasis_demo_pack.json"
    kb = KnowledgeBase.from_path(kb_path)
    provider_mode = os.getenv("MEDICAL_DEMO_PROVIDER_MODE", "deterministic_demo").strip().lower()
    if provider_mode in {"real_gpu", "real_gpu_gemma4", "gemma4_real"}:
        return ConsultService(
            knowledge_base=kb,
            generator=RealGemma4Generator(),
            provider_mode="real_gpu_gemma4",
        )
    return ConsultService(
        knowledge_base=kb,
        generator=DemoGemma4Generator(),
        provider_mode="deterministic_demo",
    )


class ConsultService:
    """Primary orchestration service."""

    def __init__(self, knowledge_base: KnowledgeBase, generator: GeneratorClient, provider_mode: str = "deterministic_demo"):
        self.knowledge_base = knowledge_base
        self.generator = generator
        self.provider_mode = provider_mode

    def health(self) -> dict:
        """Expose service health."""
        gpu = self._gpu_readiness()
        model_loaded = getattr(self.generator, "_generator", None) is not None
        chat_available = (
            self.provider_mode == "real_gpu_gemma4"
            and gpu["cuda_available"]
            and gpu["bitsandbytes_available"]
            and (model_loaded or gpu["gpu_free_memory_ready"])
        )
        return {
            "status": "ok",
            "kb_ready": True,
            "kb_path": str(self.knowledge_base.source_path),
            "model_name": self.generator.model_name,
            "provider_mode": self.provider_mode,
            "chat_available": chat_available,
            "model_loaded": model_loaded,
            **gpu,
        }

    def consult(self, request: ConsultationRequest) -> ConsultationResponse:
        """Run the full consultation flow."""
        started = time.perf_counter()
        request_id = request.client_request_id or str(uuid.uuid4())
        contexts = self.knowledge_base.search(request.patient_text, top_k=4)
        evidence_gate = evaluate_evidence_gate(request.patient_text, request.image_bytes, contexts)

        if evidence_gate.outcome == "hard_abstain":
            return self._build_abstained_response(
                request_id=request_id,
                timing_ms=self._timing_ms(started),
                gate=evidence_gate,
                needed_next_inputs=evidence_gate.missing_required_inputs or ["more detailed clinical information"],
            )

        if evidence_gate.outcome == "rag_supported":
            grounded_output = self.generator.generate(
                request.patient_text,
                contexts,
                use_rag=True,
                image_bytes=request.image_bytes,
                image_filename=request.image_filename,
            )
            if grounded_output.confidence.lower().strip() == "low" or grounded_output.diagnosis.upper() == "INSUFFICIENT":
                return self._build_abstained_response(
                    request_id=request_id,
                    timing_ms=self._timing_ms(started),
                    gate=evidence_gate,
                    needed_next_inputs=grounded_output.needed_next_inputs or ["confirmatory testing"],
                )
            return self._build_grounded_response(
                request_id=request_id,
                timing_ms=self._timing_ms(started),
                gate=evidence_gate,
                model_output=grounded_output,
                contexts=contexts,
            )

        fallback_output = self.generator.generate(
            request.patient_text,
            [],
            use_rag=False,
            image_bytes=request.image_bytes,
            image_filename=request.image_filename,
        )
        fallback_gate = evaluate_parametric_gate(evidence_gate, fallback_output)
        if fallback_gate.outcome == "hard_abstain":
            return self._build_abstained_response(
                request_id=request_id,
                timing_ms=self._timing_ms(started),
                gate=fallback_gate,
                needed_next_inputs=fallback_gate.missing_required_inputs or ["more structured clinical details"],
            )
        return self._build_parametric_response(
            request_id=request_id,
            timing_ms=self._timing_ms(started),
            gate=fallback_gate,
            model_output=fallback_output,
        )

    def chat(self, request: ChatRequest) -> ChatResponse:
        """Run the real-GPU structured assistant flow."""
        started = time.perf_counter()
        request_id = request.client_request_id or str(uuid.uuid4())
        response_mode = (request.response_mode or LIVE_GPU_MODE).strip().lower()
        latest_user_text = self._latest_user_text(request.messages)

        if response_mode == OFFICIAL_REPLAY_MODE:
            return self._official_v12d_replay_response(
                request_id=request_id,
                latest_user_text=latest_user_text,
                started=started,
            )

        if response_mode != LIVE_GPU_MODE:
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Unsupported Response Mode",
                        f"`{request.response_mode}` is not supported.",
                        "",
                        "Use `official_v12d_replay` for deck-matching evidence or `live_gpu` for fresh generation.",
                    ]
                ),
                evidence=[],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="invalid_response_mode",
                needed_next_inputs=["choose official_v12d_replay or live_gpu"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": response_mode,
                },
                response_source_mode=response_mode,
                source_label="unsupported response mode",
                fresh_generation_executed=False,
            )

        if self.provider_mode != "real_gpu_gemma4":
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Real GPU Mode Required",
                        "The GPU assistant is disabled because this backend is not running in real Gemma 4 mode.",
                        "",
                        "Start the backend with `MEDICAL_DEMO_PROVIDER_MODE=real_gpu_gemma4` to enable live model generation.",
                    ]
                ),
                evidence=[],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="real_gpu_required",
                needed_next_inputs=["restart backend in real GPU mode"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": LIVE_GPU_MODE,
                    "fresh_generation_executed": False,
                },
                response_source_mode=LIVE_GPU_MODE,
                source_label=LIVE_GPU_LABEL,
                fresh_generation_executed=False,
            )

        missing_inputs = self._chat_missing_inputs(latest_user_text)
        if missing_inputs:
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## More Clinical Detail Needed",
                        "The assistant is not generating a differential from this input because the case summary is too limited.",
                        "",
                        "### Useful next details",
                        *[f"- {item}" for item in missing_inputs],
                    ]
                ),
                evidence=[],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="needs_more_input",
                needed_next_inputs=missing_inputs,
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": LIVE_GPU_MODE,
                    "fresh_generation_executed": False,
                },
                response_source_mode=LIVE_GPU_MODE,
                source_label=LIVE_GPU_LABEL,
                fresh_generation_executed=False,
            )

        contexts, retrieval_audit = self.knowledge_base.search_with_audit(
            latest_user_text, top_k=4
        )
        retrieval_audit = self._build_chat_retrieval_audit(
            request_id=request_id,
            latest_user_text=latest_user_text,
            live_audit=retrieval_audit,
        )
        prompt = self._build_chat_prompt(request.messages)
        try:
            model_output = self.generator.generate(
                prompt,
                contexts,
                use_rag=True,
                image_bytes=request.image_bytes,
                image_filename=request.image_filename,
            )
        except Exception as exc:
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Model Unavailable",
                        "The local GPU model did not complete generation, so no clinical suggestion is being shown.",
                        "",
                        f"Technical error: `{type(exc).__name__}`",
                    ]
                ),
                evidence=[context.to_response_dict() for context in contexts[:3]],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="model_error",
                needed_next_inputs=["check GPU memory and backend logs", "retry after confirming CUDA readiness"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "error_type": type(exc).__name__,
                    "response_source_mode": LIVE_GPU_MODE,
                    "fresh_generation_executed": True,
                },
                retrieval_audit=retrieval_audit,
                response_source_mode=LIVE_GPU_MODE,
                source_label=LIVE_GPU_LABEL,
                fresh_generation_executed=True,
            )

        safety_state = "generated_support"
        if model_output.confidence.lower().strip() == "low" or model_output.insufficient_reason:
            safety_state = "low_confidence"
        if model_output.answer_markdown.strip().startswith("[Generation Error:"):
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Model Unavailable",
                        "The local GPU model returned a generation error, so no clinical suggestion is being shown.",
                        "",
                        model_output.answer_markdown.strip(),
                    ]
                ),
                evidence=[context.to_response_dict() for context in contexts[:3]],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="model_error",
                needed_next_inputs=["check GPU memory and backend logs", "retry after confirming CUDA readiness"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": LIVE_GPU_MODE,
                    "fresh_generation_executed": True,
                    **dict(model_output.runtime_metadata),
                },
                retrieval_audit=retrieval_audit,
                response_source_mode=LIVE_GPU_MODE,
                source_label=LIVE_GPU_LABEL,
                fresh_generation_executed=True,
            )

        return ChatResponse(
            request_id=request_id,
            model_name=self.generator.model_name,
            provider_mode=self.provider_mode,
            assistant_markdown=self._safety_frame_chat_answer(
                model_output.answer_markdown or model_output.reasoning
            ),
            evidence=[context.to_response_dict() for context in contexts[:3]],
            disclaimer=self._disclaimer(),
            timing_ms=self._timing_ms(started),
            safety_state=safety_state,
            needed_next_inputs=list(model_output.needed_next_inputs),
            runtime_metadata={
                "provider_mode": self.provider_mode,
                "response_source_mode": LIVE_GPU_MODE,
                "fresh_generation_executed": True,
                **dict(model_output.runtime_metadata),
            },
            retrieval_audit=retrieval_audit,
            response_source_mode=LIVE_GPU_MODE,
            source_label=LIVE_GPU_LABEL,
            fresh_generation_executed=True,
        )

    @staticmethod
    def _timing_ms(started: float) -> int:
        return int((time.perf_counter() - started) * 1000)

    def _official_v12d_replay_response(
        self,
        *,
        request_id: str,
        latest_user_text: str,
        started: float,
    ) -> ChatResponse:
        """Return the saved V12d answer used by the presentation deck."""
        case_id = self._known_v12d_case_id(
            request_id=request_id,
            latest_user_text=latest_user_text,
        )
        results_dir = self._official_v12d_results_dir()
        if not case_id:
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Known V12d Case ID Required",
                        "Official deck replay is available only for the selected V12d held-out cases.",
                        "",
                        "Include one of these case IDs in the prompt:",
                        *[f"- `{item}`" for item in KNOWN_V12D_CASE_IDS],
                    ]
                ),
                evidence=[],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="known_case_required",
                needed_next_inputs=["include one selected V12d case ID"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": OFFICIAL_REPLAY_MODE,
                    "fresh_generation_executed": False,
                    "official_results_dir": str(results_dir),
                },
                response_source_mode=OFFICIAL_REPLAY_MODE,
                source_label=OFFICIAL_REPLAY_LABEL,
                source_path=str(results_dir),
                fresh_generation_executed=False,
            )

        result_path = results_dir / f"{case_id}_result.json"
        if not result_path.exists():
            return ChatResponse(
                request_id=request_id,
                model_name=self.generator.model_name,
                provider_mode=self.provider_mode,
                assistant_markdown="\n".join(
                    [
                        "## Official Replay Unavailable",
                        f"The saved V12d result for `{case_id}` was not found.",
                        "",
                        f"Expected file: `{result_path}`",
                    ]
                ),
                evidence=[],
                disclaimer=self._disclaimer(),
                timing_ms=self._timing_ms(started),
                safety_state="official_replay_unavailable",
                needed_next_inputs=["restore the V12d held-out result artifact"],
                runtime_metadata={
                    "provider_mode": self.provider_mode,
                    "response_source_mode": OFFICIAL_REPLAY_MODE,
                    "fresh_generation_executed": False,
                    "official_results_dir": str(results_dir),
                },
                response_source_mode=OFFICIAL_REPLAY_MODE,
                source_label=OFFICIAL_REPLAY_LABEL,
                source_path=str(result_path),
                fresh_generation_executed=False,
            )

        result = json.loads(result_path.read_text(encoding="utf-8"))
        raw_response = result.get("raw_response") or {}
        metadata = result.get("metadata") or {}
        assistant_markdown = (
            result.get("assistant_markdown")
            or raw_response.get("assistant_markdown")
            or "The saved V12d response is empty."
        )
        evidence = list(result.get("evidence") or raw_response.get("evidence") or [])
        raw_runtime = dict(raw_response.get("runtime_metadata") or {})
        saved_timing_ms = raw_response.get("timing_ms")
        saved_elapsed_seconds = metadata.get("elapsed_seconds")
        runtime_metadata = {
            **raw_runtime,
            "provider_mode": raw_response.get("provider_mode")
            or metadata.get("provider_mode")
            or self.provider_mode,
            "model_name": raw_response.get("model_name")
            or metadata.get("model_name")
            or self.generator.model_name,
            "response_source_mode": OFFICIAL_REPLAY_MODE,
            "source_label": OFFICIAL_REPLAY_LABEL,
            "source_path": str(result_path),
            "official_replay_case_id": case_id,
            "fresh_generation_executed": False,
            "saved_request_id": raw_response.get("request_id") or metadata.get("request_id", ""),
            "saved_timing_ms": saved_timing_ms,
            "saved_elapsed_seconds": saved_elapsed_seconds,
            "saved_query_image_tensor_count": metadata.get("query_image_tensor_count"),
        }
        return ChatResponse(
            request_id=request_id,
            model_name=runtime_metadata["model_name"],
            provider_mode=runtime_metadata["provider_mode"],
            assistant_markdown=assistant_markdown,
            evidence=evidence,
            disclaimer=raw_response.get("disclaimer") or self._disclaimer(),
            timing_ms=self._timing_ms(started),
            safety_state=raw_response.get("safety_state") or result.get("safety_state", "generated_support"),
            needed_next_inputs=list(
                raw_response.get("needed_next_inputs")
                or ["clinician review", "confirmatory laboratory or histopathology testing"]
            ),
            runtime_metadata=runtime_metadata,
            retrieval_audit=self._official_replay_retrieval_audit(
                request_id=request_id,
                latest_user_text=latest_user_text,
                case_id=case_id,
                result=result,
                evidence=evidence,
            ),
            response_source_mode=OFFICIAL_REPLAY_MODE,
            source_label=OFFICIAL_REPLAY_LABEL,
            source_path=str(result_path),
            fresh_generation_executed=False,
        )

    @staticmethod
    def _official_v12d_results_dir() -> Path:
        return Path(
            os.getenv(
                "MEDICAL_DEMO_OFFICIAL_V12D_RESULTS_DIR",
                str(
                    Path(__file__).resolve().parents[3]
                    / "presentation"
                    / "v12d"
                    / "data"
                    / "heldout_evaluation_results"
                ),
            )
        )

    @staticmethod
    def _known_v12d_case_id(*, request_id: str, latest_user_text: str) -> str:
        match_text = f"{request_id}\n{latest_user_text}"
        return next(
            (candidate for candidate in KNOWN_V12D_CASE_IDS if candidate in match_text),
            "",
        )

    def _official_replay_retrieval_audit(
        self,
        *,
        request_id: str,
        latest_user_text: str,
        case_id: str,
        result: dict,
        evidence: list,
    ) -> dict:
        split_provenance = result.get("split_provenance") or {}
        returned_contexts = []
        for index, item in enumerate(evidence, start=1):
            context = dict(item)
            context["rank"] = index
            returned_contexts.append(context)
        return {
            "retrieval_backend": "official_v12d_saved_demo_output",
            "kb_path": split_provenance.get("runtime_retrieval_kb_source", ""),
            "top_k_requested": len(returned_contexts),
            "candidate_count": len(returned_contexts),
            "returned_count": len(returned_contexts),
            "scoring_method": (
                "saved local demo KB scores from the V12d held-out output; "
                "no live retrieval executed in replay mode"
            ),
            "returned_contexts": returned_contexts,
            "live_rerank_executed": False,
            "live_rerank_method": None,
            "rerank_boundary": (
                "Official replay mode returns the saved deck-matching V12d answer. "
                "Fresh live retriever/reranker execution is not claimed."
            ),
            "official_rerank_reference": self._official_rerank_reference(
                request_id=request_id,
                latest_user_text=f"{case_id}\n{latest_user_text}",
            ),
        }

    @staticmethod
    def _safety_frame_chat_answer(answer: str) -> str:
        """Rewrite model-format labels into decision-support language."""
        text = answer.strip()
        replacements = {
            "## DIAGNOSIS PREDICTION": "## Supportive Differential (Not a Diagnosis)",
            "**Rank 1 (Most Likely):**": "**Rank 1 supportive consideration:**",
            "**Chosen Final Diagnosis for Scoring:**": "**Most supported option in this research demo:**",
            "**Evidence Source:**": "**Evidence source used by the model:**",
        }
        for source, target in replacements.items():
            text = text.replace(source, target)
        if not text:
            text = (
                "The model returned an empty answer. Treat this as unavailable and seek clinician review."
            )
        return "\n\n".join(
            [
                "> Safety boundary: research decision support only; not ground truth, not clinical validation, and not a diagnosis system.",
                text,
            ]
        )

    @staticmethod
    def _latest_user_text(messages: List[ChatMessage]) -> str:
        for message in reversed(messages):
            if message.role.lower().strip() == "user":
                return message.content.strip()
        return ""

    @staticmethod
    def _chat_missing_inputs(text: str) -> List[str]:
        cleaned = " ".join(text.split())
        if not cleaned:
            return CHAT_REQUIRED_INPUTS
        if len(cleaned) < 40:
            return ["more detailed clinical summary", *CHAT_REQUIRED_INPUTS[:3]]
        return []

    @staticmethod
    def _build_chat_prompt(messages: List[ChatMessage]) -> str:
        rendered_messages = []
        for message in messages[-8:]:
            role = message.role.lower().strip() or "user"
            content = message.content.strip()
            if not content:
                continue
            rendered_messages.append(f"{role.upper()}: {content}")
        conversation = "\n\n".join(rendered_messages)
        return "\n".join(
            [
                "You are a research-prototype clinical decision-support assistant for leishmaniasis case reasoning.",
                "Do not claim to diagnose the patient. Do not claim the answer is ground truth.",
                "Do not claim clinical validation or expert validation.",
                "Use cautious wording: 'could support', 'should consider', 'needs confirmation'.",
                "Return concise Markdown with these sections:",
                "1. Supportive differential",
                "2. Reasoning",
                "3. Uncertainty and missing information",
                "4. Suggested confirmatory tests or clinician review",
                "5. Evidence used",
                "",
                "Conversation:",
                conversation,
            ]
        )

    def _build_chat_retrieval_audit(
        self,
        *,
        request_id: str,
        latest_user_text: str,
        live_audit: dict,
    ) -> dict:
        """Build the public retriever/reranker audit for GPU chat responses."""
        audit = {
            **dict(live_audit),
            "live_rerank_executed": False,
            "live_rerank_method": None,
            "rerank_boundary": RERANK_BOUNDARY,
            "official_rerank_reference": self._official_rerank_reference(
                request_id=request_id,
                latest_user_text=latest_user_text,
            ),
        }
        return audit

    def _official_rerank_reference(
        self,
        *,
        request_id: str,
        latest_user_text: str,
    ) -> dict:
        """Attach exact V12d official rerank trace fields when a known case is named."""
        trace_path = Path(
            os.getenv(
                "MEDICAL_DEMO_OFFICIAL_RAG_TRACE_PATH",
                str(
                    Path(__file__).resolve().parents[3]
                    / "presentation"
                    / "v12d"
                    / "data"
                    / "exact_rag_trace_appendix"
                    / "trace_summary.json"
                ),
            )
        )
        match_text = f"{request_id}\n{latest_user_text}"
        case_id = next(
            (candidate for candidate in KNOWN_V12D_CASE_IDS if candidate in match_text),
            "",
        )
        if not case_id:
            return {
                "available": False,
                "reason": "no known V12d held-out case ID found in request text or request ID",
                "source_path": str(trace_path),
            }
        if not trace_path.exists():
            return {
                "available": False,
                "reason": "official V12d trace file is unavailable",
                "case_id": case_id,
                "source_path": str(trace_path),
            }
        try:
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
            run = trace.get("official_rag_run", {})
            case = next(
                item for item in trace.get("cases", []) if item.get("case_id") == case_id
            )
            official = case.get("official_rag_trace", {})
        except Exception as exc:
            return {
                "available": False,
                "reason": f"official V12d trace file could not be read: {type(exc).__name__}",
                "case_id": case_id,
                "source_path": str(trace_path),
            }
        return {
            "available": True,
            "source_label": "official Gemma 4 experiment-pipeline trace",
            "source_path": str(trace_path),
            "case_id": case_id,
            "qid": case.get("qid", ""),
            "retriever_method": run.get("retriever_method"),
            "rerank": run.get("rerank"),
            "retrieval_top_k": run.get("retrieval_top_k"),
            "context_count": official.get("context_count", 0),
            "contexts": list(official.get("top_contexts_for_slide", [])),
            "boundary": (
                "Reference-only: this is the official rerank-enabled final context "
                "list, not a live reranker call from the demo backend."
            ),
        }

    @staticmethod
    def _gpu_readiness() -> dict:
        min_free_mib = int(os.getenv("MEDICAL_DEMO_MIN_FREE_VRAM_MIB", "12000"))
        gpu = {
            "cuda_available": False,
            "gpu_name": "",
            "gpu_memory_total_mib": None,
            "gpu_memory_free_mib": None,
            "gpu_min_free_mib": min_free_mib,
            "gpu_free_memory_ready": False,
            "bitsandbytes_available": False,
        }
        try:
            torch = importlib.import_module("torch")
            cuda_available = bool(torch.cuda.is_available())
            gpu["cuda_available"] = cuda_available
            if cuda_available:
                gpu["gpu_name"] = str(torch.cuda.get_device_name(0))
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
                    total_mib = int(total_bytes / (1024**2))
                    free_mib = int(free_bytes / (1024**2))
                    gpu["gpu_memory_total_mib"] = total_mib
                    gpu["gpu_memory_free_mib"] = free_mib
                    gpu["gpu_free_memory_ready"] = free_mib >= min_free_mib
                except Exception:
                    props = torch.cuda.get_device_properties(0)
                    gpu["gpu_memory_total_mib"] = int(props.total_memory / (1024**2))
                    gpu["gpu_free_memory_ready"] = True
        except Exception as exc:
            gpu["gpu_error"] = f"{type(exc).__name__}: {exc}"
        try:
            importlib.import_module("bitsandbytes")
            gpu["bitsandbytes_available"] = True
        except Exception:
            gpu["bitsandbytes_available"] = False
        return gpu

    def _gate_payload(self, gate: GateEvaluation, model_confidence: str) -> dict:
        return {
            "stage": gate.stage,
            "trigger_codes": list(gate.gate_trigger_codes),
            "retrieval_support_status": gate.retrieval_support_status,
            "model_confidence": model_confidence,
            "image_usable": gate.image_usable,
            "escalation_required": gate.escalation_required,
            "top_score": gate.top_score,
            "evidence_conflict_flag": gate.evidence_conflict_flag,
            "provider_mode": self.provider_mode,
        }

    def _disclaimer(self) -> str:
        return REAL_GPU_DISCLAIMER if self.provider_mode == "real_gpu_gemma4" else DISCLAIMER

    def _build_abstained_response(
        self,
        *,
        request_id: str,
        timing_ms: int,
        gate: GateEvaluation,
        needed_next_inputs: List[str],
    ) -> ConsultationResponse:
        unique_needed = list(dict.fromkeys(item for item in needed_next_inputs if item))
        answer = "\n".join(
            [
                "## Insufficient Information",
                "The app is abstaining because the available information is not safe enough for a ranked differential.",
                "",
                "### What is still needed",
                *[f"- {item}" for item in unique_needed],
                "",
                ESCALATION_TEXT,
            ]
        )
        return ConsultationResponse(
            request_id=request_id,
            model_name=self.generator.model_name,
            decision_state="abstained",
            top_diagnoses=[],
            answer_markdown=answer,
            evidence=[],
            disclaimer=self._disclaimer(),
            timing_ms=timing_ms,
            uncertainty_gate=self._gate_payload(gate, model_confidence="low"),
            needed_next_inputs=unique_needed,
            safe_to_show_ranked_differential=False,
            runtime_metadata={"provider_mode": self.provider_mode},
        )

    def _build_grounded_response(
        self,
        *,
        request_id: str,
        timing_ms: int,
        gate: GateEvaluation,
        model_output: GeneratorOutput,
        contexts: List,
    ) -> ConsultationResponse:
        diagnoses = [
            DiagnosisRank(
                rank=index,
                label=label,
                confidence_band=model_output.confidence,
                rationale=model_output.reasoning,
            )
            for index, label in enumerate(model_output.ranked_differential[:3], start=1)
        ]
        answer = model_output.answer_markdown or "\n".join(
            [
                "## Grounded Differential",
                f"Rank 1: **{model_output.diagnosis}**",
                "",
                model_output.reasoning,
                "",
                "### Supporting evidence",
                *[f"- {item}" for item in model_output.key_evidence],
            ]
        )
        return ConsultationResponse(
            request_id=request_id,
            model_name=self.generator.model_name,
            decision_state="rag_supported",
            top_diagnoses=diagnoses,
            answer_markdown=answer,
            evidence=[context.to_response_dict() for context in contexts[:3]],
            disclaimer=self._disclaimer(),
            timing_ms=timing_ms,
            uncertainty_gate=self._gate_payload(gate, model_confidence=model_output.confidence),
            needed_next_inputs=list(model_output.needed_next_inputs),
            safe_to_show_ranked_differential=True,
            runtime_metadata={
                "provider_mode": self.provider_mode,
                **dict(model_output.runtime_metadata),
            },
        )

    def _build_parametric_response(
        self,
        *,
        request_id: str,
        timing_ms: int,
        gate: GateEvaluation,
        model_output: GeneratorOutput,
    ) -> ConsultationResponse:
        diagnoses = [
            DiagnosisRank(
                rank=index,
                label=label,
                confidence_band=model_output.confidence,
                rationale="Model-only fallback. Not evidence-grounded.",
            )
            for index, label in enumerate(model_output.ranked_differential[:3], start=1)
        ]
        answer = model_output.answer_markdown or "\n".join(
            [
                "## Provisional Differential",
                "This result comes from the constrained parametric fallback path because retrieval support was not strong enough.",
                "",
                f"Most likely provisional diagnosis: **{model_output.diagnosis}**",
                "",
                model_output.reasoning,
                "",
                "### Before treating this as likely",
                *[f"- {item}" for item in model_output.needed_next_inputs],
                "",
                ESCALATION_TEXT,
            ]
        )
        return ConsultationResponse(
            request_id=request_id,
            model_name=self.generator.model_name,
            decision_state="provisional_parametric",
            top_diagnoses=diagnoses,
            answer_markdown=answer,
            evidence=[],
            disclaimer=self._disclaimer(),
            timing_ms=timing_ms,
            uncertainty_gate=self._gate_payload(gate, model_confidence=model_output.confidence),
            needed_next_inputs=list(model_output.needed_next_inputs),
            safe_to_show_ranked_differential=True,
            runtime_metadata={
                "provider_mode": self.provider_mode,
                **dict(model_output.runtime_metadata),
            },
        )
