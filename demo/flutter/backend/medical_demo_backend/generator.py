"""Generator adapters for deterministic and real-GPU demo modes."""

from __future__ import annotations

import os
import re
import sys
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .types import GeneratorOutput, RetrievedEvidence


class GeneratorClient(Protocol):
    """Minimal generator interface."""

    @property
    def model_name(self) -> str:  # pragma: no cover - protocol definition
        ...

    def generate(
        self,
        patient_text: str,
        contexts: List[RetrievedEvidence],
        use_rag: bool,
        image_bytes: Optional[bytes] = None,
        image_filename: str = "",
    ) -> GeneratorOutput:
        ...  # pragma: no cover - protocol definition


class DemoGemma4Generator:
    """Safe deterministic stand-in for the planned Gemma 4 adapter."""

    def __init__(self) -> None:
        self._model_name = os.getenv("MEDICAL_DEMO_MODEL_NAME", "google/gemma-4-E4B-it")

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        patient_text: str,
        contexts: List[RetrievedEvidence],
        use_rag: bool,
        image_bytes: Optional[bytes] = None,
        image_filename: str = "",
    ) -> GeneratorOutput:
        lowered = patient_text.lower()
        if use_rag and contexts:
            return self._generate_grounded(lowered, contexts)
        return self._generate_parametric(lowered)

    def _generate_grounded(self, lowered: str, contexts: List[RetrievedEvidence]) -> GeneratorOutput:
        label_counts = Counter(context.diagnosis_label for context in contexts)
        diagnosis, _ = label_counts.most_common(1)[0]
        has_confirmatory = any(context.confirmatory for context in contexts)
        confidence = "high" if has_confirmatory and len(contexts) >= 2 else "medium"
        ranked = [diagnosis]
        if diagnosis != "Cutaneous leishmaniasis":
            ranked.append("Cutaneous leishmaniasis")
        if diagnosis != "Visceral leishmaniasis":
            ranked.append("Visceral leishmaniasis")
        while len(ranked) < 3:
            ranked.append("Mucocutaneous leishmaniasis")
        reasoning = (
            "Retrieved evidence overlaps the symptom pattern and supports "
            f"{diagnosis.lower()}. "
            "The response is grounded in the retrieved leishmaniasis reference snippets."
        )
        key_evidence = [context.title for context in contexts[:3]]
        needed = []
        insufficient_reason = None
        if confidence == "medium":
            needed = ["confirmatory microscopy or histopathology", "travel and exposure history"]
        return GeneratorOutput(
            diagnosis=diagnosis,
            confidence=confidence,
            ranked_differential=ranked[:3],
            reasoning=reasoning,
            insufficient_reason=insufficient_reason,
            needed_next_inputs=needed,
            key_evidence=key_evidence,
        )

    def _generate_parametric(self, lowered: str) -> GeneratorOutput:
        if any(term in lowered for term in ("splenomegaly", "pancytopenia", "hepatosplenomegaly", "prolonged fever")):
            diagnosis = "Visceral leishmaniasis"
            confidence = "medium"
            needed = ["CBC and organomegaly assessment", "confirmatory parasitology or molecular test"]
        elif any(term in lowered for term in ("nasal", "mucosal", "oral ulcer", "palate", "septum")):
            diagnosis = "Mucocutaneous leishmaniasis"
            confidence = "medium"
            needed = ["ENT or mucosal examination", "confirmatory tissue sampling"]
        elif any(term in lowered for term in ("ulcer", "plaque", "papule", "nodule", "skin lesion", "lesion")):
            diagnosis = "Cutaneous leishmaniasis"
            confidence = "medium"
            needed = ["lesion image or dermatology exam", "confirmatory smear, biopsy, or PCR"]
        else:
            diagnosis = "INSUFFICIENT"
            confidence = "low"
            needed = ["clear symptom history", "travel or exposure history", "lesion description or image"]

        if diagnosis == "INSUFFICIENT":
            return GeneratorOutput(
                diagnosis=diagnosis,
                confidence=confidence,
                ranked_differential=[],
                reasoning=(
                    "The available description is too limited for a safe model-only differential. "
                    "The fallback path must abstain rather than invent support."
                ),
                insufficient_reason="Model-only reasoning remained low confidence.",
                needed_next_inputs=needed,
                key_evidence=[],
            )

        ranked = [diagnosis]
        if diagnosis != "Cutaneous leishmaniasis":
            ranked.append("Cutaneous leishmaniasis")
        if diagnosis != "Visceral leishmaniasis":
            ranked.append("Visceral leishmaniasis")
        while len(ranked) < 3:
            ranked.append("Mucocutaneous leishmaniasis")
        return GeneratorOutput(
            diagnosis=diagnosis,
            confidence=confidence,
            ranked_differential=ranked[:3],
            reasoning=(
                "This is a constrained model-only fallback based on the symptom pattern. "
                "It does not claim retrieved evidence support."
            ),
            insufficient_reason=None,
            needed_next_inputs=needed,
            key_evidence=[],
        )


class RealGemma4Generator:
    """Thin UI-facing wrapper around the thesis Gemma 4 generator."""

    provider_mode = "real_gpu_gemma4"

    def __init__(self) -> None:
        self._model_name = os.getenv("MEDICAL_DEMO_MODEL_NAME", "google/gemma-4-E4B-it")
        self._project_root = Path(
            os.getenv("MEDICAL_DEMO_PROJECT_ROOT", "/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000")
        )
        self._legacy_root = Path(os.getenv("LEGACY_ROOT", "/home/ngocnt/Leishmaniasis_v3"))
        self._codes_dir = self._project_root / "codes"
        self._cache_dir = os.getenv("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")
        self._image_size = os.getenv("GEMMA4_IMAGE_SIZE", os.getenv("MEDICAL_DEMO_GEMMA4_IMAGE_SIZE", "768"))
        self._max_query_images = os.getenv(
            "GEMMA4_MAX_QUERY_IMAGES",
            os.getenv("MEDICAL_DEMO_GEMMA4_MAX_QUERY_IMAGES", "1"),
        )
        self._temperature = float(os.getenv("MEDICAL_DEMO_GEMMA4_TEMPERATURE", "0.3"))
        self._max_tokens = int(os.getenv("MEDICAL_DEMO_GEMMA4_MAX_TOKENS", "1024"))
        seed_value = os.getenv("MEDICAL_DEMO_GEMMA4_RANDOM_SEED", "42").strip()
        self._random_seed = int(seed_value) if seed_value else None
        self._generator = None
        self._prompt_mode = None
        self._torch = None
        self._lock = threading.Lock()
        self._load_seconds: Optional[float] = None
        self._runtime_metadata: Dict[str, Any] = {}

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        patient_text: str,
        contexts: List[RetrievedEvidence],
        use_rag: bool,
        image_bytes: Optional[bytes] = None,
        image_filename: str = "",
    ) -> GeneratorOutput:
        self._ensure_loaded()
        image_path = self._write_temp_image(image_bytes, image_filename) if image_bytes else None
        generation_started = time.perf_counter()
        try:
            query = patient_text.strip()
            if image_path:
                query = f"{query}\n\nAttached patient image: {Path(image_path).name}"
            context_payloads = [self._context_to_pipeline_dict(context) for context in contexts]
            if self._torch is not None and self._torch.cuda.is_available():
                self._torch.cuda.reset_peak_memory_stats()
            print(
                "[real-gpu-ui] generating with "
                f"model={self._model_name} use_rag={use_rag} "
                f"contexts={len(context_payloads)} image_tensor={bool(image_path)}"
            )
            answer = self._generator.generate(
                query,
                context_payloads,
                query_images=[image_path] if image_path else [],
                use_rag_prompt=use_rag,
            )
        finally:
            if image_path:
                try:
                    Path(image_path).unlink(missing_ok=True)
                except OSError:
                    pass
        elapsed = time.perf_counter() - generation_started
        metadata = self._build_runtime_metadata(elapsed)
        ranked = self._parse_ranked_differential(answer)
        diagnosis = ranked[0] if ranked else "Gemma 4 generated differential"
        confidence = self._parse_confidence(answer)
        if answer.strip().startswith("[Generation Error:"):
            confidence = "low"
        return GeneratorOutput(
            diagnosis=diagnosis,
            confidence=confidence,
            ranked_differential=ranked[:3] or [diagnosis],
            reasoning="Parsed from the generated model answer shown below.",
            insufficient_reason=None if confidence != "low" else "Real model output was low confidence or errored.",
            needed_next_inputs=["clinician review", "confirmatory laboratory or histopathology testing"],
            key_evidence=[context.title for context in contexts[:3]],
            answer_markdown=answer,
            runtime_metadata=metadata,
        )

    def _ensure_loaded(self) -> None:
        if self._generator is not None:
            return
        with self._lock:
            if self._generator is not None:
                return
            if not self._codes_dir.exists():
                raise RuntimeError(f"Thesis code directory not found: {self._codes_dir}")
            sys.path.insert(0, str(self._codes_dir))
            os.environ.setdefault("PROJECT_ROOT", str(self._project_root))
            os.environ.setdefault("LEGACY_ROOT", str(self._legacy_root))
            os.environ.setdefault("STRUCTURED_CASES_SPLIT_DIR", str(self._project_root / "leishmaniasis_verified_v2"))
            os.environ.setdefault(
                "STRUCTURED_CASES_IMAGES_DIR",
                str(self._legacy_root / "data" / "leishmaniasis_multimodal" / "images"),
            )
            os.environ.setdefault("STRUCTURED_CASES_RUNS_DIR", str(self._project_root / "runs"))
            os.environ.setdefault("HF_HOME", "/mnt/data/hf")
            os.environ.setdefault("TRANSFORMERS_CACHE", self._cache_dir)
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("GEMMA4_FORCE_4BIT", "1")
            os.environ.setdefault("GEMMA4_IMAGE_SIZE", self._image_size)
            os.environ.setdefault("GEMMA4_MAX_QUERY_IMAGES", self._max_query_images)

            from configs.prompt_mode import PromptMode
            from pipeline.generators.gemma4 import Gemma4Generator as PipelineGemma4Generator
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("real_gpu_gemma4 mode requires CUDA; torch.cuda.is_available() is false")
            self._torch = torch
            print("[real-gpu-ui] loading Gemma 4 for Flutter UI requests")
            started = time.perf_counter()
            self._prompt_mode = PromptMode.BALANCED
            self._generator = PipelineGemma4Generator(
                model_path=self._model_name,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
                random_seed=self._random_seed,
                prompt_mode=self._prompt_mode,
                ordering_mode="image_first",
                use_context_image_tensors=False,
                support_image_tensor_budget=0,
            )
            self._load_seconds = time.perf_counter() - started
            self._runtime_metadata = dict(getattr(self._generator, "runtime_metadata", {}) or {})
            print(f"[real-gpu-ui] Gemma 4 ready for UI requests after {self._load_seconds:.1f}s")

    @staticmethod
    def _context_to_pipeline_dict(context: RetrievedEvidence) -> Dict[str, Any]:
        return {
            "doc_id": context.source_case_id or context.chunk_id,
            "chunk_id": context.chunk_id,
            "title": context.title,
            "diagnosis_label": context.diagnosis_label,
            "text": context.text,
            "score": context.score,
            "confirmatory": context.confirmatory,
            "image_paths": [],
        }

    @staticmethod
    def _write_temp_image(image_bytes: Optional[bytes], image_filename: str) -> Optional[str]:
        if not image_bytes:
            return None
        suffix = Path(image_filename or "uploaded.jpg").suffix.lower()
        if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
            suffix = ".jpg"
        with tempfile.NamedTemporaryFile(prefix="medical_demo_upload_", suffix=suffix, delete=False) as tmp:
            tmp.write(image_bytes)
            return tmp.name

    def _build_runtime_metadata(self, generation_latency_seconds: float) -> Dict[str, Any]:
        generator_metadata = dict(getattr(self._generator, "last_generation_metadata", {}) or {})
        runtime_metadata = dict(self._runtime_metadata)
        torch_metadata: Dict[str, Any] = {}
        if self._torch is not None and self._torch.cuda.is_available():
            torch_metadata = {
                "gpu_memory_allocated_mib": round(self._torch.cuda.memory_allocated() / (1024**2), 1),
                "gpu_memory_reserved_mib": round(self._torch.cuda.memory_reserved() / (1024**2), 1),
                "gpu_peak_memory_allocated_mib": round(self._torch.cuda.max_memory_allocated() / (1024**2), 1),
            }
        return {
            "provider_mode": self.provider_mode,
            "model_name": self._model_name,
            "model_load_seconds": self._load_seconds,
            "generation_latency_seconds": generation_latency_seconds,
            "temperature": generator_metadata.get("temperature", self._temperature),
            "max_new_tokens": generator_metadata.get("max_new_tokens", self._max_tokens),
            "do_sample": generator_metadata.get("do_sample", self._temperature > 0),
            "random_seed": generator_metadata.get("random_seed", self._random_seed),
            "prompt_context_count": generator_metadata.get("prompt_context_count"),
            "query_image_tensor_count": generator_metadata.get("query_image_tensor_count"),
            "image_tensor_fallback_used": generator_metadata.get("image_tensor_fallback_used", False),
            "image_tensor_fallback_reason": generator_metadata.get("image_tensor_fallback_reason", ""),
            **runtime_metadata,
            **torch_metadata,
        }

    @staticmethod
    def _parse_confidence(answer: str) -> str:
        lowered = re.sub(r"[*_`]", "", answer.lower())
        if re.search(r"(rank\s*1\s*)?confidence\s*:\s*high", lowered):
            return "high"
        if re.search(r"(rank\s*1\s*)?confidence\s*:\s*low", lowered):
            return "low"
        if "low confidence" in lowered or "confidence: low" in lowered:
            return "low"
        if "high confidence" in lowered or "confidence: high" in lowered:
            return "high"
        return "medium"

    @staticmethod
    def _parse_ranked_differential(answer: str) -> List[str]:
        ranked: List[str] = []
        rank_pattern = re.compile(
            r"^\s*(?:\*\*)?Rank\s*([1-3])(?:\s*\([^)]*\))?\s*:\s*(?:\*\*)?\s*([^\n#]+)",
            flags=re.IGNORECASE | re.MULTILINE,
        )
        for match in rank_pattern.finditer(answer):
            label = re.sub(r"\s+", " ", match.group(2)).strip(" -*:")
            label = re.sub(r"\*\*.*$", "", label).strip()
            label = re.sub(r"\s*\([^)]*\)\s*$", "", label).strip()
            if label:
                ranked.append(label)
        if not ranked:
            numbered_pattern = re.compile(r"^\s*([1-3])[\.\)]\s*([^\n]+)", flags=re.MULTILINE)
            for match in numbered_pattern.finditer(answer):
                label = re.sub(r"\s+", " ", match.group(2)).strip(" -*:")
                if label:
                    ranked.append(label)
        deduped: List[str] = []
        for label in ranked:
            if label not in deduped:
                deduped.append(label)
        return deduped[:3]
