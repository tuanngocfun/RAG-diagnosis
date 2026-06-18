"""
MedGemma Answer Generator.

Local HuggingFace generation using google/medgemma-4b-it.
"""

import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from packaging import version

from configs.prompt_mode import PromptMode, build_rag_prompt as _build_rag_prompt
from pipeline.diagnosis_output_parser import analyze_answer_format

HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")

_CONFIRMATORY_QUERY_PATTERNS = [
    r"amastigot",
    r"leishman[- ]donovan",
    r"bone marrow",
    r"biopsy",
    r"histopath",
    r"pcr.{0,20}positive",
    r"smear.{0,20}positive",
    r"rk39.{0,20}positive",
]
_CONTEXT_CONFIRMATORY_PATTERNS = [
    r"amastigot",
    r"leishman[- ]donovan",
    r"bone marrow",
    r"biopsy",
    r"histopath",
    r"pcr.{0,20}positive",
    r"smear.{0,20}positive",
    r"parasite.{0,20}(seen|identified)",
    r"rk39.{0,20}positive",
    r"species.{0,30}(identified|confirmed)",
]

# MedGemma multimodal generation becomes unstable once the total sequence
# (prompt + generated answer) grows beyond 1024 tokens.
MAX_TOTAL_TOKENS = int(os.environ.get("MEDGEMMA_MAX_TOTAL_TOKENS", "1024"))
MAX_VISION_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_VISION_NEW_TOKENS", "256"))
MAX_TEXT_NEW_TOKENS = int(os.environ.get("MEDGEMMA_MAX_TEXT_NEW_TOKENS", "256"))
MAX_QUERY_IMAGES = int(os.environ.get("MEDGEMMA_MAX_QUERY_IMAGES", "1"))
MAX_VISION_PROMPT_CHARS = int(os.environ.get("MEDGEMMA_MAX_VISION_PROMPT_CHARS", "2000"))
MAX_VISION_CONTEXT_CHARS = int(os.environ.get("MEDGEMMA_MAX_VISION_CONTEXT_CHARS", "500"))
PROMPT_SUFFIX_CHARS = int(os.environ.get("MEDGEMMA_PROMPT_SUFFIX_CHARS", "700"))
MIN_PROMPT_CHARS = int(os.environ.get("MEDGEMMA_MIN_PROMPT_CHARS", "700"))


def _truncate_preserving_suffix(text: str, max_chars: int, suffix_chars: int = PROMPT_SUFFIX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 64:
        return text[:max_chars]
    suffix_chars = max(0, min(suffix_chars, max_chars // 2))
    separator = "\n...\n"
    head_chars = max_chars - suffix_chars - len(separator)
    if head_chars <= 0:
        return text[-max_chars:]
    return text[:head_chars] + separator + text[-suffix_chars:]



def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    query_images: List[str] = None,
    context_images: List[str] = None,
    prompt_mode: PromptMode = PromptMode.BALANCED,
) -> str:
    return _build_rag_prompt(
        query=query,
        contexts=contexts,
        mode=prompt_mode,
        query_images=query_images,
        context_images=context_images,
        max_chars_per_context=MAX_VISION_CONTEXT_CHARS,
        include_context_images=True,
        is_text_only_model=False,
    )


class MedGemmaGenerator:
    """Generate answers using local MedGemma-4b-it."""

    MODEL_ID = "google/medgemma-4b-it"
    MIN_TRANSFORMERS = "4.50.0"

    @staticmethod
    def _resolve_device_map(device: str):
        """
        Resolve HF device_map from environment with safe defaults.

        MEDGEMMA_DEVICE_MAP:
          - auto (default)
          - gpu_only / cuda_only / single_gpu (force all modules to GPU 0)
          - cpu (force CPU)
        """
        raw = os.environ.get("MEDGEMMA_DEVICE_MAP", "auto").strip().lower()
        if raw in {"gpu_only", "cuda_only", "single_gpu"}:
            return {"": 0}
        if raw == "cpu":
            return {"": "cpu"}
        if device == "cpu":
            return {"": "cpu"}
        return "auto"

    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_dir: str = None,
        use_vision: bool = True,
        prompt_mode: PromptMode = PromptMode.BALANCED,
    ):
        try:
            import transformers
            from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer, Gemma3ForConditionalGeneration
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")

        if version.parse(transformers.__version__) < version.parse(self.MIN_TRANSFORMERS):
            raise RuntimeError(
                f"MedGemma requires transformers>={self.MIN_TRANSFORMERS}. "
                f"Detected transformers=={transformers.__version__}."
            )

        self._torch = torch
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_path or self.MODEL_ID
        self.cache_dir = cache_dir or HF_CACHE
        self.use_vision = use_vision
        self.prompt_mode = prompt_mode
        self.device_map = self._resolve_device_map(device)
        self._PIL = None
        self.last_generation_metadata: Dict[str, object] = {}

        self.decoding_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "prompt_mode": str(prompt_mode),
        }

        print(f"Loading MedGemma from {self.model_name}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Device: {self.device}")
        print(f"  torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'ALL')}")
        print(f"  Requested HF device_map: {self.device_map}")
        print(f"  Prompt mode: {self.prompt_mode}")

        if self.use_vision:
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )

        try:
            if self.use_vision:
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype="auto",
                    device_map=self.device_map,
                    trust_remote_code=True,
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    torch_dtype="auto",
                    device_map=self.device_map,
                    trust_remote_code=True,
                )
        except ValueError as e:
            if "model type `gemma3`" in str(e).lower() or "model type 'gemma3'" in str(e).lower():
                raise RuntimeError(
                    "MedGemma load failed because transformers does not support gemma3 architecture. "
                    "Upgrade environment: pip install -U 'transformers>=4.50.0' 'tokenizers>=0.21.0'"
                ) from e
            raise

        hf_device_map = getattr(self.model, "hf_device_map", None)
        if isinstance(hf_device_map, dict) and hf_device_map:
            placements = sorted({str(v) for v in hf_device_map.values()})
            print(f"  Actual HF device placement: {placements}")
            if any(p.startswith("cpu") or p == "disk" for p in placements):
                warning = (
                    "Warning: MedGemma has CPU/disk offload in hf_device_map, "
                    "which can make generation much slower. "
                    "Set MEDGEMMA_DEVICE_MAP=gpu_only to force full GPU placement."
                )
                print(warning)
                if os.environ.get("MEDGEMMA_FAIL_ON_CPU_OFFLOAD", "0") == "1":
                    raise RuntimeError(warning)

        print(f"✓ MedGemma loaded: {self.model_name} (vision={self.use_vision})")

    def _load_images(self, image_paths: List[str]) -> List:
        if self._PIL is None:
            from PIL import Image

            self._PIL = Image

        images = []
        for p in (image_paths or [])[:MAX_QUERY_IMAGES]:
            fp = Path(p)
            if not fp.exists():
                continue
            try:
                img = self._PIL.open(fp).convert("RGB")
                img = img.resize((896, 896), self._PIL.Resampling.LANCZOS)
                images.append(img)
            except Exception as e:
                print(f"Warning: Could not load image {p}: {e}")
        return images

    def _vision_input_token_budget(self) -> int:
        reserved = min(self.decoding_params["max_new_tokens"], MAX_VISION_NEW_TOKENS)
        return max(256, MAX_TOTAL_TOKENS - reserved - 1)

    def _text_input_token_budget(self) -> int:
        reserved = min(self.decoding_params["max_new_tokens"], MAX_TEXT_NEW_TOKENS)
        return max(256, MAX_TOTAL_TOKENS - reserved - 1)

    def _prepare_vision_inputs(self, prompt: str, query_images: Optional[List[str]]) -> Tuple[str, Dict]:
        images = self._load_images(query_images or [])
        if not images:
            raise ValueError("No valid query images available for MedGemma vision input")

        char_budget = min(len(prompt), MAX_VISION_PROMPT_CHARS)
        while True:
            prompt_for_model = _truncate_preserving_suffix(prompt, char_budget, PROMPT_SUFFIX_CHARS)
            content = [{"type": "image", "image": img} for img in images]
            content.append({"type": "text", "text": prompt_for_model})
            messages = [{"role": "user", "content": content}]
            inputs = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self.model.device)
            input_len = inputs["input_ids"].shape[1]
            if input_len <= self._vision_input_token_budget() or char_budget <= MIN_PROMPT_CHARS:
                return prompt_for_model, inputs

            scale = self._vision_input_token_budget() / max(input_len, 1)
            next_budget = max(
                MIN_PROMPT_CHARS,
                min(char_budget - 100, int(char_budget * scale) - 32),
            )
            if next_budget >= char_budget:
                next_budget = max(MIN_PROMPT_CHARS, char_budget - 100)
            char_budget = next_budget

    def _prepare_text_inputs(self, prompt: str) -> Tuple[str, Dict]:
        prompt_for_model = _truncate_preserving_suffix(prompt, MAX_VISION_PROMPT_CHARS, PROMPT_SUFFIX_CHARS)
        inputs = self.tokenizer(
            prompt_for_model,
            return_tensors="pt",
            truncation=True,
            max_length=self._text_input_token_budget(),
        ).to(self.device)
        return prompt_for_model, inputs

    def _generate_with_inputs(self, inputs: Dict, used_vision: bool) -> str:
        input_len = inputs["input_ids"].shape[1]
        available = max(1, MAX_TOTAL_TOKENS - input_len - 1)
        capped_new_tokens = min(
            self.decoding_params["max_new_tokens"],
            MAX_VISION_NEW_TOKENS if used_vision else MAX_TEXT_NEW_TOKENS,
            available,
        )

        with self._torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                temperature=self.decoding_params["temperature"],
                max_new_tokens=max(1, capped_new_tokens),
                do_sample=self.decoding_params["do_sample"],
                pad_token_id=self.tokenizer.eos_token_id,
            )

        if used_vision:
            answer = self.processor.decode(outputs[0][input_len:], skip_special_tokens=True)
        else:
            answer = self.tokenizer.decode(outputs[0][input_len:], skip_special_tokens=True)
        return answer.strip()

    def _generate_from_prompt(self, prompt: str, query_images: Optional[List[str]]) -> str:
        if self.use_vision and query_images:
            try:
                _, inputs = self._prepare_vision_inputs(prompt, query_images)
                return self._generate_with_inputs(inputs, used_vision=True)
            except Exception as e:
                print(f"Warning: MedGemma vision path failed, retrying text-only prompt: {type(e).__name__}: {e}")

        _, inputs = self._prepare_text_inputs(prompt)
        return self._generate_with_inputs(inputs, used_vision=False)

    def _contexts_have_confirmatory_signal(self, contexts: List[Dict]) -> bool:
        for ctx in contexts or []:
            lowered = (ctx.get("text", "") or "").lower()
            if any(re.search(pattern, lowered) for pattern in _CONTEXT_CONFIRMATORY_PATTERNS):
                return True
        return False

    def _answer_ranks_nonleish(self, answer: str) -> bool:
        analysis = analyze_answer_format(answer)
        if analysis.rank1_diagnosis_type == "Non-Leishmaniasis":
            return True
        lowered_rank1 = (analysis.rank1_diagnosis_text or "").lower()
        lowered_answer = (answer or "").lower()
        return "non-leish" in lowered_rank1 or "non-leish" in lowered_answer

    def _needs_reconciliation_retry(self, query: str, answer: str, contexts: List[Dict]) -> bool:
        if not self._answer_ranks_nonleish(answer):
            return False
        lowered_query = (query or "").lower()
        query_confirmatory = any(re.search(pattern, lowered_query) for pattern in _CONFIRMATORY_QUERY_PATTERNS)
        return query_confirmatory or self._contexts_have_confirmatory_signal(contexts)

    def _format_retry_prompt(self, prompt: str) -> str:
        return (
            prompt
            + "\n\n## OUTPUT FORMAT REQUIREMENT\n"
            + "Return exactly this schema with no extra preamble and no markdown deviations:\n"
            + "**Rank 1 (Most Likely):** [Diagnosis name]\n"
            + "**Rank 1 Diagnosis Type:** [CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other]\n"
            + "**Rank 1 Reasoning:** [1-3 concise sentences grounded in the patient and retrieved evidence]\n"
            + "**Rank 2:** [Diagnosis name]\n"
            + "**Rank 3:** [Diagnosis name]"
        )

    def _build_generation_metadata(
        self,
        contexts: List[Dict],
        answer: str,
        format_retry_count: int,
        answer_format_error: Optional[str] = None,
    ) -> Dict[str, object]:
        analysis = analyze_answer_format(answer)
        metadata = {
            "prompt_context_doc_ids": [ctx.get("doc_id") for ctx in contexts if ctx.get("doc_id")],
            "prompt_context_count": len([ctx for ctx in contexts if ctx.get("doc_id")]),
            "format_retry_count": format_retry_count,
            "answer_format_valid": analysis.answer_format_valid,
            "answer_format_error": answer_format_error or analysis.answer_format_error,
        }
        return metadata

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None,
        query_images: List[str] = None,
        context_images: List[str] = None,
        use_rag_prompt: bool = True,
    ) -> str:
        active_query_images = query_images if query_images is not None else (image_paths or [])
        active_query_images = (active_query_images or [])[:MAX_QUERY_IMAGES]
        prompt_contexts = list(contexts or [])
        prompt_context_doc_ids = [ctx.get("doc_id") for ctx in prompt_contexts if ctx.get("doc_id")]
        if use_rag_prompt:
            prompt = build_rag_prompt(
                query,
                prompt_contexts,
                query_images=active_query_images if active_query_images else None,
                context_images=context_images if context_images else None,
                prompt_mode=self.prompt_mode,
            )
        else:
            prompt = query

        format_retry_count = 0
        self.last_generation_metadata = {
            "prompt_context_doc_ids": prompt_context_doc_ids,
            "prompt_context_count": len(prompt_context_doc_ids),
            "format_retry_count": 0,
            "answer_format_valid": None,
            "answer_format_error": "",
        }

        try:
            active_prompt = prompt
            answer = self._generate_from_prompt(active_prompt, active_query_images)
            if use_rag_prompt and self._needs_reconciliation_retry(query, answer, prompt_contexts):
                active_prompt = (
                    prompt
                    + "\n\n## RECONCILIATION CHECK\n"
                    + "The selected retrieved cases contain potentially definitive leishmaniasis evidence "
                    + "(parasite visualization, biopsy, marrow finding, histopathology, or a positive confirmatory test). "
                    + "Do not rank Non-Leishmaniasis #1 unless you explicitly explain why that evidence is unreliable "
                    + "or belongs only to a retrieved reference case rather than the patient."
                )
                answer = self._generate_from_prompt(active_prompt, active_query_images)

            analysis = analyze_answer_format(answer)
            if not analysis.answer_format_valid:
                format_retry_count += 1
                answer = self._generate_from_prompt(self._format_retry_prompt(active_prompt), active_query_images)

            self.last_generation_metadata = self._build_generation_metadata(
                prompt_contexts,
                answer,
                format_retry_count=format_retry_count,
            )
            return answer
        except Exception as e:
            error_text = f"generation_exception:{type(e).__name__}"
            self.last_generation_metadata = {
                "prompt_context_doc_ids": prompt_context_doc_ids,
                "prompt_context_count": len(prompt_context_doc_ids),
                "format_retry_count": format_retry_count,
                "answer_format_valid": False,
                "answer_format_error": error_text,
            }
            return f"[Generation Error: {type(e).__name__}: {e}]"

    def generate_batch(self, samples: List[Dict], progress: bool = True) -> List[Dict]:
        results = []
        for i, sample in enumerate(samples):
            answer = self.generate(
                sample["query"],
                sample.get("contexts", []),
                image_paths=sample.get("query_images", []),
                query_images=sample.get("query_images", []),
                context_images=sample.get("context_images", []),
                use_rag_prompt=sample.get("use_rag_prompt", True),
            )
            results.append(
                {
                    **sample,
                    "answer": answer,
                    "model_name": self.model_name,
                    "decoding_params": self.decoding_params,
                    "generation_metadata": dict(self.last_generation_metadata or {}),
                }
            )
            if progress and (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{len(samples)}")
        return results


def get_medgemma_generator() -> MedGemmaGenerator:
    return MedGemmaGenerator()


if __name__ == "__main__":
    print("Testing MedGemma Generator...")
    print(f"HF Cache: {HF_CACHE}")
    try:
        gen = MedGemmaGenerator()
        print(f"✓ Loaded: {gen.model_name}")
        test_answer = gen.generate(
            "What are symptoms of visceral leishmaniasis?",
            [{"doc_id": "PMC123", "text": "Fever, weight loss, splenomegaly..."}],
        )
        print(f"✓ Generated: {test_answer[:100]}...")
    except Exception as e:
        print(f"⚠ Could not load MedGemma: {e}")
        print("  (Model may need to be downloaded first)")
