"""
Gemma 4 Answer Generator.

Local HuggingFace generation using google/gemma-4-E4B-it.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, List, Optional

from configs.prompt_mode import PromptMode, build_rag_prompt as _build_rag_prompt

HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")
GEMMA4_MODEL_ID = "google/gemma-4-E4B-it"
MAX_QUERY_IMAGES = int(os.environ.get("GEMMA4_MAX_QUERY_IMAGES", "5"))
IMAGE_SIZE = int(os.environ.get("GEMMA4_IMAGE_SIZE", "896"))
ORDERING_MODES = {"image_first", "text_first", "interleaved"}


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    query_images: Optional[List[str]] = None,
    context_images: Optional[List[str]] = None,
    prompt_mode: PromptMode = PromptMode.BALANCED,
) -> str:
    return _build_rag_prompt(
        query=query,
        contexts=contexts,
        mode=prompt_mode,
        query_images=query_images,
        context_images=context_images,
        max_chars_per_context=1800,
        include_context_images=True,
        is_text_only_model=False,
    )


class Gemma4Generator:
    """Generate answers using local Gemma 4 E4B."""

    MODEL_ID = GEMMA4_MODEL_ID

    def __init__(
        self,
        model_path: Optional[str] = None,
        device: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_dir: Optional[str] = None,
        use_vision: bool = True,
        random_seed: Optional[int] = 42,
        prompt_mode: PromptMode = PromptMode.BALANCED,
        ordering_mode: str = "image_first",
        use_context_image_tensors: bool = False,
        support_image_tensor_budget: int = 0,
    ):
        try:
            from transformers import AutoModelForImageTextToText, AutoProcessor
            import torch
        except ImportError as exc:
            raise ImportError("pip install transformers torch") from exc

        self._torch = torch
        self._PIL = None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_path or self.MODEL_ID
        self.cache_dir = cache_dir or HF_CACHE
        self.use_vision = use_vision
        self.prompt_mode = prompt_mode
        self.random_seed = random_seed
        self.ordering_mode = ordering_mode if ordering_mode in ORDERING_MODES else "image_first"
        self.use_context_image_tensors = bool(use_context_image_tensors)
        self.support_image_tensor_budget = max(0, int(support_image_tensor_budget or 0))
        self.last_generation_metadata: Dict[str, object] = {}

        if random_seed is not None:
            torch.manual_seed(random_seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(random_seed)

        self.decoding_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "random_seed": random_seed,
        }

        vram_gb = 0.0
        if torch.cuda.is_available():
            vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)

        force_4bit = os.environ.get("GEMMA4_FORCE_4BIT", "0") == "1"
        disable_4bit = os.environ.get("GEMMA4_DISABLE_4BIT", "0") == "1"
        use_quantization = bool(torch.cuda.is_available() and not disable_4bit and (force_4bit or vram_gb < 28.0))

        print(f"Loading Gemma 4 from {self.model_name}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Device: {self.device}")
        print(f"  Vision: {self.use_vision}")
        print(f"  Ordering mode: {self.ordering_mode}")
        print(f"  Context image tensors: {self.use_context_image_tensors} (budget={self.support_image_tensor_budget})")
        print(f"  GPU VRAM: {vram_gb:.1f}GB")
        print(f"  Quantization: {'4-bit' if use_quantization else 'None (HF default)'}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True,
        )
        self.tokenizer = getattr(self.processor, "tokenizer", None)

        model_kwargs = {
            "cache_dir": self.cache_dir,
            "torch_dtype": "auto",
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if use_quantization:
            try:
                from transformers import BitsAndBytesConfig

                model_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4",
                )
            except ImportError:
                print("  Warning: bitsandbytes not installed; loading without 4-bit quantization")

        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_name,
            **model_kwargs,
        )
        print(f"✓ Gemma 4 loaded: {self.model_name} (vision={self.use_vision})")

    def _load_images(self, image_paths: Optional[List[str]], max_images: int = MAX_QUERY_IMAGES) -> List:
        if self._PIL is None:
            from PIL import Image

            self._PIL = Image

        images = []
        for path in (image_paths or [])[:max(0, int(max_images))]:
            p = Path(path)
            if not p.exists():
                continue
            try:
                img = self._PIL.open(p).convert("RGB")
                img = img.resize((IMAGE_SIZE, IMAGE_SIZE), self._PIL.Resampling.LANCZOS)
                images.append(img)
            except Exception as exc:
                print(f"Warning: Could not load image {path}: {exc}")
        return images

    @staticmethod
    def _assemble_multimodal_content(
        prompt: str,
        query_images: List,
        support_images: List,
        ordering_mode: str,
    ) -> List[Dict]:
        content: List[Dict] = []

        def _append_images(images: List) -> None:
            for image in images:
                content.append({"type": "image", "image": image})

        if ordering_mode == "text_first":
            content.append({"type": "text", "text": prompt})
            _append_images(query_images)
            _append_images(support_images)
            return content

        if ordering_mode == "interleaved":
            _append_images(query_images)
            content.append({"type": "text", "text": prompt})
            _append_images(support_images)
            return content

        _append_images(query_images)
        _append_images(support_images)
        content.append({"type": "text", "text": prompt})
        return content

    def _prepare_inputs(
        self,
        prompt: str,
        query_images: Optional[List[str]],
        context_images: Optional[List[str]] = None,
    ) -> tuple[Dict, int, int]:
        query_tensor_images = (
            self._load_images(query_images, max_images=MAX_QUERY_IMAGES)
            if self.use_vision and query_images
            else []
        )
        support_tensor_images: List = []
        if self.use_vision and self.use_context_image_tensors and context_images and self.support_image_tensor_budget > 0:
            support_tensor_images = self._load_images(
                context_images,
                max_images=self.support_image_tensor_budget,
            )
        content = self._assemble_multimodal_content(
            prompt=prompt,
            query_images=query_tensor_images,
            support_images=support_tensor_images,
            ordering_mode=self.ordering_mode,
        )
        messages = [{"role": "user", "content": content}]

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )
        model_device = getattr(self.model, "device", None)
        return inputs.to(model_device or self.device), len(query_tensor_images), len(support_tensor_images)

    def _decode(self, tokens) -> str:
        if self.tokenizer is not None:
            return self.tokenizer.decode(tokens, skip_special_tokens=True)
        return self.processor.decode(tokens, skip_special_tokens=True)

    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: Optional[List[str]] = None,
        query_images: Optional[List[str]] = None,
        context_images: Optional[List[str]] = None,
        use_rag_prompt: bool = True,
    ) -> str:
        active_query_images = query_images if query_images is not None else (image_paths or [])
        prompt_context_doc_ids = [ctx.get("doc_id") for ctx in (contexts or []) if ctx.get("doc_id")]
        prompt = (
            build_rag_prompt(
                query,
                contexts,
                query_images=active_query_images if active_query_images else None,
                context_images=context_images if context_images else None,
                prompt_mode=self.prompt_mode,
            )
            if use_rag_prompt
            else query
        )
        self.last_generation_metadata = {
            "prompt_context_doc_ids": prompt_context_doc_ids,
            "prompt_context_count": len(prompt_context_doc_ids),
            "format_retry_count": 0,
            "answer_format_valid": None,
            "answer_format_error": "",
            "ordering_mode": self.ordering_mode,
            "use_context_image_tensors": self.use_context_image_tensors,
            "support_image_tensor_budget": self.support_image_tensor_budget,
            "query_image_tensor_attempt_count": min(len(active_query_images or []), MAX_QUERY_IMAGES)
            if self.use_vision
            else 0,
            "support_image_tensor_attempt_count": (
                min(len(context_images or []), self.support_image_tensor_budget)
                if self.use_vision and self.use_context_image_tensors and self.support_image_tensor_budget > 0
                else 0
            ),
            "query_image_tensor_count": 0,
            "support_image_tensor_count": 0,
            "image_tensor_fallback_used": False,
            "image_tensor_fallback_reason": "",
        }

        try:
            inputs, query_image_tensor_count, support_image_tensor_count = self._prepare_inputs(
                prompt,
                active_query_images,
                context_images=context_images,
            )
            generation_kwargs = {
                "temperature": self.decoding_params["temperature"],
                "max_new_tokens": self.decoding_params["max_new_tokens"],
                "do_sample": self.decoding_params["do_sample"],
            }
            pad_token_id = getattr(self.tokenizer, "eos_token_id", None)
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                    pad_token_id=pad_token_id,
                )
            input_len = inputs["input_ids"].shape[1]
            answer = self._decode(outputs[0][input_len:]).strip()
            self.last_generation_metadata = {
                **self.last_generation_metadata,
                "answer_format_valid": True,
                "answer_format_error": "",
                "query_image_tensor_count": query_image_tensor_count,
                "support_image_tensor_count": support_image_tensor_count,
                "image_tensor_fallback_used": False,
                "image_tensor_fallback_reason": "",
            }
            return answer
        except Exception as exc:
            err_msg = str(exc).lower()
            is_oom = "outofmemory" in err_msg or "out of memory" in err_msg or "cuda oom" in err_msg
            if is_oom and (active_query_images or context_images):
                try:
                    if self._torch.cuda.is_available():
                        self._torch.cuda.empty_cache()
                    inputs, _, _ = self._prepare_inputs(prompt, None, context_images=None)
                    with self._torch.no_grad():
                        outputs = self.model.generate(
                            **inputs,
                            temperature=self.decoding_params["temperature"],
                            max_new_tokens=self.decoding_params["max_new_tokens"],
                            do_sample=self.decoding_params["do_sample"],
                            pad_token_id=getattr(self.tokenizer, "eos_token_id", None),
                        )
                    input_len = inputs["input_ids"].shape[1]
                    answer = self._decode(outputs[0][input_len:]).strip()
                    self.last_generation_metadata = {
                        **self.last_generation_metadata,
                        "answer_format_valid": True,
                        "answer_format_error": "",
                        "query_image_tensor_count": 0,
                        "support_image_tensor_count": 0,
                        "image_tensor_fallback_used": True,
                        "image_tensor_fallback_reason": "oom_retry_without_images",
                    }
                    return answer
                except Exception as fallback_exc:
                    self.last_generation_metadata = {
                        **self.last_generation_metadata,
                        "answer_format_valid": False,
                        "answer_format_error": f"generation_exception:{type(fallback_exc).__name__}",
                        "image_tensor_fallback_used": True,
                        "image_tensor_fallback_reason": "oom_retry_without_images_failed",
                    }
                    return f"[Generation Error: {type(fallback_exc).__name__}: {fallback_exc}]"
            self.last_generation_metadata = {
                **self.last_generation_metadata,
                "answer_format_valid": False,
                "answer_format_error": f"generation_exception:{type(exc).__name__}",
                "image_tensor_fallback_used": False,
                "image_tensor_fallback_reason": "",
            }
            return f"[Generation Error: {type(exc).__name__}: {exc}]"

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
