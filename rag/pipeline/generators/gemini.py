"""
Gemini Answer Generator - New SDK (google.genai)

Uses the official google.genai SDK (successor to deprecated google-generativeai).
Per official docs: https://ai.google.dev/gemini-api/docs/text-generation

Key changes from old SDK:
- Uses genai.Client() instead of genai.configure()
- Uses client.models.generate_content() 
- Safety settings via config=types.GenerateContentConfig(safety_settings=[...])
- Default for GA models (gemini-2.5-pro) is BLOCK_NONE per official docs
"""
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple

# Import from package to ensure dotenv is loaded
from .. import GOOGLE_API_KEY

# Import centralized prompt builder
from configs.prompt_mode import build_rag_prompt as _build_rag_prompt, PromptMode


# Stable Gemini model IDs (GA June 2025)
GEMINI_MODELS = {
    "stable": "gemini-2.5-pro",      # GA June 17, 2025
    "flash": "gemini-2.5-flash",     # GA June 17, 2025
    "legacy": "gemini-2.0-flash",    # Fallback
}


def _extract_text_safely(response) -> Tuple[str, Dict]:
    """
    Defensive text extraction from Gemini response.
    
    Per GPT 5.2: Using response.text directly throws when blocked.
    Instead, check candidates and extract safely.
    
    Returns:
        Tuple of (text, metadata_dict)
    """
    meta = {"blocked": False, "finish_reason": None}
    
    # Try response.text first (works when not blocked)
    try:
        if hasattr(response, 'text') and response.text:
            return response.text, meta
    except Exception:
        pass
    
    # Check candidates for blocked content
    candidates = getattr(response, "candidates", None)
    if not candidates:
        meta["blocked"] = True
        meta["finish_reason"] = "NO_CANDIDATES"
        return "", meta
    
    cand = candidates[0]
    finish_reason = getattr(cand, "finish_reason", None)
    meta["finish_reason"] = str(finish_reason) if finish_reason else None
    
    # Check if blocked (SAFETY = blocked by safety filter)
    if finish_reason and "SAFETY" in str(finish_reason):
        meta["blocked"] = True
    
    # Try to extract text from parts
    content = getattr(cand, "content", None)
    if content:
        parts = getattr(content, "parts", None) or []
        texts = []
        for p in parts:
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        if texts:
            return "".join(texts).strip(), meta
    
    meta["blocked"] = True
    return "", meta


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    include_images: bool = True,
    query_images: List[str] = None,
    context_images: List[str] = None,
    prompt_mode: PromptMode = PromptMode.STRICT_CONTEXT
) -> str:
    """
    Build RAG prompt with diagnosis-focused output format.
    
    WRAPPER: Delegates to centralized build_rag_prompt from configs/prompt_mode.py
    Default mode is STRICT_CONTEXT (original Gemini behavior).
    
    Args:
        query: Clinical question
        contexts: Retrieved contexts
        include_images: Whether to include image references in context
        query_images: Patient image paths
        context_images: Evidence image paths from retrieved cases
        prompt_mode: Which prompt template to use (default: STRICT_CONTEXT)
    """
    return _build_rag_prompt(
        query=query,
        contexts=contexts,
        mode=prompt_mode,
        query_images=query_images,
        context_images=context_images,
        max_chars_per_context=2000,
        include_context_images=include_images,
        is_text_only_model=False
    )


class GeminiGenerator:
    """
    Generate answers using Google Gemini API (new google.genai SDK).
    
    Uses the official google.genai SDK with proper safety settings.
    """
    
    def __init__(
        self,
        model: str = None,
        include_images: bool = False,
        prompt_mode: PromptMode = PromptMode.STRICT_CONTEXT,
        temperature: float = 0.3,
        max_tokens: int = 4096,  # Increased from 1024 to avoid MAX_TOKENS truncation
        batch_poll_seconds: float = 10.0,
        batch_timeout_seconds: int = 3600,
    ):
        """
        Initialize Gemini generator with new SDK.
        
        Args:
            model: Model ID or None for stable default
            include_images: Whether to reference images in prompt
            temperature: Sampling temperature
            max_tokens: Max output tokens
        """
        try:
            from google import genai
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")
        
        # Initialize client with API key from environment
        self.client = genai.Client(api_key=GOOGLE_API_KEY)
        self._types = types
        
        # Use stable model by default
        self.model_name = model or GEMINI_MODELS["stable"]
        self.include_images = include_images
        self.prompt_mode = prompt_mode

        # Generation config
        self.generation_config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        # Safety settings - BLOCK_NONE for all categories (research use)
        # Per official docs: "Default for GA models (gemini-2.0-flash, gemini-2.5-*) is BLOCK_NONE"
        self.safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.OFF,  # OFF = no blocking
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.OFF,
            ),
        ]
        
        # Test connection
        self._test_model()
        
        # Store decoding params for logging
        self.decoding_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
            "batch_poll_seconds": batch_poll_seconds,
            "batch_timeout_seconds": batch_timeout_seconds,
            "prompt_mode": str(prompt_mode),
        }

        self.batch_poll_seconds = batch_poll_seconds
        self.batch_timeout_seconds = batch_timeout_seconds
    
    def _test_model(self):
        """Test model availability."""
        try:
            # Simple test request
            response = self.client.models.generate_content(
                model=self.model_name,
                contents="Say hello",
                config=self._types.GenerateContentConfig(
                    max_output_tokens=10
                )
            )
            print(f"✓ Using {self.model_name} with new google.genai SDK (safety=OFF)")
        except Exception as e:
            print(f"Warning: Could not test model {self.model_name}: {e}")
            # Try fallback
            for fallback in ["gemini-2.5-flash", "gemini-2.0-flash"]:
                try:
                    self.client.models.generate_content(
                        model=fallback,
                        contents="Say hello",
                        config=self._types.GenerateContentConfig(max_output_tokens=10)
                    )
                    self.model_name = fallback
                    print(f"✓ Using fallback model {fallback}")
                    return
                except:
                    continue
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None,
        query_images: List[str] = None,
        context_images: List[str] = None,
        use_rag_prompt: bool = True,
    ) -> str:
        """
        Generate answer for query given contexts with TRUE MULTIMODAL support.
        
        Sends both text contexts AND images to Gemini for generation.
        
        Args:
            query: Query text
            contexts: Retrieved contexts (from retriever, NOT qrels)
            image_paths: List of image file paths to include
            use_rag_prompt: If True, wrap query with RAG faithfulness prompt.
                           If False, use query directly (for No-RAG baseline).
        
        Returns:
            Generated answer text
        """
        active_query_images = query_images if query_images is not None else image_paths
        contents = self._build_contents(
            query=query,
            contexts=contexts,
            image_paths=active_query_images,
            query_images=active_query_images,
            context_images=context_images,
            use_rag_prompt=use_rag_prompt,
        )
        
        try:
            # Create config with safety settings
            config = self._types.GenerateContentConfig(
                temperature=self.decoding_params["temperature"],
                max_output_tokens=self.decoding_params["max_output_tokens"],
                safety_settings=self.safety_settings,
            )
            
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,  # Multimodal: images + text
                config=config,
            )
            
            # Use defensive extraction
            text, meta = _extract_text_safely(response)
            
            if text:
                return text
            
            # Blocked or empty - return fallback
            top_ids = [c.get("doc_id") for c in contexts[:5] if c.get("doc_id")]
            return (
                "Insufficient evidence in retrieved excerpts (generation returned no content). "
                f"finish_reason={meta.get('finish_reason')}. "
                + (f"Top retrieved case IDs: {', '.join(top_ids)}." if top_ids else "")
            )
            
        except Exception as e:
            return f"[Generation Error: {type(e).__name__}: {e}]"

    def _build_contents(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None,
        query_images: List[str] = None,
        context_images: List[str] = None,
        use_rag_prompt: bool = True,
    ) -> List:
        """Build multimodal contents payload shared by single and batch generation."""
        import base64
        from pathlib import Path as P

        active_query_images = query_images if query_images is not None else image_paths
        if use_rag_prompt:
            prompt = build_rag_prompt(
                query,
                contexts,
                include_images=self.include_images,
                query_images=active_query_images,
                context_images=context_images,
                prompt_mode=self.prompt_mode,
            )
        else:
            prompt = query

        contents = []
        if image_paths:
            valid_images = 0
            for img_path in image_paths[:5]:
                img_file = P(img_path)
                if not img_file.exists():
                    continue
                try:
                    with open(img_file, "rb") as f:
                        img_data = base64.b64encode(f.read()).decode("utf-8")

                    suffix = img_file.suffix.lower()
                    mime_map = {
                        ".jpg": "image/jpeg",
                        ".jpeg": "image/jpeg",
                        ".png": "image/png",
                        ".gif": "image/gif",
                        ".webp": "image/webp",
                    }
                    mime_type = mime_map.get(suffix, "image/jpeg")
                    contents.append({
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": img_data,
                        }
                    })
                    valid_images += 1
                except Exception as e:
                    print(f"Warning: Could not load image {img_path}: {e}")

            if valid_images > 0:
                prompt = f"[{valid_images} medical images included below for reference]\n\n" + prompt

        contents.append(prompt)
        return contents

    def _build_generate_config(self):
        return self._types.GenerateContentConfig(
            temperature=self.decoding_params["temperature"],
            max_output_tokens=self.decoding_params["max_output_tokens"],
            safety_settings=self.safety_settings,
        )

    def _generate_batch_via_api(
        self,
        samples: List[Dict],
        poll_seconds: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
    ) -> List[Dict]:
        """Run generation through Google Batch API with inlined requests."""
        poll_seconds = poll_seconds if poll_seconds is not None else self.batch_poll_seconds
        timeout_seconds = timeout_seconds if timeout_seconds is not None else self.batch_timeout_seconds

        reqs = []
        for i, sample in enumerate(samples):
            qid = sample.get("qid") or f"idx-{i}"
            reqs.append(
                self._types.InlinedRequest(
                    model=self.model_name,
                    contents=self._build_contents(
                        query=sample["query"],
                        contexts=sample.get("contexts", []),
                        image_paths=sample.get("image_paths"),
                        query_images=sample.get("query_images"),
                        context_images=sample.get("context_images"),
                        use_rag_prompt=sample.get("use_rag_prompt", True),
                    ),
                    metadata={"qid": qid},
                    config=self._build_generate_config(),
                )
            )

        display_name = f"leish-batch-{uuid.uuid4().hex[:8]}"
        job = self.client.batches.create(
            model=self.model_name,
            src=self._types.BatchJobSource(inlined_requests=reqs),
            config=self._types.CreateBatchJobConfig(display_name=display_name),
        )
        print(f"Started Gemini batch job: {job.name}")

        deadline = time.time() + timeout_seconds
        while True:
            job = self.client.batches.get(name=job.name)
            state = str(getattr(job, "state", "UNKNOWN"))
            if "SUCCEEDED" in state:
                break
            if any(x in state for x in ["FAILED", "CANCELLED", "EXPIRED"]):
                raise RuntimeError(f"Batch job ended in terminal error state: {state}")
            if time.time() > deadline:
                raise TimeoutError(f"Batch job timeout after {timeout_seconds}s: {state}")
            time.sleep(poll_seconds)

        dest = getattr(job, "dest", None)
        inlined_responses = getattr(dest, "inlined_responses", None) if dest else None
        if not inlined_responses:
            raise RuntimeError("Batch job succeeded but returned no inlined responses")

        sample_by_qid = {str(s.get("qid")): s for s in samples if s.get("qid") is not None}
        answers_by_qid: Dict[str, str] = {}

        for resp in inlined_responses:
            md = getattr(resp, "metadata", None) or {}
            qid = str(md.get("qid", ""))
            err = getattr(resp, "error", None)

            if err is not None:
                answers_by_qid[qid] = f"[Batch Error: {err}]"
                continue

            gen_resp = getattr(resp, "response", None)
            text, meta = _extract_text_safely(gen_resp)
            if text:
                answers_by_qid[qid] = text
                continue

            contexts = sample_by_qid.get(qid, {}).get("contexts", [])
            top_ids = [c.get("doc_id") for c in contexts[:5] if c.get("doc_id")]
            answers_by_qid[qid] = (
                "Insufficient evidence in retrieved excerpts (batch generation returned no content). "
                f"finish_reason={meta.get('finish_reason')}. "
                + (f"Top retrieved case IDs: {', '.join(top_ids)}." if top_ids else "")
            )

        results = []
        for sample in samples:
            qid = str(sample.get("qid", ""))
            answer = answers_by_qid.get(qid, "[Batch Error: missing response]")
            results.append(
                {
                    **sample,
                    "answer": answer,
                    "model_name": self.model_name,
                    "decoding_params": self.decoding_params,
                }
            )
        return results
    
    def generate_batch(
        self,
        samples: List[Dict],
        progress: bool = True,
        use_batch_api: bool = False,
        poll_seconds: Optional[float] = None,
        timeout_seconds: Optional[int] = None,
    ) -> List[Dict]:
        """
        Generate answers for multiple samples.
        
        Args:
            samples: List of {qid, query, contexts}
            progress: Show progress
        
        Returns:
            List of {qid, query, contexts, answer, model_name, ...}
        """
        if use_batch_api:
            try:
                return self._generate_batch_via_api(
                    samples=samples,
                    poll_seconds=poll_seconds,
                    timeout_seconds=timeout_seconds,
                )
            except Exception as e:
                print(f"Batch API failed, falling back to sequential generation: {e}")

        results = []
        
        for i, sample in enumerate(samples):
            answer = self.generate(
                sample["query"],
                sample.get("contexts", []),
                image_paths=sample.get("image_paths"),
                query_images=sample.get("query_images"),
                context_images=sample.get("context_images"),
                use_rag_prompt=sample.get("use_rag_prompt", True),
            )
            
            results.append({
                **sample,
                "answer": answer,
                "model_name": self.model_name,
                "decoding_params": self.decoding_params
            })
            
            if progress and (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{len(samples)}")
        
        return results


if __name__ == "__main__":
    # Test
    gen = GeminiGenerator()
    print(f"✓ Initialized Gemini: {gen.model_name}")
    
    test_answer = gen.generate(
        "What are symptoms of visceral leishmaniasis?",
        [{"doc_id": "PMC123", "text": "Fever, weight loss, splenomegaly..."}]
    )
    print(f"✓ Generated: {test_answer[:100]}...")
