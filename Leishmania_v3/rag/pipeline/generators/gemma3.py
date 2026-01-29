"""
Gemma 3 Answer Generator (12B / 27B)

Local HuggingFace generation using google/gemma-3-12b-it or google/gemma-3-27b-it.
Supports multimodal (vision) via gemma-3-12b-vision-it / gemma-3-27b-vision-it.

For evaluating RAG value on resource-limited models with Gemini 2.5 Pro as judge.
"""
import os
from typing import Dict, List, Optional
from pathlib import Path

# Import from package to ensure HF cache is set
from .. import DATA_ROOT

# Import centralized prompt builder
from ...configs.prompt_mode import build_rag_prompt as _build_rag_prompt, PromptMode

# Default cache path
HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/data4t/hf/transformers")

# Gemma 3 model variants
GEMMA3_MODELS = {
    "12b": "google/gemma-3-12b-it",
    "27b": "google/gemma-3-27b-it",
    "12b-vision": "google/gemma-3-12b-vision-it",
    "27b-vision": "google/gemma-3-27b-vision-it",
}


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    query_images: List[str] = None,
    context_images: List[str] = None,
    prompt_mode: PromptMode = PromptMode.STRICT_CONTEXT
) -> str:
    """
    Build RAG prompt for Gemma 3 - ALIGNED with Gemini for fair comparison.
    
    WRAPPER: Delegates to centralized build_rag_prompt from configs/prompt_mode.py
    Default mode is STRICT_CONTEXT (aligned with Gemini behavior).
    
    Per GPT 5.2 recommendations:
    - Same structure as gemini.py build_rag_prompt
    - Explicit REQUIRED OUTPUT FORMAT section
    - FAITHFULNESS ENFORCEMENT
    - Same output headers for LLM judge parsing
    
    Args:
        query: Clinical question
        contexts: Retrieved contexts
        query_images: Patient image paths
        context_images: Evidence image paths
        prompt_mode: Which prompt template to use (default: STRICT_CONTEXT)
    """
    return _build_rag_prompt(
        query=query,
        contexts=contexts,
        mode=prompt_mode,
        query_images=query_images,
        context_images=context_images,
        max_chars_per_context=2000,
        include_context_images=True,
        is_text_only_model=False  # Gemma3 can have vision variant
    )




class Gemma3Generator:
    """
    Generate answers using local Gemma 3 (12B or 27B).
    
    Runs on local GPU with HuggingFace transformers.
    Supports multimodal via vision variants.
    """
    
    def __init__(
        self,
        variant: str = "12b",
        device: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_dir: str = None,
        use_vision: bool = False,
        prompt_mode: PromptMode = PromptMode.STRICT_CONTEXT
    ):
        """
        Initialize Gemma 3 generator.
        
        Args:
            variant: Model size - "12b" or "27b"
            device: Device (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Max new tokens
            cache_dir: HuggingFace cache directory
            use_vision: If True, use vision-enabled variant for multimodal
            prompt_mode: Prompt template mode (STRICT_CONTEXT, BALANCED, NO_CONTEXT)
        """
        self.prompt_mode = prompt_mode
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")
        
        self._torch = torch
        self._PIL = None  # Lazy load PIL for images
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.cache_dir = cache_dir or HF_CACHE
        self.use_vision = use_vision
        
        # Select model variant
        if use_vision:
            model_key = f"{variant}-vision"
        else:
            model_key = variant
        
        if model_key not in GEMMA3_MODELS:
            raise ValueError(f"Unknown variant: {model_key}. Choose from: {list(GEMMA3_MODELS.keys())}")
        
        self.model_name = GEMMA3_MODELS[model_key]
        
        self.decoding_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0
        }
        
        # Load model
        print(f"Loading Gemma 3 from {self.model_name}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Device: {self.device}")
        print(f"  Vision: {self.use_vision}")
        
        if use_vision:
            # Use processor for vision models
            self.processor = AutoProcessor.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ Gemma 3 loaded: {self.model_name}")
    
    def _load_images(self, image_paths: List[str]) -> List:
        """Load images for vision model."""
        if self._PIL is None:
            from PIL import Image
            self._PIL = Image
        
        images = []
        for path in image_paths[:5]:  # Max 5 images
            p = Path(path)
            if p.exists():
                try:
                    img = self._PIL.Image.open(p).convert("RGB")
                    images.append(img)
                except Exception as e:
                    print(f"Warning: Could not load image {path}: {e}")
        return images
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None,
        use_rag_prompt: bool = True
    ) -> str:
        """
        Generate answer using Gemma 3.
        
        Args:
            query: Query text
            contexts: Retrieved contexts
            image_paths: Image paths for vision model
            use_rag_prompt: If True, use RAG-formatted prompt
        
        Returns:
            Generated answer text
        """
        if use_rag_prompt:
            prompt = build_rag_prompt(
                query, contexts, 
                query_images=image_paths if image_paths else None,
                prompt_mode=self.prompt_mode
            )
        else:
            prompt = query
        
        try:
            if self.use_vision and image_paths:
                # Multimodal generation
                images = self._load_images(image_paths)
                if images:
                    inputs = self.processor(
                        text=prompt,
                        images=images,
                        return_tensors="pt"
                    ).to(self.device)
                else:
                    # Fall back to text-only if no valid images
                    inputs = self.tokenizer(
                        prompt,
                        return_tensors="pt",
                        truncation=True,
                        max_length=4096
                    ).to(self.device)
            else:
                # Text-only generation
                inputs = self.tokenizer(
                    prompt,
                    return_tensors="pt",
                    truncation=True,
                    max_length=4096
                ).to(self.device)
            
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.decoding_params,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            input_len = inputs.input_ids.shape[1] if hasattr(inputs, 'input_ids') else inputs['input_ids'].shape[1]
            answer = self.tokenizer.decode(
                outputs[0][input_len:],
                skip_special_tokens=True
            )
            return answer.strip()
            
        except Exception as e:
            return f"[Generation Error: {type(e).__name__}: {e}]"
    
    def generate_batch(
        self,
        samples: List[Dict],
        progress: bool = True
    ) -> List[Dict]:
        """
        Generate answers for multiple samples.
        
        Args:
            samples: List of {qid, query, contexts, query_images}
            progress: Show progress
        
        Returns:
            List of {qid, query, contexts, answer, model_name, ...}
        """
        results = []
        
        for i, sample in enumerate(samples):
            answer = self.generate(
                sample["query"],
                sample.get("contexts", []),
                image_paths=sample.get("query_images", [])
            )
            
            results.append({
                **sample,
                "answer": answer,
                "model_name": self.model_name,
                "decoding_params": self.decoding_params
            })
            
            if progress and (i + 1) % 5 == 0:
                print(f"  Generated {i + 1}/{len(samples)}")
        
        return results


def get_gemma3_generator(variant: str = "12b", use_vision: bool = False) -> Gemma3Generator:
    """Convenience function to get Gemma 3 generator."""
    return Gemma3Generator(variant=variant, use_vision=use_vision)


if __name__ == "__main__":
    print("Testing Gemma 3 Generator...")
    print(f"HF Cache: {HF_CACHE}")
    print(f"Available variants: {list(GEMMA3_MODELS.keys())}")
    
    # Only test instantiation if model is available
    try:
        gen = Gemma3Generator(variant="12b")
        print(f"✓ Loaded: {gen.model_name}")
        
        # Test generation
        test_answer = gen.generate(
            "What are symptoms of visceral leishmaniasis?",
            [{"doc_id": "PMC123", "text": "Fever, weight loss, splenomegaly..."}]
        )
        print(f"✓ Generated: {test_answer[:100]}...")
    except Exception as e:
        print(f"⚠ Could not load Gemma 3: {e}")
        print("  (Model may need to be downloaded first)")
