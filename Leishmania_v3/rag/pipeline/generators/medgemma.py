"""
MedGemma Answer Generator

Local HuggingFace generation using google/medgemma-4b-it.
For research experiments - runs on local GPU.

Updated to have same interface as Gemma3Generator for fair comparison.
"""
import os
from typing import Dict, List, Optional

# Import from package to ensure HF cache is set
from .. import DATA_ROOT

# Import centralized prompt builder
from ...configs.prompt_mode import build_rag_prompt as _build_rag_prompt, PromptMode


# Default cache path
HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/data4t/hf/transformers")


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    query_images: List[str] = None,
    context_images: List[str] = None,
    prompt_mode: PromptMode = PromptMode.BALANCED
) -> str:
    """
    Build RAG prompt for MedGemma - BALANCED for domain-specialized models.
    
    WRAPPER: Delegates to centralized build_rag_prompt from configs/prompt_mode.py
    Default mode is BALANCED (original MedGemma behavior - synergy with parametric knowledge).
    
    Per paper "RAG-Enhanced Open SLMs for Hypertension Management":
    - RAG should NOT degrade model performance
    - For domain-specialized models, allow synergy between parametric + retrieved knowledge
    - Do NOT force override of model's medical training
    
    Args:
        query: Clinical question
        contexts: Retrieved contexts
        query_images: Patient image paths
        context_images: Evidence image paths
        prompt_mode: Which prompt template to use (default: BALANCED)
    """
    return _build_rag_prompt(
        query=query,
        contexts=contexts,
        mode=prompt_mode,
        query_images=query_images,
        context_images=context_images,
        max_chars_per_context=2000,
        include_context_images=True,
        is_text_only_model=True  # MedGemma is text-only
    )






class MedGemmaGenerator:
    """
    Generate answers using local MedGemma-4b-it.
    
    Runs on local GPU with HuggingFace transformers.
    Text-only model (no vision support).
    """
    
    MODEL_ID = "google/medgemma-4b-it"
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,  # Aligned with Gemma3
        cache_dir: str = None
    ):
        """
        Initialize MedGemma generator.
        
        Args:
            model_path: Local path or HF model ID
            device: Device (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Max new tokens
            cache_dir: HuggingFace cache directory
        """
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch")
        
        self._torch = torch
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_path or self.MODEL_ID
        self.cache_dir = cache_dir or HF_CACHE
        
        self.decoding_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0
        }
        
        # Load model
        print(f"Loading MedGemma from {self.model_name}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Device: {self.device}")
        
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
        
        print(f"✓ MedGemma loaded: {self.model_name}")
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None,
        use_rag_prompt: bool = True  # NEW: same as Gemma3Generator
    ) -> str:
        """
        Generate answer using MedGemma.
        
        Args:
            query: Query text
            contexts: Retrieved contexts
            image_paths: Ignored (MedGemma is text-only, but kept for API consistency)
            use_rag_prompt: If True, use RAG-formatted prompt; if False, use query directly
        
        Returns:
            Generated answer text
        """
        if use_rag_prompt:
            prompt = build_rag_prompt(query, contexts, image_paths if image_paths else None)
        else:
            # No-RAG mode: use query directly without retrieval prompt
            prompt = query
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=16384  # Aligned with Gemma3
            ).to(self.device)
            
            with self._torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **self.decoding_params,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode only new tokens
            answer = self.tokenizer.decode(
                outputs[0][inputs.input_ids.shape[1]:],
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


def get_medgemma_generator() -> MedGemmaGenerator:
    """Convenience function to get MedGemma generator."""
    return MedGemmaGenerator()


if __name__ == "__main__":
    print("Testing MedGemma Generator...")
    print(f"HF Cache: {HF_CACHE}")
    
    # Only test instantiation if model is available
    try:
        gen = MedGemmaGenerator()
        print(f"✓ Loaded: {gen.model_name}")
        
        # Test generation
        test_answer = gen.generate(
            "What are symptoms of visceral leishmaniasis?",
            [{"doc_id": "PMC123", "text": "Fever, weight loss, splenomegaly..."}]
        )
        print(f"✓ Generated: {test_answer[:100]}...")
    except Exception as e:
        print(f"⚠ Could not load MedGemma: {e}")
        print("  (Model may need to be downloaded first)")

