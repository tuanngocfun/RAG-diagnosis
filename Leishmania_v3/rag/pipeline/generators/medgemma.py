"""
MedGemma Answer Generator

Local HuggingFace generation using google/medgemma-4b-it.
For research experiments - runs on local GPU.
"""
import os
from typing import Dict, List, Optional

# Import from package to ensure HF cache is set
from .. import DATA_ROOT


# Default cache path
HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/data4t/hf/transformers")


def build_rag_prompt(
    query: str,
    contexts: List[Dict]
) -> str:
    """Build RAG prompt for MedGemma."""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n[Context {i}] (Case: {ctx.get('doc_id', 'unknown')})\n"
        context_text += ctx.get("text", "")[:2000]  # Aligned with Gemini for fair comparison
    
    return f"""You are a medical expert for Leishmaniasis diagnosis.
Answer based on the retrieved contexts. Cite cases using [Case: PMC...].

QUERY: {query}

CONTEXTS:
{context_text}

ANSWER:"""


class MedGemmaGenerator:
    """
    Generate answers using local MedGemma-4b-it.
    
    Runs on local GPU with HuggingFace transformers.
    """
    
    MODEL_ID = "google/medgemma-4b-it"
    
    def __init__(
        self,
        model_path: str = None,
        device: str = None,
        temperature: float = 0.3,
        max_tokens: int = 512,
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
        
        print(f"✓ MedGemma loaded on {self.device}")
    
    def generate(
        self,
        query: str,
        contexts: List[Dict],
        image_paths: List[str] = None  # Not used for text-only
    ) -> str:
        """
        Generate answer using MedGemma.
        
        Args:
            query: Query text
            contexts: Retrieved contexts
            image_paths: Ignored (MedGemma is text-only)
        
        Returns:
            Generated answer text
        """
        prompt = build_rag_prompt(query, contexts)
        
        try:
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=2048
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
            return f"[Generation Error: {e}]"
    
    def generate_batch(
        self,
        samples: List[Dict],
        progress: bool = True
    ) -> List[Dict]:
        """
        Generate answers for multiple samples.
        
        Args:
            samples: List of {qid, query, contexts}
            progress: Show progress
        
        Returns:
            List of {qid, query, contexts, answer, model_name, ...}
        """
        results = []
        
        for i, sample in enumerate(samples):
            answer = self.generate(
                sample["query"],
                sample.get("contexts", [])
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


if __name__ == "__main__":
    print("Testing MedGemma Generator...")
    print(f"HF Cache: {HF_CACHE}")
    
    # Only test instantiation if model is available
    try:
        gen = MedGemmaGenerator()
        print(f"✓ Loaded: {gen.model_name}")
    except Exception as e:
        print(f"⚠ Could not load MedGemma: {e}")
        print("  (Model may need to be downloaded first)")
