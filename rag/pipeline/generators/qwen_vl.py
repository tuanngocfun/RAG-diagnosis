"""
Qwen2.5-VL Answer Generator

Local HuggingFace generation using Qwen/Qwen2.5-VL-7B-Instruct.
Supports:
- Text + Image inputs
- 128K context window
- Dynamic resolution (no fixed image size)

VRAM: ~15GB for 7B model
"""
import os
from typing import Dict, List, Optional
from pathlib import Path

# Import from package to ensure HF cache is set
from .. import DATA_ROOT

# Default cache path
HF_CACHE = os.environ.get("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers")

# Qwen2.5-VL model variants
QWEN_VL_MODELS = {
    "7b": "Qwen/Qwen2.5-VL-7B-Instruct",
    "72b": "Qwen/Qwen2.5-VL-72B-Instruct",  # Needs multi-GPU
}


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    query_images: List[str] = None,
    context_images: List[str] = None
) -> str:
    """Build RAG prompt for Qwen VL."""
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n[Context {i}] (Case: {ctx.get('doc_id', 'unknown')})\n"
        context_text += ctx.get("text", "")[:2000]
    
    image_sections = ""
    if query_images:
        image_sections += f"\n\n## PATIENT IMAGES\n[{len(query_images)} patient image(s) attached for diagnosis]\n"
    if context_images:
        image_sections += f"\n## EVIDENCE IMAGES\n[{len(context_images)} supporting image(s) from similar cases]\n"
    
    return f"""You are a medical expert specializing in Leishmaniasis diagnosis.

IMPORTANT: This is for RESEARCH and EVALUATION purposes only.
Base your diagnosis primarily on the retrieved case excerpts.

QUERY:
{query}
{image_sections}
RETRIEVED CASES:
{context_text}

TASK: Provide a diagnosis assessment.

## DIAGNOSIS PREDICTION
**Primary Diagnosis:** [Your diagnosis]
**Diagnosis Type:** [CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other]
**Species (if determinable):** [e.g., L. donovani, L. tropica, or "Not determinable"]
**Confidence:** [High/Medium/Low]
**Evidence Source:** [Retrieved cases only / Retrieved + general knowledge]

## RANKED DIFFERENTIAL (Top-3)
1) [Most likely differential]
2) [Second differential]
3) [Third differential]

## SUPPORTING EVIDENCE
- Key findings from retrieved cases
- Cite cases using their IDs (e.g., "Case PMC123456")

DIAGNOSIS ASSESSMENT:"""


class QwenVLGenerator:
    """
    Generate answers using Qwen2.5-VL (7B or 72B).
    
    Uses transformers with qwen_vl_utils for image processing.
    Supports multimodal input (text + images).
    """
    
    def __init__(
        self,
        variant: str = "7b",
        device: str = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        cache_dir: str = None,
    ):
        """
        Initialize Qwen VL generator.
        
        Args:
            variant: Model size - "7b" or "72b"
            device: Device (auto-detected if None)
            temperature: Sampling temperature
            max_tokens: Max new tokens
            cache_dir: HuggingFace cache directory
        """
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            import torch
        except ImportError:
            raise ImportError("pip install transformers torch qwen-vl-utils")
        
        self._torch = torch
        self._PIL = None  # Lazy load
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.cache_dir = cache_dir or HF_CACHE
        
        if variant not in QWEN_VL_MODELS:
            raise ValueError(f"Unknown variant: {variant}. Choose from: {list(QWEN_VL_MODELS.keys())}")
        
        self.model_name = QWEN_VL_MODELS[variant]
        
        self.decoding_params = {
            "temperature": temperature,
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0
        }
        
        # Load model
        print(f"Loading Qwen2.5-VL from {self.model_name}...")
        print(f"  Cache dir: {self.cache_dir}")
        print(f"  Device: {self.device}")
        
        self.processor = AutoProcessor.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            trust_remote_code=True
        )
        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            cache_dir=self.cache_dir,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        print(f"✓ Qwen2.5-VL loaded: {self.model_name}")
    
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
        Generate answer using Qwen2.5-VL.
        
        Args:
            query: Query text
            contexts: Retrieved contexts
            image_paths: Image paths for multimodal input
            use_rag_prompt: If True, use RAG-formatted prompt
        
        Returns:
            Generated answer text
        """
        if use_rag_prompt:
            prompt = build_rag_prompt(query, contexts, image_paths if image_paths else None)
        else:
            prompt = query
        
        try:
            # Build messages format for Qwen VL
            messages = []
            content = []
            
            # Add images if provided
            if image_paths:
                images = self._load_images(image_paths)
                for img in images:
                    content.append({"type": "image", "image": img})
            
            # Add text
            content.append({"type": "text", "text": prompt})
            
            messages.append({"role": "user", "content": content})
            
            # Process with Qwen VL processor
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Get image inputs if any
            if image_paths:
                images = self._load_images(image_paths)
                inputs = self.processor(
                    text=[text],
                    images=images if images else None,
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            else:
                inputs = self.processor(
                    text=[text],
                    padding=True,
                    return_tensors="pt"
                ).to(self.device)
            
            # Generate
            with self._torch.no_grad():
                output_ids = self.model.generate(
                    **inputs,
                    **self.decoding_params,
                    pad_token_id=self.processor.tokenizer.pad_token_id
                )
            
            # Decode - only new tokens
            generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
            answer = self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
            
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


def get_qwen_vl_generator(variant: str = "7b") -> QwenVLGenerator:
    """Convenience function to get Qwen VL generator."""
    return QwenVLGenerator(variant=variant)


if __name__ == "__main__":
    print("Testing Qwen2.5-VL Generator...")
    print(f"HF Cache: {HF_CACHE}")
    print(f"Available variants: {list(QWEN_VL_MODELS.keys())}")
    
    try:
        gen = QwenVLGenerator(variant="7b")
        print(f"✓ Loaded: {gen.model_name}")
        
        # Test generation
        test_answer = gen.generate(
            "What are symptoms of visceral leishmaniasis?",
            [{"doc_id": "PMC123", "text": "Fever, weight loss, splenomegaly..."}]
        )
        print(f"✓ Generated: {test_answer[:100]}...")
    except Exception as e:
        print(f"⚠ Could not load Qwen2.5-VL: {e}")
        print("  (Model may need to be downloaded first)")
