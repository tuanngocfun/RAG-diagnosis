#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone MedGemma-4B-IT multimodal medical analysis system.
No RAG pipeline - direct image and text processing using MedGemma-4B-IT.

Based on existing patterns from:
- rag/retriever/medgemma4b_qdrant_crossencoder_medcpt.py
- rag/retriever/run_batch_answers_medgemma4b_test_medcpt.py
- experiments/medgemma.ipynb
"""

import os
import re
import json
import logging
import hashlib
import math
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
import torch
from PIL import Image

# Set up environment for offline usage
os.environ.setdefault("HF_HUB_OFFLINE", "1") 
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Handle newer transformers versions that prefer HF_HOME
if "TRANSFORMERS_CACHE" in os.environ and "HF_HOME" not in os.environ:
    # Set HF_HOME to parent of TRANSFORMERS_CACHE for newer transformers versions
    transformers_cache = Path(os.environ["TRANSFORMERS_CACHE"])
    os.environ["HF_HOME"] = str(transformers_cache.parent)

from transformers import (
    AutoProcessor, AutoModelForVision2Seq, AutoModelForImageTextToText,
    AutoTokenizer
)

# -------------------- Configuration --------------------
class StandaloneCFG:
    """Configuration for standalone MedGemma system"""
    # Models
    MEDGEMMA_MODEL_ID: str = "google/medgemma-4b-it"
    
    # HF cache / device
    HF_CACHE: Path = Path(os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers"))
    
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generation parameters
    MAX_NEW_TOKENS: int = 1024
    MAX_IMAGES_PER_ANSWER: int = 4
    
    # Processor stability
    USE_FAST_PROCESSORS: bool = True
    
    # Extract root for case data
    EXTRACT_ROOT: Path = Path(os.getenv("RAG_EXTRACT_ROOT", "/app/kaggle/working2/extract"))

# -------------------- Utility Functions --------------------

def resolve_local_model_dir(model_id: str, cache_dir: Path) -> str:
    """
    Return a usable path for from_pretrained(..., local_files_only=True).
    Supports:
      1) HF cache layout:   <cache>/models--org--repo/snapshots/<rev>/
      2) Flat local folder: <cache>/<repo>/
      3) Absolute path passed as model_id
    """
    def usable(d: Path) -> bool:
        if not d or not d.exists():
            return False
        has_cfg = (d / "config.json").exists()
        has_proc = (d / "preprocessor_config.json").exists() or (d / "processor_config.json").exists()
        has_tok = (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
        # Accept either processor (for vision/multimodal) or tokenizer (for text models like BGE)
        return has_cfg and (has_proc or has_tok)

    # 0) Absolute or relative explicit path
    p = Path(model_id)
    if p.exists() and usable(p):
        return str(p)

    # 1) Hugging Face cache layout
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snaps_root = repo_dir / "snapshots"
    if snaps_root.is_dir():
        # Prefer "offline-materialized"
        offline = snaps_root / "offline-materialized"
        if usable(offline):
            return str(offline)
        # Otherwise newest usable snapshot
        snaps = sorted((x for x in snaps_root.iterdir() if x.is_dir()),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        for s in snaps:
            if usable(s):
                return str(s)
    # Sometimes files are placed directly under the repo dir
    if usable(repo_dir):
        return str(repo_dir)

    # 2) Flat local folder (your case: /data4t/hf/transformers/bge-reranker-v2-m3)
    flat_dir = cache_dir / model_id.split("/")[-1]
    if usable(flat_dir):
        return str(flat_dir)

    # 3) Fall back to original HF id (will work online; offline will raise)
    return model_id

def _normalize_answer(text: str) -> str:
    """
    Enhanced normalization for MedGemma outputs to handle repetition and garbage.
    """
    if not text:
        return text
    
    t = text.strip()
    
    # Early garbage detection - truncate very long outputs
    if len(t) > 4000:
        print("⚠️ [FIX] Truncating very long answer (likely repetitive)")
        t = t[:2000] + "..."
    
    # Remove prompt leakage patterns
    t = re.sub(r'^\s*(?:user\s+)?(?:You are a medical expert[^.]*\.)?', '', t, flags=re.I)
    t = re.sub(r'Evidence Sources:\s*\[?\d+\]?[^.]*\.', '', t, flags=re.I)
    t = re.sub(r'User has uploaded \d+ files:[^.]*\.', '', t, flags=re.I)
    t = re.sub(r'Please analyze both the uploaded content[^.]*\.', '', t, flags=re.I)
    
    # Remove diagnostic instruction echoing
    t = re.sub(r'Medical Question:\s*Provide a systematic medical diagnosis for:[^.]*\.?', '', t, flags=re.I)
    t = re.sub(r'Please structure your diagnostic analysis as follows:[^.]*\.?', '', t, flags=re.I)
    t = re.sub(r'Key clinical findings from the evidence[^.]*\.?', '', t, flags=re.I)
    
    # Remove answer prefixes
    t = re.sub(r'^\s*(?:Question\s*:\s*.*?)?\b(?:Answer|ANSWER|MEDICAL ANALYSIS)\s*:\s*', '', t, flags=re.I | re.S)
    
    # Handle Final Answer sections
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail and len(tail) < len(t) * 0.8:
            t = tail
    
    # Remove repetitive case descriptions
    t = re.sub(r'(?:\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+A\s+LESION[^.]*\.?\s*){2,}', 
               '', t, flags=re.I)
    
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    
    # Enhanced sentence deduplication
    sentences = re.split(r'(?<=[.!?])(?:["\')\]\}]+)?(?:\s+)', t)
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        # Create deduplication key
        key = re.sub(r'\W+', '', sentence.lower())[:50]
        
        # Skip repetitive patterns
        if any(pattern in sentence.lower() for pattern in [
            "year-old boy from laos", "year-old man from cambodia",
            "progressed quickly from", "medical expert", "evidence sources"
        ]):
            continue
        
        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(sentence)
            
            if len(unique_sentences) >= 6:
                break
    
    result = " ".join(unique_sentences).strip()
    
    # Ensure proper ending
    if result and result[-1] not in '.!?':
        last_punct = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_punct > len(result) * 0.7:
            result = result[:last_punct + 1]
        else:
            result += "."
    
    # Final sanity check
    if len(result) < 30 or result.count(' ') < 4:
        return "Unable to generate a clear medical answer from the available information."
    
    return result

def find_case_dir(case_id: str, extract_root: Path) -> Optional[Path]:
    """Locate a case folder under EXTRACT_ROOT from a case_id."""
    # exact match
    cand = extract_root / case_id
    if cand.is_dir():
        return cand
    # sanitized match
    sanitized = re.sub(r"[^\w\-.]", "_", case_id)
    sanitized = re.sub(r"_+", "_", sanitized).strip("_")
    cand2 = extract_root / sanitized
    if cand2.is_dir():
        return cand2
    # loose match (contains)
    for p in extract_root.iterdir():
        if p.is_dir() and sanitized.lower() in p.name.lower():
            return p
    return None

def page_indices_to_paths(case_dir: Path, indices: List[int]) -> List[Path]:
    """Return image paths for 0-based page indices."""
    pages_dir = case_dir / "pages"
    out: List[Path] = []
    for i in indices:
        p = pages_dir / f"page_{i+1:04d}.png"  # 1-based filename
        if p.exists():
            out.append(p)
    return out

def select_images(seed_images: List[str], case_dir: Optional[Path], 
                 target_pages: List[int], max_images: int) -> List[str]:
    """
    Select up to max_images from seed images and case directory.
    """
    out, seen = [], set()

    # 1) seed paths first
    for p in seed_images or []:
        if p and Path(p).exists() and p not in seen:
            out.append(p)
            seen.add(p)
            if len(out) >= max_images:
                return out

    # 2) case directory pages
    if case_dir and target_pages:
        for p in page_indices_to_paths(case_dir, target_pages):
            if p.exists():
                sp = str(p)
                if sp not in seen:
                    out.append(sp)
                    seen.add(sp)
                    if len(out) >= max_images:
                        return out

    return out

# -------------------- Standalone MedGemma System --------------------

class StandaloneMedGemma4B:
    """
    Standalone MedGemma-4B-IT for direct multimodal medical analysis.
    No RAG pipeline - processes images and text directly.
    """
    
    def __init__(self, model_id: str = None):
        self.model_id = model_id or StandaloneCFG.MEDGEMMA_MODEL_ID
        self.device = StandaloneCFG.DEVICE
        self.cfg = StandaloneCFG()
        
        # Initialize model and processor
        self._load_model()
        
        print(f"[INFO] StandaloneMedGemma4B loaded on {self.device}")
        
    def _load_model(self):
        """Load MedGemma-4B-IT model and processor."""
        name_or_path = resolve_local_model_dir(self.model_id, self.cfg.HF_CACHE)

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self._dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        print(f"[INFO] Loading MedGemma model from: {name_or_path}")
        print(f"[DEBUG] HF_CACHE: {self.cfg.HF_CACHE}")
        print(f"[DEBUG] Model path exists: {Path(name_or_path).exists()}")
        
        # Check if resolved path exists
        if not Path(name_or_path).exists():
            print(f"[ERROR] Model path does not exist: {name_or_path}")
            print(f"[DEBUG] Checking cache directory: {self.cfg.HF_CACHE}")
            if self.cfg.HF_CACHE.exists():
                print(f"[DEBUG] Cache dir contents: {list(self.cfg.HF_CACHE.iterdir())[:5]}")
            raise FileNotFoundError(f"Model not found at {name_or_path}")

        # Processor - use same pattern as existing RAG code
        self.processor = AutoProcessor.from_pretrained(
            name_or_path, 
            cache_dir=str(self.cfg.HF_CACHE), 
            local_files_only=True,
            use_fast=self.cfg.USE_FAST_PROCESSORS
        )
        
        # Model - use same pattern as existing RAG code
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                name_or_path, 
                cache_dir=str(self.cfg.HF_CACHE), 
                local_files_only=True,
                device_map="auto", 
                torch_dtype=self._dtype
            ).eval()
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(
                name_or_path, 
                cache_dir=str(self.cfg.HF_CACHE), 
                local_files_only=True,
                device_map="auto", 
                torch_dtype=self._dtype
            ).eval()

        # Tokenizer (if exposed by processor) - same pattern as existing RAG code
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name_or_path, 
                    cache_dir=str(self.cfg.HF_CACHE), 
                    local_files_only=True
                )
            except Exception:
                self.tokenizer = None

    def analyze_images(self, image_paths: List[str], question: str = "", 
                      max_output_tokens: int = None) -> str:
        """
        Analyze medical images with optional question.
        
        Args:
            image_paths: List of paths to medical images
            question: Optional medical question about the images
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            Medical analysis text
        """
        if not image_paths:
            return "No images provided for analysis."
            
        # Load and validate images
        valid_images = []
        for path in image_paths[:self.cfg.MAX_IMAGES_PER_ANSWER]:
            try:
                if Path(path).exists():
                    img = Image.open(path).convert("RGB")
                    # Resize for memory efficiency
                    if max(img.size) > 1024:
                        img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                    valid_images.append(img)
            except Exception as e:
                print(f"[WARN] Failed to load image {path}: {e}")
                continue
                
        if not valid_images:
            return "No valid images could be loaded."
            
        # Prepare prompt without manual image tokens (handled by processor)
        if question:
            # Specific medical question
            prompt = f"""Analyze the provided medical image(s) to answer this question: {question}

Please provide:
1. Visual observations from the image(s)
2. Medical interpretation of findings
3. Relevant diagnostic considerations
4. Answer to the specific question

Answer:"""
        else:
            # General medical analysis
            prompt = """Analyze the provided medical image(s). Describe what you observe and provide medical interpretation of any findings.

Focus on:
1. Anatomical structures visible
2. Any pathological findings
3. Clinical significance
4. Possible diagnoses or conditions

Medical Analysis:"""
            
        return self._generate_with_images(prompt, valid_images, max_output_tokens)
    
    def answer_medical_question(self, question: str, image_paths: List[str] = None, 
                               context: str = "", max_output_tokens: int = None) -> str:
        """
        Answer a medical question with optional images and context.
        
        Args:
            question: Medical question to answer
            image_paths: Optional medical images for context
            context: Optional text context
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            Medical answer
        """
        # Load images if provided
        images = []
        if image_paths:
            for path in image_paths[:self.cfg.MAX_IMAGES_PER_ANSWER]:
                try:
                    if Path(path).exists():
                        img = Image.open(path).convert("RGB")
                        if max(img.size) > 1024:
                            img.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
                        images.append(img)
                except Exception:
                    continue
        
        # Build prompt
        prompt_parts = []
        
        if context:
            prompt_parts.append(f"Medical Context: {context}")
            
        prompt_parts.append(f"Question: {question}")
        
        if images:
            prompt_parts.append("Please analyze the provided image(s) in your response.")
            
        prompt_parts.append("Provide a clear, evidence-based medical answer:")
        
        prompt = "\n\n".join(prompt_parts)
        
        if images:
            return self._generate_with_images(prompt, images, max_output_tokens)
        else:
            return self._generate_text_only(prompt, max_output_tokens)
    
    def diagnose_case(self, case_id: str, question: str = "", 
                     seed_images: List[str] = None, max_output_tokens: int = None) -> str:
        """
        Provide diagnostic analysis for a case using images from the case directory.
        
        Args:
            case_id: Case identifier to locate in extract root
            question: Optional specific diagnostic question
            seed_images: Optional specific images to use
            max_output_tokens: Maximum tokens to generate
            
        Returns:
            Diagnostic analysis
        """
        # Find case directory
        case_dir = find_case_dir(case_id, self.cfg.EXTRACT_ROOT)
        if not case_dir:
            return f"Case directory not found for case_id: {case_id}"
            
        # Select images (first 3 pages if no seed images provided)
        target_pages = [0, 1, 2] if not seed_images else []
        image_paths = select_images(
            seed_images or [], case_dir, target_pages, self.cfg.MAX_IMAGES_PER_ANSWER
        )
        
        if not image_paths:
            return f"No images found for case: {case_id}"
            
        # Prepare diagnostic prompt with image tokens
        if question:
            diagnostic_prompt = f"""Case Analysis Request: {question}

Case ID: {case_id}

Please provide a systematic diagnostic analysis based on the medical images from this case:

1. Clinical observations from the images
2. Key diagnostic features identified  
3. Differential diagnoses to consider
4. Most likely diagnosis with supporting evidence
5. Recommended diagnostic tests or procedures

Diagnostic Assessment:"""
        else:
            diagnostic_prompt = f"""Comprehensive Case Analysis

Case ID: {case_id}

Please provide a complete diagnostic evaluation based on the medical images:

1. Visual findings and clinical observations
2. Morphological characteristics of any lesions/abnormalities  
3. Geographic/epidemiologic considerations if applicable
4. Differential diagnostic considerations
5. Most likely diagnosis with rationale
6. Suggested confirmatory tests

Medical Diagnosis:"""
            
        # Call analyze_images with the diagnostic question
        return self.analyze_images(image_paths, diagnostic_prompt, max_output_tokens)

    def _generate_with_images(self, prompt: str, images: List[Image.Image], 
                            max_output_tokens: int = None) -> str:
        """Generate response with images using MedGemma-4B-IT."""
        max_tokens = max_output_tokens or self.cfg.MAX_NEW_TOKENS
        
        try:
            # Use message-based approach like existing RAG code
            if images:
                # Create proper message structure for multimodal input
                messages = [
                    {
                        "role": "user",
                        "content": [{"type": "text", "text": prompt}] + 
                                   [{"type": "image", "image": img} for img in images]
                    }
                ]
            else:
                # Text-only message structure  
                messages = [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]

            # Try structured message processing first (like existing RAG code)
            inputs = None
            if images and self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                try:
                    inputs = self.processor.apply_chat_template(
                        messages,
                        add_generation_prompt=True,
                        tokenize=True,
                        return_dict=True,
                        return_tensors="pt",
                    )
                except Exception as e:
                    print(f"[DEBUG] Structured processing failed: {e}")
                    inputs = None
            
            # Fallback to traditional text+images processing
            if inputs is None:
                inputs = self.processor(
                    text=prompt,
                    images=images,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048  # Conservative context limit
                )
            
        except ValueError as e:
            error_msg = str(e).lower()
            if "image tokens" in error_msg or "image" in error_msg:
                print(f"[WARN] Multimodal processing failed, using text-only: {e}")
                # Complete fallback to text-only
                try:
                    inputs = self.processor(text=prompt, return_tensors="pt")
                    images = []  # Clear images since we're falling back
                except Exception as fallback_e:
                    print(f"[ERROR] Even text-only fallback failed: {fallback_e}")
                    raise
            else:
                print(f"[ERROR] Processing failed: {type(e).__name__}: {e}")
                raise
        except Exception as e:
            print(f"[ERROR] Processing failed: {type(e).__name__}: {e}")
            raise

        try:
            # Move to device
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            
            # Generation parameters
            gen_kwargs = {
                "max_new_tokens": min(max_tokens, 512),
                "min_new_tokens": 20,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 4,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
                "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None
            }
            
            # Filter out None values
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            
            # Generate
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            # Decode
            input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            if input_len > 0 and len(outputs.shape) > 1:
                new_tokens = outputs[0][input_len:]
            else:
                new_tokens = outputs[0] if len(outputs.shape) > 1 else outputs
            
            # Decode text
            if self.processor is not None:
                text = self.processor.decode(new_tokens, skip_special_tokens=True)
            elif self.tokenizer is not None:
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                text = ""
                
            return _normalize_answer(text)
            
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            return f"Error generating response: {str(e)}"

    def _generate_text_only(self, prompt: str, max_output_tokens: int = None) -> str:
        """Generate text-only response."""
        max_tokens = max_output_tokens or self.cfg.MAX_NEW_TOKENS
        
        try:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            )
            
            inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            
            gen_kwargs = {
                "max_new_tokens": min(max_tokens, 512),
                "min_new_tokens": 20,
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.2,
                "pad_token_id": self.tokenizer.pad_token_id if self.tokenizer else None,
                "eos_token_id": self.tokenizer.eos_token_id if self.tokenizer else None
            }
            
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}
            
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **gen_kwargs)
            
            input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
            new_tokens = outputs[0][input_len:] if input_len > 0 else outputs[0]
            
            if self.tokenizer is not None:
                text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            else:
                text = ""
                
            return _normalize_answer(text)
            
        except Exception as e:
            print(f"[ERROR] Text generation failed: {e}")
            return f"Error generating response: {str(e)}"

# -------------------- Main Functions --------------------

def create_medgemma_analyzer() -> StandaloneMedGemma4B:
    """Create and return a standalone MedGemma analyzer."""
    return StandaloneMedGemma4B()

def analyze_medical_images(analyzer: StandaloneMedGemma4B, image_paths: List[str], 
                          question: str = "") -> Dict[str, Any]:
    """
    Analyze medical images and return structured results.
    
    Returns:
        Dictionary with analysis text and metadata
    """
    analysis = analyzer.analyze_images(image_paths, question)
    
    return {
        "analysis": analysis,
        "image_count": len(image_paths),
        "question": question,
        "model": analyzer.model_id
    }

def batch_analyze_cases(analyzer: StandaloneMedGemma4B, case_ids: List[str], 
                       questions: List[str] = None) -> List[Dict[str, Any]]:
    """
    Analyze multiple cases in batch.
    
    Args:
        analyzer: MedGemma analyzer instance
        case_ids: List of case IDs to analyze
        questions: Optional questions for each case
        
    Returns:
        List of analysis results
    """
    results = []
    questions = questions or [""] * len(case_ids)
    
    for i, case_id in enumerate(case_ids):
        question = questions[i] if i < len(questions) else ""
        
        try:
            analysis = analyzer.diagnose_case(case_id, question)
            results.append({
                "case_id": case_id,
                "question": question,
                "analysis": analysis,
                "success": True,
                "error": None
            })
        except Exception as e:
            results.append({
                "case_id": case_id,
                "question": question,
                "analysis": "",
                "success": False,
                "error": str(e)
            })
            
    return results

# -------------------- Example Usage --------------------

def main():
    """Example usage of standalone MedGemma system."""
    # Create analyzer
    analyzer = create_medgemma_analyzer()
    
    # Example 1: Analyze specific images
    image_paths = ["/path/to/medical_image1.png", "/path/to/medical_image2.png"]
    question = "What diagnostic features are visible in these images?"
    
    if all(Path(p).exists() for p in image_paths):
        result = analyze_medical_images(analyzer, image_paths, question)
        print("Medical Image Analysis:")
        print(result["analysis"])
    
    # Example 2: Diagnose a case by case_id
    case_id = "example_case_001"
    diagnosis = analyzer.diagnose_case(case_id, "What is the most likely diagnosis?")
    print(f"\nCase Diagnosis for {case_id}:")
    print(diagnosis)
    
    # Example 3: Answer general medical question
    medical_question = "What are the key diagnostic features of cutaneous leishmaniasis?"
    answer = analyzer.answer_medical_question(medical_question)
    print(f"\nMedical Q&A:")
    print(answer)

if __name__ == "__main__":
    main()