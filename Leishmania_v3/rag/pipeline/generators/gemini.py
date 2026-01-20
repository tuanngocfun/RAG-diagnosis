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
from typing import Dict, List, Optional, Tuple

# Import from package to ensure dotenv is loaded
from .. import GOOGLE_API_KEY


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
    context_images: List[str] = None
) -> str:
    """
    Build RAG prompt with diagnosis-focused output format.
    
    Per GPT 5.2 recommendations + diagnosis accuracy fix:
    - Frame as evidence extraction for research (safety filter compliant)
    - EXPLICITLY REQUEST diagnosis prediction (CRITICAL for evaluation)
    - Structure output to include diagnosis type and species prediction
    - FAITHFULNESS ENFORCEMENT: Refuse to guess when evidence insufficient
    - Separate patient images from evidence images (per user feedback)
    """
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n[Context {i}] (Case: {ctx.get('doc_id', 'unknown')})\n"
        context_text += ctx.get("text", "")[:2000]  # Aligned with retrieval truncation
        
        if include_images and ctx.get("image_paths"):
            context_text += f"\n[{len(ctx['image_paths'])} associated medical images]"
    
    # Build image context sections (per user feedback)
    image_sections = ""
    if query_images:
        image_sections += f"\n\n## PATIENT IMAGES (from the case to diagnose)\n"
        image_sections += f"[{len(query_images)} patient image(s) attached - examine these for diagnosis]\n"
    
    if context_images:
        image_sections += f"\n## EVIDENCE IMAGES (from retrieved training cases)\n"
        image_sections += f"[{len(context_images)} supporting image(s) from similar cases for reference]\n"
    
    return f"""You are an AI research assistant helping with an academic thesis on leishmaniasis case reports.

IMPORTANT CONTEXT:
- This is strictly for RESEARCH and EVALUATION purposes
- The data consists of de-identified case reports from PubMed Central (PMC)
- Do NOT provide medical advice or treatment recommendations

RESEARCH QUERY:
{query}
{image_sections}
RETRIEVED CASE REPORT EXCERPTS:
{context_text}

## EVIDENCE PRIORITY INSTRUCTION (AUGMENTATION MODE)
Use the retrieved case excerpts as PRIMARY evidence for your diagnosis.

WHEN RETRIEVED EVIDENCE IS SUFFICIENT:
- Base your diagnosis primarily on the retrieved cases
- Cite retrieved cases using their IDs (e.g., "Case PMC123456")

WHEN RETRIEVED EVIDENCE IS INSUFFICIENT:
- You MAY supplement with your medical knowledge
- Mark such reasoning with: "(based on general medical knowledge)"
- Still provide a diagnosis assessment rather than refusing to answer

This ensures RAG augments rather than replaces your medical expertise.

TASK:
Provide a structured diagnosis assessment, prioritizing retrieved evidence.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Primary Diagnosis:** [State your diagnosis, e.g., "Cutaneous Leishmaniasis", "Visceral Leishmaniasis", "PKDL", "Mucocutaneous Leishmaniasis"]
**Diagnosis Type:** [CL, VL, MCL, PKDL, Other]
**Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Confidence:** [High/Medium/Low based on available evidence]
**Evidence Source:** [Retrieved cases only / Retrieved + general knowledge]

## SUPPORTING EVIDENCE
- Key clinical findings from retrieved cases
- Cite cases using their IDs (e.g., "Case PMC123456")

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider if evidence is limited

DIAGNOSIS ASSESSMENT:"""


class GeminiGenerator:
    """
    Generate answers using Google Gemini API (new google.genai SDK).
    
    Uses the official google.genai SDK with proper safety settings.
    """
    
    def __init__(
        self,
        model: str = None,
        include_images: bool = False,
        temperature: float = 0.3,
        max_tokens: int = 4096  # Increased from 1024 to avoid MAX_TOKENS truncation
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
            "max_output_tokens": max_tokens
        }
    
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
        use_rag_prompt: bool = True  # NEW: allows bypassing RAG prompt for No-RAG baseline
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
        import base64
        from pathlib import Path as P
        
        # Build prompt based on mode
        if use_rag_prompt:
            prompt = build_rag_prompt(query, contexts, self.include_images)
        else:
            # No-RAG mode: use query directly as prompt (already formatted)
            prompt = query
        
        # Build multimodal content list
        contents = []
        
        # Add images if provided (TRUE MULTIMODAL)
        if image_paths:
            valid_images = 0
            for img_path in image_paths[:5]:  # Max 5 images per request
                img_file = P(img_path)
                if img_file.exists():
                    try:
                        # Read and encode image
                        with open(img_file, "rb") as f:
                            img_data = base64.b64encode(f.read()).decode("utf-8")
                        
                        # Determine MIME type
                        suffix = img_file.suffix.lower()
                        mime_map = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg",
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp"
                        }
                        mime_type = mime_map.get(suffix, "image/jpeg")
                        
                        # Add image part using new SDK format
                        contents.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": img_data
                            }
                        })
                        valid_images += 1
                    except Exception as e:
                        print(f"Warning: Could not load image {img_path}: {e}")
            
            if valid_images > 0:
                # Add image context note to prompt
                prompt = f"[{valid_images} medical images included below for reference]\n\n" + prompt
        
        # Add text prompt
        contents.append(prompt)
        
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
