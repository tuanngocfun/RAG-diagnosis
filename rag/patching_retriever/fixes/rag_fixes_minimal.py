#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed version of critical components to address the answer generation issues
"""

import re
import logging
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

def improved_normalize_answer(text: str) -> str:
    """
    Improved answer normalization that better handles the garbage output issues
    """
    if not text:
        return text
    
    t = text.strip()
    
    # Early detection of garbage/repetitive output
    if len(t) > 3000:  # Very long answers are usually garbage
        logger.warning("⚠️ Truncating very long answer (likely repetitive)")
        t = t[:1500] + "..."
    
    # Remove common garbage patterns first
    garbage_patterns = [
        # Remove training data artifacts
        r'\b(?:AIzaSy[\w-]+|gsk_[\w-]+|sk-or-v1-[\w-]+)\b',  # API keys
        r'\b(?:TRANSFORMERS_CACHE|HF_HUB_OFFLINE|PYTHONPATH)\b',  # Environment vars
        r'\b(?:docker run|--env-file|--gpus|127\.0\.0\.1)\b',  # Docker commands
        # Remove repetitive medical term chains
        r'\b(?:medicine|medical|clinical|diagnosis|treatment|therapeutic|pharmacological|epidemiological|pathogenesis|immunology|microscopy|histopathology|dermatologic|gastroenterologists|endocrinology|nephrologists|pulmonologist|allergist|immunologiest|cardiologist|neurologist|psychiatrist|pediatricians|surgeons|anesthesiologists|radiologists|pathologists|microbiologists){5,}',
        # Remove long technical chains
        r'\b(?:physics|mathematics|statistics|probability|calculus|differential|equations|integral|algebra|matrix|operations|vector|calculations|tensor|products|eigenvalue|problems|eigenvector|decomposition|fourier|transform|laplace){3,}',
    ]
    
    for pattern in garbage_patterns:
        t = re.sub(pattern, "", t, flags=re.I)
    
    # Remove obvious case title repetitions more aggressively
    t = re.sub(r'(?:\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+A\s+LESION[^.]*\.?\s*){2,}', 
               "", t, flags=re.I)
    
    # Remove repetitive phrases about case progression
    t = re.sub(r'(?:The lesion progressed quickly from a sore to eat through[^.]*\.?\s*){2,}', 
               "The lesion progressed quickly.", t, flags=re.I)
    
    # Clean up answer prefixes
    t = re.sub(r'^\s*(?:user\s+)?(?:Question\s*:\s*.*?)?\b(?:Answer|ANSWER|Medical Analysis|MEDICAL ANALYSIS)\s*:\s*', 
               "", t, flags=re.I | re.S)
    
    # Handle Final Answer blocks
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail and len(tail) < len(t) * 0.8:  # Only use if it's a reasonable portion
            t = tail
    
    # Remove LaTeX and formatting artifacts
    t = re.sub(r'\\boxed\\{([^}]*)\\}', r'\1', t)
    t = re.sub(r'\$\$?(.*?)\$\$?', r'\1', t, flags=re.S)
    
    # Normalize whitespace
    t = re.sub(r'\s+', ' ', t).strip()
    
    # Enhanced sentence deduplication - handle medical repetition better
    sentences = re.split(r'(?<=[.!?])(?:["\')\]\}]+)?(?:\s+)', t)
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:  # Skip very short fragments
            continue
        
        # Create deduplication key - normalize medical terms
        key = re.sub(r'\W+', '', sentence.lower())
        key = re.sub(r'\d+', 'N', key)  # Normalize numbers
        key = re.sub(r'(leishmaniasis?|leishmania|cutaneous|visceral|mucocutaneous)', 'LEISH', key)
        
        # Skip obvious repetitive case descriptions
        if any(pattern in sentence.lower() for pattern in [
            "year-old boy from laos", "year-old man from cambodia",
            "progressed quickly from a sore", "lesion on the"
        ]):
            continue
        
        # Only add if not seen and substantial
        if key and key not in seen and len(key) > 10:
            seen.add(key)
            unique_sentences.append(sentence)
            
            # Limit total sentences to prevent runaway answers
            if len(unique_sentences) >= 8:
                break
    
    result = " ".join(unique_sentences).strip()
    
    # Final cleanup - remove trailing incomplete sentences
    if result and not result[-1] in '.!?':
        # Find last complete sentence
        last_period = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_period > len(result) * 0.7:  # If last sentence is reasonable length
            result = result[:last_period + 1]
        else:
            result += "."
    
    # Final sanity check - if result is too short or seems broken, provide fallback
    if len(result) < 50 or result.count(' ') < 5:
        return "Unable to generate a clear medical answer from the available information."
    
    return result

def build_focused_medical_prompt(question: str, context: str = "", max_context_length: int = 2000) -> str:
    """
    Build a focused medical prompt that reduces garbage output
    """
    # Detect question type for better prompting
    question_lower = question.lower()
    
    if any(term in question_lower for term in ["cure", "treat", "therapy", "management"]):
        prompt_type = "TREATMENT"
        instruction = "Provide a concise medical treatment answer focusing on therapeutic options and dosages."
    elif any(term in question_lower for term in ["diagnosis", "identify", "what is", "disease"]):
        prompt_type = "DIAGNOSIS"  
        instruction = "Provide a clear diagnostic assessment based on the clinical findings."
    elif any(term in question_lower for term in ["clinical", "presentation", "features", "symptoms"]):
        prompt_type = "CLINICAL"
        instruction = "Describe the relevant clinical features and presentation."
    else:
        prompt_type = "GENERAL"
        instruction = "Provide a focused medical response to the question."
    
    # Build concise context
    if context and len(context) > max_context_length:
        # Prioritize medical content
        sentences = context.split('.')
        medical_sentences = []
        for sent in sentences:
            if any(term in sent.lower() for term in [
                'leishmania', 'amastigote', 'treatment', 'diagnosis', 'lesion', 
                'clinical', 'patient', 'microscopy', 'biopsy'
            ]):
                medical_sentences.append(sent.strip())
        
        context = '. '.join(medical_sentences[:5])  # Limit to 5 relevant sentences
        if context:
            context += '.'
    
    # Create focused prompt
    prompt = f"""Medical Question: {question}

{instruction}

Context: {context}

Please provide a direct, concise answer (2-4 sentences maximum) without repeating case descriptions or including unrelated medical terminology."""

    return prompt

def enhanced_answer_generation(model, question: str, image_paths: List[str], 
                             hits: List[Dict[str, Any]], max_tokens: int = 512) -> str:
    """
    Enhanced answer generation with better prompt engineering and output control
    """
    try:
        # Build focused context from hits
        context_parts = []
        for hit in hits[:3]:  # Limit to top 3 hits
            excerpt = hit.get('text_excerpt', '').strip()
            if excerpt and len(excerpt) > 30:
                # Clean the excerpt
                excerpt = re.sub(r'\s+', ' ', excerpt)
                if len(excerpt) > 200:
                    excerpt = excerpt[:197] + "..."
                context_parts.append(excerpt)
        
        context = " ".join(context_parts)
        
        # Build focused prompt
        focused_prompt = build_focused_medical_prompt(question, context)
        
        # Generate with strict limits to prevent runaway generation
        answer = model.answer(
            focused_prompt,
            [Path(p) for p in image_paths[:2]],  # Limit to 2 images max
            spans=[],  # No spans to avoid regurgitation
            max_output_tokens=min(max_tokens, 256),  # Strict token limit
            context_text="",  # Use prompt context instead
            images_per_answer=min(2, len(image_paths))
        )
        
        # Apply improved normalization
        normalized_answer = improved_normalize_answer(answer)
        
        # Final quality check
        if len(normalized_answer) < 30:
            return "Unable to generate a reliable medical response from the available information."
        
        return normalized_answer
        
    except Exception as e:
        logger.error(f"Enhanced answer generation failed: {e}")
        return "Error generating medical response. Please try again with a different question."

# Export the fixes
__all__ = ['improved_normalize_answer', 'build_focused_medical_prompt', 'enhanced_answer_generation']