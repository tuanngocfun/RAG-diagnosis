"""
Standardized Query Templates for RAGAS Evaluation

Based on best practices from:
- RadLE: Expert persona, specific diagnosis output
- GaRAGe: Grounding constraints, deflection handling
- Lessons Learned (EHR IR): Fixed question phrasing
- Grok 4.1: Step-by-step reasoning instruction

Usage:
    from pipeline.query_templates import build_diagnosis_query, SYSTEM_PROMPT
    
    query = build_diagnosis_query(
        clinical_description="45-year-old male with erythematous plaques...",
        has_image=True
    )
"""

from typing import Optional, Dict, Any
from dataclasses import dataclass


# =============================================================================
# SYSTEM PROMPT (Fixed across all evaluation runs)
# =============================================================================

SYSTEM_PROMPT = """You are a board-certified dermatologist and infectious disease specialist.

Your task is to analyze the patient presentation and any retrieved medical context to provide a diagnosis.

**CONSTRAINTS:**
1. Base your diagnosis ONLY on the provided clinical information and retrieved context
2. If the provided information is insufficient for a confident diagnosis, respond with "INSUFFICIENT" and list what additional information would be needed
3. Provide step-by-step clinical reasoning before stating your final diagnosis
4. Do not assume information not explicitly stated in the case

**OUTPUT FORMAT:**
Provide your response as valid JSON with this structure:
{
  "diagnosis": "<single specific diagnosis>",
  "leish_type": "<CL|VL|MCL|PKDL|unknown>",
  "confidence": "<high|medium|low>",
  "key_evidence": ["evidence 1", "evidence 2", "..."],
  "reasoning": "<step-by-step clinical reasoning>",
  "insufficient_reason": null or "<what is missing if confidence is low>"
}
"""

# =============================================================================
# FIXED QUESTION (Identical across all cases for metric stability)
# =============================================================================

DIAGNOSIS_QUESTION = "What is the most likely diagnosis for this patient?"

DIAGNOSIS_QUESTION_WITH_TYPE = (
    "What is the most likely diagnosis for this patient? "
    "If leishmaniasis, specify the clinical form (CL, VL, MCL, PKDL)."
)


# =============================================================================
# QUERY TEMPLATES
# =============================================================================

# Template for text-only queries (Q1)
TEXT_ONLY_TEMPLATE = """**PATIENT CASE:**
Clinical presentation: {clinical_description}

**QUESTION:** {question}
"""

# Template for multimodal queries (Q1+Q3 combined)
MULTIMODAL_TEMPLATE = """**PATIENT CASE:**
Clinical presentation: {clinical_description}

**CLINICAL IMAGE:** [Attached as multimodal input]

**QUESTION:** {question}
"""

# Template for image-only queries (Q3)
IMAGE_ONLY_TEMPLATE = """**CLINICAL IMAGE:** [Attached as multimodal input]

Based on the clinical image alone, {question}
"""


# =============================================================================
# QUERY BUILDER FUNCTIONS
# =============================================================================

@dataclass
class StandardizedQuery:
    """A standardized query for evaluation."""
    query_text: str
    question_only: str  # Just the question (for RAGAS query field)
    full_prompt: str    # Full prompt with context (for generation)
    query_type: str
    has_image: bool


def build_diagnosis_query(
    clinical_description: str,
    has_image: bool = False,
    image_only: bool = False,
    include_leish_type: bool = True
) -> StandardizedQuery:
    """
    Build a standardized diagnosis query.
    
    Args:
        clinical_description: Clean case text (without disease name)
        has_image: Whether clinical image is attached
        image_only: Whether this is an image-only query (Q3)
        include_leish_type: Whether to ask for Leishmania type specification
    
    Returns:
        StandardizedQuery with consistent formatting
    """
    question = DIAGNOSIS_QUESTION_WITH_TYPE if include_leish_type else DIAGNOSIS_QUESTION
    
    if image_only:
        template = IMAGE_ONLY_TEMPLATE
        query_type = "Q3_image_diagnosis"
        query_text = template.format(question=question)
    elif has_image:
        template = MULTIMODAL_TEMPLATE
        query_type = "Q1_Q3_multimodal_diagnosis"
        query_text = template.format(
            clinical_description=clinical_description,
            question=question
        )
    else:
        template = TEXT_ONLY_TEMPLATE
        query_type = "Q1_text_diagnosis"
        query_text = template.format(
            clinical_description=clinical_description,
            question=question
        )
    
    return StandardizedQuery(
        query_text=query_text,
        question_only=question,
        full_prompt=query_text,
        query_type=query_type,
        has_image=has_image or image_only
    )


def build_rag_prompt(
    query: StandardizedQuery,
    retrieved_contexts: list,
    system_prompt: str = SYSTEM_PROMPT
) -> Dict[str, Any]:
    """
    Build the full RAG prompt for generation.
    
    Args:
        query: StandardizedQuery from build_diagnosis_query
        retrieved_contexts: List of retrieved text contexts
        system_prompt: System instructions (use default SYSTEM_PROMPT)
    
    Returns:
        Dict with 'system', 'user', and 'contexts' keys
    """
    # Format retrieved contexts
    context_str = "\n\n".join([
        f"[Context {i+1}]: {ctx}" 
        for i, ctx in enumerate(retrieved_contexts)
    ])
    
    user_prompt = f"""**RETRIEVED MEDICAL CONTEXT:**
{context_str}

---

{query.query_text}
"""
    
    return {
        "system": system_prompt,
        "user": user_prompt,
        "contexts": retrieved_contexts,
        "question_for_ragas": query.question_only
    }


# =============================================================================
# SANITIZATION (Remove disease name leakage)
# =============================================================================

import re

LEAK_PATTERNS = [
    # Leishmania species
    r"leishmania\s*\w*",
    r"l\.\s*(donovani|infantum|tropica|major|braziliensis|mexicana|amazonensis)",
    # Disease forms
    r"visceral\s*leishmaniasis",
    r"cutaneous\s*leishmaniasis",
    r"mucocutaneous\s*leishmaniasis",
    r"post.?kala.?azar",
    r"pkdl",
    r"kala.?azar",
    # Generic diagnosis terms
    r"leishmaniasis",
    r"leishmanial",
]

LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


def sanitize_case_text(text: str) -> str:
    """
    Remove disease name mentions to prevent answer leakage.
    
    Args:
        text: Raw case text
    
    Returns:
        Cleaned text with disease names removed
    """
    return LEAK_REGEX.sub("[CONDITION]", text)


# =============================================================================
# CONVENIENCE FUNCTION FOR MIGRATION
# =============================================================================

def migrate_old_query(old_query: str, old_query_type: str) -> StandardizedQuery:
    """
    Convert old-style queries to new standardized format.
    
    Args:
        old_query: Old query text (e.g., "Patient with extensive plaques...")
        old_query_type: Old query type (e.g., "Q1_symptom_only")
    
    Returns:
        StandardizedQuery in new format
    """
    # Clean the old query
    if old_query.startswith("Patient with "):
        clinical_description = old_query[13:]  # Remove "Patient with "
    elif old_query.startswith("Patient"):
        clinical_description = old_query[7:].strip()  # Remove "Patient"
    else:
        clinical_description = old_query
    
    # Determine if image query
    is_image = "Q3" in old_query_type or old_query.startswith("[IMAGE:")
    image_only = old_query_type == "Q3_image_only"
    
    # Sanitize
    clinical_description = sanitize_case_text(clinical_description)
    
    return build_diagnosis_query(
        clinical_description=clinical_description,
        has_image=is_image,
        image_only=image_only
    )


if __name__ == "__main__":
    # Demo usage
    print("=== Standardized Query Templates ===\n")
    
    # Text-only query
    q1 = build_diagnosis_query(
        clinical_description="45-year-old male with extensive erythematous plaques on forehead and face",
        has_image=False
    )
    print(f"Q1 (Text-only):\n{q1.query_text}\n")
    
    # Multimodal query
    q2 = build_diagnosis_query(
        clinical_description="7-year-old cat with facial swelling and eye lesions",
        has_image=True
    )
    print(f"Q1+Q3 (Multimodal):\n{q2.query_text}\n")
    
    # Image-only query
    q3 = build_diagnosis_query(
        clinical_description="",
        image_only=True
    )
    print(f"Q3 (Image-only):\n{q3.query_text}\n")
    
    print(f"Question for RAGAS: {q1.question_only}")
