"""
Prompt Mode: Configuration for RAG prompt strategies.

Per GPT 5.2 recommendation:
- strict_context: "Only use context, ignore prior knowledge" (current baseline, harms MedGemma)
- balanced: "Use context as evidence, but can use prior knowledge with marking" (recommended)
- no_context: "Same prompt format but empty contexts" (control for ablation)

Key insight: MedGemma performs worse with RAG (-6%) because strict_context
forces it to override its strong medical parametric knowledge.

IMPORTANT: These templates are exact copies of what was in gemini.py and medgemma.py
to ensure NO BEHAVIOR CHANGE during refactoring.
"""
from enum import Enum
from typing import Optional, List, Dict


class PromptMode(str, Enum):
    """Available prompt modes for RAG integration."""
    STRICT_CONTEXT = "strict_context"  # Force context override (current baseline - gemini/gemma3)
    BALANCED = "balanced"              # Synergy between context and parametric (medgemma)
    NO_CONTEXT = "no_context"          # Control: same format, empty contexts


# =============================================================================
# Prompt Templates - EXACT COPIES from original generator files
# =============================================================================

# STRICT_CONTEXT: Exact copy from gemini.py build_rag_prompt() 
# Used by: GeminiGenerator, Gemma3Generator (default)
PROMPT_TEMPLATE_STRICT = """You are an AI research assistant helping with an academic thesis on leishmaniasis case reports.

IMPORTANT CONTEXT:
- This is strictly for RESEARCH and EVALUATION purposes
- The data consists of de-identified case reports from PubMed Central (PMC)
- Do NOT provide medical advice or treatment recommendations

RESEARCH QUERY:
{query}
{images_section}
RETRIEVED CASE REPORT EXCERPTS:
{contexts}

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

# BALANCED: Exact copy from medgemma.py build_rag_prompt()
# Used by: MedGemmaGenerator (default)
PROMPT_TEMPLATE_BALANCED = """You are a medical expert specializing in Leishmaniasis diagnosis.

IMPORTANT: This is for RESEARCH and EVALUATION purposes only.

CLINICAL QUERY:
{query}
{images_section}
REFERENCE CASES FROM LITERATURE:
{contexts}

## INSTRUCTIONS (BALANCED RAG)
You have access to both:
1. Your medical training and knowledge of Leishmaniasis
2. Reference cases retrieved from published case reports

CONSIDER the reference cases as SUPPORTING EVIDENCE:
- If reference cases align with your medical knowledge, cite them to support your diagnosis
- If reference cases seem to conflict, use your medical expertise to reconcile or note the discrepancy
- Your diagnosis should reflect your best clinical judgment, informed by both sources

TASK:
Provide a structured diagnosis assessment.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Primary Diagnosis:** [State your diagnosis, e.g., "Cutaneous Leishmaniasis", "Visceral Leishmaniasis", "PKDL", "Mucocutaneous Leishmaniasis"]
**Diagnosis Type:** [CL, VL, MCL, PKDL, Other]
**Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Confidence:** [High/Medium/Low based on available evidence]
**Evidence Source:** [Medical knowledge / Reference cases / Both]

## SUPPORTING EVIDENCE
- Key clinical findings supporting your diagnosis
- Reference cases if they support your conclusion (e.g., "Similar to Case PMC123456")

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider

DIAGNOSIS ASSESSMENT:"""

# NO_CONTEXT: For no-RAG baseline ablation
PROMPT_TEMPLATE_NO_CONTEXT = """You are a medical expert specializing in Leishmaniasis diagnosis.

IMPORTANT: This is for RESEARCH and EVALUATION purposes only.

CLINICAL QUERY:
{query}
{images_section}
TASK:
Using your medical knowledge, provide a structured diagnosis assessment.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Primary Diagnosis:** [State your diagnosis, e.g., "Cutaneous Leishmaniasis", "Visceral Leishmaniasis", "PKDL", "Mucocutaneous Leishmaniasis"]
**Diagnosis Type:** [CL, VL, MCL, PKDL, Other]
**Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Confidence:** [High/Medium/Low based on available evidence]
**Evidence Source:** [Medical knowledge]

## SUPPORTING EVIDENCE
- Key clinical findings supporting your diagnosis

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider

DIAGNOSIS ASSESSMENT:"""

PROMPT_TEMPLATES = {
    PromptMode.STRICT_CONTEXT: PROMPT_TEMPLATE_STRICT,
    PromptMode.BALANCED: PROMPT_TEMPLATE_BALANCED,
    PromptMode.NO_CONTEXT: PROMPT_TEMPLATE_NO_CONTEXT,
}


def format_contexts(
    contexts: List[Dict],
    max_chars_per_context: int = 2000,
    include_images: bool = True
) -> str:
    """
    Format retrieved contexts for prompt insertion.
    
    EXACT formatting from gemini.py/gemma3.py/medgemma.py build_rag_prompt().
    """
    if not contexts:
        return "No relevant case excerpts retrieved."
    
    context_text = ""
    for i, ctx in enumerate(contexts, 1):
        context_text += f"\n[Context {i}] (Case: {ctx.get('doc_id', 'unknown')})\n"
        context_text += ctx.get("text", "")[:max_chars_per_context]
        
        # Include image reference if available (from gemini.py/gemma3.py)
        if include_images and ctx.get("image_paths"):
            context_text += f"\n[{len(ctx['image_paths'])} associated medical images]"
    
    return context_text


def format_images_section(
    query_images: Optional[List[str]] = None,
    context_images: Optional[List[str]] = None,
    is_text_only_model: bool = False
) -> str:
    """
    Format images section for prompt.
    
    EXACT formatting from gemini.py/gemma3.py/medgemma.py build_rag_prompt().
    """
    image_sections = ""
    
    if query_images:
        image_sections += f"\n\n## PATIENT IMAGES (from the case to diagnose)\n"
        if is_text_only_model:
            image_sections += f"[{len(query_images)} patient image(s) - model is text-only, clinical descriptions used]\n"
        else:
            image_sections += f"[{len(query_images)} patient image(s) attached - examine these for diagnosis]\n"
    
    if context_images:
        image_sections += f"\n## EVIDENCE IMAGES (from retrieved training cases)\n"
        image_sections += f"[{len(context_images)} supporting image(s) from similar cases for reference]\n"
    
    return image_sections


def get_prompt_template(mode: PromptMode) -> str:
    """Get prompt template for given mode."""
    return PROMPT_TEMPLATES[mode]


def build_rag_prompt(
    query: str,
    contexts: List[Dict],
    mode: PromptMode = PromptMode.STRICT_CONTEXT,
    query_images: Optional[List[str]] = None,
    context_images: Optional[List[str]] = None,
    max_chars_per_context: int = 2000,
    include_context_images: bool = True,
    is_text_only_model: bool = False
) -> str:
    """
    Build RAG prompt with given mode.
    
    This is the SINGLE SOURCE OF TRUTH for prompt building.
    All generators should use this function.
    
    Args:
        query: Clinical question
        contexts: Retrieved context dictionaries
        mode: Prompt mode (strict_context/balanced/no_context)
        query_images: Optional list of query image paths
        context_images: Optional list of context image paths
        max_chars_per_context: Max chars per context to avoid truncation
        include_context_images: Whether to include image references in context
        is_text_only_model: If True, note that model can't process images
        
    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(mode)
    
    # For no_context mode, use empty contexts
    if mode == PromptMode.NO_CONTEXT:
        contexts_str = ""
    else:
        contexts_str = format_contexts(
            contexts,
            max_chars_per_context=max_chars_per_context,
            include_images=include_context_images
        )
    
    images_section = format_images_section(
        query_images=query_images,
        context_images=context_images,
        is_text_only_model=is_text_only_model
    )
    
    return template.format(
        contexts=contexts_str,
        images_section=images_section,
        query=query
    )


# =============================================================================
# Mode Descriptions (for documentation/logging)
# =============================================================================

PROMPT_MODE_DESCRIPTIONS = {
    PromptMode.STRICT_CONTEXT: (
        "Forces model to use ONLY retrieved contexts, ignoring prior knowledge. "
        "May harm specialized medical models (e.g., MedGemma) that have strong "
        "parametric knowledge. Use for testing retrieval quality."
    ),
    PromptMode.BALANCED: (
        "Allows model to integrate retrieved evidence with prior medical knowledge. "
        "Recommended for specialized medical models. Prior knowledge must be marked. "
        "Best for production use with medical SLMs."
    ),
    PromptMode.NO_CONTEXT: (
        "Control condition: same prompt format but no contexts provided. "
        "Use for ablation studies comparing RAG vs no-RAG performance."
    )
}
