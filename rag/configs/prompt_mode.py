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


RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION = "rag_balanced_image_grounding_v1"
RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES = "explicit_patient_image_grounding_required_when_images_present"
RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION = "rag_q3_nocontext_image_grounding_v1"
RAG_Q3_NOCONTEXT_IMAGE_GROUNDING_PROMPT_CONTRACT_NOTES = "q3_rag_arm_fallback_with_image_grounding_guardrail"


# =============================================================================
# Prompt Templates - EXACT COPIES from original generator files
# =============================================================================

# STRICT_CONTEXT: Exact copy from gemini.py build_rag_prompt() 
# Used by: GeminiGenerator, Gemma3Generator (default)
PROMPT_TEMPLATE_STRICT = """You are an AI research assistant helping with an academic thesis on infectious disease case reports, including leishmaniasis and non-leishmaniasis conditions.

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

CRITICAL DIAGNOSTIC GUARDRAIL:
- Do NOT assume the case is leishmaniasis.
- If evidence does not support leishmaniasis, set primary diagnosis to "Non-Leishmaniasis".

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Rank 1 (Most Likely):** [Diagnosis name]
**Rank 1 Diagnosis Type:** [CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other]
**Rank 1 Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Rank 1 Confidence:** [High/Medium/Low]
**Rank 2:** [Diagnosis name]
**Rank 3:** [Diagnosis name]
**Chosen Final Diagnosis for Scoring:** [Repeat Rank 1 diagnosis text]
**Evidence Source:** [Retrieved cases only / Retrieved + general knowledge]

## SUPPORTING EVIDENCE
- Key clinical findings from retrieved cases
- Cite cases using their IDs (e.g., "Case PMC123456")

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider if evidence is limited

DIAGNOSIS ASSESSMENT:"""

# BALANCED: Exact copy from medgemma.py build_rag_prompt()
# Used by: MedGemmaGenerator (default)
PROMPT_TEMPLATE_BALANCED = """You are a medical expert in dermatology and infectious disease differential diagnosis.

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
- Use retrieved cases to ground your reasoning, not just as optional background
- Explicitly reconcile query findings against retrieved cases when they appear to conflict
- Treat definitive parasite/pathology evidence (amastigotes, biopsy confirmation, positive confirmatory PCR/smear) as stronger than negative serology alone
- Do NOT rank Non-Leishmaniasis as Rank 1 if the query includes definitive leishmaniasis evidence unless you clearly explain why that evidence is unreliable or belongs only to a retrieved reference case
- Keep the final answer concise and non-repetitive; avoid generic filler or repeated bullet points

## VISUAL GROUNDING REQUIREMENT
- If patient images are attached, you MUST explicitly ground your Rank 1 reasoning in at least one concrete patient-image finding
- Distinguish patient-image findings from retrieved reference/supporting images; do not blur the two sources
- Do not invent visual details that are not visible in the attached patient image(s) or not stated in the query
- If the patient image is low-quality, nondiagnostic, or visually ambiguous, say that explicitly instead of pretending the image is decisive

TASK:
Provide a structured diagnosis assessment.

CRITICAL DIAGNOSTIC GUARDRAIL:
- Do NOT assume the case is leishmaniasis.
- Use "Non-Leishmaniasis" when evidence favors alternative etiologies.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Rank 1 (Most Likely):** [Diagnosis name]
**Rank 1 Diagnosis Type:** [CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other]
**Rank 1 Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Rank 1 Confidence:** [High/Medium/Low]
**Rank 2:** [Diagnosis name]
**Rank 3:** [Diagnosis name]
**Chosen Final Diagnosis for Scoring:** [Repeat Rank 1 diagnosis text]
**Evidence Source:** [Medical knowledge / Reference cases / Both]

## SUPPORTING EVIDENCE
- Provide 3-5 concise bullets only
- Prioritize definitive patient findings first, then supporting reference cases (e.g., "Similar to Case PMC123456")
- If retrieved evidence conflicts with your final diagnosis, explain the conflict briefly instead of ignoring it

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider

DIAGNOSIS ASSESSMENT:"""

# NO_CONTEXT: For no-RAG baseline ablation
PROMPT_TEMPLATE_NO_CONTEXT = """You are a medical expert in dermatology and infectious disease differential diagnosis.

IMPORTANT: This is for RESEARCH and EVALUATION purposes only.

CLINICAL QUERY:
{query}
{images_section}
## INSTRUCTIONS (NO-RAG CONTROL)
Use only the clinical query, attached patient images, and your medical knowledge.
Do not claim support from retrieved cases, literature excerpts, or unseen documents.
Treat definitive parasite/pathology evidence stated in the query (for example amastigotes, biopsy confirmation, marrow confirmation, or a positive confirmatory PCR/smear) as stronger than negative serology alone.
Do not rank Non-Leishmaniasis as Rank 1 if the query itself contains definitive leishmaniasis evidence unless you clearly explain why that evidence is unreliable, non-confirmatory, or outweighed by stronger contradictory findings.
Keep the final answer concise and non-repetitive; avoid generic filler or repeated bullet points.

TASK:
Using your medical knowledge, provide a structured diagnosis assessment.

CRITICAL DIAGNOSTIC GUARDRAIL:
- Do NOT assume the case is leishmaniasis.
- Use "Non-Leishmaniasis" when evidence does not support leishmaniasis.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Rank 1 (Most Likely):** [Diagnosis name]
**Rank 1 Diagnosis Type:** [CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other]
**Rank 1 Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Rank 1 Confidence:** [High/Medium/Low]
**Rank 2:** [Diagnosis name]
**Rank 3:** [Diagnosis name]
**Chosen Final Diagnosis for Scoring:** [Repeat Rank 1 diagnosis text]
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
