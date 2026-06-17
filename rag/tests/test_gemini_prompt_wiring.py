from configs.prompt_mode import PromptMode
from pipeline.generators.gemini import GeminiGenerator


def _make_generator(prompt_mode: PromptMode, include_images: bool = True) -> GeminiGenerator:
    generator = GeminiGenerator.__new__(GeminiGenerator)
    generator.prompt_mode = prompt_mode
    generator.include_images = include_images
    return generator


def test_gemini_build_contents_uses_balanced_prompt_and_preserves_schema():
    generator = _make_generator(PromptMode.BALANCED)

    contents = generator._build_contents(
        query="Provide a ranked differential diagnosis.\n\nClinical Context: Pancytopenia and splenomegaly.",
        contexts=[{"doc_id": "PMC1", "text": "Example retrieved case text."}],
        query_images=[],
        context_images=[],
        use_rag_prompt=True,
    )

    prompt = contents[-1]
    assert "## INSTRUCTIONS (BALANCED RAG)" in prompt
    assert "REFERENCE CASES FROM LITERATURE" in prompt
    assert "Chosen Final Diagnosis for Scoring" in prompt
    assert "Evidence Source" in prompt
    assert "## SUPPORTING EVIDENCE" in prompt
    assert "## DIFFERENTIAL CONSIDERATIONS" in prompt
    assert "## EVIDENCE PRIORITY INSTRUCTION (AUGMENTATION MODE)" not in prompt


def test_gemini_build_contents_respects_no_context_passthrough_for_norag():
    generator = _make_generator(PromptMode.NO_CONTEXT)
    no_rag_prompt = "## INSTRUCTIONS (NO-RAG CONTROL)\nChosen Final Diagnosis for Scoring\nEvidence Source"

    contents = generator._build_contents(
        query=no_rag_prompt,
        contexts=[],
        query_images=[],
        context_images=[],
        use_rag_prompt=False,
    )

    prompt = contents[-1]
    assert prompt == no_rag_prompt
    assert "## INSTRUCTIONS (NO-RAG CONTROL)" in prompt
    assert "REFERENCE CASES FROM LITERATURE" not in prompt
