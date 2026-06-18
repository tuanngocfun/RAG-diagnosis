from configs.prompt_mode import PromptMode, build_rag_prompt
from pipeline.run_baseline_norag import build_norag_prompt


def test_no_context_prompt_includes_query_only_guardrail():
    prompt = build_rag_prompt(
        query="Clinical Context: Bone marrow aspirate showed amastigotes despite negative serology.",
        contexts=[],
        mode=PromptMode.NO_CONTEXT,
        query_images=["/tmp/query-image.jpg"],
    )

    assert "## INSTRUCTIONS (NO-RAG CONTROL)" in prompt
    assert "Treat definitive parasite/pathology evidence stated in the query" in prompt
    assert "Do not rank Non-Leishmaniasis as Rank 1 if the query itself contains definitive leishmaniasis evidence" in prompt
    assert "REFERENCE CASES FROM LITERATURE" not in prompt


def test_build_norag_prompt_uses_shared_no_context_contract():
    prompt = build_norag_prompt(
        query_text="Provide a ranked differential diagnosis.\n\nClinical Context: Pancytopenia and marrow amastigotes.",
        query_images=["/tmp/query-image.jpg"],
    )

    assert "## INSTRUCTIONS (NO-RAG CONTROL)" in prompt
    assert "Do not claim support from retrieved cases" in prompt
    assert "REFERENCE CASES FROM LITERATURE" not in prompt
    assert "Chosen Final Diagnosis for Scoring" in prompt
    assert "Evidence Source" in prompt
    assert "## SUPPORTING EVIDENCE" in prompt
    assert "## DIFFERENTIAL CONSIDERATIONS" in prompt
