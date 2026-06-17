from pipeline.build_nonleish_train_variants import CASE_TEXT_MAX_TOKENS, build_case_text
from pipeline.chunker import chunk_fixed_size, estimate_tokens


def test_build_case_text_prefers_source_narrative_and_keeps_structured_evidence():
    case_record = {
        "prompt1_case_prompt": "Short fallback prompt that should not lead the row.",
        "evidence": "Bone marrow biopsy confirmed Histoplasma capsulatum with PAS-positive organisms.",
        "rationale": "The tissue confirmation and disseminated pattern strongly support histoplasmosis.",
        "primary_disease_mapped": "Disseminated Histoplasmosis",
        "diagnosis_type": "Non-Leishmaniasis",
    }
    source_case = {
        "case_text": "A transplant recipient presented with fever, pancytopenia, and hepatosplenomegaly. Bone marrow evaluation was pursued after persistent clinical decline.",
        "abstract": "Abstract fallback that should not replace the narrative.",
    }

    result = build_case_text(case_record, source_case)

    assert result.text.startswith("Clinical Narrative:")
    assert "A transplant recipient presented with fever" in result.text
    assert "Short fallback prompt" not in result.text
    assert "Key Diagnostic Evidence:" in result.text
    assert "Bone marrow biopsy confirmed Histoplasma capsulatum" in result.text
    assert "Reference Diagnosis:" in result.text
    assert "Diagnosis: Disseminated Histoplasmosis." in result.text
    assert result.sections == [
        "Clinical Narrative",
        "Key Diagnostic Evidence",
        "Reasoning Summary",
        "Reference Diagnosis",
    ]


def test_build_case_text_trims_narrative_before_dropping_evidence():
    long_narrative = " ".join(
        f"Sentence {idx}. Fever, splenomegaly, and pancytopenia persisted despite empiric therapy."
        for idx in range(200)
    )
    case_record = {
        "evidence": "Bone marrow biopsy confirmed intracellular yeasts consistent with Histoplasma capsulatum.",
        "rationale": "Confirmed tissue diagnosis outweighs broad infectious differentials in this immunocompromised host.",
        "primary_disease_mapped": "Disseminated Histoplasmosis",
        "diagnosis_type": "Non-Leishmaniasis",
    }
    source_case = {"case_text": long_narrative}

    result = build_case_text(case_record, source_case)
    chunks = chunk_fixed_size(result.text, "PMC_TEST")

    assert "Key Diagnostic Evidence:" in result.text
    assert "Bone marrow biopsy confirmed intracellular yeasts" in result.text
    assert "Reference Diagnosis:" in result.text
    assert estimate_tokens(result.text) <= CASE_TEXT_MAX_TOKENS
    assert len(chunks) <= 2
