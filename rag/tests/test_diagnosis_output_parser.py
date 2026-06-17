from pipeline.diagnosis_output_parser import analyze_answer_format, compute_family_metric_details


def test_cutaneous_rank1_markdown_is_parseable():
    analysis = analyze_answer_format("**Rank 1: Cutaneous Leishmaniasis (CL)**")
    assert analysis.answer_format_valid is True
    assert analysis.rank1_diagnosis_type == "CL"
    assert analysis.diagnosis_family == "cutaneous_family"


def test_visceral_inference_from_rank1_text():
    analysis = analyze_answer_format("**Rank 1: Leishmaniasis (likely visceral leishmaniasis - VL)**")
    assert analysis.answer_format_valid is True
    assert analysis.rank1_diagnosis_type == "VL"
    assert analysis.diagnosis_family == "visceral_family"


def test_inline_markdown_rank1_is_parseable():
    analysis = analyze_answer_format("Rank 1: **Cutaneous Leishmaniasis (CL)**")
    assert analysis.answer_format_valid is True
    assert analysis.rank1_diagnosis_type == "CL"


def test_nonleish_with_cl_abbreviation_stays_nonleish():
    analysis = analyze_answer_format("**Rank 1: Sporotrichosis (CL)**")
    assert analysis.answer_format_valid is True
    assert analysis.rank1_diagnosis_type == "Non-Leishmaniasis"
    assert analysis.diagnosis_family == "nonleish_family"


def test_malformed_output_is_flagged():
    analysis = analyze_answer_format("The diagnosis is probably leishmaniasis")
    assert analysis.answer_format_valid is False
    assert analysis.answer_format_error == "missing_rank1_line"


def test_family_metric_uses_rank1_text_fallback():
    details = compute_family_metric_details(
        "**Rank 1: Cutaneous Leishmaniasis (CL)**",
        {"diagnosis_type": "CL"},
    )
    assert details["diagnosis_family"] == "cutaneous_family"
    assert details["diagnosis_family_accuracy"] == 1.0
