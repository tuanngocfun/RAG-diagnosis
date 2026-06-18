from medical_demo_backend.generator import RealGemma4Generator


def test_rank_parser_ignores_rank_metadata_lines():
    answer = """
## DIAGNOSIS PREDICTION
**Rank 1 (Most Likely):** Cutaneous Leishmaniasis
**Rank 1 Diagnosis Type:** CL
**Rank 1 Species (if determinable):** Not determinable
**Rank 1 Confidence:** High
**Rank 2:** Non-Leishmaniasis (Bacterial/Fungal infection)
**Rank 3:** Other parasitic infection
**Chosen Final Diagnosis for Scoring:** Cutaneous Leishmaniasis
"""

    assert RealGemma4Generator._parse_ranked_differential(answer) == [
        "Cutaneous Leishmaniasis",
        "Non-Leishmaniasis",
        "Other parasitic infection",
    ]


def test_confidence_parser_handles_markdown_rank_confidence():
    answer = "**Rank 1 Confidence:** High"

    assert RealGemma4Generator._parse_confidence(answer) == "high"
