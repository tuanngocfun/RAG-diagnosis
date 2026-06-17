from medical_demo_backend.generator import DemoGemma4Generator
from medical_demo_backend.kb import KnowledgeBase
from medical_demo_backend.service import ConsultService
from medical_demo_backend.types import ConsultationRequest


def _service() -> ConsultService:
    from pathlib import Path

    kb = KnowledgeBase.from_path(
        Path(__file__).resolve().parents[2] / "kb" / "leishmaniasis_demo_pack.json"
    )
    return ConsultService(kb, DemoGemma4Generator())


def test_weak_retrieval_enters_parametric_fallback():
    response = _service().consult(
        ConsultationRequest(
            patient_text="Chronic skin lesion with ulcerated border after travel to an endemic region.",
            client_request_id="case-weak",
        )
    )
    assert response.decision_state == "provisional_parametric"
    assert response.uncertainty_gate["stage"] == "parametric_fallback"
    assert response.evidence == []


def test_low_confidence_fallback_abstains():
    response = _service().consult(
        ConsultationRequest(
            patient_text="Rash.",
            client_request_id="case-abstain",
        )
    )
    assert response.decision_state == "abstained"
    assert response.safe_to_show_ranked_differential is False
    assert response.top_diagnoses == []


def test_supported_rag_returns_evidence():
    response = _service().consult(
        ConsultationRequest(
            patient_text=(
                "Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes."
            ),
            client_request_id="case-rag",
        )
    )
    assert response.decision_state == "rag_supported"
    assert response.safe_to_show_ranked_differential is True
    assert len(response.evidence) >= 1
    assert response.uncertainty_gate["retrieval_support_status"] == "supported"


def test_abstained_response_hides_ranked_differential():
    response = _service().consult(
        ConsultationRequest(
            patient_text="",
            client_request_id="case-no-input",
        )
    )
    assert response.decision_state == "abstained"
    assert response.safe_to_show_ranked_differential is False
    assert response.top_diagnoses == []


def test_chat_answer_safety_frame_rewrites_benchmark_labels():
    answer = _service()._safety_frame_chat_answer(
        "## DIAGNOSIS PREDICTION\n"
        "**Rank 1 (Most Likely):** Cutaneous leishmaniasis\n"
        "**Chosen Final Diagnosis for Scoring:** Cutaneous leishmaniasis"
    )

    assert "Safety boundary" in answer
    assert "Supportive Differential (Not a Diagnosis)" in answer
    assert "Rank 1 supportive consideration" in answer
    assert "Most supported option in this research demo" in answer
    assert "Chosen Final Diagnosis for Scoring" not in answer
