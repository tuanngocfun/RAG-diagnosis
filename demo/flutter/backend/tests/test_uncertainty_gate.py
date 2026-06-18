from medical_demo_backend.kb import KnowledgeBase
from medical_demo_backend.uncertainty_gate import evaluate_evidence_gate, evaluate_parametric_gate
from medical_demo_backend.generator import DemoGemma4Generator


def _kb() -> KnowledgeBase:
    from pathlib import Path

    return KnowledgeBase.from_path(
        Path(__file__).resolve().parents[2] / "kb" / "leishmaniasis_demo_pack.json"
    )


def test_supported_evidence_allows_rag_path():
    contexts = _kb().search(
        "Ulcerated plaque on the arm after sandfly exposure with amastigotes seen on smear.",
        top_k=4,
    )
    gate = evaluate_evidence_gate(
        "Ulcerated plaque on the arm after sandfly exposure with amastigotes seen on smear.",
        None,
        contexts,
    )
    assert gate.outcome == "rag_supported"
    assert gate.retrieval_support_status == "supported"


def test_conflicting_evidence_triggers_abstention():
    contexts = _kb().search(
        "Patient has ulcerative skin lesions, prolonged fever, splenomegaly, and both cutaneous and visceral clues.",
        top_k=4,
    )
    gate = evaluate_evidence_gate(
        "Patient has ulcerative skin lesions, prolonged fever, splenomegaly, and both cutaneous and visceral clues.",
        None,
        contexts,
    )
    assert gate.outcome == "hard_abstain"
    assert "evidence_conflict" in gate.gate_trigger_codes


def test_parametric_fallback_low_confidence_becomes_abstain():
    evidence_gate = evaluate_evidence_gate(
        "Vague rash with no travel history provided.",
        None,
        _kb().search("Vague rash with no travel history provided.", top_k=4),
    )
    output = DemoGemma4Generator().generate("Vague rash with no travel history provided.", [], use_rag=False)
    gate = evaluate_parametric_gate(evidence_gate, output)
    assert gate.outcome == "hard_abstain"
    assert "low_model_confidence" in gate.gate_trigger_codes
