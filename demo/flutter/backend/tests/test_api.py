import io
import json
from pathlib import Path

from medical_demo_backend.api import create_app
from medical_demo_backend.kb import KnowledgeBase
from medical_demo_backend.service import ConsultService
from medical_demo_backend.types import GeneratorOutput


def _invoke_json(path: str, payload: dict, method: str = "POST", app=None):
    body = json.dumps(payload).encode("utf-8")
    app = app or create_app()
    environ = {
        "REQUEST_METHOD": method,
        "PATH_INFO": path,
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": str(len(body)),
        "wsgi.input": io.BytesIO(body),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    response_body = b"".join(app(environ, start_response))
    return captured["status"], json.loads(response_body.decode("utf-8"))


class _FakeRealGenerator:
    model_name = "google/gemma-4-E4B-it"

    def __init__(self, *, error: bool = False, error_string: bool = False):
        self.error = error
        self.error_string = error_string
        self.calls = []

    def generate(self, patient_text, contexts, use_rag, image_bytes=None, image_filename=""):
        self.calls.append(
            {
                "patient_text": patient_text,
                "contexts": contexts,
                "use_rag": use_rag,
                "image_bytes": image_bytes,
                "image_filename": image_filename,
            }
        )
        if self.error:
            raise RuntimeError("fake GPU failure")
        if self.error_string:
            return GeneratorOutput(
                diagnosis="INSUFFICIENT",
                confidence="low",
                ranked_differential=[],
                reasoning="generation error",
                insufficient_reason="generation error",
                needed_next_inputs=["retry"],
                answer_markdown="[Generation Error: CUDA out of memory]",
                runtime_metadata={"gpu_name": "Fake GPU"},
            )
        return GeneratorOutput(
            diagnosis="Cutaneous leishmaniasis",
            confidence="medium",
            ranked_differential=["Cutaneous leishmaniasis"],
            reasoning="The case pattern overlaps retrieved cutaneous leishmaniasis evidence.",
            insufficient_reason=None,
            needed_next_inputs=["confirmatory smear, biopsy, or PCR", "clinician review"],
            answer_markdown=(
                "## Supportive differential\n"
                "- Cutaneous leishmaniasis could be considered.\n\n"
                "## Reasoning\n"
                "This is decision support only, not a diagnosis or ground truth."
            ),
            runtime_metadata={
                "gpu_name": "Fake GPU",
                "quantization_mode": "4-bit",
                "generation_latency_seconds": 0.1,
            },
        )


def _real_gpu_app(fake_generator):
    kb = KnowledgeBase.from_path(
        Path(__file__).resolve().parents[2] / "kb" / "leishmaniasis_demo_pack.json"
    )
    service = ConsultService(
        kb,
        fake_generator,
        provider_mode="real_gpu_gemma4",
    )
    return create_app(service=service)


def test_root_endpoint_lists_valid_demo_urls():
    app = create_app()
    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": "0",
        "wsgi.input": io.BytesIO(b""),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    payload = b"".join(app(environ, start_response))
    body = payload.decode("utf-8")
    headers = dict(captured["headers"])
    assert captured["status"] == "200 OK"
    assert headers["Content-Type"] == "text/html; charset=utf-8"
    assert "/health" in body
    assert "POST /v1/chat" in body
    assert "http://127.0.0.1:8021" in body
    assert "REAL_CASES_AUDIENCE_VIEW.html" in body


def test_health_endpoint():
    app = create_app()
    environ = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/health",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": "0",
        "wsgi.input": io.BytesIO(b""),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    payload = b"".join(app(environ, start_response))
    body = json.loads(payload.decode("utf-8"))
    assert captured["status"] == "200 OK"
    assert body["status"] == "ok"
    assert body["provider_mode"] == "deterministic_demo"
    assert "chat_available" in body
    assert "cuda_available" in body
    assert "gpu_name" in body
    assert "gpu_memory_total_mib" in body
    assert "gpu_memory_free_mib" in body
    headers = dict(captured["headers"])
    assert headers["Access-Control-Allow-Origin"] == "*"


def test_options_preflight_for_consult_endpoint():
    app = create_app()
    environ = {
        "REQUEST_METHOD": "OPTIONS",
        "PATH_INFO": "/v1/consult",
        "CONTENT_TYPE": "application/json",
        "CONTENT_LENGTH": "0",
        "wsgi.input": io.BytesIO(b""),
    }
    captured = {}

    def start_response(status, headers):
        captured["status"] = status
        captured["headers"] = headers

    payload = b"".join(app(environ, start_response))
    headers = dict(captured["headers"])
    assert captured["status"] == "204 No Content"
    assert payload == b""
    assert headers["Access-Control-Allow-Origin"] == "*"
    assert "POST" in headers["Access-Control-Allow-Methods"]
    assert "Content-Type" in headers["Access-Control-Allow-Headers"]


def test_api_contract_for_abstained_response():
    status, body = _invoke_json("/v1/consult", {"patient_text": ""})
    assert status == "200 OK"
    assert body["decision_state"] == "abstained"
    assert body["top_diagnoses"] == []
    assert body["safe_to_show_ranked_differential"] is False


def test_api_contract_for_provisional_response():
    status, body = _invoke_json(
        "/v1/consult",
        {"patient_text": "Chronic skin lesion with ulcerated border after travel to an endemic region."},
    )
    assert status == "200 OK"
    assert body["decision_state"] == "provisional_parametric"
    assert body["uncertainty_gate"]["stage"] == "parametric_fallback"
    assert body["safe_to_show_ranked_differential"] is True
    assert body["evidence"] == []


def test_api_contract_for_supported_response():
    status, body = _invoke_json(
        "/v1/consult",
        {
            "patient_text": "Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes."
        },
    )
    assert status == "200 OK"
    assert body["decision_state"] == "rag_supported"
    assert body["uncertainty_gate"]["retrieval_support_status"] == "supported"
    assert len(body["evidence"]) >= 1
    assert {"title", "diagnosis_label", "score", "confirmatory"} <= set(body["evidence"][0])


def test_chat_endpoint_requires_real_gpu_provider():
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Ulcerated plaque after sandfly exposure with amastigotes on smear.",
                }
            ]
        },
    )
    assert status == "200 OK"
    assert body["safety_state"] == "real_gpu_required"
    assert body["provider_mode"] == "deterministic_demo"
    assert body["evidence"] == []
    assert body["retrieval_audit"] == {}


def test_chat_endpoint_with_fake_real_gpu_generator_returns_answer_and_evidence():
    fake = _FakeRealGenerator()
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Adult patient with an ulcerated plaque on the forearm after sandfly exposure "
                        "and a smear showing amastigotes."
                    ),
                }
            ],
            "client_request_id": "chat-success",
        },
        app=_real_gpu_app(fake),
    )
    assert status == "200 OK"
    assert body["request_id"] == "chat-success"
    assert body["provider_mode"] == "real_gpu_gemma4"
    assert body["safety_state"] == "generated_support"
    assert "Supportive differential" in body["assistant_markdown"]
    assert len(body["evidence"]) >= 1
    assert body["runtime_metadata"]["gpu_name"] == "Fake GPU"
    assert body["response_source_mode"] == "live_gpu"
    assert body["source_label"] == "fresh local Gemma 4 GPU generation"
    assert body["fresh_generation_executed"] is True
    audit = body["retrieval_audit"]
    assert audit["retrieval_backend"] == "local_demo_lexical"
    assert audit["top_k_requested"] == 4
    assert audit["candidate_count"] >= audit["returned_count"] >= 1
    assert audit["live_rerank_executed"] is False
    assert audit["live_rerank_method"] is None
    assert "no separate re-ranker contract" in audit["rerank_boundary"]
    assert audit["returned_contexts"][0]["rank"] == 1
    assert {"chunk_id", "source_case_id", "score", "diagnosis_label"} <= set(
        audit["returned_contexts"][0]
    )
    assert audit["official_rerank_reference"]["available"] is False
    assert fake.calls[0]["use_rag"] is True


def test_chat_endpoint_official_v12d_replay_returns_saved_answer_without_generator_call(
    tmp_path, monkeypatch
):
    results_dir = tmp_path / "heldout_evaluation_results"
    results_dir.mkdir()
    trace_path = tmp_path / "trace_summary.json"
    trace_path.write_text(
        json.dumps(
            {
                "official_rag_run": {
                    "retriever_method": "hybrid",
                    "rerank": True,
                    "retrieval_top_k": 20,
                },
                "cases": [
                    {
                        "case_id": "PMC7516301_01",
                        "qid": "PMC7516301_01::Q1_Q3_multimodal_diagnosis",
                        "official_rag_trace": {
                            "context_count": 1,
                            "top_contexts_for_slide": [
                                {
                                    "rank": 1,
                                    "doc_id": "PMC7528117_01",
                                    "score": 0.033,
                                    "diagnosis_type": "MCL",
                                    "label_source": "train_verified",
                                    "text_prefix_260": "official context",
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    saved_answer = (
        "> Safety boundary: research decision support only.\n\n"
        "**Rank 1 supportive consideration:** Mucocutaneous Leishmaniasis (Mucosal form)\n"
        "**Rank 1 Diagnosis Type:** MCL\n"
        "**Rank 1 Confidence:** Medium"
    )
    (results_dir / "PMC7516301_01_result.json").write_text(
        json.dumps(
            {
                "case_id": "PMC7516301_01",
                "assistant_markdown": saved_answer,
                "evidence": [
                    {
                        "chunk_id": "mcl-002",
                        "source_case_id": "case-mcl-002",
                        "title": "MCL evidence",
                        "diagnosis_label": "Mucocutaneous leishmaniasis",
                        "excerpt": "mucosal disease evidence",
                        "score": 0.3617,
                        "confirmatory": True,
                    }
                ],
                "metadata": {
                    "provider_mode": "real_gpu_gemma4",
                    "model_name": "google/gemma-4-E4B-it",
                    "request_id": "v12d_saved_PMC7516301_01",
                    "elapsed_seconds": 65.78,
                    "query_image_tensor_count": 1,
                },
                "split_provenance": {
                    "runtime_retrieval_kb_source": "/demo/kb.json",
                },
                "raw_response": {
                    "assistant_markdown": saved_answer,
                    "disclaimer": "Decision support only.",
                    "model_name": "google/gemma-4-E4B-it",
                    "provider_mode": "real_gpu_gemma4",
                    "request_id": "v12d_saved_PMC7516301_01",
                    "runtime_metadata": {
                        "query_image_tensor_count": 1,
                        "prompt_context_count": 3,
                    },
                    "safety_state": "generated_support",
                    "timing_ms": 65780,
                    "needed_next_inputs": ["clinician review"],
                },
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDICAL_DEMO_OFFICIAL_V12D_RESULTS_DIR", str(results_dir))
    monkeypatch.setenv("MEDICAL_DEMO_OFFICIAL_RAG_TRACE_PATH", str(trace_path))

    fake = _FakeRealGenerator()
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please replay PMC7516301_01 for the defense deck.",
                }
            ],
            "client_request_id": "deck-replay-PMC7516301_01",
            "response_mode": "official_v12d_replay",
        },
        app=_real_gpu_app(fake),
    )

    assert status == "200 OK"
    assert fake.calls == []
    assert body["assistant_markdown"] == saved_answer
    assert body["response_source_mode"] == "official_v12d_replay"
    assert body["source_label"] == "official V12d experiment-pipeline replay"
    assert body["fresh_generation_executed"] is False
    assert body["source_path"].endswith("PMC7516301_01_result.json")
    assert body["runtime_metadata"]["saved_request_id"] == "v12d_saved_PMC7516301_01"
    assert body["runtime_metadata"]["saved_query_image_tensor_count"] == 1
    assert body["retrieval_audit"]["retrieval_backend"] == "official_v12d_saved_demo_output"
    assert body["retrieval_audit"]["returned_contexts"][0]["chunk_id"] == "mcl-002"
    assert body["retrieval_audit"]["official_rerank_reference"]["available"] is True


def test_chat_endpoint_official_replay_requires_known_case_id(monkeypatch, tmp_path):
    monkeypatch.setenv("MEDICAL_DEMO_OFFICIAL_V12D_RESULTS_DIR", str(tmp_path))
    fake = _FakeRealGenerator()
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": "Please replay a case, but no selected case ID is present.",
                }
            ],
            "response_mode": "official_v12d_replay",
        },
        app=_real_gpu_app(fake),
    )

    assert status == "200 OK"
    assert fake.calls == []
    assert body["safety_state"] == "known_case_required"
    assert body["fresh_generation_executed"] is False
    assert "PMC7516301_01" in body["assistant_markdown"]


def test_chat_endpoint_attaches_official_v12d_rerank_reference(tmp_path, monkeypatch):
    trace_path = tmp_path / "trace_summary.json"
    trace_path.write_text(
        json.dumps(
            {
                "version": "V12d",
                "official_rag_run": {
                    "retriever_method": "hybrid",
                    "rerank": True,
                    "retrieval_top_k": 20,
                },
                "cases": [
                    {
                        "case_id": "PMC7516301_01",
                        "qid": "PMC7516301_01::Q1_Q3_multimodal_diagnosis",
                        "official_rag_trace": {
                            "context_count": 10,
                            "top_contexts_for_slide": [
                                {
                                    "rank": 1,
                                    "doc_id": "PMC7528117_01",
                                    "score": 0.03301807008521583,
                                    "diagnosis_type": "MCL",
                                    "label_source": "train_verified",
                                    "text_prefix_260": "Exact official context prefix.",
                                    "text_char_count": 1157,
                                }
                            ],
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEDICAL_DEMO_OFFICIAL_RAG_TRACE_PATH", str(trace_path))

    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "PMC7516301_01 adult patient with an ulcerated plaque on the forearm "
                        "after sandfly exposure and a smear showing amastigotes."
                    ),
                }
            ],
            "client_request_id": "manual-test-PMC7516301_01",
        },
        app=_real_gpu_app(_FakeRealGenerator()),
    )

    assert status == "200 OK"
    reference = body["retrieval_audit"]["official_rerank_reference"]
    assert reference["available"] is True
    assert reference["source_label"] == "official Gemma 4 experiment-pipeline trace"
    assert reference["case_id"] == "PMC7516301_01"
    assert reference["qid"] == "PMC7516301_01::Q1_Q3_multimodal_diagnosis"
    assert reference["retriever_method"] == "hybrid"
    assert reference["rerank"] is True
    assert reference["retrieval_top_k"] == 20
    assert reference["contexts"][0]["doc_id"] == "PMC7528117_01"


def test_chat_endpoint_vague_input_needs_more_information():
    fake = _FakeRealGenerator()
    status, body = _invoke_json(
        "/v1/chat",
        {"messages": [{"role": "user", "content": "Rash."}]},
        app=_real_gpu_app(fake),
    )
    assert status == "200 OK"
    assert body["safety_state"] == "needs_more_input"
    assert body["needed_next_inputs"]
    assert body["retrieval_audit"] == {}
    assert fake.calls == []


def test_chat_endpoint_model_exception_maps_to_safe_error():
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Adult patient with chronic ulcerated plaque after endemic travel, "
                        "with exam findings concerning for cutaneous leishmaniasis."
                    ),
                }
            ]
        },
        app=_real_gpu_app(_FakeRealGenerator(error=True)),
    )
    assert status == "200 OK"
    assert body["safety_state"] == "model_error"
    assert "no clinical suggestion" in body["assistant_markdown"]
    assert body["retrieval_audit"]["retrieval_backend"] == "local_demo_lexical"


def test_chat_endpoint_generation_error_string_maps_to_safe_error():
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Adult patient with chronic ulcerated plaque after endemic travel, "
                        "with exam findings concerning for cutaneous leishmaniasis."
                    ),
                }
            ]
        },
        app=_real_gpu_app(_FakeRealGenerator(error_string=True)),
    )
    assert status == "200 OK"
    assert body["safety_state"] == "model_error"
    assert "Model Unavailable" in body["assistant_markdown"]


def test_chat_endpoint_passes_optional_image_to_generator():
    fake = _FakeRealGenerator()
    status, body = _invoke_json(
        "/v1/chat",
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "Adult patient with an ulcerated plaque on the forearm after sandfly exposure "
                        "and a smear showing amastigotes."
                    ),
                }
            ],
            "image_base64": "aW1hZ2UtYnl0ZXM=",
            "image_filename": "lesion.jpg",
        },
        app=_real_gpu_app(fake),
    )
    assert status == "200 OK"
    assert body["safety_state"] == "generated_support"
    assert fake.calls[0]["image_bytes"] == b"image-bytes"
    assert fake.calls[0]["image_filename"] == "lesion.jpg"
