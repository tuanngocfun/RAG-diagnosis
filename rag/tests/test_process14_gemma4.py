import importlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from configs import prompt_mode as prompt_mode_config
from configs.prompt_mode import PromptMode
from pipeline import answer_generator
from pipeline import audit_run_artifacts
from pipeline import config as pipeline_config
from pipeline import pseudolabel_adapter
from pipeline import retriever as pipeline_retriever
from pipeline import run_baseline_norag
from pipeline import run_multimodal_eval
from pipeline.generators import gemma4 as gemma4_module
from pipeline.pseudolabel_adapter import BuildStats


class _DummyGemma4Generator:
    init_kwargs = None
    calls = []

    def __init__(self, **kwargs):
        type(self).init_kwargs = dict(kwargs)
        type(self).calls = []
        self.model_name = kwargs.get("model_path", "missing-model-path")
        self.last_generation_metadata = {}
        self.decoding_params = {"temperature": 0.3, "max_new_tokens": 32, "do_sample": True}

    def generate(self, query, contexts, image_paths=None, query_images=None, context_images=None, use_rag_prompt=True):
        type(self).calls.append(
            {
                "query": query,
                "contexts": list(contexts or []),
                "image_paths": list(image_paths or []),
                "query_images": list(query_images or []),
                "context_images": list(context_images or []),
                "use_rag_prompt": use_rag_prompt,
            }
        )
        self.last_generation_metadata = {
            "prompt_context_doc_ids": [ctx.get("doc_id") for ctx in (contexts or []) if ctx.get("doc_id")],
            "prompt_context_count": len(contexts or []),
            "format_retry_count": 0,
            "answer_format_valid": True,
            "answer_format_error": "",
            "ordering_mode": type(self).init_kwargs.get("ordering_mode", "image_first"),
            "use_context_image_tensors": bool(type(self).init_kwargs.get("use_context_image_tensors", False)),
            "support_image_tensor_budget": int(type(self).init_kwargs.get("support_image_tensor_budget", 0) or 0),
            "query_image_tensor_attempt_count": len(query_images or []),
            "support_image_tensor_attempt_count": min(
                len(context_images or []),
                int(type(self).init_kwargs.get("support_image_tensor_budget", 0) or 0),
            ),
            "query_image_tensor_count": len(query_images or []),
            "support_image_tensor_count": min(
                len(context_images or []),
                int(type(self).init_kwargs.get("support_image_tensor_budget", 0) or 0),
            ),
            "image_tensor_fallback_used": False,
            "image_tensor_fallback_reason": "",
        }
        return "dummy answer"


def test_balanced_prompt_contract_requires_visual_grounding():
    prompt = prompt_mode_config.build_rag_prompt(
        query="What is the diagnosis?",
        contexts=[{"doc_id": "train-1", "text": "Reference case"}],
        mode=PromptMode.BALANCED,
        query_images=["/tmp/query.jpg"],
        context_images=["/tmp/context.jpg"],
        is_text_only_model=False,
    )

    assert prompt_mode_config.RAG_IMAGE_GROUNDING_PROMPT_CONTRACT_VERSION == "rag_balanced_image_grounding_v1"
    assert "## VISUAL GROUNDING REQUIREMENT" in prompt
    assert "MUST explicitly ground your Rank 1 reasoning" in prompt


def test_generate_answers_routes_gemma4_to_dedicated_generator(tmp_path, monkeypatch):
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "qid": "case-1::Q1_diagnosis",
                "query": "What is the diagnosis?",
                "query_type": "Q1_diagnosis",
                "contexts": [],
                "query_images": [],
                "context_images": [],
                "ground_truth": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"case_id": "train-1", "images": [], "diagnosis_type": "VL"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"VL": 1}, "warnings": []},
    )

    output_path = answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
    )

    assert output_path.exists()
    assert _DummyGemma4Generator.init_kwargs["model_path"] == "google/gemma-4-E4B-it"
    rows = [json.loads(line) for line in output_path.open("r", encoding="utf-8") if line.strip()]
    assert rows[0]["model_name"] == "google/gemma-4-E4B-it"


def test_generate_answers_keeps_q3_retrieved_image_contexts(tmp_path, monkeypatch):
    query_image = tmp_path / "query_image.jpg"
    query_image.write_bytes(b"image placeholder")
    images_dir = tmp_path / "images"
    context_image_dir = images_dir / "train-1"
    context_image_dir.mkdir(parents=True)
    (context_image_dir / "context_image.jpg").write_bytes(b"context image placeholder")

    retrieval_path = tmp_path / "retrieval.jsonl"
    original_query = "What is the diagnosis from the clinical image?"
    retrieval_path.write_text(
        json.dumps(
            {
                "qid": "case-1::Q3_image_diagnosis",
                "query": original_query,
                "query_type": "Q3_image_diagnosis",
                "contexts": [
                    {
                        "doc_id": "train-1",
                        "score": 0.71,
                        "text": "Biopsy confirmed amastigotes in a cutaneous leishmaniasis lesion.",
                    }
                ],
                "query_images": [str(query_image)],
                "context_images": [],
                "ground_truth": {"diagnosis": "cutaneous leishmaniasis", "diagnosis_type": "CL", "species": ""},
                "ground_truth_pseudolabel": {
                    "diagnosis": "cutaneous leishmaniasis",
                    "diagnosis_type": "CL",
                    "species": "",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps(
            {
                "case_id": "train-1",
                "images": [{"file": "context_image.jpg"}],
                "diagnosis_type": "CL",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"CL": 1}, "warnings": []},
    )

    output_path = answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
    )

    rows = [json.loads(line) for line in output_path.open("r", encoding="utf-8") if line.strip()]
    assert rows[0]["query"] == original_query
    assert rows[0]["query_type"] == "Q3_image_diagnosis"
    assert rows[0]["generation_mode"] == "rag_prompt"
    assert rows[0]["gating_info"].startswith("[ROUTER] Q3 -> IMAGE-RAG")
    assert len(rows[0]["contexts"]) == 1
    assert len(rows[0]["context_images"]) == 1

    call = _DummyGemma4Generator.calls[0]
    assert call["query"] == original_query
    assert call["use_rag_prompt"] is True
    assert len(call["contexts"]) == 1
    assert len(call["context_images"]) == 1


def test_generate_answers_uses_q3_medical_no_context_fallback_without_mutating_saved_query(tmp_path, monkeypatch):
    query_image = tmp_path / "query_image.jpg"
    query_image.write_bytes(b"image placeholder")

    retrieval_path = tmp_path / "retrieval.jsonl"
    original_query = "What is the diagnosis from the clinical image?"
    retrieval_path.write_text(
        json.dumps(
            {
                "qid": "case-1::Q3_image_diagnosis",
                "query": original_query,
                "query_type": "Q3_image_diagnosis",
                "contexts": [],
                "query_images": [str(query_image)],
                "context_images": [],
                "ground_truth": {"diagnosis": "cutaneous leishmaniasis", "diagnosis_type": "CL", "species": ""},
                "ground_truth_pseudolabel": {
                    "diagnosis": "cutaneous leishmaniasis",
                    "diagnosis_type": "CL",
                    "species": "",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"case_id": "train-1", "images": [], "diagnosis_type": "CL"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"CL": 1}, "warnings": []},
    )

    output_path = answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
    )

    rows = [json.loads(line) for line in output_path.open("r", encoding="utf-8") if line.strip()]
    assert rows[0]["query"] == original_query
    assert rows[0]["generation_mode"] == "norag_prompt"
    assert rows[0]["gating_info"] == "[ROUTER] Q3 -> NO-RAG (no image-RAG contexts)"
    assert rows[0]["prompt_contract_version"] == "rag_q3_nocontext_image_grounding_v1"
    assert rows[0]["contexts"] == []
    assert rows[0]["context_images"] == []

    call = _DummyGemma4Generator.calls[0]
    assert call["query"] != original_query
    assert original_query in call["query"]
    assert "## INSTRUCTIONS (NO-RAG CONTROL)" in call["query"]
    assert "## PATIENT IMAGES" in call["query"]
    assert "## VISUAL GROUNDING REQUIREMENT" in call["query"]
    assert call["use_rag_prompt"] is False
    assert call["contexts"] == []


def test_generate_answers_records_context_k_and_ordering_mode(tmp_path, monkeypatch):
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "qid": "case-1::Q1_diagnosis",
                "query": "What is the diagnosis?",
                "query_type": "Q1_diagnosis",
                "contexts": [
                    {"doc_id": f"train-{idx}", "score": 0.9 - (idx * 0.01), "text": "confirmatory biopsy evidence"}
                    for idx in range(1, 7)
                ],
                "query_images": [],
                "context_images": [],
                "ground_truth": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        "".join(
            json.dumps({"case_id": f"train-{idx}", "images": [], "diagnosis_type": "VL"}) + "\n"
            for idx in range(1, 7)
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"VL": 6}, "warnings": []},
    )

    output_path = answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
        context_k=3,
        ordering_mode="text_first",
        use_context_image_tensors=True,
        support_image_tensor_budget=2,
    )

    rows = [json.loads(line) for line in output_path.open("r", encoding="utf-8") if line.strip()]
    assert len(rows[0]["contexts"]) == 3
    assert rows[0]["context_k"] == 3
    assert rows[0]["ordering_mode"] == "text_first"
    assert rows[0]["use_context_image_tensors"] is True
    assert rows[0]["support_image_tensor_budget"] == 2
    assert rows[0]["prompt_contract_version"] == "rag_balanced_image_grounding_v1"
    assert _DummyGemma4Generator.init_kwargs["ordering_mode"] == "text_first"
    assert _DummyGemma4Generator.init_kwargs["use_context_image_tensors"] is True
    assert _DummyGemma4Generator.init_kwargs["support_image_tensor_budget"] == 2


def test_gemma4_prepare_inputs_uses_support_image_tensor_budget_exactly():
    generator = gemma4_module.Gemma4Generator.__new__(gemma4_module.Gemma4Generator)
    generator.use_vision = True
    generator.use_context_image_tensors = True
    generator.support_image_tensor_budget = 7
    generator.ordering_mode = "image_first"
    generator.device = "cpu"
    generator.model = SimpleNamespace(device="cpu")

    class _DummyInputs(dict):
        def to(self, _device):
            return self

    class _DummyProcessor:
        def apply_chat_template(self, messages, **_kwargs):
            assert messages[0]["content"][-1]["type"] == "text"
            return _DummyInputs()

    load_calls = []

    def _fake_load_images(image_paths, max_images):
        load_calls.append({"image_paths": list(image_paths or []), "max_images": max_images})
        return list((image_paths or [])[:max_images])

    generator.processor = _DummyProcessor()
    generator._load_images = _fake_load_images

    _, query_tensor_count, support_tensor_count = gemma4_module.Gemma4Generator._prepare_inputs(
        generator,
        prompt="prompt",
        query_images=["q1", "q2"],
        context_images=[f"s{idx}" for idx in range(10)],
    )

    assert query_tensor_count == 2
    assert support_tensor_count == 7
    assert load_calls[0]["max_images"] == gemma4_module.MAX_QUERY_IMAGES
    assert load_calls[1]["max_images"] == 7


def test_gemma4_generate_stamps_image_tensor_fallback_metadata():
    generator = gemma4_module.Gemma4Generator.__new__(gemma4_module.Gemma4Generator)
    generator.use_vision = True
    generator.use_context_image_tensors = True
    generator.support_image_tensor_budget = 1
    generator.ordering_mode = "image_first"
    generator.prompt_mode = PromptMode.BALANCED
    generator.device = "cpu"
    generator.model_name = "google/gemma-4-E4B-it"
    generator.decoding_params = {"temperature": 0.3, "max_new_tokens": 32, "do_sample": True}
    generator.tokenizer = SimpleNamespace(eos_token_id=0)
    class _NoGrad:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    class _DummyTokenTensor:
        shape = (1, 3)

    generator._torch = SimpleNamespace(
        no_grad=lambda: _NoGrad(),
        cuda=SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None),
    )

    class _DummyInputs(dict):
        pass

    calls = {"prepare": []}

    def _fake_prepare_inputs(prompt, query_images, context_images=None):
        calls["prepare"].append(
            {
                "query_images": list(query_images or []),
                "context_images": list(context_images or []),
            }
        )
        return _DummyInputs({"input_ids": _DummyTokenTensor()}), 2 if query_images else 0, 1 if context_images else 0

    class _DummyModel:
        def __init__(self):
            self.device = "cpu"
            self.calls = 0

        def generate(self, **_kwargs):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("CUDA out of memory")
            return [[0, 1, 2, 3]]

    generator.model = _DummyModel()
    generator._prepare_inputs = _fake_prepare_inputs
    generator._decode = lambda _tokens: "fallback answer"

    answer = gemma4_module.Gemma4Generator.generate(
        generator,
        query="What is the diagnosis?",
        contexts=[{"doc_id": "train-1", "text": "Biopsy evidence"}],
        query_images=["q1", "q2"],
        context_images=["s1"],
        use_rag_prompt=False,
    )

    assert answer == "fallback answer"
    assert calls["prepare"][0]["query_images"] == ["q1", "q2"]
    assert calls["prepare"][0]["context_images"] == ["s1"]
    assert calls["prepare"][1]["query_images"] == []
    assert calls["prepare"][1]["context_images"] == []
    assert generator.last_generation_metadata["query_image_tensor_attempt_count"] == 2
    assert generator.last_generation_metadata["support_image_tensor_attempt_count"] == 1
    assert generator.last_generation_metadata["image_tensor_fallback_used"] is True
    assert generator.last_generation_metadata["image_tensor_fallback_reason"] == "oom_retry_without_images"
    assert generator.last_generation_metadata["query_image_tensor_count"] == 0
    assert generator.last_generation_metadata["support_image_tensor_count"] == 0


def test_gemma4_multimodal_ordering_modes():
    query_images = ["query-1", "query-2"]
    support_images = ["support-1"]

    image_first = gemma4_module.Gemma4Generator._assemble_multimodal_content(
        prompt="prompt",
        query_images=query_images,
        support_images=support_images,
        ordering_mode="image_first",
    )
    assert [item["type"] for item in image_first] == ["image", "image", "image", "text"]

    text_first = gemma4_module.Gemma4Generator._assemble_multimodal_content(
        prompt="prompt",
        query_images=query_images,
        support_images=support_images,
        ordering_mode="text_first",
    )
    assert [item["type"] for item in text_first] == ["text", "image", "image", "image"]

    interleaved = gemma4_module.Gemma4Generator._assemble_multimodal_content(
        prompt="prompt",
        query_images=query_images,
        support_images=support_images,
        ordering_mode="interleaved",
    )
    assert [item["type"] for item in interleaved] == ["image", "image", "text", "image"]


def test_run_baseline_norag_defaults_gemma4_model_id(tmp_path, monkeypatch):
    query_file = tmp_path / "queries.jsonl"
    query_file.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "query_type": "Q1_diagnosis",
                "question": "What is the diagnosis?",
                "clinical_context": "fever and splenomegaly",
                "formatted_query": "What is the diagnosis?\n\nClinical Context: fever and splenomegaly",
                "query_images": [],
                "ground_truth": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    test_jsonl = tmp_path / "test.jsonl"
    test_jsonl.write_text(json.dumps({"case_id": "case-1", "images": []}) + "\n", encoding="utf-8")

    stats = BuildStats(
        train_rows=106,
        test_rows=56,
        query_rows=136,
        query_mixed56_rows=56,
        dataset_version="p14_v7",
        train_source="train",
        test_source="test",
        output_dir=str(tmp_path),
        train_path=str(tmp_path / "train_p14_v7_normalized.jsonl"),
        test_path=str(tmp_path / "test_p14_v7_normalized.jsonl"),
        query_path=str(query_file),
        query_mixed56_path=str(tmp_path / "eval_queries_p14_v7_mixed56.jsonl"),
        qrels_verified_path=str(tmp_path / "qrels_p14_v7_verified.json"),
        qrels_pseudolabel_path=str(tmp_path / "qrels_p14_v7_pseudolabel.json"),
    )

    monkeypatch.setattr(run_baseline_norag, "RUNS_DIR", tmp_path)
    monkeypatch.setattr(run_baseline_norag, "TEST_JSONL", test_jsonl)
    monkeypatch.setattr(run_baseline_norag, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(run_baseline_norag, "build_pseudolabel_artifacts", lambda **kwargs: stats)
    monkeypatch.setattr(run_baseline_norag, "time", SimpleNamespace(sleep=lambda *_args, **_kwargs: None))

    run_dir = run_baseline_norag.run_baseline_norag(
        query_types=["Q1_diagnosis"],
        run_id="gemma4_norag_test",
        generator_type="gemma4",
        generator_model=None,
        queries_file=str(query_file),
        evaluate=False,
    )

    summary = json.load((run_dir / "summary.json").open("r", encoding="utf-8"))
    assert _DummyGemma4Generator.init_kwargs["model_path"] == "google/gemma-4-E4B-it"
    assert summary["generator"] == "google/gemma-4-E4B-it"
    assert summary["pseudolabel_artifacts"]["dataset_version"] == "p14_v7"


def test_generate_answers_writes_generation_sidecar_and_keeps_run_config_retrieval_scoped(tmp_path, monkeypatch):
    query_image = tmp_path / "query_image.jpg"
    query_image.write_bytes(b"image placeholder")
    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "qid": "case-1::Q1_diagnosis",
                        "query": "What is the diagnosis?",
                        "query_type": "Q1_diagnosis",
                        "contexts": [{"doc_id": "train-1", "score": 0.8, "text": "confirmatory biopsy evidence"}],
                        "query_images": [],
                        "context_images": [],
                        "ground_truth": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                        "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                    }
                ),
                json.dumps(
                    {
                        "qid": "case-2::Q3_image_diagnosis",
                        "query": "What is the diagnosis from the image?",
                        "query_type": "Q3_image_diagnosis",
                        "contexts": [],
                        "query_images": [str(query_image)],
                        "context_images": [],
                        "ground_truth": {"diagnosis": "x", "diagnosis_type": "CL", "species": ""},
                        "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "CL", "species": ""},
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"case_id": "train-1", "images": [], "diagnosis_type": "VL"}) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "run_config.json").write_text(
        json.dumps(
            {
                "retrieval_top_k": 12,
                "experiment_controls": {"retrieval_top_k": 12},
                "retrieval_contract": {"sentinel": "retrieval_only"},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (tmp_path / "summary.json").write_text(json.dumps({"run_id": "unit"}, indent=2), encoding="utf-8")

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"VL": 1, "CL": 1}, "warnings": []},
    )

    answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
        context_k=5,
        ordering_mode="image_first",
    )

    generation_contract = json.load((tmp_path / "answer_generation_contract.json").open("r", encoding="utf-8"))
    persisted_run_config = json.load((tmp_path / "run_config.json").open("r", encoding="utf-8"))
    persisted_summary = json.load((tmp_path / "summary.json").open("r", encoding="utf-8"))

    assert generation_contract["prompt_contract_version"] == "mixed"
    assert generation_contract["prompt_contract_summary"]["by_version"]["rag_balanced_image_grounding_v1"]["count"] == 1
    assert generation_contract["prompt_contract_summary"]["by_version"]["rag_q3_nocontext_image_grounding_v1"]["count"] == 1
    assert generation_contract["retrieval_top_k_inherited"] == 12
    assert generation_contract["multimodal_usage_summary"]["rows_with_query_images"] == 1
    assert generation_contract["multimodal_usage_summary"]["rows_with_context_images_available"] == 0
    assert generation_contract["multimodal_usage_summary"]["rows_with_query_image_tensors"] == 1
    assert generation_contract["multimodal_usage_summary"]["rows_with_support_image_tensors"] == 0
    assert generation_contract["multimodal_usage_summary"]["image_tensor_fallback_count"] == 0
    assert generation_contract["multimodal_usage_summary"]["true_multimodal_support_active"] is False
    assert "answer_generation_contract" not in persisted_run_config
    assert persisted_run_config["retrieval_contract"]["sentinel"] == "retrieval_only"
    assert persisted_summary["answer_generation_contract"]["prompt_contract_version"] == "mixed"
    assert persisted_summary["answer_generation_contract"]["multimodal_usage_summary"]["rows_with_query_image_tensors"] == 1


def test_generate_answers_stamps_true_multimodal_support_usage_summary(tmp_path, monkeypatch):
    query_image = tmp_path / "query_image.jpg"
    query_image.write_bytes(b"image placeholder")
    images_dir = tmp_path / "images"
    context_image_dir = images_dir / "train-1"
    context_image_dir.mkdir(parents=True)
    (context_image_dir / "context_image.jpg").write_bytes(b"context image placeholder")

    retrieval_path = tmp_path / "retrieval.jsonl"
    retrieval_path.write_text(
        json.dumps(
            {
                "qid": "case-1::Q1_Q3_multimodal_diagnosis",
                "query": "What is the diagnosis?",
                "query_type": "Q1_Q3_multimodal_diagnosis",
                "contexts": [{"doc_id": "train-1", "score": 0.8, "text": "confirmatory biopsy evidence"}],
                "query_images": [str(query_image)],
                "context_images": [],
                "ground_truth": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
                "ground_truth_pseudolabel": {"diagnosis": "x", "diagnosis_type": "VL", "species": ""},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    train_jsonl = tmp_path / "train.jsonl"
    train_jsonl.write_text(
        json.dumps({"case_id": "train-1", "images": [{"file": "context_image.jpg"}], "diagnosis_type": "VL"}) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(answer_generator, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(answer_generator, "IMAGES_DIR", images_dir)
    monkeypatch.setattr(answer_generator, "Gemma4Generator", _DummyGemma4Generator)
    monkeypatch.setattr(
        answer_generator,
        "get_dataset_support_snapshot",
        lambda: {"train_diagnosis_type_counts": {"VL": 1}, "warnings": []},
    )

    output_path = answer_generator.generate_answers(
        tmp_path,
        generator_type="gemma4",
        model_variant="4b",
        output_file="answers_rag_std.jsonl",
        prompt_mode=PromptMode.BALANCED,
        ordering_mode="image_first",
        use_context_image_tensors=True,
        support_image_tensor_budget=1,
    )

    rows = [json.loads(line) for line in output_path.open("r", encoding="utf-8") if line.strip()]
    generation_contract = json.load((tmp_path / "answer_generation_contract.json").open("r", encoding="utf-8"))

    assert rows[0]["query_image_tensor_attempt_count"] == 1
    assert rows[0]["support_image_tensor_attempt_count"] == 1
    assert rows[0]["query_image_tensor_count"] == 1
    assert rows[0]["support_image_tensor_count"] == 1
    assert rows[0]["image_tensor_fallback_used"] is False
    assert generation_contract["multimodal_usage_summary"]["rows_with_query_image_tensors"] == 1
    assert generation_contract["multimodal_usage_summary"]["rows_with_support_image_tensors"] == 1
    assert generation_contract["multimodal_usage_summary"]["true_multimodal_support_active"] is True


def test_resolve_secondary_pseudolabel_qrels_path_prefers_active_artifact(monkeypatch):
    expected = Path("/tmp/qrels_p14_v7_pseudolabel.json")
    monkeypatch.setattr(
        run_multimodal_eval,
        "get_pseudolabel_artifact_paths",
        lambda dataset_version=None: {"qrels_pseudolabel": expected},
    )
    path = run_multimodal_eval._resolve_secondary_pseudolabel_qrels_path(None)
    assert path == expected


def test_config_reload_uses_dataset_version_env(monkeypatch):
    monkeypatch.setenv("STRUCTURED_CASES_DATASET_VERSION", "p14_v7")
    reloaded = importlib.reload(pipeline_config)
    assert reloaded.DATASET_VERSION == "p14_v7"
    assert reloaded.get_dataset_artifact_filenames()["query"] == "eval_queries_p14_v7.jsonl"
    assert "cases_text_e5_1024_p14_v7" in reloaded.COLLECTIONS

    monkeypatch.setenv("STRUCTURED_CASES_DATASET_VERSION", "v163_pseudolabel_v2")
    importlib.reload(pipeline_config)


def test_config_reload_supports_phase1b_mixed_train_alias(monkeypatch):
    monkeypatch.setenv("STRUCTURED_CASES_DATASET_VERSION", "p14_v7_phase1b_tierAB")
    reloaded = importlib.reload(pipeline_config)

    assert reloaded.DATASET_VERSION == "p14_v7_phase1b_tierAB"
    assert reloaded.get_dataset_artifact_filenames()["train"] == "nonleish_additions/generated/train_phase1b_tierAB.jsonl"
    assert reloaded.get_dataset_artifact_filenames()["query"] == "eval_queries_p14_v7.jsonl"
    assert reloaded.get_dataset_base_version() == "p14_v7"
    assert reloaded.dataset_reuses_shared_eval_artifacts() is True
    assert reloaded.is_strict_retrieval_mode("p14_v7_phase1b_tierAB") is True
    assert "cases_text_e5_1024_p14_v7_phase1b_tierAB" in reloaded.COLLECTIONS

    monkeypatch.setenv("STRUCTURED_CASES_DATASET_VERSION", "v163_pseudolabel_v2")
    importlib.reload(pipeline_config)


def test_build_pseudolabel_artifacts_reuses_existing_shared_eval_artifacts_for_phase1b_alias(tmp_path, monkeypatch):
    alias_train = tmp_path / "nonleish_additions" / "generated" / "train_phase1b_tierAB.jsonl"
    alias_train.parent.mkdir(parents=True)
    alias_train.write_text(
        "\n".join(
            [
                json.dumps({"case_id": "train-1", "diagnosis_type": "VL", "is_leishmaniasis": True}),
                json.dumps({"case_id": "train-2", "diagnosis_type": "Non-Leishmaniasis", "is_leishmaniasis": False}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    test_path = tmp_path / "test_p14_v7_normalized.jsonl"
    query_path = tmp_path / "eval_queries_p14_v7.jsonl"
    query_mixed56_path = tmp_path / "eval_queries_p14_v7_mixed56.jsonl"
    qrels_verified_path = tmp_path / "qrels_p14_v7_verified.json"
    qrels_pseudolabel_path = tmp_path / "qrels_p14_v7_pseudolabel.json"
    test_path.write_text(json.dumps({"case_id": "test-1"}) + "\n", encoding="utf-8")
    query_path.write_text(json.dumps({"case_id": "test-1", "query_type": "Q1_diagnosis"}) + "\n", encoding="utf-8")
    query_mixed56_path.write_text(json.dumps({"case_id": "test-1", "query_type": "Q1_diagnosis"}) + "\n", encoding="utf-8")
    qrels_verified_path.write_text("{}", encoding="utf-8")
    qrels_pseudolabel_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(pseudolabel_adapter, "_default_source_results", lambda dataset_version=None: (tmp_path / "train_results.jsonl", tmp_path / "test_results.jsonl"))
    monkeypatch.setattr(
        pseudolabel_adapter,
        "get_pseudolabel_artifact_paths",
        lambda output_suffix="", dataset_version=None, output_dir=None: {
            "train": alias_train,
            "test": test_path,
            "query": query_path,
            "query_mixed56": query_mixed56_path,
            "qrels_verified": qrels_verified_path,
            "qrels_pseudolabel": qrels_pseudolabel_path,
        },
    )

    stats = pseudolabel_adapter.build_pseudolabel_artifacts(
        dataset_version="p14_v7_phase1b_tierAB",
        output_dir=tmp_path,
    )

    assert stats.dataset_version == "p14_v7_phase1b_tierAB"
    assert stats.train_rows == 2
    assert stats.test_rows == 1
    assert stats.query_rows == 1
    assert stats.query_mixed56_rows == 1
    assert Path(stats.train_path) == alias_train


def test_resolve_bm25_index_path_strict_rejects_legacy_fallback(tmp_path, monkeypatch):
    (tmp_path / "bm25_index.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(pipeline_retriever, "SPLIT_DIR", tmp_path)
    monkeypatch.setattr(pipeline_retriever, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(pipeline_retriever, "DATASET_VERSION", "p14_v7")

    try:
        pipeline_retriever.resolve_bm25_index_path("p14_v7", strict=True)
    except FileNotFoundError as exc:
        assert "Strict retrieval mode" in str(exc)
    else:
        raise AssertionError("strict p14_v7 retrieval should reject legacy BM25 fallback")


def test_required_resource_matrix_makes_caption_optional_for_zero_coverage_images_pilot(monkeypatch):
    monkeypatch.setattr(run_multimodal_eval, "DATASET_VERSION", "p14_v7")
    matrix = run_multimodal_eval._build_required_resource_matrix(
        query_types=["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"],
        method="hybrid",
        image_search_mode="images",
        rerank=True,
        strict_mode=True,
        strip_query_images=False,
        ablation_scope="",
        caption_support={
            "total_image_entries": 77,
            "caption_entries": 0,
            "caption_coverage_ratio": 0.0,
        },
    )

    assert matrix["canonical_full_pilot"] is True
    assert matrix["dense_collection"]["required_at_bootstrap"] is True
    assert matrix["caption_collection"]["required_at_bootstrap"] is False
    assert matrix["caption_collection"]["required_for_runtime_use"] is False
    assert matrix["image_collection"]["required_for_runtime_use"] is True


def test_required_resource_matrix_still_requires_caption_when_lane_expected_or_coverage_nonzero(monkeypatch):
    monkeypatch.setattr(run_multimodal_eval, "DATASET_VERSION", "p14_v7")
    captions_mode = run_multimodal_eval._build_required_resource_matrix(
        query_types=["Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"],
        method="hybrid",
        image_search_mode="captions",
        rerank=True,
        strict_mode=True,
        strip_query_images=False,
        ablation_scope="",
        caption_support={
            "total_image_entries": 77,
            "caption_entries": 0,
            "caption_coverage_ratio": 0.0,
        },
    )
    assert captions_mode["caption_collection"]["required_at_bootstrap"] is True
    assert captions_mode["caption_collection"]["required_for_runtime_use"] is True

    nonzero_coverage = run_multimodal_eval._build_required_resource_matrix(
        query_types=["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"],
        method="hybrid",
        image_search_mode="images",
        rerank=True,
        strict_mode=True,
        strip_query_images=False,
        ablation_scope="",
        caption_support={
            "total_image_entries": 77,
            "caption_entries": 5,
            "caption_coverage_ratio": 5 / 77,
        },
    )
    assert nonzero_coverage["caption_collection"]["required_at_bootstrap"] is True
    assert nonzero_coverage["caption_collection"]["required_for_runtime_use"] is False


def test_validate_canonical_pilot_contract_requires_hybrid_and_rerank(monkeypatch):
    monkeypatch.setattr(run_multimodal_eval, "DATASET_VERSION", "p14_v7")
    try:
        run_multimodal_eval._validate_canonical_pilot_contract(
            query_types=["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"],
            method="bm25",
            rerank=False,
            image_search_mode="images",
            strip_query_images=False,
            ablation_scope="",
        )
    except ValueError as exc:
        assert "requires --method hybrid" in str(exc)
    else:
        raise AssertionError("canonical p14_v7 pilot should require hybrid + rerank")


class _DummyLane1Retriever:
    def __init__(self, strict_resources=False):
        self.collection_name = "cases_text_e5_1024_p14_v7"
        self.client = object()
        self.strict_resources = strict_resources
        self.resource_events = []
        self._usage = {
            "dense_lane_query_count": 0,
            "bm25_lane_query_count": 0,
            "hybrid_lane_query_count": 0,
            f"collection_queries::{self.collection_name}": 0,
        }

    def retrieve_hybrid(self, query, top_k=20):
        self._usage["hybrid_lane_query_count"] += 1
        self._usage["dense_lane_query_count"] += 1
        self._usage["bm25_lane_query_count"] += 1
        self._usage[f"collection_queries::{self.collection_name}"] += 1
        return [("train-1", 0.9)]

    def retrieve_bm25(self, query, top_k=20):
        self._usage["bm25_lane_query_count"] += 1
        return [("train-1", 0.9)]

    def retrieve_e5(self, query, top_k=20):
        self._usage["dense_lane_query_count"] += 1
        return [("train-1", 0.9)]

    def get_resource_contract(self):
        return {
            "strict_mode": self.strict_resources,
            "dense_collection": self.collection_name,
            "bm25_index": {
                "resolved_path": "/tmp/bm25_index_p14_v7.json",
                "expected_path": "/tmp/bm25_index_p14_v7.json",
                "exists": True,
                "fallback_used": False,
                "fingerprint": {"path": "/tmp/bm25_index_p14_v7.json", "exists": True, "hash": "abc123"},
            },
            "resource_events": [],
            "usage": dict(self._usage),
        }

    def get_usage_snapshot(self):
        return dict(self._usage)


class _DummyLane2Retriever:
    def __init__(self, strict_resources=False):
        self.caption_collection = "captions_biomedclip_512_p14_v7"
        self.image_collection = "images_biomedclip_512_p14_v7"
        self.client = object()
        self.strict_resources = strict_resources
        self._usage = {}

    def retrieve_by_caption(self, query, top_k=20):
        return []

    def retrieve_by_image_path(self, image_path, top_k=20, search_captions=True):
        return []

    def get_resource_contract(self):
        return {
            "strict_mode": self.strict_resources,
            "caption_collection": self.caption_collection,
            "image_collection": self.image_collection,
            "usage": dict(self._usage),
        }

    def get_usage_snapshot(self):
        return dict(self._usage)


class _DummyReranker:
    def rerank(self, query, candidates, top_k=20):
        return [(cid, text, score) for cid, text, score in candidates[:top_k]]


def test_run_multimodal_eval_rejects_context_budget_above_retrieval_budget():
    try:
        run_multimodal_eval.run_multimodal_evaluation(
            qrels_file="unused.json",
            query_types=["Q1_diagnosis"],
            method="hybrid",
            rerank=True,
            run_id="should_not_run",
            image_search_mode="images",
            queries_file="unused.jsonl",
            dataset_pack="auto",
            context_k=5,
            retrieval_top_k=3,
        )
    except ValueError as exc:
        assert "retrieval_top_k must be >= context_k" in str(exc)
    else:
        raise AssertionError("retrieval_top_k < context_k should fail fast")


def test_run_multimodal_eval_writes_retrieval_contract_and_usage(tmp_path, monkeypatch):
    run_root = tmp_path / "runs"
    run_root.mkdir()
    query_file = tmp_path / "eval_queries_p14_v7.jsonl"
    query_file.write_text(
        json.dumps(
            {
                "case_id": "case-1",
                "query_type": "Q1_diagnosis",
                "question": "What is the diagnosis?",
                "clinical_context": "fever and splenomegaly",
                "formatted_query": "What is the diagnosis?\n\nClinical Context: fever and splenomegaly",
                "query_images": [],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    qrels_path = tmp_path / "qrels_p14_v7_verified.json"
    qrels_path.write_text(json.dumps({"case-1": {"train-1": 3}}, indent=2), encoding="utf-8")

    train_jsonl = tmp_path / "train_p14_v7_normalized.jsonl"
    train_jsonl.write_text(
        json.dumps({"case_id": "train-1", "case_text": "confirmatory diagnosis", "images": [], "diagnosis_type": "VL"})
        + "\n",
        encoding="utf-8",
    )
    test_jsonl = tmp_path / "test_p14_v7_normalized.jsonl"
    test_jsonl.write_text(
        json.dumps({"case_id": "case-1", "diagnosis": "VL", "diagnosis_type": "VL", "species": "", "images": []})
        + "\n",
        encoding="utf-8",
    )

    def fake_eval(all_retrieved, qrels, k_values, return_per_query=False):
        per_query = {
            "case-1::Q1_diagnosis": {
                "recall@5": 1.0,
                "ndcg@5": 1.0,
                "precision@5": 0.2,
                "mrr": 1.0,
                "ap": 1.0,
            }
        }
        return SimpleNamespace(
            recall_at_k={5: 1.0, 10: 1.0},
            ndcg_at_k={5: 1.0, 10: 1.0},
            precision_at_k={5: 0.2, 10: 0.1},
            mrr=1.0,
            map_score=1.0,
            per_query_metrics=per_query if return_per_query else per_query,
        )

    monkeypatch.setattr(run_multimodal_eval, "RUNS_DIR", run_root)
    monkeypatch.setattr(run_multimodal_eval, "SPLIT_DIR", tmp_path)
    monkeypatch.setattr(run_multimodal_eval, "TRAIN_JSONL", train_jsonl)
    monkeypatch.setattr(run_multimodal_eval, "TEST_JSONL", test_jsonl)
    monkeypatch.setattr(run_multimodal_eval, "TRAIN_NORMALIZED", train_jsonl)
    monkeypatch.setattr(run_multimodal_eval, "TEST_NORMALIZED", test_jsonl)
    monkeypatch.setattr(run_multimodal_eval, "build_pseudolabel_artifacts", lambda **kwargs: None)
    monkeypatch.setattr(run_multimodal_eval, "get_runtime_metadata", lambda: {
        "dataset_version": "p14_v7",
        "train_jsonl": str(train_jsonl),
        "train_jsonl_hash": "trainhash",
        "train_jsonl_rows": 1,
        "test_jsonl": str(test_jsonl),
        "test_jsonl_hash": "testhash",
        "test_jsonl_rows": 1,
    })
    monkeypatch.setattr(run_multimodal_eval, "get_dataset_support_snapshot", lambda: {"warnings": []})
    monkeypatch.setattr(run_multimodal_eval, "Lane1Retriever", _DummyLane1Retriever)
    monkeypatch.setattr(run_multimodal_eval, "Lane2Retriever", _DummyLane2Retriever)
    monkeypatch.setattr(run_multimodal_eval, "get_medcpt_reranker", lambda: _DummyReranker())
    monkeypatch.setattr(run_multimodal_eval, "evaluate_retrieval", fake_eval)
    monkeypatch.setattr(run_multimodal_eval, "get_collection_snapshot", lambda client, name: {
        "name": name,
        "exists": True,
        "points_count": 5,
        "vector_size": 1024 if "cases_text" in name else 512,
        "error": None,
    })
    monkeypatch.setattr(run_multimodal_eval, "is_strict_retrieval_mode", lambda dataset_version=None: False)
    monkeypatch.setitem(sys.modules, "pipeline.run_catalog", SimpleNamespace(update_catalog=lambda: None))

    run_dir = run_multimodal_eval.run_multimodal_evaluation(
        qrels_file="qrels_p14_v7_verified.json",
        query_types=["Q1_diagnosis"],
        method="hybrid",
        rerank=True,
        k_values=[5, 10],
        run_id="unit_retrieval_contract",
        image_search_mode="images",
        queries_file=str(query_file),
        dataset_pack="auto",
        context_k=5,
        retrieval_top_k=12,
        ordering_mode="image_first",
    )

    run_config = json.load((run_dir / "run_config.json").open("r", encoding="utf-8"))
    summary = json.load((run_dir / "summary.json").open("r", encoding="utf-8"))
    retrieval_contract = run_config["retrieval_contract"]

    assert retrieval_contract["resolved_resources"]["dense_collection"] == "cases_text_e5_1024_p14_v7"
    assert Path(retrieval_contract["resolved_resources"]["bm25_index"]["resolved_path"]).name == "bm25_index_p14_v7.json"
    assert run_config["caption_support"]["coverage"]["caption_entries"] == 0
    assert run_config["caption_support"]["required_at_bootstrap"] is False
    assert summary["caption_support"]["lane_exercised"] is False
    assert summary["retrieval_usage"]["dense_lane_query_count"] == 1
    assert summary["retrieval_usage"]["bm25_lane_query_count"] == 1
    assert summary["retrieval_usage"]["rerank_query_count"] == 1
    assert summary["retrieval_usage"]["resources_exercised"]["dense_lane"] is True
    assert summary["retrieval_usage"]["resources_exercised"]["rerank"] is True
    assert run_config["context_k"] == 5
    assert run_config["retrieval_top_k"] == 12
    assert summary["experiment_controls"]["context_k"] == 5
    assert summary["experiment_controls"]["retrieval_top_k"] == 12


def test_audit_run_artifacts_checks_retrieval_contract_usage(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_jsonl = tmp_path / "train_p14_v7_normalized.jsonl"
    train_jsonl.write_text(json.dumps({"case_id": "train-1"}) + "\n", encoding="utf-8")
    test_jsonl = tmp_path / "test_p14_v7_normalized.jsonl"
    test_jsonl.write_text(json.dumps({"case_id": "case-1"}) + "\n", encoding="utf-8")

    run_config = {
        "run_id": "unit",
        "qrels_file": "qrels_p14_v7_verified.json",
        "queries_file": str(tmp_path / "eval_queries_p14_v7.jsonl"),
        "image_search_mode": "images",
        "retriever_method": "hybrid",
        "rerank": True,
        "runtime_metadata": {
            "train_jsonl": str(train_jsonl),
            "train_jsonl_hash": pipeline_config.get_path_fingerprint(train_jsonl)["hash"],
            "train_jsonl_rows": 1,
            "test_jsonl": str(test_jsonl),
            "test_jsonl_hash": pipeline_config.get_path_fingerprint(test_jsonl)["hash"],
            "test_jsonl_rows": 1,
        },
        "retrieval_contract": {
            "strict_mode": True,
            "resolved_resources": {
                "dense_collection": "cases_text_e5_1024_p14_v7",
                "caption_collection": "captions_biomedclip_512_p14_v7",
                "image_collection": "images_biomedclip_512_p14_v7",
                "bm25_index": {"resolved_path": str(tmp_path / "bm25_index_p14_v7.json")},
            },
            "collections_at_start": {
                "dense_collection": {"exists": True},
                "caption_collection": {"exists": True},
                "image_collection": {"exists": True},
            },
            "usage": {
                "resources_exercised": {
                    "dense_lane": True,
                    "bm25_lane": True,
                    "caption_lane": False,
                    "image_lane": True,
                    "rerank": True,
                },
                "dense_lane_query_count": 1,
                "bm25_lane_query_count": 1,
                "caption_query_count": 0,
                "image_query_count": 1,
                "rerank_query_count": 1,
            },
        },
        "evaluation_contract": {"baseline_equivalent": True, "warnings": []},
    }
    summary = {
        "run_id": "unit",
        "retrieval_contract": run_config["retrieval_contract"],
        "retrieval_usage": run_config["retrieval_contract"]["usage"],
        "evaluation_contract": run_config["evaluation_contract"],
    }

    (tmp_path / "eval_queries_p14_v7.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "bm25_index_p14_v7.json").write_text("{}", encoding="utf-8")
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    argv = [
        "audit_run_artifacts.py",
        "--run-dir",
        str(run_dir),
        "--expected-train-jsonl",
        str(train_jsonl),
        "--expected-test-jsonl",
        str(test_jsonl),
        "--expected-qrels",
        "qrels_p14_v7_verified.json",
        "--expected-queries",
        "eval_queries_p14_v7.jsonl",
        "--expected-image-search",
        "images",
        "--expected-method",
        "hybrid",
        "--require-rerank",
        "--require-strict-retrieval",
        "--expected-bm25-index",
        str(tmp_path / "bm25_index_p14_v7.json"),
        "--expected-dense-collection",
        "cases_text_e5_1024_p14_v7",
        "--expected-caption-collection",
        "captions_biomedclip_512_p14_v7",
        "--expected-image-collection",
        "images_biomedclip_512_p14_v7",
        "--require-resource-usage",
        "dense_lane,bm25_lane,image_lane,rerank",
        "--require-baseline-equivalent",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert audit_run_artifacts.main() == 0


def test_audit_run_artifacts_allows_missing_caption_collection_for_zero_coverage_images_pilot(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_optional_caption"
    run_dir.mkdir()
    train_jsonl = tmp_path / "train_p14_v7_normalized.jsonl"
    train_jsonl.write_text(json.dumps({"case_id": "train-1"}) + "\n", encoding="utf-8")
    test_jsonl = tmp_path / "test_p14_v7_normalized.jsonl"
    test_jsonl.write_text(json.dumps({"case_id": "case-1"}) + "\n", encoding="utf-8")
    (tmp_path / "eval_queries_p14_v7.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "bm25_index_p14_v7.json").write_text("{}", encoding="utf-8")

    caption_support = {
        "coverage": {
            "total_image_entries": 77,
            "caption_entries": 0,
            "caption_coverage_ratio": 0.0,
        },
        "collection_name": "captions_biomedclip_512_p14_v7",
        "collection_exists": False,
        "lane_expected": False,
        "lane_exercised": False,
        "required_at_bootstrap": False,
        "absence_reason": "caption_collection_absent_due_to_zero_caption_coverage",
    }
    run_config = {
        "run_id": "unit_optional_caption",
        "qrels_file": "qrels_p14_v7_verified.json",
        "queries_file": str(tmp_path / "eval_queries_p14_v7.jsonl"),
        "image_search_mode": "images",
        "retriever_method": "hybrid",
        "rerank": True,
        "runtime_metadata": {
            "train_jsonl": str(train_jsonl),
            "train_jsonl_hash": pipeline_config.get_path_fingerprint(train_jsonl)["hash"],
            "train_jsonl_rows": 1,
            "test_jsonl": str(test_jsonl),
            "test_jsonl_hash": pipeline_config.get_path_fingerprint(test_jsonl)["hash"],
            "test_jsonl_rows": 1,
        },
        "caption_support": caption_support,
        "retrieval_contract": {
            "strict_mode": True,
            "resolved_resources": {
                "dense_collection": "cases_text_e5_1024_p14_v7",
                "caption_collection": "captions_biomedclip_512_p14_v7",
                "image_collection": "images_biomedclip_512_p14_v7",
                "bm25_index": {"resolved_path": str(tmp_path / "bm25_index_p14_v7.json")},
            },
            "collections_at_start": {
                "dense_collection": {"exists": True},
                "caption_collection": {"exists": False},
                "image_collection": {"exists": True},
            },
            "caption_support": caption_support,
            "usage": {
                "resources_exercised": {
                    "dense_lane": True,
                    "bm25_lane": True,
                    "caption_lane": False,
                    "image_lane": True,
                    "rerank": True,
                },
                "dense_lane_query_count": 1,
                "bm25_lane_query_count": 1,
                "caption_query_count": 0,
                "image_query_count": 1,
                "rerank_query_count": 1,
            },
        },
        "evaluation_contract": {"baseline_equivalent": True, "warnings": []},
    }
    summary = {
        "run_id": "unit_optional_caption",
        "caption_support": caption_support,
        "retrieval_contract": run_config["retrieval_contract"],
        "retrieval_usage": run_config["retrieval_contract"]["usage"],
        "evaluation_contract": run_config["evaluation_contract"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    argv = [
        "audit_run_artifacts.py",
        "--run-dir",
        str(run_dir),
        "--expected-train-jsonl",
        str(train_jsonl),
        "--expected-test-jsonl",
        str(test_jsonl),
        "--expected-qrels",
        "qrels_p14_v7_verified.json",
        "--expected-queries",
        "eval_queries_p14_v7.jsonl",
        "--expected-image-search",
        "images",
        "--expected-method",
        "hybrid",
        "--require-rerank",
        "--require-strict-retrieval",
        "--expected-bm25-index",
        str(tmp_path / "bm25_index_p14_v7.json"),
        "--expected-dense-collection",
        "cases_text_e5_1024_p14_v7",
        "--expected-caption-collection",
        "captions_biomedclip_512_p14_v7",
        "--expected-image-collection",
        "images_biomedclip_512_p14_v7",
        "--require-resource-usage",
        "dense_lane,bm25_lane,image_lane,rerank",
        "--require-baseline-equivalent",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert audit_run_artifacts.main() == 0


def test_audit_run_artifacts_rejects_inconsistent_optional_caption_contract(tmp_path, monkeypatch):
    run_dir = tmp_path / "run_bad_optional_caption"
    run_dir.mkdir()
    train_jsonl = tmp_path / "train_p14_v7_normalized.jsonl"
    train_jsonl.write_text(json.dumps({"case_id": "train-1"}) + "\n", encoding="utf-8")
    test_jsonl = tmp_path / "test_p14_v7_normalized.jsonl"
    test_jsonl.write_text(json.dumps({"case_id": "case-1"}) + "\n", encoding="utf-8")
    (tmp_path / "eval_queries_p14_v7.jsonl").write_text("{}", encoding="utf-8")
    (tmp_path / "bm25_index_p14_v7.json").write_text("{}", encoding="utf-8")

    caption_support = {
        "coverage": {
            "total_image_entries": 77,
            "caption_entries": 1,
            "caption_coverage_ratio": 1 / 77,
        },
        "collection_name": "captions_biomedclip_512_p14_v7",
        "collection_exists": False,
        "lane_expected": False,
        "lane_exercised": False,
        "required_at_bootstrap": False,
        "absence_reason": "caption_collection_absent_due_to_zero_caption_coverage",
    }
    run_config = {
        "run_id": "unit_bad_optional_caption",
        "qrels_file": "qrels_p14_v7_verified.json",
        "queries_file": str(tmp_path / "eval_queries_p14_v7.jsonl"),
        "image_search_mode": "images",
        "retriever_method": "hybrid",
        "rerank": True,
        "runtime_metadata": {
            "train_jsonl": str(train_jsonl),
            "train_jsonl_hash": pipeline_config.get_path_fingerprint(train_jsonl)["hash"],
            "train_jsonl_rows": 1,
            "test_jsonl": str(test_jsonl),
            "test_jsonl_hash": pipeline_config.get_path_fingerprint(test_jsonl)["hash"],
            "test_jsonl_rows": 1,
        },
        "caption_support": caption_support,
        "retrieval_contract": {
            "strict_mode": True,
            "resolved_resources": {
                "dense_collection": "cases_text_e5_1024_p14_v7",
                "caption_collection": "captions_biomedclip_512_p14_v7",
                "image_collection": "images_biomedclip_512_p14_v7",
                "bm25_index": {"resolved_path": str(tmp_path / "bm25_index_p14_v7.json")},
            },
            "collections_at_start": {
                "dense_collection": {"exists": True},
                "caption_collection": {"exists": False},
                "image_collection": {"exists": True},
            },
            "caption_support": caption_support,
            "usage": {
                "resources_exercised": {
                    "dense_lane": True,
                    "bm25_lane": True,
                    "caption_lane": False,
                    "image_lane": True,
                    "rerank": True,
                },
                "dense_lane_query_count": 1,
                "bm25_lane_query_count": 1,
                "caption_query_count": 0,
                "image_query_count": 1,
                "rerank_query_count": 1,
            },
        },
        "evaluation_contract": {"baseline_equivalent": True, "warnings": []},
    }
    summary = {
        "run_id": "unit_bad_optional_caption",
        "caption_support": caption_support,
        "retrieval_contract": run_config["retrieval_contract"],
        "retrieval_usage": run_config["retrieval_contract"]["usage"],
        "evaluation_contract": run_config["evaluation_contract"],
    }
    (run_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    argv = [
        "audit_run_artifacts.py",
        "--run-dir",
        str(run_dir),
        "--expected-train-jsonl",
        str(train_jsonl),
        "--expected-test-jsonl",
        str(test_jsonl),
        "--expected-qrels",
        "qrels_p14_v7_verified.json",
        "--expected-queries",
        "eval_queries_p14_v7.jsonl",
        "--expected-image-search",
        "images",
        "--expected-method",
        "hybrid",
        "--require-rerank",
        "--require-strict-retrieval",
        "--expected-bm25-index",
        str(tmp_path / "bm25_index_p14_v7.json"),
        "--expected-dense-collection",
        "cases_text_e5_1024_p14_v7",
        "--expected-caption-collection",
        "captions_biomedclip_512_p14_v7",
        "--expected-image-collection",
        "images_biomedclip_512_p14_v7",
        "--require-resource-usage",
        "dense_lane,bm25_lane,image_lane,rerank",
        "--require-baseline-equivalent",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    assert audit_run_artifacts.main() == 1
