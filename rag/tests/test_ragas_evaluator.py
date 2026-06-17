import asyncio
import sys
from types import ModuleType, SimpleNamespace
from pathlib import Path

import pipeline.ragas_evaluator as ragas_evaluator_module
from pipeline.ragas_evaluator import RAGAsLibraryEvaluator, _select_ragas_src


def test_select_ragas_src_prefers_candidate_with_version_file(tmp_path):
    broken = tmp_path / "broken" / "src" / "ragas"
    broken.mkdir(parents=True)
    (broken / "__init__.py").write_text("", encoding="utf-8")

    working = tmp_path / "working" / "src" / "ragas"
    working.mkdir(parents=True)
    (working / "__init__.py").write_text("", encoding="utf-8")
    (working / "_version.py").write_text("__version__ = '0.0-test'\n", encoding="utf-8")

    selected = _select_ragas_src([broken.parent, working.parent])

    assert selected == Path(working.parent)


def test_evaluate_reasoning_recall_logs_fallback_judge_model(monkeypatch):
    class FakeModels:
        def __init__(self):
            self.calls = []

        def generate_content(self, model, contents, config):
            self.calls.append(model)
            if model == "primary-model":
                raise RuntimeError("primary unavailable")
            return SimpleNamespace(text='{"matched_groundtruth_indices":[1],"explanation":"ok"}')

    fake_models = FakeModels()

    class FakeClient:
        def __init__(self, api_key):
            self.api_key = api_key
            self.models = fake_models

    fake_google = ModuleType("google")
    fake_google.genai = SimpleNamespace(Client=FakeClient)
    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setattr(ragas_evaluator_module, "JUDGE_MODEL", "secondary-model")
    monkeypatch.setattr(ragas_evaluator_module, "JUDGE_MODEL_FALLBACK", "tertiary-model")

    evaluator = RAGAsLibraryEvaluator.__new__(RAGAsLibraryEvaluator)
    evaluator.api_key = "test-key"
    evaluator.model_name = "primary-model"

    result = asyncio.run(
        evaluator.evaluate_reasoning_recall(
            groundtruth_points=["Point A", "Point B"],
            predicted_steps=["Predicted A"],
        )
    )

    assert fake_models.calls == ["primary-model", "secondary-model"]
    assert result["judge_model"] == "secondary-model"
    assert result["requested_judge_model"] == "primary-model"
    assert result["method"] == "llm_judge_fallback_model"
    assert result["matched_groundtruth_indices"] == [1]
