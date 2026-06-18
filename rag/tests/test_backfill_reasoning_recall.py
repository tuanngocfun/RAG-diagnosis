import json

from pipeline.backfill_reasoning_recall import backfill_run_dir


class FakeReasoningEvaluator:
    def __init__(self):
        self.calls = []

    async def evaluate_reasoning_recall_for_sample(self, qid: str, answer: str):
        self.calls.append((qid, answer))
        return {
            "recall": 1.0,
            "method": "llm_judge",
            "groundtruth_count": 2,
            "matched_groundtruth_count": 2,
            "matched_groundtruth_indices": [1, 2],
            "matched_groundtruth_points": ["A", "B"],
            "unmatched_groundtruth_points": [],
            "explanation": "all matched",
            "source_path": "/tmp/v7.jsonl",
            "source_id": "p14_v7_reaudit_shared56",
            "predicted_reasoning_steps": ["A", "B"],
            "judge_model": "gemini-2.5-pro",
            "requested_judge_model": "gemini-2.5-pro",
        }


def _write_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def test_backfill_run_dir_is_idempotent(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    answers_rows = [
        {
            "qid": "PMC1::Q1_diagnosis",
            "answer": "1. Step A\n2. Step B",
        }
    ]
    phase_a_rows = [
        {
            "qid": "PMC1::Q1_diagnosis",
            "phase": "phase_a_metrics",
            "label_track": "retrieval_metrics",
            "answers_sha256": "abc",
            "eval_scope_id": "scope",
            "traces": {},
        }
    ]
    ragas_rows = [
        {
            "qid": "PMC1::Q1_diagnosis",
            "traces": {},
        }
    ]

    _write_jsonl(run_dir / "answers_rag_std.jsonl", answers_rows)
    _write_jsonl(run_dir / "phase_a_metrics.jsonl", phase_a_rows)
    _write_jsonl(run_dir / "ragas.jsonl", ragas_rows)
    _write_json(run_dir / "summary.json", {"metrics": {"grounded_accuracy": None}})

    evaluator = FakeReasoningEvaluator()
    first = backfill_run_dir(run_dir, evaluator=evaluator)

    assert first["phase_a_updated"] == 1
    assert first["ragas_updated"] == 1
    assert first["summary_updated"] is True
    assert len(evaluator.calls) == 2

    with open(run_dir / "phase_a_metrics.jsonl", "r", encoding="utf-8") as f:
        phase_row = json.loads(next(f))
    with open(run_dir / "ragas.jsonl", "r", encoding="utf-8") as f:
        ragas_row = json.loads(next(f))
    with open(run_dir / "summary.json", "r", encoding="utf-8") as f:
        summary = json.load(f)

    assert phase_row["reasoning_recall"] == 1.0
    assert ragas_row["reasoning_recall_source_id"] == "p14_v7_reaudit_shared56"
    assert ragas_row["traces"]["reasoning_recall_diagnostics"]["judge_model"] == "gemini-2.5-pro"
    assert summary["ragas_metrics_verified"]["reasoning_recall"] == 1.0

    second = backfill_run_dir(run_dir, evaluator=evaluator)

    assert second["phase_a_updated"] == 0
    assert second["phase_a_skipped"] == 1
    assert second["ragas_updated"] == 0
    assert second["ragas_skipped"] == 1
    assert len(evaluator.calls) == 2
