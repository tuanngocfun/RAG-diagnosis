import json

import pytest

from pipeline.run_ragas_phased import (
    ACTIVE_BATCH_STATE_FLAGS,
    ActiveDiagnosisBatch,
    PHASE_B,
    PHASE_LABEL_TRACK,
    _annotate_phase_row,
    _new_manifest,
    _run_diagnosis_phase,
    _save_manifest,
    _validate_phase_output,
)


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def test_validate_phase_output_accepts_complete_phase_b(tmp_path):
    ordered_qids = ["case_a::Q1", "case_b::Q1"]
    answers_sha256 = "sha-ok"
    rows = []
    for qid in ordered_qids:
        row = _annotate_phase_row(
            {
                "qid": qid,
                "diagnosis_accuracy": 1.0,
                "diagnosis_type_accuracy": 1.0,
                "gt_rank": 1,
                "top3_hit": 1.0,
                "l3_top1_correct": 1.0,
                "fallback_level": "type_exact",
                "rank_source": "judge",
                "judge_gt_rank": 1,
                "parser_gt_rank": 1,
                "judge_parser_disagreement": False,
                "diagnosis_reasoning": "ok",
                "diagnosis_method": "llm_judge_batch",
            },
            run_id="run123",
            answers_sha256=answers_sha256,
            phase=PHASE_B,
            label_track=PHASE_LABEL_TRACK[PHASE_B],
        )
        rows.append(row)

    phase_path = tmp_path / "phase_b_verified.jsonl"
    _write_jsonl(phase_path, rows)

    validated = _validate_phase_output(
        run_dir=tmp_path,
        phase=PHASE_B,
        ordered_qids=ordered_qids,
        answers_sha256=answers_sha256,
        run_id="run123",
    )

    assert list(validated) == ordered_qids


def test_validate_phase_output_rejects_missing_required_field(tmp_path):
    ordered_qids = ["case_a::Q1"]
    answers_sha256 = "sha-ok"
    broken_row = _annotate_phase_row(
        {
            "qid": ordered_qids[0],
            "diagnosis_accuracy": 1.0,
            "diagnosis_type_accuracy": 1.0,
            "gt_rank": 1,
            "top3_hit": 1.0,
            "l3_top1_correct": 1.0,
            "fallback_level": "type_exact",
            "rank_source": "judge",
            "judge_gt_rank": 1,
            "parser_gt_rank": 1,
            "judge_parser_disagreement": False,
            "diagnosis_reasoning": "ok",
            "diagnosis_method": "llm_judge_batch",
        },
        run_id="run123",
        answers_sha256=answers_sha256,
        phase=PHASE_B,
        label_track=PHASE_LABEL_TRACK[PHASE_B],
    )
    broken_row.pop("diagnosis_method")
    _write_jsonl(tmp_path / "phase_b_verified.jsonl", [broken_row])

    with pytest.raises(RuntimeError, match="missing required fields"):
        _validate_phase_output(
            run_dir=tmp_path,
            phase=PHASE_B,
            ordered_qids=ordered_qids,
            answers_sha256=answers_sha256,
            run_id="run123",
        )


def test_run_diagnosis_phase_preserves_active_batch_without_resubmit(tmp_path, monkeypatch):
    ordered_qids = ["case_a::Q1"]
    answers_sha256 = "sha-ok"
    manifest = _new_manifest(
        run_dir=tmp_path,
        answers_file="answers.jsonl",
        answers_sha256=answers_sha256,
        selected_qids_sha256="qid-sha",
        ordered_qids=ordered_qids,
        judge_model="gemini-2.5-pro",
    )
    manifest["phases"][PHASE_B]["status"] = "running"
    manifest["phases"][PHASE_B]["batch_job_id"] = "batches/existing"
    manifest["phases"][PHASE_B]["batch_judge_model"] = "gemini-2.5-pro"
    manifest_path = tmp_path / "eval_manifest.json"
    _save_manifest(manifest_path, manifest)

    submit_calls = {"count": 0}

    def _fail_submit(*args, **kwargs):
        submit_calls["count"] += 1
        raise AssertionError("submit should not be called for an active existing batch")

    def _active_batch(*args, **kwargs):
        raise ActiveDiagnosisBatch(
            job_id="batches/existing",
            state=f"JOB_STATE_{ACTIVE_BATCH_STATE_FLAGS[0]}",
            timeout_seconds=7200,
        )

    monkeypatch.setattr("pipeline.run_ragas_phased._submit_diagnosis_batch", _fail_submit)
    monkeypatch.setattr("pipeline.run_ragas_phased._poll_diagnosis_batch", _active_batch)

    samples_by_qid = {
        ordered_qids[0]: {
            "answer": "answer",
            "query": "clinical query",
            "ground_truth": {"diagnosis": "Visceral Leishmaniasis", "diagnosis_type": "VL", "species": ""},
            "query_images": [],
        }
    }

    with pytest.raises(ActiveDiagnosisBatch) as excinfo:
        _run_diagnosis_phase(
            phase=PHASE_B,
            evaluator=object(),
            run_dir=tmp_path,
            ordered_qids=ordered_qids,
            samples_by_qid=samples_by_qid,
            manifest_path=manifest_path,
            manifest=manifest,
            answers_sha256=answers_sha256,
            poll_seconds=0.01,
            timeout_seconds=7200,
            diagnosis_batch_api=True,
            resume=True,
        )

    assert excinfo.value.job_id == "batches/existing"
    assert excinfo.value.phase == PHASE_B
    assert submit_calls["count"] == 0
    assert manifest["phases"][PHASE_B]["status"] == "running"
    assert manifest["phases"][PHASE_B]["batch_job_id"] == "batches/existing"
    assert manifest["phases"][PHASE_B]["last_observed_batch_state"] == f"JOB_STATE_{ACTIVE_BATCH_STATE_FLAGS[0]}"
