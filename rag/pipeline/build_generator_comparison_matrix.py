#!/usr/bin/env python3
"""Build a generator comparison matrix from completed run artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Optional

from .compare_rag_norag import (
    _enrich_rows_with_answer_metadata,
    _load_ragas_rows,
    _load_run_contract_metadata,
    _resolve_run_dir,
    _summarize_by_bucket,
)
from .config import RUNS_DIR


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric(summary: Dict[str, Dict[str, object]], bucket: str, metric: str) -> Optional[float]:
    return summary.get(bucket, {}).get("metrics", {}).get(metric)


def _safe_delta(lhs: Optional[float], rhs: Optional[float]) -> Optional[float]:
    if lhs is None or rhs is None:
        return None
    return lhs - rhs


def _load_manifest_judge_model(run_dir: Path) -> Optional[str]:
    manifest = run_dir / "eval_manifest.json"
    if manifest.exists():
        payload = _load_json(manifest)
        judge_model = payload.get("judge_model")
        if judge_model:
            return str(judge_model)

    run_config = run_dir / "run_config.json"
    if run_config.exists():
        payload = _load_json(run_config)
        merged_from = payload.get("merged_from_chunk_runs") or []
        for chunk_path in merged_from:
            chunk_manifest = Path(chunk_path) / "eval_manifest.json"
            if not chunk_manifest.exists():
                continue
            judge_model = _load_json(chunk_manifest).get("judge_model")
            if judge_model:
                return str(judge_model)
    return None


def _load_generator_model(run_dir: Path) -> Optional[str]:
    run_config = run_dir / "run_config.json"
    if run_config.exists():
        payload = _load_json(run_config)
        generator_model = payload.get("generator_model")
        if generator_model:
            return str(generator_model)

    seed_config = run_dir / "seed_sweep_config.json"
    if seed_config.exists():
        payload = _load_json(seed_config)
        explicit_model = payload.get("model")
        if explicit_model:
            return str(explicit_model)
        generator = payload.get("generator")
        variant = payload.get("variant")
        if generator == "medgemma":
            return "google/medgemma-4b-it"
        if generator and variant:
            return f"{generator}:{variant}"
        if generator:
            return str(generator)

    summary = run_dir / "summary.json"
    if summary.exists():
        payload = _load_json(summary)
        generator_model = payload.get("generator")
        if generator_model:
            return str(generator_model)

    return None


def _raw_answers_preserved(run_dir: Path, is_rag: bool) -> bool:
    if is_rag:
        return any((run_dir / name).exists() for name in ("answers_rag_std.jsonl", "answers.jsonl"))
    return (run_dir / "answers_norag.jsonl").exists() and (run_dir / "answers_gemini.jsonl").exists()


def _normalize_prompt_mode(value: object) -> str:
    text = str(value or "unknown")
    if text.startswith("PromptMode."):
        return text.split(".", 1)[1].lower()
    return text


def _arm_matching_notes(
    *,
    label: str,
    is_rag: bool,
    generator_model: Optional[str],
    judge_model: Optional[str],
    decision_support_level: Optional[str],
) -> List[str]:
    notes: List[str] = []
    if generator_model and "gemini" in generator_model.lower():
        notes.append("Gemini API generation has no exact seed parity with the frozen MedGemma line.")
        if judge_model and "gemini" in judge_model.lower():
            notes.append("Same-model-family generator/judge setup.")
    if not is_rag:
        notes.append("Diagnosis-only evaluation mirrors the frozen MedGemma no-RAG protocol.")
    if decision_support_level == "final_control_candidate":
        notes.append("Eligible for final matched-control interpretation.")
    return notes


def _build_arm_row(run_dir: Path, *, label: str, is_rag: bool) -> Dict[str, object]:
    contract_meta = _load_run_contract_metadata(run_dir)
    rows = _enrich_rows_with_answer_metadata(run_dir, _load_ragas_rows(run_dir))
    summary = _summarize_by_bucket(list(rows.values()))
    generator_model = _load_generator_model(run_dir)
    judge_model = _load_manifest_judge_model(run_dir)
    same_family = bool(
        generator_model
        and judge_model
        and "gemini" in generator_model.lower()
        and "gemini" in judge_model.lower()
    )

    return {
        "label": label,
        "run_dir": str(run_dir),
        "generator_model": generator_model,
        "judge_model": judge_model,
        "same_family_generator_judge": same_family,
        "prompt_mode": _normalize_prompt_mode(contract_meta.get("prompt_mode")),
        "prompt_contract_version": contract_meta.get("prompt_contract_version"),
        "control_type": contract_meta.get("control_type"),
        "overall_verified_diagnosis": _metric(summary, "all", "diagnosis_accuracy"),
        "leish_verified_diagnosis": _metric(summary, "leish", "diagnosis_accuracy"),
        "nonleish_verified_diagnosis": _metric(summary, "nonleish", "diagnosis_accuracy"),
        "leish_diagnosis_type": _metric(summary, "leish", "diagnosis_type_accuracy"),
        "nonleish_diagnosis_type": _metric(summary, "nonleish", "diagnosis_type_accuracy"),
        "raw_answers_preserved": _raw_answers_preserved(run_dir, is_rag=is_rag),
        "matching_notes": _arm_matching_notes(
            label=label,
            is_rag=is_rag,
            generator_model=generator_model,
            judge_model=judge_model,
            decision_support_level=contract_meta.get("decision_support_level"),
        ),
    }


def _load_compare(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing comparison JSON: {path}")
    return _load_json(path)


def _fmt(value: Optional[float]) -> str:
    return "NA" if value is None else f"{value:.4f}"


def _build_output_path(*run_dirs: Path) -> Path:
    filename = "generator_matrix_" + "_vs_".join(run_dir.name for run_dir in run_dirs) + ".json"
    if len(filename) <= 240:
        return RUNS_DIR / filename

    digest = hashlib.sha1("::".join(run_dir.name for run_dir in run_dirs).encode("utf-8")).hexdigest()[:12]
    stubs = "_vs_".join(run_dir.name[:40] for run_dir in run_dirs)
    return RUNS_DIR / f"generator_matrix_{stubs}_{digest}.json"


def build_generator_comparison_matrix(
    *,
    medgemma_rag: str,
    medgemma_norag: str,
    medgemma_compare: str,
    gemini_rag: str,
    gemini_norag: str,
    gemini_compare: str,
    gemma3_rag: str,
    gemma3_norag: str,
    gemma3_compare: str,
) -> Path:
    medgemma_rag_dir = _resolve_run_dir(medgemma_rag)
    medgemma_norag_dir = _resolve_run_dir(medgemma_norag)
    gemini_rag_dir = _resolve_run_dir(gemini_rag)
    gemini_norag_dir = _resolve_run_dir(gemini_norag)
    gemma3_rag_dir = _resolve_run_dir(gemma3_rag)
    gemma3_norag_dir = _resolve_run_dir(gemma3_norag)

    medgemma_compare_json = _load_compare(Path(medgemma_compare))
    gemini_compare_json = _load_compare(Path(gemini_compare))
    gemma3_compare_json = _load_compare(Path(gemma3_compare))

    arms = [
        _build_arm_row(medgemma_rag_dir, label="MedGemma RAG", is_rag=True),
        _build_arm_row(medgemma_norag_dir, label="MedGemma no-RAG", is_rag=False),
        _build_arm_row(gemini_rag_dir, label="Gemini RAG", is_rag=True),
        _build_arm_row(gemini_norag_dir, label="Gemini no-RAG", is_rag=False),
        _build_arm_row(gemma3_rag_dir, label="Gemma 3 4B RAG", is_rag=True),
        _build_arm_row(gemma3_norag_dir, label="Gemma 3 4B no-RAG", is_rag=False),
    ]

    medgemma_delta = medgemma_compare_json["metric_deltas"]
    gemini_delta = gemini_compare_json["metric_deltas"]
    gemma3_delta = gemma3_compare_json["metric_deltas"]
    cross_model = {
        "gemini_vs_medgemma": {
            "overall_verified_diagnosis": _safe_delta(
                arms[2]["overall_verified_diagnosis"], arms[0]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis": _safe_delta(
                arms[2]["leish_verified_diagnosis"], arms[0]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis": _safe_delta(
                arms[2]["nonleish_verified_diagnosis"], arms[0]["nonleish_verified_diagnosis"]
            ),
            "overall_verified_diagnosis_no_rag": _safe_delta(
                arms[3]["overall_verified_diagnosis"], arms[1]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis_no_rag": _safe_delta(
                arms[3]["leish_verified_diagnosis"], arms[1]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis_no_rag": _safe_delta(
                arms[3]["nonleish_verified_diagnosis"], arms[1]["nonleish_verified_diagnosis"]
            ),
        },
        "gemma3_vs_medgemma": {
            "overall_verified_diagnosis": _safe_delta(
                arms[4]["overall_verified_diagnosis"], arms[0]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis": _safe_delta(
                arms[4]["leish_verified_diagnosis"], arms[0]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis": _safe_delta(
                arms[4]["nonleish_verified_diagnosis"], arms[0]["nonleish_verified_diagnosis"]
            ),
            "overall_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["overall_verified_diagnosis"], arms[1]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["leish_verified_diagnosis"], arms[1]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["nonleish_verified_diagnosis"], arms[1]["nonleish_verified_diagnosis"]
            ),
        },
        "gemma3_vs_gemini": {
            "overall_verified_diagnosis": _safe_delta(
                arms[4]["overall_verified_diagnosis"], arms[2]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis": _safe_delta(
                arms[4]["leish_verified_diagnosis"], arms[2]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis": _safe_delta(
                arms[4]["nonleish_verified_diagnosis"], arms[2]["nonleish_verified_diagnosis"]
            ),
            "overall_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["overall_verified_diagnosis"], arms[3]["overall_verified_diagnosis"]
            ),
            "leish_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["leish_verified_diagnosis"], arms[3]["leish_verified_diagnosis"]
            ),
            "nonleish_verified_diagnosis_no_rag": _safe_delta(
                arms[5]["nonleish_verified_diagnosis"], arms[3]["nonleish_verified_diagnosis"]
            ),
        },
    }

    output_path = _build_output_path(medgemma_rag_dir, gemini_rag_dir, gemma3_rag_dir)
    report = {
        "arms": arms,
        "within_model_deltas": {
            "medgemma": medgemma_delta,
            "gemini": gemini_delta,
            "gemma3": gemma3_delta,
        },
        "cross_model_deltas": cross_model,
        "matching_notes": [
            "This comparison is maximally parallel / as apples-to-apples as possible.",
            "It is not perfectly identical because Gemini API generation does not expose exact seed parity.",
            "The Gemini pair uses a same-model-family generator/judge setup.",
            "No-RAG arms use diagnosis-only evaluation because retrieval-grounded metrics are not meaningful without external contexts.",
        ],
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    markdown_path = output_path.with_suffix(".md")
    lines = [
        "# Generator Comparison Matrix",
        "",
        "- This comparison is maximally parallel / as apples-to-apples as possible.",
        "- It is not perfectly identical: Gemini API generation has no exact seed parity with the frozen MedGemma line.",
        "- The Gemini pair uses a same-model-family generator/judge setup.",
        "- No-RAG uses diagnosis-only evaluation to mirror the frozen MedGemma no-RAG protocol.",
        "",
        "| Arm | Generator | Judge | Same Family | Prompt Mode | Prompt Contract | Overall Dx | Leish Dx | Non-Leish Dx | Leish Type | Non-Leish Type | Raw Answers |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for arm in arms:
        lines.append(
            "| {label} | {generator_model} | {judge_model} | {same_family} | {prompt_mode} | {prompt_contract} | {overall} | {leish} | {nonleish} | {leish_type} | {nonleish_type} | {raw_answers} |".format(
                label=arm["label"],
                generator_model=arm["generator_model"] or "unknown",
                judge_model=arm["judge_model"] or "unknown",
                same_family="yes" if arm["same_family_generator_judge"] else "no",
                prompt_mode=arm["prompt_mode"] or "unknown",
                prompt_contract=arm["prompt_contract_version"] or "unknown",
                overall=_fmt(arm["overall_verified_diagnosis"]),
                leish=_fmt(arm["leish_verified_diagnosis"]),
                nonleish=_fmt(arm["nonleish_verified_diagnosis"]),
                leish_type=_fmt(arm["leish_diagnosis_type"]),
                nonleish_type=_fmt(arm["nonleish_diagnosis_type"]),
                raw_answers="yes" if arm["raw_answers_preserved"] else "no",
            )
        )

    lines.extend(
        [
            "",
            "## Within-Model RAG vs No-RAG",
            f"- MedGemma overall verified diagnosis delta: {_fmt(medgemma_delta['all']['diagnosis_accuracy'])}",
            f"- MedGemma Leish verified diagnosis delta: {_fmt(medgemma_delta['leish']['diagnosis_accuracy'])}",
            f"- MedGemma Non-Leish verified diagnosis delta: {_fmt(medgemma_delta['nonleish']['diagnosis_accuracy'])}",
            f"- Gemini overall verified diagnosis delta: {_fmt(gemini_delta['all']['diagnosis_accuracy'])}",
            f"- Gemini Leish verified diagnosis delta: {_fmt(gemini_delta['leish']['diagnosis_accuracy'])}",
            f"- Gemini Non-Leish verified diagnosis delta: {_fmt(gemini_delta['nonleish']['diagnosis_accuracy'])}",
            f"- Gemma 3 overall verified diagnosis delta: {_fmt(gemma3_delta['all']['diagnosis_accuracy'])}",
            f"- Gemma 3 Leish verified diagnosis delta: {_fmt(gemma3_delta['leish']['diagnosis_accuracy'])}",
            f"- Gemma 3 Non-Leish verified diagnosis delta: {_fmt(gemma3_delta['nonleish']['diagnosis_accuracy'])}",
            "",
            "## Cross-Model Deltas",
            f"- RAG overall verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['overall_verified_diagnosis'])}",
            f"- RAG Leish verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['leish_verified_diagnosis'])}",
            f"- RAG Non-Leish verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['nonleish_verified_diagnosis'])}",
            f"- no-RAG overall verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['overall_verified_diagnosis_no_rag'])}",
            f"- no-RAG Leish verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['leish_verified_diagnosis_no_rag'])}",
            f"- no-RAG Non-Leish verified diagnosis (Gemini - MedGemma): {_fmt(cross_model['gemini_vs_medgemma']['nonleish_verified_diagnosis_no_rag'])}",
            f"- RAG overall verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['overall_verified_diagnosis'])}",
            f"- RAG Leish verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['leish_verified_diagnosis'])}",
            f"- RAG Non-Leish verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['nonleish_verified_diagnosis'])}",
            f"- no-RAG overall verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['overall_verified_diagnosis_no_rag'])}",
            f"- no-RAG Leish verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['leish_verified_diagnosis_no_rag'])}",
            f"- no-RAG Non-Leish verified diagnosis (Gemma 3 - MedGemma): {_fmt(cross_model['gemma3_vs_medgemma']['nonleish_verified_diagnosis_no_rag'])}",
            f"- RAG overall verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['overall_verified_diagnosis'])}",
            f"- RAG Leish verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['leish_verified_diagnosis'])}",
            f"- RAG Non-Leish verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['nonleish_verified_diagnosis'])}",
            f"- no-RAG overall verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['overall_verified_diagnosis_no_rag'])}",
            f"- no-RAG Leish verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['leish_verified_diagnosis_no_rag'])}",
            f"- no-RAG Non-Leish verified diagnosis (Gemma 3 - Gemini): {_fmt(cross_model['gemma3_vs_gemini']['nonleish_verified_diagnosis_no_rag'])}",
        ]
    )
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Saved generator matrix JSON to {output_path}")
    print(f"Saved generator matrix Markdown to {markdown_path}")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a six-arm generator comparison matrix")
    parser.add_argument("--medgemma-rag", required=True)
    parser.add_argument("--medgemma-norag", required=True)
    parser.add_argument("--medgemma-compare", required=True)
    parser.add_argument("--gemini-rag", required=True)
    parser.add_argument("--gemini-norag", required=True)
    parser.add_argument("--gemini-compare", required=True)
    parser.add_argument("--gemma3-rag", required=True)
    parser.add_argument("--gemma3-norag", required=True)
    parser.add_argument("--gemma3-compare", required=True)
    args = parser.parse_args()

    build_generator_comparison_matrix(
        medgemma_rag=args.medgemma_rag,
        medgemma_norag=args.medgemma_norag,
        medgemma_compare=args.medgemma_compare,
        gemini_rag=args.gemini_rag,
        gemini_norag=args.gemini_norag,
        gemini_compare=args.gemini_compare,
        gemma3_rag=args.gemma3_rag,
        gemma3_norag=args.gemma3_norag,
        gemma3_compare=args.gemma3_compare,
    )
