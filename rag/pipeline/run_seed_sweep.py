#!/usr/bin/env python3
"""Run seed sweep on a fixed retrieval run to quantify generation randomness.

This script reuses an existing `retrieval.jsonl` and creates per-seed run dirs,
then generates answers with the selected generator and optionally runs RAGAS evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from configs.prompt_mode import PromptMode
from pipeline.answer_generator import generate_answers
from pipeline.config import get_runtime_metadata
from pipeline.ragas_evaluator import run_ragas_evaluation


def _parse_metric_value(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        v = float(value)
        return None if math.isnan(v) else v
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        try:
            v = float(s)
        except ValueError:
            return None
        return None if math.isnan(v) else v
    return None


def _copy_base_artifacts(src: Path, dst: Path) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for name in ["retrieval.jsonl", "queries.json", "run_config.json", "summary.json", "metrics_per_query.csv"]:
        p = src / name
        if p.exists():
            shutil.copy2(p, dst / name)


def _load_inherited_retrieval_top_k(base_run_dir: Path) -> Optional[int]:
    candidate_paths = [base_run_dir / "run_config.json", base_run_dir / "summary.json"]
    for candidate_path in candidate_paths:
        if not candidate_path.exists():
            continue
        with open(candidate_path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        direct_value = payload.get("retrieval_top_k")
        if direct_value is not None:
            return int(direct_value)
        experiment_controls = payload.get("experiment_controls") or {}
        inherited_value = experiment_controls.get("retrieval_top_k")
        if inherited_value is not None:
            return int(inherited_value)
    return None


def run_seed_sweep(
    base_run_dir: Path,
    seeds: List[int],
    generator: str,
    model: str | None,
    variant: str,
    prompt_mode: PromptMode,
    force_rag: bool,
    output_file: str,
    delay: float,
    evaluate: bool,
    judge_model: str | None,
    use_batch_api: bool,
    batch_poll_seconds: float,
    batch_timeout_seconds: int,
    judge_batch_api: bool,
    judge_batch_poll_seconds: float,
    judge_batch_timeout_seconds: int,
    eval_resume: bool,
    evaluate_retrieval_metrics: bool,
    strip_query_images: bool,
    ablation_scope: str,
    context_k: Optional[int],
    ordering_mode: str,
    use_context_image_tensors: bool,
    support_image_tensor_budget: int,
) -> List[Path]:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dirs: List[Path] = []
    inherited_retrieval_top_k = _load_inherited_retrieval_top_k(base_run_dir)

    for seed in seeds:
        run_name = f"{base_run_dir.name}_seed{seed}_{generator}_{ts}"
        run_dir = base_run_dir.parent / run_name
        _copy_base_artifacts(base_run_dir, run_dir)

        with open(run_dir / "seed_sweep_config.json", "w") as f:
            json.dump(
                {
                    "base_run": str(base_run_dir),
                    "seed": seed,
                    "generator": generator,
                    "model": model,
                    "variant": variant,
                    "prompt_mode": str(prompt_mode),
                    "force_rag": force_rag,
                    "output_file": output_file,
                    "judge_model": judge_model,
                    "evaluate": evaluate,
                    "use_batch_api": use_batch_api,
                    "batch_poll_seconds": batch_poll_seconds,
                    "batch_timeout_seconds": batch_timeout_seconds,
                    "judge_batch_api": judge_batch_api,
                    "judge_batch_poll_seconds": judge_batch_poll_seconds,
                    "judge_batch_timeout_seconds": judge_batch_timeout_seconds,
                    "strip_query_images": strip_query_images,
                    "ablation_scope": ablation_scope,
                    "context_k": context_k,
                    "ordering_mode": ordering_mode,
                    "use_context_image_tensors": use_context_image_tensors,
                    "support_image_tensor_budget": support_image_tensor_budget,
                    "retrieval_top_k_inherited": inherited_retrieval_top_k,
                    "retrieval_top_k_inherited_from_base_run": inherited_retrieval_top_k is not None,
                    "runtime_metadata": get_runtime_metadata(),
                },
                f,
                indent=2,
            )

        generate_answers(
            run_dir,
            retrieval_file="retrieval.jsonl",
            generator_type=generator,
            model_variant=variant,
            prompt_mode=prompt_mode,
            force_rag=force_rag,
            output_file=output_file,
            model=model,
            random_seed=seed,
            use_batch_api=use_batch_api,
            batch_poll_seconds=batch_poll_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
            strip_query_images=strip_query_images,
            ablation_scope=ablation_scope,
            context_k=context_k,
            ordering_mode=ordering_mode,
            use_context_image_tensors=use_context_image_tensors,
            support_image_tensor_budget=support_image_tensor_budget,
        )

        if evaluate:
            run_ragas_evaluation(
                run_dir,
                answers_file=output_file,
                judge_model=judge_model,
                delay_seconds=delay,
                resume=eval_resume,
                diagnosis_batch_api=judge_batch_api,
                diagnosis_batch_poll_seconds=judge_batch_poll_seconds,
                diagnosis_batch_timeout_seconds=judge_batch_timeout_seconds,
                evaluate_retrieval_metrics=evaluate_retrieval_metrics,
            )

        out_dirs.append(run_dir)

    return out_dirs


def print_summary(run_dirs: List[Path]) -> None:
    print("\n=== Seed Sweep Summary ===")
    print("run_id\tn\tl3_top1\ttop3_hit\tdiag\ttype")
    for run_dir in run_dirs:
        ragas = run_dir / "ragas.jsonl"
        if not ragas.exists():
            print(f"{run_dir.name}\tNA\tNA\tNA\tNA\tNA")
            continue
        rows = [json.loads(line) for line in ragas.open() if line.strip()]
        n = len(rows)
        if n == 0:
            print(f"{run_dir.name}\t0\tNA\tNA\tNA\tNA")
            continue
        l3_vals = [_parse_metric_value(r.get("l3_top1_correct")) for r in rows]
        top3_vals = [_parse_metric_value(r.get("top3_hit")) for r in rows]
        diag_vals = [_parse_metric_value(r.get("diagnosis_accuracy")) for r in rows]
        type_vals = [_parse_metric_value(r.get("diagnosis_type_accuracy")) for r in rows]
        l3_vals = [v for v in l3_vals if v is not None]
        top3_vals = [v for v in top3_vals if v is not None]
        diag_vals = [v for v in diag_vals if v is not None]
        type_vals = [v for v in type_vals if v is not None]
        l3 = sum(l3_vals) / len(l3_vals) if l3_vals else float("nan")
        top3 = sum(top3_vals) / len(top3_vals) if top3_vals else float("nan")
        diag = sum(diag_vals) / len(diag_vals) if diag_vals else float("nan")
        dtype = sum(type_vals) / len(type_vals) if type_vals else float("nan")
        l3_str = f"{l3:.4f}" if not math.isnan(l3) else "NA"
        top3_str = f"{top3:.4f}" if not math.isnan(top3) else "NA"
        diag_str = f"{diag:.4f}" if not math.isnan(diag) else "NA"
        type_str = f"{dtype:.4f}" if not math.isnan(dtype) else "NA"
        print(f"{run_dir.name}\t{n}\t{l3_str}\t{top3_str}\t{diag_str}\t{type_str}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run fixed-retrieval seed sweep")
    parser.add_argument("--base-run", required=True, help="Run directory containing retrieval.jsonl")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--generator", default="gemma3")
    parser.add_argument(
        "--model",
        default=None,
        help="Explicit model ID: Gemini model for --generator gemini, or local HF model ID for gemma3/gemma4/medgemma",
    )
    parser.add_argument("--variant", default="4b")
    parser.add_argument(
        "--prompt-mode",
        choices=["strict_context", "balanced", "no_context"],
        default="balanced",
        help="Prompt mode used during generation (must match base run for fair seed analysis)",
    )
    parser.add_argument(
        "--force-rag",
        action="store_true",
        help="Force RAG mode in generator (recommended for controlled RAG robustness sweeps)",
    )
    parser.add_argument(
        "--output-file",
        default="answers_rag_std.jsonl",
        help="Answers filename to write and evaluate",
    )
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument("--judge-model", default=None, help="Optional judge model passed to RAGAS evaluator")
    parser.add_argument("--no-eval", action="store_true", help="Generate only, skip RAGAS evaluation")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use Google Batch API for Gemini generation")
    parser.add_argument("--batch-poll-seconds", type=float, default=10.0,
                        help="Polling interval for Gemini batch jobs")
    parser.add_argument("--batch-timeout-seconds", type=int, default=3600,
                        help="Timeout for Gemini batch jobs")
    parser.add_argument("--judge-batch-api", action="store_true",
                        help="Use Gemini Batch API for diagnosis judge evaluation")
    parser.add_argument("--judge-batch-poll-seconds", type=float, default=10.0,
                        help="Polling interval for diagnosis judge batch jobs")
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=7200,
                        help="Timeout for diagnosis judge batch jobs")
    parser.add_argument(
        "--eval-resume",
        action="store_true",
        help="Resume existing ragas.jsonl for the run dir (default: fresh evaluation)",
    )
    parser.add_argument(
        "--diagnosis-only-eval",
        action="store_true",
        help="Disable retrieval metrics in RAGAS and evaluate diagnosis metrics only",
    )
    parser.add_argument(
        "--strip-query-images",
        action="store_true",
        help="Generator-side hard image-off ablation: remove query_images before generation",
    )
    parser.add_argument(
        "--ablation-scope",
        default="",
        help="Optional ablation scope label stored in generated artifacts",
    )
    parser.add_argument(
        "--context-k",
        type=int,
        default=None,
        help="Optional explicit prompt context budget passed through to generation",
    )
    parser.add_argument(
        "--ordering-mode",
        choices=["image_first", "text_first", "interleaved"],
        default="image_first",
        help="Gemma4 multimodal ordering mode for patient/query images",
    )
    parser.add_argument(
        "--use-context-image-tensors",
        action="store_true",
        help="Pass retrieved support images as real Gemma4 image tensors instead of prompt-only references",
    )
    parser.add_argument(
        "--support-image-tensor-budget",
        type=int,
        default=0,
        help="Maximum number of retrieved support images to attach as Gemma4 image tensors",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_dirs = run_seed_sweep(
        base_run_dir=Path(args.base_run),
        seeds=args.seeds,
        generator=args.generator,
        model=args.model,
        variant=args.variant,
        prompt_mode=PromptMode(args.prompt_mode),
        force_rag=args.force_rag,
        output_file=args.output_file,
        delay=args.delay,
        evaluate=not args.no_eval,
        judge_model=args.judge_model,
        use_batch_api=args.batch_api,
        batch_poll_seconds=args.batch_poll_seconds,
        batch_timeout_seconds=args.batch_timeout_seconds,
        judge_batch_api=args.judge_batch_api,
        judge_batch_poll_seconds=args.judge_batch_poll_seconds,
        judge_batch_timeout_seconds=args.judge_batch_timeout_seconds,
        eval_resume=args.eval_resume,
        evaluate_retrieval_metrics=not args.diagnosis_only_eval,
        strip_query_images=args.strip_query_images,
        ablation_scope=(args.ablation_scope or ("generator_only_image_strip" if args.strip_query_images else "")),
        context_k=args.context_k,
        ordering_mode=args.ordering_mode,
        use_context_image_tensors=args.use_context_image_tensors,
        support_image_tensor_budget=args.support_image_tensor_budget,
    )
    print_summary(run_dirs)
