#!/usr/bin/env python3
"""Run controlled RAG context ablations for leish23 experiments.

Ablation design:
- Build one shared retrieval run (top contexts from retriever)
- Create variant runs by filtering/truncating retrieval contexts
- Generate answers + run RAGAS for each variant

This avoids re-running retrieval for every k/constraint condition and keeps
all downstream settings identical for fair comparison.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from configs.prompt_mode import PromptMode
from pipeline.answer_generator import generate_answers
from pipeline.config import SPLIT_DIR
from pipeline.ragas_evaluator import run_ragas_evaluation
from pipeline.run_multimodal_eval import run_multimodal_evaluation


@dataclass
class Variant:
    name: str
    context_k: int
    constraint: str  # "none" | "qrels_only"


DEFAULT_VARIANTS: List[Variant] = [
    Variant(name="k3", context_k=3, constraint="none"),
    Variant(name="k5", context_k=5, constraint="none"),
    Variant(name="k8", context_k=8, constraint="none"),
    Variant(name="k5_qrels", context_k=5, constraint="qrels_only"),
]


def _load_jsonl(path: Path) -> List[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _write_jsonl(path: Path, rows: List[dict]) -> None:
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _assert_active_split_is_23() -> None:
    test_path = SPLIT_DIR / "test.jsonl"
    eval_path = SPLIT_DIR / "eval_queries_v163.jsonl"

    test_n = sum(1 for line in test_path.open() if line.strip())
    eval_rows = _load_jsonl(eval_path)
    eval_mm_n = sum(1 for row in eval_rows if row.get("query_type") == "Q1_Q3_multimodal_diagnosis")

    if test_n != 23 or eval_mm_n != 23:
        raise RuntimeError(
            f"Active split is not pure leish23 multimodal: test={test_n}, eval_multimodal={eval_mm_n}"
        )


def _load_qrels_case_map(qrels_file: str) -> Dict[str, set]:
    with open(SPLIT_DIR / qrels_file) as f:
        raw = json.load(f)
    return {case_id: set(doc_scores.keys()) for case_id, doc_scores in raw.items()}


def _build_variant_retrieval(
    base_rows: List[dict],
    qrels_case_map: Dict[str, set],
    variant: Variant,
) -> List[dict]:
    out = []
    for row in base_rows:
        contexts = row.get("contexts", [])
        if not contexts:
            out.append(row)
            continue

        case_id = row["qid"].split("::")[0]
        filtered = contexts

        if variant.constraint == "qrels_only":
            allowed = qrels_case_map.get(case_id, set())
            filtered = [ctx for ctx in contexts if ctx.get("doc_id") in allowed]

        new_row = dict(row)
        new_row["contexts"] = filtered
        out.append(new_row)

    return out


def _copy_base_artifacts(base_run_dir: Path, dst_run_dir: Path) -> None:
    dst_run_dir.mkdir(parents=True, exist_ok=True)
    for filename in ["run_config.json", "queries.json", "summary.json", "metrics_per_query.csv"]:
        src = base_run_dir / filename
        if src.exists():
            shutil.copy2(src, dst_run_dir / filename)


def run_ablation(
    run_prefix: str,
    qrels_file: str,
    generator_type: str,
    model_variant: str,
    delay_seconds: float,
    variants: List[Variant],
    agentic_lite: bool,
) -> List[Path]:
    _assert_active_split_is_23()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_run_id = f"{run_prefix}_base_{ts}"
    print(f"[1/3] Building shared retrieval run: {base_run_id}")

    base_run_dir = run_multimodal_evaluation(
        qrels_file=qrels_file,
        run_id=base_run_id,
        method="hybrid",
        query_types=["Q1_Q3_multimodal_diagnosis"],
        k_values=[5],
        image_search_mode="captions",
        rerank=False,
        agentic_lite=agentic_lite,
    )

    base_rows = _load_jsonl(base_run_dir / "retrieval.jsonl")
    qrels_case_map = _load_qrels_case_map(qrels_file)

    print("[2/3] Creating variant retrieval files")
    output_dirs: List[Path] = []
    for variant in variants:
        variant_id = f"{run_prefix}_{variant.name}_{ts}"
        variant_dir = base_run_dir.parent / variant_id
        _copy_base_artifacts(base_run_dir, variant_dir)

        variant_rows = _build_variant_retrieval(base_rows, qrels_case_map, variant)
        _write_jsonl(variant_dir / "retrieval.jsonl", variant_rows)

        with open(variant_dir / "ablation_config.json", "w") as f:
            json.dump(
                {
                    "base_run_id": base_run_id,
                    "variant": {
                        "name": variant.name,
                        "context_k": variant.context_k,
                        "constraint": variant.constraint,
                    },
                    "generator_type": generator_type,
                    "model_variant": model_variant,
                    "qrels_file": qrels_file,
                },
                f,
                indent=2,
            )

        output_dirs.append(variant_dir)

    print("[3/3] Generating answers + running RAGAS for each variant")
    for variant_dir in output_dirs:
        print(f"  -> {variant_dir.name}")
        generate_answers(
            variant_dir,
            retrieval_file="retrieval.jsonl",
            generator_type=generator_type,
            model_variant=model_variant,
            prompt_mode=PromptMode("balanced"),
            force_rag=True,
            output_file="answers_rag_std.jsonl",
            context_k=variant.context_k,
        )
        run_ragas_evaluation(
            variant_dir,
            answers_file="answers_rag_std.jsonl",
            delay_seconds=delay_seconds,
        )

    return output_dirs


def print_summary_table(run_dirs: List[Path]) -> None:
    print("\n=== Context Ablation Summary ===")
    print("run_id\tn\tdiag\ttype")
    for run_dir in run_dirs:
        ragas_path = run_dir / "ragas.jsonl"
        if not ragas_path.exists():
            print(f"{run_dir.name}\t0\tNA\tNA")
            continue
        rows = _load_jsonl(ragas_path)
        n = len(rows)
        if n == 0:
            print(f"{run_dir.name}\t0\tNA\tNA")
            continue
        diag = sum((row.get("diagnosis_accuracy") or 0) for row in rows) / n
        dtype = sum((row.get("diagnosis_type_accuracy") or 0) for row in rows) / n
        print(f"{run_dir.name}\t{n}\t{diag:.4f}\t{dtype:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run context-k and qrels-constrained RAG ablations")
    parser.add_argument("--run-prefix", default="gemma3_4b_rag_v163_leish23_ablate")
    parser.add_argument("--qrels", default="qrels_clinical_strict.json")
    parser.add_argument("--generator", default="gemma3")
    parser.add_argument("--variant", default="4b")
    parser.add_argument("--delay", type=float, default=1.5)
    parser.add_argument(
        "--agentic-lite",
        action="store_true",
        help="Enable one-step retrieve-evaluate-refine retrieval loop in base retrieval run",
    )
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help=(
            "Optional variant specs as name:k:constraint, e.g. "
            "k3:3:none k5:5:none k5q:5:qrels_only"
        ),
    )
    return parser.parse_args()


def _parse_variants(specs: List[str] | None) -> List[Variant]:
    if not specs:
        return DEFAULT_VARIANTS

    parsed: List[Variant] = []
    for spec in specs:
        try:
            name, k_str, constraint = spec.split(":")
            parsed.append(Variant(name=name, context_k=int(k_str), constraint=constraint))
        except Exception as exc:
            raise ValueError(
                f"Invalid variant spec '{spec}'. Expected name:k:constraint"
            ) from exc
    return parsed


if __name__ == "__main__":
    args = parse_args()
    variants = _parse_variants(args.variants)
    run_dirs = run_ablation(
        run_prefix=args.run_prefix,
        qrels_file=args.qrels,
        generator_type=args.generator,
        model_variant=args.variant,
        delay_seconds=args.delay,
        variants=variants,
        agentic_lite=args.agentic_lite,
    )
    print_summary_table(run_dirs)
