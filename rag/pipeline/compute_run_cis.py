#!/usr/bin/env python3
"""Compute confidence intervals for diagnosis metrics from RAGAS run outputs.

Features:
- Per-run metric mean + bootstrap CI
- Paired run comparison (delta mean + paired bootstrap CI)

Usage examples:
  python -m rag.pipeline.compute_run_cis --run rag/runs/run_a --run rag/runs/run_b
  python -m rag.pipeline.compute_run_cis --run rag/runs/rag --run rag/runs/norag --paired
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Tuple


@dataclass
class MetricSummary:
    n_total: int
    n_used: int
    n_missing: int
    mean: float
    ci_low: float
    ci_high: float


def load_ragas_rows(run_dir: Path) -> List[dict]:
    ragas_path = run_dir / "ragas.jsonl"
    if not ragas_path.exists():
        raise FileNotFoundError(f"No ragas.jsonl in {run_dir}")
    with open(ragas_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def parse_metric_value(value: object) -> Optional[float]:
    """Convert a raw metric value to float; return None for missing/invalid values."""
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        val = float(value)
        if math.isnan(val):
            return None
        return val
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        try:
            val = float(s)
        except ValueError:
            return None
        if math.isnan(val):
            return None
        return val
    return None


def bootstrap_mean_ci(
    values: List[float],
    confidence: float = 0.95,
    n_boot: int = 10000,
    seed: int = 42,
) -> MetricSummary:
    if not values:
        return MetricSummary(
            n_total=0,
            n_used=0,
            n_missing=0,
            mean=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
        )

    rng = Random(seed)
    n = len(values)
    mean_val = sum(values) / n

    boot_means: List[float] = []
    for _ in range(n_boot):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        boot_means.append(sum(sample) / n)

    boot_means.sort()
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n_boot)
    hi_idx = int((1.0 - alpha) * n_boot) - 1
    lo_idx = max(0, min(lo_idx, n_boot - 1))
    hi_idx = max(0, min(hi_idx, n_boot - 1))

    return MetricSummary(
        n_total=n,
        n_used=n,
        n_missing=0,
        mean=mean_val,
        ci_low=boot_means[lo_idx],
        ci_high=boot_means[hi_idx],
    )


def extract_metric_values(
    rows: List[dict],
    metric: str,
    missing_policy: str,
) -> Tuple[List[float], int, int, int]:
    """Extract metric values according to missing policy.

    Returns:
        (values, n_total, n_used, n_missing)
    """
    values: List[float] = []
    n_total = len(rows)
    n_missing = 0

    for row in rows:
        raw = row.get(metric)
        parsed = parse_metric_value(raw)
        if parsed is None:
            n_missing += 1
            if missing_policy == "zero":
                values.append(0.0)
            elif missing_policy == "error":
                qid = row.get("qid", "<unknown_qid>")
                raise ValueError(
                    f"Missing/invalid metric '{metric}' at qid='{qid}' with value={raw!r}"
                )
            # "skip" => do nothing
        else:
            values.append(parsed)

    return values, n_total, len(values), n_missing


def summarize_run(
    run_dir: Path,
    metric_keys: List[str],
    confidence: float,
    n_boot: int,
    seed: int,
    missing_policy: str,
) -> Dict[str, MetricSummary]:
    rows = load_ragas_rows(run_dir)
    out: Dict[str, MetricSummary] = {}
    for metric in metric_keys:
        vals, n_total, n_used, n_missing = extract_metric_values(rows, metric, missing_policy)
        summary = bootstrap_mean_ci(vals, confidence=confidence, n_boot=n_boot, seed=seed)
        summary.n_total = n_total
        summary.n_used = n_used
        summary.n_missing = n_missing
        out[metric] = summary
    return out


def paired_delta_ci(
    run_a: Path,
    run_b: Path,
    metric: str,
    confidence: float,
    n_boot: int,
    seed: int,
    missing_policy: str,
) -> Tuple[int, int, float, float, float]:
    rows_a = {r["qid"]: r for r in load_ragas_rows(run_a)}
    rows_b = {r["qid"]: r for r in load_ragas_rows(run_b)}

    common_qids = sorted(set(rows_a.keys()) & set(rows_b.keys()))
    if not common_qids:
        return 0, 0, float("nan"), float("nan"), float("nan")

    deltas: List[float] = []
    n_missing_pairs = 0
    for qid in common_qids:
        va = parse_metric_value(rows_a[qid].get(metric))
        vb = parse_metric_value(rows_b[qid].get(metric))
        if va is None or vb is None:
            n_missing_pairs += 1
            if missing_policy == "zero":
                va = 0.0 if va is None else va
                vb = 0.0 if vb is None else vb
            elif missing_policy == "error":
                raise ValueError(
                    f"Missing/invalid paired metric '{metric}' for qid='{qid}': "
                    f"run_a={rows_a[qid].get(metric)!r}, run_b={rows_b[qid].get(metric)!r}"
                )
            else:
                continue
        deltas.append(float(va) - float(vb))

    if not deltas:
        return len(common_qids), n_missing_pairs, float("nan"), float("nan"), float("nan")

    summary = bootstrap_mean_ci(deltas, confidence=confidence, n_boot=n_boot, seed=seed)
    return len(common_qids), n_missing_pairs, summary.mean, summary.ci_low, summary.ci_high


def paired_permutation_pvalue(
    run_a: Path,
    run_b: Path,
    metric: str,
    seed: int,
    n_perm: int,
    missing_policy: str,
) -> Tuple[int, int, float]:
    """Two-sided paired permutation p-value via random sign flips on paired deltas."""
    rows_a = {r["qid"]: r for r in load_ragas_rows(run_a)}
    rows_b = {r["qid"]: r for r in load_ragas_rows(run_b)}
    common_qids = sorted(set(rows_a.keys()) & set(rows_b.keys()))

    deltas: List[float] = []
    n_missing_pairs = 0
    for qid in common_qids:
        va = parse_metric_value(rows_a[qid].get(metric))
        vb = parse_metric_value(rows_b[qid].get(metric))
        if va is None or vb is None:
            n_missing_pairs += 1
            if missing_policy == "zero":
                va = 0.0 if va is None else va
                vb = 0.0 if vb is None else vb
            elif missing_policy == "error":
                raise ValueError(
                    f"Missing/invalid paired metric '{metric}' for qid='{qid}': "
                    f"run_a={rows_a[qid].get(metric)!r}, run_b={rows_b[qid].get(metric)!r}"
                )
            else:
                continue
        deltas.append(float(va) - float(vb))

    if not deltas:
        return len(common_qids), n_missing_pairs, float("nan")

    rng = Random(seed)
    observed = abs(sum(deltas) / len(deltas))
    extreme = 0
    for _ in range(n_perm):
        signed = [d if rng.random() < 0.5 else -d for d in deltas]
        stat = abs(sum(signed) / len(signed))
        if stat >= observed:
            extreme += 1

    # add-one smoothing
    p_value = (extreme + 1) / (n_perm + 1)
    return len(common_qids), n_missing_pairs, p_value


def holm_bonferroni(pvals: List[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values in original order."""
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted_sorted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = (m - rank) * pvals[idx]

    # enforce monotonicity on sorted adjusted values
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_sorted = [min(1.0, x) for x in adjusted_sorted]

    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute bootstrap confidence intervals for run metrics")
    parser.add_argument("--run", action="append", required=True, help="Run directory path (repeatable)")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["diagnosis_accuracy", "diagnosis_type_accuracy"],
        help="Metric keys in ragas.jsonl",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-policy",
        choices=["skip", "error", "zero"],
        default="skip",
        help="How to handle missing/invalid metric values (default: skip)",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="Also compute paired deltas for the first two --run values (run1 - run2)",
    )
    parser.add_argument(
        "--paired-permutation",
        action="store_true",
        help="Also compute paired permutation p-values for first two --run values",
    )
    parser.add_argument(
        "--permutations",
        type=int,
        default=10000,
        help="Number of random sign-flip permutations (default: 10000)",
    )
    parser.add_argument(
        "--holm",
        action="store_true",
        help="Apply Holm-Bonferroni correction to paired permutation p-values across metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dirs = [Path(r) for r in args.run]

    print("=== Per-run bootstrap CI ===")
    print("run\tmetric\tn_total\tn_used\tn_missing\tmean\tci_low\tci_high")
    for run_dir in run_dirs:
        summaries = summarize_run(
            run_dir=run_dir,
            metric_keys=args.metrics,
            confidence=args.confidence,
            n_boot=args.bootstrap,
            seed=args.seed,
            missing_policy=args.missing_policy,
        )
        for metric in args.metrics:
            s = summaries[metric]
            print(
                f"{run_dir.name}\t{metric}\t{s.n_total}\t{s.n_used}\t{s.n_missing}\t"
                f"{s.mean:.4f}\t{s.ci_low:.4f}\t{s.ci_high:.4f}"
            )

    if args.paired and len(run_dirs) >= 2:
        a, b = run_dirs[0], run_dirs[1]
        print("\n=== Paired delta bootstrap CI ===")
        print("comparison\tmetric\tn_common\tn_missing_pairs\tdelta_mean\tci_low\tci_high")
        for metric in args.metrics:
            n_common, n_missing_pairs, mean_delta, lo, hi = paired_delta_ci(
                run_a=a,
                run_b=b,
                metric=metric,
                confidence=args.confidence,
                n_boot=args.bootstrap,
                seed=args.seed,
                missing_policy=args.missing_policy,
            )
            print(
                f"{a.name}-{b.name}\t{metric}\t{n_common}\t{n_missing_pairs}\t"
                f"{mean_delta:.4f}\t{lo:.4f}\t{hi:.4f}"
            )

    if args.paired_permutation and len(run_dirs) >= 2:
        a, b = run_dirs[0], run_dirs[1]
        print("\n=== Paired permutation p-values ===")
        print("comparison\tmetric\tn_common\tn_missing_pairs\tp_value\tp_holm")

        raw_pvals: List[float] = []
        rows: List[Tuple[str, int, int, float]] = []
        for metric in args.metrics:
            n_common, n_missing_pairs, p_val = paired_permutation_pvalue(
                run_a=a,
                run_b=b,
                metric=metric,
                seed=args.seed,
                n_perm=args.permutations,
                missing_policy=args.missing_policy,
            )
            rows.append((metric, n_common, n_missing_pairs, p_val))
            raw_pvals.append(p_val)

        adjusted = holm_bonferroni(raw_pvals) if args.holm else [float("nan")] * len(raw_pvals)
        for (metric, n_common, n_missing_pairs, p_val), p_adj in zip(rows, adjusted):
            p_adj_str = f"{p_adj:.6f}" if not math.isnan(p_adj) else "NA"
            print(
                f"{a.name}-{b.name}\t{metric}\t{n_common}\t{n_missing_pairs}\t"
                f"{p_val:.6f}\t{p_adj_str}"
            )


if __name__ == "__main__":
    main()