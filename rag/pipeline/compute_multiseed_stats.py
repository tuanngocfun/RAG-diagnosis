#!/usr/bin/env python3
"""Compute multi-seed, case-aware statistics for grouped RAGAS runs.

Each group is a named set of run directories (typically one directory per seed).
The script reports per-group means with both case bootstrap CIs and hierarchical
bootstrap CIs, then performs paired comparisons between the first two groups.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from random import Random
from typing import Dict, List, Optional, Tuple


RUNS_ROOT = Path("rag/runs")


@dataclass
class GroupMetricSummary:
    group: str
    metric: str
    n_runs: int
    n_qids: int
    mean: float
    case_ci_low: float
    case_ci_high: float
    hier_ci_low: float
    hier_ci_high: float


def parse_metric_value(value: object) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        val = float(value)
        return None if math.isnan(val) else val
    if isinstance(value, str):
        s = value.strip().lower()
        if s in {"", "nan", "none", "null", "na", "n/a"}:
            return None
        try:
            val = float(s)
        except ValueError:
            return None
        return None if math.isnan(val) else val
    return None


def load_run_rows(run_dir: Path) -> Dict[str, dict]:
    ragas_path = run_dir / "ragas.jsonl"
    if not ragas_path.exists():
        hint = ""
        if not run_dir.exists():
            hint = " (directory not found; pass full path or a run name under rag/runs)"
        raise FileNotFoundError(f"No ragas.jsonl in {run_dir}{hint}")
    rows: Dict[str, dict] = {}
    with open(ragas_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            qid = row.get("qid")
            if qid:
                rows[qid] = row
    return rows


def percentile_ci(values: List[float], confidence: float) -> Tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    ordered = sorted(values)
    n = len(ordered)
    alpha = (1.0 - confidence) / 2.0
    lo_idx = int(alpha * n)
    hi_idx = int((1.0 - alpha) * n) - 1
    lo_idx = max(0, min(lo_idx, n - 1))
    hi_idx = max(0, min(hi_idx, n - 1))
    return ordered[lo_idx], ordered[hi_idx]


def build_group_values(
    run_dirs: List[Path],
    metric: str,
    missing_policy: str,
) -> Dict[str, List[float]]:
    qid_values: Dict[str, List[float]] = {}
    for run_dir in run_dirs:
        rows = load_run_rows(run_dir)
        for qid, row in rows.items():
            val = parse_metric_value(row.get(metric))
            if val is None:
                if missing_policy == "zero":
                    val = 0.0
                elif missing_policy == "error":
                    raise ValueError(
                        f"Missing/invalid metric '{metric}' in {run_dir} for qid='{qid}'"
                    )
                else:
                    continue
            qid_values.setdefault(qid, []).append(float(val))
    return qid_values


def case_bootstrap_ci(
    qid_values: Dict[str, List[float]],
    confidence: float,
    n_boot: int,
    seed: int,
) -> Tuple[float, float, float]:
    qids = sorted(qid_values.keys())
    if not qids:
        return float("nan"), float("nan"), float("nan")
    qid_means = [sum(qid_values[q]) / len(qid_values[q]) for q in qids]
    mean_val = sum(qid_means) / len(qid_means)

    rng = Random(seed)
    n = len(qid_means)
    boot = []
    for _ in range(n_boot):
        sample = [qid_means[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    lo, hi = percentile_ci(boot, confidence)
    return mean_val, lo, hi


def hierarchical_bootstrap_ci(
    qid_values: Dict[str, List[float]],
    confidence: float,
    n_boot: int,
    seed: int,
) -> Tuple[float, float]:
    qids = sorted(qid_values.keys())
    if not qids:
        return float("nan"), float("nan")

    rng = Random(seed)
    n_q = len(qids)
    boot = []
    for _ in range(n_boot):
        sampled_qids = [qids[rng.randrange(n_q)] for _ in range(n_q)]
        vals = []
        for qid in sampled_qids:
            per_seed = qid_values[qid]
            vals.append(per_seed[rng.randrange(len(per_seed))])
        boot.append(sum(vals) / len(vals))
    return percentile_ci(boot, confidence)


def paired_delta_stats(
    group_a: Dict[str, List[float]],
    group_b: Dict[str, List[float]],
    confidence: float,
    n_boot: int,
    n_perm: int,
    seed: int,
) -> Tuple[int, float, float, float, float]:
    common_qids = sorted(set(group_a.keys()) & set(group_b.keys()))
    if not common_qids:
        return 0, float("nan"), float("nan"), float("nan"), float("nan")

    per_qid_delta = [
        (sum(group_a[q]) / len(group_a[q])) - (sum(group_b[q]) / len(group_b[q]))
        for q in common_qids
    ]
    observed = sum(per_qid_delta) / len(per_qid_delta)

    rng = Random(seed)
    n = len(per_qid_delta)
    boot = []
    for _ in range(n_boot):
        sample = [per_qid_delta[rng.randrange(n)] for _ in range(n)]
        boot.append(sum(sample) / n)
    ci_low, ci_high = percentile_ci(boot, confidence)

    extreme = 0
    abs_obs = abs(observed)
    for _ in range(n_perm):
        signed = [d if rng.random() < 0.5 else -d for d in per_qid_delta]
        stat = abs(sum(signed) / len(signed))
        if stat >= abs_obs:
            extreme += 1
    p_val = (extreme + 1) / (n_perm + 1)
    return len(common_qids), observed, ci_low, ci_high, p_val


def holm_bonferroni(pvals: List[float]) -> List[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvals[i])
    adjusted_sorted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted_sorted[rank] = (m - rank) * pvals[idx]
    for i in range(1, m):
        adjusted_sorted[i] = max(adjusted_sorted[i], adjusted_sorted[i - 1])
    adjusted_sorted = [min(1.0, x) for x in adjusted_sorted]
    adjusted = [0.0] * m
    for rank, idx in enumerate(order):
        adjusted[idx] = adjusted_sorted[rank]
    return adjusted


def parse_group_arg(value: str) -> Tuple[str, List[Path]]:
    if "=" not in value:
        raise ValueError(f"Invalid --group '{value}'. Expected format name=run1,run2,...")
    name, runs_str = value.split("=", 1)
    name = name.strip()
    raw_paths = [p.strip() for p in runs_str.split(",") if p.strip()]
    run_paths: List[Path] = []
    for raw in raw_paths:
        p = Path(raw)
        # Convenience: allow bare run names by resolving under rag/runs.
        if not p.exists() and not p.is_absolute() and "/" not in raw:
            candidate = RUNS_ROOT / raw
            if candidate.exists():
                p = candidate
        run_paths.append(p)
    if not name:
        raise ValueError(f"Invalid group name in '{value}'")
    if not run_paths:
        raise ValueError(f"No runs provided for group '{name}'")
    return name, run_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute grouped multi-seed statistics from ragas.jsonl runs")
    parser.add_argument(
        "--group",
        action="append",
        required=True,
        help="Group definition: name=run_dir1,run_dir2,... (repeatable)",
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=["l3_top1_correct", "top3_hit", "diagnosis_accuracy", "diagnosis_type_accuracy"],
        help="Metric keys from ragas.jsonl",
    )
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--hier-bootstrap", type=int, default=5000)
    parser.add_argument("--permutations", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--missing-policy",
        choices=["skip", "error", "zero"],
        default="skip",
        help="How to handle missing/invalid metric values",
    )
    parser.add_argument(
        "--holm",
        action="store_true",
        help="Apply Holm correction to paired p-values across metrics",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    groups = [parse_group_arg(g) for g in args.group]

    cache: Dict[Tuple[str, str], Dict[str, List[float]]] = {}

    print("=== Grouped multi-seed summaries ===")
    print("group\tmetric\tn_runs\tn_qids\tmean\tcase_ci_low\tcase_ci_high\thier_ci_low\thier_ci_high")
    for group_name, run_dirs in groups:
        for metric in args.metrics:
            qid_values = build_group_values(
                run_dirs=run_dirs,
                metric=metric,
                missing_policy=args.missing_policy,
            )
            cache[(group_name, metric)] = qid_values
            mean_val, case_lo, case_hi = case_bootstrap_ci(
                qid_values=qid_values,
                confidence=args.confidence,
                n_boot=args.bootstrap,
                seed=args.seed,
            )
            hier_lo, hier_hi = hierarchical_bootstrap_ci(
                qid_values=qid_values,
                confidence=args.confidence,
                n_boot=args.hier_bootstrap,
                seed=args.seed + 1,
            )
            print(
                f"{group_name}\t{metric}\t{len(run_dirs)}\t{len(qid_values)}\t"
                f"{mean_val:.4f}\t{case_lo:.4f}\t{case_hi:.4f}\t{hier_lo:.4f}\t{hier_hi:.4f}"
            )

    if len(groups) < 2:
        return

    (a_name, _), (b_name, _) = groups[0], groups[1]
    print("\n=== Paired grouped comparisons (first group - second group) ===")
    print("comparison\tmetric\tn_common_qids\tdelta_mean\tci_low\tci_high\tp_value\tp_holm")

    raw_pvals: List[float] = []
    rows: List[Tuple[str, int, float, float, float, float]] = []
    for metric in args.metrics:
        q_a = cache[(a_name, metric)]
        q_b = cache[(b_name, metric)]
        n_common, delta, lo, hi, p_val = paired_delta_stats(
            group_a=q_a,
            group_b=q_b,
            confidence=args.confidence,
            n_boot=args.bootstrap,
            n_perm=args.permutations,
            seed=args.seed,
        )
        rows.append((metric, n_common, delta, lo, hi, p_val))
        raw_pvals.append(p_val)

    adjusted = holm_bonferroni(raw_pvals) if args.holm else [float("nan")] * len(raw_pvals)
    for (metric, n_common, delta, lo, hi, p_val), p_adj in zip(rows, adjusted):
        p_adj_str = f"{p_adj:.6f}" if not math.isnan(p_adj) else "NA"
        print(
            f"{a_name}-{b_name}\t{metric}\t{n_common}\t{delta:.4f}\t"
            f"{lo:.4f}\t{hi:.4f}\t{p_val:.6f}\t{p_adj_str}"
        )


if __name__ == "__main__":
    main()
