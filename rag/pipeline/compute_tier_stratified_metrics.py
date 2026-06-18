#!/usr/bin/env python3
"""Compute tier-stratified evaluation metrics for one or more run folders.

This script joins:
1) Case quality tiers from Prompt 1->3 outputs (results.jsonl)
2) Per-query evaluation outputs from run folders (ragas.jsonl)

Buckets reported per run:
- A
- B
- C
- A+B

Default metrics:
- diagnosis_accuracy
- diagnosis_type_accuracy
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{i}: {exc}") from exc
    return rows


def qid_to_case_id(qid: str) -> str:
    if "::" in qid:
        return qid.split("::", 1)[0]
    return qid


def to_float(value: Any) -> float:
    if value is None:
        raise ValueError("None metric")
    return float(value)


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def fmt(x: float) -> str:
    return f"{x:.4f}"


def load_tier_map(results_jsonl: Path) -> Dict[str, str]:
    rows = read_jsonl(results_jsonl)
    out: Dict[str, str] = {}
    for row in rows:
        cid = str(row.get("case_id", "")).strip()
        tier = str(row.get("confidence_tier", "")).strip().upper()
        if not cid:
            continue
        if tier not in {"A", "B", "C"}:
            continue
        out[cid] = tier
    return out


def analyze_run(run_dir: Path, tier_map: Dict[str, str], metrics: List[str]) -> Dict[str, Any]:
    ragas_path = run_dir / "ragas.jsonl"
    if not ragas_path.exists():
        raise FileNotFoundError(f"Missing ragas.jsonl in {run_dir}")

    rows = read_jsonl(ragas_path)

    buckets = {
        "A": {m: [] for m in metrics},
        "B": {m: [] for m in metrics},
        "C": {m: [] for m in metrics},
        "A+B": {m: [] for m in metrics},
    }

    counts = {"A": 0, "B": 0, "C": 0, "A+B": 0}
    unmapped_qids: List[str] = []

    for row in rows:
        qid = str(row.get("qid", "")).strip()
        if not qid:
            continue
        case_id = qid_to_case_id(qid)
        tier = tier_map.get(case_id)
        if tier not in {"A", "B", "C"}:
            unmapped_qids.append(qid)
            continue

        counts[tier] += 1
        if tier in {"A", "B"}:
            counts["A+B"] += 1

        for m in metrics:
            try:
                v = to_float(row.get(m))
            except Exception:
                continue
            buckets[tier][m].append(v)
            if tier in {"A", "B"}:
                buckets["A+B"][m].append(v)

    total_mapped = counts["A"] + counts["B"] + counts["C"]
    pct_tier_c = (counts["C"] / total_mapped * 100.0) if total_mapped else 0.0

    return {
        "run_dir": str(run_dir),
        "n_rows": len(rows),
        "n_mapped": total_mapped,
        "n_unmapped": len(unmapped_qids),
        "pct_tier_c": pct_tier_c,
        "counts": counts,
        "buckets": buckets,
        "unmapped_qids": unmapped_qids,
    }


def to_markdown(results_path: Path, analyses: List[Dict[str, Any]], metrics: List[str]) -> str:
    lines: List[str] = []
    lines.append("# Tier-Stratified Evaluation Metrics")
    lines.append("")
    lines.append(f"- Tier source: {results_path}")
    lines.append(f"- Metrics: {', '.join(metrics)}")
    lines.append("")

    for analysis in analyses:
        run_name = Path(analysis["run_dir"]).name
        lines.append(f"## {run_name}")
        lines.append("")
        lines.append(f"- Total rows in ragas.jsonl: {analysis['n_rows']}")
        lines.append(f"- Mapped rows with known tier: {analysis['n_mapped']}")
        lines.append(f"- Unmapped rows: {analysis['n_unmapped']}")
        lines.append(f"- Tier C share (mapped rows): {analysis['pct_tier_c']:.2f}%")
        lines.append("")

        header = "| Bucket | n | " + " | ".join([f"{m} mean" for m in metrics]) + " |"
        sep = "|---|---:|" + "|".join(["---:" for _ in metrics]) + "|"
        lines.append(header)
        lines.append(sep)

        for bucket in ["A", "B", "C", "A+B"]:
            n = analysis["counts"][bucket]
            vals = []
            for m in metrics:
                vals.append(fmt(mean(analysis["buckets"][bucket][m])))
            lines.append(f"| {bucket} | {n} | " + " | ".join(vals) + " |")

        if analysis["unmapped_qids"]:
            lines.append("")
            lines.append("Unmapped qids:")
            for qid in analysis["unmapped_qids"]:
                lines.append(f"- {qid}")

        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compute tier-stratified run metrics from ragas.jsonl files")
    p.add_argument("--tier-results-jsonl", required=True, help="Path to Prompt 1->3 results.jsonl with confidence_tier")
    p.add_argument("--run", dest="runs", action="append", required=True,
                   help="Run directory containing ragas.jsonl (repeat for multiple runs)")
    p.add_argument("--metric", dest="metrics", action="append",
                   default=["diagnosis_accuracy", "diagnosis_type_accuracy"],
                   help="Metric key in ragas.jsonl (repeatable)")
    p.add_argument("--output-md", default=None, help="Optional markdown output path")
    return p


def main() -> None:
    args = build_parser().parse_args()
    tier_path = Path(args.tier_results_jsonl)
    if not tier_path.exists():
        raise FileNotFoundError(f"Tier results file not found: {tier_path}")

    tier_map = load_tier_map(tier_path)
    metrics = list(dict.fromkeys(args.metrics))

    analyses: List[Dict[str, Any]] = []
    for run in args.runs:
        run_dir = Path(run)
        analyses.append(analyze_run(run_dir, tier_map, metrics))

    md = to_markdown(tier_path, analyses, metrics)

    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(md, encoding="utf-8")
        print(f"Wrote {out}")
    else:
        print(md)


if __name__ == "__main__":
    main()
