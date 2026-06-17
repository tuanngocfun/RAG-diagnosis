#!/usr/bin/env python3
"""Build provenance reports for non-leish train variant artifacts.

This utility traces, for one or more phase names:
1) Producer manifests and validations
2) Reconstructed generation commands
3) Confirmed command evidence from shell history
4) Downstream run/config/log consumers
5) Optional v7/gpt5mini branch references for disambiguation
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DEFAULT_PROJECT_ROOT = Path("/home/ngocnt/experiments/structured_cases_v4")
DEFAULT_LEGACY_ROOT = Path("/home/ngocnt/Leishmania_v3")
DEFAULT_PHASES: Tuple[str, ...] = ("phase1a_tierA", "phase1b_tierAB")


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return read_json(path)
    except Exception:
        return None


def uniq(seq: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for item in seq:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def to_rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except Exception:
        return str(path)


def resolve_case_list(case_list_raw: str, project_root: Path, manifest_path: Path) -> Optional[Path]:
    raw = Path(case_list_raw)
    candidates: List[Path] = []

    if raw.is_absolute():
        candidates.append(raw)
    else:
        # Most manifests were written from project_root/codes working dir.
        candidates.append((project_root / "codes" / raw).resolve())
        candidates.append((project_root / raw).resolve())
        candidates.append((manifest_path.parent / raw).resolve())

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def read_history(path: Path) -> List[Tuple[int, str]]:
    rows: List[Tuple[int, str]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for idx, line in enumerate(f, start=1):
            rows.append((idx, line.rstrip("\n")))
    return rows


def history_matches(rows: Sequence[Tuple[int, str]], phase_name: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for line_no, line in rows:
        if "build_nonleish_train_variants" not in line:
            continue
        if phase_name not in line:
            continue
        out.append((line_no, line.strip()))
    return out


def history_exports(rows: Sequence[Tuple[int, str]], output_path: str) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    needle = output_path.strip()
    if not needle:
        return out
    for line_no, line in rows:
        if "STRUCTURED_CASES_TRAIN_JSONL" not in line:
            continue
        if needle not in line:
            continue
        out.append((line_no, line.strip()))
    return out


def materialize_likely(manifest: Dict[str, Any]) -> bool:
    image_audit = manifest.get("image_audit_by_case") or {}
    if not isinstance(image_audit, dict):
        return False
    for audit in image_audit.values():
        if not isinstance(audit, dict):
            continue
        if int(audit.get("materialized_image_count", 0) or 0) > 0:
            return True
    return False


def collect_mentions(files: Sequence[Path], needle: str, max_hits: int) -> List[Dict[str, Any]]:
    hits: List[Dict[str, Any]] = []
    if not needle:
        return hits

    for path in files:
        if len(hits) >= max_hits:
            break
        if not path.exists() or not path.is_file():
            continue
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line_no, line in enumerate(f, start=1):
                    if needle in line:
                        hits.append(
                            {
                                "path": str(path),
                                "line": line_no,
                                "snippet": line.strip(),
                            }
                        )
                        if len(hits) >= max_hits:
                            break
        except Exception:
            continue
    return hits


def collect_run_mentions(runs_dir: Path, needle: str, max_hits: int) -> List[Dict[str, Any]]:
    files: List[Path] = []
    for pattern in ("**/run_config.json", "**/summary.json", "**/*.log"):
        files.extend(sorted(runs_dir.glob(pattern)))
    return collect_mentions(files, needle=needle, max_hits=max_hits)


def collect_v7_code_references(v7_root: Path, target_basenames: Sequence[str], max_hits: int) -> List[Dict[str, Any]]:
    files: List[Path] = []
    for pattern in ("**/*.py", "**/*.md", "**/*.sh"):
        files.extend(sorted(v7_root.glob(pattern)))

    hits: List[Dict[str, Any]] = []
    for basename in uniq([b for b in target_basenames if b]):
        basename_hits = collect_mentions(files, needle=basename, max_hits=max_hits)
        for item in basename_hits:
            item["target_basename"] = basename
            hits.append(item)
            if len(hits) >= max_hits:
                return hits
    return hits


def extract_validation_counts(validation: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not validation:
        return {}
    stats = validation.get("stats") or {}
    base_counts = validation.get("base_counts") or {}
    result_counts = validation.get("result_counts") or {}
    consistency = validation.get("consistency") or {}
    return {
        "requested_case_count": stats.get("requested_case_count"),
        "added_case_count": stats.get("added_case_count"),
        "base_train_pseudolabel_v2": base_counts.get("train_pseudolabel_v2"),
        "result_train_pseudolabel_v2": result_counts.get("train_pseudolabel_v2"),
        "baseline_rows_preserved": consistency.get("baseline_rows_preserved"),
    }


def build_phase_report(
    project_root: Path,
    runs_dir: Path,
    generated_dir: Path,
    history_rows: Sequence[Tuple[int, str]],
    phase_name: str,
    max_run_mentions: int,
) -> Dict[str, Any]:
    manifest_path = generated_dir / f"merge_manifest_{phase_name}.json"
    validation_path = generated_dir / f"merge_validation_{phase_name}.json"

    manifest = safe_load_json(manifest_path)
    validation = safe_load_json(validation_path)

    if not manifest:
        return {
            "phase_name": phase_name,
            "manifest_path": str(manifest_path),
            "validation_path": str(validation_path),
            "exists": False,
            "error": "Manifest file is missing or unreadable",
        }

    outputs = manifest.get("outputs") or {}
    target_pseudolabel = str(outputs.get("train_pseudolabel_v2", ""))
    target_raw = str(outputs.get("train_raw", ""))

    case_list_raw = str(manifest.get("case_list_file", ""))
    case_list_resolved = resolve_case_list(case_list_raw, project_root=project_root, manifest_path=manifest_path)

    reconstructed_cmd_parts = [
        f"cd {project_root / 'codes'}",
        "python -m pipeline.build_nonleish_train_variants",
        f"--phase-name {phase_name}",
        f"--case-list {case_list_resolved if case_list_resolved else case_list_raw}",
    ]
    if materialize_likely(manifest):
        reconstructed_cmd_parts.append("--materialize-images")
    reconstructed_command = " \\\n  ".join(reconstructed_cmd_parts[1:])

    phase_history = history_matches(history_rows, phase_name=phase_name)
    export_history = history_exports(history_rows, output_path=target_pseudolabel)

    pseudolabel_mentions = collect_run_mentions(
        runs_dir=runs_dir,
        needle=target_pseudolabel,
        max_hits=max_run_mentions,
    )
    raw_mentions = collect_run_mentions(
        runs_dir=runs_dir,
        needle=target_raw,
        max_hits=max_run_mentions,
    )

    return {
        "phase_name": phase_name,
        "exists": True,
        "manifest_path": str(manifest_path),
        "validation_path": str(validation_path),
        "created_at_utc": manifest.get("created_at_utc"),
        "case_list_file_raw": case_list_raw,
        "case_list_file_resolved": str(case_list_resolved) if case_list_resolved else None,
        "selected_case_count": len(manifest.get("selected_case_ids") or []),
        "added_case_count_manifest": len(manifest.get("added_case_ids") or []),
        "outputs": outputs,
        "validation_counts": extract_validation_counts(validation),
        "reconstructed_command": reconstructed_command,
        "reconstructed_command_cwd": str(project_root / "codes"),
        "history_build_commands": [{"line": n, "text": t} for n, t in phase_history],
        "history_export_commands": [{"line": n, "text": t} for n, t in export_history],
        "downstream_mentions_pseudolabel": pseudolabel_mentions,
        "downstream_mentions_raw": raw_mentions,
    }


def collect_gpt5mini_branch(generated_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for path in sorted(generated_dir.glob("merge_manifest_*gpt5mini*.json")):
        manifest = safe_load_json(path)
        if not manifest:
            continue
        rows.append(
            {
                "manifest_path": str(path),
                "phase_name": manifest.get("phase_name"),
                "inputs": manifest.get("inputs") or {},
                "outputs": manifest.get("outputs") or {},
            }
        )
    return rows


def to_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Non-Leish Train Provenance Report")
    lines.append("")
    lines.append(f"- Generated at UTC: {report['generated_at_utc']}")
    lines.append(f"- Project root: {report['project_root']}")
    lines.append(f"- Direct producer script: {report['producer_script']}")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Phase | Manifest | Target train_pseudolabel_v2 | Requested | Added | Base->Result |")
    lines.append("|---|---|---|---:|---:|---|")
    for phase in report["phases"]:
        if not phase.get("exists"):
            lines.append(f"| {phase['phase_name']} | MISSING | - | - | - | - |")
            continue
        vc = phase.get("validation_counts") or {}
        base_cnt = vc.get("base_train_pseudolabel_v2")
        res_cnt = vc.get("result_train_pseudolabel_v2")
        train_target = str((phase.get("outputs") or {}).get("train_pseudolabel_v2", ""))
        lines.append(
            "| {phase} | {manifest} | {target} | {requested} | {added} | {base}->{res} |".format(
                phase=phase["phase_name"],
                manifest=Path(phase["manifest_path"]).name,
                target=Path(train_target).name if train_target else "-",
                requested=vc.get("requested_case_count", "-"),
                added=vc.get("added_case_count", "-"),
                base=base_cnt if base_cnt is not None else "-",
                res=res_cnt if res_cnt is not None else "-",
            )
        )
    lines.append("")

    for phase in report["phases"]:
        lines.append(f"## Phase {phase['phase_name']}")
        lines.append("")
        if not phase.get("exists"):
            lines.append(f"- Error: {phase.get('error', 'Unknown error')}")
            lines.append("")
            continue

        lines.append(f"- Manifest: {phase['manifest_path']}")
        lines.append(f"- Validation: {phase['validation_path']}")
        lines.append(f"- Manifest created_at_utc: {phase.get('created_at_utc')}")
        lines.append(f"- Case list (raw): {phase.get('case_list_file_raw')}")
        lines.append(f"- Case list (resolved): {phase.get('case_list_file_resolved') or 'UNRESOLVED'}")

        outputs = phase.get("outputs") or {}
        lines.append(f"- Output train_raw: {outputs.get('train_raw')}")
        lines.append(f"- Output train_pseudolabel_v2: {outputs.get('train_pseudolabel_v2')}")

        vc = phase.get("validation_counts") or {}
        lines.append(
            "- Validation counts: requested={req}, added={added}, base_train_pseudolabel_v2={base}, result_train_pseudolabel_v2={res}".format(
                req=vc.get("requested_case_count"),
                added=vc.get("added_case_count"),
                base=vc.get("base_train_pseudolabel_v2"),
                res=vc.get("result_train_pseudolabel_v2"),
            )
        )
        lines.append("")

        lines.append("### Reconstructed Generation Command")
        lines.append("")
        lines.append("```bash")
        lines.append(f"cd {phase['reconstructed_command_cwd']}")
        lines.append(phase["reconstructed_command"])
        lines.append("```")
        lines.append("")

        lines.append("### Confirmed Shell History Build Commands")
        lines.append("")
        history_build = phase.get("history_build_commands") or []
        if history_build:
            for row in history_build:
                lines.append(f"- .bash_history:{row['line']} {row['text']}")
        else:
            lines.append("- None found for this phase")
        lines.append("")

        lines.append("### Confirmed Shell History Export Commands")
        lines.append("")
        history_exports = phase.get("history_export_commands") or []
        if history_exports:
            for row in history_exports:
                lines.append(f"- .bash_history:{row['line']} {row['text']}")
        else:
            lines.append("- None found for this phase")
        lines.append("")

        lines.append("### Downstream Mentions (pseudolabel target)")
        lines.append("")
        p_hits = phase.get("downstream_mentions_pseudolabel") or []
        if p_hits:
            for hit in p_hits:
                lines.append(f"- {hit['path']}:{hit['line']} {hit['snippet']}")
        else:
            lines.append("- None found")
        lines.append("")

        lines.append("### Downstream Mentions (raw target)")
        lines.append("")
        r_hits = phase.get("downstream_mentions_raw") or []
        if r_hits:
            for hit in r_hits:
                lines.append(f"- {hit['path']}:{hit['line']} {hit['snippet']}")
        else:
            lines.append("- None found")
        lines.append("")

    lines.append("## v7 and gpt5mini Branch Check")
    lines.append("")
    lines.append(f"- v7 code root: {report['v7_code_root']}")
    lines.append(f"- Target basenames scanned: {', '.join(report['target_basenames'])}")
    lines.append("")

    v7_hits = report.get("v7_code_references") or []
    if v7_hits:
        lines.append("Found direct references in v7 code files:")
        for hit in v7_hits:
            lines.append(
                f"- {hit['path']}:{hit['line']} ({hit.get('target_basename', '?')}) {hit['snippet']}"
            )
    else:
        lines.append("No direct references found in v7 code files for the target basenames.")
    lines.append("")

    gpt5_rows = report.get("gpt5mini_manifests") or []
    if gpt5_rows:
        lines.append("Adjacent gpt5mini manifests in nonleish_additions/generated:")
        for row in gpt5_rows:
            outputs = row.get("outputs") or {}
            lines.append(f"- {row['manifest_path']}")
            lines.append(f"  phase_name={row.get('phase_name')}")
            lines.append(f"  merged_train={outputs.get('merged_train')}")
            lines.append(f"  standalone_nonleish={outputs.get('standalone_nonleish')}")
    else:
        lines.append("No gpt5mini merge manifests found in generated/.")
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Build provenance reports for non-leish generated train artifacts")
    p.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT), help="Path to structured_cases_v4 root")
    p.add_argument("--legacy-root", default=str(DEFAULT_LEGACY_ROOT), help="Path to Leishmania_v3 root")
    p.add_argument(
        "--phase",
        action="append",
        dest="phases",
        default=None,
        help="Phase name (repeatable). Defaults to phase1a_tierA and phase1b_tierAB.",
    )
    p.add_argument(
        "--history-path",
        default=str(Path.home() / ".bash_history"),
        help="Shell history file for command evidence",
    )
    p.add_argument("--max-run-mentions", type=int, default=12, help="Cap mention rows per phase and target")
    p.add_argument(
        "--output-json",
        default=None,
        help="Optional output JSON path (default: generated/provenance_report_nonleish_phase_outputs.json)",
    )
    p.add_argument(
        "--output-md",
        default=None,
        help="Optional output markdown path (default: generated/provenance_report_nonleish_phase_outputs.md)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()

    project_root = Path(args.project_root).resolve()
    legacy_root = Path(args.legacy_root).resolve()
    runs_dir = project_root / "runs"
    generated_dir = project_root / "leishmaniasis_verified_v2" / "nonleish_additions" / "generated"
    producer_script = project_root / "codes" / "pipeline" / "build_nonleish_train_variants.py"

    phases = args.phases if args.phases else list(DEFAULT_PHASES)
    phases = uniq(phases)

    history_rows = read_history(Path(args.history_path).expanduser())

    phase_reports = [
        build_phase_report(
            project_root=project_root,
            runs_dir=runs_dir,
            generated_dir=generated_dir,
            history_rows=history_rows,
            phase_name=phase,
            max_run_mentions=args.max_run_mentions,
        )
        for phase in phases
    ]

    target_basenames: List[str] = []
    for phase in phase_reports:
        outputs = phase.get("outputs") or {}
        out = outputs.get("train_pseudolabel_v2")
        if out:
            target_basenames.append(Path(str(out)).name)

    v7_code_root = legacy_root / "rag" / "testing" / "multimodal" / "v7"
    v7_refs = collect_v7_code_references(
        v7_root=v7_code_root,
        target_basenames=target_basenames,
        max_hits=20,
    )

    report: Dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "legacy_root": str(legacy_root),
        "producer_script": str(producer_script),
        "history_path": str(Path(args.history_path).expanduser()),
        "phases": phase_reports,
        "target_basenames": target_basenames,
        "v7_code_root": str(v7_code_root),
        "v7_code_references": v7_refs,
        "gpt5mini_manifests": collect_gpt5mini_branch(generated_dir=generated_dir),
    }

    default_json = generated_dir / "provenance_report_nonleish_phase_outputs.json"
    default_md = generated_dir / "provenance_report_nonleish_phase_outputs.md"

    output_json = Path(args.output_json).resolve() if args.output_json else default_json
    output_md = Path(args.output_md).resolve() if args.output_md else default_md

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    output_md.write_text(to_markdown(report), encoding="utf-8")

    print(f"Wrote JSON report: {output_json}")
    print(f"Wrote markdown report: {output_md}")
    print(f"Phases analyzed: {', '.join(phases)}")


if __name__ == "__main__":
    main()
