"""
Prepare standalone non-leish Prompt1-3 input rows from infectious candidates.

This script:
1) Selects rows by case_id from infectious_candidates_cases.jsonl
2) Maps case_image_files/captions to images[] entries used by medcase_multimodal_pipeline
3) Resolves and materializes linked image files into a dedicated image root
4) Writes an audit manifest for image coverage and linkage quality
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CASE_ID_PREFIX = "PMC"


@dataclass
class BuildSummary:
    requested_case_count: int
    selected_case_count: int
    missing_case_count: int
    materialized_case_count: int
    materialized_image_count: int


def parse_maybe_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [text]

    if isinstance(parsed, list):
        return [str(v).strip() for v in parsed if str(v).strip()]
    if parsed is None:
        return []
    return [str(parsed).strip()]


def read_case_ids(path: Path) -> List[str]:
    ordered: List[str] = []
    seen = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            case_id = raw.strip().upper()
            if not case_id:
                continue
            if not case_id.startswith(CASE_ID_PREFIX):
                raise ValueError(f"Invalid case_id in list: {case_id}")
            if case_id in seen:
                continue
            seen.add(case_id)
            ordered.append(case_id)
    return ordered


def read_jsonl_selected(path: Path, wanted_case_ids: Iterable[str]) -> Dict[str, Dict[str, Any]]:
    wanted = set(wanted_case_ids)
    selected: Dict[str, Dict[str, Any]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            case_id = str(row.get("case_id", "")).upper()
            if case_id in wanted and case_id not in selected:
                selected[case_id] = row
                if len(selected) == len(wanted):
                    break

    return selected


def load_inventory(path: Path) -> Tuple[Dict[Tuple[str, str], List[Dict[str, str]]], Dict[Tuple[str, str], List[Dict[str, str]]], Dict[str, int], Dict[str, int]]:
    by_case_requested: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    by_article_requested: Dict[Tuple[str, str], List[Dict[str, str]]] = {}
    case_counts: Dict[str, int] = {}
    article_counts: Dict[str, int] = {}

    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            case_id = str(row.get("case_id", "")).strip().upper()
            article_id = str(row.get("article_id", "")).strip().upper()
            requested_file = str(row.get("requested_file", "")).strip()

            if case_id:
                case_counts[case_id] = case_counts.get(case_id, 0) + 1
            if article_id:
                article_counts[article_id] = article_counts.get(article_id, 0) + 1

            if case_id and requested_file:
                by_case_requested.setdefault((case_id, requested_file), []).append(row)
            if article_id and requested_file:
                by_article_requested.setdefault((article_id, requested_file), []).append(row)

    return by_case_requested, by_article_requested, case_counts, article_counts


def resolve_inventory_local_path(row: Dict[str, str], aligned_images_root: Path) -> Optional[Path]:
    raw_path = str(row.get("local_path", "")).strip()
    if raw_path:
        candidate = Path(raw_path)
        if candidate.exists():
            return candidate

        legacy_prefix = "/data1t/lab/ngocnt/Leishmania_v3"
        home_prefix = "/home/ngocnt/Leishmania_v3"
        if raw_path.startswith(legacy_prefix):
            rewritten = Path(home_prefix + raw_path[len(legacy_prefix):])
            if rewritten.exists():
                return rewritten

    inv_case_id = str(row.get("case_id", "")).strip().upper()
    resolved_file = str(row.get("resolved_file", "")).strip()
    if inv_case_id and resolved_file:
        fallback = aligned_images_root / inv_case_id / resolved_file
        if fallback.exists():
            return fallback

    return None


def materialize_case_images(
    *,
    case_id: str,
    article_id: str,
    source_case: Dict[str, Any],
    inventory_by_case_requested: Dict[Tuple[str, str], List[Dict[str, str]]],
    inventory_by_article_requested: Dict[Tuple[str, str], List[Dict[str, str]]],
    case_inventory_counts: Dict[str, int],
    article_inventory_counts: Dict[str, int],
    aligned_images_root: Path,
    images_target_root: Path,
    materialize_images: bool,
) -> Tuple[List[Dict[str, str]], List[str], Dict[str, Any]]:
    requested_files = parse_maybe_list(source_case.get("case_image_files"))
    requested_captions = parse_maybe_list(source_case.get("case_image_captions"))

    images: List[Dict[str, str]] = []
    query_paths: List[str] = []
    missing_requested_files: List[str] = []
    unresolved_requested_files: List[str] = []
    fallback_article_used = False

    for idx, requested_file in enumerate(requested_files):
        caption = requested_captions[idx] if idx < len(requested_captions) else ""
        images.append({"file": requested_file, "caption": caption})

        rows = list(inventory_by_case_requested.get((case_id, requested_file), []))
        used_article_fallback = False
        if not rows:
            rows = list(inventory_by_article_requested.get((article_id, requested_file), []))
            used_article_fallback = bool(rows)
            fallback_article_used = fallback_article_used or used_article_fallback

        if not rows:
            missing_requested_files.append(requested_file)
            continue

        resolved_path: Optional[Path] = None
        for row in rows:
            resolved_path = resolve_inventory_local_path(row, aligned_images_root)
            if resolved_path is not None:
                break

        if resolved_path is None:
            unresolved_requested_files.append(requested_file)
            continue

        if materialize_images:
            target_dir = images_target_root / case_id
            target_dir.mkdir(parents=True, exist_ok=True)
            target_path = target_dir / requested_file
            if not target_path.exists():
                try:
                    target_path.symlink_to(resolved_path)
                except OSError:
                    shutil.copy2(resolved_path, target_path)
            query_paths.append(str(target_path))

    audit = {
        "source_image_count": int(source_case.get("image_count", 0) or 0),
        "requested_file_count": len(requested_files),
        "inventory_rows_case_id": case_inventory_counts.get(case_id, 0),
        "inventory_rows_article_id": article_inventory_counts.get(article_id, 0),
        "fallback_article_used": fallback_article_used,
        "linked_image_count": len(images),
        "materialized_image_count": len(query_paths),
        "missing_requested_files": sorted(missing_requested_files),
        "missing_requested_file_count": len(missing_requested_files),
        "unresolved_requested_files": sorted(unresolved_requested_files),
        "unresolved_requested_file_count": len(unresolved_requested_files),
    }
    return images, query_paths, audit


def select_diagnosis(source_case: Dict[str, Any]) -> str:
    candidates = parse_maybe_list(source_case.get("matched_disease_candidates"))
    for candidate in candidates:
        value = candidate.strip()
        if value:
            return value
    return "Non-Leishmaniasis"


def build_input_row(
    *,
    source_case: Dict[str, Any],
    images: List[Dict[str, str]],
    query_image_paths: List[str],
    phase_name: str,
    source_jsonl: Path,
    image_audit: Dict[str, Any],
) -> Dict[str, Any]:
    case_id = str(source_case.get("case_id", "")).upper()
    article_id = str(source_case.get("article_id") or case_id.rsplit("_", 1)[0]).upper()
    diagnosis = select_diagnosis(source_case)

    return {
        "case_id": case_id,
        "article_id": article_id,
        "pmcid": source_case.get("pmcid", article_id),
        "pmid": source_case.get("pmid", ""),
        "doi": source_case.get("doi", ""),
        "journal": source_case.get("journal", ""),
        "year": source_case.get("year", ""),
        "license": source_case.get("license", ""),
        "title": source_case.get("title", ""),
        "abstract": source_case.get("abstract", ""),
        "case_text": source_case.get("case_text", ""),
        "images": images,
        "query_image_paths": query_image_paths,
        "diagnosis": diagnosis,
        "diagnosis_type": "Non-Leishmaniasis",
        "species": "",
        "is_leishmaniasis": False,
        "split": f"train_{phase_name}",
        "source_jsonl": str(source_jsonl),
        "reference_label": {
            "diagnosis": diagnosis,
            "diagnosis_type": "Non-Leishmaniasis",
            "species": "",
            "is_leishmaniasis": False,
            "license": source_case.get("license", ""),
            "provenance": "infectious_candidates_matched_disease_candidate_0",
            "clinician_validated": False,
            "is_ground_truth": False,
        },
        "pipeline_metadata": {
            "selection_phase": phase_name,
            "input_builder": "prepare_nonleish_prompt123_inputs_v1",
            "diagnosis_source": "matched_disease_candidates[0]",
            "model_openai_reasoner": "gpt-5-mini",
            "model_openai_judge": "gpt-5-mini",
            "model_gemini_quality_filter": "gemini-2.5-pro",
            "image_linkage": image_audit,
        },
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare non-leish prompt1-3 input rows from infectious candidates")
    parser.add_argument(
        "--phase-name",
        default="phase1a_tierA_gpt5mini_gemini25pro",
        help="Suffix used in split and manifest metadata",
    )
    parser.add_argument(
        "--case-list",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/"
            "nonleish_additions/phase1a_tierA_case_ids.txt"
        ),
    )
    parser.add_argument(
        "--source-jsonl",
        type=Path,
        default=Path(
            "/home/ngocnt/Leishmania_v3/data/Infectious_disease_v2/infectious_candidates_cases.jsonl"
        ),
    )
    parser.add_argument(
        "--image-inventory-csv",
        type=Path,
        default=Path(
            "/home/ngocnt/Leishmania_v3/data/Infectious_disease_v2/aligned_case_images/image_inventory.csv"
        ),
    )
    parser.add_argument(
        "--aligned-images-root",
        type=Path,
        default=Path(
            "/home/ngocnt/Leishmania_v3/data/Infectious_disease_v2/aligned_case_images"
        ),
    )
    parser.add_argument(
        "--images-target-root",
        type=Path,
        default=Path(
            "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/"
            "nonleish_phase1a_tierA_gpt5mini_gemini25pro/images"
        ),
    )
    parser.add_argument(
        "--output-jsonl",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/"
            "nonleish_additions/generated/train_phase1a_tierA_raw_gpt5mini_gemini25pro.jsonl"
        ),
    )
    parser.add_argument(
        "--audit-json",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/"
            "nonleish_additions/generated/image_materialization_audit_phase1a_tierA_gpt5mini_gemini25pro.json"
        ),
    )
    parser.add_argument(
        "--materialize-images",
        action="store_true",
        help="Symlink/copy linked images into images-target-root and populate query_image_paths",
    )
    parser.add_argument(
        "--fail-on-missing",
        action="store_true",
        help="Fail when any requested case_id is missing in source-jsonl",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    case_ids = read_case_ids(args.case_list)
    selected = read_jsonl_selected(args.source_jsonl, case_ids)

    missing_ids = [case_id for case_id in case_ids if case_id not in selected]
    if missing_ids and args.fail_on_missing:
        raise ValueError(f"Missing case_ids in source-jsonl: {missing_ids}")

    (
        inventory_by_case_requested,
        inventory_by_article_requested,
        case_inventory_counts,
        article_inventory_counts,
    ) = load_inventory(args.image_inventory_csv)

    output_rows: List[Dict[str, Any]] = []
    image_audit_by_case: Dict[str, Dict[str, Any]] = {}

    for case_id in case_ids:
        if case_id not in selected:
            continue

        source_case = selected[case_id]
        article_id = str(source_case.get("article_id") or case_id.rsplit("_", 1)[0]).upper()

        images, query_image_paths, image_audit = materialize_case_images(
            case_id=case_id,
            article_id=article_id,
            source_case=source_case,
            inventory_by_case_requested=inventory_by_case_requested,
            inventory_by_article_requested=inventory_by_article_requested,
            case_inventory_counts=case_inventory_counts,
            article_inventory_counts=article_inventory_counts,
            aligned_images_root=args.aligned_images_root,
            images_target_root=args.images_target_root,
            materialize_images=args.materialize_images,
        )
        image_audit_by_case[case_id] = image_audit

        output_rows.append(
            build_input_row(
                source_case=source_case,
                images=images,
                query_image_paths=query_image_paths,
                phase_name=args.phase_name,
                source_jsonl=args.source_jsonl,
                image_audit=image_audit,
            )
        )

    write_jsonl(args.output_jsonl, output_rows)

    materialized_case_count = sum(1 for row in output_rows if row.get("query_image_paths"))
    materialized_image_count = sum(len(row.get("query_image_paths", [])) for row in output_rows)
    summary = BuildSummary(
        requested_case_count=len(case_ids),
        selected_case_count=len(output_rows),
        missing_case_count=len(missing_ids),
        materialized_case_count=materialized_case_count,
        materialized_image_count=materialized_image_count,
    )

    audit_payload = {
        "phase_name": args.phase_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "case_list_file": str(args.case_list),
        "source_jsonl": str(args.source_jsonl),
        "image_inventory_csv": str(args.image_inventory_csv),
        "aligned_images_root": str(args.aligned_images_root),
        "images_target_root": str(args.images_target_root),
        "materialize_images": bool(args.materialize_images),
        "summary": summary.__dict__,
        "requested_case_ids": case_ids,
        "missing_case_ids": missing_ids,
        "selected_case_ids": [row["case_id"] for row in output_rows],
        "output_jsonl": str(args.output_jsonl),
        "image_audit_by_case": image_audit_by_case,
    }
    args.audit_json.parent.mkdir(parents=True, exist_ok=True)
    args.audit_json.write_text(json.dumps(audit_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Prepared non-leish Prompt1-3 input ===")
    print(f"phase_name={args.phase_name}")
    print(f"requested_case_count={summary.requested_case_count}")
    print(f"selected_case_count={summary.selected_case_count}")
    print(f"missing_case_count={summary.missing_case_count}")
    print(f"materialized_case_count={summary.materialized_case_count}")
    print(f"materialized_image_count={summary.materialized_image_count}")
    print(f"output_jsonl={args.output_jsonl}")
    print(f"audit_json={args.audit_json}")


if __name__ == "__main__":
    main()
