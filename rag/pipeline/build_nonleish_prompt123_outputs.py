"""
Build standalone and merged non-leish artifacts from Prompt1-3 results.

Inputs:
- results.jsonl from medcase_multimodal_pipeline.py (structure-only Prompt1-3 run)
- base train_p14_v7_normalized.jsonl

Outputs:
- standalone non-leish normalized JSONL (p14-style schema)
- merged train JSONL (base p14 + new non-leish rows)
- merge manifest and validation JSON files
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set


SILVER_LABEL_DISCLAIMER = (
    "Evaluation labels are silver labels derived from GPT-5-mini and Gemini 2.5 Pro "
    "pipeline outputs; they are not clinician ground truth."
)


@dataclass
class MergeStats:
    expected_case_count: int
    standalone_case_count: int
    added_case_count: int
    collision_case_count: int
    missing_expected_case_count: int


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def read_case_list(path: Optional[Path]) -> List[str]:
    if path is None:
        return []
    if not path.exists():
        raise FileNotFoundError(f"Expected case list file not found: {path}")

    out: List[str] = []
    seen: Set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            case_id = raw.strip().upper()
            if not case_id:
                continue
            if case_id in seen:
                continue
            seen.add(case_id)
            out.append(case_id)
    return out


def parse_tagged_field(raw_text: str, tag_name: str) -> str:
    if not raw_text:
        return ""
    pattern = rf"<{re.escape(tag_name)}>(.*?)</{re.escape(tag_name)}>"
    match = re.search(pattern, raw_text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def extract_case_prompt(result: Dict[str, Any]) -> str:
    prompt1 = result.get("prompt1_converted_case") or {}
    if not isinstance(prompt1, dict):
        return ""
    case_prompt = str(prompt1.get("case_prompt", "")).strip()
    if case_prompt:
        return case_prompt
    return parse_tagged_field(str(prompt1.get("raw", "")), "case_prompt")


def extract_final_diagnosis(result: Dict[str, Any]) -> str:
    prompt1 = result.get("prompt1_converted_case") or {}
    if not isinstance(prompt1, dict):
        return ""
    final_dx = str(prompt1.get("final_diagnosis", "")).strip()
    if final_dx:
        return final_dx
    return parse_tagged_field(str(prompt1.get("raw", "")), "final_diagnosis")


def diagnosis_type_from_text(text: str) -> str:
    lowered = (text or "").lower()
    if not lowered:
        return ""
    if "non-leish" in lowered or "not leish" in lowered:
        return "Non-Leishmaniasis"
    if "post-kala" in lowered or "pkdl" in lowered:
        return "PKDL"
    if "mucocutaneous" in lowered:
        return "MCL"
    if "visceral" in lowered or "kala-azar" in lowered or "kala azar" in lowered:
        return "VL"
    if "cutaneous" in lowered:
        return "CL"
    if "ocular" in lowered:
        return "Ocular"
    if "veterinary" in lowered or "canine" in lowered:
        return "Veterinary"
    return "Other"


def to_image_entries(image_paths_used: List[str]) -> Dict[str, List[Any]]:
    query_image_paths = [str(path) for path in (image_paths_used or []) if str(path).strip()]
    images = [{"file": Path(path).name, "caption": ""} for path in query_image_paths]
    return {
        "images": images,
        "query_image_paths": query_image_paths,
    }


def build_label_contract(dataset_version: str) -> Dict[str, str]:
    return {
        "dataset_version": dataset_version,
        "ground_truth_status": "silver_reference_only",
        "verified_track": "reference_label propagated from the structured multimodal pipeline, not clinician gold ground truth",
        "pseudolabel_track": "Prompt-1 final_diagnosis extracted from the structured multimodal pipeline, not clinician gold ground truth",
        "disclaimer": SILVER_LABEL_DISCLAIMER,
    }


def normalize_result_row(
    result: Dict[str, Any],
    *,
    split_name: str,
    dataset_version: str,
    model_openai_reasoner: str,
    model_openai_judge: str,
    model_gemini_quality_filter: str,
) -> Optional[Dict[str, Any]]:
    case_id = str(result.get("case_id", "")).strip()
    if not case_id:
        return None

    case_text = extract_case_prompt(result)
    if not case_text:
        return None

    reference = result.get("reference_label") or {}
    verified_diagnosis = str(reference.get("diagnosis", "")).strip() or "Non-Leishmaniasis"
    verified_diagnosis_type = str(reference.get("diagnosis_type", "")).strip() or "Non-Leishmaniasis"
    verified_species = str(reference.get("species", "")).strip()
    verified_is_leish = reference.get("is_leishmaniasis")
    if verified_is_leish is None:
        verified_is_leish = False

    pseudo_dx = extract_final_diagnosis(result) or verified_diagnosis
    pseudo_type = diagnosis_type_from_text(pseudo_dx) or verified_diagnosis_type
    pseudo_is_leish = None if not pseudo_type else pseudo_type != "Non-Leishmaniasis"

    images_payload = to_image_entries(result.get("image_paths_used") or [])

    pipeline_metadata = dict(result.get("pipeline_metadata") or {})
    pipeline_metadata.update(
        {
            "model_openai_reasoner": model_openai_reasoner,
            "model_openai_judge": model_openai_judge,
            "model_gemini_quality_filter": model_gemini_quality_filter,
            "normalization_builder": "build_nonleish_prompt123_outputs_v1",
        }
    )

    return {
        "case_id": case_id,
        "case_text": case_text,
        "images": images_payload["images"],
        "diagnosis": verified_diagnosis,
        "diagnosis_type": verified_diagnosis_type,
        "species": verified_species,
        "is_leishmaniasis": bool(verified_is_leish),
        "labels": {
            "verified": {
                "diagnosis": verified_diagnosis,
                "diagnosis_type": verified_diagnosis_type,
                "species": verified_species,
                "is_leishmaniasis": bool(verified_is_leish),
            },
            "pseudolabel": {
                "diagnosis": pseudo_dx,
                "diagnosis_type": pseudo_type,
                "species": verified_species,
                "is_leishmaniasis": pseudo_is_leish,
            },
        },
        "query_image_paths": images_payload["query_image_paths"],
        "split": split_name,
        "source_jsonl": result.get("source_jsonl", ""),
        "reference_label": reference,
        "pipeline_metadata": pipeline_metadata,
        "label_contract": build_label_contract(dataset_version),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build non-leish standalone and merged p14 outputs from Prompt1-3 results")
    parser.add_argument(
        "--phase-name",
        default="phase1a_tierA_gpt5mini_gemini25pro",
    )
    parser.add_argument(
        "--results-jsonl",
        type=Path,
        default=Path(
            "/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/"
            "nonleish_phase1a_tierA_gpt5mini_gemini25pro/results.jsonl"
        ),
    )
    parser.add_argument(
        "--base-train-jsonl",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/"
            "train_p14_v7_normalized.jsonl"
        ),
    )
    parser.add_argument(
        "--expected-case-list",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/"
            "nonleish_additions/phase1a_tierA_case_ids.txt"
        ),
    )
    parser.add_argument(
        "--standalone-output",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/nonleish_additions/generated/"
            "train_nonleish_p14_v7_phase1a_tierA_gpt5mini_gemini25pro.jsonl"
        ),
    )
    parser.add_argument(
        "--merged-output",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/nonleish_additions/generated/"
            "train_p14_v7_merged_phase1a_tierA_gpt5mini_gemini25pro.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest-output",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/nonleish_additions/generated/"
            "merge_manifest_phase1a_tierA_gpt5mini_gemini25pro.json"
        ),
    )
    parser.add_argument(
        "--validation-output",
        type=Path,
        default=Path(
            "/home/ngocnt/experiments/structured_cases_v4/leishmaniasis_verified_v2/nonleish_additions/generated/"
            "merge_validation_phase1a_tierA_gpt5mini_gemini25pro.json"
        ),
    )
    parser.add_argument(
        "--split-name",
        default="train_nonleish_phase1a_tierA_gpt5mini_gemini25pro",
    )
    parser.add_argument(
        "--dataset-version",
        default="p14_v7_nonleish_phase1a_gpt5mini_gemini25pro",
    )
    parser.add_argument("--model-openai-reasoner", default="gpt-5-mini")
    parser.add_argument("--model-openai-judge", default="gpt-5-mini")
    parser.add_argument("--model-gemini-quality-filter", default="gemini-2.5-pro")
    parser.add_argument(
        "--fail-on-missing-expected",
        action="store_true",
        help="Fail if expected-case-list contains IDs missing from standalone output",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result_rows = read_jsonl(args.results_jsonl)
    normalized_rows: List[Dict[str, Any]] = []
    skipped_case_ids: List[str] = []

    for result in result_rows:
        normalized = normalize_result_row(
            result,
            split_name=args.split_name,
            dataset_version=args.dataset_version,
            model_openai_reasoner=args.model_openai_reasoner,
            model_openai_judge=args.model_openai_judge,
            model_gemini_quality_filter=args.model_gemini_quality_filter,
        )
        if normalized is None:
            case_id = str(result.get("case_id", "")).strip()
            if case_id:
                skipped_case_ids.append(case_id)
            continue
        normalized_rows.append(normalized)

    write_jsonl(args.standalone_output, normalized_rows)

    base_rows = read_jsonl(args.base_train_jsonl)
    base_case_ids = {str(row.get("case_id", "")).strip() for row in base_rows if str(row.get("case_id", "")).strip()}

    added_rows: List[Dict[str, Any]] = []
    collision_case_ids: List[str] = []
    for row in normalized_rows:
        case_id = row["case_id"]
        if case_id in base_case_ids:
            collision_case_ids.append(case_id)
            continue
        added_rows.append(row)

    merged_rows = base_rows + added_rows
    write_jsonl(args.merged_output, merged_rows)

    expected_case_ids = read_case_list(args.expected_case_list)
    expected_set = set(expected_case_ids)
    produced_set = {row["case_id"] for row in normalized_rows}
    missing_expected_case_ids = sorted(expected_set - produced_set)
    unexpected_case_ids = sorted(produced_set - expected_set)

    if missing_expected_case_ids and args.fail_on_missing_expected:
        raise ValueError(f"Missing expected case_ids in standalone output: {missing_expected_case_ids}")

    stats = MergeStats(
        expected_case_count=len(expected_set),
        standalone_case_count=len(normalized_rows),
        added_case_count=len(added_rows),
        collision_case_count=len(collision_case_ids),
        missing_expected_case_count=len(missing_expected_case_ids),
    )

    manifest = {
        "phase_name": args.phase_name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_results_jsonl": str(args.results_jsonl),
        "base_train_jsonl": str(args.base_train_jsonl),
        "expected_case_list": str(args.expected_case_list) if args.expected_case_list else None,
        "standalone_output": str(args.standalone_output),
        "merged_output": str(args.merged_output),
        "selected_case_ids": [row["case_id"] for row in normalized_rows],
        "added_case_ids": [row["case_id"] for row in added_rows],
        "collision_case_ids": sorted(collision_case_ids),
        "missing_expected_case_ids": missing_expected_case_ids,
        "unexpected_case_ids": unexpected_case_ids,
        "skipped_case_ids": sorted(set(skipped_case_ids)),
        "models": {
            "openai_reasoner": args.model_openai_reasoner,
            "openai_judge": args.model_openai_judge,
            "gemini_quality_filter": args.model_gemini_quality_filter,
        },
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    merged_case_ids = [str(row.get("case_id", "")).strip() for row in merged_rows if str(row.get("case_id", "")).strip()]
    merged_case_id_set = set(merged_case_ids)

    base_prefix_preserved = merged_rows[: len(base_rows)] == base_rows
    all_added_non_leish = all(not bool(row.get("is_leishmaniasis", True)) for row in added_rows)

    validation = {
        "stats": stats.__dict__,
        "counts": {
            "base_train_rows": len(base_rows),
            "standalone_rows": len(normalized_rows),
            "added_rows": len(added_rows),
            "merged_rows": len(merged_rows),
        },
        "consistency": {
            "merged_count_matches": len(merged_rows) == len(base_rows) + len(added_rows),
            "no_duplicate_case_ids": len(merged_case_ids) == len(merged_case_id_set),
            "baseline_prefix_preserved": base_prefix_preserved,
            "all_added_non_leish": all_added_non_leish,
            "missing_expected_case_count": len(missing_expected_case_ids),
        },
        "details": {
            "collision_case_ids": sorted(collision_case_ids),
            "missing_expected_case_ids": missing_expected_case_ids,
            "unexpected_case_ids": unexpected_case_ids,
            "skipped_case_ids": sorted(set(skipped_case_ids)),
        },
    }
    args.validation_output.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Built non-leish Prompt1-3 outputs ===")
    print(f"phase_name={args.phase_name}")
    print(f"standalone_rows={len(normalized_rows)}")
    print(f"added_rows={len(added_rows)}")
    print(f"collision_rows={len(collision_case_ids)}")
    print(f"missing_expected_case_count={len(missing_expected_case_ids)}")
    print(f"standalone_output={args.standalone_output}")
    print(f"merged_output={args.merged_output}")
    print(f"manifest_output={args.manifest_output}")
    print(f"validation_output={args.validation_output}")


if __name__ == "__main__":
    main()
