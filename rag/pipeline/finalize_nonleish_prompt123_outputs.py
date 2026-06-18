"""
Finalize Prompt1-3 outputs into standalone and merged p14_v7 JSONL artifacts.

Inputs:
- Prompt1-3 results.jsonl from medcase_multimodal_pipeline
- Prepared raw input JSONL (for stable images/query_image_paths carry-through)
- Base train_p14_v7_normalized.jsonl

Outputs:
- Standalone non-leish normalized JSONL (model-named)
- Merged p14_v7 + non-leish JSONL (model-named)
- Manifest and validation JSON files
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


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


def get_label_contract(dataset_version: str) -> Dict[str, str]:
    return {
        "dataset_version": dataset_version,
        "ground_truth_status": "silver_reference_only",
        "verified_track": "reference_label propagated from the structured multimodal pipeline, not clinician gold ground truth",
        "pseudolabel_track": "Prompt-1 final_diagnosis extracted from the structured multimodal pipeline, not clinician gold ground truth",
        "disclaimer": "Evaluation labels are silver labels derived from GPT-5-mini and Gemini 2.5 Pro pipeline outputs; they are not clinician ground truth.",
    }


def diagnosis_type_from_text(text: str) -> str:
    lowered = (text or "").strip().lower()
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


def extract_case_prompt(result_row: Dict[str, Any]) -> str:
    prompt1 = result_row.get("prompt1_converted_case") or {}
    if isinstance(prompt1, dict):
        return str(prompt1.get("case_prompt") or "").strip()
    return ""


def extract_final_diagnosis(result_row: Dict[str, Any]) -> str:
    prompt1 = result_row.get("prompt1_converted_case") or {}
    if isinstance(prompt1, dict):
        return str(prompt1.get("final_diagnosis") or "").strip()
    return ""


def to_normalized_nonleish_row(
    result_row: Dict[str, Any],
    prepared_raw_row: Dict[str, Any],
    split_name: str,
    source_results_path: Path,
    label_contract: Dict[str, str],
    model_tag: str,
) -> Dict[str, Any]:
    case_id = str(result_row.get("case_id") or "")
    reference = result_row.get("reference_label") or {}

    verified = {
        "diagnosis": str(reference.get("diagnosis") or prepared_raw_row.get("diagnosis") or ""),
        "diagnosis_type": str(reference.get("diagnosis_type") or prepared_raw_row.get("diagnosis_type") or "Non-Leishmaniasis"),
        "species": str(reference.get("species") or prepared_raw_row.get("species") or ""),
        "is_leishmaniasis": bool(reference.get("is_leishmaniasis", prepared_raw_row.get("is_leishmaniasis", False))),
    }

    pseudo_dx = extract_final_diagnosis(result_row)
    pseudo_type = diagnosis_type_from_text(pseudo_dx)
    pseudolabel = {
        "diagnosis": pseudo_dx,
        "diagnosis_type": pseudo_type if pseudo_type else verified["diagnosis_type"],
        "species": verified["species"],
        "is_leishmaniasis": False,
    }

    base_meta = result_row.get("pipeline_metadata") or {}
    merged_meta = dict(base_meta)
    merged_meta.update(
        {
            "model_bundle": model_tag,
            "result_origin": str(source_results_path),
            "nonleish_variant": True,
            "output_role": "standalone_nonleish_prompt123",
        }
    )

    return {
        "case_id": case_id,
        "case_text": extract_case_prompt(result_row),
        "images": prepared_raw_row.get("images") or [],
        "diagnosis": verified["diagnosis"],
        "diagnosis_type": verified["diagnosis_type"],
        "species": verified["species"],
        "is_leishmaniasis": False,
        "labels": {
            "verified": verified,
            "pseudolabel": pseudolabel,
        },
        "query_image_paths": prepared_raw_row.get("query_image_paths") or [],
        "split": split_name,
        "source_jsonl": str(source_results_path),
        "reference_label": {
            **verified,
            "license": str(reference.get("license") or prepared_raw_row.get("license") or ""),
            "provenance": "nonleish_prompt123_reference_label",
            "clinician_validated": False,
            "is_ground_truth": False,
        },
        "pipeline_metadata": merged_meta,
        "label_contract": label_contract,
    }


@dataclass
class FinalizeStats:
    requested_cases: int
    results_rows: int
    normalized_rows: int
    merged_rows: int
    collisions: int


def main() -> None:
    parser = argparse.ArgumentParser(description="Finalize non-leish Prompt1-3 outputs into standalone + merged p14 artifacts")
    parser.add_argument("--results-jsonl", required=True, type=Path)
    parser.add_argument("--prepared-raw-jsonl", required=True, type=Path)
    parser.add_argument("--base-train-p14", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--phase-name", default="phase1a_tierA")
    parser.add_argument("--dataset-version", default="p14_v7")
    parser.add_argument("--model-tag", default="gpt5mini_gemini25pro")
    args = parser.parse_args()

    results_rows = read_jsonl(args.results_jsonl)
    raw_rows = read_jsonl(args.prepared_raw_jsonl)
    base_rows = read_jsonl(args.base_train_p14)

    raw_by_case = {str(r.get("case_id") or ""): r for r in raw_rows}
    label_contract = get_label_contract(args.dataset_version)

    split_name = f"train_{args.dataset_version}_{args.phase_name}_{args.model_tag}_nonleish"
    normalized_rows: List[Dict[str, Any]] = []
    missing_from_raw: List[str] = []

    for result_row in results_rows:
        case_id = str(result_row.get("case_id") or "")
        raw_row = raw_by_case.get(case_id)
        if raw_row is None:
            missing_from_raw.append(case_id)
            continue
        normalized_rows.append(
            to_normalized_nonleish_row(
                result_row=result_row,
                prepared_raw_row=raw_row,
                split_name=split_name,
                source_results_path=args.results_jsonl,
                label_contract=label_contract,
                model_tag=args.model_tag,
            )
        )

    base_case_ids = {str(r.get("case_id") or "") for r in base_rows}
    collisions = [r["case_id"] for r in normalized_rows if r["case_id"] in base_case_ids]
    add_rows = [r for r in normalized_rows if r["case_id"] not in base_case_ids]
    merged_rows = base_rows + add_rows

    standalone_path = args.output_dir / (
        f"train_{args.dataset_version}_nonleish_{args.phase_name}_{args.model_tag}.jsonl"
    )
    merged_path = args.output_dir / (
        f"train_{args.dataset_version}_with_nonleish_{args.phase_name}_{args.model_tag}.jsonl"
    )
    manifest_path = args.output_dir / (
        f"merge_manifest_{args.phase_name}_{args.model_tag}.json"
    )
    validation_path = args.output_dir / (
        f"merge_validation_{args.phase_name}_{args.model_tag}.json"
    )

    n_standalone = write_jsonl(standalone_path, normalized_rows)
    n_merged = write_jsonl(merged_path, merged_rows)

    stats = FinalizeStats(
        requested_cases=len(raw_rows),
        results_rows=len(results_rows),
        normalized_rows=n_standalone,
        merged_rows=n_merged,
        collisions=len(collisions),
    )

    manifest = {
        "phase_name": args.phase_name,
        "model_tag": args.model_tag,
        "dataset_version": args.dataset_version,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "results_jsonl": str(args.results_jsonl),
            "prepared_raw_jsonl": str(args.prepared_raw_jsonl),
            "base_train_p14": str(args.base_train_p14),
        },
        "outputs": {
            "standalone_nonleish": str(standalone_path),
            "merged_train": str(merged_path),
            "validation": str(validation_path),
        },
        "stats": stats.__dict__,
        "missing_from_raw": missing_from_raw,
        "collisions": collisions,
        "added_case_ids": [r["case_id"] for r in add_rows],
    }

    validation = {
        "standalone_case_count": n_standalone,
        "base_case_count": len(base_rows),
        "added_case_count": len(add_rows),
        "merged_case_count": n_merged,
        "expected_merged_case_count": len(base_rows) + len(add_rows),
        "all_added_nonleish": all(not bool(r.get("is_leishmaniasis", True)) for r in add_rows),
        "all_added_have_label_contract": all("label_contract" in r for r in add_rows),
        "all_added_model_tagged": all((r.get("pipeline_metadata") or {}).get("model_bundle") == args.model_tag for r in add_rows),
        "no_collisions_added": len(collisions) == 0,
        "missing_from_raw_count": len(missing_from_raw),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    with validation_path.open("w", encoding="utf-8") as f:
        json.dump(validation, f, ensure_ascii=False, indent=2)

    print("=== Finalize Non-Leish Prompt1-3 Outputs ===")
    print(f"standalone={standalone_path}")
    print(f"merged={merged_path}")
    print(f"manifest={manifest_path}")
    print(f"validation={validation_path}")
    print(json.dumps(stats.__dict__, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
