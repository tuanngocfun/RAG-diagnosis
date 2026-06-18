#!/usr/bin/env python3
"""Run held-out case evaluation for the presentation appendix.

The script uses the thesis split artifacts as the source of truth:

- official experimental retrieval corpus:
  nonleish_additions/generated/train_phase1b_tierAB.jsonl
- held-out evaluation set: test_p14_v7_normalized.jsonl / eval queries

The backend may use a separate local demo KB. That runtime KB is recorded
independently from the official corpus used for split-exclusion verification.
The script writes fresh local-backend outputs to
data/heldout_evaluation_results/.
"""

from __future__ import annotations

import base64
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import requests
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SPLIT_ROOTS = (
    Path(
        "/media/sf_master-thesis/demo-4/multimodal/"
        "for_gemini25flash-lite/recent_patched_v9/structured_cases_v4_2/"
        "leishmaniasis_verified_v2"
    ),
    Path(
        "/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/"
        "details_analysis/rtx_titan/structured_cases_v4/"
        "leishmaniasis_verified_v2"
    ),
)
DEFAULT_IMAGE_ROOTS = (
    Path("/media/sf_master-thesis/rag/data/leishmaniasis_multimodal/images"),
    Path("/home/ngocnt/Leishmaniasis_v3/data/leishmaniasis_multimodal/images"),
)


def resolve_root(env_name: str, defaults: tuple[Path, ...]) -> Path:
    configured = os.environ.get(env_name)
    candidates = ([Path(configured)] if configured else []) + list(defaults)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


SPLIT_ROOT = resolve_root("LEISH_SPLIT_ROOT", DEFAULT_SPLIT_ROOTS)
IMAGE_ROOT = resolve_root("LEISH_IMAGE_ROOT", DEFAULT_IMAGE_ROOTS)
TEST_JSONL = SPLIT_ROOT / "test_p14_v7_normalized.jsonl"
TRAIN_JSONL = (
    SPLIT_ROOT
    / "nonleish_additions"
    / "generated"
    / "train_phase1b_tierAB.jsonl"
)
EVAL_QUERY_JSONL = SPLIT_ROOT / "eval_queries_p14_v7_mixed56.jsonl"
OUTPUT_DIR = Path(
    os.environ.get(
        "LEISH_HELDOUT_OUTPUT_DIR",
        str(PROJECT_ROOT / "data" / "heldout_evaluation_results"),
    )
).expanduser()
BACKEND_BASE_URL = os.environ.get(
    "LEISH_BACKEND_URL", "http://127.0.0.1:8010"
).rstrip("/")
BACKEND_HEALTH_URL = f"{BACKEND_BASE_URL}/health"
BACKEND_CHAT_URL = f"{BACKEND_BASE_URL}/v1/chat"
REQUEST_PREFIX = os.environ.get("LEISH_CLIENT_REQUEST_PREFIX", "v12c_heldout_eval")
DEVICE_PLATFORM = os.environ.get("LEISH_DEVICE_PLATFORM", "v12c_heldout_evaluation")

SELECTED_CASES = [
    {
        "case_id": "PMC7516301_01",
        "category": "success",
        "notes": "Held-out MCL case; success-style functional example.",
    },
    {
        "case_id": "PMC7456484_01",
        "category": "limitation",
        "notes": "Held-out PKDL case; subtype-differentiation limitation.",
    },
    {
        "case_id": "PMC10026180_04",
        "category": "specificity",
        "notes": "Held-out non-leish case; specificity and safety-boundary example.",
    },
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def by_case_id(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(row.get("case_id", ""), []).append(row)
    return grouped


def normalize_image_path(path: str | None) -> Path | None:
    if not path:
        return None
    direct = Path(path)
    corrected = Path(
        path.replace(
            "/home/ngocnt/Leishmania_v3/",
            "/home/ngocnt/Leishmaniasis_v3/",
        )
    )
    candidates = [direct, corrected]

    marker = "/data/leishmaniasis_multimodal/images/"
    if marker in path:
        relative = path.split(marker, 1)[1]
        candidates.append(IMAGE_ROOT / relative)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def first_image_path(test_case: dict[str, Any], eval_query: dict[str, Any]) -> Path:
    candidates: list[str] = []

    image_path = eval_query.get("image_path")
    if isinstance(image_path, str):
        candidates.append(image_path)

    query_images = eval_query.get("query_images")
    if isinstance(query_images, list):
        candidates.extend(path for path in query_images if isinstance(path, str))

    query_image_paths = test_case.get("query_image_paths")
    if isinstance(query_image_paths, list):
        candidates.extend(path for path in query_image_paths if isinstance(path, str))
    elif isinstance(query_image_paths, str):
        candidates.append(query_image_paths)

    for candidate in candidates:
        normalized = normalize_image_path(candidate)
        if normalized and normalized.exists():
            return normalized

    raise FileNotFoundError(f"No resolved image found for {test_case.get('case_id')}")


def expected_label(eval_query: dict[str, Any]) -> str:
    label = eval_query.get("ground_truth", {})
    diagnosis = label.get("diagnosis", "Unknown")
    diagnosis_type = label.get("diagnosis_type", "Unknown")
    if diagnosis_type and diagnosis_type not in diagnosis:
        return f"{diagnosis} ({diagnosis_type})"
    return diagnosis


def image_to_base64(path: Path) -> str:
    with open(path, "rb") as handle:
        return base64.b64encode(handle.read()).decode("utf-8")


def save_input_png(case_id: str, source_path: Path) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{case_id}_input.png"
    with Image.open(source_path) as image:
        image.thumbnail((900, 900), Image.Resampling.LANCZOS)
        image.save(output_path, "PNG")
    return output_path


def backend_health() -> dict[str, Any]:
    response = requests.get(BACKEND_HEALTH_URL, timeout=10)
    response.raise_for_status()
    health = response.json()
    if health.get("status") != "ok" or not health.get("chat_available"):
        raise RuntimeError(f"Backend is not ready for chat: {health}")
    return health


def run_case(
    case_config: dict[str, str],
    test_case: dict[str, Any],
    eval_query: dict[str, Any],
    health: dict[str, Any],
) -> dict[str, Any]:
    case_id = case_config["case_id"]
    clinical_text = eval_query.get("clinical_context") or test_case.get("case_text") or ""
    if not clinical_text.strip():
        raise ValueError(f"{case_id} has no blinded clinical text")

    image_path = first_image_path(test_case, eval_query)
    png_path = save_input_png(case_id, image_path)

    payload = {
        "messages": [{"role": "user", "content": clinical_text}],
        "image_base64": image_to_base64(image_path),
        "image_filename": image_path.name,
        "client_request_id": f"{REQUEST_PREFIX}_{case_id}",
        "device_platform": DEVICE_PLATFORM,
    }

    started = time.time()
    response = requests.post(BACKEND_CHAT_URL, json=payload, timeout=300)
    response.raise_for_status()
    elapsed = round(time.time() - started, 2)
    raw = response.json()

    reference_label = expected_label(eval_query)
    return {
        "case_id": case_id,
        "silver_reference_label": reference_label,
        "expected_diagnosis": reference_label,
        "category": case_config["category"],
        "notes": case_config["notes"],
        "clinical_text": clinical_text,
        "image_path": str(image_path),
        "input_png_path": str(png_path),
        "assistant_markdown": raw.get("assistant_markdown", ""),
        "safety_state": raw.get("safety_state", "unknown"),
        "evidence": raw.get("evidence", []),
        "metadata": {
            "provider_mode": raw.get("provider_mode", "unknown"),
            "model_name": raw.get("model_name", "unknown"),
            "elapsed_seconds": elapsed,
            "request_id": raw.get("request_id", ""),
            "query_image_tensor_count": raw.get("runtime_metadata", {}).get(
                "query_image_tensor_count", "unknown"
            ),
        },
        "split_provenance": {
            "held_out_source": str(TEST_JSONL),
            "eval_query_source": str(EVAL_QUERY_JSONL),
            "clinical_retrieval_corpus_source": str(TRAIN_JSONL),
            "runtime_retrieval_kb_source": health.get("kb_path", "unknown"),
            "test_count": 1,
            "eval_query_count": 1,
            "train_count": 0,
            "source_terms": {
                "official_experimental_retrieval_corpus": (
                    "121-case phase1b Tier A+B train artifact; used for "
                    "split-exclusion verification"
                ),
                "saved_demo_runtime_kb": (
                    "small local defense demo KB used by the preserved backend "
                    "output"
                ),
                "held_out_evaluation_set": "56 non-indexed evaluation cases",
            },
            "label_status": eval_query.get("label_contract", {}).get(
                "ground_truth_status", "silver_reference_only"
            ),
        },
        "label_contract": eval_query.get("label_contract", {}),
        "raw_response": raw,
        "success": True,
        "error": None,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Split root: {SPLIT_ROOT}")
    print(f"Image root: {IMAGE_ROOT}")
    print(f"Backend: {BACKEND_BASE_URL}")

    for path in [TEST_JSONL, TRAIN_JSONL, EVAL_QUERY_JSONL]:
        if not path.exists():
            raise FileNotFoundError(path)

    test_rows = load_jsonl(TEST_JSONL)
    train_rows = load_jsonl(TRAIN_JSONL)
    eval_rows = load_jsonl(EVAL_QUERY_JSONL)
    test_by_id = by_case_id(test_rows)
    train_by_id = by_case_id(train_rows)
    eval_by_id = by_case_id(eval_rows)

    health = backend_health()
    print("Backend health:")
    print(json.dumps(health, indent=2, ensure_ascii=False))

    results: list[dict[str, Any]] = []
    for case_config in SELECTED_CASES:
        case_id = case_config["case_id"]
        test_matches = test_by_id.get(case_id, [])
        eval_matches = eval_by_id.get(case_id, [])
        train_matches = train_by_id.get(case_id, [])

        if len(test_matches) != 1:
            raise RuntimeError(f"{case_id}: expected 1 held-out row, found {len(test_matches)}")
        if len(eval_matches) != 1:
            raise RuntimeError(f"{case_id}: expected 1 eval-query row, found {len(eval_matches)}")
        if train_matches:
            raise RuntimeError(f"{case_id}: found in clinical retrieval corpus/train split")

        print(f"\nRunning held-out case: {case_id}")
        result = run_case(
            case_config, test_matches[0], eval_matches[0], health
        )
        result_path = OUTPUT_DIR / f"{case_id}_result.json"
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2, ensure_ascii=False)
        print(f"Saved {result_path}")
        results.append(result)
        time.sleep(2)

    summary = {
        "test_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "case_source": "56-case held-out evaluation set",
        "official_experimental_retrieval_corpus": {
            "description": "121-case phase1b Tier A+B train artifact",
            "path": str(TRAIN_JSONL),
            "use": "split-exclusion verification",
        },
        "runtime_retrieval_kb": {
            "description": "small local defense demo KB",
            "path": health.get("kb_path", "unknown"),
            "use": "retrieval support for the saved backend outputs",
        },
        "total_cases": len(results),
        "successful_cases": sum(1 for result in results if result.get("success")),
        "backend_health": health,
        "results": results,
    }
    summary_path = OUTPUT_DIR / "evaluation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"\nSaved {summary_path}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
