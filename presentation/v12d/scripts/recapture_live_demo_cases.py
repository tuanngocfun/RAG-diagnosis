#!/usr/bin/env python3
"""Run the v12d/cases inputs through the live Flutter backend.

This script is intentionally separate from the official V12d replay path. It
uses the exact case text/image artifacts under ``v12d/cases`` and writes fresh
``live_gpu`` backend responses under ``v12d/re-capture`` for deck alignment and
manual evidence review.
"""

from __future__ import annotations

import base64
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CASES_ROOT = PROJECT_ROOT / "cases"
OUTPUT_ROOT = PROJECT_ROOT / "re-capture"
BACKEND_ROOT = os.getenv("MEDICAL_DEMO_BACKEND_URL", "http://127.0.0.1:8010").rstrip("/")

CASES = [
    {
        "case_id": "PMC7516301_01",
        "role": "concordant MCL live-demo case",
        "text_path": CASES_ROOT / "PMC7516301_01" / "input_texts.txt",
        "image_path": CASES_ROOT / "PMC7516301_01" / "image.png",
    },
    {
        "case_id": "PMC7456484_01",
        "role": "PKDL subtype-resolution limitation",
        "text_path": CASES_ROOT / "PMC7456484_01" / "input_texts.txt",
        "image_path": CASES_ROOT / "PMC7456484_01" / "image.png",
    },
    {
        "case_id": "PMC10026180_04",
        "role": "AT/nose-rash full-text stress test",
        "text_path": CASES_ROOT / "PMC10026180_04" / "full_texts" / "input_texts.txt",
        "image_path": CASES_ROOT / "PMC10026180_04" / "full_texts" / "image.png",
    },
]


def extract_field(markdown: str, label: str) -> str:
    match = re.search(re.escape(label) + r"\s*(.*)", markdown)
    return match.group(1).strip() if match else ""


def extract_rank_fields(markdown: str) -> dict[str, str]:
    return {
        "rank1": (
            extract_field(markdown, "**Rank 1 supportive consideration:**")
            or extract_field(markdown, "**Rank 1 (Most Likely):**")
            or extract_field(markdown, "**Rank 1:**")
        ),
        "rank1_type": extract_field(markdown, "**Rank 1 Diagnosis Type:**"),
        "rank1_confidence": extract_field(markdown, "**Rank 1 Confidence:**"),
        "supported_option": (
            extract_field(markdown, "**Most supported option in this research demo:**")
            or extract_field(markdown, "**Chosen Final Diagnosis for Scoring:**")
        ),
    }


def image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def health() -> dict[str, Any]:
    response = requests.get(f"{BACKEND_ROOT}/health", timeout=15)
    response.raise_for_status()
    return response.json()


def run_case(case: dict[str, Any], timestamp: str, output_dir: Path) -> dict[str, Any]:
    case_id = case["case_id"]
    text = case["text_path"].read_text(encoding="utf-8").strip()
    image_path = case["image_path"]
    payload = {
        "client_request_id": f"v12d_recapture_live_gpu_{case_id}_{timestamp}",
        "device_platform": "v12d_recapture_live_demo_alignment",
        "response_mode": "live_gpu",
        "image_base64": image_base64(image_path),
        "image_filename": image_path.name,
        "messages": [{"role": "user", "content": text}],
    }
    started = time.perf_counter()
    response = requests.post(f"{BACKEND_ROOT}/v1/chat", json=payload, timeout=420)
    response.raise_for_status()
    result = response.json()
    elapsed = round(time.perf_counter() - started, 3)
    rank_fields = extract_rank_fields(result.get("assistant_markdown", ""))
    wrapped = {
        "case_id": case_id,
        "role": case["role"],
        "timestamp": timestamp,
        "text_path": str(case["text_path"]),
        "image_path": str(image_path),
        "backend_url": BACKEND_ROOT,
        "elapsed_seconds": elapsed,
        "rank_fields": rank_fields,
        "raw_response": result,
    }
    out_path = output_dir / f"{case_id}_live_gpu_result.json"
    out_path.write_text(json.dumps(wrapped, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{case_id}: {rank_fields['rank1_type']} | {rank_fields['rank1']} ({elapsed}s)")
    return wrapped


def main() -> int:
    timestamp = os.getenv("V12D_RECAPTURE_TIMESTAMP") or datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = OUTPUT_ROOT / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("V12D LIVE DEMO CASE RE-CAPTURE")
    print("=" * 80)
    print(f"timestamp={timestamp}")
    print(f"backend={BACKEND_ROOT}")
    backend_health = health()
    (output_dir / "backend_health_before.json").write_text(
        json.dumps(backend_health, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    if backend_health.get("status") != "ok":
        raise SystemExit(f"Backend not healthy: {backend_health}")
    if backend_health.get("provider_mode") != "real_gpu_gemma4":
        raise SystemExit(f"Expected real_gpu_gemma4, got {backend_health.get('provider_mode')}")
    if backend_health.get("chat_available") is not True:
        raise SystemExit(f"Chat unavailable: {backend_health}")
    if backend_health.get("model_loaded") is not True:
        print("NOTE: model_loaded=false before first live request; first request may load the model.")

    results = [run_case(case, timestamp, output_dir) for case in CASES]
    backend_health_after = health()
    (output_dir / "backend_health_after.json").write_text(
        json.dumps(backend_health_after, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = {
        "timestamp": timestamp,
        "backend_url": BACKEND_ROOT,
        "case_count": len(results),
        "results": [
            {
                "case_id": item["case_id"],
                "role": item["role"],
                "elapsed_seconds": item["elapsed_seconds"],
                **item["rank_fields"],
                "safety_state": item["raw_response"].get("safety_state"),
                "response_source_mode": item["raw_response"].get("response_source_mode"),
                "fresh_generation_executed": item["raw_response"].get("fresh_generation_executed"),
                "image_tensors": (item["raw_response"].get("runtime_metadata") or {}).get(
                    "query_image_tensor_count"
                ),
            }
            for item in results
        ],
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    latest_path = OUTPUT_ROOT / "latest_live_recapture_manifest.json"
    latest_path.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "output_dir": str(output_dir),
                "summary": str(output_dir / "summary.json"),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved summary: {output_dir / 'summary.json'}")
    print(f"Updated manifest: {latest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
