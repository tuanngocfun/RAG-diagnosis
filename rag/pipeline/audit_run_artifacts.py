#!/usr/bin/env python3
"""
Audit whether a saved run used the expected dataset and evaluation contract.

This is intentionally lightweight and reads only saved artifacts from disk.
It does not rebuild, reindex, or rerun evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

CAPTION_COLLECTION_ABSENT_ZERO_COVERAGE_REASON = (
    "caption_collection_absent_due_to_zero_caption_coverage"
)


def _load_json(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""

    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def _fingerprint(path: Path) -> Dict[str, object]:
    resolved = path.expanduser().resolve()
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "hash": _file_hash(resolved),
        "rows": _count_jsonl_rows(resolved) if resolved.suffix == ".jsonl" else None,
        "mtime": _mtime_iso(resolved),
    }


def _print_check(ok: bool, label: str, detail: str) -> None:
    status = "OK" if ok else "FAIL"
    print(f"{status:4} {label}: {detail}")


def _check_expected_dataset(
    *,
    label: str,
    expected_path: Optional[str],
    runtime_metadata: Dict[str, object],
    runtime_path_key: str,
    runtime_hash_key: str,
    runtime_rows_key: str,
) -> int:
    if not expected_path:
        return 0

    expected = _fingerprint(Path(expected_path))
    actual_path = runtime_metadata.get(runtime_path_key)
    actual_hash = runtime_metadata.get(runtime_hash_key)
    actual_rows = runtime_metadata.get(runtime_rows_key)

    failures = 0
    if not actual_path:
        _print_check(False, label, "run_config.json is missing runtime dataset metadata")
        return 1

    actual_resolved = str(Path(str(actual_path)).expanduser().resolve())
    ok_path = actual_resolved == expected["path"]
    _print_check(ok_path, f"{label} path", f"saved={actual_resolved} expected={expected['path']}")
    failures += 0 if ok_path else 1

    if actual_hash:
        ok_hash = str(actual_hash) == str(expected["hash"])
        _print_check(ok_hash, f"{label} hash", f"saved={actual_hash} expected={expected['hash']}")
        failures += 0 if ok_hash else 1
    else:
        _print_check(False, f"{label} hash", "run_config.json does not contain a saved file hash")
        failures += 1

    if actual_rows is not None and expected["rows"] is not None:
        ok_rows = int(actual_rows) == int(expected["rows"])
        _print_check(ok_rows, f"{label} rows", f"saved={actual_rows} expected={expected['rows']}")
        failures += 0 if ok_rows else 1

    return failures


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit a saved run against expected artifacts")
    parser.add_argument("--run-dir", required=True, help="Run directory to audit")
    parser.add_argument("--expected-train-jsonl", default=None, help="Expected active train JSONL path")
    parser.add_argument("--expected-test-jsonl", default=None, help="Expected active test JSONL path")
    parser.add_argument("--expected-qrels", default=None, help="Expected qrels filename")
    parser.add_argument("--expected-queries", default=None, help="Expected eval queries filename")
    parser.add_argument("--expected-image-search", default=None, help="Expected image search mode")
    parser.add_argument("--expected-method", default=None, help="Expected retrieval method")
    parser.add_argument("--require-rerank", action="store_true", help="Fail if rerank was not enabled")
    parser.add_argument("--require-strict-retrieval", action="store_true", help="Fail if strict retrieval mode was not enabled")
    parser.add_argument("--expected-bm25-index", default=None, help="Expected resolved BM25 index path or filename")
    parser.add_argument("--expected-dense-collection", default=None, help="Expected dense collection name")
    parser.add_argument("--expected-caption-collection", default=None, help="Expected caption collection name")
    parser.add_argument("--expected-image-collection", default=None, help="Expected image collection name")
    parser.add_argument(
        "--require-resource-usage",
        default=None,
        help="Comma-separated runtime usage flags to require (dense_lane,bm25_lane,caption_lane,image_lane,rerank)",
    )
    parser.add_argument(
        "--require-baseline-equivalent",
        action="store_true",
        help="Fail if evaluation_contract.baseline_equivalent is not true",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir).expanduser().resolve()
    run_config_path = run_dir / "run_config.json"
    summary_path = run_dir / "summary.json"

    failures = 0

    if not run_config_path.exists():
        _print_check(False, "run_config.json", f"missing at {run_config_path}")
        return 1
    _print_check(True, "run_config.json", str(run_config_path))

    if not summary_path.exists():
        _print_check(False, "summary.json", f"missing at {summary_path}")
        return 1
    _print_check(True, "summary.json", str(summary_path))

    run_config = _load_json(run_config_path)
    summary = _load_json(summary_path)
    runtime_metadata = run_config.get("runtime_metadata") or {}
    evaluation_contract = (
        run_config.get("evaluation_contract")
        or summary.get("evaluation_contract")
        or {}
    )
    retrieval_contract = (
        run_config.get("retrieval_contract")
        or summary.get("retrieval_contract")
        or {}
    )
    caption_support = (
        run_config.get("caption_support")
        or summary.get("caption_support")
        or retrieval_contract.get("caption_support")
        or {}
    )
    retrieval_usage = (
        summary.get("retrieval_usage")
        or retrieval_contract.get("usage")
        or {}
    )
    qrels_audit = summary.get("qrels_audit") or {}

    print(f"Run: {run_dir}")
    print(f"Run ID: {run_config.get('run_id', run_dir.name)}")

    failures += _check_expected_dataset(
        label="train_jsonl",
        expected_path=args.expected_train_jsonl,
        runtime_metadata=runtime_metadata,
        runtime_path_key="train_jsonl",
        runtime_hash_key="train_jsonl_hash",
        runtime_rows_key="train_jsonl_rows",
    )
    failures += _check_expected_dataset(
        label="test_jsonl",
        expected_path=args.expected_test_jsonl,
        runtime_metadata=runtime_metadata,
        runtime_path_key="test_jsonl",
        runtime_hash_key="test_jsonl_hash",
        runtime_rows_key="test_jsonl_rows",
    )

    if args.expected_qrels:
        actual_qrels = run_config.get("qrels_file")
        ok = actual_qrels == args.expected_qrels
        _print_check(ok, "qrels_file", f"saved={actual_qrels} expected={args.expected_qrels}")
        failures += 0 if ok else 1

    if args.expected_queries:
        actual_queries = Path(str(run_config.get("queries_file", ""))).name
        ok = actual_queries == args.expected_queries
        _print_check(ok, "queries_file", f"saved={actual_queries} expected={args.expected_queries}")
        failures += 0 if ok else 1

    if args.expected_image_search:
        actual_image_search = run_config.get("image_search_mode")
        ok = actual_image_search == args.expected_image_search
        _print_check(ok, "image_search_mode", f"saved={actual_image_search} expected={args.expected_image_search}")
        failures += 0 if ok else 1

    if args.expected_method:
        actual_method = run_config.get("retriever_method")
        ok = actual_method == args.expected_method
        _print_check(ok, "retriever_method", f"saved={actual_method} expected={args.expected_method}")
        failures += 0 if ok else 1

    if args.require_rerank:
        actual_rerank = bool(run_config.get("rerank"))
        _print_check(actual_rerank, "rerank", f"saved={actual_rerank} expected=True")
        failures += 0 if actual_rerank else 1

    if args.require_strict_retrieval:
        strict_mode = bool(retrieval_contract.get("strict_mode"))
        _print_check(strict_mode, "strict_retrieval", f"saved={strict_mode} expected=True")
        failures += 0 if strict_mode else 1

    if retrieval_contract:
        resolved_resources = retrieval_contract.get("resolved_resources") or {}
        bm25_info = resolved_resources.get("bm25_index") or {}
        collection_checks = [
            ("dense_collection", args.expected_dense_collection, resolved_resources.get("dense_collection")),
            ("caption_collection", args.expected_caption_collection, resolved_resources.get("caption_collection")),
            ("image_collection", args.expected_image_collection, resolved_resources.get("image_collection")),
        ]
        for label, expected, actual in collection_checks:
            if not expected:
                continue
            ok = actual == expected
            _print_check(ok, label, f"saved={actual} expected={expected}")
            failures += 0 if ok else 1

        if args.expected_bm25_index:
            actual_path = str(bm25_info.get("resolved_path") or "")
            actual_name = Path(actual_path).name if actual_path else ""
            expected_path = str(args.expected_bm25_index)
            ok = actual_path == expected_path or actual_name == expected_path
            _print_check(ok, "bm25_index", f"saved={actual_path} expected={expected_path}")
            failures += 0 if ok else 1

        collection_snapshots = retrieval_contract.get("collections_at_start") or {}
        for label, expected in (
            ("dense_collection", args.expected_dense_collection),
            ("caption_collection", args.expected_caption_collection),
            ("image_collection", args.expected_image_collection),
        ):
            if not expected:
                continue
            snapshot = collection_snapshots.get(label) or {}
            if label != "caption_collection":
                ok = bool(snapshot.get("exists"))
                _print_check(ok, f"{label} exists", f"saved={snapshot}")
                failures += 0 if ok else 1
                continue

            caption_exists = bool(snapshot.get("exists"))
            caption_required = bool(caption_support.get("required_at_bootstrap", True))
            caption_lane_expected = bool(caption_support.get("lane_expected"))
            caption_lane_exercised = bool(caption_support.get("lane_exercised"))
            caption_coverage = caption_support.get("coverage") or {}
            caption_entries = int(caption_coverage.get("caption_entries") or 0)
            caption_absence_reason = caption_support.get("absence_reason")

            if caption_required:
                ok = caption_exists
                _print_check(ok, "caption_collection exists", f"saved={snapshot} required=True")
                failures += 0 if ok else 1
                continue

            consistency_checks = [
                (not caption_lane_expected, "caption_collection optional lane_expected", f"saved={caption_lane_expected} expected=False"),
                (not caption_lane_exercised, "caption_collection optional lane_exercised", f"saved={caption_lane_exercised} expected=False"),
                (caption_entries == 0, "caption_collection optional caption_entries", f"saved={caption_entries} expected=0"),
            ]
            for ok, check_label, detail in consistency_checks:
                _print_check(ok, check_label, detail)
                failures += 0 if ok else 1

            if caption_exists:
                _print_check(True, "caption_collection exists", f"saved={snapshot} required=False")
            else:
                ok = caption_absence_reason == CAPTION_COLLECTION_ABSENT_ZERO_COVERAGE_REASON
                _print_check(
                    ok,
                    "caption_collection absence_reason",
                    f"saved={caption_absence_reason} expected={CAPTION_COLLECTION_ABSENT_ZERO_COVERAGE_REASON}",
                )
                failures += 0 if ok else 1

    if args.require_resource_usage:
        required_usage = [item.strip() for item in str(args.require_resource_usage).split(",") if item.strip()]
        exercised = retrieval_usage.get("resources_exercised") or {}
        for key in required_usage:
            ok = bool(exercised.get(key))
            _print_check(ok, f"resource_usage::{key}", f"saved={exercised.get(key)} expected=True")
            failures += 0 if ok else 1

    if args.require_baseline_equivalent:
        is_equivalent = bool(evaluation_contract.get("baseline_equivalent"))
        _print_check(
            is_equivalent,
            "baseline_equivalent",
            f"saved={evaluation_contract.get('baseline_equivalent')} warnings={evaluation_contract.get('warnings', [])}",
        )
        failures += 0 if is_equivalent else 1

    if qrels_audit:
        print(
            "Primary qrels coverage: "
            f"train_docs_missing={qrels_audit.get('train_docs_missing_from_primary_qrels_count')} "
            f"nonleish_missing={qrels_audit.get('nonleish_train_docs_missing_from_primary_qrels_count')} "
            f"augmented_missing={qrels_audit.get('augmented_train_docs_missing_from_primary_qrels_count')}"
        )
    if retrieval_usage:
        print(
            "Retrieval usage: "
            f"dense={retrieval_usage.get('dense_lane_query_count')} "
            f"bm25={retrieval_usage.get('bm25_lane_query_count')} "
            f"caption={retrieval_usage.get('caption_query_count')} "
            f"image={retrieval_usage.get('image_query_count')} "
            f"rerank={retrieval_usage.get('rerank_query_count')}"
        )

    if failures:
        print(f"AUDIT FAILED with {failures} issue(s)")
        return 1

    print("AUDIT PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
