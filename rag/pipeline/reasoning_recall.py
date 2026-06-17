"""Reasoning recall helpers for structured-cases evaluation.

This module provides:
- Groundtruth reasoning source loading from JSONL artifacts.
- Parsing of model answer reasoning traces.
- Normalization helpers for LLM-as-judge recall payloads.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


REASONING_RECALL_PROMPT = """You are an experienced medical expert tasked with comparing two sets of diagnostic reasons.

Return JSON only with these keys:
- matched_groundtruth_indices: array of integers (1-indexed)
- matched_groundtruth_points: array of strings
- unmatched_groundtruth_points: array of strings
- recall: number
- explanation: string

Instructions:
- Groundtruth reasoning points come from the source case report.
- Predicted reasoning steps come from the model answer.
- For each groundtruth reasoning point, decide whether predicted reasoning contains an equivalent justification.
- A match can be looser in wording, but must preserve the same clinical reason.
- Focus on recall, not precision.
- recall = matched_groundtruth_count / total_groundtruth_points
- If there are zero groundtruth points, return recall = 0.
"""


_NUMBERED_LINE_RE = re.compile(r"^\s*(\d+)\s*[\).:-]\s*(.+?)\s*$")
_BULLET_LINE_RE = re.compile(r"^\s*[-*]\s+(.+?)\s*$")
_MULTISPACE_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"[*_`#>]")

_SKIP_PREFIXES = (
    "primary diagnosis",
    "diagnosis type",
    "species",
    "confidence",
)

_KNOWN_REASONING_SOURCE_IDS = (
    (
        "p14_test_held_out_structure_only_v7_local_reaudit/results.jsonl",
        "p14_v7_reaudit_shared56",
    ),
    (
        "test_pseudolabel_v2_strict/results.jsonl",
        "test_pseudolabel_v2_strict_shared56",
    ),
    (
        "legacy_process14_reasoning_source.jsonl",
        "legacy_process14_reasoning_source",
    ),
    (
        "legacy_process14_reasoning_source_scaffold.jsonl",
        "legacy_process14_reasoning_source_scaffold",
    ),
)


def _clean_text(text: str) -> str:
    s = _MD_RE.sub("", str(text or "")).strip()
    s = _MULTISPACE_RE.sub(" ", s)
    return s


def parse_numbered_reasoning_points(raw_text: str) -> List[str]:
    """Parse numbered reasoning lines into normalized points.

    Expected formats:
    - `1. ...`
    - `2) ...`
    - `3: ...`
    """
    if not raw_text:
        return []

    lines = [line.strip() for line in str(raw_text).splitlines() if line.strip()]
    points: List[str] = []

    for line in lines:
        m = _NUMBERED_LINE_RE.match(line)
        if m:
            item = _clean_text(m.group(2))
            if item:
                points.append(item)

    # Fallback to bullet parsing if numbered parsing found nothing.
    if not points:
        for line in lines:
            m = _BULLET_LINE_RE.match(line)
            if m:
                item = _clean_text(m.group(1))
                if item:
                    points.append(item)

    # Last fallback: split long text into sentence-like chunks.
    if not points:
        chunks = re.split(r"(?<=[.!?])\s+", _clean_text(raw_text))
        points = [chunk for chunk in chunks if chunk]

    # Deduplicate while preserving order.
    seen = set()
    deduped: List[str] = []
    for p in points:
        k = p.lower()
        if k in seen:
            continue
        seen.add(k)
        deduped.append(p)
    return deduped


def parse_predicted_reasoning_steps(answer: str, max_steps: int = 12) -> List[str]:
    """Parse reasoning-like steps from a model answer.

    This function uses a permissive parser because answer formats vary across models.
    """
    if not answer:
        return []

    lines = [line.strip() for line in str(answer).splitlines() if line.strip()]
    steps: List[str] = []

    for line in lines:
        low = line.lower()
        if any(low.startswith(prefix) for prefix in _SKIP_PREFIXES):
            continue

        m_num = _NUMBERED_LINE_RE.match(line)
        if m_num:
            cleaned = _clean_text(m_num.group(2))
            if cleaned:
                steps.append(cleaned)
            continue

        m_bullet = _BULLET_LINE_RE.match(line)
        if m_bullet:
            cleaned = _clean_text(m_bullet.group(1))
            if cleaned:
                steps.append(cleaned)

    if not steps:
        text = _clean_text(answer)
        chunks = re.split(r"(?<=[.!?])\s+", text)
        for chunk in chunks:
            c = chunk.strip()
            if not c:
                continue
            low = c.lower()
            if any(low.startswith(prefix) for prefix in _SKIP_PREFIXES):
                continue
            steps.append(c)
            if len(steps) >= max_steps:
                break

    # Normalize and cap length.
    normalized: List[str] = []
    seen = set()
    for step in steps:
        cleaned = _clean_text(step)
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(cleaned)
        if len(normalized) >= max_steps:
            break
    return normalized


def extract_case_id_from_qid(qid: str) -> str:
    if not qid:
        return ""
    parts = str(qid).split("::", 1)
    return parts[0].strip()


def identify_reasoning_source_id(source_path: str | Path) -> str:
    normalized = str(source_path or "").replace("\\", "/")
    for suffix, source_id in _KNOWN_REASONING_SOURCE_IDS:
        if normalized.endswith(suffix):
            return source_id
    path_obj = Path(normalized) if normalized else Path("unknown_reasoning_source")
    return path_obj.stem or "unknown_reasoning_source"


def load_reasoning_source_map(source_paths: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Load case_id -> reasoning metadata map from JSONL sources.

    First source in the list has priority for duplicate case_ids.
    """
    case_map: Dict[str, Dict[str, Any]] = {}

    for source_path in source_paths:
        if not source_path or not source_path.exists():
            continue
        try:
            with open(source_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    case_id = str(row.get("case_id", "")).strip()
                    if not case_id or case_id in case_map:
                        continue

                    diagnostic_reasoning = (
                        (row.get("prompt1_converted_case") or {}).get("diagnostic_reasoning")
                        or ""
                    )
                    points = parse_numbered_reasoning_points(diagnostic_reasoning)
                    case_map[case_id] = {
                        "groundtruth_reasoning_points": points,
                        "source_path": str(source_path),
                        "source_id": identify_reasoning_source_id(source_path),
                        "groundtruth_raw": diagnostic_reasoning,
                    }
        except Exception:
            # Best-effort loading: skip malformed sources and continue.
            continue

    return case_map


def resolve_groundtruth_payload(
    qid: str,
    case_reasoning_map: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    case_id = extract_case_id_from_qid(qid)
    payload = case_reasoning_map.get(case_id) or {}
    return {
        "case_id": case_id,
        "groundtruth_points": payload.get("groundtruth_reasoning_points") or [],
        "source_path": str(payload.get("source_path") or ""),
        "source_id": str(payload.get("source_id") or ""),
        "groundtruth_raw": str(payload.get("groundtruth_raw") or ""),
    }


def resolve_groundtruth_points(
    qid: str,
    case_reasoning_map: Dict[str, Dict[str, Any]],
) -> Tuple[List[str], str]:
    payload = resolve_groundtruth_payload(qid, case_reasoning_map)
    return payload["groundtruth_points"], payload["source_path"]


def normalize_reasoning_recall_result(
    parsed_payload: Dict[str, Any],
    groundtruth_points: List[str],
) -> Dict[str, Any]:
    """Normalize arbitrary judge payload into deterministic recall output."""
    gt_points = list(groundtruth_points or [])
    total = len(gt_points)

    matched_indices: List[int] = []
    raw_indices = parsed_payload.get("matched_groundtruth_indices")
    if isinstance(raw_indices, list):
        for idx in raw_indices:
            try:
                i = int(idx)
            except Exception:
                continue
            if 1 <= i <= total and i not in matched_indices:
                matched_indices.append(i)

    matched_points = [gt_points[i - 1] for i in matched_indices]
    matched_set = set(matched_indices)
    unmatched_points = [p for i, p in enumerate(gt_points, start=1) if i not in matched_set]

    recall = 0.0
    if total > 0:
        recall = len(matched_points) / float(total)

    explanation = str(parsed_payload.get("explanation") or "")

    return {
        "matched_groundtruth_indices": matched_indices,
        "matched_groundtruth_points": matched_points,
        "unmatched_groundtruth_points": unmatched_points,
        "matched_groundtruth_count": len(matched_points),
        "groundtruth_count": total,
        "recall": recall,
        "explanation": explanation,
    }


def build_reasoning_recall_user_payload(
    groundtruth_points: List[str],
    predicted_steps: List[str],
) -> str:
    payload = {
        "groundtruth_reasoning_points": groundtruth_points,
        "predicted_reasoning_steps": predicted_steps,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
