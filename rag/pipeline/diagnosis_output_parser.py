"""Shared parsing helpers for structured diagnosis outputs."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple


SUPPORTED_DIAGNOSIS_TYPES = {
    "CL": ["cutaneous leishmaniasis", "cutaneous", "oriental sore", "baghdad boil", "delhi boil"],
    "VL": ["visceral leishmaniasis", "visceral", "kala-azar", "kala azar"],
    "MCL": ["mucocutaneous leishmaniasis", "mucocutaneous", "espundia"],
    "PKDL": ["post-kala-azar dermal leishmaniasis", "post kala-azar", "pkdl"],
    "DCL": ["diffuse cutaneous leishmaniasis", "dcl"],
    "DsCL": ["disseminated cutaneous leishmaniasis", "dscl", "disseminated cutaneous"],
    "LCL": ["localized cutaneous leishmaniasis", "lcl", "localized cutaneous"],
    "LR": ["leishmaniasis recidivans", "lr", "recidivans"],
    "Ocular": ["ocular leishmaniasis", "ocular"],
    "Veterinary": ["veterinary leishmaniasis", "canine leishmaniasis", "veterinary"],
    "Non-Leishmaniasis": ["non-leishmaniasis", "non leishmaniasis", "not leishmaniasis", "other"],
}

DIAGNOSIS_FAMILY_MAP = {
    "cutaneous_family": {"CL", "LCL", "DCL", "DsCL", "LR"},
    "visceral_family": {"VL", "PKDL"},
    "mucosal_family": {"MCL", "Ocular"},
    "veterinary_family": {"Veterinary"},
    "nonleish_family": {"Non-Leishmaniasis"},
}

LEISH_KEYWORDS = [
    "leish",
    "kala azar",
    "kala-azar",
    "espundia",
    "pkdl",
    "recidivans",
]
NONLEISH_KEYWORDS = [
    "non leish",
    "non-leish",
    "not leish",
    "without leish",
]
NONSPECIFIC_TYPE_ALIASES = {
    "cutaneous",
    "visceral",
    "mucocutaneous",
    "ocular",
    "veterinary",
    "other",
}
ABBREVIATION_TYPE_TOKENS = {
    "cl",
    "vl",
    "mcl",
    "pkdl",
    "dcl",
    "dscl",
    "lcl",
    "lr",
}
STRIP_MARKUP_RE = re.compile(r"[*_`#>\[\]]")
RANK1_TYPE_RE = re.compile(
    r"rank\s*1(?:\s*\([^)]*\))?\s*diagnosis\s*type\s*:\s*(.+)",
    re.IGNORECASE,
)
RANK_N_TYPE_RE = re.compile(
    r"rank\s*(\d+)(?:\s*\([^)]*\))?\s*diagnosis\s*type\s*:\s*(.+)",
    re.IGNORECASE,
)
RANK1_DIAGNOSIS_RE = re.compile(
    r"rank\s*1(?:\s*\([^)]*\))?\s*:\s*(.+)",
    re.IGNORECASE,
)
RANK_N_DIAGNOSIS_RE = re.compile(
    r"rank\s*(\d+)(?:\s*\([^)]*\))?\s*:\s*(.+)",
    re.IGNORECASE,
)
PAREN_VALUE_RE = re.compile(r"\(([^()]+)\)")
TOKEN_SPLIT_RE = re.compile(r"[\s,;/|]+")
BROAD_GT_TYPES = {"CL", "VL"}


@dataclass
class AnswerFormatAnalysis:
    rank1_diagnosis_text: Optional[str]
    rank1_diagnosis_type: Optional[str]
    diagnosis_family: Optional[str]
    diagnosis_family_source: Optional[str]
    answer_format_valid: bool
    answer_format_error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def normalize_diagnosis(diagnosis: str) -> str:
    if not diagnosis:
        return ""
    normalized = diagnosis.lower().strip()
    normalized = re.sub(r"[^\w\s-]", "", normalized)
    normalized = " ".join(normalized.split())
    return normalized


def _strip_markup(text: str) -> str:
    cleaned = STRIP_MARKUP_RE.sub("", text or "")
    cleaned = cleaned.replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(cleaned.split()).strip()


def _clean_field_value(value: str) -> str:
    cleaned = _strip_markup(value)
    cleaned = cleaned.strip(" :-")
    return cleaned


def _line_candidates(answer: str):
    for raw_line in (answer or "").splitlines():
        line = _clean_field_value(raw_line)
        if line:
            yield line


def canonicalize_diagnosis_type(type_text: str) -> Optional[str]:
    normalized = normalize_diagnosis(type_text or "")
    if not normalized:
        return None

    upper = normalized.upper()
    if upper in SUPPORTED_DIAGNOSIS_TYPES:
        return upper

    for canonical, aliases in SUPPORTED_DIAGNOSIS_TYPES.items():
        if normalized == canonical.lower():
            return canonical
        if any(normalized == normalize_diagnosis(alias) for alias in aliases):
            return canonical

    if "non leish" in normalized:
        return "Non-Leishmaniasis"
    return None


def diagnosis_family_from_type(diagnosis_type: Optional[str]) -> Optional[str]:
    canonical = canonicalize_diagnosis_type(diagnosis_type or "")
    if canonical is None:
        return None
    for family, members in DIAGNOSIS_FAMILY_MAP.items():
        if canonical in members:
            return family
    return None


def extract_rank1_diagnosis_type(answer: str) -> Optional[str]:
    for line in _line_candidates(answer):
        match = RANK1_TYPE_RE.search(line)
        if not match:
            continue
        return canonicalize_diagnosis_type(_clean_field_value(match.group(1)))
    return None


def extract_rank1_diagnosis_text(answer: str) -> Optional[str]:
    for line in _line_candidates(answer):
        if "diagnosis type" in line.lower():
            continue
        match = RANK1_DIAGNOSIS_RE.search(line)
        if not match:
            continue
        value = _clean_field_value(match.group(1))
        if value:
            return value
    return None


def extract_ranked_diagnosis_texts(answer: str, max_rank: int = 3) -> Dict[int, str]:
    ranked: Dict[int, str] = {}
    for line in _line_candidates(answer):
        if "diagnosis type" in line.lower():
            continue
        match = RANK_N_DIAGNOSIS_RE.search(line)
        if not match:
            continue
        try:
            rank = int(match.group(1))
        except Exception:
            continue
        if rank < 1 or rank > max_rank or rank in ranked:
            continue
        value = _clean_field_value(match.group(2))
        if value:
            ranked[rank] = value
    return ranked


def extract_ranked_diagnosis_types(answer: str, max_rank: int = 3) -> Dict[int, Optional[str]]:
    ranked: Dict[int, Optional[str]] = {}
    for line in _line_candidates(answer):
        match = RANK_N_TYPE_RE.search(line)
        if not match:
            continue
        try:
            rank = int(match.group(1))
        except Exception:
            continue
        if rank < 1 or rank > max_rank or rank in ranked:
            continue
        ranked[rank] = canonicalize_diagnosis_type(_clean_field_value(match.group(2)))
    return ranked


def _is_explicit_nonleish(text: str) -> bool:
    normalized = normalize_diagnosis(text)
    if not normalized:
        return False
    return any(keyword in normalized for keyword in NONLEISH_KEYWORDS)


def _is_leish_like(text: str) -> bool:
    normalized = normalize_diagnosis(text)
    if not normalized:
        return False
    if _is_explicit_nonleish(normalized):
        return False
    if any(keyword in normalized for keyword in LEISH_KEYWORDS):
        return True
    for canonical, aliases in SUPPORTED_DIAGNOSIS_TYPES.items():
        if canonical == "Non-Leishmaniasis":
            continue
        candidates = [canonical] + aliases
        for candidate in candidates:
            alias_norm = normalize_diagnosis(candidate)
            if not alias_norm or alias_norm in NONSPECIFIC_TYPE_ALIASES or alias_norm in ABBREVIATION_TYPE_TOKENS:
                continue
            if alias_norm in normalized:
                return True
    return False


def _extract_abbreviation_candidates(text: str):
    tokens = []
    for group in PAREN_VALUE_RE.findall(text or ""):
        tokens.extend(TOKEN_SPLIT_RE.split(group))
    normalized = text.replace("(", " ").replace(")", " ").replace("-", " ")
    tokens.extend(TOKEN_SPLIT_RE.split(normalized))
    for token in tokens:
        cleaned = token.strip().strip(".:,;")
        if cleaned:
            yield cleaned


def infer_diagnosis_type_from_text(text: Optional[str]) -> Optional[str]:
    cleaned = _clean_field_value(text or "")
    normalized = normalize_diagnosis(cleaned)
    if not normalized:
        return None

    if _is_explicit_nonleish(cleaned):
        return "Non-Leishmaniasis"

    leish_like = _is_leish_like(cleaned)
    alias_hits = []
    for canonical, aliases in SUPPORTED_DIAGNOSIS_TYPES.items():
        if canonical == "Non-Leishmaniasis":
            continue
        candidates = [canonical] + aliases
        for candidate in candidates:
            alias_norm = normalize_diagnosis(candidate)
            if not alias_norm or alias_norm in NONSPECIFIC_TYPE_ALIASES:
                continue
            if alias_norm in ABBREVIATION_TYPE_TOKENS and not leish_like:
                continue
            if alias_norm in normalized:
                alias_hits.append((len(alias_norm), canonical))
    if alias_hits:
        alias_hits.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return alias_hits[0][1]

    if leish_like:
        for token in _extract_abbreviation_candidates(cleaned):
            canonical = canonicalize_diagnosis_type(token)
            if canonical and canonical != "Non-Leishmaniasis":
                return canonical
        return None

    return "Non-Leishmaniasis"


def infer_diagnosis_type_from_rank1_text(rank1_text: Optional[str]) -> Optional[str]:
    return infer_diagnosis_type_from_text(rank1_text)


def _canonical_ground_truth_type(ground_truth: Optional[Dict[str, Any]]) -> Tuple[Optional[str], str]:
    ground_truth = ground_truth or {}

    gt_type = canonicalize_diagnosis_type(str(ground_truth.get("diagnosis_type") or ""))
    if gt_type:
        return gt_type, "diagnosis_type"

    inferred_from_dx = infer_diagnosis_type_from_text(str(ground_truth.get("diagnosis") or ""))
    if inferred_from_dx:
        return inferred_from_dx, "diagnosis_text_inferred"

    return None, "missing_taxonomy"


def _match_type_with_fallback(pred_type: Optional[str], gt_type: Optional[str]) -> Tuple[bool, str]:
    if pred_type is None or gt_type is None:
        return False, "unscorable_missing_type"

    if pred_type == gt_type:
        return True, "type_exact"

    # When GT only specifies broad CL/VL, allow family-level subtype match.
    if gt_type in BROAD_GT_TYPES:
        pred_family = diagnosis_family_from_type(pred_type)
        gt_family = diagnosis_family_from_type(gt_type)
        if pred_family and gt_family and pred_family == gt_family:
            return True, f"family_fallback_from_{gt_type}"

    return False, "type_mismatch_fail"


def evaluate_ranked_diagnosis_contract(
    answer: str,
    ground_truth: Optional[Dict[str, Any]],
    max_rank: int = 3,
) -> Dict[str, Any]:
    """Deterministic parser-side rank contract used when judge rank metadata is unavailable."""
    result: Dict[str, Any] = {
        "gt_rank": None,
        "top3_hit": None,
        "l3_top1_correct": None,
        "fallback_level": "unscorable_missing_reference",
        "rank_source": "parser",
        "gt_type_canonical": None,
        "gt_type_source": "missing_taxonomy",
        "rank1_diagnosis_text": None,
        "rank1_diagnosis_type": None,
    }

    if not ground_truth:
        return result

    ranked_texts = extract_ranked_diagnosis_texts(answer, max_rank=max_rank)
    ranked_types = extract_ranked_diagnosis_types(answer, max_rank=max_rank)
    for rank, text in ranked_texts.items():
        if ranked_types.get(rank) is None:
            ranked_types[rank] = infer_diagnosis_type_from_text(text)

    rank1_text = ranked_texts.get(1) or extract_rank1_diagnosis_text(answer)
    rank1_type = ranked_types.get(1) or extract_rank1_diagnosis_type(answer)
    result["rank1_diagnosis_text"] = rank1_text
    result["rank1_diagnosis_type"] = rank1_type

    gt_type, gt_type_source = _canonical_ground_truth_type(ground_truth)
    result["gt_type_canonical"] = gt_type
    result["gt_type_source"] = gt_type_source

    has_rank_signal = bool(ranked_texts) or bool(ranked_types)

    if gt_type is None:
        gt_diag_norm = normalize_diagnosis(str((ground_truth or {}).get("diagnosis") or ""))
        if not gt_diag_norm:
            result["fallback_level"] = "unscorable_missing_taxonomy"
            return result

        matched_rank: Optional[int] = None
        for rank in range(1, max_rank + 1):
            pred_norm = normalize_diagnosis(ranked_texts.get(rank, ""))
            if pred_norm and (gt_diag_norm in pred_norm or pred_norm in gt_diag_norm):
                matched_rank = rank
                break

        if matched_rank is None:
            if has_rank_signal:
                result["gt_rank"] = 0
                result["top3_hit"] = 0.0
                result["l3_top1_correct"] = 0.0
                result["fallback_level"] = "diagnosis_text_fallback_no_match"
            else:
                result["fallback_level"] = "unscorable_missing_rank_lines"
            return result

        result["gt_rank"] = matched_rank
        result["top3_hit"] = 1.0
        result["l3_top1_correct"] = 1.0 if matched_rank == 1 else 0.0
        result["fallback_level"] = "diagnosis_text_fallback_match"
        return result

    matched_rank = None
    matched_level = "type_mismatch_fail"
    for rank in range(1, max_rank + 1):
        is_match, level = _match_type_with_fallback(ranked_types.get(rank), gt_type)
        if is_match:
            matched_rank = rank
            matched_level = level
            break

    if matched_rank is None:
        if has_rank_signal:
            result["gt_rank"] = 0
            result["top3_hit"] = 0.0
            result["l3_top1_correct"] = 0.0
            result["fallback_level"] = (
                matched_level if gt_type_source == "diagnosis_type" else f"{matched_level}({gt_type_source})"
            )
        else:
            result["fallback_level"] = "unscorable_missing_rank_lines"
        return result

    result["gt_rank"] = matched_rank
    result["top3_hit"] = 1.0
    result["l3_top1_correct"] = 1.0 if matched_rank == 1 else 0.0
    result["fallback_level"] = (
        matched_level if gt_type_source == "diagnosis_type" else f"{matched_level}({gt_type_source})"
    )
    return result


def analyze_answer_format(answer: str) -> AnswerFormatAnalysis:
    if not (answer or "").strip():
        return AnswerFormatAnalysis(
            rank1_diagnosis_text=None,
            rank1_diagnosis_type=None,
            diagnosis_family=None,
            diagnosis_family_source=None,
            answer_format_valid=False,
            answer_format_error="empty_answer",
        )

    if str(answer).startswith("[Generation Error:"):
        return AnswerFormatAnalysis(
            rank1_diagnosis_text=None,
            rank1_diagnosis_type=None,
            diagnosis_family=None,
            diagnosis_family_source=None,
            answer_format_valid=False,
            answer_format_error="generation_error",
        )

    explicit_type = extract_rank1_diagnosis_type(answer)
    rank1_text = extract_rank1_diagnosis_text(answer)

    diagnosis_type = explicit_type
    family_source = None
    if explicit_type is not None:
        family_source = "explicit_type_line"
    elif rank1_text:
        diagnosis_type = infer_diagnosis_type_from_rank1_text(rank1_text)
        if diagnosis_type == "Non-Leishmaniasis":
            family_source = "rank1_text_nonleish"
        elif diagnosis_type is not None:
            normalized_rank1 = normalize_diagnosis(rank1_text)
            if normalize_diagnosis(diagnosis_type) in normalized_rank1:
                family_source = "rank1_text_alias"
            else:
                family_source = "rank1_text_abbreviation"

    diagnosis_family = diagnosis_family_from_type(diagnosis_type)
    if not rank1_text:
        return AnswerFormatAnalysis(
            rank1_diagnosis_text=None,
            rank1_diagnosis_type=diagnosis_type,
            diagnosis_family=diagnosis_family,
            diagnosis_family_source=family_source,
            answer_format_valid=False,
            answer_format_error="missing_rank1_line",
        )

    if diagnosis_type is None:
        return AnswerFormatAnalysis(
            rank1_diagnosis_text=rank1_text,
            rank1_diagnosis_type=None,
            diagnosis_family=None,
            diagnosis_family_source=None,
            answer_format_valid=False,
            answer_format_error="unparseable_rank1_diagnosis",
        )

    return AnswerFormatAnalysis(
        rank1_diagnosis_text=rank1_text,
        rank1_diagnosis_type=diagnosis_type,
        diagnosis_family=diagnosis_family,
        diagnosis_family_source=family_source,
        answer_format_valid=True,
        answer_format_error="",
    )


def compute_family_metric_details(answer: str, ground_truth: Optional[Dict]) -> Dict[str, object]:
    analysis = analyze_answer_format(answer)
    ground_truth_family = diagnosis_family_from_type((ground_truth or {}).get("diagnosis_type"))
    diagnosis_family_accuracy = None
    if analysis.diagnosis_family is not None and ground_truth_family is not None:
        diagnosis_family_accuracy = 1.0 if analysis.diagnosis_family == ground_truth_family else 0.0
    return {
        "diagnosis_family": analysis.diagnosis_family,
        "diagnosis_family_accuracy": diagnosis_family_accuracy,
        "diagnosis_family_source": analysis.diagnosis_family_source,
        "answer_format_valid": analysis.answer_format_valid,
        "answer_format_error": analysis.answer_format_error,
        "rank1_diagnosis_type": analysis.rank1_diagnosis_type,
        "rank1_diagnosis_text": analysis.rank1_diagnosis_text,
    }


def compute_family_metric(answer: str, ground_truth: Optional[Dict]):
    details = compute_family_metric_details(answer, ground_truth)
    return details["diagnosis_family"], details["diagnosis_family_accuracy"]
