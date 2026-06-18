"""Shared confirmatory-signal helpers for retrieval and generation analysis."""

from __future__ import annotations

import re
from typing import Dict, Iterable, Optional


CONFIRMATORY_PATTERNS = (
    r"final diagnosis",
    r"diagnos(?:is|ed) with",
    r"confirmed",
    r"amastigot",
    r"leishman[- ]donovan",
    r"biopsy",
    r"histopath",
    r"bone marrow",
    r"pcr.{0,20}positive",
    r"smear.{0,20}positive",
    r"parasite.{0,20}(seen|identified)",
    r"rk39.{0,20}positive",
    r"species.{0,30}(identified|confirmed)",
    r"responded to",
    r"treated with amphotericin",
    r"treated with antimonial",
)
STRONG_CONFIRMATORY_PATTERNS = (
    r"amastigot",
    r"leishman[- ]donovan",
    r"biopsy",
    r"histopath",
    r"bone marrow",
    r"pcr.{0,20}positive",
    r"smear.{0,20}positive",
    r"parasite.{0,20}(seen|identified)",
    r"rk39.{0,20}positive",
    r"species.{0,30}(identified|confirmed)",
)
NEGATIVE_PATTERNS = (
    r"suspected",
    r"considered",
    r"differential",
    r"ruled out",
    r"negative",
    r"inadequate response",
    r"no evidence of",
    r"not diagnostic",
)
LEISH_QUERY_PATTERNS = (
    r"leish",
    r"amastigot",
    r"kala-azar",
    r"leishman[- ]donovan",
    r"rk39",
    r"bone marrow",
    r"splenomegaly",
    r"hepatosplenomegaly",
    r"mucocutaneous",
    r"cutaneous",
    r"pkdl",
)


def context_support_score(text: str) -> float:
    lowered = (text or "").lower()
    pos = sum(1.0 for pat in CONFIRMATORY_PATTERNS if re.search(pat, lowered))
    neg = sum(1.0 for pat in NEGATIVE_PATTERNS if re.search(pat, lowered))
    return pos - (0.75 * neg)


def context_has_confirmatory_signal(text: str) -> bool:
    lowered = (text or "").lower()
    return any(re.search(pat, lowered) for pat in STRONG_CONFIRMATORY_PATTERNS)


def query_has_leish_signal(query: str) -> bool:
    lowered = (query or "").lower()
    return any(re.search(pat, lowered) for pat in LEISH_QUERY_PATTERNS)


def query_has_confirmatory_signal(query: str) -> bool:
    lowered = (query or "").lower()
    return any(re.search(pat, lowered) for pat in CONFIRMATORY_PATTERNS)


def contexts_have_confirmatory_signal(
    contexts: Iterable[Dict[str, object]],
    *,
    selected_doc_ids: Optional[Iterable[str]] = None,
) -> bool:
    selected = {str(doc_id) for doc_id in (selected_doc_ids or []) if doc_id}
    limit_to_selected = bool(selected)
    for ctx in contexts:
        doc_id = str(ctx.get("doc_id") or "")
        if limit_to_selected and doc_id not in selected:
            continue
        if context_has_confirmatory_signal(str(ctx.get("text") or "")):
            return True
    return False
