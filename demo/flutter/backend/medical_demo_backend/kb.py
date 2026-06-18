"""Knowledge base loading and simple lexical retrieval."""

from __future__ import annotations

import json
import re
from collections import Counter
from math import sqrt
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set, Tuple

from .types import KnowledgeChunk, RetrievedEvidence

_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "to",
    "with",
    "year",
    "years",
    "old",
    "patient",
}


def tokenize(text: str) -> List[str]:
    """Convert text into simple lowercased tokens."""
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [word for word in words if len(word) > 2 and word not in _STOPWORDS]


class KnowledgeBase:
    """Small demo knowledge base with lexical scoring."""

    def __init__(self, chunks: List[KnowledgeChunk], source_path: Path):
        self.chunks = list(chunks)
        self.source_path = source_path
        self._token_sets: List[Set[str]] = [set(tokenize(chunk.text + " " + " ".join(chunk.tags))) for chunk in chunks]

    @classmethod
    def from_path(cls, path: Path) -> "KnowledgeBase":
        payload = json.loads(path.read_text(encoding="utf-8"))
        chunks = [
            KnowledgeChunk(
                chunk_id=str(item["chunk_id"]),
                source_case_id=str(item["source_case_id"]),
                title=str(item["title"]),
                diagnosis_label=str(item["diagnosis_label"]),
                text=str(item["text"]),
                tags=[str(tag) for tag in item.get("tags", [])],
                confirmatory=bool(item.get("confirmatory", False)),
            )
            for item in payload
        ]
        return cls(chunks=chunks, source_path=path)

    def search_with_audit(self, query: str, top_k: int = 4) -> Tuple[List[RetrievedEvidence], Dict[str, Any]]:
        """Return top lexical matches plus a transparent retrieval audit."""
        query_tokens = tokenize(query)
        audit: Dict[str, Any] = {
            "retrieval_backend": "local_demo_lexical",
            "kb_path": str(self.source_path),
            "top_k_requested": top_k,
            "candidate_count": 0,
            "returned_count": 0,
            "scoring_method": (
                "lexical token-overlap score with tag/title/confirmatory bonuses"
            ),
            "query_token_count": len(query_tokens),
            "returned_contexts": [],
        }
        if not query_tokens:
            return [], audit

        query_counts = Counter(query_tokens)
        query_set = set(query_tokens)
        scored: List[RetrievedEvidence] = []
        for chunk, chunk_tokens in zip(self.chunks, self._token_sets):
            overlap = query_set & chunk_tokens
            if not overlap:
                continue
            overlap_score = len(overlap) / max(sqrt(len(query_set) * len(chunk_tokens)), 1.0)
            tag_bonus = 0.0
            title_bonus = 0.0
            for tag in chunk.tags:
                if tag.lower() in query_set:
                    tag_bonus += 0.08
            for token in tokenize(chunk.title):
                if token in query_counts:
                    title_bonus += 0.05
            confirmatory_bonus = 0.08 if chunk.confirmatory else 0.0
            score = round(overlap_score + tag_bonus + title_bonus + confirmatory_bonus, 4)
            scored.append(
                RetrievedEvidence(
                    chunk_id=chunk.chunk_id,
                    source_case_id=chunk.source_case_id,
                    title=chunk.title,
                    diagnosis_label=chunk.diagnosis_label,
                    text=chunk.text,
                    score=score,
                    confirmatory=chunk.confirmatory,
                )
            )
        scored.sort(key=lambda item: item.score, reverse=True)
        returned = scored[: max(1, top_k)]
        audit["candidate_count"] = len(scored)
        audit["returned_count"] = len(returned)
        audit["returned_contexts"] = [
            {
                "rank": rank,
                **item.to_response_dict(),
            }
            for rank, item in enumerate(returned, start=1)
        ]
        return returned, audit

    def search(self, query: str, top_k: int = 4) -> List[RetrievedEvidence]:
        """Return top lexical matches."""
        contexts, _audit = self.search_with_audit(query, top_k=top_k)
        return contexts
