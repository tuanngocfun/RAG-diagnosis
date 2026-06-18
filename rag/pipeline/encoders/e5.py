"""
E5 Encoder for Lane 1 Text Retrieval.

Uses intfloat/multilingual-e5-large-instruct as symmetric bi-encoder
(same model for query and document encoding).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..config import MODELS


DEFAULT_REPO_ID = "intfloat/multilingual-e5-large-instruct"


def _existing_path(*candidates: Optional[Union[str, Path]]) -> Optional[Path]:
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if path.exists():
            return path
    return None


def _resolve_snapshot_dir(model_root: Path) -> Optional[Path]:
    if (model_root / "config.json").exists():
        return model_root

    refs_main = model_root / "refs" / "main"
    if refs_main.exists():
        revision = refs_main.read_text(encoding="utf-8").strip()
        snapshot_dir = model_root / "snapshots" / revision
        if snapshot_dir.exists():
            return snapshot_dir

    snapshots_dir = model_root / "snapshots"
    if snapshots_dir.exists():
        snapshots = sorted(path for path in snapshots_dir.iterdir() if path.is_dir())
        if snapshots:
            return snapshots[-1]

    return None


def _resolve_cache_folder(cache_folder: Optional[Union[str, Path]]) -> Path:
    if cache_folder:
        return Path(cache_folder).expanduser()

    hf_home = os.environ.get("HF_HOME")
    return _existing_path(
        os.environ.get("SENTENCE_TRANSFORMERS_HOME"),
        Path(hf_home) / "sentence-transformers" if hf_home else None,
        "/data1t/lab/hf-cache/sentence-transformers",
    ) or Path("/data1t/lab/hf-cache/sentence-transformers")


def _resolve_model_source(model_path: Optional[Union[str, Path]]) -> Union[str, Path]:
    if model_path:
        candidate = Path(str(model_path)).expanduser()
        if candidate.exists():
            snapshot_dir = _resolve_snapshot_dir(candidate)
            return snapshot_dir or candidate
        return str(model_path)

    local_hint = MODELS.get("e5_large")
    if isinstance(local_hint, Path) and local_hint.exists():
        snapshot_dir = _resolve_snapshot_dir(local_hint)
        if snapshot_dir is not None:
            return snapshot_dir

    return DEFAULT_REPO_ID


class E5Encoder:
    """E5-large encoder for text embedding."""

    def __init__(
        self,
        model_path: Union[str, Path, None] = None,
        device: str = None,
        cache_folder: Union[str, Path, None] = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        resolved_cache = _resolve_cache_folder(cache_folder)
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(resolved_cache))

        resolved_model = _resolve_model_source(model_path)

        self.device = device
        self.model_name = str(resolved_model)
        self.cache_folder = resolved_cache

        try:
            self.model = SentenceTransformer(
                self.model_name,
                device=device,
                cache_folder=str(resolved_cache),
            )
        except Exception as exc:
            raise RuntimeError(
                "Failed to load E5 encoder. "
                f"model_source={self.model_name} cache_folder={resolved_cache}. "
                "If this server does not already have the E5 model cached, you need to download "
                "intfloat/multilingual-e5-large-instruct or point the workflow at an existing local cache."
            ) from exc

        self.dimension = 1024

    def encode_query(
        self,
        query: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        if isinstance(query, str):
            query = [query]

        formatted = [f"query: {q}" for q in query]
        embeddings = self.model.encode(
            formatted,
            normalize_embeddings=normalize,
            show_progress_bar=len(formatted) > 10,
        )
        return embeddings

    def encode_document(
        self,
        document: Union[str, List[str]],
        normalize: bool = True,
    ) -> np.ndarray:
        if isinstance(document, str):
            document = [document]

        formatted = [f"passage: {d}" for d in document]
        embeddings = self.model.encode(
            formatted,
            normalize_embeddings=normalize,
            show_progress_bar=len(formatted) > 10,
        )
        return embeddings

    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
        normalize: bool = True,
    ) -> np.ndarray:
        prefix = "query: " if is_query else "passage: "
        formatted = [f"{prefix}{t}" for t in texts]

        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True,
        )
        return embeddings


_encoder = None


def get_e5_encoder(model_path: Union[str, Path, None] = None) -> E5Encoder:
    global _encoder
    if _encoder is None:
        _encoder = E5Encoder(model_path)
    return _encoder


if __name__ == "__main__":
    encoder = E5Encoder()

    query = "What are the symptoms of visceral leishmaniasis?"
    doc = "Visceral leishmaniasis presents with fever, weight loss, and splenomegaly."

    q_emb = encoder.encode_query(query)
    d_emb = encoder.encode_document(doc)

    similarity = np.dot(q_emb[0], d_emb[0])

    print(f"Query embedding shape: {q_emb.shape}")
    print(f"Document embedding shape: {d_emb.shape}")
    print(f"Similarity: {similarity:.4f}")
