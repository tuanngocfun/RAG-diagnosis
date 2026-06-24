"""
RAG Package - Leishmania Multimodal RAG Pipeline

This package contains:
- configs/: Shared configuration schemas (prompt_mode, context_mode, manifest)
- cli/: Command-line entrypoints for running experiments
- pipeline/: Core RAG pipeline components (retrieval, generation, evaluation)
- runs/: Output directories for experiment runs
"""

# Re-export key components for convenience
from .configs.prompt_mode import PromptMode, build_rag_prompt
from .configs.context_mode import ContextMode, get_context_selection_params
from .configs.manifest_schema import create_manifest, validate_manifest
from .configs.query_types import normalize_query_type, CANONICAL_QUERY_TYPES

__all__ = [
    "PromptMode",
    "build_rag_prompt",
    "ContextMode",
    "get_context_selection_params",
    "create_manifest",
    "validate_manifest",
    "normalize_query_type",
    "CANONICAL_QUERY_TYPES",
]
