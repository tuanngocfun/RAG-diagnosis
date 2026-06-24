"""
Query Types: Canonical query type definitions and normalization.

Per GPT 5.2 recommendation: Query type mismatch (RAG uses Q1_symptom_only 
vs No-RAG uses Q1_diagnosis) is a major confounder. This module provides
a single canonical mapping for all query types.
"""
from typing import Optional, Dict


# =============================================================================
# Canonical Query Types
# =============================================================================

CANONICAL_QUERY_TYPES = {
    # === PRIMARY QUERY TYPES (use these) ===
    "Q1_symptom_only": {
        "description": "Clinical symptoms description only, no exposure history",
        "includes_symptoms": True,
        "includes_exposure": False,
        "includes_images": False,
        "task": "diagnosis"
    },
    "Q1_symptom_exposure": {
        "description": "Clinical symptoms with geographic/exposure history",
        "includes_symptoms": True,
        "includes_exposure": True,
        "includes_images": False,
        "task": "diagnosis"
    },
    "Q3_image_only": {
        "description": "Clinical image(s) only, no text description",
        "includes_symptoms": False,
        "includes_exposure": False,
        "includes_images": True,
        "task": "diagnosis"
    },
    "Q1_Q3_multimodal": {
        "description": "Combined symptoms + exposure + images (full multimodal)",
        "includes_symptoms": True,
        "includes_exposure": True,
        "includes_images": True,
        "task": "diagnosis"
    },
    
    # === LEGACY / ALIAS TYPES (map to canonical) ===
    # These are kept for backward compatibility with existing runs
}


# Alias mapping: legacy names → canonical names
QUERY_TYPE_ALIASES = {
    # Common variations
    "symptom_only": "Q1_symptom_only",
    "symptom-only": "Q1_symptom_only",
    "symptoms_only": "Q1_symptom_only",
    
    "symptom_exposure": "Q1_symptom_exposure",
    "symptom-exposure": "Q1_symptom_exposure",
    "symptoms_exposure": "Q1_symptom_exposure",
    
    "image_only": "Q3_image_only",
    "image-only": "Q3_image_only",
    "images_only": "Q3_image_only",
    
    "multimodal": "Q1_Q3_multimodal",
    "multimodal_diagnosis": "Q1_Q3_multimodal",
    "Q1_Q3_multimodal_diagnosis": "Q1_Q3_multimodal",
    
    # Old naming conventions
    "Q1_diagnosis": "Q1_symptom_only",  # Legacy: diagnosis without specifying input type
    "diagnosis": "Q1_symptom_only",
}


def normalize_query_type(query_type: str) -> str:
    """
    Normalize query type to canonical form.
    
    Args:
        query_type: Input query type (may be alias or canonical)
        
    Returns:
        Canonical query type string
        
    Raises:
        ValueError: If query type is unknown
    """
    # Already canonical
    if query_type in CANONICAL_QUERY_TYPES:
        return query_type
    
    # Check aliases
    if query_type in QUERY_TYPE_ALIASES:
        return QUERY_TYPE_ALIASES[query_type]
    
    # Try case-insensitive match
    query_type_lower = query_type.lower()
    for alias, canonical in QUERY_TYPE_ALIASES.items():
        if alias.lower() == query_type_lower:
            return canonical
    
    raise ValueError(
        f"Unknown query type: '{query_type}'. "
        f"Valid types: {list(CANONICAL_QUERY_TYPES.keys())}"
    )


def get_query_type_config(query_type: str) -> Dict:
    """Get configuration for a query type."""
    canonical = normalize_query_type(query_type)
    return CANONICAL_QUERY_TYPES[canonical]


def is_multimodal_query(query_type: str) -> bool:
    """Check if query type includes images."""
    config = get_query_type_config(query_type)
    return config.get("includes_images", False)


def list_canonical_query_types() -> list:
    """List all canonical query types."""
    return list(CANONICAL_QUERY_TYPES.keys())


# =============================================================================
# Query Set IDs (for manifest)
# =============================================================================

# Predefined query sets for consistent experiments
QUERY_SETS = {
    "test_v143_symptom": {
        "description": "29 test cases, Q1_symptom_only queries",
        "query_types": ["Q1_symptom_only"],
        "dataset_version": "v143",
        "n_queries": 29
    },
    "test_v143_multimodal": {
        "description": "29 test cases, multimodal queries (symptoms + images)",
        "query_types": ["Q1_Q3_multimodal"],
        "dataset_version": "v143",
        "n_queries": 29
    },
    "test_v143_full": {
        "description": "29 test cases, all query types",
        "query_types": ["Q1_symptom_only", "Q1_symptom_exposure", "Q3_image_only", "Q1_Q3_multimodal"],
        "dataset_version": "v143",
        "n_queries": 116  # 29 * 4
    }
}
