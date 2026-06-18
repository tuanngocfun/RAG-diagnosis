"""
Run Contract: manifest.json schema definition.

Per GPT 5.2 recommendation: Each run folder MUST contain manifest.json 
to be considered comparable. Runs without manifest are excluded from analysis.
"""
from typing import Dict, Any, Optional
from datetime import datetime
import json


# JSON Schema for manifest.json
MANIFEST_SCHEMA = {
    "type": "object",
    "required": [
        "run_id",
        "dataset_version",
        "query_set_id", 
        "prompt_mode",
        "context_mode",
        "rag_mode",
        "model_name",
        "timestamp"
    ],
    "properties": {
        # === IDENTIFICATION ===
        "run_id": {
            "type": "string",
            "description": "Unique identifier for this run (e.g., 'medgemma4b_rag_v3_20260127')"
        },
        "timestamp": {
            "type": "string",
            "format": "date-time",
            "description": "ISO 8601 timestamp of run start"
        },
        "git_commit": {
            "type": "string",
            "description": "Git commit hash for reproducibility (optional but recommended)"
        },
        
        # === DATASET ===
        "dataset_version": {
            "type": "string",
            "description": "Dataset version identifier (e.g., 'v143' = 143 verified cases)"
        },
        "query_set_id": {
            "type": "string",
            "description": "Canonical query set used (e.g., 'Q1_symptom_only', 'Q1_diagnosis')"
        },
        "n_queries": {
            "type": "integer",
            "description": "Number of queries evaluated"
        },
        
        # === RAG CONFIGURATION ===
        "rag_mode": {
            "type": "string",
            "enum": ["rag", "no_rag"],
            "description": "'rag' = retrieval augmented, 'no_rag' = parametric only"
        },
        "context_mode": {
            "type": "string",
            "description": "Context selection mode: 'top_k:10', 'quality_threshold:0.6', 'dynamic_k'"
        },
        "prompt_mode": {
            "type": "string",
            "enum": ["strict_context", "balanced", "no_context"],
            "description": "Prompt strategy for context integration"
        },
        
        # === MODEL ===
        "model_name": {
            "type": "string",
            "description": "Generator model (e.g., 'gemini-2.5-pro', 'medgemma-4b-it', 'gemma3-12b')"
        },
        "model_config": {
            "type": "object",
            "description": "Model-specific parameters",
            "properties": {
                "temperature": {"type": "number"},
                "top_p": {"type": "number"},
                "top_k": {"type": "integer"},
                "max_tokens": {"type": "integer"}
            }
        },
        
        # === RETRIEVER (for RAG mode) ===
        "retriever": {
            "type": "object",
            "description": "Retriever configuration (null for no_rag)",
            "properties": {
                "encoder": {"type": "string"},
                "method": {"type": "string", "enum": ["dense", "sparse", "hybrid"]},
                "top_k": {"type": "integer"},
                "collection": {"type": "string"}
            }
        },
        
        # === EVALUATION ===
        "judge_model": {
            "type": "string",
            "description": "Model used for LLM-as-Judge evaluation"
        },
        "metrics": {
            "type": "object",
            "description": "Summary metrics (populated after evaluation)",
            "properties": {
                "diagnosis_accuracy": {"type": "number"},
                "context_relevance": {"type": "number"},
                "faithfulness": {"type": "number"}
            }
        }
    }
}


def create_manifest(
    run_id: str,
    dataset_version: str,
    query_set_id: str,
    rag_mode: str,
    context_mode: str,
    prompt_mode: str,
    model_name: str,
    model_config: Optional[Dict] = None,
    retriever: Optional[Dict] = None,
    judge_model: str = "gemini-2.5-pro",
    n_queries: Optional[int] = None,
    git_commit: Optional[str] = None
) -> Dict[str, Any]:
    """Create a valid manifest dictionary."""
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset_version": dataset_version,
        "query_set_id": query_set_id,
        "rag_mode": rag_mode,
        "context_mode": context_mode,
        "prompt_mode": prompt_mode,
        "model_name": model_name,
        "model_config": model_config or {},
        "retriever": retriever,
        "judge_model": judge_model,
    }
    
    if n_queries is not None:
        manifest["n_queries"] = n_queries
    if git_commit:
        manifest["git_commit"] = git_commit
    
    return manifest


def validate_manifest(manifest: Dict[str, Any]) -> tuple[bool, list[str]]:
    """
    Validate manifest against required fields.
    
    Returns:
        (is_valid, list of missing/invalid fields)
    """
    errors = []
    required = MANIFEST_SCHEMA["required"]
    
    for field in required:
        if field not in manifest:
            errors.append(f"Missing required field: {field}")
    
    # Validate enum values
    if "rag_mode" in manifest and manifest["rag_mode"] not in ["rag", "no_rag"]:
        errors.append(f"Invalid rag_mode: {manifest['rag_mode']}")
    
    if "prompt_mode" in manifest and manifest["prompt_mode"] not in ["strict_context", "balanced", "no_context"]:
        errors.append(f"Invalid prompt_mode: {manifest['prompt_mode']}")
    
    return len(errors) == 0, errors


def load_manifest(run_dir) -> Optional[Dict[str, Any]]:
    """Load manifest.json from run directory."""
    from pathlib import Path
    manifest_path = Path(run_dir) / "manifest.json"
    
    if not manifest_path.exists():
        return None
    
    with open(manifest_path) as f:
        return json.load(f)


def save_manifest(manifest: Dict[str, Any], run_dir) -> None:
    """Save manifest.json to run directory."""
    from pathlib import Path
    manifest_path = Path(run_dir) / "manifest.json"
    
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
