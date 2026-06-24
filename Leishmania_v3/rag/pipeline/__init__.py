"""
RAG Pipeline - __init__.py

CRITICAL: Load dotenv and set HF cache paths BEFORE any model imports.
This ensures all model downloads go to /data4t/hf/ not ~/.cache/
"""
import os
from pathlib import Path

# Load .env FIRST before any other imports (override=True to ensure vars are set)
from dotenv import load_dotenv
_env_path = Path(__file__).parent.parent.parent / ".env"
load_dotenv(_env_path, override=True)

# Force HF cache paths - these MUST be set before any HF imports
_HF_PATHS = {
    "HF_HOME": "/data4t/hf",
    "HUGGINGFACE_HUB_CACHE": "/data4t/hf/hub",
    "TRANSFORMERS_CACHE": "/data4t/hf/transformers",
    "SENTENCE_TRANSFORMERS_HOME": "/data4t/hf/sentence-transformers"
}
for key, default in _HF_PATHS.items():
    # Use .env value if present, otherwise use default
    os.environ[key] = os.environ.get(key) or default

# Now safe to import config (which may trigger model imports)
from .config import (
    GOOGLE_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    MODELS,
    COLLECTIONS,
    DATA_ROOT,
    TRAIN_JSONL,
    TEST_JSONL,
    IMAGES_DIR,
    RUNS_DIR,
    JUDGE_MODEL,
    JUDGE_CONFIG,
    GENERATOR_MODEL,
    GENERATOR_MODEL_MEDGEMMA,
    log_env_summary,
)

__all__ = [
    "GOOGLE_API_KEY",
    "QDRANT_API_KEY", 
    "QDRANT_URL",
    "MODELS",
    "COLLECTIONS",
    "DATA_ROOT",
    "TRAIN_JSONL",
    "TEST_JSONL",
    "IMAGES_DIR",
    "RUNS_DIR",
    "JUDGE_MODEL",
    "JUDGE_CONFIG",
    "GENERATOR_MODEL",
    "GENERATOR_MODEL_MEDGEMMA",
    "log_env_summary",
]
