"""
RAG Pipeline package bootstrap.

Loads credentials and cache paths before model imports so the copied
structured-cases workspace can run directly on this server.
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover - optional in lightweight test envs
    def load_dotenv(*args, **kwargs):  # type: ignore[override]
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_legacy_override = os.environ.get("LEGACY_ROOT")
if _legacy_override:
    LEGACY_PROJECT_ROOT = Path(_legacy_override).expanduser()
else:
    _legacy_home = Path("/home/ngocnt/Leishmania_v3")
    LEGACY_PROJECT_ROOT = _legacy_home if _legacy_home.exists() else Path("/data1t/lab/ngocnt/Leishmania_v3")


def _resolve_env_file() -> Path:
    override = os.environ.get("STRUCTURED_CASES_ENV_FILE")
    if override:
        return Path(override).expanduser()

    for candidate in [PROJECT_ROOT / ".env", LEGACY_PROJECT_ROOT / ".env"]:
        if candidate.exists():
            return candidate

    return PROJECT_ROOT / ".env"


_env_path = _resolve_env_file()
if _env_path.exists():
    load_dotenv(_env_path, override=False)

_DEFAULT_HF_ROOT = Path('/data1t/lab/hf-cache')
if not _DEFAULT_HF_ROOT.exists():
    _DEFAULT_HF_ROOT = Path('/mnt/data/hf')

_HF_PATHS = {
    "HF_HOME": str(_DEFAULT_HF_ROOT),
    "HUGGINGFACE_HUB_CACHE": str(_DEFAULT_HF_ROOT / 'hub'),
    "TRANSFORMERS_CACHE": str(_DEFAULT_HF_ROOT / 'transformers'),
    "SENTENCE_TRANSFORMERS_HOME": str(_DEFAULT_HF_ROOT / 'sentence-transformers'),
}
for key, default in _HF_PATHS.items():
    os.environ[key] = os.environ.get(key) or default

from .config import (
    ADAPTIVE_RAG,
    CHUNK_CONFIG,
    COLLECTIONS,
    DATA_ROOT,
    DATASET_VERSION,
    ENTITY_LINKS,
    ENV_PATH,
    GENERATOR_MODEL,
    GENERATOR_MODEL_GEMMA4,
    GENERATOR_MODEL_MEDGEMMA,
    GOOGLE_API_KEY,
    IMAGES_DIR,
    JUDGE_CONFIG,
    JUDGE_MODEL,
    LEGACY_PROJECT_ROOT,
    MODELS,
    PROJECT_ROOT,
    QDRANT_API_KEY,
    QDRANT_URL,
    RUNS_DIR,
    SPLIT_DIR,
    TEST_JSONL,
    TRAIN_JSONL,
    get_dataset_support_snapshot,
    get_runtime_metadata,
    log_env_summary,
)

__all__ = [
    "ADAPTIVE_RAG",
    "CHUNK_CONFIG",
    "COLLECTIONS",
    "DATA_ROOT",
    "DATASET_VERSION",
    "ENTITY_LINKS",
    "ENV_PATH",
    "GENERATOR_MODEL",
    "GENERATOR_MODEL_GEMMA4",
    "GENERATOR_MODEL_MEDGEMMA",
    "GOOGLE_API_KEY",
    "IMAGES_DIR",
    "JUDGE_CONFIG",
    "JUDGE_MODEL",
    "LEGACY_PROJECT_ROOT",
    "MODELS",
    "PROJECT_ROOT",
    "QDRANT_API_KEY",
    "QDRANT_URL",
    "RUNS_DIR",
    "SPLIT_DIR",
    "TEST_JSONL",
    "TRAIN_JSONL",
    "get_dataset_support_snapshot",
    "get_runtime_metadata",
    "log_env_summary",
]
