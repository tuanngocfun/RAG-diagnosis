"""
RAG Pipeline Configuration.

Resolves the copied structured-cases workspace first, then falls back to the
legacy Leishmania_v3 tree for shared assets such as images, credentials, and
vendored code.
"""

import hashlib
import json
import os
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_legacy_override = os.getenv("LEGACY_ROOT")
if _legacy_override:
    LEGACY_PROJECT_ROOT = Path(_legacy_override).expanduser()
else:
    _legacy_home = Path("/home/ngocnt/Leishmania_v3")
    LEGACY_PROJECT_ROOT = _legacy_home if _legacy_home.exists() else Path("/data1t/lab/ngocnt/Leishmania_v3")


def _normalize_path(value: Optional[os.PathLike | str]) -> Optional[Path]:
    if value is None:
        return None
    return Path(value).expanduser()


def _resolve_path(env_name: str, default_candidates: Iterable[Path]) -> Path:
    override = _normalize_path(os.getenv(env_name))
    if override is not None:
        return override

    candidates = [Path(p) for p in default_candidates]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return candidates[0]


def _load_dotenv_from_candidates() -> Path:
    env_path = _resolve_path(
        "STRUCTURED_CASES_ENV_FILE",
        [
            PROJECT_ROOT / ".env",
            LEGACY_PROJECT_ROOT / ".env",
        ],
    )
    if env_path.exists():
        load_dotenv(env_path, override=False)
    return env_path


ENV_PATH = _load_dotenv_from_candidates()

# =============================================================================
# API Credentials
# =============================================================================
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_URL = os.getenv("QDRANT_URL")

# Validate required keys
if not QDRANT_API_KEY or not QDRANT_URL:
    raise ValueError("QDRANT_API_KEY and QDRANT_URL must be set in .env")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY must be set in .env")

# =============================================================================
# Model Paths (Local HuggingFace Cache)
# =============================================================================
HF_CACHE = Path(os.environ.get("TRANSFORMERS_CACHE", "/mnt/data/hf/transformers"))

MODELS = {
    # Lane 1: Text IR
    "e5_large": HF_CACHE / "models--intfloat--multilingual-e5-large-instruct",
    "medcpt_query": HF_CACHE / "MedCPT-Query-Encoder",
    "medcpt_cross": HF_CACHE / "MedCPT-Cross-Encoder",
    # Lane 2: Vision-Language
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
    # Generation
    "medgemma": HF_CACHE / "models--google--medgemma-4b-it",
}

LEGACY_DATASET_VERSION = "v163_pseudolabel_v2"
DATASET_ARTIFACT_ALIASES: Dict[str, Dict[str, object]] = {
    "p14_v7_phase1b_tierAB": {
        "base_dataset_version": "p14_v7",
        "reuse_existing_eval_artifacts": True,
        "artifact_filenames": {
            "train": "nonleish_additions/generated/train_phase1b_tierAB.jsonl",
            "test": "test_p14_v7_normalized.jsonl",
            "query": "eval_queries_p14_v7.jsonl",
            "query_mixed56": "eval_queries_p14_v7_mixed56.jsonl",
            "qrels_verified": "qrels_p14_v7_verified.json",
            "qrels_pseudolabel": "qrels_p14_v7_pseudolabel.json",
        },
    },
}


def get_dataset_alias_metadata(dataset_version: Optional[str] = None) -> Dict[str, object]:
    version = str(dataset_version or DATASET_VERSION or LEGACY_DATASET_VERSION).strip() or LEGACY_DATASET_VERSION
    metadata = DATASET_ARTIFACT_ALIASES.get(version) or {}
    return {
        "dataset_version": version,
        "base_dataset_version": metadata.get("base_dataset_version", version),
        "reuse_existing_eval_artifacts": bool(metadata.get("reuse_existing_eval_artifacts", False)),
        "artifact_filenames": dict(metadata.get("artifact_filenames") or {}),
    }


def get_dataset_base_version(dataset_version: Optional[str] = None) -> str:
    metadata = get_dataset_alias_metadata(dataset_version)
    return str(metadata.get("base_dataset_version") or metadata["dataset_version"]).strip()


def dataset_reuses_shared_eval_artifacts(dataset_version: Optional[str] = None) -> bool:
    metadata = get_dataset_alias_metadata(dataset_version)
    return bool(metadata.get("reuse_existing_eval_artifacts", False))


def get_dataset_artifact_filenames(dataset_version: Optional[str] = None) -> Dict[str, str]:
    version = str(dataset_version or DATASET_VERSION or LEGACY_DATASET_VERSION).strip() or LEGACY_DATASET_VERSION
    alias_metadata = get_dataset_alias_metadata(version)
    aliased_filenames = alias_metadata.get("artifact_filenames") or {}
    if aliased_filenames:
        return {
            "train": str(aliased_filenames["train"]),
            "test": str(aliased_filenames["test"]),
            "query": str(aliased_filenames["query"]),
            "query_mixed56": str(aliased_filenames["query_mixed56"]),
            "qrels_verified": str(aliased_filenames["qrels_verified"]),
            "qrels_pseudolabel": str(aliased_filenames["qrels_pseudolabel"]),
        }
    if version == LEGACY_DATASET_VERSION:
        return {
            "train": "train_pseudolabel_v2_normalized.jsonl",
            "test": "test_pseudolabel_v2_normalized.jsonl",
            "query": "eval_queries_v163_pseudolabel_v2.jsonl",
            "query_mixed56": "eval_queries_v163_mixed56.jsonl",
            "qrels_verified": "qrels_pseudolabel_verified.json",
            "qrels_pseudolabel": "qrels_pseudolabel_label.json",
        }
    return {
        "train": f"train_{version}_normalized.jsonl",
        "test": f"test_{version}_normalized.jsonl",
        "query": f"eval_queries_{version}.jsonl",
        "query_mixed56": f"eval_queries_{version}_mixed56.jsonl",
        "qrels_verified": f"qrels_{version}_verified.json",
        "qrels_pseudolabel": f"qrels_{version}_pseudolabel.json",
    }


# =============================================================================
# Qdrant Collections - Explicit Dimension Naming per GPT 5.2 Feedback
# =============================================================================
DATASET_VERSION = str(
    os.getenv("STRUCTURED_CASES_DATASET_VERSION", LEGACY_DATASET_VERSION)
).strip() or LEGACY_DATASET_VERSION
DATASET_ARTIFACTS = get_dataset_artifact_filenames(DATASET_VERSION)
STRICT_RETRIEVAL_DATASET_VERSIONS = {"p14_v7", "p14_v7_phase1b_tierAB"}

COLLECTIONS = {
    f"cases_text_e5_1024_{DATASET_VERSION}": {"dim": 1024, "distance": "Cosine", "encoder": "E5"},
    f"cases_text_bm25_{DATASET_VERSION}": {"type": "sparse"},
    f"captions_biomedclip_512_{DATASET_VERSION}": {"dim": 512, "distance": "Cosine", "encoder": "BiomedCLIP"},
    f"images_biomedclip_512_{DATASET_VERSION}": {"dim": 512, "distance": "Cosine", "encoder": "BiomedCLIP"},
    "cases_text_e5_1024": {"dim": 1024, "distance": "Cosine", "encoder": "E5"},
    "captions_biomedclip_512": {"dim": 512, "distance": "Cosine", "encoder": "BiomedCLIP"},
    "images_biomedclip_512": {"dim": 512, "distance": "Cosine", "encoder": "BiomedCLIP"},
}

# =============================================================================
# Data Paths
# =============================================================================
SPLIT_DIR = _resolve_path(
    "STRUCTURED_CASES_SPLIT_DIR",
    [
        PROJECT_ROOT / "leishmaniasis_verified_v2",
        PROJECT_ROOT / "data" / "leishmaniasis_verified_v2",
        LEGACY_PROJECT_ROOT / "data" / "leishmaniasis_verified_v2",
    ],
)
def get_dataset_artifact_path(kind: str, dataset_version: Optional[str] = None, split_dir: Optional[Path] = None) -> Path:
    base_dir = split_dir or SPLIT_DIR
    artifacts = DATASET_ARTIFACTS if dataset_version in (None, DATASET_VERSION) else get_dataset_artifact_filenames(dataset_version)
    return base_dir / artifacts[kind]


IMAGES_DIR = _resolve_path(
    "STRUCTURED_CASES_IMAGES_DIR",
    [
        PROJECT_ROOT / "leishmaniasis_multimodal" / "images",
        PROJECT_ROOT / "data" / "leishmaniasis_multimodal" / "images",
        LEGACY_PROJECT_ROOT / "data" / "leishmaniasis_multimodal" / "images",
    ],
)
DATA_ROOT = IMAGES_DIR.parent
RUNS_DIR = _resolve_path(
    "STRUCTURED_CASES_RUNS_DIR",
    [
        PROJECT_ROOT / "runs",
        PROJECT_ROOT,
    ],
)

TRAIN_JSONL = _normalize_path(
    os.getenv("STRUCTURED_CASES_TRAIN_JSONL") or os.getenv("TRAIN_JSONL_OVERRIDE")
) or (
    get_dataset_artifact_path("train")
)
TEST_JSONL = _normalize_path(
    os.getenv("STRUCTURED_CASES_TEST_JSONL") or os.getenv("TEST_JSONL_OVERRIDE")
) or (
    get_dataset_artifact_path("test")
)

# Legacy paths (for reference only - DO NOT USE for new experiments)
_SPLIT_DIR_V1 = LEGACY_PROJECT_ROOT / "data" / "leishmaniasis_verified"
_SPLIT_DIR_OLD = LEGACY_PROJECT_ROOT / "data" / "leishmaniasis_split"
_TRAIN_JSONL_OLD = DATA_ROOT / "train_clean.jsonl"
_TEST_JSONL_OLD = DATA_ROOT / "test_clean.jsonl"

ENTITY_LINKS = SPLIT_DIR / "case_entity_links.jsonl"

# =============================================================================
# Chunking Parameters
# =============================================================================
CHUNK_CONFIG = {
    "max_tokens": 400,
    "overlap_tokens": 50,
    "strategies": ["fixed", "section", "semantic"],
}

# =============================================================================
# Evaluation Parameters
# =============================================================================
EVAL_CONFIG = {
    "k_values": [5, 10],
    "query_types": ["symptom_only", "symptom_exposure", "image_only"],
}

# =============================================================================
# LLM Judge (Gemini 2.5 Pro for RAGAs)
# =============================================================================
JUDGE_MODEL = "gemini-2.5-pro"
JUDGE_MODEL_FALLBACK = "gemini-2.5-flash"
JUDGE_CONFIG = {
    "temperature": 0,
    "thinking_budget": 1024,
    "max_retries": 3,
    "retry_delay": 2,
}

# =============================================================================
# Generation Models
# =============================================================================
GENERATOR_MODEL = "gemini-2.5-pro"
GENERATOR_MODEL_MEDGEMMA = "google/medgemma-4b-it"
GENERATOR_MODEL_GEMMA4 = "google/gemma-4-E4B-it"
SILVER_LABEL_DISCLAIMER = (
    "Evaluation labels are silver labels derived from GPT-5-mini and Gemini 2.5 Pro "
    "pipeline outputs; they are not clinician ground truth."
)

# =============================================================================
# Adaptive RAG Configuration
# =============================================================================
ADAPTIVE_RAG = {
    "enabled": True,
    "thresholds": {
        "Q1_diagnosis": 0.018,
        "Q3_image_diagnosis": 0.35,
        "Q1_Q3_multimodal_diagnosis": 0.025,
        "default": 0.015,
    },
    "margin_threshold": 0.002,
    "soft_gating": True,
    "low_confidence_k": 1,
    "high_confidence_k": 5,
    "use_norag_prompt_on_fallback": True,
}


def _count_diagnosis_types(path: Path) -> Dict[str, int]:
    counts: Counter = Counter()
    if not path.exists():
        return {}

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            diagnosis_type = str(row.get("diagnosis_type", "") or "UNKNOWN").strip() or "UNKNOWN"
            counts[diagnosis_type] += 1

    return dict(sorted(counts.items()))


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0

    with open(path, "r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _file_hash(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return ""

    digest = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()[:12]


def _mtime_iso(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds")


def get_path_fingerprint(path_like: os.PathLike | str) -> Dict[str, object]:
    path = Path(path_like).expanduser().resolve()
    return {
        "path": str(path),
        "exists": path.exists(),
        "hash": _file_hash(path),
        "rows": _count_jsonl_rows(path) if path.suffix == ".jsonl" else None,
        "mtime": _mtime_iso(path),
    }


def is_strict_retrieval_mode(dataset_version: Optional[str] = None) -> bool:
    override = os.getenv("STRUCTURED_CASES_STRICT_RETRIEVAL")
    if override is not None:
        return str(override).strip().lower() not in {"", "0", "false", "no", "off"}
    version = str(dataset_version or DATASET_VERSION or "").strip()
    return version in STRICT_RETRIEVAL_DATASET_VERSIONS


def get_dataset_support_snapshot() -> Dict[str, object]:
    train_counts = _count_diagnosis_types(TRAIN_JSONL)
    test_counts = _count_diagnosis_types(TEST_JSONL)
    warnings: List[str] = []
    if train_counts.get("Non-Leishmaniasis", 0) == 0:
        warnings.append(
            "Train retrieval corpus contains 0 Non-Leishmaniasis cases; non-leish retrieval support is unavailable."
        )
    return {
        "train_diagnosis_type_counts": train_counts,
        "test_diagnosis_type_counts": test_counts,
        "warnings": warnings,
    }


def get_runtime_metadata() -> Dict[str, object]:
    return {
        "project_root": str(PROJECT_ROOT),
        "legacy_project_root": str(LEGACY_PROJECT_ROOT),
        "env_file": str(ENV_PATH),
        "env_file_exists": ENV_PATH.exists(),
        "split_dir": str(SPLIT_DIR),
        "split_dir_exists": SPLIT_DIR.exists(),
        "data_root": str(DATA_ROOT),
        "data_root_exists": DATA_ROOT.exists(),
        "images_dir": str(IMAGES_DIR),
        "images_dir_exists": IMAGES_DIR.exists(),
        "runs_dir": str(RUNS_DIR),
        "runs_dir_exists": RUNS_DIR.exists(),
        "dataset_version": DATASET_VERSION,
        "dataset_artifacts": DATASET_ARTIFACTS,
        "train_jsonl": str(TRAIN_JSONL),
        "train_jsonl_exists": TRAIN_JSONL.exists(),
        "train_jsonl_rows": _count_jsonl_rows(TRAIN_JSONL),
        "train_jsonl_hash": _file_hash(TRAIN_JSONL),
        "train_jsonl_mtime": _mtime_iso(TRAIN_JSONL),
        "test_jsonl": str(TEST_JSONL),
        "test_jsonl_exists": TEST_JSONL.exists(),
        "test_jsonl_rows": _count_jsonl_rows(TEST_JSONL),
        "test_jsonl_hash": _file_hash(TEST_JSONL),
        "test_jsonl_mtime": _mtime_iso(TEST_JSONL),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "ALL"),
        "transformers_cache": os.environ.get("TRANSFORMERS_CACHE", "NOT SET"),
        "hf_home": os.environ.get("HF_HOME", "NOT SET"),
        "silver_label_disclaimer": SILVER_LABEL_DISCLAIMER,
    }


# =============================================================================
# Environment Summary (for reproducibility logging)
# =============================================================================
def log_env_summary() -> None:
    """Print environment configuration for reproducibility."""
    runtime = get_runtime_metadata()
    print("=" * 60)
    print("ENVIRONMENT SUMMARY")
    print("=" * 60)
    print(f"  PROJECT_ROOT:             {runtime['project_root']}")
    print(f"  LEGACY_PROJECT_ROOT:      {runtime['legacy_project_root']}")
    print(f"  ENV_FILE:                 {runtime['env_file']}")
    print(f"  HF_HOME:                  {os.environ.get('HF_HOME', 'NOT SET')}")
    print(f"  TRANSFORMERS_CACHE:       {os.environ.get('TRANSFORMERS_CACHE', 'NOT SET')}")
    print(f"  SENTENCE_TRANSFORMERS_HOME: {os.environ.get('SENTENCE_TRANSFORMERS_HOME', 'NOT SET')}")
    print(f"  DATA_ROOT:                {DATA_ROOT}")
    print(f"  SPLIT_DIR:                {SPLIT_DIR}")
    print(f"  IMAGES_DIR:               {IMAGES_DIR}")
    print(f"  RUNS_DIR:                 {RUNS_DIR}")
    print(f"  CUDA_VISIBLE_DEVICES:     {runtime['cuda_visible_devices']}")
    print(f"  QDRANT_URL:               {QDRANT_URL}")
    print("=" * 60)
