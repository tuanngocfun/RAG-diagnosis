"""
Run Catalog - Track and manage evaluation runs

Creates and maintains runs_catalog.yaml with:
- Run ID, timestamp, git commit
- Evaluation config (qrels, queries, chunking)
- Model versions
- Artifact hashes for reproducibility
- Data fingerprints (per GPT 5.2 feedback)
"""
import json
import hashlib
import subprocess
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
from dataclasses import dataclass, asdict, field

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from . import RUNS_DIR, DATA_ROOT, QDRANT_URL, QDRANT_API_KEY


@dataclass
class RunMetadata:
    """Metadata for a single run with full reproducibility info."""
    run_id: str
    created_at: str
    
    # Evaluation config
    qrels_file: str
    qrels_hash: str
    query_type: str
    n_queries: int
    
    # Model config
    retriever_method: str
    reranker_model: Optional[str] = None
    generator_model: Optional[str] = None
    
    # Chunking config
    chunking_strategy: str = "fixed"
    chunk_max_tokens: int = 400
    
    # Artifact hashes
    retrieval_hash: str = ""
    answers_hash: str = ""
    ragas_hash: str = ""
    
    # Git info
    git_commit: str = ""
    
    # Metrics summary
    metrics: Dict = field(default_factory=dict)
    
    # === DATA FINGERPRINTS (per GPT 5.2 feedback) ===
    train_file: str = ""
    train_hash: str = ""
    test_file: str = ""
    test_hash: str = ""
    
    # Qdrant collection info
    collections_info: List[Dict] = field(default_factory=list)
    
    # Library versions
    library_versions: Dict[str, str] = field(default_factory=dict)


def get_file_hash(filepath: Path) -> str:
    """Get MD5 hash of file contents."""
    if not filepath.exists():
        return ""
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()[:12]


def get_git_commit() -> str:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(DATA_ROOT.parent)
        )
        return result.stdout.strip()[:8]
    except:
        return "unknown"


def _resolve_run_path(path_value: object) -> Path | None:
    if not path_value:
        return None
    return Path(str(path_value)).expanduser()


def _find_answers_artifact(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("answers*.jsonl"))
    return candidates[0] if candidates else None


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries for reproducibility."""
    versions = {}
    
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except ImportError:
        versions["transformers"] = "NOT_INSTALLED"
    
    try:
        import qdrant_client
        # qdrant_client doesn't have __version__, use importlib.metadata
        try:
            from importlib.metadata import version
            versions["qdrant_client"] = version("qdrant-client")
        except Exception:
            versions["qdrant_client"] = "installed"
    except ImportError:
        versions["qdrant_client"] = "NOT_INSTALLED"
    
    try:
        import open_clip
        versions["open_clip"] = getattr(open_clip, "__version__", "0.x")
    except ImportError:
        versions["open_clip"] = "NOT_INSTALLED"
    
    try:
        import torch
        versions["torch"] = str(torch.__version__)
    except ImportError:
        versions["torch"] = "NOT_INSTALLED"
    
    try:
        import sentence_transformers
        versions["sentence_transformers"] = sentence_transformers.__version__
    except ImportError:
        versions["sentence_transformers"] = "NOT_INSTALLED"
    
    return versions


def get_qdrant_collections_info() -> List[Dict]:
    """Get info about relevant Qdrant collections."""
    from qdrant_client import QdrantClient
    
    collections_info = []
    collection_names = [
        "cases_text_e5", "cases_text_e5_1024",
        "captions_biomedclip", "captions_biomedclip_512",
        "images_biomedclip_512"
    ]
    
    try:
        client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            check_compatibility=False,
        )
        
        for col_name in collection_names:
            try:
                info = client.get_collection(col_name)
                # Handle vector config that may be object or dict (multi-vector)
                vector_config = info.config.params.vectors
                if hasattr(vector_config, 'size'):
                    dim = vector_config.size
                    distance = str(vector_config.distance)
                elif isinstance(vector_config, dict):
                    # Multi-vector: get first vector config
                    first_vec = next(iter(vector_config.values()), {})
                    if hasattr(first_vec, 'size'):
                        dim = first_vec.size
                        distance = str(first_vec.distance)
                    else:
                        dim = first_vec.get('size', 0)
                        distance = str(first_vec.get('distance', 'unknown'))
                else:
                    dim = 0
                    distance = 'unknown'
                
                collections_info.append({
                    "name": col_name,
                    "points": info.points_count,
                    "dim": dim,
                    "distance": distance
                })
            except:
                pass  # Collection doesn't exist
    except Exception as e:
        collections_info.append({"error": str(e)})
    
    return collections_info


def create_run_metadata(run_dir: Path) -> RunMetadata:
    """Create metadata for a run from its artifacts with full fingerprinting."""
    run_dir = Path(run_dir)
    
    # Load config
    config_path = run_dir / "run_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {}
    
    # Load summary for metrics
    summary_path = run_dir / "summary.json"
    if summary_path.exists():
        with open(summary_path) as f:
            metrics = json.load(f).get("metrics", {})
    else:
        metrics = {}

    runtime_metadata = config.get("runtime_metadata") or {}
    
    # Get qrels hash
    qrels_file = config.get("qrels_file", "qrels_grade3.json")
    split_dir = _resolve_run_path(runtime_metadata.get("split_dir"))
    qrels_path = Path(qrels_file)
    if not qrels_path.is_absolute():
        if split_dir is not None:
            qrels_path = split_dir / qrels_file
        else:
            qrels_path = DATA_ROOT / qrels_file
    
    # Count queries
    queries_path = run_dir / "queries.json"
    n_queries = 0
    if queries_path.exists():
        with open(queries_path) as f:
            n_queries = len(json.load(f))
    
    query_types = config.get("query_types")
    if isinstance(query_types, list) and query_types:
        query_type = ",".join(str(q) for q in query_types)
    else:
        query_type = config.get("query_type", "Q1_symptom_only")

    answers_artifact = _find_answers_artifact(run_dir)

    train_path = _resolve_run_path(runtime_metadata.get("train_jsonl"))
    test_path = _resolve_run_path(runtime_metadata.get("test_jsonl"))
    train_file = str(train_path) if train_path is not None else ""
    test_file = str(test_path) if test_path is not None else ""
    train_hash = str(runtime_metadata.get("train_jsonl_hash") or "")
    test_hash = str(runtime_metadata.get("test_jsonl_hash") or "")
    if not train_hash and train_path is not None:
        train_hash = get_file_hash(train_path)
    if not test_hash and test_path is not None:
        test_hash = get_file_hash(test_path)

    return RunMetadata(
        run_id=run_dir.name,
        created_at=config.get("created_at", datetime.now().isoformat()),
        qrels_file=qrels_file,
        qrels_hash=get_file_hash(qrels_path),
        query_type=query_type,
        n_queries=n_queries,
        retriever_method=config.get("retriever_method", "hybrid"),
        reranker_model="ncbi/MedCPT-Cross-Encoder" if config.get("rerank") else None,
        generator_model=config.get("generator_model"),
        chunking_strategy=config.get("chunking_strategy", "fixed"),
        retrieval_hash=get_file_hash(run_dir / "retrieval.jsonl"),
        answers_hash=get_file_hash(answers_artifact) if answers_artifact is not None else "",
        ragas_hash=get_file_hash(run_dir / "ragas.jsonl"),
        git_commit=get_git_commit(),
        metrics=metrics,
        # Data fingerprints
        train_file=train_file,
        train_hash=train_hash,
        test_file=test_file,
        test_hash=test_hash,
        collections_info=get_qdrant_collections_info(),
        library_versions=get_library_versions()
    )



def update_catalog() -> Path:
    """Update runs_catalog with all runs in RUNS_DIR."""
    catalog = []
    
    # Directories to skip (not actual evaluation runs)
    skip_dirs = {"reindex_manifests", "__pycache__", ".git"}
    
    for run_dir in sorted(RUNS_DIR.iterdir()):
        if run_dir.is_dir() and not run_dir.name.startswith("."):
            # Skip non-run directories
            if run_dir.name in skip_dirs:
                continue
            try:
                meta = create_run_metadata(run_dir)
                catalog.append(asdict(meta))
            except Exception as e:
                print(f"Warning: Could not process {run_dir.name}: {e}")
    
    # Save catalog
    catalog_path = RUNS_DIR / "runs_catalog.yaml"
    
    if YAML_AVAILABLE:
        with open(catalog_path, "w") as f:
            yaml.dump(catalog, f, default_flow_style=False, sort_keys=False)
    else:
        # Fallback to JSON
        catalog_path = RUNS_DIR / "runs_catalog.json"
        with open(catalog_path, "w") as f:
            json.dump(catalog, f, indent=2)
    
    print(f"✓ Updated catalog with {len(catalog)} runs: {catalog_path}")
    
    return catalog_path


def print_catalog_summary():
    """Print summary table of all runs."""
    catalog_path = RUNS_DIR / "runs_catalog.yaml"
    if not catalog_path.exists():
        catalog_path = RUNS_DIR / "runs_catalog.json"
    
    if not catalog_path.exists():
        print("No catalog found. Run update_catalog() first.")
        return
    
    if catalog_path.suffix == ".yaml" and YAML_AVAILABLE:
        with open(catalog_path) as f:
            catalog = yaml.safe_load(f)
    else:
        with open(catalog_path) as f:
            catalog = json.load(f)
    
    print("\n" + "="*80)
    print("RUNS CATALOG SUMMARY")
    print("="*80)
    print(f"{'Run ID':<25} {'Method':<15} {'Queries':<8} {'nDCG@5':<10} {'MRR':<10}")
    print("-"*80)
    
    for run in catalog:
        metrics = run.get("metrics", {})
        ndcg = metrics.get("ndcg", {}).get("@5", 0)
        mrr = metrics.get("mrr", 0)
        
        print(f"{run['run_id']:<25} {run['retriever_method']:<15} "
              f"{run['n_queries']:<8} {ndcg:<10.4f} {mrr:<10.4f}")


if __name__ == "__main__":
    update_catalog()
    print_catalog_summary()
