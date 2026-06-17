"""
Reproducible Reindexing Script for RAG Pipeline

This script provides full reproducibility for indexing operations:
1. Logs all library versions
2. Computes data file hashes
3. Records Qdrant collection states
4. Saves a complete run manifest

Usage:
    python -m rag.pipeline.reindex_pipeline [--lane1] [--lane2] [--images] [--all]
"""
import json
import hashlib
import argparse
from collections import Counter
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from .config import (
    DATA_ROOT,
    TRAIN_JSONL,
    RUNS_DIR,
    QDRANT_URL,
    QDRANT_API_KEY,
    DATASET_VERSION,
)
from .indexer import QdrantIndexer, stable_hash_id
from .chunker import chunk_cases


@dataclass
class ReindexManifest:
    """Manifest for reproducibility tracking."""
    timestamp: str
    git_commit: str = ""
    
    # Data files
    train_file: str = ""
    train_file_hash: str = ""
    train_case_count: int = 0
    
    # Library versions
    versions: Dict[str, str] = field(default_factory=dict)
    
    # Indexing results
    collections_indexed: List[Dict] = field(default_factory=list)
    
    # Environment
    qdrant_url: str = ""
    hf_home: str = ""


def get_file_hash(filepath: Path) -> str:
    """Compute MD5 hash of file."""
    if not filepath.exists():
        return "NOT_FOUND"
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()


def get_library_versions() -> Dict[str, str]:
    """Get versions of key libraries."""
    versions = {}
    
    try:
        import transformers
        versions["transformers"] = transformers.__version__
    except:
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
        versions["open_clip"] = getattr(open_clip, "__version__", "unknown")
    except:
        versions["open_clip"] = "NOT_INSTALLED"
    
    try:
        import sentence_transformers
        versions["sentence_transformers"] = sentence_transformers.__version__
    except:
        versions["sentence_transformers"] = "NOT_INSTALLED"
    
    try:
        import torch
        versions["torch"] = str(torch.__version__)  # Convert to string for YAML
    except ImportError:
        versions["torch"] = "NOT_INSTALLED"
    
    try:
        import numpy
        versions["numpy"] = numpy.__version__
    except:
        versions["numpy"] = "NOT_INSTALLED"
    
    return versions


def get_git_commit() -> str:
    """Get current git commit hash."""
    import subprocess
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True,
            cwd=str(DATA_ROOT.parent)
        )
        return result.stdout.strip()[:12]
    except:
        return "unknown"


def get_qdrant_collection_info(client, collection_name: str) -> Optional[Dict]:
    """Get collection info from Qdrant with proper attribute handling."""
    try:
        info = client.get_collection(collection_name)
        
        # Handle missing vectors_count attribute
        vectors_count = getattr(info, 'vectors_count', info.points_count)
        
        # Get vector dimension safely
        try:
            vector_config = info.config.params.vectors
            if hasattr(vector_config, 'size'):
                vector_size = vector_config.size
            elif isinstance(vector_config, dict):
                vector_size = vector_config.get('size', 0)
            else:
                vector_size = 0
            distance = str(getattr(vector_config, 'distance', 'unknown'))
        except AttributeError:
            vector_size = 0
            distance = "unknown"
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": vectors_count,
            "vector_size": vector_size,
            "distance": distance,
        }
    except Exception as e:
        print(f"Warning: Could not get info for {collection_name}: {e}")
        return None


def summarize_diagnosis_types(cases: List[Dict]) -> Dict[str, int]:
    counts: Counter = Counter()
    for case in cases:
        diagnosis_type = str(case.get("diagnosis_type", "") or "UNKNOWN").strip() or "UNKNOWN"
        counts[diagnosis_type] += 1
    return dict(sorted(counts.items()))


def reindex_lane1(
    cases: List[Dict],
    indexer: QdrantIndexer,
    collection_name: str = f"cases_text_e5_1024_{DATASET_VERSION}",
    strategy: str = "fixed",
    recreate: bool = True
) -> Dict:
    """
    Reindex Lane 1 (E5 text chunks).
    
    Args:
        recreate: If True, delete and recreate collection to avoid stale points
    
    Returns:
        Dict with indexing stats
    """
    print(f"\n{'='*60}")
    print("LANE 1: E5 Text Chunks (1024d)")
    print(f"{'='*60}")
    
    # Recreate collection if requested (recommended for clean reindex)
    if recreate:
        print(f"Recreating collection '{collection_name}' for clean index...")
        indexer.create_collection(collection_name, 1024, recreate=True)
    
    # Chunk cases
    print(f"Chunking with strategy='{strategy}'...")
    chunks = chunk_cases(cases, strategy=strategy)
    print(f"Created {len(chunks)} chunks from {len(cases)} cases")
    
    # Index
    print(f"\nIndexing to '{collection_name}'...")
    n_indexed = indexer.index_chunks_e5(chunks, collection_name)
    
    return {
        "collection": collection_name,
        "encoder": "E5",
        "dimension": 1024,
        "cases": len(cases),
        "chunks": len(chunks),
        "points_indexed": n_indexed,
        "strategy": strategy,
        "recreated": recreate
    }


def reindex_lane2_captions(
    cases: List[Dict],
    indexer: QdrantIndexer,
    collection_name: str = f"captions_biomedclip_512_{DATASET_VERSION}",
    recreate: bool = True
) -> Dict:
    """
    Reindex Lane 2 captions (BiomedCLIP 512d).
    
    Args:
        recreate: If True, delete and recreate collection to avoid stale points
    
    Returns:
        Dict with indexing stats
    """
    print(f"\n{'='*60}")
    print("LANE 2: BiomedCLIP Captions (512d)")
    print(f"{'='*60}")
    
    # Recreate collection if requested
    if recreate:
        print(f"Recreating collection '{collection_name}' for clean index...")
        indexer.create_collection(collection_name, 512, recreate=True)
    
    cases_with_images = [c for c in cases if c.get("images")]
    print(f"Cases with images: {len(cases_with_images)}")
    
    n_indexed = indexer.index_captions_biomedclip(cases, collection_name)
    
    return {
        "collection": collection_name,
        "encoder": "BiomedCLIP",
        "dimension": 512,
        "cases_with_images": len(cases_with_images),
        "points_indexed": n_indexed,
        "recreated": recreate
    }


def reindex_lane2_images(
    cases: List[Dict],
    indexer: QdrantIndexer,
    collection_name: str = f"images_biomedclip_512_{DATASET_VERSION}",
    recreate: bool = True
) -> Dict:
    """
    Reindex Lane 2 images (BiomedCLIP 512d).
    
    Args:
        recreate: If True, delete and recreate collection to avoid stale points
    
    Returns:
        Dict with indexing stats
    """
    print(f"\n{'='*60}")
    print("LANE 2: BiomedCLIP Images (512d)")
    print(f"{'='*60}")
    
    # Recreate collection if requested
    if recreate:
        print(f"Recreating collection '{collection_name}' for clean index...")
        indexer.create_collection(collection_name, 512, recreate=True)
    
    n_indexed = indexer.index_images_biomedclip(cases, collection_name)
    
    return {
        "collection": collection_name,
        "encoder": "BiomedCLIP",
        "dimension": 512,
        "modality": "image",
        "points_indexed": n_indexed,
        "recreated": recreate
    }


def run_reindex(
    train_file: str = "train.jsonl",
    lane1: bool = False,
    lane2_captions: bool = False,
    lane2_images: bool = False,
    all_lanes: bool = False,
    recreate: bool = True
) -> Path:
    """
    Run reproducible reindexing.
    
    Args:
        train_file: Training data file (use clean version)
        lane1: Reindex Lane 1 (E5 text)
        lane2_captions: Reindex Lane 2 captions (BiomedCLIP)
        lane2_images: Reindex Lane 2 images (BiomedCLIP)
        all_lanes: Reindex all lanes
        recreate: If True, delete and recreate collections (recommended)
    
    Returns:
        Path to manifest file
    """
    import os
    
    if all_lanes:
        lane1 = lane2_captions = lane2_images = True
    
    if not any([lane1, lane2_captions, lane2_images]):
        print("No lanes selected. Use --lane1, --lane2, --images, or --all")
        return None
    
    # Initialize manifest
    manifest = ReindexManifest(
        timestamp=datetime.now().isoformat(),
        git_commit=get_git_commit(),
        versions=get_library_versions(),
        qdrant_url=QDRANT_URL,
        hf_home=os.environ.get("HF_HOME", "NOT_SET")
    )
    
    # Load data
    requested_train = Path(train_file)
    if requested_train.is_absolute():
        train_path = requested_train
    else:
        # Prefer active split path used by TRAIN_JSONL; fallback to DATA_ROOT.
        candidate = Path(TRAIN_JSONL).parent / requested_train
        train_path = candidate if candidate.exists() else (DATA_ROOT / requested_train)

    print(f"\n{'='*60}")
    print("LOADING DATA")
    print(f"{'='*60}")
    print(f"Train file: {train_path}")
    
    manifest.train_file = str(train_path)
    manifest.train_file_hash = get_file_hash(train_path)
    
    with open(train_path) as f:
        cases = [json.loads(line) for line in f]
    manifest.train_case_count = len(cases)
    diagnosis_type_counts = summarize_diagnosis_types(cases)
    non_leish_count = diagnosis_type_counts.get("Non-Leishmaniasis", 0)
    
    print(f"Loaded {len(cases)} cases")
    print(f"Non-Leishmaniasis cases: {non_leish_count}")
    print(f"Diagnosis type distribution: {diagnosis_type_counts}")
    print(f"File hash: {manifest.train_file_hash}")
    
    # Print versions
    print(f"\n{'='*60}")
    print("LIBRARY VERSIONS")
    print(f"{'='*60}")
    for lib, ver in manifest.versions.items():
        print(f"  {lib}: {ver}")
    
    # Initialize indexer
    indexer = QdrantIndexer()
    
    # Run indexing (with recreate for clean index)
    if lane1:
        result = reindex_lane1(cases, indexer, recreate=recreate)
        manifest.collections_indexed.append(result)
    
    if lane2_captions:
        result = reindex_lane2_captions(cases, indexer, recreate=recreate)
        manifest.collections_indexed.append(result)
    
    if lane2_images:
        result = reindex_lane2_images(cases, indexer, recreate=recreate)
        manifest.collections_indexed.append(result)
    
    # Get final collection states
    print(f"\n{'='*60}")
    print("FINAL COLLECTION STATES")
    print(f"{'='*60}")
    
    for col_result in manifest.collections_indexed:
        info = get_qdrant_collection_info(indexer.client, col_result["collection"])
        if info:
            col_result["final_state"] = info
            print(f"  {info['name']}: {info['points_count']} points, {info['vector_size']}d")
    
    # Save manifest
    manifest_dir = RUNS_DIR / "reindex_manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    
    manifest_name = f"reindex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    manifest_path = manifest_dir / manifest_name
    
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2)
    
    print(f"\n{'='*60}")
    print("MANIFEST SAVED")
    print(f"{'='*60}")
    print(f"  Path: {manifest_path}")
    
    return manifest_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproducible RAG Pipeline Reindexing")
    parser.add_argument("--train-file", default="train.jsonl",
                        help="Training data file (default: train.jsonl)")
    parser.add_argument("--lane1", action="store_true",
                        help="Reindex Lane 1 (E5 text chunks)")
    parser.add_argument("--lane2", action="store_true",
                        help="Reindex Lane 2 captions (BiomedCLIP)")
    parser.add_argument("--images", action="store_true",
                        help="Reindex Lane 2 images (BiomedCLIP)")
    parser.add_argument("--all", action="store_true",
                        help="Reindex all lanes")
    
    args = parser.parse_args()
    
    run_reindex(
        train_file=args.train_file,
        lane1=args.lane1,
        lane2_captions=args.lane2,
        lane2_images=args.images,
        all_lanes=args.all
    )
