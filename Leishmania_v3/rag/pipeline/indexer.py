"""
Qdrant Indexer for RAG Pipeline

Indexes train cases into Qdrant collections:
- cases_text_e5_1024: Dense vectors from E5 encoder (1024d)
- captions_biomedclip_512: Dense vectors from BiomedCLIP text encoder (512d)
- images_biomedclip_512: Dense vectors from BiomedCLIP image encoder (512d)

Per GPT 5.2 Feedback:
- Uses stable hashing (hashlib.sha256) instead of Python hash()
- Explicit dimension-based collection naming
- True BiomedCLIP for Lane 2 (not E5 fallback)
"""
import json
import hashlib
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import numpy as np

from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import (
    VectorParams,
    Distance,
    PointStruct,
)

from .config import (
    QDRANT_API_KEY,
    QDRANT_URL,
    COLLECTIONS,
    TRAIN_JSONL,
    IMAGES_DIR,
)
from .chunker import chunk_cases, Chunk
from .encoders.e5 import get_e5_encoder
from .encoders.bm25 import get_bm25_retriever


def stable_hash_id(point_id: str) -> int:
    """
    Generate a stable, reproducible integer ID from a string.
    
    Uses SHA-256 hash truncated to 63 bits for Qdrant compatibility.
    This is deterministic across Python processes (unlike hash()).
    
    Args:
        point_id: String identifier
    
    Returns:
        Positive int64 suitable for Qdrant point ID
    """
    hash_bytes = hashlib.sha256(point_id.encode('utf-8')).digest()
    # Use first 8 bytes, interpret as unsigned int, then mask to 63 bits
    hash_int = int.from_bytes(hash_bytes[:8], byteorder='big', signed=False)
    return hash_int % (2**63)  # Ensure positive int64


class QdrantIndexer:
    """
    Indexer for Qdrant vector database.
    
    Supports:
    - Lane 1: E5 text chunks (1024d)
    - Lane 2: BiomedCLIP captions and images (512d)
    """
    
    def __init__(self):
        """Initialize Qdrant client."""
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )
    
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        distance: str = "Cosine",
        recreate: bool = False
    ) -> None:
        """
        Create a Qdrant collection if it doesn't exist.
        
        Args:
            collection_name: Name of the collection
            dimension: Vector dimension
            distance: Distance metric (Cosine, Euclidean, Dot)
            recreate: If True, delete and recreate existing collection
        """
        distance_map = {
            "Cosine": Distance.COSINE,
            "Euclidean": Distance.EUCLID,
            "Dot": Distance.DOT,
        }
        
        # Check if exists
        collections = self.client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            if recreate:
                print(f"Deleting existing collection '{collection_name}'...")
                self.client.delete_collection(collection_name)
            else:
                print(f"Collection '{collection_name}' already exists")
                return
        
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=dimension,
                distance=distance_map.get(distance, Distance.COSINE)
            )
        )
        print(f"Created collection '{collection_name}' (dim={dimension}, dist={distance})")
    
    def index_chunks_e5(
        self,
        chunks: List[Chunk],
        collection_name: str = "cases_text_e5_1024",
        batch_size: int = 32
    ) -> int:
        """
        Index text chunks using E5 encoder (1024d).
        
        Args:
            chunks: List of Chunk objects
            collection_name: Target collection name
            batch_size: Batch size for encoding
        
        Returns:
            Number of indexed points
        """
        encoder = get_e5_encoder()
        
        # Create collection if needed
        self.create_collection(collection_name, encoder.dimension)
        
        # Process in batches
        points = []
        
        for i in tqdm(range(0, len(chunks), batch_size), desc="Encoding chunks (E5)"):
            batch = chunks[i:i + batch_size]
            texts = [c.text for c in batch]
            
            # Encode as documents
            embeddings = encoder.encode_document(texts)
            
            for j, chunk in enumerate(batch):
                point_id = f"{chunk.case_id}_{chunk.chunk_idx}"
                
                points.append(PointStruct(
                    id=stable_hash_id(point_id),  # FIXED: stable hash
                    vector=embeddings[j].tolist(),
                    payload={
                        "case_id": chunk.case_id,
                        "chunk_idx": chunk.chunk_idx,
                        "text": chunk.text[:500],  # Truncate for payload
                        "strategy": chunk.strategy,
                        "start_char": chunk.start_char,
                        "end_char": chunk.end_char,
                    }
                ))
        
        # Upsert to Qdrant
        self.client.upsert(
            collection_name=collection_name,
            points=points
        )
        
        print(f"Indexed {len(points)} chunks to '{collection_name}'")
        return len(points)
    
    def index_captions_biomedclip(
        self,
        cases: List[Dict],
        collection_name: str = "captions_biomedclip_512",
        batch_size: int = 32
    ) -> int:
        """
        Index image captions using BiomedCLIP text encoder (512d).
        
        This is TRUE multimodal: captions are encoded into the same 
        embedding space as images, enabling cross-modal retrieval.
        
        Args:
            cases: List of case dictionaries with 'images' field
            collection_name: Target collection name
            batch_size: Batch size for encoding
        
        Returns:
            Number of indexed points
        """
        from .encoders.biomedclip import get_biomedclip_encoder
        encoder = get_biomedclip_encoder()
        
        # Extract all captions
        captions = []
        for case in cases:
            case_id = case.get("case_id", "")
            for img in case.get("images", []):
                caption = img.get("caption", "")
                # Use 'file' key (actual data structure) or 'file_name' for compatibility
                file_name = img.get("file") or img.get("file_name", "")
                if caption:
                    captions.append({
                        "case_id": case_id,
                        "image_id": file_name,
                        "caption": caption
                    })
        
        if not captions:
            print("No captions found")
            return 0
        
        # Create 512d collection
        self.create_collection(collection_name, encoder.dimension)
        
        # Encode in batches
        points = []
        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions (BiomedCLIP)"):
            batch = captions[i:i + batch_size]
            texts = [c["caption"] for c in batch]
            
            # Use BiomedCLIP text encoder
            embeddings = encoder.encode_text(texts)
            
            for j, cap in enumerate(batch):
                point_id = f"{cap['case_id']}_{cap['image_id']}"
                points.append(PointStruct(
                    id=stable_hash_id(point_id),  # FIXED: stable hash
                    vector=embeddings[j].tolist(),
                    payload={
                        "case_id": cap["case_id"],
                        "image_id": cap["image_id"],
                        "caption": cap["caption"]
                    }
                ))
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} captions to '{collection_name}'")
        return len(points)
    
    def index_images_biomedclip(
        self,
        cases: List[Dict],
        collection_name: str = "images_biomedclip_512",
        batch_size: int = 16
    ) -> int:
        """
        Index actual images using BiomedCLIP image encoder (512d).
        
        This enables TRUE cross-modal retrieval: text query → image results.
        
        Args:
            cases: List of case dictionaries with 'images' field
            collection_name: Target collection name
            batch_size: Batch size for encoding (lower due to image memory)
        
        Returns:
            Number of indexed points
        """
        from .encoders.biomedclip import get_biomedclip_encoder
        encoder = get_biomedclip_encoder()
        
        # Create 512d collection
        self.create_collection(collection_name, encoder.dimension)
        
        # Extract all valid image paths
        images = []
        for case in cases:
            case_id = case.get("case_id", "")
            for img in case.get("images", []):
                file_name = img.get("file") or img.get("file_name", "")
                if file_name:
                    # Images are in subdirectory by case_id
                    img_path = IMAGES_DIR / case_id / file_name
                    if img_path.exists():
                        images.append({
                            "case_id": case_id,
                            "image_id": file_name,
                            "path": str(img_path),
                            "caption": img.get("caption", "")
                        })
        
        if not images:
            print("No images found on disk")
            return 0
        
        print(f"Found {len(images)} images on disk")
        
        # Encode in batches
        points = []
        for i in tqdm(range(0, len(images), batch_size), desc="Encoding images (BiomedCLIP)"):
            batch = images[i:i + batch_size]
            paths = [img["path"] for img in batch]
            
            try:
                # Use BiomedCLIP image encoder
                embeddings = encoder.encode_image(paths)
                
                for j, img in enumerate(batch):
                    point_id = f"img_{img['case_id']}_{img['image_id']}"
                    points.append(PointStruct(
                        id=stable_hash_id(point_id),
                        vector=embeddings[j].tolist(),
                        payload={
                            "case_id": img["case_id"],
                            "image_id": img["image_id"],
                            "caption": img["caption"],
                            "path": img["path"]
                        }
                    ))
            except Exception as e:
                print(f"Error encoding batch {i}: {e}")
                continue
        
        if points:
            self.client.upsert(collection_name=collection_name, points=points)
            print(f"Indexed {len(points)} images to '{collection_name}'")
        
        return len(points)
    
    # Legacy method for backward compatibility
    def index_captions(
        self,
        cases: List[Dict],
        collection_name: str = "captions_biomedclip_512",
        use_biomedclip: bool = True
    ) -> int:
        """
        Index image captions (legacy method, redirects to index_captions_biomedclip).
        
        Args:
            cases: List of case dictionaries with 'images' field
            collection_name: Target collection name
            use_biomedclip: If True, use BiomedCLIP (recommended)
        
        Returns:
            Number of indexed points
        """
        if use_biomedclip:
            return self.index_captions_biomedclip(cases, collection_name)
        else:
            # Fallback to E5 (NOT recommended - causes dimension mismatch)
            print("WARNING: Using E5 for captions. Collection name suggests BiomedCLIP!")
            return self._index_captions_e5(cases, collection_name)
    
    def _index_captions_e5(
        self,
        cases: List[Dict],
        collection_name: str
    ) -> int:
        """Legacy E5-based caption indexing (NOT recommended)."""
        encoder = get_e5_encoder()
        
        captions = []
        for case in cases:
            for img in case.get("images", []):
                caption = img.get("caption", "")
                file_name = img.get("file") or img.get("file_name", "")
                if caption:
                    captions.append({
                        "case_id": case["case_id"],
                        "image_id": file_name,
                        "caption": caption
                    })
        
        if not captions:
            return 0
        
        self.create_collection(collection_name, encoder.dimension)
        
        texts = [c["caption"] for c in captions]
        embeddings = encoder.encode_document(texts)
        
        points = []
        for i, cap in enumerate(captions):
            point_id = f"{cap['case_id']}_{cap['image_id']}"
            points.append(PointStruct(
                id=stable_hash_id(point_id),
                vector=embeddings[i].tolist(),
                payload={
                    "case_id": cap["case_id"],
                    "image_id": cap["image_id"],
                    "caption": cap["caption"]
                }
            ))
        
        self.client.upsert(collection_name=collection_name, points=points)
        print(f"Indexed {len(points)} captions to '{collection_name}' (E5)")
        return len(points)
    
    def get_collection_info(self, collection_name: str) -> Dict:
        """Get collection info with proper attribute handling for different qdrant versions."""
        info = self.client.get_collection(collection_name)
        
        # Handle different qdrant_client versions - some use vectors_count, some don't
        vectors_count = getattr(info, 'vectors_count', None)
        if vectors_count is None:
            vectors_count = getattr(info, 'points_count', 0)  # Fallback
        
        # Get vector dimension from config
        try:
            vector_config = info.config.params.vectors
            if hasattr(vector_config, 'size'):
                dim = vector_config.size
            elif isinstance(vector_config, dict):
                dim = vector_config.get('size', 0)
            else:
                dim = 0
        except AttributeError:
            dim = 0
        
        return {
            "name": collection_name,
            "points_count": info.points_count,
            "vectors_count": vectors_count,
            "dimension": dim,
            "status": str(info.status) if hasattr(info, 'status') else "unknown"
        }


def index_train_data(
    strategy: str = "fixed",
    max_tokens: int = 400,
    overlap_tokens: int = 50,
    index_images: bool = True
) -> Dict:
    """
    Full indexing pipeline for train data.
    
    Args:
        strategy: Chunking strategy
        max_tokens: Max tokens per chunk
        overlap_tokens: Overlap tokens
        index_images: Whether to index actual images (requires BiomedCLIP)
    
    Returns:
        Dict with indexing stats
    """
    from .config import DATASET_VERSION, DATA_ROOT
    
    # Versioned collection names per GPT 5.2 checkpoint #2
    E5_COLLECTION = f"cases_text_e5_1024_{DATASET_VERSION}"
    CAPTIONS_COLLECTION = f"captions_biomedclip_512_{DATASET_VERSION}"
    IMAGES_COLLECTION = f"images_biomedclip_512_{DATASET_VERSION}"
    
    # Load train cases
    print("=" * 60)
    print(f"RAG Indexing Pipeline - Dataset Version: {DATASET_VERSION}")
    print("=" * 60)
    print(f"\nLoading train cases from {TRAIN_JSONL}...")
    with open(TRAIN_JSONL) as f:
        cases = [json.loads(line) for line in f]
    print(f"Loaded {len(cases)} train cases")
    
    # Chunk cases
    print(f"\nChunking with strategy='{strategy}'...")
    chunks = chunk_cases(
        cases,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens
    )
    print(f"Created {len(chunks)} chunks")
    
    # Index to Qdrant
    indexer = QdrantIndexer()
    
    # Lane 1: E5 text chunks (versioned)
    print(f"\nIndexing text chunks with E5 (1024d) to '{E5_COLLECTION}'...")
    n_chunks = indexer.index_chunks_e5(chunks, E5_COLLECTION)
    
    # Lane 2: BiomedCLIP captions (versioned)
    print(f"\nIndexing captions with BiomedCLIP (512d) to '{CAPTIONS_COLLECTION}'...")
    n_captions = indexer.index_captions_biomedclip(cases, CAPTIONS_COLLECTION)
    
    # Lane 2: BiomedCLIP images (optional, requires GPU)
    n_images = 0
    if index_images:
        print(f"\nIndexing images with BiomedCLIP (512d) to '{IMAGES_COLLECTION}'...")
        n_images = indexer.index_images_biomedclip(cases, IMAGES_COLLECTION)
    
    # Also fit BM25
    print("\nFitting BM25 index...")
    bm25 = get_bm25_retriever()
    bm25.fit_from_cases(cases)
    
    # Save BM25 index (versioned)
    bm25_path = DATA_ROOT / f"bm25_index_{DATASET_VERSION}.json"
    bm25.save(bm25_path)
    print(f"Saved BM25 index to {bm25_path}")
    
    return {
        "dataset_version": DATASET_VERSION,
        "train_cases": len(cases),
        "chunks": n_chunks,
        "captions": n_captions,
        "images": n_images,
        "strategy": strategy,
        "collections": {
            "e5": E5_COLLECTION,
            "captions": CAPTIONS_COLLECTION,
            "images": IMAGES_COLLECTION
        }
    }



if __name__ == "__main__":
    stats = index_train_data(strategy="fixed", index_images=False)
    print("\n" + "="*50)
    print("Indexing Complete!")
    for k, v in stats.items():
        print(f"  {k}: {v}")
