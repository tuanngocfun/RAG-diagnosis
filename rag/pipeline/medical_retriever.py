"""
Medical Retriever with BioClinical-ModernBERT + SPLADE

Specialized text retriever for medical domain using:
- Dense: NeuML/bioclinical-modernbert-base-embeddings (768d, 8192 tokens)
- Sparse: NeuML/pubmedbert-base-splade (learned sparse, term expansion)

Per Perplexity Pro recommendation for improved context relevance.
"""
import json
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import QDRANT_API_KEY, QDRANT_URL, DATA_ROOT, DATASET_VERSION, SPLIT_DIR


class MedicalRetriever:
    """
    Medical-specialized text retriever using BioClinical-ModernBERT + SPLADE.
    
    Replacement for Lane1Retriever (E5 + BM25) in medical domain.
    """
    
    # Collection names for new indexes
    COLLECTION_BIOCLINICAL = f"cases_text_bioclinical_768_{DATASET_VERSION}"
    
    def __init__(
        self,
        collection_name: str = None,
        use_splade: bool = True
    ):
        """
        Initialize medical retriever.
        
        Args:
            collection_name: Custom collection name (default: auto-versioned)
            use_splade: If True, use PubMedBERT-SPLADE; else fall back to BM25
        """
        self.client = QdrantClient(
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
            check_compatibility=False,
        )
        self.collection_name = collection_name or self.COLLECTION_BIOCLINICAL
        self.use_splade = use_splade
        
        self._encoder = None
        self._splade = None
        self._bm25 = None
    
    @property
    def encoder(self):
        """Lazy load BioClinical-ModernBERT encoder."""
        if self._encoder is None:
            from .encoders.bioclinical_modernbert import get_bioclinical_encoder
            self._encoder = get_bioclinical_encoder()
        return self._encoder
    
    @property
    def sparse_encoder(self):
        """Lazy load sparse encoder (SPLADE or BM25)."""
        if self.use_splade:
            if self._splade is None:
                try:
                    from txtai.vectors import SparseVectors
                    self._splade = SparseVectors(
                        "NeuML/pubmedbert-base-splade",
                        path="/mnt/data/hf/transformers"
                    )
                    print("✓ PubMedBERT-SPLADE loaded")
                except Exception as e:
                    print(f"SPLADE load failed ({e}), falling back to BM25")
                    self.use_splade = False
                    return self.sparse_encoder  # Recursive call to get BM25
            return self._splade
        else:
            if self._bm25 is None:
                from .encoders.bm25 import get_bm25_retriever
                self._bm25 = get_bm25_retriever()
                # Load versioned BM25 index
                bm25_path = DATA_ROOT / f"bm25_index_{DATASET_VERSION}.json"
                if bm25_path.exists():
                    self._bm25.load(bm25_path)
                else:
                    legacy_path = DATA_ROOT / "bm25_index.json"
                    if legacy_path.exists():
                        self._bm25.load(legacy_path)
            return self._bm25
    
    def retrieve_dense(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using BioClinical-ModernBERT dense vectors (768d).
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score) tuples
        """
        # Encode query
        query_emb = self.encoder.encode_query(query)
        if len(query_emb.shape) == 2:
            query_emb = query_emb[0]
        
        # Search Qdrant
        try:
            response = self.client.query_points(
                collection_name=self.collection_name,
                query=query_emb.tolist(),
                limit=top_k * 2,  # Get more to deduplicate case_ids
            )
            results = response.points
        except Exception as e:
            print(f"Dense retrieval failed: {e}")
            return []
        
        # Aggregate by case_id (best chunk per case)
        case_scores = {}
        for hit in results:
            case_id = hit.payload.get("case_id")
            if case_id not in case_scores or hit.score > case_scores[case_id]:
                case_scores[case_id] = hit.score
        
        # Sort and return top_k
        sorted_cases = sorted(case_scores.items(), key=lambda x: -x[1])[:top_k]
        return sorted_cases
    
    def retrieve_sparse(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using sparse matching (SPLADE or BM25).
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score) tuples
        """
        if self.use_splade:
            # SPLADE retrieval via search against pre-indexed sparse vectors
            # For now, fall back to BM25 as SPLADE indexing is more complex
            # TODO: Implement full SPLADE index
            print("SPLADE full index not implemented, using BM25")
            self.use_splade = False
            return self.retrieve_sparse(query, top_k)
        else:
            return self.sparse_encoder.retrieve(query, top_k)
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        dense_weight: float = 0.6
    ) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval with RRF fusion of BioClinical + BM25.
        
        Args:
            query: Query text
            top_k: Number of results
            rrf_k: RRF constant
            dense_weight: Weight for dense (0-1), sparse gets 1-dense_weight
        
        Returns:
            List of (case_id, score) tuples
        """
        # Get results from both
        dense_results = self.retrieve_dense(query, top_k * 2)
        sparse_results = self.retrieve_sparse(query, top_k * 2)
        
        # RRF fusion
        scores = defaultdict(float)
        
        for rank, (case_id, _) in enumerate(dense_results):
            scores[case_id] += dense_weight / (rrf_k + rank + 1)
        
        for rank, (case_id, _) in enumerate(sparse_results):
            scores[case_id] += (1 - dense_weight) / (rrf_k + rank + 1)
        
        # Sort and return
        sorted_cases = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return sorted_cases


def index_with_bioclinical(
    train_jsonl: Path = None,
    collection_name: str = None,
    batch_size: int = 32
) -> int:
    """
    Index training corpus with BioClinical-ModernBERT.
    
    Args:
        train_jsonl: Path to training JSONL file
        collection_name: Target collection name
        batch_size: Batch size for encoding
    
    Returns:
        Number of points indexed
    """
    from qdrant_client.http.models import VectorParams, Distance, PointStruct
    import uuid
    
    # Defaults
    if train_jsonl is None:
        train_jsonl = SPLIT_DIR / "train.jsonl"
    if collection_name is None:
        collection_name = MedicalRetriever.COLLECTION_BIOCLINICAL
    
    # Load encoder
    from .encoders.bioclinical_modernbert import get_bioclinical_encoder
    encoder = get_bioclinical_encoder()
    
    # Load training cases
    print(f"Loading training cases from {train_jsonl}...")
    with open(train_jsonl) as f:
        cases = [json.loads(line) for line in f]
    print(f"Loaded {len(cases)} cases")
    
    # Initialize Qdrant client
    client = QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
        check_compatibility=False,
    )
    
    # Create or recreate collection
    print(f"Creating collection: {collection_name} (768d, cosine)")
    try:
        client.delete_collection(collection_name)
    except:
        pass
    
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=768,
            distance=Distance.COSINE
        )
    )
    
    # Index each case (full text, no chunking for now since ModernBERT supports 8192 tokens)
    points = []
    for i, case in enumerate(cases):
        case_id = case["case_id"]
        text = case.get("case_text", "")
        
        if not text:
            continue
        
        # Encode (ModernBERT can handle up to 8192 tokens)
        embedding = encoder.encode(text[:8000])  # Safe limit
        if len(embedding.shape) == 2:
            embedding = embedding[0]
        
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload={
                "case_id": case_id,
                "diagnosis": case.get("diagnosis", ""),
                "diagnosis_type": case.get("diagnosis_type", "")
            }
        )
        points.append(point)
        
        # Batch upsert
        if len(points) >= batch_size:
            client.upsert(collection_name=collection_name, points=points)
            print(f"  Indexed {i + 1}/{len(cases)}")
            points = []
    
    # Final batch
    if points:
        client.upsert(collection_name=collection_name, points=points)
    
    print(f"✓ Indexed {len(cases)} cases to {collection_name}")
    return len(cases)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Index corpus")
    parser.add_argument("--test", action="store_true", help="Test retrieval")
    args = parser.parse_args()
    
    if args.index:
        index_with_bioclinical()
    
    if args.test:
        retriever = MedicalRetriever()
        query = "fever hepatosplenomegaly pancytopenia endemic area"
        print(f"Query: {query}")
        
        print("\nBioClinical Dense Results:")
        for case_id, score in retriever.retrieve_dense(query, top_k=5):
            print(f"  {case_id}: {score:.4f}")
        
        print("\nHybrid Results:")
        for case_id, score in retriever.retrieve_hybrid(query, top_k=5):
            print(f"  {case_id}: {score:.4f}")
