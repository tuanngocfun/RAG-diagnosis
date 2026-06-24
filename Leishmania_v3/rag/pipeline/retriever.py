"""
Retriever for RAG Pipeline

Implements Lane 1 (Text) + Lane 2 (Vision-Language) + RRF Fusion

Per GPT 5.2 Feedback:
- Lane 2 uses TRUE BiomedCLIP (512d), not E5 fallback
- Cross-modal retrieval: text query → image/caption results
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

from qdrant_client import QdrantClient
from qdrant_client.http import models

from .config import QDRANT_API_KEY, QDRANT_URL, DATA_ROOT, DATASET_VERSION


class Lane1Retriever:
    """
    Lane 1: Text retrieval using E5 (1024d) + BM25 with optional RRF fusion.
    """
    
    def __init__(self, collection_name: str = None):
        """Initialize retriever."""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Use versioned collection name by default (GPT 5.2 checkpoint #2)
        self.collection_name = collection_name or f"cases_text_e5_1024_{DATASET_VERSION}"
        self._encoder = None
        self._bm25 = None
    
    @property
    def encoder(self):
        """Lazy load E5 encoder."""
        if self._encoder is None:
            from .encoders.e5 import get_e5_encoder
            self._encoder = get_e5_encoder()
        return self._encoder
    
    @property
    def bm25(self):
        """Lazy load BM25 retriever."""
        if self._bm25 is None:
            from .encoders.bm25 import get_bm25_retriever
            self._bm25 = get_bm25_retriever()
            # Load versioned BM25 index (GPT 5.2 checkpoint #2)
            bm25_path = DATA_ROOT / f"bm25_index_{DATASET_VERSION}.json"
            if bm25_path.exists():
                self._bm25.load(bm25_path)
            else:
                # Fallback to legacy path
                legacy_path = DATA_ROOT / "bm25_index.json"
                if legacy_path.exists():
                    print(f"Warning: Using legacy BM25 index from {legacy_path}")
                    self._bm25.load(legacy_path)
        return self._bm25
    
    def retrieve_e5(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using E5 dense vectors (1024d).
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score) tuples
        """
        # Encode query
        query_emb = self.encoder.encode_query(query)[0]
        
        # Search Qdrant using query_points API
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_emb.tolist(),
            limit=top_k * 2,  # Get more to deduplicate case_ids
        )
        results = response.points
        
        # Aggregate by case_id (best chunk per case)
        case_scores = {}
        for hit in results:
            case_id = hit.payload.get("case_id")
            if case_id not in case_scores or hit.score > case_scores[case_id]:
                case_scores[case_id] = hit.score
        
        # Sort and return top_k
        sorted_cases = sorted(case_scores.items(), key=lambda x: -x[1])[:top_k]
        return sorted_cases
    
    def retrieve_bm25(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve using BM25 sparse matching.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score) tuples
        """
        return self.bm25.retrieve(query, top_k)
    
    def retrieve_hybrid(
        self,
        query: str,
        top_k: int = 10,
        rrf_k: int = 60,
        e5_weight: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Hybrid retrieval with RRF fusion of E5 + BM25.
        
        Args:
            query: Query text
            top_k: Number of results
            rrf_k: RRF constant
            e5_weight: Weight for E5 (0-1), BM25 gets 1-e5_weight
        
        Returns:
            List of (case_id, score) tuples
        """
        # Get results from both
        e5_results = self.retrieve_e5(query, top_k * 2)
        bm25_results = self.retrieve_bm25(query, top_k * 2)
        
        # RRF fusion
        scores = defaultdict(float)
        
        for rank, (case_id, _) in enumerate(e5_results):
            scores[case_id] += e5_weight / (rrf_k + rank + 1)
        
        for rank, (case_id, _) in enumerate(bm25_results):
            scores[case_id] += (1 - e5_weight) / (rrf_k + rank + 1)
        
        # Sort and return
        sorted_cases = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return sorted_cases


class Lane2Retriever:
    """
    Lane 2: Vision-Language retrieval using BiomedCLIP (512d).
    
    TRUE multimodal: text query → BiomedCLIP embedding → search caption/image collections.
    Both captions and images are in the same 512d embedding space.
    """
    
    def __init__(
        self,
        caption_collection: str = None,
        image_collection: str = None
    ):
        """Initialize retriever with versioned collection names."""
        self.client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Use versioned collection names by default (GPT 5.2 checkpoint #2)
        self.caption_collection = caption_collection or f"captions_biomedclip_512_{DATASET_VERSION}"
        self.image_collection = image_collection or f"images_biomedclip_512_{DATASET_VERSION}"
        self._encoder = None
    
    @property
    def encoder(self):
        """Lazy load BiomedCLIP encoder (512d)."""
        if self._encoder is None:
            from .encoders.biomedclip import get_biomedclip_encoder
            self._encoder = get_biomedclip_encoder()
        return self._encoder
    
    def retrieve_by_caption(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve via caption similarity using BiomedCLIP text encoding.
        
        Query is encoded with BiomedCLIP text encoder and searched against
        captions that were also encoded with BiomedCLIP.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score, metadata) tuples
        """
        # Encode query with BiomedCLIP text encoder
        query_emb = self.encoder.encode_text(query)[0]
        
        try:
            response = self.client.query_points(
                collection_name=self.caption_collection,
                query=query_emb.tolist(),
                limit=top_k,
            )
            results = response.points
            
            return [
                (hit.payload.get("case_id"), hit.score, hit.payload)
                for hit in results
            ]
        except Exception as e:
            print(f"Caption retrieval failed: {e}")
            return []
    
    def retrieve_by_image(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve images via text→image cross-modal search.
        
        Query is encoded with BiomedCLIP text encoder and searched against
        images that were encoded with BiomedCLIP image encoder.
        
        This is TRUE cross-modal retrieval in aligned embedding space.
        
        Args:
            query: Query text
            top_k: Number of results
        
        Returns:
            List of (case_id, score, metadata) tuples with image paths
        """
        # Encode query with BiomedCLIP text encoder
        query_emb = self.encoder.encode_text(query)[0]
        
        try:
            response = self.client.query_points(
                collection_name=self.image_collection,
                query=query_emb.tolist(),
                limit=top_k,
            )
            results = response.points
            
            return [
                (hit.payload.get("case_id"), hit.score, hit.payload)
                for hit in results
            ]
        except Exception as e:
            print(f"Image retrieval failed: {e}")
            return []
    
    def retrieve_by_image_path(
        self,
        image_path: str,
        top_k: int = 10,
        search_captions: bool = True
    ) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve by IMAGE query (Q3 image-only evaluation).
        
        Unlike retrieve_by_image() which uses text→image, this method
        encodes an actual image file and searches for similar images/captions.
        
        This enables TRUE image-query multimodal retrieval per MMed-RAG standard.
        
        Args:
            image_path: Path to query image file
            top_k: Number of results
            search_captions: If True, search captions; if False, search images
        
        Returns:
            List of (case_id, score, metadata) tuples
        """
        from pathlib import Path
        
        # Validate image exists
        if not Path(image_path).exists():
            print(f"Image not found: {image_path}")
            return []
        
        # Encode image with BiomedCLIP image encoder
        image_emb = self.encoder.encode_image(image_path)[0]
        
        # Search collection (captions or images)
        collection = self.caption_collection if search_captions else self.image_collection
        
        try:
            response = self.client.query_points(
                collection_name=collection,
                query=image_emb.tolist(),
                limit=top_k,
            )
            results = response.points
            
            return [
                (hit.payload.get("case_id"), hit.score, hit.payload)
                for hit in results
            ]
        except Exception as e:
            print(f"Image-query retrieval failed: {e}")
            return []
    
    def retrieve_multimodal(
        self,
        query: str,
        top_k: int = 10,
        caption_weight: float = 0.5
    ) -> List[Tuple[str, float, Dict]]:
        """
        Combined caption + image retrieval with RRF fusion.
        
        Args:
            query: Query text
            top_k: Number of results
            caption_weight: Weight for caption results (0-1)
        
        Returns:
            Fused results as (case_id, score, metadata) tuples
        """
        caption_results = self.retrieve_by_caption(query, top_k * 2)
        image_results = self.retrieve_by_image(query, top_k * 2)
        
        # RRF fusion by case_id
        scores = defaultdict(float)
        metadata = {}
        
        for rank, (case_id, _, payload) in enumerate(caption_results):
            scores[case_id] += caption_weight / (60 + rank + 1)
            metadata[case_id] = payload
        
        for rank, (case_id, _, payload) in enumerate(image_results):
            scores[case_id] += (1 - caption_weight) / (60 + rank + 1)
            if case_id not in metadata:
                metadata[case_id] = payload
            else:
                # Merge image info
                metadata[case_id]["image_path"] = payload.get("path")
        
        # Sort and return
        sorted_cases = sorted(scores.items(), key=lambda x: -x[1])[:top_k]
        return [(case_id, score, metadata.get(case_id, {})) for case_id, score in sorted_cases]


def rrf_fusion(
    lane1_results: List[Tuple[str, float]],
    lane2_results: List[Tuple[str, float]],
    k: int = 60,
    lane1_weight: float = 0.6
) -> List[Tuple[str, float]]:
    """
    Reciprocal Rank Fusion of Lane 1 and Lane 2 results.
    
    Args:
        lane1_results: Results from Lane 1 (text)
        lane2_results: Results from Lane 2 (vision-language)
        k: RRF constant
        lane1_weight: Weight for Lane 1
    
    Returns:
        Fused results as (case_id, score) tuples
    """
    scores = defaultdict(float)
    
    for rank, (case_id, _) in enumerate(lane1_results):
        scores[case_id] += lane1_weight / (k + rank + 1)
    
    for rank, item in enumerate(lane2_results):
        case_id = item[0] if isinstance(item, tuple) else item
        scores[case_id] += (1 - lane1_weight) / (k + rank + 1)
    
    return sorted(scores.items(), key=lambda x: -x[1])


def two_lane_retrieval(
    query: str,
    top_k: int = 10,
    lane1_weight: float = 0.6
) -> List[Tuple[str, float]]:
    """
    Full two-lane retrieval with RRF fusion.
    
    Lane 1: E5 + BM25 hybrid (text)
    Lane 2: BiomedCLIP (vision-language)
    
    Args:
        query: Query text
        top_k: Number of results
        lane1_weight: Weight for Lane 1 in fusion
    
    Returns:
        Fused results as (case_id, score) tuples
    """
    lane1 = Lane1Retriever()
    lane2 = Lane2Retriever()
    
    # Get results from both lanes
    lane1_results = lane1.retrieve_hybrid(query, top_k * 2)
    lane2_caption = lane2.retrieve_by_caption(query, top_k * 2)
    
    # Convert Lane 2 to (case_id, score) format
    lane2_results = [(r[0], r[1]) for r in lane2_caption]
    
    # Fuse
    fused = rrf_fusion(lane1_results, lane2_results, lane1_weight=lane1_weight)
    
    return fused[:top_k]


if __name__ == "__main__":
    # Test retriever
    print("Testing Lane 1 (E5 + BM25)...")
    retriever = Lane1Retriever()
    
    query = "fever and splenomegaly in endemic area"
    print(f"Query: {query}")
    
    print("\nE5 Results:")
    for case_id, score in retriever.retrieve_e5(query, top_k=5):
        print(f"  {case_id}: {score:.4f}")
    
    print("\nTesting Lane 2 (BiomedCLIP)...")
    lane2 = Lane2Retriever()
    
    print("\nCaption Results:")
    for case_id, score, payload in lane2.retrieve_by_caption(query, top_k=5):
        print(f"  {case_id}: {score:.4f}")
