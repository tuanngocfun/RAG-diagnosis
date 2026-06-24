"""
E5 Encoder for Lane 1 Text Retrieval

Uses intfloat/multilingual-e5-large-instruct as symmetric bi-encoder
(same model for query and document encoding)
"""
import torch
from pathlib import Path
from typing import List, Union
import numpy as np
from sentence_transformers import SentenceTransformer


class E5Encoder:
    """
    E5-large encoder for text embedding.
    
    Symmetric bi-encoder: same model encodes both queries and documents.
    Follows E5 format: "query: ..." for queries, "passage: ..." for documents.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = None,
        device: str = None,
        cache_folder: str = "/data4t/hf/sentence-transformers"
    ):
        """
        Initialize E5 encoder.
        
        Args:
            model_path: Path to local model or HF model name
            device: Device to use (auto-detected if None)
            cache_folder: Cache folder for sentence-transformers
        """
        import os
        os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", cache_folder)
        
        if model_path is None:
            model_path = "intfloat/multilingual-e5-large-instruct"
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model = SentenceTransformer(
            str(model_path), 
            device=device,
            cache_folder=cache_folder
        )
        self.dimension = 1024  # E5-large dimension
    
    def encode_query(
        self,
        query: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode query text(s).
        
        Args:
            query: Single query or list of queries
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Query embeddings as numpy array
        """
        if isinstance(query, str):
            query = [query]
        
        # Add E5 query prefix
        formatted = [f"query: {q}" for q in query]
        
        embeddings = self.model.encode(
            formatted,
            normalize_embeddings=normalize,
            show_progress_bar=len(formatted) > 10
        )
        
        return embeddings
    
    def encode_document(
        self,
        document: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode document text(s).
        
        Args:
            document: Single document or list of documents
            normalize: Whether to L2-normalize embeddings
        
        Returns:
            Document embeddings as numpy array
        """
        if isinstance(document, str):
            document = [document]
        
        # Add E5 passage prefix
        formatted = [f"passage: {d}" for d in document]
        
        embeddings = self.model.encode(
            formatted,
            normalize_embeddings=normalize,
            show_progress_bar=len(formatted) > 10
        )
        
        return embeddings
    
    def encode_batch(
        self,
        texts: List[str],
        is_query: bool = False,
        batch_size: int = 32,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Batch encode texts.
        
        Args:
            texts: List of texts to encode
            is_query: Whether texts are queries (vs documents)
            batch_size: Batch size for encoding
            normalize: Whether to L2-normalize
        
        Returns:
            Embeddings as numpy array
        """
        prefix = "query: " if is_query else "passage: "
        formatted = [f"{prefix}{t}" for t in texts]
        
        embeddings = self.model.encode(
            formatted,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=True
        )
        
        return embeddings


# Singleton instance
_encoder = None


def get_e5_encoder(model_path: Union[str, Path] = None) -> E5Encoder:
    """Get singleton E5 encoder instance."""
    global _encoder
    if _encoder is None:
        _encoder = E5Encoder(model_path)
    return _encoder


if __name__ == "__main__":
    # Test encoder
    encoder = E5Encoder()
    
    query = "What are the symptoms of visceral leishmaniasis?"
    doc = "Visceral leishmaniasis presents with fever, weight loss, and splenomegaly."
    
    q_emb = encoder.encode_query(query)
    d_emb = encoder.encode_document(doc)
    
    # Compute similarity
    similarity = np.dot(q_emb[0], d_emb[0])
    
    print(f"Query embedding shape: {q_emb.shape}")
    print(f"Document embedding shape: {d_emb.shape}")
    print(f"Similarity: {similarity:.4f}")
