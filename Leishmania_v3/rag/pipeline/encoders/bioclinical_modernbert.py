"""
BioClinical-ModernBERT Encoder for Medical Text Retrieval

A specialized text encoder trained on PubMed + MIMIC-IV clinical notes.
Features:
- 95.54% performance on PubMed tasks  
- 8,192 token context (vs 512 for E5)
- 768-dimensional embeddings

Per Perplexity Pro recommendation for medical domain RAG.
"""
import os
from pathlib import Path
from typing import List, Union

import numpy as np
import torch

# Set cache before imports
os.environ.setdefault("TRANSFORMERS_CACHE", "/data4t/hf/transformers")
os.environ.setdefault("HF_HOME", "/data4t/hf/transformers")


class BioClinicalModernBERTEncoder:
    """
    BioClinical-ModernBERT encoder for medical text embeddings.
    
    Replacement for E5-large-v2 in the Lane 1 text retrieval.
    """
    
    MODEL_NAME = "NeuML/bioclinical-modernbert-base-embeddings"
    CACHE_DIR = "/data4t/hf/transformers"
    EMBEDDING_DIM = 768
    MAX_LENGTH = 8192  # Much longer than E5's 512
    
    _instance = None
    
    def __init__(self, device: str = None):
        """Initialize encoder with specified device."""
        from sentence_transformers import SentenceTransformer
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        print(f"Loading BioClinical-ModernBERT on {device}...")
        self.model = SentenceTransformer(
            self.MODEL_NAME,
            cache_folder=self.CACHE_DIR,
            device=device
        )
        print(f"✓ BioClinical-ModernBERT loaded (dim={self.EMBEDDING_DIM})")
    
    @classmethod
    def get_instance(cls, device: str = None) -> "BioClinicalModernBERTEncoder":
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls(device=device)
        return cls._instance
    
    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 32,
        show_progress: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            normalize: L2 normalize embeddings (for cosine similarity)
        
        Returns:
            Numpy array of embeddings (N, 768)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=normalize,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_query(self, query: str) -> np.ndarray:
        """Encode a single query."""
        return self.encode(query, normalize=True)
    
    def encode_passages(
        self,
        passages: List[str],
        batch_size: int = 32,
        show_progress: bool = True
    ) -> np.ndarray:
        """Encode multiple passages for indexing."""
        return self.encode(
            passages,
            batch_size=batch_size,
            show_progress=show_progress,
            normalize=True
        )


def get_bioclinical_encoder(device: str = None) -> BioClinicalModernBERTEncoder:
    """Get singleton BioClinical-ModernBERT encoder instance."""
    return BioClinicalModernBERTEncoder.get_instance(device=device)


if __name__ == "__main__":
    # Test the encoder
    encoder = get_bioclinical_encoder()
    
    # Test medical text
    test_texts = [
        "Patient presented with fever, hepatosplenomegaly, and pancytopenia",
        "Cutaneous leishmaniasis presenting as nodular skin lesions",
        "Visceral leishmaniasis confirmed by bone marrow aspiration"
    ]
    
    embeddings = encoder.encode(test_texts)
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Embedding norm: {np.linalg.norm(embeddings[0]):.4f}")
    
    # Test similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(embeddings[:1], embeddings[1:])
    print(f"Similarity to CL: {sim[0, 0]:.4f}")
    print(f"Similarity to VL: {sim[0, 1]:.4f}")
