"""
MedCPT Cross-Encoder Reranker

Uses ncbi/MedCPT-Cross-Encoder for reranking retrieved results.
Cross-encoder provides more accurate relevance scores than bi-encoder.
"""
import torch
from pathlib import Path
from typing import List, Dict, Tuple, Union
import numpy as np

from transformers import AutoTokenizer, AutoModelForSequenceClassification


class MedCPTReranker:
    """
    MedCPT cross-encoder for reranking.
    
    Takes query-document pairs and produces relevance scores.
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "ncbi/MedCPT-Cross-Encoder",
        device: str = None
    ):
        """
        Initialize MedCPT cross-encoder.
        
        Args:
            model_path: Path to model or HF model name
            device: Device (auto-detected if None)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(model_path))
        self.model = self.model.to(device)
        self.model.eval()
    
    def score(
        self,
        query: str,
        documents: List[str],
        batch_size: int = 16
    ) -> List[float]:
        """
        Score query-document pairs.
        
        Args:
            query: Query text
            documents: List of document texts
            batch_size: Batch size for scoring
        
        Returns:
            List of relevance scores
        """
        scores = []
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            
            # Create pairs
            pairs = [(query, doc) for doc in batch_docs]
            
            # Tokenize
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)
            
            # Score
            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
            
            if batch_scores.ndim == 0:
                batch_scores = [float(batch_scores)]
            else:
                batch_scores = batch_scores.tolist()
            
            scores.extend(batch_scores)
        
        return scores
    
    def rerank(
        self,
        query: str,
        candidates: List[Tuple[str, str, float]],
        top_k: int = None
    ) -> List[Tuple[str, str, float]]:
        """
        Rerank candidates using cross-encoder.
        
        Args:
            query: Query text
            candidates: List of (doc_id, doc_text, original_score) tuples
            top_k: Return top K after reranking (None = all)
        
        Returns:
            Reranked list of (doc_id, doc_text, new_score) tuples
        """
        if not candidates:
            return []
        
        doc_ids = [c[0] for c in candidates]
        doc_texts = [c[1] for c in candidates]
        
        # Score
        scores = self.score(query, doc_texts)
        
        # Combine with IDs
        results = list(zip(doc_ids, doc_texts, scores))
        
        # Sort by score
        results.sort(key=lambda x: -x[2])
        
        if top_k:
            results = results[:top_k]
        
        return results


# Singleton
_reranker = None


def get_medcpt_reranker() -> MedCPTReranker:
    """Get singleton MedCPT reranker."""
    global _reranker
    if _reranker is None:
        _reranker = MedCPTReranker()
    return _reranker


if __name__ == "__main__":
    print("Testing MedCPT Reranker...")
    
    reranker = MedCPTReranker()
    print("✓ Loaded MedCPT Cross-Encoder")
    
    query = "What are the symptoms of visceral leishmaniasis?"
    docs = [
        "Visceral leishmaniasis presents with fever, weight loss, and splenomegaly.",
        "Cutaneous leishmaniasis causes skin ulcers.",
        "The patient had a headache and cough."
    ]
    
    scores = reranker.score(query, docs)
    print("\nScores:")
    for doc, score in zip(docs, scores):
        print(f"  {score:.4f}: {doc[:50]}...")
