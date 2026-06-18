"""
BM25 Sparse Retriever for Baseline Comparison

Uses rank_bm25 for sparse keyword-based retrieval.
Serves as baseline per CliniqIR paper methodology.
"""
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np
from collections import defaultdict

try:
    from rank_bm25 import BM25Okapi
except ImportError:
    raise ImportError("Please install rank_bm25: pip install rank_bm25")


class BM25Retriever:
    """
    BM25 sparse retriever for baseline comparison.
    
    Per CliniqIR paper: BM25 as baseline to compare against dense retrievers.
    """
    
    def __init__(self):
        """Initialize empty BM25 retriever."""
        self.bm25 = None
        self.doc_ids = []
        self.tokenized_docs = []
    
    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization with lowercasing."""
        # Remove punctuation and lowercase
        import re
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        return text.split()
    
    def fit(self, documents: List[Dict[str, str]]) -> None:
        """
        Fit BM25 on documents.
        
        Args:
            documents: List of dicts with 'id' and 'text' keys
        """
        self.doc_ids = [doc["id"] for doc in documents]
        self.tokenized_docs = [self._tokenize(doc["text"]) for doc in documents]
        
        self.bm25 = BM25Okapi(self.tokenized_docs)
    
    def fit_from_cases(
        self,
        cases: List[Dict],
        text_field: str = "case_text",
        id_field: str = "case_id"
    ) -> None:
        """
        Fit BM25 from case data.
        
        Args:
            cases: List of case dictionaries
            text_field: Field containing text
            id_field: Field containing ID
        """
        documents = [
            {"id": case[id_field], "text": case.get(text_field, "")}
            for case in cases
            if case.get(text_field)
        ]
        self.fit(documents)
    
    def retrieve(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Retrieve top-k documents for query.
        
        Args:
            query: Query text
            top_k: Number of results to return
        
        Returns:
            List of (doc_id, score) tuples
        """
        if self.bm25 is None:
            raise ValueError("BM25 not fitted. Call fit() first.")
        
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.doc_ids[idx], float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0  # Filter zero-score results
        ]
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        top_k: int = 10
    ) -> List[List[Tuple[str, float]]]:
        """
        Batch retrieve for multiple queries.
        
        Args:
            queries: List of query texts
            top_k: Number of results per query
        
        Returns:
            List of result lists
        """
        return [self.retrieve(q, top_k) for q in queries]
    
    def save(self, path: Path) -> None:
        """Save BM25 index to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "doc_ids": self.doc_ids,
            "tokenized_docs": self.tokenized_docs
        }
        
        with open(path, "w") as f:
            json.dump(data, f)
    
    def load(self, path: Path) -> None:
        """Load BM25 index from file."""
        with open(path) as f:
            data = json.load(f)
        
        self.doc_ids = data["doc_ids"]
        self.tokenized_docs = data["tokenized_docs"]
        self.bm25 = BM25Okapi(self.tokenized_docs)


# Singleton instance
_retriever = None


def get_bm25_retriever() -> BM25Retriever:
    """Get singleton BM25 retriever instance."""
    global _retriever
    if _retriever is None:
        _retriever = BM25Retriever()
    return _retriever


if __name__ == "__main__":
    # Test BM25
    documents = [
        {"id": "doc1", "text": "Visceral leishmaniasis causes fever and splenomegaly."},
        {"id": "doc2", "text": "Cutaneous leishmaniasis presents with skin ulcers."},
        {"id": "doc3", "text": "Treatment includes liposomal amphotericin B."},
    ]
    
    retriever = BM25Retriever()
    retriever.fit(documents)
    
    results = retriever.retrieve("fever and enlarged spleen", top_k=3)
    print("Query: 'fever and enlarged spleen'")
    for doc_id, score in results:
        print(f"  {doc_id}: {score:.4f}")
