#!/usr/bin/env python3
"""
Text-Only RAG Baseline (No Knowledge Graph)

This baseline uses only text embeddings for retrieval, without any KG enhancement.
Used to demonstrate the value of KG-enhanced retrieval in comparison.

Usage:
    python text_only_rag.py --data ../../data/leishmaniasis_multimodal/train.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple
import numpy as np

# Optional: sentence-transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("Note: sentence-transformers not installed. Using TF-IDF fallback.")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class TextOnlyRAG:
    """
    Baseline RAG system using only text embeddings.
    No Knowledge Graph enhancement.
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model_name = model_name
        self.documents = []
        self.embeddings = None
        self.case_ids = []
        
        if HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading SentenceTransformer model: {model_name}")
            self.model = SentenceTransformer(model_name)
            self.use_transformers = True
        else:
            print("Using TF-IDF vectorizer as fallback")
            self.vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
            self.use_transformers = False
    
    def index_documents(self, documents: List[Dict]):
        """
        Index documents for retrieval.
        Each document should have 'case_id' and 'case_text'.
        """
        self.documents = documents
        self.case_ids = [d['case_id'] for d in documents]
        
        texts = [d.get('case_text', '') or '' for d in documents]
        
        print(f"Indexing {len(texts)} documents...")
        
        if self.use_transformers:
            self.embeddings = self.model.encode(texts, show_progress_bar=True)
        else:
            self.embeddings = self.vectorizer.fit_transform(texts)
        
        print(f"✓ Indexed {len(self.documents)} documents")
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve top-k most similar documents for a query.
        
        Returns: [(case_id, similarity_score, document), ...]
        """
        if self.use_transformers:
            query_embedding = self.model.encode([query])
            similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        else:
            query_vec = self.vectorizer.transform([query])
            similarities = cosine_similarity(query_vec, self.embeddings)[0]
        
        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]
        
        results = []
        for idx in top_indices:
            results.append((
                self.case_ids[idx],
                float(similarities[idx]),
                self.documents[idx]
            ))
        
        return results
    
    def evaluate(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Evaluate retrieval performance.
        
        test_queries: List of dicts with 'query' and 'relevant_case_ids'
        """
        precisions = []
        recalls = []
        mrrs = []
        
        for q in test_queries:
            query = q['query']
            relevant = set(q.get('relevant_case_ids', []))
            
            results = self.retrieve(query, top_k=k)
            retrieved = [r[0] for r in results]
            
            # Precision@K
            hits = len(set(retrieved) & relevant)
            precision = hits / k if k > 0 else 0
            precisions.append(precision)
            
            # Recall@K
            recall = hits / len(relevant) if relevant else 0
            recalls.append(recall)
            
            # MRR
            for rank, case_id in enumerate(retrieved, 1):
                if case_id in relevant:
                    mrrs.append(1.0 / rank)
                    break
            else:
                mrrs.append(0)
        
        return {
            "precision@k": np.mean(precisions),
            "recall@k": np.mean(recalls),
            "mrr": np.mean(mrrs),
            "k": k,
            "num_queries": len(test_queries)
        }


def load_data(path: Path) -> List[Dict]:
    """Load JSONL data."""
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text-Only RAG Baseline")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--model", type=str, default="all-MiniLM-L6-v2")
    args = parser.parse_args()
    
    print("=" * 60)
    print("TEXT-ONLY RAG BASELINE (No Knowledge Graph)")
    print("=" * 60)
    
    # Load data
    data = load_data(Path(args.data))
    print(f"Loaded {len(data)} records")
    
    # Initialize and index
    rag = TextOnlyRAG(model_name=args.model)
    rag.index_documents(data)
    
    # Demo query
    print("\n📝 Demo retrieval:")
    query = "visceral leishmaniasis fever hepatosplenomegaly"
    results = rag.retrieve(query, top_k=3)
    
    for case_id, score, doc in results:
        print(f"  {case_id}: score={score:.3f}")
        print(f"    {doc.get('case_text', '')[:100]}...")
    
    print("\n✅ Text-Only RAG baseline ready!")
    print("   Use rag.retrieve(query) for retrieval")
    print("   Use rag.evaluate(test_queries) for evaluation")


if __name__ == "__main__":
    main()
