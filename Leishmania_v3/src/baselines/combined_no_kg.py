#!/usr/bin/env python3
"""
Combined Multimodal Baseline (No Knowledge Graph)

Combines text and image embeddings for multimodal retrieval,
but WITHOUT any Knowledge Graph enhancement.

Usage:
    python combined_no_kg.py --data ../../data/leishmaniasis_multimodal/train.jsonl
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

from text_only_rag import TextOnlyRAG
from image_only import ImageOnlyBaseline


class CombinedNoKGBaseline:
    """
    Multimodal baseline combining text + image without KG.
    """
    
    def __init__(self, 
                 text_weight: float = 0.7,
                 image_weight: float = 0.3,
                 text_model: str = "all-MiniLM-L6-v2"):
        self.text_weight = text_weight
        self.image_weight = image_weight
        
        self.text_rag = TextOnlyRAG(model_name=text_model)
        self.image_baseline = ImageOnlyBaseline()
        
        self.documents = []
        self.case_ids = []
    
    def index(self, 
              documents: List[Dict], 
              images_dir: Optional[Path] = None):
        """
        Index both text and images.
        
        documents: List of dicts with 'case_id', 'case_text', 'images'
        images_dir: Directory containing extracted images
        """
        self.documents = documents
        self.case_ids = [d['case_id'] for d in documents]
        
        # Index text
        print("\n📝 Indexing text...")
        self.text_rag.index_documents(documents)
        
        # Index images if available
        if images_dir and images_dir.exists():
            print("\n🖼️ Indexing images...")
            self.image_baseline.index_images(images_dir, documents)
    
    def retrieve(self, 
                 query_text: str, 
                 query_image: Optional[Path] = None,
                 top_k: int = 5) -> List[Tuple[str, float, Dict]]:
        """
        Retrieve using combined text + image similarity.
        
        If query_image is provided, combines both modalities.
        Otherwise, uses only text.
        """
        # Get text scores
        text_results = self.text_rag.retrieve(query_text, top_k=len(self.case_ids))
        text_scores = {r[0]: r[1] for r in text_results}
        
        # Normalize text scores
        max_text = max(text_scores.values()) if text_scores else 1
        text_scores = {k: v / max_text for k, v in text_scores.items()}
        
        # Get image scores if query image provided
        if query_image and query_image.exists():
            image_results = self.image_baseline.retrieve_by_image(query_image, top_k=len(self.case_ids))
            image_scores = {r[0]: r[1] for r in image_results}
            
            # Normalize image scores
            max_img = max(image_scores.values()) if image_scores else 1
            image_scores = {k: v / max_img for k, v in image_scores.items()}
        else:
            image_scores = {}
        
        # Combine scores
        combined_scores = {}
        for case_id in self.case_ids:
            text_score = text_scores.get(case_id, 0)
            image_score = image_scores.get(case_id, 0)
            
            if image_scores:
                combined = (self.text_weight * text_score + 
                           self.image_weight * image_score)
            else:
                combined = text_score
            
            combined_scores[case_id] = combined
        
        # Sort and return top-k
        sorted_cases = sorted(combined_scores.items(), key=lambda x: -x[1])[:top_k]
        
        # Build results with full documents
        doc_lookup = {d['case_id']: d for d in self.documents}
        results = []
        for case_id, score in sorted_cases:
            results.append((case_id, score, doc_lookup.get(case_id, {})))
        
        return results
    
    def evaluate(self, test_queries: List[Dict], k: int = 5) -> Dict:
        """
        Evaluate combined retrieval.
        
        test_queries: List with 'query_text', 'query_image' (optional), 'relevant_case_ids'
        """
        precisions = []
        recalls = []
        mrrs = []
        
        for q in test_queries:
            query_text = q.get('query_text', q.get('query', ''))
            query_image = Path(q['query_image']) if q.get('query_image') else None
            relevant = set(q.get('relevant_case_ids', []))
            
            results = self.retrieve(query_text, query_image, top_k=k)
            retrieved = [r[0] for r in results]
            
            hits = len(set(retrieved) & relevant)
            precisions.append(hits / k if k > 0 else 0)
            recalls.append(hits / len(relevant) if relevant else 0)
            
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
            "num_queries": len(test_queries),
            "text_weight": self.text_weight,
            "image_weight": self.image_weight
        }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Combined Multimodal Baseline (No KG)")
    parser.add_argument("--data", type=str, required=True, help="Path to JSONL data")
    parser.add_argument("--images", type=str, help="Path to images directory")
    parser.add_argument("--text-weight", type=float, default=0.7)
    parser.add_argument("--image-weight", type=float, default=0.3)
    args = parser.parse_args()
    
    print("=" * 60)
    print("COMBINED MULTIMODAL BASELINE (No Knowledge Graph)")
    print("=" * 60)
    
    # Load data
    with open(args.data) as f:
        data = [json.loads(line) for line in f]
    print(f"Loaded {len(data)} records")
    
    # Initialize
    baseline = CombinedNoKGBaseline(
        text_weight=args.text_weight,
        image_weight=args.image_weight
    )
    
    images_dir = Path(args.images) if args.images else None
    baseline.index(data, images_dir)
    
    # Demo
    print("\n📝 Demo retrieval:")
    query = "visceral leishmaniasis fever hepatosplenomegaly amphotericin"
    results = baseline.retrieve(query, top_k=3)
    
    for case_id, score, doc in results:
        print(f"  {case_id}: score={score:.3f}")
    
    print("\n✅ Combined (No KG) baseline ready!")


if __name__ == "__main__":
    main()
