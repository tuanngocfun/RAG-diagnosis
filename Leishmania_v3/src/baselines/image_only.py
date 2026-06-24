#!/usr/bin/env python3
"""
Image-Only Baseline (No Text, No Knowledge Graph)

This baseline uses only image embeddings for visual similarity search.
No clinical text or KG entities are used.

Usage:
    python image_only.py --images ../../data/leishmaniasis_multimodal/images/
"""

import json
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

# Optional: PIL for image loading
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Optional: torch for embeddings
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class ImageOnlyBaseline:
    """
    Baseline using only image visual features.
    No text or Knowledge Graph enhancement.
    """
    
    def __init__(self, model_name: str = "clip"):
        self.model_name = model_name
        self.image_embeddings = {}
        self.case_id_to_images = {}
        
        if model_name == "clip" and HAS_TORCH:
            try:
                import clip
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                self.use_clip = True
                print(f"Loaded CLIP model on {self.device}")
            except ImportError:
                print("CLIP not installed. Using random embeddings for demo.")
                self.use_clip = False
        else:
            self.use_clip = False
            print("Using random embeddings for demo (install torch + clip for real use)")
    
    def _get_image_embedding(self, image_path: Path) -> np.ndarray:
        """Get embedding for a single image."""
        if self.use_clip and HAS_PIL:
            image = Image.open(image_path).convert("RGB")
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.model.encode_image(image_input)
            
            return embedding.cpu().numpy().flatten()
        else:
            # Random embedding for demo
            return np.random.randn(512)
    
    def index_images(self, images_dir: Path, metadata: Optional[List[Dict]] = None):
        """
        Index all images for retrieval.
        
        images_dir: Directory containing case_id/image.webp structure
        metadata: Optional list of records with 'case_id' and 'images' fields
        """
        print(f"Indexing images from {images_dir}...")
        
        if metadata:
            # Use metadata to find images
            for record in metadata:
                case_id = record['case_id']
                case_images = record.get('images', [])
                
                if case_images:
                    self.case_id_to_images[case_id] = []
                    
                    for img_info in case_images:
                        img_path = images_dir / case_id / img_info['file']
                        if img_path.exists():
                            embedding = self._get_image_embedding(img_path)
                            self.image_embeddings[str(img_path)] = embedding
                            self.case_id_to_images[case_id].append(str(img_path))
        else:
            # Scan directory
            for case_dir in images_dir.iterdir():
                if case_dir.is_dir():
                    case_id = case_dir.name
                    self.case_id_to_images[case_id] = []
                    
                    for img_path in case_dir.glob("*.webp"):
                        embedding = self._get_image_embedding(img_path)
                        self.image_embeddings[str(img_path)] = embedding
                        self.case_id_to_images[case_id].append(str(img_path))
        
        print(f"✓ Indexed {len(self.image_embeddings)} images from {len(self.case_id_to_images)} cases")
    
    def retrieve_by_image(self, query_image_path: Path, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Retrieve similar cases based on image similarity.
        
        Returns: [(case_id, max_similarity_score), ...]
        """
        query_embedding = self._get_image_embedding(query_image_path)
        
        # Calculate similarity to all indexed images
        case_scores = {}
        
        for img_path, embedding in self.image_embeddings.items():
            similarity = np.dot(query_embedding, embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(embedding)
            )
            
            # Find which case this image belongs to
            for case_id, images in self.case_id_to_images.items():
                if img_path in images:
                    if case_id not in case_scores or similarity > case_scores[case_id]:
                        case_scores[case_id] = similarity
                    break
        
        # Sort by similarity
        sorted_cases = sorted(case_scores.items(), key=lambda x: -x[1])
        
        return sorted_cases[:top_k]
    
    def get_case_embedding(self, case_id: str) -> Optional[np.ndarray]:
        """Get average embedding for a case (average of all its images)."""
        images = self.case_id_to_images.get(case_id, [])
        if not images:
            return None
        
        embeddings = [self.image_embeddings[img] for img in images]
        return np.mean(embeddings, axis=0)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Image-Only Baseline")
    parser.add_argument("--images", type=str, required=True, help="Path to images directory")
    parser.add_argument("--data", type=str, help="Optional path to JSONL metadata")
    args = parser.parse_args()
    
    print("=" * 60)
    print("IMAGE-ONLY BASELINE (No Text, No Knowledge Graph)")
    print("=" * 60)
    
    baseline = ImageOnlyBaseline()
    
    metadata = None
    if args.data:
        with open(args.data) as f:
            metadata = [json.loads(line) for line in f]
    
    baseline.index_images(Path(args.images), metadata)
    
    print("\n✅ Image-Only baseline ready!")


if __name__ == "__main__":
    main()
