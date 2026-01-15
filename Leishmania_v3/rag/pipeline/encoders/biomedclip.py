"""
BiomedCLIP Encoder for Lane 2 Vision-Language Retrieval

Uses microsoft/BiomedCLIP for aligned text-image embeddings (512d)
Enables true cross-modal retrieval: text → images
"""
import torch
from pathlib import Path
from typing import List, Union, Optional
import numpy as np

try:
    from open_clip import create_model_and_transforms, get_tokenizer
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False


class BiomedCLIPEncoder:
    """
    BiomedCLIP encoder for vision-language embeddings.
    
    Projects both text and images into aligned 512d space.
    """
    
    MODEL_NAME = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    
    def __init__(self, device: str = None):
        """
        Initialize BiomedCLIP encoder.
        
        Args:
            device: Device to use (auto-detected if None)
        """
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError(
                "Please install open_clip: pip install open_clip_torch"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        
        # Load model
        self.model, self.preprocess_train, self.preprocess_val = create_model_and_transforms(
            self.MODEL_NAME
        )
        self.model = self.model.to(device)
        self.model.eval()
        
        self.tokenizer = get_tokenizer(self.MODEL_NAME)
        self.dimension = 512
    
    def encode_text(
        self,
        texts: Union[str, List[str]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode text(s) into BiomedCLIP embedding space.
        
        Args:
            texts: Single text or list of texts
            normalize: Whether to L2-normalize
        
        Returns:
            Text embeddings (N, 512)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        tokens = self.tokenizer(texts).to(self.device)
        
        with torch.no_grad():
            text_features = self.model.encode_text(tokens)
            
            if normalize:
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        return text_features.cpu().numpy()
    
    def encode_image(
        self,
        images: Union[str, Path, List[Union[str, Path]]],
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode image(s) into BiomedCLIP embedding space.
        
        Args:
            images: Image path(s) or PIL Image(s)
            normalize: Whether to L2-normalize
        
        Returns:
            Image embeddings (N, 512)
        """
        from PIL import Image
        
        if isinstance(images, (str, Path)):
            images = [images]
        
        # Load and preprocess
        processed = []
        for img_path in images:
            if isinstance(img_path, (str, Path)):
                img = Image.open(img_path).convert("RGB")
            else:
                img = img_path  # Assume PIL Image
            processed.append(self.preprocess_val(img))
        
        batch = torch.stack(processed).to(self.device)
        
        with torch.no_grad():
            image_features = self.model.encode_image(batch)
            
            if normalize:
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        
        return image_features.cpu().numpy()
    
    def encode_batch(
        self,
        items: List,
        mode: str = "text",
        batch_size: int = 32
    ) -> np.ndarray:
        """
        Batch encode items.
        
        Args:
            items: List of texts or image paths
            mode: 'text' or 'image'
            batch_size: Batch size
        
        Returns:
            Embeddings as numpy array
        """
        embeddings = []
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            if mode == "text":
                emb = self.encode_text(batch)
            else:
                emb = self.encode_image(batch)
            
            embeddings.append(emb)
        
        return np.vstack(embeddings)


# Singleton
_encoder = None


def get_biomedclip_encoder() -> BiomedCLIPEncoder:
    """Get singleton BiomedCLIP encoder."""
    global _encoder
    if _encoder is None:
        _encoder = BiomedCLIPEncoder()
    return _encoder


if __name__ == "__main__":
    print("Testing BiomedCLIP encoder...")
    
    encoder = BiomedCLIPEncoder()
    print(f"✓ Loaded BiomedCLIP (dim={encoder.dimension})")
    
    # Test text
    text = "Skin ulcer with raised borders on the forearm"
    text_emb = encoder.encode_text(text)
    print(f"✓ Text embedding shape: {text_emb.shape}")
