"""
Generators Package

Separates different answer generators for cleaner code:
- gemini.py: API-based Gemini generation
- medgemma.py: Local HuggingFace MedGemma
- gemma3.py: Local Gemma 3 12B/27B (NEW: for resource-limited eval)
- qwen_vl.py: Local Qwen2.5-VL 7B/72B (NEW: best multimodal)
"""
from .gemini import GeminiGenerator
from .medgemma import MedGemmaGenerator
from .gemma3 import Gemma3Generator
from .qwen_vl import QwenVLGenerator

__all__ = ["GeminiGenerator", "MedGemmaGenerator", "Gemma3Generator", "QwenVLGenerator"]


