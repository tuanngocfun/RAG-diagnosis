"""
Generators Package

Separates different answer generators for cleaner code:
- gemini.py: API-based Gemini generation
- medgemma.py: Local HuggingFace MedGemma
"""
from .gemini import GeminiGenerator
from .medgemma import MedGemmaGenerator

__all__ = ["GeminiGenerator", "MedGemmaGenerator"]
