"""
Generators Package

Separates different answer generators for cleaner code:
- gemini.py: API-based Gemini generation
- medgemma.py: Local HuggingFace MedGemma
- gemma3.py: Local Gemma 3 12B/27B (NEW: for resource-limited eval)
- qwen_vl.py: Local Qwen2.5-VL 7B/72B (NEW: best multimodal)

Prompt modes are centralized in configs/prompt_mode.py:
- STRICT_CONTEXT: Force context over parametric knowledge (Gemini/Gemma3 default)
- BALANCED: Synergy with parametric knowledge (MedGemma default)
- NO_CONTEXT: No-RAG baseline
"""
from .gemini import GeminiGenerator
from .medgemma import MedGemmaGenerator
from .gemma3 import Gemma3Generator
from .qwen_vl import QwenVLGenerator

# Re-export prompt mode for convenience (using correct relative import path)
from ...configs.prompt_mode import PromptMode, build_rag_prompt

__all__ = [
    "GeminiGenerator",
    "MedGemmaGenerator",
    "Gemma3Generator",
    "QwenVLGenerator",
    "PromptMode",
    "build_rag_prompt",
]


