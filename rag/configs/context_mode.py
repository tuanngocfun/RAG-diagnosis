"""
Context Mode: Configuration for context selection strategies.

Per GPT 5.2 recommendation:
- Selection metric (cheap, online): retriever/reranker score
- Reporting metric (expensive, offline): RAGAS context_relevance

Context Modes:
- top_k:K       → Take top K contexts regardless of score
- quality_threshold:t → Take contexts with score >= t (min 3 guaranteed)
- dynamic_k    → Choose K that maximizes quality/length tradeoff
"""
from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple


class ContextMode(str, Enum):
    """Available context selection modes."""
    TOP_K = "top_k"               # Fixed number of contexts
    QUALITY_THRESHOLD = "quality_threshold"  # Score-based filtering
    DYNAMIC_K = "dynamic_k"       # Adaptive selection
    
    @classmethod
    def parse(cls, mode_str: str) -> Tuple["ContextMode", Dict[str, Any]]:
        """
        Parse mode string like 'top_k:10' or 'quality_threshold:0.6'.
        
        Returns:
            (ContextMode, params_dict)
        """
        if ":" in mode_str:
            mode, param = mode_str.split(":", 1)
        else:
            mode, param = mode_str, None
            
        mode = cls(mode)
        params = {}
        
        if mode == cls.TOP_K:
            params["k"] = int(param) if param else 10
        elif mode == cls.QUALITY_THRESHOLD:
            params["threshold"] = float(param) if param else 0.6
            params["min_k"] = 3  # Guarantee minimum contexts
        elif mode == cls.DYNAMIC_K:
            params["max_k"] = int(param) if param else 10
            params["quality_drop_threshold"] = 0.1  # Stop when quality drops by this
            
        return mode, params


@dataclass
class ContextSelectionConfig:
    """Configuration for context selection."""
    mode: ContextMode
    params: Dict[str, Any]
    
    # Score source for online selection
    score_source: str = "retriever"  # "retriever" or "reranker"
    
    # Whether to use normalized scores (0-1)
    normalize_scores: bool = True


def get_context_selection_params(mode_str: str) -> ContextSelectionConfig:
    """Get context selection configuration from mode string."""
    mode, params = ContextMode.parse(mode_str)
    return ContextSelectionConfig(mode=mode, params=params)


def select_contexts(
    contexts: List[Dict],
    config: ContextSelectionConfig
) -> List[Dict]:
    """
    Select contexts based on configuration.
    
    Args:
        contexts: List of context dicts with 'score' field (pre-sorted by score desc)
        config: Selection configuration
        
    Returns:
        Selected contexts
    """
    if not contexts:
        return []
    
    mode = config.mode
    params = config.params
    
    if mode == ContextMode.TOP_K:
        k = params.get("k", 10)
        return contexts[:k]
    
    elif mode == ContextMode.QUALITY_THRESHOLD:
        threshold = params.get("threshold", 0.6)
        min_k = params.get("min_k", 3)
        
        # Select all above threshold
        selected = [c for c in contexts if c.get("score", 0) >= threshold]
        
        # Ensure minimum
        if len(selected) < min_k:
            selected = contexts[:min_k]
            
        return selected
    
    elif mode == ContextMode.DYNAMIC_K:
        max_k = params.get("max_k", 10)
        drop_threshold = params.get("quality_drop_threshold", 0.1)
        
        selected = [contexts[0]] if contexts else []
        
        for i in range(1, min(len(contexts), max_k)):
            score_drop = contexts[i-1].get("score", 0) - contexts[i].get("score", 0)
            
            # Stop if quality drops significantly
            if score_drop > drop_threshold:
                break
                
            selected.append(contexts[i])
            
        return selected
    
    # Fallback: return all
    return contexts


# Predefined configurations for experiments
CONTEXT_MODES = {
    "top_k:3": "High quality, low length (3 contexts)",
    "top_k:5": "Medium quality, medium length (5 contexts)",
    "top_k:10": "Standard baseline (10 contexts)",
    "quality_threshold:0.6": "Quality-filtered (score >= 0.6)",
    "quality_threshold:0.5": "Relaxed quality filter (score >= 0.5)",
    "dynamic_k": "Adaptive quality-length tradeoff"
}
