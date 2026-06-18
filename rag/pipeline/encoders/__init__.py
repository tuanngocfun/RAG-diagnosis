"""
Encoders Package Init

Uses lazy imports to avoid loading torch when not needed
"""

def get_e5_encoder(*args, **kwargs):
    """Lazy load E5 encoder."""
    from .e5 import E5Encoder
    return E5Encoder(*args, **kwargs)

def get_bm25_retriever():
    """Get BM25 retriever (no torch required)."""
    from .bm25 import BM25Retriever
    return BM25Retriever()

# Only export functions, not classes directly
__all__ = [
    "get_e5_encoder",
    "get_bm25_retriever",
]
