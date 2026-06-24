"""
Text Chunker for RAG Pipeline

Implements 3 chunking strategies as per Q1 journal standards:
1. Fixed-size: 400 tokens with 50 overlap
2. Section-aware: Split by clinical section headers
3. Semantic: LLM-based logical unit splitting (simplified)
"""
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass


@dataclass
class Chunk:
    """Represents a text chunk."""
    case_id: str
    chunk_idx: int
    text: str
    start_char: int
    end_char: int
    strategy: str


# Clinical section headers commonly found in case reports
SECTION_HEADERS = [
    r"(?:case\s*)?presentation",
    r"history\s*of\s*present\s*illness",
    r"physical\s*examination",
    r"laboratory\s*findings",
    r"imaging",
    r"diagnosis",
    r"treatment",
    r"outcome",
    r"discussion",
    r"introduction",
    r"clinical\s*findings",
    r"investigations?",
    r"management",
    r"follow.?up",
]

SECTION_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:" + "|".join(SECTION_HEADERS) + r")[\s:]*(?:\n|$)",
    re.IGNORECASE | re.MULTILINE
)


def estimate_tokens(text: str) -> int:
    """Rough token estimation (words * 1.3)."""
    return int(len(text.split()) * 1.3)


def chunk_fixed_size(
    text: str,
    case_id: str,
    max_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Strategy 1: Fixed-size chunking with overlap.
    
    Args:
        text: Input text
        case_id: Case identifier
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
    
    Returns:
        List of Chunk objects
    """
    words = text.split()
    max_words = int(max_tokens / 1.3)
    overlap_words = int(overlap_tokens / 1.3)
    
    chunks = []
    start_idx = 0
    chunk_idx = 0
    
    while start_idx < len(words):
        end_idx = min(start_idx + max_words, len(words))
        chunk_words = words[start_idx:end_idx]
        chunk_text = " ".join(chunk_words)
        
        # Calculate character positions
        start_char = len(" ".join(words[:start_idx])) + (1 if start_idx > 0 else 0)
        end_char = start_char + len(chunk_text)
        
        chunks.append(Chunk(
            case_id=case_id,
            chunk_idx=chunk_idx,
            text=chunk_text,
            start_char=start_char,
            end_char=end_char,
            strategy="fixed"
        ))
        
        chunk_idx += 1
        start_idx = end_idx - overlap_words if end_idx < len(words) else end_idx
        
        # Avoid infinite loop
        if start_idx >= end_idx:
            break
    
    return chunks


def chunk_section_aware(
    text: str,
    case_id: str,
    max_tokens: int = 400
) -> List[Chunk]:
    """
    Strategy 2: Section-aware chunking.
    Split by clinical section headers, then subdivide if too long.
    
    Args:
        text: Input text
        case_id: Case identifier
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of Chunk objects
    """
    # Find section boundaries
    matches = list(SECTION_PATTERN.finditer(text))
    
    if not matches:
        # No sections found, fall back to fixed-size
        return chunk_fixed_size(text, case_id, max_tokens)
    
    chunks = []
    chunk_idx = 0
    
    # Process text before first section
    if matches[0].start() > 0:
        intro_text = text[:matches[0].start()].strip()
        if intro_text:
            for sub_chunk in _subdivide_if_needed(intro_text, max_tokens):
                chunks.append(Chunk(
                    case_id=case_id,
                    chunk_idx=chunk_idx,
                    text=sub_chunk,
                    start_char=text.find(sub_chunk),
                    end_char=text.find(sub_chunk) + len(sub_chunk),
                    strategy="section"
                ))
                chunk_idx += 1
    
    # Process each section
    for i, match in enumerate(matches):
        section_start = match.start()
        section_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[section_start:section_end].strip()
        
        if not section_text:
            continue
        
        for sub_chunk in _subdivide_if_needed(section_text, max_tokens):
            chunks.append(Chunk(
                case_id=case_id,
                chunk_idx=chunk_idx,
                text=sub_chunk,
                start_char=section_start,
                end_char=section_end,
                strategy="section"
            ))
            chunk_idx += 1
    
    return chunks


def _subdivide_if_needed(text: str, max_tokens: int) -> List[str]:
    """Subdivide text if it exceeds max_tokens."""
    if estimate_tokens(text) <= max_tokens:
        return [text]
    
    # Split by paragraphs first
    paragraphs = text.split("\n\n")
    
    result = []
    current = []
    current_tokens = 0
    
    for para in paragraphs:
        para_tokens = estimate_tokens(para)
        
        if current_tokens + para_tokens <= max_tokens:
            current.append(para)
            current_tokens += para_tokens
        else:
            if current:
                result.append("\n\n".join(current))
            current = [para]
            current_tokens = para_tokens
    
    if current:
        result.append("\n\n".join(current))
    
    return result


def chunk_semantic(
    text: str,
    case_id: str,
    max_tokens: int = 400
) -> List[Chunk]:
    """
    Strategy 3: Semantic chunking (simplified).
    Split by sentence boundaries, grouping related sentences.
    
    Note: Full semantic chunking would use LLM, this is a lightweight version.
    
    Args:
        text: Input text
        case_id: Case identifier
        max_tokens: Maximum tokens per chunk
    
    Returns:
        List of Chunk objects
    """
    # Split by sentences
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    chunk_idx = 0
    current_sentences = []
    current_tokens = 0
    current_start = 0
    
    for sent in sentences:
        sent_tokens = estimate_tokens(sent)
        
        if current_tokens + sent_tokens <= max_tokens:
            current_sentences.append(sent)
            current_tokens += sent_tokens
        else:
            if current_sentences:
                chunk_text = " ".join(current_sentences)
                chunks.append(Chunk(
                    case_id=case_id,
                    chunk_idx=chunk_idx,
                    text=chunk_text,
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                    strategy="semantic"
                ))
                chunk_idx += 1
                current_start += len(chunk_text) + 1
            
            current_sentences = [sent]
            current_tokens = sent_tokens
    
    # Add remaining
    if current_sentences:
        chunk_text = " ".join(current_sentences)
        chunks.append(Chunk(
            case_id=case_id,
            chunk_idx=chunk_idx,
            text=chunk_text,
            start_char=current_start,
            end_char=current_start + len(chunk_text),
            strategy="semantic"
        ))
    
    return chunks


def chunk_text(
    text: str,
    case_id: str,
    strategy: str = "fixed",
    max_tokens: int = 400,
    overlap_tokens: int = 50
) -> List[Chunk]:
    """
    Chunk text using specified strategy.
    
    Args:
        text: Input text
        case_id: Case identifier
        strategy: One of "fixed", "section", "semantic"
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap for fixed-size strategy
    
    Returns:
        List of Chunk objects
    """
    if strategy == "fixed":
        return chunk_fixed_size(text, case_id, max_tokens, overlap_tokens)
    elif strategy == "section":
        return chunk_section_aware(text, case_id, max_tokens)
    elif strategy == "semantic":
        return chunk_semantic(text, case_id, max_tokens)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def chunk_cases(
    cases: List[Dict],
    strategy: str = "fixed",
    text_field: str = "case_text",
    **kwargs
) -> List[Chunk]:
    """
    Chunk all cases.
    
    Args:
        cases: List of case dictionaries
        strategy: Chunking strategy
        text_field: Field containing text to chunk
        **kwargs: Additional arguments for chunking
    
    Returns:
        List of all chunks
    """
    all_chunks = []
    
    for case in cases:
        text = case.get(text_field, "")
        if not text:
            continue
        
        chunks = chunk_text(
            text=text,
            case_id=case["case_id"],
            strategy=strategy,
            **kwargs
        )
        all_chunks.extend(chunks)
    
    return all_chunks


if __name__ == "__main__":
    # Test chunking
    sample_text = """
    A 45-year-old male presented with fever, weight loss, and abdominal swelling 
    for 3 months. He had a history of travel to endemic areas in India.
    
    Physical Examination: The patient was pale and cachectic. Splenomegaly and 
    hepatomegaly were noted on abdominal examination.
    
    Laboratory Findings: Complete blood count showed pancytopenia. Liver function 
    tests were mildly elevated.
    
    Diagnosis: Bone marrow aspiration revealed amastigotes.
    
    Treatment: The patient was started on liposomal amphotericin B with good response.
    """
    
    for strategy in ["fixed", "section", "semantic"]:
        chunks = chunk_text(sample_text, "TEST_001", strategy)
        print(f"\n{strategy.upper()} chunking: {len(chunks)} chunks")
        for c in chunks:
            print(f"  Chunk {c.chunk_idx}: {len(c.text)} chars, ~{estimate_tokens(c.text)} tokens")
