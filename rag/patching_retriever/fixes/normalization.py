#!/usr/bin/env python3
"""
Fix for the normalize answer function - create a simpler version to prevent over-filtering
"""

import re as _re

def _normalize_answer_fixed(text: str) -> str:
    """Simplified normalization to avoid over-filtering medical answers."""
    if not text:
        print("[DEBUG] Empty input to normalize")
        return text
    
    print(f"[DEBUG] Input to normalize: {repr(text[:200])}")
    
    t = text.strip()
    
    # Remove obvious repetitions of case titles
    t = _re.sub(r"\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+A\s+LESION[^.]*\.\s*\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+A\s+LESION[^.]*\.", "", t, flags=_re.I)
    
    # Remove echoed prompts 
    t = _re.sub(r"^\s*(?:Question\s*:\s*.*?)?\b(?:Answer|ANSWER|MEDICAL ANALYSIS)\s*:\s*", "", t, flags=_re.I | _re.S)
    t = _re.sub(r"^(?:final\s+)?answer\s*:\s*", "", t, flags=_re.I).strip()
    
    # Handle multiple "Final Answer:" blocks - keep the last one
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail:
            t = tail
    
    # Remove LaTeX wrappers
    t = _re.sub(r"\\boxed\\{([^}]*)\\}", r"\1", t)
    t = _re.sub(r"\$\$?(.*?)\$\$?", r"\1", t, flags=_re.S)
    
    # Normalize whitespace
    t = _re.sub(r"\s+", " ", t).strip()
    
    # Simple deduplication - only remove exact duplicates
    sentences = [s.strip() for s in _re.split(r'[.!?]+', t) if s.strip()]
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        # Create a simple key for deduplication
        key = _re.sub(r'\W+', '', sentence.lower())
        if key and key not in seen and len(sentence) > 5:  # Much more lenient
            seen.add(key)
            unique_sentences.append(sentence)
    
    result = '. '.join(unique_sentences).strip()
    
    # Ensure proper ending
    if result and result[-1] not in '.!?':
        result += '.'
    
    print(f"[DEBUG] Normalized output: {repr(result[:200])}")
    return result


# Apply the fix by replacing the function in the run_batch_answers file
def apply_fix():
    import shutil
    
    file_path = "/home/students/Leishmania/rag/retriever/run_batch_answers_medgemma4b_test_medcpt.py"
    backup_path = file_path + ".backup"
    
    # Create backup first
    shutil.copy2(file_path, backup_path)
    print(f"Backup created: {backup_path}")
    
    # Read the original file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Find and replace the normalize function
    import re
    
    # Pattern to match the entire function
    pattern = r'def _normalize_answer\(text: str\) -> str:(.*?)(?=\ndef|\nclass|\nif __name__|$)'
    
    replacement = '''def _normalize_answer(text: str) -> str:
    """Simplified normalization to avoid over-filtering medical answers."""
    if not text:
        print("[DEBUG] Empty input to normalize")
        return text
    
    print(f"[DEBUG] Input to normalize: {repr(text[:200])}")
    
    t = text.strip()
    
    # Remove obvious repetitions of case titles
    t = _re.sub(r"\\b\\d+\\s+\\d+\\s+A\\s+\\d+-YEAR-OLD\\s+[A-Z\\s]+\\s+WITH\\s+A\\s+LESION[^.]*\\.\\s*\\b\\d+\\s+\\d+\\s+A\\s+\\d+-YEAR-OLD\\s+[A-Z\\s]+\\s+WITH\\s+A\\s+LESION[^.]*\\.", "", t, flags=_re.I)
    
    # Remove echoed prompts 
    t = _re.sub(r"^\\s*(?:Question\\s*:\\s*.*?)?\\b(?:Answer|ANSWER|MEDICAL ANALYSIS)\\s*:\\s*", "", t, flags=_re.I | _re.S)
    t = _re.sub(r"^(?:final\\s+)?answer\\s*:\\s*", "", t, flags=_re.I).strip()
    
    # Handle multiple "Final Answer:" blocks - keep the last one
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail:
            t = tail
    
    # Remove LaTeX wrappers
    t = _re.sub(r"\\\\boxed\\\\{([^}]*)\\\\}", r"\\1", t)
    t = _re.sub(r"\\$\\$?(.*?)\\$\\$?", r"\\1", t, flags=_re.S)
    
    # Normalize whitespace
    t = _re.sub(r"\\s+", " ", t).strip()
    
    # Simple deduplication - only remove exact duplicates
    sentences = [s.strip() for s in _re.split(r'[.!?]+', t) if s.strip()]
    unique_sentences = []
    seen = set()
    
    for sentence in sentences:
        # Create a simple key for deduplication
        key = _re.sub(r'\\W+', '', sentence.lower())
        if key and key not in seen and len(sentence) > 5:  # Much more lenient
            seen.add(key)
            unique_sentences.append(sentence)
    
    result = '. '.join(unique_sentences).strip()
    
    # Ensure proper ending
    if result and result[-1] not in '.!?':
        result += '.'
    
    print(f"[DEBUG] Normalized output: {repr(result[:200])}")
    return result

'''
    
    # Apply the replacement
    new_content = re.sub(pattern, replacement, content, flags=re.DOTALL)
    
    if new_content != content:
        with open(file_path, 'w') as f:
            f.write(new_content)
        print("Fix applied successfully!")
        return True
    else:
        print("Pattern not found - manual fix needed")
        return False

if __name__ == "__main__":
    apply_fix()