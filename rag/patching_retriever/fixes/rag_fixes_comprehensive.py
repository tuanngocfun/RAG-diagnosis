"""
Comprehensive fixes for the RAG pipeline issues
"""

# Fix 1: Better normalization function to replace the garbage _normalize_answer
NORMALIZE_ANSWER_FIX = '''
def _normalize_answer(text: str) -> str:
    """
    Fixed normalization that handles garbage output, repetition, and prompt leakage
    """
    if not text:
        return text
    
    t = text.strip()
    
    # Early garbage detection - truncate very long outputs
    if len(t) > 4000:
        logging.warning("Truncating very long answer (likely repetitive)")
        t = t[:2000] + "..."
    
    # Remove prompt leakage patterns first
    t = re.sub(r'^\\s*(?:user\\s+)?(?:You are a medical expert[^.]*\\.)?', '', t, flags=re.I)
    t = re.sub(r'Evidence Sources:\\s*\\[?\\d+\\]?[^.]*\\.', '', t, flags=re.I)
    t = re.sub(r'User has uploaded \\d+ files:[^.]*\\.', '', t, flags=re.I)
    
    # Remove answer prefixes and labels  
    t = re.sub(r'^\\s*(?:Question\\s*:\\s*.*?)?\\b(?:Answer|ANSWER|MEDICAL ANALYSIS)\\s*:\\s*', '', t, flags=re.I | re.S)
    
    # Handle Final Answer sections
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail and len(tail) < len(t) * 0.8:
            t = tail
    
    # Remove repetitive case descriptions
    t = re.sub(r'(?:\\b\\d+\\s+\\d+\\s+A\\s+\\d+-YEAR-OLD\\s+[A-Z\\s]+\\s+WITH\\s+A\\s+LESION[^.]*\\.?\\s*){2,}', 
               '', t, flags=re.I)
    t = re.sub(r'(?:The lesion progressed quickly from a sore to eat through[^.]*\\.?\\s*){2,}', 
               'The lesion progressed quickly.', t, flags=re.I)
    
    # Remove LaTeX and formatting
    t = re.sub(r'\\\\boxed\\\\{([^}]*)\\\\}', r'\\1', t)
    t = re.sub(r'\\$\\$?(.*?)\\$\\$?', r'\\1', t, flags=re.S)
    
    # Normalize whitespace
    t = re.sub(r'\\s+', ' ', t).strip()
    
    # Enhanced sentence deduplication
    sentences = re.split(r'(?<=[.!?])(?:["\\')\\]\\}]+)?(?:\\s+)', t)
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        # Create deduplication key
        key = re.sub(r'\\W+', '', sentence.lower())[:50]  # First 50 chars
        
        # Skip obvious repetitive patterns
        if any(pattern in sentence.lower() for pattern in [
            "year-old boy from laos", "year-old man from cambodia",
            "progressed quickly from", "medical expert"
        ]):
            continue
        
        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(sentence)
            
            # Limit sentences to prevent runaway
            if len(unique_sentences) >= 6:
                break
    
    result = " ".join(unique_sentences).strip()
    
    # Ensure proper ending
    if result and result[-1] not in '.!?':
        last_punct = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_punct > len(result) * 0.7:
            result = result[:last_punct + 1]
        else:
            result += "."
    
    # Final sanity check
    if len(result) < 30 or result.count(' ') < 4:
        return "Unable to generate a clear medical answer from the available information."
    
    return result
'''

# Fix 2: Better prompt building to prevent garbage generation
PROMPT_FIX = '''
def _build_focused_prompt(question: str, spans: List[Tuple[str, str]], context: str) -> str:
    """
    Build a focused prompt that reduces garbage output
    """
    # Detect question type
    q_lower = question.lower()
    
    if any(term in q_lower for term in ["cure", "treat", "therapy"]):
        instruction = "Provide a concise treatment answer (2-3 sentences maximum)."
    elif any(term in q_lower for term in ["diagnosis", "identify", "what is"]):
        instruction = "Provide a clear diagnostic assessment (2-3 sentences maximum)."
    else:
        instruction = "Provide a focused medical answer (2-3 sentences maximum)."
    
    # Build concise evidence
    evidence_text = ""
    if spans:
        # Take only the most relevant spans
        relevant_spans = []
        for span_text, citation in spans[:3]:
            if len(span_text.strip()) > 20:
                relevant_spans.append(span_text.strip()[:200])
        
        if relevant_spans:
            evidence_text = " ".join(relevant_spans)
    
    # Combine with context, keeping it short
    all_context = f"{context} {evidence_text}".strip()
    if len(all_context) > 1000:
        all_context = all_context[:1000] + "..."
    
    # Create focused prompt
    prompt = f"""Question: {question}

{instruction}

Medical Evidence: {all_context}

Answer:"""
    
    return prompt
'''

# Fix 3: Better image handling to prevent token format issues
IMAGE_HANDLING_FIX = '''
# In the answer() method, add better image validation:
def _validate_images(self, image_paths: List[Path], max_images: int = 2) -> List[Path]:
    """Validate and limit images to prevent token format issues"""
    valid_paths = []
    
    for path in image_paths[:max_images]:
        try:
            if path.exists() and path.stat().st_size > 0:
                # Check if it's a valid image
                from PIL import Image
                with Image.open(path) as img:
                    if img.size[0] > 50 and img.size[1] > 50:  # Reasonable size
                        valid_paths.append(path)
        except Exception:
            continue
    
    return valid_paths[:max_images]  # Hard limit to prevent issues
'''

# Fix 4: Enhanced answer generation with safeguards
GENERATION_FIX = '''
def generate_answer_with_safeguards(self, prompt: str, image_paths: List[Path], max_tokens: int = 512) -> str:
    """
    Generate answer with safeguards against garbage output
    """
    try:
        # Validate inputs
        validated_images = self._validate_images(image_paths, max_images=2)
        
        # Limit token generation strictly
        safe_max_tokens = min(max_tokens, 256)  # Conservative limit
        
        # Use simpler generation parameters
        generation_kwargs = {
            "max_new_tokens": safe_max_tokens,
            "temperature": 0.7,  # Not too creative
            "do_sample": True,
            "pad_token_id": self.tokenizer.eos_token_id,
            "repetition_penalty": 1.1,  # Prevent repetition
        }
        
        # Generate with proper error handling
        if validated_images:
            # Try multimodal generation
            try:
                inputs = self.processor(
                    text=prompt,
                    images=[Image.open(p) for p in validated_images],
                    return_tensors="pt"
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                
                # Decode properly
                generated_ids = outputs[0][len(inputs["input_ids"][0]):]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                
            except Exception as e:
                logging.warning(f"Multimodal generation failed, falling back to text-only: {e}")
                # Fallback to text-only
                inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(**inputs, **generation_kwargs)
                generated_ids = outputs[0][len(inputs["input_ids"][0]):]
                answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        else:
            # Text-only generation
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, **generation_kwargs)
            generated_ids = outputs[0][len(inputs["input_ids"][0]):]
            answer = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        return answer.strip()
        
    except Exception as e:
        logging.error(f"Generation failed: {e}")
        return "Unable to generate a medical response due to technical issues."
'''

print("Generated comprehensive fixes for the RAG pipeline issues:")
print("1. Fixed normalization to handle garbage output and prompt leakage")
print("2. Improved prompt building to reduce verbose generation") 
print("3. Better image validation to prevent token format issues")
print("4. Enhanced generation with safeguards and fallbacks")
print("\\nThese fixes should be applied to the medgemma4b_qdrant_crossencoder_medcpt.py file")