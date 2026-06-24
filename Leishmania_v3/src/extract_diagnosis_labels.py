#!/usr/bin/env python3
"""
LLM-Based Diagnosis Extraction using Gemini 3 Pro

Extracts structured diagnosis labels from Leishmaniasis case reports
using multimodal input (text + images) for Q1 journal quality ground truth.

Methodology follows Q1 standards:
- Few-shot prompting with examples
- Evidence span extraction
- Structured JSON output
- Confidence scoring

Output schema per case:
{
    "case_id": "PMC123456_01",
    "diagnosis": "Cutaneous Leishmaniasis",
    "species": "L. braziliensis",
    "confirmation_method": "PCR",
    "evidence_span": "PCR confirmed L. braziliensis...",
    "confidence": "high",
    "is_leishmaniasis": true
}
"""
import os
import sys
import json
import base64
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime

# Ensure proper paths
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
MODEL_NAME = "gemini-3-pro-preview"  # Gemini 3 Pro Preview
FALLBACK_MODEL = "gemini-2.5-pro"

# Real data paths (from user's dataset)
DATA_ROOT = Path(__file__).parent.parent / "data" / "leishmaniasis_multimodal"
INPUT_FILE = DATA_ROOT / "leishmaniasis_multimodal.jsonl"  # 406 cases
IMAGES_DIR = DATA_ROOT / "images"  # 155 case folders with images
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "leishmaniasis_split"


@dataclass
class DiagnosisLabel:
    """Structured diagnosis label for a case."""
    case_id: str
    diagnosis: str
    diagnosis_type: str  # Cutaneous, Visceral, Mucocutaneous, PKDL, Other
    species: Optional[str]
    confirmation_method: Optional[str]  # PCR, biopsy, serology, clinical
    evidence_span: str
    confidence: str  # high, medium, low
    is_leishmaniasis: bool
    raw_llm_response: Optional[str] = None
    error: Optional[str] = None


# Few-shot examples for the prompt
FEW_SHOT_EXAMPLES = """
### Example 1 (Confirmed Leishmaniasis):
Case Text: "A 35-year-old male presented with chronic ulcer on the arm. PCR analysis confirmed Leishmania major infection. The patient was treated with intralesional antimony."
Diagnosis:
```json
{
  "diagnosis": "Cutaneous Leishmaniasis",
  "diagnosis_type": "Cutaneous",
  "species": "Leishmania major",
  "confirmation_method": "PCR",
  "evidence_span": "PCR analysis confirmed Leishmania major infection",
  "confidence": "high",
  "is_leishmaniasis": true
}
```

### Example 2 (Confirmed Leishmaniasis):
Case Text: "The patient had hepatosplenomegaly and pancytopenia. Bone marrow aspirate showed amastigotes. Diagnosed with kala-azar and treated with liposomal amphotericin B."
Diagnosis:
```json
{
  "diagnosis": "Visceral Leishmaniasis (Kala-azar)",
  "diagnosis_type": "Visceral",
  "species": null,
  "confirmation_method": "bone marrow biopsy",
  "evidence_span": "Bone marrow aspirate showed amastigotes. Diagnosed with kala-azar",
  "confidence": "high",
  "is_leishmaniasis": true
}
```

### Example 3 (NOT Leishmaniasis):
Case Text: "Trypanosoma cruzi was detected by PCR. The patient was diagnosed with Chagas disease and started on benznidazole therapy."
Diagnosis:
```json
{
  "diagnosis": "Chagas Disease",
  "diagnosis_type": "Other",
  "species": "Trypanosoma cruzi",
  "confirmation_method": "PCR",
  "evidence_span": "Trypanosoma cruzi was detected by PCR. The patient was diagnosed with Chagas disease",
  "confidence": "high",
  "is_leishmaniasis": false
}
```

### Example 4 (NOT Leishmaniasis):
Case Text: "Patient presents with cutaneous metastases from gastric adenocarcinoma. Histopathology confirmed signet-ring cell carcinoma."
Diagnosis:
```json
{
  "diagnosis": "Gastric Cancer with Cutaneous Metastases",
  "diagnosis_type": "Other",
  "species": null,
  "confirmation_method": "histopathology",
  "evidence_span": "Histopathology confirmed signet-ring cell carcinoma",
  "confidence": "high",
  "is_leishmaniasis": false
}
```
"""

DIAGNOSIS_PROMPT = """You are an expert medical diagnostician specializing in tropical infectious diseases.

Your task is to extract the FINAL CONFIRMED DIAGNOSIS from a medical case report.

## Important Instructions:
1. Read the case text and any images carefully
2. Identify the FINAL diagnosis (not differential diagnoses)
3. Look for confirmation methods: PCR, biopsy, serology, culture, microscopy
4. Extract the exact sentence(s) that confirm the diagnosis
5. Determine if this is actually Leishmaniasis or a different disease
6. Be honest - if the case is NOT about Leishmaniasis, mark is_leishmaniasis: false

## Leishmaniasis Types:
- **Cutaneous (CL)**: Skin ulcers/lesions
- **Visceral (VL/Kala-azar)**: Hepatosplenomegaly, pancytopenia, fever
- **Mucocutaneous (MCL)**: Nasal/oral mucosal involvement
- **PKDL**: Post-kala-azar dermal leishmaniasis
- **Other**: Not leishmaniasis (Chagas, TB, cancer, etc.)

## Few-Shot Examples:
{examples}

---

## Now extract diagnosis from this case:

### Title:
{title}

### Case Text:
{case_text}

### Response Format:
Return ONLY a valid JSON object with these fields:
```json
{{
  "diagnosis": "Full diagnosis name",
  "diagnosis_type": "Cutaneous|Visceral|Mucocutaneous|PKDL|Other",
  "species": "Leishmania species or null",
  "confirmation_method": "PCR|biopsy|serology|culture|microscopy|clinical|null",
  "evidence_span": "The exact sentence(s) confirming diagnosis",
  "confidence": "high|medium|low",
  "is_leishmaniasis": true|false
}}
```
"""


import time
from typing import Tuple

# Define the structured output schema for diagnosis extraction
# Note: Google GenAI SDK doesn't support ["string", "null"] - use STRING and make optional via required
DIAGNOSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "diagnosis": {
            "type": "string",
            "description": "Full diagnosis name"
        },
        "diagnosis_type": {
            "type": "string",
            "enum": ["Cutaneous", "Visceral", "Mucocutaneous", "PKDL", "Other"],
            "description": "Type of diagnosis"
        },
        "species": {
            "type": "string",
            "description": "Leishmania species if identified, or empty string if not identified"
        },
        "confirmation_method": {
            "type": "string",
            "description": "Method used to confirm diagnosis (PCR, biopsy, serology, culture, microscopy, clinical) or empty string"
        },
        "evidence_span": {
            "type": "string",
            "description": "The exact sentence(s) confirming the diagnosis"
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence level in the diagnosis extraction"
        },
        "is_leishmaniasis": {
            "type": "boolean",
            "description": "Whether this case is about Leishmaniasis"
        }
    },
    "required": ["diagnosis", "diagnosis_type", "evidence_span", "confidence", "is_leishmaniasis"]
}


class DiagnosisExtractor:
    """Extract diagnosis labels using Gemini with structured output and robust error handling."""
    
    # Retry configuration
    MAX_RETRIES = 3
    BASE_DELAY = 2.0  # seconds
    
    def __init__(self, model: str = None, api_key: str = None):
        self.api_key = api_key or GOOGLE_API_KEY
        self.model_name = model or MODEL_NAME
        
        try:
            from google import genai
            from google.genai import types
            self.client = genai.Client(api_key=self.api_key)
            self._types = types
            self._genai = genai
            self._test_model()
        except ImportError:
            raise ImportError("pip install google-genai")
    
    def _test_model(self):
        """Test model availability, fallback if needed."""
        try:
            self.client.models.generate_content(
                model=self.model_name,
                contents="Say hello",
                config=self._types.GenerateContentConfig(max_output_tokens=10)
            )
            print(f"✓ Using {self.model_name}")
        except Exception as e:
            print(f"Warning: {self.model_name} not available: {e}")
            self.model_name = FALLBACK_MODEL
            print(f"✓ Fallback to {self.model_name}")
    
    def _load_image(self, image_path: str) -> Optional[dict]:
        """Load and encode image for multimodal input."""
        path = Path(image_path)
        if not path.exists():
            return None
        
        try:
            with open(path, "rb") as f:
                img_data = base64.b64encode(f.read()).decode("utf-8")
            
            suffix = path.suffix.lower()
            mime_map = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".webp": "image/webp"
            }
            mime_type = mime_map.get(suffix, "image/jpeg")
            
            return {
                "inline_data": {
                    "mime_type": mime_type,
                    "data": img_data
                }
            }
        except Exception:
            return None
    
    def _extract_text_safely(self, response) -> Tuple[str, Dict]:
        """
        Defensive text extraction from Gemini response.
        
        Per GPT 5.2: Using response.text directly throws when blocked.
        Instead, check candidates and extract safely.
        
        Returns:
            Tuple of (text, metadata_dict)
        """
        meta = {"blocked": False, "finish_reason": None}
        
        # Try response.text first (works when not blocked)
        try:
            if hasattr(response, 'text') and response.text:
                return response.text, meta
        except Exception:
            pass
        
        # Try response.parsed for structured output
        try:
            if hasattr(response, 'parsed') and response.parsed:
                # Convert parsed object to JSON string
                return json.dumps(response.parsed), meta
        except Exception:
            pass
        
        # Check candidates for blocked content
        candidates = getattr(response, "candidates", None)
        if not candidates:
            meta["blocked"] = True
            meta["finish_reason"] = "NO_CANDIDATES"
            return "", meta
        
        cand = candidates[0]
        finish_reason = getattr(cand, "finish_reason", None)
        meta["finish_reason"] = str(finish_reason) if finish_reason else None
        
        # Check if blocked by safety filter only (MAX_TOKENS is not blocked - it has partial content)
        if finish_reason and "SAFETY" in str(finish_reason):
            meta["blocked"] = True
        
        # For MAX_TOKENS, we may still have partial text to parse - don't mark as blocked
        
        # Try to extract text from parts
        content = getattr(cand, "content", None)
        if content:
            parts = getattr(content, "parts", None) or []
            texts = []
            for p in parts:
                t = getattr(p, "text", None)
                if t:
                    texts.append(t)
            if texts:
                return "".join(texts).strip(), meta
        
        meta["blocked"] = True
        return "", meta
    
    def _parse_json_response(self, text: str) -> dict:
        """Parse JSON from LLM response."""
        if not text:
            return {"error": "Empty response"}
            
        # Try to extract JSON from markdown code block
        if "```json" in text:
            start = text.find("```json") + 7
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        elif "```" in text:
            start = text.find("```") + 3
            end = text.find("```", start)
            if end > start:
                text = text[start:end].strip()
        
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            # Try to fix common issues: incomplete JSON
            if text.startswith("{") and not text.endswith("}"):
                # Try to complete the JSON
                text = text.rsplit(",", 1)[0]  # Remove incomplete last field
                text += "}"
                try:
                    return json.loads(text)
                except:
                    pass
            return {"error": f"Failed to parse JSON: {str(e)}", "raw": text[:200]}
    
    def _get_head_tail_text(self, case_text: str, head_size: int = 7000, tail_size: int = 7000) -> str:
        """
        Get HEAD + TAIL of case text to capture both intro and conclusion.
        
        Per GPT 5.2: Final diagnosis often appears at the end of case reports.
        Cutting only the first 8000 chars may miss critical diagnosis statements.
        """
        if len(case_text) <= head_size + tail_size:
            return case_text
        
        head = case_text[:head_size]
        tail = case_text[-tail_size:]
        
        return f"{head}\n\n[...middle section truncated for brevity...]\n\n{tail}"
    
    def _generate_with_retry(self, contents: list, config) -> Tuple[Any, str]:
        """
        Generate content with exponential backoff retry for rate limits.
        
        Returns:
            Tuple of (response, error_message)
        """
        last_error = None
        
        for attempt in range(self.MAX_RETRIES):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=contents,
                    config=config
                )
                return response, None
                
            except Exception as e:
                error_str = str(e)
                last_error = error_str
                
                # Check if rate limited
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    delay = self.BASE_DELAY * (2 ** attempt)
                    print(f"    Rate limited, waiting {delay}s before retry {attempt + 1}/{self.MAX_RETRIES}")
                    time.sleep(delay)
                else:
                    # Non-retryable error
                    return None, error_str
        
        return None, f"Max retries exceeded: {last_error}"
    
    def extract_diagnosis(
        self,
        case_id: str,
        title: str,
        case_text: str,
        image_paths: Optional[List[str]] = None,
        image_captions: Optional[List[str]] = None,
        image_types: Optional[List[str]] = None
    ) -> DiagnosisLabel:
        """
        Extract diagnosis from a single case using multimodal input.
        
        Uses:
        - HEAD+TAIL text extraction to catch diagnoses at document end
        - Structured output (response_schema) for guaranteed valid JSON
        - Defensive text extraction for None/blocked responses
        - Exponential backoff retry for rate limits
        """
        # Use HEAD+TAIL extraction instead of just head
        processed_text = self._get_head_tail_text(case_text)
        
        # Build prompt (simplified since we're using response_schema)
        prompt = f"""You are an expert medical diagnostician specializing in tropical infectious diseases.

Extract the FINAL CONFIRMED DIAGNOSIS from this medical case report.

## Instructions:
1. Read the case text and any images carefully
2. Identify the FINAL diagnosis (not differential diagnoses)
3. Look for confirmation methods: PCR, biopsy, serology, culture, microscopy
4. Extract the exact sentence(s) that confirm the diagnosis
5. Determine if this is actually Leishmaniasis or a different disease
6. Be honest - if the case is NOT about Leishmaniasis, mark is_leishmaniasis: false

## Leishmaniasis Types:
- **Cutaneous (CL)**: Skin ulcers/lesions
- **Visceral (VL/Kala-azar)**: Hepatosplenomegaly, pancytopenia, fever
- **Mucocutaneous (MCL)**: Nasal/oral mucosal involvement
- **PKDL**: Post-kala-azar dermal leishmaniasis
- **Other**: Not leishmaniasis (Chagas, TB, cancer, etc.)

{FEW_SHOT_EXAMPLES}

---

## Case to analyze:

### Title:
{title}

### Case Text:
{processed_text}
"""
        
        # Build multimodal content
        contents = []
        
        # Add images with their metadata (max 5)
        if image_paths:
            valid_images = 0
            for idx, img_path in enumerate(image_paths[:5]):
                img_content = self._load_image(img_path)
                if img_content:
                    contents.append(img_content)
                    valid_images += 1
                    
                    # Add caption and type context after each image
                    img_context = []
                    if image_captions and idx < len(image_captions) and image_captions[idx]:
                        img_context.append(f"Caption: {image_captions[idx]}")
                    if image_types and idx < len(image_types) and image_types[idx]:
                        img_context.append(f"Image type: {image_types[idx]}")
                    if img_context:
                        contents.append(" | ".join(img_context) + "\n")
            
            if valid_images > 0:
                contents.append(f"\n[{valid_images} medical images shown above]\n\n")
        
        # Add text prompt
        contents.append(prompt)
        
        # Generate with structured output (response_schema)
        try:
            config = self._types.GenerateContentConfig(
                temperature=0.0,  # Low for deterministic extraction
                max_output_tokens=8096,  # Increased to avoid MAX_TOKENS cutoff
                response_mime_type="application/json",
                response_schema=DIAGNOSIS_SCHEMA,
                safety_settings=[
                    self._types.SafetySetting(
                        category=c,
                        threshold=self._types.HarmBlockThreshold.OFF
                    )
                    for c in [
                        self._types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                        self._types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        self._types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        self._types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    ]
                ]
            )
            
            # Use retry logic
            response, error = self._generate_with_retry(contents, config)
            
            if error:
                return DiagnosisLabel(
                    case_id=case_id,
                    diagnosis="Error",
                    diagnosis_type="Other",
                    species=None,
                    confirmation_method=None,
                    evidence_span="",
                    confidence="low",
                    is_leishmaniasis=False,
                    error=error
                )
            
            # Defensive text extraction
            raw_text, meta = self._extract_text_safely(response)
            
            if meta.get("blocked"):
                return DiagnosisLabel(
                    case_id=case_id,
                    diagnosis="Blocked",
                    diagnosis_type="Other",
                    species=None,
                    confirmation_method=None,
                    evidence_span="",
                    confidence="low",
                    is_leishmaniasis=False,
                    error=f"Response blocked: {meta.get('finish_reason', 'unknown')}"
                )
            
            if not raw_text:
                return DiagnosisLabel(
                    case_id=case_id,
                    diagnosis="Empty",
                    diagnosis_type="Other",
                    species=None,
                    confirmation_method=None,
                    evidence_span="",
                    confidence="low",
                    is_leishmaniasis=False,
                    error="Empty response from model"
                )
            
            # Parse response (should be valid JSON due to response_schema)
            parsed = self._parse_json_response(raw_text)
            
            if "error" in parsed:
                return DiagnosisLabel(
                    case_id=case_id,
                    diagnosis="Unknown",
                    diagnosis_type="Other",
                    species=None,
                    confirmation_method=None,
                    evidence_span="",
                    confidence="low",
                    is_leishmaniasis=False,
                    raw_llm_response=raw_text,
                    error=parsed.get("error")
                )
            
            return DiagnosisLabel(
                case_id=case_id,
                diagnosis=parsed.get("diagnosis", "Unknown"),
                diagnosis_type=parsed.get("diagnosis_type", "Other"),
                species=parsed.get("species"),
                confirmation_method=parsed.get("confirmation_method"),
                evidence_span=parsed.get("evidence_span", ""),
                confidence=parsed.get("confidence", "low"),
                is_leishmaniasis=parsed.get("is_leishmaniasis", False),
                raw_llm_response=raw_text
            )
            
        except Exception as e:
            return DiagnosisLabel(
                case_id=case_id,
                diagnosis="Error",
                diagnosis_type="Other",
                species=None,
                confirmation_method=None,
                evidence_span="",
                confidence="low",
                is_leishmaniasis=False,
                error=str(e)
            )


def extract_all_diagnoses(
    input_file: Path,
    output_file: Path,
    images_dir: Path = IMAGES_DIR,
    limit: int = None,
    resume: bool = False
) -> Path:
    """
    Extract diagnoses from all cases in a JSONL file.
    
    Args:
        input_file: Input JSONL with cases
        output_file: Output JSONL with diagnosis labels
        images_dir: Directory containing case images
        limit: Optional limit on number of cases to process
        resume: If True, load existing output and only retry errors
    
    Returns:
        Path to output file
    """
    extractor = DiagnosisExtractor()
    
    # Load cases
    with open(input_file) as f:
        cases = [json.loads(l) for l in f]
    
    if limit:
        cases = cases[:limit]
    
    # Build case_id to case mapping
    case_map = {c["case_id"]: c for c in cases}
    
    # Load existing results if resuming
    existing_results = {}
    cases_to_process = []
    
    if resume and output_file.exists():
        print(f"Resuming from {output_file}...")
        with open(output_file) as f:
            for line in f:
                result = json.loads(line)
                case_id = result["case_id"]
                existing_results[case_id] = result
                
                # Check if this needs retry (has error)
                if result.get("error"):
                    if case_id in case_map:
                        cases_to_process.append(case_map[case_id])
                        print(f"  Will retry: {case_id} (error: {result.get('error', '')[:50]}...)")
        
        # Keep all successful results
        print(f"Loaded {len(existing_results)} existing results")
        print(f"  Successful (keeping): {len(existing_results) - len(cases_to_process)}")
        print(f"  Errors (retrying): {len(cases_to_process)}")
    else:
        cases_to_process = cases
    
    if not cases_to_process:
        print("No cases to process - all successful!")
        return output_file
    
    print(f"Processing {len(cases_to_process)} cases...")
    
    stats = {"total": 0, "is_leish": 0, "not_leish": 0, "errors": 0, "retried": 0}
    
    for i, case in enumerate(cases_to_process):
        case_id = case["case_id"]
        title = case.get("title", "")
        case_text = case.get("case_text", "")
        
        # Extract images from JSONL metadata (rich data with captions, types, labels)
        images_meta = case.get("images", [])
        image_paths = []
        image_captions = []
        image_types = []
        
        for img in images_meta[:5]:  # Max 5 images
            file_name = img.get("file", "")
            if file_name:
                # Build full path: images_dir / case_id / filename
                img_path = images_dir / case_id / file_name
                if img_path.exists():
                    image_paths.append(str(img_path))
                    # Collect rich metadata
                    caption = img.get("caption", "")
                    img_type = img.get("image_type", "")
                    img_subtype = img.get("image_subtype", "")
                    labels = img.get("labels_supervised", "")
                    
                    if caption:
                        image_captions.append(caption)
                    if img_type:
                        type_info = f"{img_type}"
                        if img_subtype and img_subtype != img_type:
                            type_info += f" ({img_subtype})"
                        image_types.append(type_info)
        
        # Extract diagnosis with images and their metadata
        label = extractor.extract_diagnosis(
            case_id=case_id,
            title=title,
            case_text=case_text,
            image_paths=image_paths,
            image_captions=image_captions,
            image_types=image_types
        )
        
        # Update existing results with new extraction
        existing_results[case_id] = asdict(label)
        
        # Update stats
        stats["total"] += 1
        if case_id in case_map:
            stats["retried"] += 1
        if label.error:
            stats["errors"] += 1
        elif label.is_leishmaniasis:
            stats["is_leish"] += 1
        else:
            stats["not_leish"] += 1
        
        # Progress
        if (i + 1) % 10 == 0 or (i + 1) == len(cases_to_process):
            print(f"  Processed {i + 1}/{len(cases_to_process)} "
                  f"(Leish: {stats['is_leish']}, Other: {stats['not_leish']}, Errors: {stats['errors']})")
    
    # Build final results in original order
    final_results = []
    for case in cases:
        case_id = case["case_id"]
        if case_id in existing_results:
            final_results.append(existing_results[case_id])
    
    # Save results
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for r in final_results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    # Calculate final stats from all results
    final_stats = {"total": len(final_results), "is_leish": 0, "not_leish": 0, "errors": 0}
    for r in final_results:
        if r.get("error"):
            final_stats["errors"] += 1
        elif r.get("is_leishmaniasis"):
            final_stats["is_leish"] += 1
        else:
            final_stats["not_leish"] += 1
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"DIAGNOSIS EXTRACTION COMPLETE")
    print(f"{'='*60}")
    print(f"Total cases: {final_stats['total']}")
    print(f"Confirmed Leishmaniasis: {final_stats['is_leish']} ({100*final_stats['is_leish']/final_stats['total']:.1f}%)")
    print(f"NOT Leishmaniasis: {final_stats['not_leish']} ({100*final_stats['not_leish']/final_stats['total']:.1f}%)")
    print(f"Remaining errors: {final_stats['errors']}")
    if resume:
        print(f"Retried this run: {stats['retried']}")
    print(f"\nOutput saved to: {output_file}")
    print(f"{'='*60}")
    
    return output_file


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract diagnosis labels using Gemini 3 Pro")
    parser.add_argument(
        "--input", "-i",
        type=Path,
        default=INPUT_FILE,  # leishmaniasis_multimodal.jsonl (406 cases)
        help="Input JSONL file"
    )
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=OUTPUT_DIR / "cases_with_diagnosis.jsonl",
        help="Output JSONL file"
    )
    parser.add_argument(
        "--limit", "-n",
        type=int,
        default=None,
        help="Limit number of cases to process (for testing)"
    )
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from existing output, only retry cases with errors"
    )
    
    args = parser.parse_args()
    
    extract_all_diagnoses(
        input_file=args.input,
        output_file=args.output,
        limit=args.limit,
        resume=args.resume
    )
