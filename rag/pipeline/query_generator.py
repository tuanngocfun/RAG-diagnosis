"""
Query Generator for RAG Evaluation

Generates 3 query types to avoid data leakage:
- Q1: Symptom-only (no diagnosis/pathogen names)
- Q2: Symptom + Exposure (add geography, lesion site)
- Q3: Image-only (for cases with images)

Per GPT 5.2 feedback: Also filters treatment terms to avoid indirect leakage

Standardized Query Support (v2.0):
- Uses query_templates.py for fixed question phrasing
- Outputs StandardizedEvalQuery objects with explicit diagnosis questions
- Separates query from clinical context for proper RAGAS evaluation
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass

# Import standardized templates
from .query_templates import (
    build_diagnosis_query,
    sanitize_case_text,
    StandardizedQuery,
    DIAGNOSIS_QUESTION_WITH_TYPE,
)

# Disease/pathogen terms to filter out
# BUG FIX (Claude 4.5 Opus): Added \b word boundaries to prevent matching
# within words like 'extensive', 'perioral', 'HIV', 'improved', 'University'
LEAK_PATTERNS = [
    # Leishmania species
    r"\bleishmania\s*\w*",
    r"\bl\.\s*(donovani|infantum|tropica|major|braziliensis|mexicana|amazonensis)\b",
    # Disease forms
    r"\bvisceral\s*leishmaniasis\b",
    r"\bcutaneous\s*leishmaniasis\b",
    r"\bmucocutaneous\s*leishmaniasis\b",
    r"\bpost.?kala.?azar\b",
    r"\bpkdl\b",
    r"\bkala.?azar\b",
    # Generic diagnosis terms
    r"\bleishmaniasis\b",
    r"\bleishmanial\b",
    # Treatment terms (per GPT 5.2 feedback)
    r"\bamphotericin\s*b?\b",
    r"\bliposomal\s*amphotericin\b",
    r"\bmiltefosine\b",
    r"\bsodium\s*stibogluconate\b",
    r"\bmeglumine\s*antimoniate\b",
    r"\bpentamidine\b",
    r"\bparomomycin\b",
    r"\bantimonial\b",
    # Dosage patterns (keep without word boundary - numbers are ok)
    r"\d+\s*mg(/kg)?(/day)?",
    r"\d+\s*mL",
    # CRITICAL FIX: Added \b to prevent matching within words like
    # 'extensive' (IV), 'perioral' (oral), 'improved' (IM), 'University' (IV)
    r"\b(IV|IM|oral|intramuscular|intravenous)\b\s*(administration|route)?",
]

# Compile patterns
LEAK_REGEX = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


@dataclass
class Query:
    """Represents a query for evaluation."""
    case_id: str
    query_type: str  # Q1, Q2, Q3
    text: Optional[str] = None
    image_path: Optional[str] = None
    ground_truth_case_ids: List[str] = None


def filter_leak_terms(text: str) -> str:
    """Remove diagnosis/pathogen/treatment terms that would leak the answer."""
    return LEAK_REGEX.sub("[REDACTED]", text)


def extract_symptoms(case_text: str, max_words: int = 150) -> str:
    """
    Extract symptom-related sentences from case text.
    Filters out diagnosis AND treatment statements (per GPT 5.2).
    """
    # Split into sentences
    sentences = re.split(r'[.!?]', case_text)
    
    # Keywords indicating symptom descriptions
    # EXPANDED per root cause analysis - added tiredness, complaint, malaise, etc.
    symptom_keywords = [
        "presented", "complained", "complaint", "symptoms", "fever", "lesion",
        "swelling", "pain", "weight loss", "fatigue", "weakness", "tiredness",
        "malaise", "lethargy", "splenomegaly", "hepatomegaly", "ulcer", "nodule", 
        "rash", "anemia", "lymphadenopathy", "hyperpigmentation", "pancytopenia",
        "thrombocytopenia", "cytopenias", "intermittent", "chronic"
    ]
    
    # Keywords indicating diagnosis (to exclude)
    diagnosis_keywords = [
        "diagnosed", "diagnosis", "confirmed", "revealed", "identified",
        "positive for", "test showed", "biopsy showed"
    ]
    
    # Keywords indicating treatment (to exclude - per GPT 5.2 feedback)
    treatment_keywords = [
        "treatment", "treated", "started on", "administered",
        "received", "given", "prescribed", "therapy", "regimen",
        "completed", "course of", "responded to"
    ]
    
    selected = []
    word_count = 0
    
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
            
        # Skip diagnosis sentences
        if any(kw in sent.lower() for kw in diagnosis_keywords):
            continue
        
        # Skip treatment sentences (NEW - per GPT 5.2)
        if any(kw in sent.lower() for kw in treatment_keywords):
            continue
            
        # Prefer symptom sentences
        if any(kw in sent.lower() for kw in symptom_keywords):
            filtered = filter_leak_terms(sent)
            words = filtered.split()
            if word_count + len(words) <= max_words:
                selected.append(filtered)
                word_count += len(words)
    
    # FALLBACK: If no symptom sentences found, use first 2 non-diagnosis sentences
    if not selected:
        fallback_count = 0
        for sent in sentences:
            sent = sent.strip()
            if not sent or len(sent) < 20:  # Skip very short sentences
                continue
            if any(kw in sent.lower() for kw in diagnosis_keywords):
                continue
            if any(kw in sent.lower() for kw in treatment_keywords):
                continue
            filtered = filter_leak_terms(sent)
            if filtered and len(filtered) > 20:
                selected.append(filtered)
                fallback_count += 1
                if fallback_count >= 2:
                    break
                
    return ". ".join(selected) + "." if selected else ""


def extract_demographics(case: Dict) -> str:
    """
    Extract age, gender from case metadata OR case_text.
    
    BUG FIX: Original only checked metadata fields which were empty
    for all 29 test cases. Now also parses from case_text.
    """
    parts = []
    
    # First, try metadata fields
    if case.get("age"):
        parts.append(f"{case['age']}-year-old")
    if case.get("gender"):
        parts.append(case["gender"].lower())
    
    # If no metadata, try to extract from case_text
    if not parts and case.get("case_text"):
        case_text = case["case_text"][:500]  # Only check first 500 chars
        
        # Pattern: "50-year-old Caucasian male" or "A 32-year-old female"
        age_pattern = r"(\d{1,3})-year-old"
        gender_pattern = r"(\d{1,3})-year-old\s+(?:\w+\s+)?(male|female|man|woman|boy|girl)"
        
        # Try combined pattern first
        match = re.search(gender_pattern, case_text, re.IGNORECASE)
        if match:
            parts.append(f"{match.group(1)}-year-old")
            gender = match.group(2).lower()
            # Normalize gender terms
            if gender in ("man", "boy"):
                gender = "male"
            elif gender in ("woman", "girl"):
                gender = "female"
            parts.append(gender)
        else:
            # Try age-only pattern
            match = re.search(age_pattern, case_text)
            if match:
                parts.append(f"{match.group(1)}-year-old")
    
    return " ".join(parts) + " patient" if parts else "Patient"


def extract_exposure_info(case_text: str) -> str:
    """Extract geography, travel, exposure information."""
    exposure_keywords = [
        "travel", "visited", "endemic", "resident", "from",
        "exposure", "contact", "outdoor", "rural", "forest"
    ]
    
    sentences = re.split(r'[.!?]', case_text)
    exposure_info = []
    
    for sent in sentences:
        sent = sent.strip()
        if any(kw in sent.lower() for kw in exposure_keywords):
            filtered = filter_leak_terms(sent)
            if len(filtered) > 10:  # Skip very short fragments
                exposure_info.append(filtered)
                
    return ". ".join(exposure_info[:2])  # Max 2 sentences


def generate_q1_symptom_only(case: Dict) -> Query:
    """
    Q1: Symptom-only query
    Demographics + symptoms, NO diagnosis or pathogen names
    """
    demographics = extract_demographics(case)
    symptoms = extract_symptoms(case.get("case_text", ""))
    
    query_text = f"{demographics} with {symptoms}" if symptoms else demographics
    
    return Query(
        case_id=case["case_id"],
        query_type="Q1_symptom_only",
        text=query_text
    )


def generate_q2_symptom_exposure(case: Dict) -> Query:
    """
    Q2: Symptom + Exposure query
    Q1 + geography, travel history, exposure
    """
    q1 = generate_q1_symptom_only(case)
    exposure = extract_exposure_info(case.get("case_text", ""))
    
    query_text = q1.text
    if exposure:
        query_text += f" | Exposure: {exposure}"
    
    return Query(
        case_id=case["case_id"],
        query_type="Q2_symptom_exposure",
        text=query_text
    )


def generate_q3_image_only(case: Dict, images_dir: Path) -> Optional[Query]:
    """
    Q3: Image-only query
    Returns query with image path, no text context
    """
    images = case.get("images", [])
    if not images:
        return None
    
    # Use first image
    first_image = images[0]
    # Use 'file' key (actual data) or 'file_name' for compatibility
    file_name = first_image.get("file") or first_image.get("file_name", "")
    # Images are in subdirectory by case_id
    case_id = case.get("case_id", "")
    image_path = images_dir / case_id / file_name
    
    if image_path.exists():
        return Query(
            case_id=case_id,
            query_type="Q3_image_only",
            image_path=str(image_path)
        )
    return None


def generate_all_queries(
    cases: List[Dict],
    images_dir: Path,
    query_types: List[str] = None
) -> List[Query]:
    """
    Generate all query types for test cases.
    
    Args:
        cases: List of case dictionaries
        images_dir: Path to images directory
        query_types: List of types to generate (default: all)
    
    Returns:
        List of Query objects
    """
    if query_types is None:
        query_types = ["Q1", "Q2", "Q3"]
    
    queries = []
    
    for case in cases:
        if "Q1" in query_types:
            queries.append(generate_q1_symptom_only(case))
        
        if "Q2" in query_types:
            queries.append(generate_q2_symptom_exposure(case))
        
        if "Q3" in query_types:
            q3 = generate_q3_image_only(case, images_dir)
            if q3:
                queries.append(q3)
    
    return queries


# =============================================================================
# STANDARDIZED QUERY GENERATION (v2.0 - for proper RAGAS evaluation)
# =============================================================================

@dataclass
class GroundTruth:
    """Ground truth for diagnosis accuracy evaluation."""
    diagnosis: str
    diagnosis_type: str  # CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, Other
    species: str = ""


@dataclass
class StandardizedEvalQuery:
    """Query formatted for RAGAS evaluation.
    
    CRITICAL: query_images are from the TEST case being queried,
    NOT from retrieved TRAIN cases (which go into context_images).
    """
    case_id: str
    query_type: str
    # The fixed question (for RAGAS query field)
    question: str
    # Cleaned clinical context (for RAGAS context field)  
    clinical_context: str
    # Images from TEST case (for multimodal query)
    query_images: List[str] = None
    # Ground truth for diagnosis accuracy
    ground_truth: GroundTruth = None
    # Full formatted query (for display/debugging)
    formatted_query: str = None
    # Legacy field (deprecated - use query_images)
    image_path: Optional[str] = None


def extract_ground_truth(case: Dict) -> GroundTruth:
    """
    Extract ground truth diagnosis information from a case.
    
    Expected case fields (from train.jsonl/test.jsonl):
    - diagnosis: Full diagnosis name
    - diagnosis_type: CL, VL, MCL, PKDL, DCL, DsCL, LCL, LR, Ocular, Veterinary, Non-Leishmaniasis, etc.
    - species: Leishmania species if available
    """
    diagnosis = case.get("diagnosis", "")
    diagnosis_type = case.get("diagnosis_type", "")
    species = case.get("species", "")
    
    # Normalize diagnosis_type to standard abbreviations
    type_mapping = {
        "cutaneous": "CL",
        "visceral": "VL", 
        "mucocutaneous": "MCL",
        "post-kala-azar": "PKDL",
        "post kala-azar": "PKDL",
        "pkdl": "PKDL",
    }
    normalized_type = type_mapping.get(diagnosis_type.lower(), diagnosis_type)
    
    return GroundTruth(
        diagnosis=diagnosis,
        diagnosis_type=normalized_type,
        species=species
    )


def extract_query_images(case: Dict, images_dir: Path) -> List[str]:
    """
    Extract all valid image paths from a TEST case.
    
    These are the images that belong TO THE QUERY (test case),
    not images from retrieved train cases.
    
    Returns:
        List of absolute image file paths that exist
    """
    query_images = []
    case_id = case.get("case_id", "")
    
    for img in case.get("images", []):
        filename = img.get("file") or img.get("file_name", "")
        if filename:
            img_path = images_dir / case_id / filename
            if img_path.exists():
                query_images.append(str(img_path))
    
    return query_images


def generate_standardized_q1(case: Dict) -> StandardizedEvalQuery:
    """
    Generate standardized Q1 query with fixed question.
    
    Key change: Query is now a fixed question, not raw case text.
    Clinical description goes into context, not query.
    """
    # Extract and clean clinical description
    demographics = extract_demographics(case)
    symptoms = extract_symptoms(case.get("case_text", ""))
    
    # This is the CONTEXT, not the query
    clinical_context = f"{demographics} with {symptoms}" if symptoms else demographics
    clinical_context = sanitize_case_text(clinical_context)
    
    # Build standardized query
    std_query = build_diagnosis_query(
        clinical_description=clinical_context,
        has_image=False
    )
    
    return StandardizedEvalQuery(
        case_id=case["case_id"],
        query_type="Q1_diagnosis",
        question=std_query.question_only,  # Fixed question
        clinical_context=clinical_context,  # Goes into context
        ground_truth=extract_ground_truth(case),  # For diagnosis accuracy
        formatted_query=std_query.query_text
    )


def generate_standardized_q2(case: Dict) -> StandardizedEvalQuery:
    """
    Generate standardized Q2 query with exposure info.
    """
    # Get Q1 base
    q1 = generate_standardized_q1(case)
    
    # Add exposure info to context
    exposure = extract_exposure_info(case.get("case_text", ""))
    clinical_context = q1.clinical_context
    if exposure:
        clinical_context += f" | Exposure: {sanitize_case_text(exposure)}"
    
    # Rebuild with updated context
    std_query = build_diagnosis_query(
        clinical_description=clinical_context,
        has_image=False
    )
    
    return StandardizedEvalQuery(
        case_id=case["case_id"],
        query_type="Q2_diagnosis_exposure",
        question=std_query.question_only,
        clinical_context=clinical_context,
        ground_truth=extract_ground_truth(case),  # For diagnosis accuracy
        formatted_query=std_query.query_text
    )


def generate_standardized_q3(case: Dict, images_dir: Path) -> Optional[StandardizedEvalQuery]:
    """
    Generate standardized Q3 (image-only) query.
    """
    images = case.get("images", [])
    if not images:
        return None
    
    first_image = images[0]
    file_name = first_image.get("file") or first_image.get("file_name", "")
    case_id = case.get("case_id", "")
    image_path = images_dir / case_id / file_name
    
    if not image_path.exists():
        return None
    
    std_query = build_diagnosis_query(
        clinical_description="",
        image_only=True
    )
    
    # Extract query images from TEST case
    query_images = extract_query_images(case, images_dir)
    
    return StandardizedEvalQuery(
        case_id=case_id,
        query_type="Q3_image_diagnosis",
        question=std_query.question_only,
        clinical_context="[Image only - no text context]",
        query_images=query_images,
        ground_truth=extract_ground_truth(case),
        image_path=str(image_path),  # Legacy
        formatted_query=std_query.query_text
    )


def generate_standardized_multimodal(case: Dict, images_dir: Path) -> Optional[StandardizedEvalQuery]:
    """
    Generate standardized Q1+Q3 combined multimodal query.
    """
    # Check for images
    images = case.get("images", [])
    if not images:
        return None
    
    first_image = images[0]
    file_name = first_image.get("file") or first_image.get("file_name", "")
    case_id = case.get("case_id", "")
    image_path = images_dir / case_id / file_name
    
    if not image_path.exists():
        return None
    
    # Get clinical context
    demographics = extract_demographics(case)
    symptoms = extract_symptoms(case.get("case_text", ""))
    clinical_context = f"{demographics} with {symptoms}" if symptoms else demographics
    clinical_context = sanitize_case_text(clinical_context)
    
    std_query = build_diagnosis_query(
        clinical_description=clinical_context,
        has_image=True
    )
    
    # Extract ALL query images from TEST case
    query_images = extract_query_images(case, images_dir)
    
    return StandardizedEvalQuery(
        case_id=case_id,
        query_type="Q1_Q3_multimodal_diagnosis",
        question=std_query.question_only,
        clinical_context=clinical_context,
        query_images=query_images,  # All images from TEST case
        ground_truth=extract_ground_truth(case),
        image_path=str(image_path),  # Legacy: first image only
        formatted_query=std_query.query_text
    )


def generate_standardized_queries(
    cases: List[Dict],
    images_dir: Path,
    query_types: List[str] = None
) -> List[StandardizedEvalQuery]:
    """
    Generate standardized queries for all cases.
    
    Args:
        cases: List of case dictionaries
        images_dir: Path to images directory
        query_types: Which types to generate (default: Q1 only)
    
    Returns:
        List of StandardizedEvalQuery objects with fixed question phrasing
    """
    if query_types is None:
        query_types = ["Q1"]  # Default to Q1 for stable metrics
    
    queries = []
    
    for case in cases:
        if "Q1" in query_types:
            queries.append(generate_standardized_q1(case))
        
        if "Q2" in query_types:
            queries.append(generate_standardized_q2(case))
        
        if "Q3" in query_types:
            q3 = generate_standardized_q3(case, images_dir)
            if q3:
                queries.append(q3)
        
        if "MULTIMODAL" in query_types:
            mm = generate_standardized_multimodal(case, images_dir)
            if mm:
                queries.append(mm)
    
    return queries


def load_and_generate(
    test_jsonl: Path,
    images_dir: Path,
    output_path: Optional[Path] = None,
    standardized: bool = True
) -> List:
    """
    Load test cases and generate queries.
    
    Args:
        test_jsonl: Path to test cases JSONL
        images_dir: Path to images directory
        output_path: Optional path to save queries
        standardized: If True, use new standardized format (v2.0)
    
    Returns:
        List of Query or StandardizedEvalQuery objects
    """
    # Load test cases
    with open(test_jsonl) as f:
        cases = [json.loads(line) for line in f]
    
    # Generate queries
    if standardized:
        queries = generate_standardized_queries(cases, images_dir, ["Q1", "Q3", "MULTIMODAL"])
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for q in queries:
                    # Serialize ground truth if present
                    ground_truth_dict = None
                    if q.ground_truth:
                        ground_truth_dict = {
                            "diagnosis": q.ground_truth.diagnosis,
                            "diagnosis_type": q.ground_truth.diagnosis_type,
                            "species": q.ground_truth.species
                        }
                    
                    f.write(json.dumps({
                        "case_id": q.case_id,
                        "query_type": q.query_type,
                        "question": q.question,
                        "clinical_context": q.clinical_context,
                        "query_images": q.query_images or [],  # TEST case images
                        "ground_truth": ground_truth_dict,
                        "image_path": q.image_path,  # Legacy
                        "formatted_query": q.formatted_query
                    }) + "\n")
    else:
        # Legacy behavior
        queries = generate_all_queries(cases, images_dir)
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                for q in queries:
                    f.write(json.dumps({
                        "case_id": q.case_id,
                        "query_type": q.query_type,
                        "text": q.text,
                        "image_path": q.image_path
                    }) + "\n")
    
    return queries


if __name__ == "__main__":
    from .config import TEST_JSONL, IMAGES_DIR, SPLIT_DIR, DATASET_VERSION
    
    queries = load_and_generate(
        TEST_JSONL,
        IMAGES_DIR,
        SPLIT_DIR / f"eval_queries_{DATASET_VERSION}.jsonl"
    )
    
    print(f"Generated {len(queries)} queries")
    for qt in ["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"]:
        count = sum(1 for q in queries if q.query_type == qt)
        print(f"  {qt}: {count}")
