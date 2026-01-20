"""
Answer Generator - Main Module

Orchestrates answer generation for RAG evaluation.
Uses separate generator modules:
- generators/gemini.py: API-based Gemini
- generators/medgemma.py: Local MedGemma

Creates answers.jsonl with:
- qid, query, contexts, answer, model_name, citations, image_paths, decoding_params
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

# Import from package (ensures HF cache is set)
from . import DATA_ROOT, TRAIN_JSONL, RUNS_DIR
from .generators import GeminiGenerator, MedGemmaGenerator


# Derive paths
IMAGES_DIR = DATA_ROOT / "images"


@dataclass
class AnswerRecord:
    """Single answer record for answers.jsonl
    
    CRITICAL (per GPT 5.2):
    - query_images: Images from TEST case (the query)
    - context_images: Images from TRAIN cases (retrieved contexts)
    """
    qid: str
    query: str
    contexts: List[Dict]  # From retriever, NOT from qrels
    answer: str
    model_name: str
    citations: List[str] = field(default_factory=list)
    # NEW: Separated per GPT 5.2
    query_images: List[str] = field(default_factory=list)  # TEST case
    context_images: List[str] = field(default_factory=list)  # TRAIN cases
    ground_truth: Optional[Dict] = None  # For diagnosis accuracy
    # Legacy (deprecated - use context_images)
    image_paths: List[str] = field(default_factory=list)
    decoding_params: Dict = field(default_factory=dict)


def enrich_contexts_with_images(
    contexts: List[Dict],
    train_cases: Dict
) -> tuple[List[Dict], List[str]]:
    """
    Add image paths to contexts from train_cases.
    
    IMPORTANT: Does NOT modify which contexts are selected.
    Only enriches existing retrieved contexts with image metadata.
    
    Returns:
        Updated contexts and list of all image paths
    """
    all_image_paths = []
    
    for ctx in contexts:
        doc_id = ctx.get("doc_id")
        if doc_id in train_cases:
            case = train_cases[doc_id]
            images = case.get("images", [])
            # Use 'file' key (per MultiCaRe dataset) with fallback to 'file_name'
            # Include case_id directory in path structure
            ctx_images = []
            for img in images:
                filename = img.get("file") or img.get("file_name", "")
                if filename:
                    # Images stored in: IMAGES_DIR / case_id / filename
                    ctx_images.append(str(IMAGES_DIR / doc_id / filename))
            ctx["image_paths"] = ctx_images
            all_image_paths.extend(ctx_images)
    
    return contexts, all_image_paths


def extract_citations(answer: str, contexts: List[Dict]) -> List[str]:
    """Extract cited case IDs from answer text."""
    return [
        ctx["doc_id"] for ctx in contexts
        if ctx.get("doc_id") and ctx["doc_id"] in answer
    ]


def generate_answers(
    run_dir: Path,
    retrieval_file: str = "retrieval.jsonl",
    generator_type: str = "gemini",  # "gemini" or "medgemma"
    output_file: str = None,  # Custom output filename (default: answers.jsonl)
    **generator_kwargs
) -> Path:
    """
    Generate answers for all queries in a run.
    
    SCIENTIFIC HYGIENE:
    - Uses ONLY contexts from retriever (retrieval.jsonl)
    - Does NOT consult qrels when selecting contexts
    - Image paths are metadata enrichment only
    
    Args:
        run_dir: Path to run directory
        retrieval_file: Retrieval JSONL from retriever
        generator_type: "gemini" or "medgemma"
        output_file: Output filename (default: answers.jsonl)
        **generator_kwargs: Args for generator
    
    Returns:
        Path to answers.jsonl
    """
    run_dir = Path(run_dir)
    
    # Load retrieval results (from retriever, NOT qrels)
    retrieval_path = run_dir / retrieval_file
    if not retrieval_path.exists():
        raise FileNotFoundError(f"No {retrieval_file} in {run_dir}")
    
    with open(retrieval_path) as f:
        samples = [json.loads(l) for l in f]
    
    # Load train cases for image enrichment only
    with open(TRAIN_JSONL) as f:
        train_cases = {}
        for line in f:
            case = json.loads(line)
            train_cases[case["case_id"]] = case
    
    # Initialize generator
    if generator_type == "medgemma":
        generator = MedGemmaGenerator(**generator_kwargs)
    else:
        generator = GeminiGenerator(**generator_kwargs)
    
    print(f"Generating answers for {len(samples)} queries with {generator.model_name}...")
    
    # Generate
    records = []
    for i, sample in enumerate(samples):
        qid = sample["qid"]
        query = sample["query"]
        contexts = sample.get("contexts", [])
        
        # Enrich with image paths (metadata only) - these are CONTEXT images (TRAIN)
        contexts, context_images = enrich_contexts_with_images(contexts, train_cases)
        
        # Query images are from the TEST case (passed in retrieval sample)
        query_images = sample.get("query_images", [])
        ground_truth = sample.get("ground_truth", None)
        
        # Log image loading status (per Claude 4.5 Fix 4)
        if query_images:
            valid_count = sum(1 for p in query_images if Path(p).exists())
            print(f"  [{qid}] query_images: {valid_count}/{len(query_images)} valid")
        
        # Generate answer - pass query_images for TRUE multimodal (per GPT 5.2 fix)
        # query_images = images from TEST case (the patient to diagnose)
        # context_images = images from TRAIN cases (retrieved evidence)
        answer = generator.generate(
            query, 
            contexts, 
            image_paths=query_images[:5]  # Send TEST case images to Gemini
        )
        
        # Extract citations
        citations = extract_citations(answer, contexts)
        
        record = AnswerRecord(
            qid=qid,
            query=query,
            contexts=contexts,
            answer=answer,
            model_name=generator.model_name,
            citations=citations,
            query_images=query_images[:5],      # TEST case images
            context_images=context_images[:5],   # TRAIN case images
            ground_truth=ground_truth,
            image_paths=query_images[:5],        # FIXED: was context_images (BUG!)
            decoding_params=generator.decoding_params
        )
        records.append(asdict(record))
        
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")
    
    # Save
    output_name = output_file or "answers.jsonl"
    output_path = run_dir / output_name
    with open(output_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    
    print(f"✓ Generated {len(records)} answers to {output_path}")
    
    return output_path


if __name__ == "__main__":
    # Generate for phase3 run
    run_dir = RUNS_DIR / "phase3_hybrid"
    if run_dir.exists():
        generate_answers(run_dir, generator_type="gemini")
