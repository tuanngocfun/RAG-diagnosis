#!/usr/bin/env python3
"""
No-RAG Baseline Generator

Generates diagnosis answers using ONLY the query (clinical context + images)
without any retrieved contexts. This serves as a baseline to compare against
RAG-enhanced generation.

Comparison purpose:
- RAG system: Query → Retrieve similar cases → Generate with evidence
- No-RAG baseline: Query → Generate directly from LLM parametric knowledge

Usage:
    python -m rag.pipeline.run_baseline_norag --run-id baseline_norag_v1
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from .config import SPLIT_DIR, RUNS_DIR, TEST_JSONL, IMAGES_DIR, DATASET_VERSION
from .generators import GeminiGenerator, Gemma3Generator, MedGemmaGenerator


@dataclass
class BaselineRunConfig:
    """Configuration for no-RAG baseline run."""
    run_id: str
    queries_file: str
    query_types: List[str]
    generator_model: str
    created_at: str
    is_rag: bool = False  # Flag to distinguish from RAG runs


@dataclass
class BaselineResult:
    """Result from no-RAG baseline generation."""
    qid: str
    query_type: str
    query: str
    clinical_context: str
    query_images: List[str]
    answer: str
    model_name: str
    ground_truth: Optional[Dict] = None
    # Empty contexts to distinguish from RAG
    contexts: List[Dict] = field(default_factory=list)
    context_images: List[str] = field(default_factory=list)


def build_norag_prompt(
    clinical_context: str,
    question: str,
    has_images: bool = False
) -> str:
    """
    Build prompt for no-RAG baseline - ALIGNED with RAG prompt structure.
    
    Per expert review (GPT 5.2, Gemini 3 Pro, Q1 papers):
    - Same structure as build_rag_prompt in gemini.py
    - Only difference: knowledge source (parametric vs retrieved)
    - Same output format ensures fair comparison
    """
    image_section = ""
    if has_images:
        image_section = "\n## PATIENT IMAGES\n[Clinical images attached - examine for diagnosis]\n"
    
    return f"""You are an AI research assistant helping with an academic thesis on leishmaniasis case reports.

IMPORTANT CONTEXT:
- This is strictly for RESEARCH and EVALUATION purposes
- The data consists of de-identified case reports from PubMed Central (PMC)
- Do NOT provide medical advice or treatment recommendations

RESEARCH QUERY:
{clinical_context}

{question}
{image_section}
## INSTRUCTION (PARAMETRIC KNOWLEDGE)
Base your diagnosis on your medical knowledge and training.
Provide your best clinical assessment. If uncertain, state your confidence level and list differentials.

TASK:
Provide a structured diagnosis assessment.

REQUIRED OUTPUT FORMAT:

## DIAGNOSIS PREDICTION
**Primary Diagnosis:** [Your diagnosis, e.g., "Cutaneous Leishmaniasis", "Visceral Leishmaniasis", "PKDL", "Mucocutaneous Leishmaniasis"]
**Diagnosis Type:** [CL, VL, MCL, PKDL, Other]
**Species (if determinable):** [e.g., "L. donovani", "L. tropica", "L. major", or "Not determinable"]
**Confidence:** [High/Medium/Low]

## SUPPORTING EVIDENCE
- Key clinical findings supporting your diagnosis
- Reasoning based on symptoms and presentation

## DIFFERENTIAL CONSIDERATIONS
- Alternative diagnoses to consider

DIAGNOSIS ASSESSMENT:"""


def run_baseline_norag(
    query_types: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    generator_model: str = None,
    generator_type: str = "gemini",  # "gemini", "gemma3", or "medgemma"
    model_variant: str = "12b",  # For Gemma3: "12b" or "27b" (ignored for medgemma)
) -> Path:
    """
    Run no-RAG baseline generation.
    
    Generates answers using ONLY query information (no retrieved contexts).
    This provides a fair comparison baseline for the RAG system.
    
    Args:
        query_types: Query types to evaluate (default: all multimodal)
        run_id: Run identifier (auto-generated if None)
        generator_model: Gemini model to use
    
    Returns:
        Path to run directory
    """
    # Defaults
    if query_types is None:
        query_types = ["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"]
    
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"baseline_norag_{timestamp}"
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")
    print(f"NO-RAG BASELINE GENERATION: {run_id}")
    print(f"{'='*60}")
    
    # Load queries from SPLIT_DIR
    queries_file = SPLIT_DIR / f"eval_queries_{DATASET_VERSION}.jsonl"
    if not queries_file.exists():
        queries_file = SPLIT_DIR / "eval_queries.jsonl"
    
    print(f"Loading queries from: {queries_file}")
    with open(queries_file) as f:
        all_queries = [json.loads(l) for l in f]
    
    # Filter by type
    queries = [q for q in all_queries if q["query_type"] in query_types]
    print(f"Query types: {query_types}")
    print(f"Total queries: {len(queries)}")
    
    # Initialize generator based on type
    if generator_type == "gemma3":
        generator = Gemma3Generator(variant=model_variant)
        print(f"Generator: Gemma 3 {model_variant} (local)")
    elif generator_type == "medgemma":
        generator = MedGemmaGenerator()
        print(f"Generator: MedGemma 4B (local, text-only)")
    else:
        generator = GeminiGenerator(model=generator_model, include_images=True)
        print(f"Generator: {generator.model_name}")
    
    # Save config
    config = BaselineRunConfig(
        run_id=run_id,
        queries_file=str(queries_file),
        query_types=query_types,
        generator_model=generator.model_name,
        created_at=datetime.now().isoformat(),
        is_rag=False
    )
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Save queries
    with open(run_dir / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    # Generate answers WITHOUT retrieval
    results = []
    stats = {"attempted": 0, "success": 0, "errors": 0}
    
    for i, q in enumerate(queries):
        qid = f"{q['case_id']}::{q['query_type']}"
        clinical_context = q.get("clinical_context", "")
        question = q.get("question", "What is the diagnosis?")
        query_images = q.get("query_images", [])
        ground_truth = q.get("ground_truth")
        
        stats["attempted"] += 1
        
        # Build no-RAG prompt (no contexts!)
        prompt = build_norag_prompt(
            clinical_context=clinical_context,
            question=question,
            has_images=len(query_images) > 0
        )
        
        try:
            # Generate with images but NO contexts, bypassing RAG faithfulness prompt
            answer = generator.generate(
                query=prompt,
                contexts=[],  # NO CONTEXTS - this is the key difference!
                image_paths=query_images[:5] if query_images else None,
                use_rag_prompt=False  # CRITICAL: bypass RAG faithfulness constraint
            )
            stats["success"] += 1
        except Exception as e:
            print(f"  Error on {qid}: {e}")
            answer = f"[ERROR: {str(e)[:100]}]"
            stats["errors"] += 1
        
        result = BaselineResult(
            qid=qid,
            query_type=q["query_type"],
            query=prompt,
            clinical_context=clinical_context,
            query_images=query_images,
            answer=answer,
            model_name=generator.model_name,
            ground_truth=ground_truth,
            contexts=[],  # EMPTY - no retrieval
            context_images=[]  # EMPTY - no retrieval
        )
        results.append(asdict(result))
        
        if (i + 1) % 5 == 0:
            print(f"  Processed {i + 1}/{len(queries)}")
        
        # Rate limiting
        time.sleep(1)
    
    # Save results
    with open(run_dir / "answers_norag.jsonl", "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Also save as retrieval.jsonl format for compatibility with RAGAS evaluator
    with open(run_dir / "retrieval.jsonl", "w") as f:
        for r in results:
            # Map to retrieval format expected by ragas_evaluator
            retrieval_record = {
                "qid": r["qid"],
                "query_type": r["query_type"],
                "query": r["clinical_context"],  # Use clinical context as query
                "clinical_context": r["clinical_context"],
                "contexts": [],  # NO CONTEXTS
                "query_images": r["query_images"],
                "context_images": [],
                "ground_truth": r["ground_truth"],
                "stage": "norag_baseline"
            }
            f.write(json.dumps(retrieval_record) + "\n")
    
    # Save answers format for RAGAS
    with open(run_dir / "answers_gemini.jsonl", "w") as f:
        for r in results:
            answer_record = {
                "qid": r["qid"],
                "query": r["clinical_context"],
                "contexts": [],
                "answer": r["answer"],
                "model_name": r["model_name"],
                "query_images": r["query_images"],
                "context_images": [],
                "ground_truth": r["ground_truth"]
            }
            f.write(json.dumps(answer_record) + "\n")
    
    # Save summary
    summary = {
        "run_id": run_id,
        "is_rag": False,
        "query_types": query_types,
        "n_queries": len(queries),
        "stats": stats,
        "generator": generator.model_name,
        "note": "No-RAG baseline - answers generated WITHOUT retrieved contexts"
    }
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"NO-RAG BASELINE COMPLETE")
    print(f"{'='*60}")
    print(f"Queries: {stats['attempted']}")
    print(f"Success: {stats['success']}")
    print(f"Errors: {stats['errors']}")
    print(f"Output: {run_dir}")
    print(f"{'='*60}")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="No-RAG Baseline Generator")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--query-types", nargs="+", 
                        default=["Q1_diagnosis", "Q3_image_diagnosis", "Q1_Q3_multimodal_diagnosis"],
                        help="Query types to evaluate")
    parser.add_argument("--model", default=None, help="Gemini model")
    
    args = parser.parse_args()
    
    run_baseline_norag(
        query_types=args.query_types,
        run_id=args.run_id,
        generator_model=args.model
    )
