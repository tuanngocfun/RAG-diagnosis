#!/usr/bin/env python3
"""
Quick test script for RAG prompt fix validation.
Tests with 10 samples to compare augmented RAG vs No-RAG.
"""
import json
import sys
from pathlib import Path
from datetime import datetime

# Add pipeline to path
sys.path.insert(0, str(Path(__file__).parent))

from pipeline.config import TEST_JSONL, RUNS_DIR, IMAGES_DIR
from pipeline.generators.gemini import GeminiGenerator, build_rag_prompt
from pipeline.run_baseline_norag import build_norag_prompt
from pipeline.query_generator import (
    generate_standardized_q1,
    extract_ground_truth,
    extract_query_images,
)


def run_quick_test(sample_size: int = 10):
    """Run quick A/B test: RAG (augmented) vs No-RAG on small sample."""
    
    # Load test cases
    with open(TEST_JSONL) as f:
        test_cases = [json.loads(line) for line in f][:sample_size]
    
    print(f"Testing with {len(test_cases)} samples")
    print("=" * 60)
    
    # Initialize generator
    generator = GeminiGenerator()
    
    # Store results
    rag_results = []
    norag_results = []
    
    for i, case in enumerate(test_cases):
        case_id = case["case_id"]
        print(f"\n[{i+1}/{len(test_cases)}] Processing {case_id}...")
        
        # Generate Q1 query
        query = generate_standardized_q1(case)
        ground_truth = extract_ground_truth(case)
        query_images = extract_query_images(case, IMAGES_DIR)
        
        clinical_context = query.clinical_context
        question = query.question
        
        # --- No-RAG ---
        norag_prompt = build_norag_prompt(clinical_context, question, has_images=len(query_images) > 0)
        norag_answer = generator.generate(
            norag_prompt, 
            contexts=[],  # No contexts
            image_paths=query_images[:5],
            use_rag_prompt=False  # Use query directly
        )
        
        norag_results.append({
            "case_id": case_id,
            "answer": norag_answer,
            "ground_truth": ground_truth.__dict__ if ground_truth else None
        })
        
        # --- RAG (with mock contexts for testing) ---
        # Load retrieval contexts from existing run
        existing_retrieval = RUNS_DIR / "multimodal_rag_full" / "retrieval.jsonl"
        contexts = []
        if existing_retrieval.exists():
            with open(existing_retrieval) as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("qid", "").startswith(case_id):
                        contexts = data.get("contexts", [])[:5]
                        break
        
        rag_answer = generator.generate(
            query.formatted_query,
            contexts=contexts,
            image_paths=query_images[:5],
            use_rag_prompt=True  # Use augmentation prompt
        )
        
        rag_results.append({
            "case_id": case_id,
            "answer": rag_answer,
            "contexts_count": len(contexts),
            "ground_truth": ground_truth.__dict__ if ground_truth else None
        })
        
        # Quick preview
        print(f"  Ground truth: {ground_truth.diagnosis if ground_truth else 'N/A'}")
        print(f"  NoRAG first 100 chars: {norag_answer[:100]}...")
        print(f"  RAG first 100 chars: {rag_answer[:100]}...")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = RUNS_DIR / f"prompt_fix_quick_test_{timestamp}"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "norag_answers.jsonl", "w") as f:
        for r in norag_results:
            f.write(json.dumps(r) + "\n")
    
    with open(output_dir / "rag_answers.jsonl", "w") as f:
        for r in rag_results:
            f.write(json.dumps(r) + "\n")
    
    print(f"\n{'=' * 60}")
    print(f"Results saved to: {output_dir}")
    print(f"NoRAG: {len(norag_results)} answers")
    print(f"RAG: {len(rag_results)} answers")
    
    # Quick check for "Insufficient evidence" in RAG answers
    insufficient_count = sum(1 for r in rag_results if "Insufficient evidence" in r["answer"])
    print(f"\nRAG 'Insufficient evidence' count: {insufficient_count}/{len(rag_results)}")
    
    return output_dir


if __name__ == "__main__":
    sample_size = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    run_quick_test(sample_size)
