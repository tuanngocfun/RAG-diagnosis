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
    PYTHONPATH="$PROJECT_ROOT/codes" python -m pipeline.run_baseline_norag --run-id baseline_norag_v1
"""
import json
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict, field

from .config import (
    DATASET_VERSION,
    IMAGES_DIR,
    RUNS_DIR,
    SILVER_LABEL_DISCLAIMER,
    SPLIT_DIR,
    TEST_JSONL,
    get_dataset_artifact_filenames,
    get_dataset_support_snapshot,
    get_runtime_metadata,
)
from .generators import GeminiGenerator, Gemma3Generator, Gemma4Generator, MedGemmaGenerator
from .query_templates import DIAGNOSIS_QUESTION_WITH_TYPE
from .image_resolver import normalize_query_image_paths, resolve_case_image_paths
from .pseudolabel_adapter import build_pseudolabel_artifacts
from configs.prompt_mode import PromptMode, build_rag_prompt as build_prompt_contract


MATCHED_NORAG_CONTROL_TYPE = "matched_norag"
MATCHED_NORAG_PROMPT_CONTRACT_VERSION = "norag_matched_v2_query_guardrail"
MATCHED_NORAG_PROMPT_CONTRACT_NOTES = "query_only_definitive_evidence_guardrail"


@dataclass
class BaselineRunConfig:
    """Configuration for no-RAG baseline run."""
    run_id: str
    queries_file: str
    query_types: List[str]
    generator_model: str
    created_at: str
    is_rag: bool = False  # Flag to distinguish from RAG runs
    control_type: str = MATCHED_NORAG_CONTROL_TYPE
    prompt_mode: str = PromptMode.NO_CONTEXT.value
    prompt_contract_version: str = MATCHED_NORAG_PROMPT_CONTRACT_VERSION
    prompt_contract_notes: str = MATCHED_NORAG_PROMPT_CONTRACT_NOTES
    ablation_scope: str = ""


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
    ground_truth_pseudolabel: Optional[Dict] = None
    ablation_scope: str = ""
    query_images_stripped: bool = False
    # Empty contexts to distinguish from RAG
    contexts: List[Dict] = field(default_factory=list)
    context_images: List[str] = field(default_factory=list)


def _resolve_queries_file(dataset_pack: str, explicit_queries_file: Optional[str]) -> Path:
    """Resolve evaluation queries file for no-RAG runs."""
    artifact_names = get_dataset_artifact_filenames(DATASET_VERSION)
    if explicit_queries_file:
        qpath = Path(explicit_queries_file)
        if not qpath.is_absolute():
            qpath = SPLIT_DIR / explicit_queries_file
        return qpath

    if dataset_pack == "mixed56":
        return SPLIT_DIR / artifact_names["query_mixed56"]
    if dataset_pack == "test":
        return SPLIT_DIR / "eval_queries_v163.jsonl"

    # auto/default behavior (legacy-compatible)
    qpath = SPLIT_DIR / artifact_names["query"]
    if qpath.exists():
        return qpath
    return SPLIT_DIR / "eval_queries.jsonl"


def build_norag_prompt(
    query_text: str,
    query_images: Optional[List[str]] = None,
) -> str:
    """
    Build prompt for the matched no-RAG control using the shared no-context contract.
    """
    return build_prompt_contract(
        query=query_text,
        contexts=[],
        mode=PromptMode.NO_CONTEXT,
        query_images=query_images or None,
        context_images=None,
        include_context_images=False,
        is_text_only_model=False,
    )


def run_baseline_norag(
    query_types: Optional[List[str]] = None,
    run_id: Optional[str] = None,
    generator_model: str = None,
    generator_type: str = "gemini",  # "gemini", "gemma3", "gemma4", or "medgemma"
    model_variant: str = "12b",  # For Gemma3: "12b" or "27b" (ignored for medgemma)
    random_seed: Optional[int] = 42,
    dataset_pack: str = "auto",
    queries_file: Optional[str] = None,
    use_batch_api: bool = False,
    batch_poll_seconds: float = 10.0,
    batch_timeout_seconds: int = 3600,
    evaluate: bool = True,
    judge_model: Optional[str] = None,
    judge_batch_api: bool = False,
    judge_batch_poll_seconds: float = 10.0,
    judge_batch_timeout_seconds: int = 7200,
    pseudolabel_train_results: Optional[str] = None,
    pseudolabel_test_results: Optional[str] = None,
    pseudolabel_suffix: str = "",
    pseudolabel_force: bool = False,
    strip_query_images: bool = False,
    ablation_scope: str = "",
) -> Path:
    """
    Run no-RAG baseline generation.
    
    Generates answers using ONLY query information (no retrieved contexts).
    This provides a fair comparison baseline for the RAG system.
    
    Args:
        query_types: Query types to evaluate (default: Q1_Q3 multimodal only)
        run_id: Run identifier (auto-generated if None)
        generator_model: Gemini model to use
    
    Returns:
        Path to run directory
    """
    # Defaults
    if query_types is None:
        query_types = ["Q1_Q3_multimodal_diagnosis"]
    
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"baseline_norag_{timestamp}"
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"{'='*60}")

    pseudolabel_stats = None
    try:
        pseudolabel_stats = build_pseudolabel_artifacts(
            force=pseudolabel_force,
            train_results_path=pseudolabel_train_results,
            test_results_path=pseudolabel_test_results,
            output_suffix=pseudolabel_suffix,
            dataset_version=DATASET_VERSION,
        )
        print(
            "Prepared pseudolabel artifacts: "
            f"train={pseudolabel_stats.train_rows}, "
            f"test={pseudolabel_stats.test_rows}, "
            f"queries={pseudolabel_stats.query_rows}, "
            f"mixed56_queries={pseudolabel_stats.query_mixed56_rows}"
        )
    except Exception as e:
        print(f"Warning: Could not refresh pseudolabel artifacts: {e}")
    print(f"NO-RAG BASELINE GENERATION: {run_id}")
    print(f"{'='*60}")
    if strip_query_images:
        if not ablation_scope:
            ablation_scope = "generator_only_image_strip"
        print(
            "Ablation mode enabled: generator-side query image stripping only; "
            f"ablation_scope={ablation_scope}"
        )

    if pseudolabel_stats and pseudolabel_suffix and not queries_file:
        if dataset_pack == "mixed56":
            queries_file = pseudolabel_stats.query_mixed56_path
        elif dataset_pack == "auto":
            queries_file = pseudolabel_stats.query_path
    
    # Load queries from SPLIT_DIR
    queries_file = _resolve_queries_file(dataset_pack, queries_file)
    if not queries_file.exists():
        raise FileNotFoundError(f"Queries file not found: {queries_file}")
    
    print(f"Loading queries from: {queries_file}")
    with open(queries_file) as f:
        all_queries = [json.loads(l) for l in f]
    
    # Filter by type
    queries = [q for q in all_queries if q["query_type"] in query_types]
    print(f"Query types: {query_types}")
    print(f"Total queries: {len(queries)}")
    
    # Initialize generator based on type
    # Check if we have image queries (same logic as answer_generator.py for parity)
    has_image_queries = any(
        "Q3" in q.get("query_type", "") or "multimodal" in q.get("query_type", "").lower()
        for q in queries
    )
    if strip_query_images:
        has_image_queries = False
    
    if generator_type == "gemma3":
        model_path = generator_model
        generator = Gemma3Generator(
            variant=model_variant,
            use_vision=has_image_queries,
            random_seed=random_seed,
            prompt_mode=PromptMode.NO_CONTEXT,
            model_path=model_path,
        )
        print(
            f"Generator: {generator.model_name} "
            f"(engine={generator_type}, vision={has_image_queries})"
        )
    elif generator_type == "gemma4":
        model_path = generator_model or "google/gemma-4-E4B-it"
        generator = Gemma4Generator(
            use_vision=has_image_queries,
            random_seed=random_seed,
            prompt_mode=PromptMode.NO_CONTEXT,
            model_path=model_path,
        )
        print(
            f"Generator: {generator.model_name} "
            f"(engine=gemma4, vision={has_image_queries})"
        )
    elif generator_type == "medgemma":
        generator = MedGemmaGenerator(
            model_path=generator_model,
            use_vision=has_image_queries,
            prompt_mode=PromptMode.NO_CONTEXT,
        )
        print(f"Generator: {generator.model_name} (engine=medgemma, vision={has_image_queries})")
    else:
        generator = GeminiGenerator(
            model=generator_model,
            include_images=True,
            prompt_mode=PromptMode.NO_CONTEXT,
            batch_poll_seconds=batch_poll_seconds,
            batch_timeout_seconds=batch_timeout_seconds,
        )
        print(f"Generator: {generator.model_name}")
    
    # Save config
    runtime_metadata = get_runtime_metadata()
    dataset_support = get_dataset_support_snapshot()
    config = BaselineRunConfig(
        run_id=run_id,
        queries_file=str(queries_file),
        query_types=query_types,
        generator_model=generator.model_name,
        created_at=datetime.now().isoformat(),
        is_rag=False,
        ablation_scope=ablation_scope,
    )
    run_config_payload = asdict(config)
    run_config_payload["runtime_metadata"] = runtime_metadata
    run_config_payload["corpus_support"] = dataset_support
    if pseudolabel_stats is not None:
        run_config_payload["pseudolabel_artifacts"] = {
            "dataset_version": pseudolabel_stats.dataset_version,
            "train_source": pseudolabel_stats.train_source,
            "test_source": pseudolabel_stats.test_source,
            "suffix": pseudolabel_stats.suffix,
            "output_dir": pseudolabel_stats.output_dir,
            "train_path": pseudolabel_stats.train_path,
            "test_path": pseudolabel_stats.test_path,
            "query_path": pseudolabel_stats.query_path,
            "query_mixed56_path": pseudolabel_stats.query_mixed56_path,
            "qrels_verified_path": pseudolabel_stats.qrels_verified_path,
            "qrels_pseudolabel_path": pseudolabel_stats.qrels_pseudolabel_path,
        }
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(run_config_payload, f, indent=2)
    
    test_cases = {}
    if TEST_JSONL.exists():
        with open(TEST_JSONL) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                test_cases[obj["case_id"]] = obj
    else:
        print(f"Warning: canonical test file not found for image resolution: {TEST_JSONL}")

    normalized_queries = []
    for q in queries:
        case_id = q["case_id"]
        test_case = test_cases.get(case_id)
        query_images = resolve_case_image_paths(test_case, max_images=5)
        if not query_images:
            fallback_inputs = list(q.get("query_images") or [])
            if q.get("image_path"):
                fallback_inputs.append(q["image_path"])
            query_images = normalize_query_image_paths(
                case_id=case_id,
                query_images=fallback_inputs,
                images_dir=IMAGES_DIR,
                max_images=5,
            )
        normalized_q = dict(q)
        if strip_query_images:
            normalized_q["query_images"] = []
            normalized_q["image_path"] = None
            normalized_q["query_images_stripped"] = bool(query_images)
        else:
            normalized_q["query_images"] = query_images
            normalized_q["image_path"] = query_images[0] if query_images else None
            normalized_q["query_images_stripped"] = False
        normalized_queries.append(normalized_q)
    queries = normalized_queries

    # Save queries
    with open(run_dir / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    # Generate answers WITHOUT retrieval
    results = []
    stats = {"attempted": 0, "success": 0, "errors": 0}

    prompts = []
    
    for i, q in enumerate(queries):
        qid = f"{q['case_id']}::{q['query_type']}"
        clinical_context = q.get("clinical_context", "")
        question = q.get("question") or DIAGNOSIS_QUESTION_WITH_TYPE
        query_text = q.get("formatted_query") or f"{question}\n\nClinical Context: {clinical_context}"
        query_images = q.get("query_images", [])
        query_images_stripped = bool(q.get("query_images_stripped", False))
        ground_truth = q.get("ground_truth")
        ground_truth_pseudolabel = q.get("ground_truth_pseudolabel")
        
        stats["attempted"] += 1
        
        # Build matched no-RAG prompt (same contract as RAG, minus retrieved contexts)
        prompt = build_norag_prompt(query_text=query_text, query_images=query_images)
        prompts.append(
            {
                "qid": qid,
                "query_type": q["query_type"],
                "query": query_text,
                "generation_prompt": prompt,
                "clinical_context": clinical_context,
                "query_images": query_images,
                "query_images_stripped": query_images_stripped,
                "ablation_scope": ablation_scope,
                "ground_truth": ground_truth,
                "ground_truth_pseudolabel": ground_truth_pseudolabel,
            }
        )

    answers_by_qid = {}
    if generator_type == "gemini" and use_batch_api and hasattr(generator, "generate_batch"):
        print("Using Gemini Batch API for no-RAG generation")
        batch_inputs = [
            {
                "qid": p["qid"],
                "query": p["generation_prompt"],
                "contexts": [],
                "image_paths": p["query_images"][:5] if p["query_images"] else None,
                "use_rag_prompt": False,
            }
            for p in prompts
        ]
        batch_results = generator.generate_batch(
            batch_inputs,
            progress=True,
            use_batch_api=True,
            poll_seconds=batch_poll_seconds,
            timeout_seconds=batch_timeout_seconds,
        )
        for br in batch_results:
            qid = br.get("qid")
            ans = br.get("answer", "")
            answers_by_qid[qid] = ans
            if ans.startswith("[ERROR"):
                stats["errors"] += 1
            else:
                stats["success"] += 1
    else:
        for i, p in enumerate(prompts):
            try:
                answer = generator.generate(
                    query=p["generation_prompt"],
                    contexts=[],
                    image_paths=p["query_images"][:5] if p["query_images"] else None,
                    use_rag_prompt=False,
                )
                stats["success"] += 1
            except Exception as e:
                print(f"  Error on {p['qid']}: {e}")
                answer = f"[ERROR: {str(e)[:100]}]"
                stats["errors"] += 1
            answers_by_qid[p["qid"]] = answer
            if (i + 1) % 5 == 0:
                print(f"  Processed {i + 1}/{len(prompts)}")
            time.sleep(1)

    for p in prompts:
        result = BaselineResult(
            qid=p["qid"],
            query_type=p["query_type"],
            query=p["query"],
            clinical_context=p["clinical_context"],
            query_images=p["query_images"],
            answer=answers_by_qid.get(p["qid"], ""),
            model_name=generator.model_name,
            ground_truth=p["ground_truth"],
            ground_truth_pseudolabel=p["ground_truth_pseudolabel"],
            ablation_scope=p.get("ablation_scope", ""),
            query_images_stripped=bool(p.get("query_images_stripped", False)),
            contexts=[],
            context_images=[],
        )
        results.append(asdict(result))
    
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
                "query": r["query"],
                "clinical_context": r["clinical_context"],
                "contexts": [],  # NO CONTEXTS
                "query_images": r["query_images"],
                "context_images": [],
                "ground_truth": r["ground_truth"],
                "ground_truth_pseudolabel": r.get("ground_truth_pseudolabel"),
                "stage": "norag_baseline",
                "control_type": MATCHED_NORAG_CONTROL_TYPE,
                "prompt_mode": PromptMode.NO_CONTEXT.value,
                "prompt_contract_version": MATCHED_NORAG_PROMPT_CONTRACT_VERSION,
                "ablation_scope": r.get("ablation_scope", ""),
                "query_images_stripped": bool(r.get("query_images_stripped", False)),
            }
            f.write(json.dumps(retrieval_record) + "\n")
    
    # Save answers format for RAGAS
    with open(run_dir / "answers_gemini.jsonl", "w") as f:
        for r in results:
            answer_record = {
                "qid": r["qid"],
                "query": r["query"],
                "contexts": [],
                "answer": r["answer"],
                "model_name": r["model_name"],
                "query_images": r["query_images"],
                "context_images": [],
                "ground_truth": r["ground_truth"],
                "ground_truth_pseudolabel": r.get("ground_truth_pseudolabel"),
                "generation_mode": "norag_prompt",
                "retrieval_support_status": "no_rag_baseline",
                "control_type": MATCHED_NORAG_CONTROL_TYPE,
                "prompt_mode": PromptMode.NO_CONTEXT.value,
                "prompt_contract_version": MATCHED_NORAG_PROMPT_CONTRACT_VERSION,
                "ablation_scope": r.get("ablation_scope", ""),
                "query_images_stripped": bool(r.get("query_images_stripped", False)),
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
        "runtime_metadata": runtime_metadata,
        "corpus_support": dataset_support,
        "note": "No-RAG baseline - answers generated WITHOUT retrieved contexts",
        "control_type": MATCHED_NORAG_CONTROL_TYPE,
        "prompt_mode": PromptMode.NO_CONTEXT.value,
        "prompt_contract_version": MATCHED_NORAG_PROMPT_CONTRACT_VERSION,
        "prompt_contract_notes": MATCHED_NORAG_PROMPT_CONTRACT_NOTES,
        "ablation_scope": ablation_scope,
        "strip_query_images": strip_query_images,
        "pseudolabel_suffix": pseudolabel_suffix,
        "silver_label_disclaimer": SILVER_LABEL_DISCLAIMER,
    }
    if pseudolabel_stats is not None:
        summary["pseudolabel_artifacts"] = {
            "dataset_version": pseudolabel_stats.dataset_version,
            "train_source": pseudolabel_stats.train_source,
            "test_source": pseudolabel_stats.test_source,
            "suffix": pseudolabel_stats.suffix,
            "output_dir": pseudolabel_stats.output_dir,
            "query_path": pseudolabel_stats.query_path,
            "query_mixed56_path": pseudolabel_stats.query_mixed56_path,
            "qrels_verified_path": pseudolabel_stats.qrels_verified_path,
            "qrels_pseudolabel_path": pseudolabel_stats.qrels_pseudolabel_path,
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

    if evaluate:
        print("\nRunning phased RAGAS evaluation...")
        from .ragas_evaluator import run_ragas_evaluation
        run_ragas_evaluation(
            run_dir,
            answers_file="answers_gemini.jsonl",
            judge_model=judge_model,
            delay_seconds=1.5,
            diagnosis_batch_api=judge_batch_api,
            diagnosis_batch_poll_seconds=judge_batch_poll_seconds,
            diagnosis_batch_timeout_seconds=judge_batch_timeout_seconds,
        )

    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="No-RAG Baseline Generator")
    parser.add_argument("--run-id", default=None, help="Run identifier")
    parser.add_argument("--query-types", nargs="+", 
                        default=["Q1_Q3_multimodal_diagnosis"],
                        help="Query types to evaluate (default: Q1_Q3 multimodal only)")
    parser.add_argument("--generator", default="gemini", choices=["gemini", "gemma3", "gemma4", "medgemma"],
                        help="Generator type: gemini (default), gemma3, gemma4, medgemma")
    parser.add_argument("--variant", default="12b", help="Model variant (gemma3: 4b/12b/27b, gemma4: 4b)")
    parser.add_argument("--model", default=None,
                        help="Explicit model ID: Gemini model for --generator gemini, or local HF model ID for gemma3/gemma4/medgemma")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stochastic generators")
    parser.add_argument("--judge-model", default=None, help="Optional judge model for RAGAS evaluation")
    parser.add_argument("--dataset-pack", choices=["auto", "test", "mixed56"], default="auto",
                        help="Select predefined eval query pack")
    parser.add_argument("--queries-file", default=None,
                        help="Optional explicit eval queries JSONL path (overrides dataset-pack)")
    parser.add_argument("--pseudolabel-train-results", default=None,
                        help="Optional override train pseudolabel results.jsonl source path")
    parser.add_argument("--pseudolabel-test-results", default=None,
                        help="Optional override test pseudolabel results.jsonl source path")
    parser.add_argument("--pseudolabel-suffix", default="",
                        help="Optional suffix for versioned pseudolabel artifacts")
    parser.add_argument("--pseudolabel-force", action="store_true",
                        help="Force pseudolabel artifact rebuild even when outputs are up to date")
    parser.add_argument("--batch-api", action="store_true",
                        help="Use Google Batch API for Gemini generation")
    parser.add_argument("--batch-poll-seconds", type=float, default=10.0,
                        help="Polling interval for Gemini batch jobs")
    parser.add_argument("--batch-timeout-seconds", type=int, default=3600,
                        help="Timeout for Gemini batch jobs")
    parser.add_argument("--judge-batch-api", action="store_true",
                        help="Use Gemini Batch API for diagnosis judge evaluation")
    parser.add_argument("--judge-batch-poll-seconds", type=float, default=10.0,
                        help="Polling interval for diagnosis judge batch jobs")
    parser.add_argument("--judge-batch-timeout-seconds", type=int, default=7200,
                        help="Timeout for diagnosis judge batch jobs")
    parser.add_argument("--strip-query-images", action="store_true",
                        help="Generator-side hard image-off ablation: remove query_images before generation")
    parser.add_argument("--ablation-scope", default="",
                        help="Optional ablation scope label stored in generated artifacts")
    parser.add_argument("--no-eval", dest="evaluate", action="store_false",
                        help="Generate only, skip phased RAGAS evaluation")
    parser.set_defaults(evaluate=True)
    
    args = parser.parse_args()
    
    run_dir = run_baseline_norag(
        query_types=args.query_types,
        run_id=args.run_id,
        generator_model=args.model,
        generator_type=args.generator,
        model_variant=args.variant,
        random_seed=args.seed,
        dataset_pack=args.dataset_pack,
        queries_file=args.queries_file,
        use_batch_api=args.batch_api,
        batch_poll_seconds=args.batch_poll_seconds,
        batch_timeout_seconds=args.batch_timeout_seconds,
        evaluate=args.evaluate,
        judge_model=args.judge_model,
        judge_batch_api=args.judge_batch_api,
        judge_batch_poll_seconds=args.judge_batch_poll_seconds,
        judge_batch_timeout_seconds=args.judge_batch_timeout_seconds,
        pseudolabel_train_results=args.pseudolabel_train_results,
        pseudolabel_test_results=args.pseudolabel_test_results,
        pseudolabel_suffix=args.pseudolabel_suffix,
        pseudolabel_force=args.pseudolabel_force,
        strip_query_images=args.strip_query_images,
        ablation_scope=(args.ablation_scope or ("generator_only_image_strip" if args.strip_query_images else "")),
    )
