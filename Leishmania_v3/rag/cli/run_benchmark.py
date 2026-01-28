"""
Benchmark Runner: Controlled experiments across dimensions.

Per GPT 5.2 recommendation: Each benchmark run changes ONLY ONE variable:
- (model) or (rag_mode) or (context_mode) or (prompt_mode)

Dimensions:
1. Model: medgemma-4b-it, gemini-2.5-pro, gemma3-12b
2. RAG mode: rag, no_rag
3. Context mode: top_k:3, top_k:10, quality_threshold:0.6, dynamic_k
4. Prompt mode: strict_context, balanced, no_context

Output:
- Per-sample JSONL for debugging
- Summary CSV for analysis
- manifest.json for reproducibility
"""
import json
import csv
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, asdict
import argparse

# Import configs
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from configs.manifest_schema import create_manifest, save_manifest, validate_manifest
from configs.context_mode import ContextMode, get_context_selection_params, select_contexts
from configs.prompt_mode import PromptMode, build_rag_prompt
from configs.query_types import normalize_query_type, QUERY_SETS


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    # Identification
    run_id: str
    
    # Dimensions
    model_name: str
    rag_mode: str  # "rag" or "no_rag"
    context_mode: str  # e.g., "top_k:10"
    prompt_mode: str  # "strict_context", "balanced", "no_context"
    
    # Dataset
    dataset_version: str = "v143"
    query_set_id: str = "test_v143_symptom"
    
    # Model config
    temperature: float = 0.0
    max_tokens: int = 4096
    
    # Retriever config (for RAG mode)
    retriever_encoder: str = "bioclinical"
    retriever_method: str = "hybrid"
    retriever_top_k: int = 10
    
    # Evaluation
    judge_model: str = "gemini-2.5-pro"


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""
    run_id: str
    model_name: str
    rag_mode: str
    context_mode: str
    prompt_mode: str
    
    # Metrics
    diagnosis_accuracy_mean: float
    diagnosis_accuracy_std: float
    context_relevance_mean: Optional[float]
    faithfulness_mean: Optional[float]
    
    # Sample counts
    n_samples: int
    n_correct: int
    
    # Timing
    total_time_seconds: float


def get_git_commit() -> Optional[str]:
    """Get current git commit hash."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent.parent.parent
        )
        if result.returncode == 0:
            return result.stdout.strip()[:8]
    except Exception:
        pass
    return None


def generate_run_id(config: BenchmarkConfig) -> str:
    """Generate run ID from config."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    model_short = config.model_name.split("/")[-1].replace("-", "")[:10]
    return f"{model_short}_{config.rag_mode}_{config.prompt_mode}_{timestamp}"


class BenchmarkRunner:
    """
    Run controlled experiments across RAG dimensions.
    
    Usage:
        runner = BenchmarkRunner(runs_dir="/path/to/runs")
        
        # Single run
        result = runner.run_single(config)
        
        # Sweep one dimension
        results = runner.sweep_dimension(
            base_config,
            dimension="context_mode",
            values=["top_k:3", "top_k:5", "top_k:10"]
        )
    """
    
    def __init__(self, runs_dir: Path = None):
        if runs_dir is None:
            runs_dir = Path(__file__).parent.parent / "runs"
        self.runs_dir = Path(runs_dir)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        
    def run_single(self, config: BenchmarkConfig) -> BenchmarkResult:
        """
        Run a single benchmark with given configuration.
        
        This is a template - actual implementation would:
        1. Load queries based on query_set_id
        2. Initialize retriever and generator based on config
        3. Run retrieval + generation for each query
        4. Evaluate with LLM judge
        5. Save results and return summary
        """
        run_id = config.run_id or generate_run_id(config)
        run_dir = self.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"BENCHMARK: {run_id}")
        print(f"{'='*60}")
        print(f"  Model:        {config.model_name}")
        print(f"  RAG mode:     {config.rag_mode}")
        print(f"  Context mode: {config.context_mode}")
        print(f"  Prompt mode:  {config.prompt_mode}")
        print(f"  Query set:    {config.query_set_id}")
        
        start_time = datetime.now()
        
        # Create and save manifest
        manifest = create_manifest(
            run_id=run_id,
            dataset_version=config.dataset_version,
            query_set_id=config.query_set_id,
            rag_mode=config.rag_mode,
            context_mode=config.context_mode,
            prompt_mode=config.prompt_mode,
            model_name=config.model_name,
            model_config={
                "temperature": config.temperature,
                "max_tokens": config.max_tokens
            },
            retriever={
                "encoder": config.retriever_encoder,
                "method": config.retriever_method,
                "top_k": config.retriever_top_k
            } if config.rag_mode == "rag" else None,
            judge_model=config.judge_model,
            git_commit=get_git_commit()
        )
        
        is_valid, errors = validate_manifest(manifest)
        if not is_valid:
            raise ValueError(f"Invalid manifest: {errors}")
        
        save_manifest(manifest, run_dir)
        print(f"  ✓ Saved manifest.json")
        
        # TODO: Implement actual benchmark logic
        # This is a placeholder that returns dummy results
        # Actual implementation would:
        # 1. from pipeline.medical_retriever import MedicalRetriever
        # 2. from pipeline.generators import get_generator
        # 3. from pipeline.ragas_evaluator import evaluate_answers
        
        end_time = datetime.now()
        elapsed = (end_time - start_time).total_seconds()
        
        # Placeholder result
        result = BenchmarkResult(
            run_id=run_id,
            model_name=config.model_name,
            rag_mode=config.rag_mode,
            context_mode=config.context_mode,
            prompt_mode=config.prompt_mode,
            diagnosis_accuracy_mean=0.0,  # TODO: actual value
            diagnosis_accuracy_std=0.0,
            context_relevance_mean=None if config.rag_mode == "no_rag" else 0.0,
            faithfulness_mean=None if config.rag_mode == "no_rag" else 0.0,
            n_samples=0,  # TODO: actual count
            n_correct=0,
            total_time_seconds=elapsed
        )
        
        print(f"  ✓ Completed in {elapsed:.1f}s")
        print(f"  → Results saved to: {run_dir}")
        
        return result
    
    def sweep_dimension(
        self,
        base_config: BenchmarkConfig,
        dimension: str,
        values: List[Any]
    ) -> List[BenchmarkResult]:
        """
        Sweep one dimension while holding others constant.
        
        Args:
            base_config: Base configuration
            dimension: Which dimension to vary ("model_name", "rag_mode", "context_mode", "prompt_mode")
            values: List of values to try for that dimension
            
        Returns:
            List of results for each value
        """
        print(f"\n{'#'*60}")
        print(f"DIMENSION SWEEP: {dimension}")
        print(f"Values: {values}")
        print(f"{'#'*60}")
        
        results = []
        
        for value in values:
            # Create config copy with modified dimension
            config_dict = asdict(base_config)
            config_dict[dimension] = value
            config_dict["run_id"] = ""  # Will be auto-generated
            
            config = BenchmarkConfig(**config_dict)
            result = self.run_single(config)
            results.append(result)
        
        # Save sweep summary
        self._save_sweep_summary(results, dimension, values)
        
        return results
    
    def _save_sweep_summary(
        self,
        results: List[BenchmarkResult],
        dimension: str,
        values: List[Any]
    ) -> None:
        """Save sweep results to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        csv_path = self.runs_dir / f"sweep_{dimension}_{timestamp}.csv"
        
        fieldnames = [
            "run_id", "model_name", "rag_mode", "context_mode", "prompt_mode",
            "diagnosis_accuracy_mean", "diagnosis_accuracy_std",
            "context_relevance_mean", "faithfulness_mean",
            "n_samples", "n_correct", "total_time_seconds"
        ]
        
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for result in results:
                writer.writerow(asdict(result))
        
        print(f"\n✓ Sweep summary saved: {csv_path}")


# =============================================================================
# Predefined Experiment Configurations
# =============================================================================

def get_context_quality_experiment() -> List[BenchmarkConfig]:
    """
    Context Quality vs Length Experiment.
    
    Holds constant: model (medgemma), rag_mode (rag), prompt_mode (balanced)
    Varies: context_mode
    """
    base = {
        "model_name": "google/medgemma-4b-it",
        "rag_mode": "rag",
        "prompt_mode": "balanced",
        "dataset_version": "v143",
        "query_set_id": "test_v143_symptom"
    }
    
    context_modes = [
        "top_k:3",           # Few high-quality
        "top_k:5",           # Medium
        "top_k:10",          # Baseline (many, potentially noisy)
        "quality_threshold:0.6",  # Quality filtered
        "dynamic_k"          # Adaptive
    ]
    
    return [
        BenchmarkConfig(run_id="", context_mode=cm, **base)
        for cm in context_modes
    ]


def get_rag_vs_norag_experiment() -> List[BenchmarkConfig]:
    """
    RAG vs No-RAG Experiment (per model).
    
    Holds constant: prompt_mode (balanced), context_mode (top_k:10)
    Varies: rag_mode, model_name
    """
    configs = []
    
    models = [
        "google/medgemma-4b-it",
        "gemini-2.5-pro",
        "google/gemma-3-12b-it"
    ]
    
    for model in models:
        for rag_mode in ["rag", "no_rag"]:
            configs.append(BenchmarkConfig(
                run_id="",
                model_name=model,
                rag_mode=rag_mode,
                context_mode="top_k:10" if rag_mode == "rag" else "none",
                prompt_mode="balanced" if rag_mode == "rag" else "no_context",
                dataset_version="v143",
                query_set_id="test_v143_symptom"
            ))
    
    return configs


def get_prompt_mode_experiment() -> List[BenchmarkConfig]:
    """
    Prompt Mode Experiment (test if balanced helps MedGemma).
    
    Holds constant: model (medgemma), rag_mode (rag), context_mode (top_k:10)
    Varies: prompt_mode
    """
    base = {
        "model_name": "google/medgemma-4b-it",
        "rag_mode": "rag",
        "context_mode": "top_k:10",
        "dataset_version": "v143",
        "query_set_id": "test_v143_symptom"
    }
    
    return [
        BenchmarkConfig(run_id="", prompt_mode="strict_context", **base),
        BenchmarkConfig(run_id="", prompt_mode="balanced", **base),
    ]


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run controlled RAG benchmark experiments"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Single run
    single_parser = subparsers.add_parser("single", help="Run single benchmark")
    single_parser.add_argument("--model", required=True, help="Model name")
    single_parser.add_argument("--rag-mode", required=True, choices=["rag", "no_rag"])
    single_parser.add_argument("--context-mode", default="top_k:10")
    single_parser.add_argument("--prompt-mode", default="balanced", 
                               choices=["strict_context", "balanced", "no_context"])
    single_parser.add_argument("--query-set", default="test_v143_symptom")
    
    # Sweep dimension
    sweep_parser = subparsers.add_parser("sweep", help="Sweep one dimension")
    sweep_parser.add_argument("--dimension", required=True,
                              choices=["model_name", "rag_mode", "context_mode", "prompt_mode"])
    sweep_parser.add_argument("--values", nargs="+", required=True)
    sweep_parser.add_argument("--base-model", default="google/medgemma-4b-it")
    sweep_parser.add_argument("--base-rag-mode", default="rag")
    sweep_parser.add_argument("--base-context-mode", default="top_k:10")
    sweep_parser.add_argument("--base-prompt-mode", default="balanced")
    
    # Predefined experiments
    exp_parser = subparsers.add_parser("experiment", help="Run predefined experiment")
    exp_parser.add_argument("--name", required=True,
                            choices=["context_quality", "rag_vs_norag", "prompt_mode"])
    
    args = parser.parse_args()
    
    runner = BenchmarkRunner()
    
    if args.command == "single":
        config = BenchmarkConfig(
            run_id="",
            model_name=args.model,
            rag_mode=args.rag_mode,
            context_mode=args.context_mode,
            prompt_mode=args.prompt_mode,
            query_set_id=args.query_set
        )
        runner.run_single(config)
        
    elif args.command == "sweep":
        base_config = BenchmarkConfig(
            run_id="",
            model_name=args.base_model,
            rag_mode=args.base_rag_mode,
            context_mode=args.base_context_mode,
            prompt_mode=args.base_prompt_mode
        )
        runner.sweep_dimension(base_config, args.dimension, args.values)
        
    elif args.command == "experiment":
        if args.name == "context_quality":
            configs = get_context_quality_experiment()
        elif args.name == "rag_vs_norag":
            configs = get_rag_vs_norag_experiment()
        elif args.name == "prompt_mode":
            configs = get_prompt_mode_experiment()
        else:
            raise ValueError(f"Unknown experiment: {args.name}")
        
        for config in configs:
            runner.run_single(config)
    
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
