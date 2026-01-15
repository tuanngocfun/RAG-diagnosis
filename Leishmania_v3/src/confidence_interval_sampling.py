#!/usr/bin/env python3
"""
Confidence Interval Sampling for Q1 Journal Validation

Implements stratified random sampling with bootstrap confidence intervals
for validating Gemini 3 Pro diagnosis accuracy.

Per Q1 journal standards:
- Stratified sampling ensures representation of all Leishmaniasis types
- Bootstrap CI provides robust uncertainty estimates
- Cohen's Kappa measures inter-rater reliability

Usage:
    python confidence_interval_sampling.py --sample-size 30
    python confidence_interval_sampling.py --all-cases  # validate all 143
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import math

# Paths
DATA_ROOT = Path(__file__).parent.parent / "data"
VERIFIED_DIR = DATA_ROOT / "leishmaniasis_verified"
INPUT_FILE = VERIFIED_DIR / "all_verified.jsonl"
OUTPUT_DIR = VERIFIED_DIR / "validation_samples"

RANDOM_SEED = 42


@dataclass
class ValidationSample:
    """A sample selected for manual validation."""
    case_id: str
    diagnosis: str
    diagnosis_type: str
    species: str
    confidence: str
    evidence_span: str
    # Fields to be filled by human reviewer
    reviewer_agrees: Optional[bool] = None
    reviewer_diagnosis: Optional[str] = None
    reviewer_notes: Optional[str] = None


def cochran_sample_size(population: int, confidence: float = 0.95, margin: float = 0.05, proportion: float = 0.5) -> int:
    """
    Calculate required sample size using Cochran's formula with finite population correction.
    
    Args:
        population: Total population size
        confidence: Confidence level (0.95 = 95%)
        margin: Margin of error (0.05 = ±5%)
        proportion: Expected proportion (0.5 = maximum variance)
    
    Returns:
        Required sample size
    """
    # Z-score for confidence level
    z_scores = {0.90: 1.645, 0.95: 1.96, 0.99: 2.576}
    z = z_scores.get(confidence, 1.96)
    
    # Cochran's formula for infinite population
    n0 = (z**2 * proportion * (1 - proportion)) / (margin**2)
    
    # Finite population correction
    n = n0 / (1 + ((n0 - 1) / population))
    
    return math.ceil(n)


def stratified_sample(cases: List[dict], sample_size: int, seed: int = RANDOM_SEED) -> List[dict]:
    """
    Select stratified random sample by diagnosis_type.
    
    Ensures each stratum is represented proportionally.
    """
    random.seed(seed)
    
    # Group by diagnosis_type
    strata = {}
    for case in cases:
        dtype = case.get("diagnosis_type", "Unknown")
        if dtype not in strata:
            strata[dtype] = []
        strata[dtype].append(case)
    
    # Calculate samples per stratum (proportional allocation)
    total = len(cases)
    samples = []
    
    for dtype, stratum_cases in strata.items():
        # Proportional allocation
        stratum_size = max(1, round(sample_size * len(stratum_cases) / total))
        # Don't over-sample
        stratum_size = min(stratum_size, len(stratum_cases))
        
        selected = random.sample(stratum_cases, stratum_size)
        samples.extend(selected)
    
    # If we have fewer samples than requested, try to add more
    remaining = sample_size - len(samples)
    if remaining > 0:
        already_selected = {s["case_id"] for s in samples}
        available = [c for c in cases if c["case_id"] not in already_selected]
        if available:
            additional = random.sample(available, min(remaining, len(available)))
            samples.extend(additional)
    
    return samples


def bootstrap_ci(values: List[bool], n_bootstrap: int = 10000, confidence: float = 0.95) -> tuple:
    """
    Calculate bootstrap confidence interval for a proportion.
    
    Args:
        values: List of boolean values (True = correct, False = incorrect)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level
    
    Returns:
        Tuple of (mean, lower_ci, upper_ci)
    """
    if not values:
        return (0.0, 0.0, 0.0)
    
    random.seed(RANDOM_SEED)
    n = len(values)
    
    # Bootstrap resampling
    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = random.choices(values, k=n)
        bootstrap_means.append(sum(sample) / n)
    
    # Sort for percentiles
    bootstrap_means.sort()
    
    # CI bounds
    alpha = 1 - confidence
    lower_idx = int(n_bootstrap * (alpha / 2))
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    
    mean = sum(values) / n
    lower = bootstrap_means[lower_idx]
    upper = bootstrap_means[upper_idx]
    
    return (mean, lower, upper)


def cohens_kappa(rater1: List[bool], rater2: List[bool]) -> float:
    """
    Calculate Cohen's Kappa for inter-rater reliability.
    
    Args:
        rater1: List of boolean judgments from rater 1
        rater2: List of boolean judgments from rater 2
    
    Returns:
        Kappa score (-1 to 1, where 1 = perfect agreement)
    """
    if len(rater1) != len(rater2):
        raise ValueError("Raters must have same number of judgments")
    
    n = len(rater1)
    if n == 0:
        return 0.0
    
    # Count agreements and disagreements
    both_true = sum(1 for a, b in zip(rater1, rater2) if a and b)
    both_false = sum(1 for a, b in zip(rater1, rater2) if not a and not b)
    
    # Observed agreement
    po = (both_true + both_false) / n
    
    # Expected agreement by chance
    p1_true = sum(rater1) / n
    p2_true = sum(rater2) / n
    pe = (p1_true * p2_true) + ((1 - p1_true) * (1 - p2_true))
    
    # Kappa
    if pe == 1:
        return 1.0
    kappa = (po - pe) / (1 - pe)
    
    return kappa


def generate_validation_sheet(samples: List[dict], output_file: Path):
    """Generate a validation sheet for human reviewers."""
    validation_items = []
    
    for i, case in enumerate(samples, 1):
        item = {
            "sample_number": i,
            "case_id": case["case_id"],
            "llm_diagnosis": case.get("diagnosis", ""),
            "llm_diagnosis_type": case.get("diagnosis_type", ""),
            "llm_species": case.get("species", ""),
            "llm_confidence": case.get("confidence", ""),
            "evidence_span": case.get("evidence_span", "")[:500],  # Truncate for readability
            # Fields for reviewer
            "reviewer_agrees": None,  # True/False
            "reviewer_diagnosis": "",  # Fill if disagrees
            "reviewer_notes": "",
        }
        validation_items.append(item)
    
    with open(output_file, "w") as f:
        json.dump(validation_items, f, indent=2)
    
    return validation_items


def analyze_validation_results(results_file: Path) -> dict:
    """
    Analyze completed validation results.
    
    Args:
        results_file: Path to completed validation JSON
    
    Returns:
        Analysis dictionary with accuracy and CI
    """
    with open(results_file) as f:
        results = json.load(f)
    
    # Extract agreements
    agreements = [r["reviewer_agrees"] for r in results if r["reviewer_agrees"] is not None]
    
    if not agreements:
        return {"error": "No validation results found"}
    
    # Calculate accuracy with CI
    mean, lower, upper = bootstrap_ci(agreements)
    
    # By diagnosis type
    by_type = {}
    for r in results:
        dtype = r.get("llm_diagnosis_type", "Unknown")
        if dtype not in by_type:
            by_type[dtype] = []
        if r["reviewer_agrees"] is not None:
            by_type[dtype].append(r["reviewer_agrees"])
    
    type_accuracy = {}
    for dtype, vals in by_type.items():
        if vals:
            m, l, u = bootstrap_ci(vals)
            type_accuracy[dtype] = {
                "accuracy": round(m * 100, 1),
                "ci_lower": round(l * 100, 1),
                "ci_upper": round(u * 100, 1),
                "n": len(vals)
            }
    
    return {
        "total_validated": len(agreements),
        "overall_accuracy": round(mean * 100, 1),
        "ci_95_lower": round(lower * 100, 1),
        "ci_95_upper": round(upper * 100, 1),
        "accuracy_by_type": type_accuracy,
    }


def main():
    parser = argparse.ArgumentParser(description="Confidence Interval Sampling for Validation")
    parser.add_argument("--sample-size", "-n", type=int, default=None,
                        help="Number of cases to sample (default: Cochran formula)")
    parser.add_argument("--all-cases", action="store_true",
                        help="Validate all cases (no sampling)")
    parser.add_argument("--analyze", type=Path, default=None,
                        help="Analyze completed validation results")
    parser.add_argument("--seed", type=int, default=RANDOM_SEED,
                        help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Analyze mode
    if args.analyze:
        print(f"Analyzing validation results from: {args.analyze}")
        results = analyze_validation_results(args.analyze)
        print(json.dumps(results, indent=2))
        return
    
    # Load verified cases
    if not INPUT_FILE.exists():
        print(f"Error: Input file not found: {INPUT_FILE}")
        print("Run create_verified_split.py first to create the verified dataset.")
        return
    
    with open(INPUT_FILE) as f:
        cases = [json.loads(line) for line in f]
    
    print("=" * 60)
    print("Confidence Interval Sampling for Q1 Journal Validation")
    print("=" * 60)
    print(f"\nTotal verified cases: {len(cases)}")
    
    # Determine sample size
    if args.all_cases:
        sample_size = len(cases)
        selected = cases
        print(f"Mode: Validate ALL cases")
    else:
        if args.sample_size:
            sample_size = args.sample_size
        else:
            # Cochran formula for 95% CI, ±5% margin
            sample_size = cochran_sample_size(len(cases), confidence=0.95, margin=0.05)
        
        print(f"\n--- Sample Size Calculation ---")
        print(f"Population: {len(cases)}")
        print(f"Confidence Level: 95%")
        print(f"Margin of Error: ±5%")
        print(f"Required Sample Size: {sample_size}")
        
        # Stratified sampling
        selected = stratified_sample(cases, sample_size, seed=args.seed)
        print(f"\nActual samples selected: {len(selected)}")
    
    # Show distribution of selected samples
    print("\n--- Selected Sample Distribution ---")
    type_dist = Counter(c.get("diagnosis_type", "Unknown") for c in selected)
    for dtype, count in sorted(type_dist.items(), key=lambda x: -x[1]):
        print(f"  {dtype}: {count} ({100*count/len(selected):.1f}%)")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate validation sheet
    output_file = OUTPUT_DIR / f"validation_sheet_n{len(selected)}.json"
    generate_validation_sheet(selected, output_file)
    print(f"\n--- Output ---")
    print(f"Validation sheet saved: {output_file}")
    
    # Instructions
    print("\n" + "=" * 60)
    print("NEXT STEPS FOR VALIDATION")
    print("=" * 60)
    print("""
1. Open the validation sheet JSON file
2. For each case, review the 'evidence_span' and assess if the 
   LLM diagnosis is correct
3. Set 'reviewer_agrees' to true/false
4. If you disagree, provide 'reviewer_diagnosis'
5. Run this script with --analyze to compute accuracy + CI

Example:
    python confidence_interval_sampling.py --analyze validation_sheet_n30.json
""")


if __name__ == "__main__":
    main()
