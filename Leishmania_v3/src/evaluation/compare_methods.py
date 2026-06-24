#!/usr/bin/env python3
"""
Compare KG Construction Methods

Compares:
1. Your method (multicare_pipeline)
2. AutoRD (external_kg/autord)
3. Baselines (no KG)

Usage:
    python compare_methods.py --config comparison_config.json
"""

import json
import sys
from pathlib import Path
from typing import Dict, List
from collections import defaultdict

from metrics import (
    entity_extraction_metrics,
    kg_quality_metrics,
    compare_methods as compare_results
)


def load_kg(path: Path) -> Dict:
    """Load a KG from JSON file."""
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {"entities": [], "relations": []}


def load_links(path: Path) -> List[Dict]:
    """Load case-entity links from JSONL."""
    links = []
    if path.exists():
        with open(path) as f:
            for line in f:
                links.append(json.loads(line))
    return links


def count_entity_types(kg: Dict) -> Dict[str, int]:
    """Count entities by type."""
    counts = defaultdict(int)
    for e in kg.get("entities", []):
        counts[e.get("entity_type", "Other")] += 1
    return dict(counts)


def compare_kg_methods(
    data_root: Path,
    methods: Dict[str, Dict[str, Path]]
) -> Dict:
    """
    Compare multiple KG construction methods.
    
    methods: Dict mapping method name to paths
        {"yours": {"kg": Path, "links": Path}, ...}
    """
    results = {}
    
    for method_name, paths in methods.items():
        print(f"\n📊 Evaluating: {method_name}")
        
        kg = load_kg(paths.get("kg"))
        links = load_links(paths.get("links"))
        
        # Basic stats
        entity_counts = count_entity_types(kg)
        
        # Quality metrics
        quality = kg_quality_metrics(kg, links)
        
        results[method_name] = {
            "total_entities": quality["total_entities"],
            "total_links": len(links),
            "entity_types": entity_counts,
            "cases_covered": quality["cases_with_entities"],
            "avg_entities_per_case": quality["avg_entities_per_case"]
        }
        
        print(f"   Entities: {quality['total_entities']}")
        print(f"   Links: {len(links)}")
        print(f"   Cases covered: {quality['cases_with_entities']}")
    
    # Compare
    comparison = compare_results({
        name: {
            "total_entities": r["total_entities"],
            "avg_entities_per_case": r["avg_entities_per_case"],
            "coverage": r["cases_covered"]
        }
        for name, r in results.items()
    })
    
    return {
        "method_results": results,
        "comparison": comparison
    }


def generate_comparison_table(results: Dict) -> str:
    """Generate markdown comparison table."""
    methods = list(results["method_results"].keys())
    
    table = "| Metric | " + " | ".join(methods) + " |\n"
    table += "|" + "---|" * (len(methods) + 1) + "\n"
    
    # Add rows for each metric
    metrics = ["total_entities", "total_links", "cases_covered", "avg_entities_per_case"]
    
    for metric in metrics:
        row = f"| {metric} |"
        for method in methods:
            value = results["method_results"][method].get(metric, "N/A")
            if isinstance(value, float):
                row += f" {value:.2f} |"
            else:
                row += f" {value} |"
        table += row + "\n"
    
    return table


def main():
    print("=" * 60)
    print("KNOWLEDGE GRAPH METHOD COMPARISON")
    print("=" * 60)
    
    # Define paths
    data_root = Path(__file__).parent.parent.parent / "data"
    multimodal_dir = data_root / "leishmaniasis_multimodal"
    external_dir = data_root / "external_kg"
    
    methods = {
        "Your Method": {
            "kg": multimodal_dir / "leishmaniasis_kg_extended.json",
            "links": multimodal_dir / "case_entity_links.jsonl"
        },
        "AutoRD Format": {
            "kg": external_dir / "autord" / "autord_output.json",
            "links": Path("/dev/null")  # AutoRD uses different format
        }
    }
    
    # Run comparison
    results = compare_kg_methods(data_root, methods)
    
    # Print table
    print("\n" + "=" * 60)
    print("COMPARISON TABLE")
    print("=" * 60)
    print(generate_comparison_table(results))
    
    # Save results
    output_path = Path(__file__).parent / "comparison_results.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results to {output_path}")


if __name__ == "__main__":
    main()
