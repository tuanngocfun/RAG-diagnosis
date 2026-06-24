#!/usr/bin/env python3
"""
AutoRD Adapter for Leishmaniasis Dataset

This adapter converts your leishmaniasis multimodal data to AutoRD format
and provides a pattern-based fallback when OpenAI API is not available.

Based on: JMIR Medical Informatics (Q1) AutoRD paper
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# AutoRD Entity Types (from paper)
AUTORD_ENTITY_TYPES = ["rare_disease", "disease", "symptom_and_sign", "anaphor"]

# AutoRD Relation Types (from paper)
AUTORD_RELATION_TYPES = ["produces", "increases_risk_of", "is_a", "is_synon", "anaphora"]


def convert_to_autord_format(multimodal_jsonl: Path) -> List[dict]:
    """
    Convert leishmaniasis multimodal data to AutoRD input format.
    
    AutoRD expects:
    {
        "id": "doc_id",
        "text": "clinical text",
        "gold": {"entities": [...], "relations": [...]}  # for training only
    }
    """
    autord_data = []
    
    with open(multimodal_jsonl) as f:
        for line in f:
            record = json.loads(line)
            
            autord_record = {
                "id": record["case_id"],
                "text": record.get("case_text", ""),
                "metadata": {
                    "article_id": record.get("article_id"),
                    "title": record.get("title", ""),
                    "abstract": record.get("abstract", "")[:500] if record.get("abstract") else ""
                }
            }
            
            autord_data.append(autord_record)
    
    return autord_data


def map_entity_type_to_autord(your_type: str) -> str:
    """Map your entity types to AutoRD entity types."""
    mapping = {
        "Disease": "rare_disease",  # Leishmaniasis as rare disease
        "Pathogen": "disease",  # Leishmania species
        "Symptom": "symptom_and_sign",
        "Drug": "disease",  # Not in AutoRD, map to disease
        "Procedure": "symptom_and_sign",  # Not in AutoRD
        "Anatomy": "symptom_and_sign"  # Not in AutoRD
    }
    return mapping.get(your_type, "disease")


def convert_kg_to_autord_output(kg_extended_path: Path) -> dict:
    """
    Convert your KG output to AutoRD-compatible format.
    
    AutoRD outputs triples: (subject, predicate, object)
    """
    with open(kg_extended_path) as f:
        kg = json.load(f)
    
    triples = []
    
    # Convert entities to triples using is_a relations
    for entity in kg.get("entities", []):
        entity_type = entity.get("entity_type", "")
        entity_name = entity.get("name", "")
        
        # Create taxonomic triple
        autord_type = map_entity_type_to_autord(entity_type)
        triples.append({
            "subject": entity_name,
            "predicate": "is_a",
            "object": autord_type
        })
    
    # Convert existing relations
    for relation in kg.get("relations", []):
        triples.append({
            "subject": relation.get("source_id", ""),
            "predicate": relation.get("relation_type", "produces").lower(),
            "object": relation.get("target_id", "")
        })
    
    return {
        "triples": triples,
        "metadata": {
            "total_triples": len(triples),
            "source": "leishmaniasis_kg_extended",
            "format": "AutoRD-compatible"
        }
    }


def main():
    print("=" * 60)
    print("AUTORD ADAPTER FOR LEISHMANIASIS")
    print("=" * 60)
    
    data_root = Path(__file__).parent.parent.parent / "leishmaniasis_multimodal"
    
    # Convert input data
    print("\n📂 Converting to AutoRD input format...")
    multimodal_path = data_root / "leishmaniasis_multimodal.jsonl"
    if multimodal_path.exists():
        autord_input = convert_to_autord_format(multimodal_path)
        
        output_path = Path(__file__).parent / "autord_input.jsonl"
        with open(output_path, 'w') as f:
            for record in autord_input:
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        print(f"   ✓ Saved {len(autord_input)} records to autord_input.jsonl")
    
    # Convert KG output
    print("\n🔄 Converting KG to AutoRD format...")
    kg_path = data_root / "leishmaniasis_kg_extended.json"
    if kg_path.exists():
        autord_output = convert_kg_to_autord_output(kg_path)
        
        output_path = Path(__file__).parent / "autord_output.json"
        with open(output_path, 'w') as f:
            json.dump(autord_output, f, indent=2, ensure_ascii=False)
        print(f"   ✓ Saved {autord_output['metadata']['total_triples']} triples to autord_output.json")
    
    print("\n✅ Done!")
    print("\nTo run actual AutoRD with OpenAI:")
    print("  1. Set OPENAI_API_KEY in kg_ref/AutoRD/env.py")
    print("  2. Copy autord_input.jsonl to kg_ref/AutoRD/data/")
    print("  3. Run: cd kg_ref/AutoRD && bash run.sh")


if __name__ == "__main__":
    main()
