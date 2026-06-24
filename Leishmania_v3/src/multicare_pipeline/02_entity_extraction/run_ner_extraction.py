#!/usr/bin/env python3
"""
Named Entity Recognition (NER) extraction from clinical case text.

This script uses pattern matching and optional spaCy/scispaCy models to extract
clinical entities from case_text:
- Disease/Diagnosis
- Pathogen/Species  
- Symptom/Sign
- Drug/Treatment
- Test/Procedure
- Anatomy/Body site

Can run in two modes:
1. Pattern-only (fast, no ML dependencies)
2. With scispaCy (more accurate, requires: pip install scispacy en_ner_bc5cdr_md)
"""

import json
import re
import sys
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, List
from collections import defaultdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from sibling folder (00_config)
import importlib.util

def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

config_dir = Path(__file__).parent.parent / "00_config"
config = load_module("config", config_dir / "config.py")
entity_schema = load_module("entity_schema", config_dir / "entity_schema.py")

MULTIMODAL_JSONL = config.MULTIMODAL_JSONL
OUTPUT_ROOT = config.OUTPUT_ROOT
LEISHMANIA_DISEASES = entity_schema.LEISHMANIA_DISEASES
LEISHMANIA_PATHOGENS = entity_schema.LEISHMANIA_PATHOGENS
LEISHMANIA_SYMPTOMS = entity_schema.LEISHMANIA_SYMPTOMS
LEISHMANIA_DRUGS = entity_schema.LEISHMANIA_DRUGS
LEISHMANIA_PROCEDURES = entity_schema.LEISHMANIA_PROCEDURES
LEISHMANIA_ANATOMY = entity_schema.LEISHMANIA_ANATOMY


@dataclass
class ExtractedEntity:
    """Entity extracted via NER."""
    text: str
    entity_type: str
    normalized_name: Optional[str] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    confidence: float = 1.0
    source: str = "pattern"  # or "scispacy", "medcat"


def build_pattern_matchers() -> dict:
    """Build regex patterns for entity extraction."""
    matchers = {}
    
    # Disease patterns (from LEISHMANIA_DISEASES)
    disease_patterns = []
    for disease_id, info in LEISHMANIA_DISEASES.items():
        names = [info["name"]] + info.get("synonyms", [])
        for name in names:
            # Escape special chars and make case-insensitive
            pattern = re.escape(name)
            disease_patterns.append((pattern, disease_id, info["name"]))
    matchers["Disease"] = disease_patterns
    
    # Pathogen patterns (from LEISHMANIA_PATHOGENS)
    pathogen_patterns = []
    for pathogen_id, info in LEISHMANIA_PATHOGENS.items():
        names = [info["name"]] + info.get("synonyms", [])
        for name in names:
            pattern = re.escape(name).replace(r"\.", r"\.?")  # Allow optional periods
            pathogen_patterns.append((pattern, pathogen_id, info["name"]))
    matchers["Pathogen"] = pathogen_patterns
    
    # Symptom patterns
    symptom_patterns = [(re.escape(s), s, s) for s in LEISHMANIA_SYMPTOMS]
    matchers["Symptom"] = symptom_patterns
    
    # Drug patterns
    drug_patterns = [(re.escape(d), d, d) for d in LEISHMANIA_DRUGS]
    matchers["Drug"] = drug_patterns
    
    # Procedure patterns
    procedure_patterns = [(re.escape(p), p, p) for p in LEISHMANIA_PROCEDURES]
    matchers["Procedure"] = procedure_patterns
    
    # Anatomy patterns
    anatomy_patterns = [(re.escape(a), a, a) for a in LEISHMANIA_ANATOMY]
    matchers["Anatomy"] = anatomy_patterns
    
    return matchers


def extract_entities_pattern(text: str, matchers: dict) -> List[ExtractedEntity]:
    """Extract entities using pattern matching."""
    entities = []
    text_lower = text.lower()
    
    for entity_type, patterns in matchers.items():
        for pattern, entity_id, normalized_name in patterns:
            # Find all matches
            for match in re.finditer(pattern, text_lower, re.IGNORECASE):
                entities.append(ExtractedEntity(
                    text=text[match.start():match.end()],
                    entity_type=entity_type,
                    normalized_name=normalized_name,
                    start_char=match.start(),
                    end_char=match.end(),
                    confidence=0.9,
                    source="pattern"
                ))
    
    # Deduplicate by position (keep first match at each position)
    seen_positions = set()
    unique_entities = []
    for entity in entities:
        pos_key = (entity.start_char, entity.end_char)
        if pos_key not in seen_positions:
            seen_positions.add(pos_key)
            unique_entities.append(entity)
    
    return unique_entities


def try_scispacy_extraction(text: str) -> List[ExtractedEntity]:
    """
    Try to extract entities using scispaCy if available.
    
    Install with:
        pip install scispacy
        pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz
    """
    try:
        import spacy
        
        # Try to load the model if not already loaded
        if not hasattr(try_scispacy_extraction, '_nlp'):
            try:
                nlp = spacy.load("en_ner_bc5cdr_md")
                try_scispacy_extraction._nlp = nlp
                print("  ✓ Loaded scispaCy model en_ner_bc5cdr_md")
            except OSError:
                # Model not installed
                try_scispacy_extraction._nlp = None
        
        nlp = try_scispacy_extraction._nlp
        if nlp is None:
            return []
        
        doc = nlp(text)
        entities = []
        
        for ent in doc.ents:
            # Map scispaCy labels to our schema
            if ent.label_ == "DISEASE":
                entity_type = "Disease"
            elif ent.label_ == "CHEMICAL":
                entity_type = "Drug"
            else:
                entity_type = ent.label_
            
            entities.append(ExtractedEntity(
                text=ent.text,
                entity_type=entity_type,
                normalized_name=ent.text,
                start_char=ent.start_char,
                end_char=ent.end_char,
                confidence=0.85,
                source="scispacy"
            ))
        
        return entities
    
    except ImportError:
        return []


def process_case(record: dict, matchers: dict, use_scispacy: bool = False) -> dict:
    """Process a single case record for NER."""
    case_id = record["case_id"]
    case_text = record.get("case_text", "") or ""
    abstract = record.get("abstract", "") or ""
    captions = " ".join([img.get("caption", "") for img in record.get("images", [])])
    
    # Combine all text sources
    full_text = f"{case_text}\n\n{abstract}\n\n{captions}"
    
    # Extract entities using patterns
    pattern_entities = extract_entities_pattern(full_text, matchers)
    
    # Optionally add scispaCy entities
    scispacy_entities = []
    if use_scispacy:
        scispacy_entities = try_scispacy_extraction(case_text)
    
    # Merge entities (pattern entities take precedence for our domain)
    all_entities = pattern_entities + scispacy_entities
    
    # Group by entity type
    entities_by_type = defaultdict(list)
    for entity in all_entities:
        entities_by_type[entity.entity_type].append(asdict(entity))
    
    return {
        "case_id": case_id,
        "article_id": record.get("article_id"),
        "entities": dict(entities_by_type),
        "entity_counts": {k: len(v) for k, v in entities_by_type.items()},
        "total_entities": len(all_entities)
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract clinical entities via NER")
    parser.add_argument("--use-scispacy", action="store_true", 
                        help="Also use scispaCy for extraction (requires installation)")
    args = parser.parse_args()
    
    print("=" * 60)
    print("CLINICAL ENTITY EXTRACTION (NER)")
    print("=" * 60)
    
    # Build pattern matchers
    print("\n🔧 Building pattern matchers...")
    matchers = build_pattern_matchers()
    for entity_type, patterns in matchers.items():
        print(f"  {entity_type}: {len(patterns)} patterns")
    
    # Load multimodal dataset
    print(f"\n📂 Loading {MULTIMODAL_JSONL}...")
    records = []
    with open(MULTIMODAL_JSONL) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"✓ Loaded {len(records)} records")
    
    # Process each record
    print(f"\n🔍 Extracting entities (scispaCy: {'enabled' if args.use_scispacy else 'disabled'})...")
    results = []
    total_counts = defaultdict(int)
    
    for i, record in enumerate(records):
        if (i + 1) % 50 == 0:
            print(f"  Processed {i + 1}/{len(records)}...")
        
        result = process_case(record, matchers, use_scispacy=args.use_scispacy)
        results.append(result)
        
        for entity_type, count in result["entity_counts"].items():
            total_counts[entity_type] += count
    
    # Save results
    output_path = OUTPUT_ROOT / "extracted_entities.jsonl"
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 EXTRACTION SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {len(records)}")
    print(f"\nEntities extracted by type:")
    for entity_type in ["Disease", "Pathogen", "Symptom", "Drug", "Procedure", "Anatomy"]:
        count = total_counts.get(entity_type, 0)
        bar = "█" * min(count // 10, 30)
        print(f"  {entity_type:12s}: {count:4d} {bar}")
    
    print(f"\nTotal entities: {sum(total_counts.values())}")
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
