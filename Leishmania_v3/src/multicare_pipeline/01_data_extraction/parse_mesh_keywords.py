#!/usr/bin/env python3
"""
Parse MeSH terms and keywords from MultiCaRe metadata for disease/treatment entities.

This script extracts pre-labeled clinical entities from:
1. mesh_terms (e.g., "Leishmaniasis, Visceral / diagnosis")
2. major_mesh_terms
3. keywords (e.g., "visceral leishmaniasis", "miltefosine")

Output: parsed_mesh_terms.jsonl with structured entity annotations
"""

import json
import re
import sys
from pathlib import Path
from typing import Optional
from dataclasses import dataclass, asdict

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from sibling folder (00_config)
from importlib import import_module
import importlib.util

config_path = Path(__file__).parent.parent / "00_config" / "config.py"
spec = importlib.util.spec_from_file_location("config", config_path)
config = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config)

MULTIMODAL_JSONL = config.MULTIMODAL_JSONL
OUTPUT_ROOT = config.OUTPUT_ROOT

# Entity patterns from MeSH qualifiers
MESH_QUALIFIERS_TO_PROCEDURE = {
    "diagnosis", "diagnostic imaging", "pathology", "microbiology",
    "isolation & purification", "immunology", "genetics"
}

MESH_QUALIFIERS_TO_TREATMENT = {
    "drug therapy", "therapy", "therapeutic use", "administration & dosage",
    "surgery", "rehabilitation"
}

MESH_QUALIFIERS_TO_ANATOMY = {
    "anatomy & histology", "injuries", "abnormalities"
}


@dataclass
class ParsedMeshEntity:
    """Structured entity extracted from MeSH term."""
    term: str
    qualifier: Optional[str]
    inferred_type: str  # Disease, Drug, Procedure, Anatomy, Other
    original_mesh: str


def parse_mesh_term(mesh_term: str) -> ParsedMeshEntity:
    """
    Parse a single MeSH term into entity components.
    
    Examples:
        "Leishmaniasis, Visceral / diagnosis" → term="Leishmaniasis, Visceral", qualifier="diagnosis"
        "Amphotericin B / therapeutic use" → term="Amphotericin B", qualifier="therapeutic use"
    """
    # Split on " / " to separate term from qualifier
    parts = mesh_term.split(" / ")
    term = parts[0].strip()
    qualifier = parts[1].strip() if len(parts) > 1 else None
    
    # Infer entity type from term and qualifier
    inferred_type = "Other"
    
    # Check if it's a disease by common patterns
    if any(pattern in term.lower() for pattern in 
           ["leishmaniasis", "disease", "syndrome", "disorder", "infection", 
            "carcinoma", "neoplasm", "tumor", "cancer"]):
        inferred_type = "Disease"
    
    # Check if it's a drug
    elif any(pattern in term.lower() for pattern in 
             ["amphotericin", "miltefosine", "antimoniate", "antibiotic", 
              "antineoplastic", "antiprotozoal", "therapeutic"]):
        inferred_type = "Drug"
    elif qualifier and qualifier.lower() in {"therapeutic use", "administration & dosage"}:
        inferred_type = "Drug"
    
    # Check if it's a procedure/test
    elif qualifier and qualifier.lower() in MESH_QUALIFIERS_TO_PROCEDURE:
        inferred_type = "Procedure"
    
    # Check if it's anatomy
    elif any(pattern in term.lower() for pattern in 
             ["liver", "spleen", "bone marrow", "lymph node", "skin", "kidney"]):
        inferred_type = "Anatomy"
    
    return ParsedMeshEntity(
        term=term,
        qualifier=qualifier,
        inferred_type=inferred_type,
        original_mesh=mesh_term
    )


def parse_keywords(keywords: list[str]) -> list[dict]:
    """
    Parse keywords into potential entity mentions.
    
    Keywords often contain disease names, drug names, or anatomical terms
    without the structured format of MeSH.
    """
    entities = []
    
    # Leishmaniasis disease patterns
    disease_patterns = [
        (r"visceral\s+leishmaniasis", "Disease", "Visceral Leishmaniasis"),
        (r"cutaneous\s+leishmaniasis", "Disease", "Cutaneous Leishmaniasis"),
        (r"mucocutaneous\s+leishmaniasis", "Disease", "Mucocutaneous Leishmaniasis"),
        (r"diffuse\s+cutaneous\s+leishmaniasis", "Disease", "Diffuse Cutaneous Leishmaniasis"),
        (r"leishmaniosis", "Disease", "Leishmaniasis"),
        (r"kala[- ]?azar", "Disease", "Visceral Leishmaniasis"),
    ]
    
    # Drug patterns
    drug_patterns = [
        (r"amphotericin", "Drug", "Amphotericin B"),
        (r"miltefosine", "Drug", "Miltefosine"),
        (r"pentavalent\s+antimonial", "Drug", "Pentavalent Antimonial"),
        (r"sodium\s+stibogluconate", "Drug", "Sodium Stibogluconate"),
        (r"pentamidine", "Drug", "Pentamidine"),
    ]
    
    # Pathogen patterns
    pathogen_patterns = [
        (r"leishmania\s+\(?l\.?\)?\s*donovani", "Pathogen", "Leishmania donovani"),
        (r"leishmania\s+\(?l\.?\)?\s*infantum", "Pathogen", "Leishmania infantum"),
        (r"leishmania\s+\(?l\.?\)?\s*major", "Pathogen", "Leishmania major"),
        (r"leishmania\s+\(?l\.?\)?\s*tropica", "Pathogen", "Leishmania tropica"),
        (r"leishmania\s+\(?l\.?\)?\s*braziliensis", "Pathogen", "Leishmania braziliensis"),
        (r"leishmania\s+\(?l\.?\)?\s*amazonensis", "Pathogen", "Leishmania amazonensis"),
    ]
    
    all_patterns = disease_patterns + drug_patterns + pathogen_patterns
    
    for keyword in keywords:
        keyword_lower = keyword.lower()
        for pattern, entity_type, normalized_name in all_patterns:
            if re.search(pattern, keyword_lower):
                entities.append({
                    "keyword": keyword,
                    "entity_type": entity_type,
                    "normalized_name": normalized_name
                })
                break
    
    return entities


def parse_mesh_string(mesh_str: str) -> list[str]:
    """
    Parse a MeSH terms string that looks like a Python list but has complex formatting.
    
    Examples:
        "[Case Reports]" → ["Case Reports"]
        "[Leishmaniasis, Visceral / diagnosis, 'Some Term', Another]" → [...]
    """
    if not mesh_str or mesh_str == "None" or mesh_str == "[]":
        return []
    
    # Remove outer brackets
    mesh_str = mesh_str.strip()
    if mesh_str.startswith("[") and mesh_str.endswith("]"):
        mesh_str = mesh_str[1:-1]
    
    if not mesh_str:
        return []
    
    # Split carefully - MeSH terms can contain commas internally
    # Pattern: split on ", " but not within quoted strings
    terms = []
    current_term = ""
    in_quotes = False
    quote_char = None
    
    i = 0
    while i < len(mesh_str):
        char = mesh_str[i]
        
        # Handle quotes
        if char in ["'", '"'] and (i == 0 or mesh_str[i-1] != '\\'):
            if not in_quotes:
                in_quotes = True
                quote_char = char
            elif char == quote_char:
                in_quotes = False
                quote_char = None
            current_term += char
        elif char == ',' and not in_quotes:
            # This is a delimiter
            term = current_term.strip().strip("'\"")
            if term:
                terms.append(term)
            current_term = ""
        else:
            current_term += char
        i += 1
    
    # Don't forget the last term
    term = current_term.strip().strip("'\"")
    if term:
        terms.append(term)
    
    return terms


def process_case_record(record: dict) -> dict:
    """Process a single case record for MeSH/keyword entities."""
    case_id = record["case_id"]
    
    parsed_mesh = []
    parsed_keywords = []
    
    # Parse MeSH terms using robust parser
    mesh_terms_str = record.get("mesh_terms", "")
    if mesh_terms_str and mesh_terms_str != "None":
        # Skip common non-informative terms
        skip_terms = {
            "Case Reports", "Humans", "Male", "Female", "Adult", "Aged", 
            "Middle Aged", "Young Adult", "Child", "Infant", "Adolescent",
            "Research Support, Non-U.S. Gov't", "Research Support, N.I.H., Extramural"
        }
        
        mesh_list = parse_mesh_string(mesh_terms_str)
        for mesh in mesh_list:
            if mesh and mesh not in skip_terms:
                parsed = parse_mesh_term(mesh)
                parsed_mesh.append(asdict(parsed))
    
    # Parse keywords using robust parser
    keywords_str = record.get("keywords", "")
    if keywords_str and keywords_str != "None":
        keywords_list = parse_mesh_string(keywords_str)  # Same format
        parsed_keywords = parse_keywords(keywords_list)
    
    return {
        "case_id": case_id,
        "article_id": record.get("article_id"),
        "mesh_entities": parsed_mesh,
        "keyword_entities": parsed_keywords,
        "disease_count": sum(1 for e in parsed_mesh if e["inferred_type"] == "Disease"),
        "drug_count": sum(1 for e in parsed_mesh if e["inferred_type"] == "Drug"),
        "pathogen_count": len([e for e in parsed_keywords if e.get("entity_type") == "Pathogen"])
    }


def main():
    print("=" * 60)
    print("PARSING MESH TERMS AND KEYWORDS")
    print("=" * 60)
    
    # Load multimodal dataset
    print(f"\n📂 Loading {MULTIMODAL_JSONL}...")
    records = []
    with open(MULTIMODAL_JSONL) as f:
        for line in f:
            records.append(json.loads(line))
    print(f"✓ Loaded {len(records)} records")
    
    # Process each record
    print("\n🔍 Parsing MeSH terms and keywords...")
    results = []
    total_mesh = 0
    total_keywords = 0
    disease_found = 0
    drug_found = 0
    pathogen_found = 0
    
    for record in records:
        parsed = process_case_record(record)
        results.append(parsed)
        total_mesh += len(parsed["mesh_entities"])
        total_keywords += len(parsed["keyword_entities"])
        disease_found += parsed["disease_count"]
        drug_found += parsed["drug_count"]
        pathogen_found += parsed["pathogen_count"]
    
    # Save results
    output_path = OUTPUT_ROOT / "parsed_mesh_terms.jsonl"
    print(f"\n💾 Saving to {output_path}...")
    with open(output_path, 'w') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 PARSING SUMMARY")
    print("=" * 60)
    print(f"Total records processed: {len(records)}")
    print(f"MeSH entities found: {total_mesh}")
    print(f"  - Diseases: {disease_found}")
    print(f"  - Drugs: {drug_found}")
    print(f"Keyword entities found: {total_keywords}")
    print(f"  - Pathogens: {pathogen_found}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
