#!/usr/bin/env python3
"""
Build Knowledge Graph entities from extracted data.

This script combines:
1. MeSH/keyword parsed entities
2. NER extracted entities  
3. Existing leishmaniasis_kg.json entities

Output: Extended KG with all entity types linked to case IDs
"""

import json
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set

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

OUTPUT_ROOT = config.OUTPUT_ROOT
DATA_ROOT = config.DATA_ROOT
EntityType = entity_schema.EntityType
LEISHMANIA_DISEASES = entity_schema.LEISHMANIA_DISEASES
LEISHMANIA_PATHOGENS = entity_schema.LEISHMANIA_PATHOGENS


def load_existing_kg() -> dict:
    """Load the existing leishmaniasis_kg.json if it exists."""
    kg_path = DATA_ROOT / "leishmaniasis_kg.json"
    if kg_path.exists():
        with open(kg_path) as f:
            return json.load(f)
    return {"entities": [], "relations": []}


def load_mesh_entities() -> List[dict]:
    """Load entities from MeSH parsing."""
    mesh_path = OUTPUT_ROOT / "parsed_mesh_terms.jsonl"
    if not mesh_path.exists():
        print(f"  ⚠ {mesh_path} not found. Run parse_mesh_keywords.py first.")
        return []
    
    entities = []
    with open(mesh_path) as f:
        for line in f:
            record = json.loads(line)
            # Add mesh entities
            for mesh_ent in record.get("mesh_entities", []):
                mesh_ent["case_id"] = record["case_id"]
                mesh_ent["source"] = "mesh"
                entities.append(mesh_ent)
            # Add keyword entities
            for kw_ent in record.get("keyword_entities", []):
                kw_ent["case_id"] = record["case_id"]
                kw_ent["source"] = "keyword"
                entities.append(kw_ent)
    return entities


def load_ner_entities() -> List[dict]:
    """Load entities from NER extraction."""
    ner_path = OUTPUT_ROOT / "extracted_entities.jsonl"
    if not ner_path.exists():
        print(f"  ⚠ {ner_path} not found. Run run_ner_extraction.py first.")
        return []
    
    entities = []
    with open(ner_path) as f:
        for line in f:
            record = json.loads(line)
            case_id = record["case_id"]
            for entity_type, ent_list in record.get("entities", {}).items():
                for ent in ent_list:
                    ent["case_id"] = case_id
                    ent["entity_type"] = entity_type
                    entities.append(ent)
    return entities


def normalize_entity_name(name: str, entity_type: str) -> str:
    """Normalize entity name for deduplication."""
    name_lower = name.lower().strip()
    
    # Disease normalization
    if entity_type == "Disease":
        for disease_id, info in LEISHMANIA_DISEASES.items():
            for syn in [info["name"].lower()] + [s.lower() for s in info.get("synonyms", [])]:
                if syn in name_lower or name_lower in syn:
                    return info["name"]
    
    # Pathogen normalization
    if entity_type == "Pathogen":
        for pathogen_id, info in LEISHMANIA_PATHOGENS.items():
            for syn in [info["name"].lower()] + [s.lower() for s in info.get("synonyms", [])]:
                if syn in name_lower or name_lower in syn:
                    return info["name"]
    
    return name.title() if len(name) < 30 else name


def build_entity_nodes(
    existing_kg: dict,
    mesh_entities: List[dict],
    ner_entities: List[dict]
) -> Dict[str, dict]:
    """Build unique entity nodes from all sources."""
    
    # Start with existing KG entities
    entities = {}
    for ent in existing_kg.get("entities", []):
        key = (ent["entity_type"], ent["name"])
        entities[key] = {
            "id": ent["id"],
            "name": ent["name"],
            "entity_type": ent["entity_type"],
            "description": ent.get("description", ""),
            "synonyms": ent.get("synonyms", []),
            "case_ids": set(),
            "confidence": ent.get("confidence", 1.0),
            "sources": {"existing_kg"}
        }
    
    # Add MeSH entities
    for ent in mesh_entities:
        entity_type = ent.get("inferred_type") or ent.get("entity_type", "Other")
        name = ent.get("normalized_name") or ent.get("term", "")
        if not name or entity_type == "Other":
            continue
        
        name = normalize_entity_name(name, entity_type)
        key = (entity_type, name)
        
        if key not in entities:
            entities[key] = {
                "id": f"{entity_type}_{len(entities)}",
                "name": name,
                "entity_type": entity_type,
                "description": "",
                "synonyms": [],
                "case_ids": set(),
                "confidence": 0.9,
                "sources": set()
            }
        
        entities[key]["case_ids"].add(ent.get("case_id", ""))
        entities[key]["sources"].add("mesh" if ent.get("source") == "mesh" else "keyword")
    
    # Add NER entities
    for ent in ner_entities:
        entity_type = ent.get("entity_type", "Other")
        name = ent.get("normalized_name") or ent.get("text", "")
        if not name:
            continue
        
        name = normalize_entity_name(name, entity_type)
        key = (entity_type, name)
        
        if key not in entities:
            entities[key] = {
                "id": f"{entity_type}_{len(entities)}",
                "name": name,
                "entity_type": entity_type,
                "description": "",
                "synonyms": [],
                "case_ids": set(),
                "confidence": 0.85,
                "sources": set()
            }
        
        entities[key]["case_ids"].add(ent.get("case_id", ""))
        entities[key]["sources"].add("ner")
    
    return entities


def build_case_entity_links(entities: Dict[str, dict]) -> List[dict]:
    """Build case-entity relationship links."""
    links = []
    for (entity_type, name), entity in entities.items():
        for case_id in entity["case_ids"]:
            if case_id:
                links.append({
                    "case_id": case_id,
                    "entity_id": entity["id"],
                    "entity_name": name,
                    "entity_type": entity_type,
                    "sources": list(entity["sources"])
                })
    return links


def main():
    print("=" * 60)
    print("BUILDING KNOWLEDGE GRAPH ENTITIES")
    print("=" * 60)
    
    # Load existing KG
    print("\n📂 Loading existing KG...")
    existing_kg = load_existing_kg()
    print(f"  Existing entities: {len(existing_kg.get('entities', []))}")
    print(f"  Existing relations: {len(existing_kg.get('relations', []))}")
    
    # Load extracted entities
    print("\n📂 Loading extracted entities...")
    mesh_entities = load_mesh_entities()
    print(f"  MeSH/keyword entities: {len(mesh_entities)}")
    
    ner_entities = load_ner_entities()
    print(f"  NER entities: {len(ner_entities)}")
    
    # Build entity nodes
    print("\n🔧 Building entity nodes...")
    entities = build_entity_nodes(existing_kg, mesh_entities, ner_entities)
    
    # Convert sets to lists for JSON serialization
    entities_list = []
    for (entity_type, name), ent in entities.items():
        ent_copy = ent.copy()
        ent_copy["case_ids"] = list(ent_copy["case_ids"])
        ent_copy["sources"] = list(ent_copy["sources"])
        entities_list.append(ent_copy)
    
    # Build case-entity links
    print("🔗 Building case-entity links...")
    links = build_case_entity_links(entities)
    
    # Save extended KG
    extended_kg = {
        "entities": entities_list,
        "relations": existing_kg.get("relations", []),
        "metadata": {
            "total_entities": len(entities_list),
            "total_case_links": len(links),
            "entity_types": list(set(e["entity_type"] for e in entities_list))
        }
    }
    
    kg_path = OUTPUT_ROOT / "leishmaniasis_kg_extended.json"
    print(f"\n💾 Saving extended KG to {kg_path}...")
    with open(kg_path, 'w') as f:
        json.dump(extended_kg, f, indent=2, ensure_ascii=False)
    
    # Save case-entity links
    links_path = OUTPUT_ROOT / "case_entity_links.jsonl"
    print(f"💾 Saving case-entity links to {links_path}...")
    with open(links_path, 'w') as f:
        for link in links:
            f.write(json.dumps(link, ensure_ascii=False) + '\n')
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 KNOWLEDGE GRAPH SUMMARY")
    print("=" * 60)
    
    # Count by entity type
    type_counts = defaultdict(int)
    for ent in entities_list:
        type_counts[ent["entity_type"]] += 1
    
    for entity_type in ["Disease", "Pathogen", "Symptom", "Drug", "Procedure", "Anatomy", "Other"]:
        count = type_counts.get(entity_type, 0)
        if count > 0:
            bar = "█" * min(count, 30)
            print(f"  {entity_type:12s}: {count:4d} {bar}")
    
    print(f"\nTotal unique entities: {len(entities_list)}")
    print(f"Total case-entity links: {len(links)}")
    
    print("\n✅ Done!")


if __name__ == "__main__":
    main()
