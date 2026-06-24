#!/usr/bin/env python3
"""
PheKnowLator Adapter for Leishmaniasis

Converts between your KG format and PheKnowLator's RDF/OWL format.
Based on: Bioinformatics (Q1) paper.

Note: Full PheKnowLator requires:
- pip install pkt_kg
- Ontology files (OWL format)
- Edge lists in specific format
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple


# PheKnowLator uses RDF URIs for entities
ENTITY_TYPE_URIS = {
    "Disease": "http://purl.obolibrary.org/obo/DOID_4",
    "Pathogen": "http://purl.obolibrary.org/obo/NCBITaxon_5658",  # Leishmania
    "Symptom": "http://purl.obolibrary.org/obo/SYMP_0000462",
    "Drug": "http://purl.obolibrary.org/obo/CHEBI_23888",
    "Procedure": "http://purl.obolibrary.org/obo/OBI_0000011",
    "Anatomy": "http://purl.obolibrary.org/obo/UBERON_0001062"
}

RELATION_URIS = {
    "CAUSES": "http://purl.obolibrary.org/obo/RO_0002410",
    "HAS_SYMPTOM": "http://purl.obolibrary.org/obo/RO_0002452",
    "TREATED_WITH": "http://purl.obolibrary.org/obo/RO_0002302",
    "DIAGNOSED_BY": "http://purl.obolibrary.org/obo/RO_0002451"
}


def convert_entities_to_nodes(entities: List[Dict]) -> List[Dict]:
    """
    Convert your entities to PheKnowLator node format.
    
    PheKnowLator expects:
    {
        "node_id": "URI",
        "node_type": "class_uri",
        "node_label": "human readable name"
    }
    """
    nodes = []
    
    for i, entity in enumerate(entities):
        entity_type = entity.get("entity_type", "Other")
        type_uri = ENTITY_TYPE_URIS.get(entity_type, "http://example.org/entity")
        
        node = {
            "node_id": f"http://example.org/leishmania/{entity.get('id', i)}",
            "node_type": type_uri,
            "node_label": entity.get("name", ""),
            "node_synonyms": entity.get("synonyms", []),
            "original_type": entity_type
        }
        nodes.append(node)
    
    return nodes


def convert_relations_to_edges(relations: List[Dict]) -> List[Dict]:
    """
    Convert your relations to PheKnowLator edge format.
    
    PheKnowLator expects:
    {
        "subject": "node_uri",
        "predicate": "relation_uri",
        "object": "node_uri"
    }
    """
    edges = []
    
    for rel in relations:
        rel_type = rel.get("relation_type", "RELATED_TO")
        predicate_uri = RELATION_URIS.get(rel_type, "http://purl.obolibrary.org/obo/RO_0002610")
        
        edge = {
            "subject": f"http://example.org/leishmania/{rel.get('source_id', '')}",
            "predicate": predicate_uri,
            "object": f"http://example.org/leishmania/{rel.get('target_id', '')}"
        }
        edges.append(edge)
    
    return edges


def generate_edge_list_file(edges: List[Dict], output_path: Path):
    """
    Generate PheKnowLator-compatible edge list TSV.
    
    Format: subject\tpredicate\tobject
    """
    with open(output_path, 'w') as f:
        f.write("subject\tpredicate\tobject\n")
        for edge in edges:
            f.write(f"{edge['subject']}\t{edge['predicate']}\t{edge['object']}\n")


def convert_kg_to_pheknowlator(kg_path: Path, output_dir: Path):
    """
    Full conversion from your KG to PheKnowLator format.
    
    Creates:
    - nodes.json
    - edges.json
    - edge_list.txt (for pkt_kg input)
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load your KG
    with open(kg_path) as f:
        kg = json.load(f)
    
    # Convert
    nodes = convert_entities_to_nodes(kg.get("entities", []))
    edges = convert_relations_to_edges(kg.get("relations", []))
    
    # Save nodes
    nodes_path = output_dir / "pkt_nodes.json"
    with open(nodes_path, 'w') as f:
        json.dump(nodes, f, indent=2)
    
    # Save edges
    edges_path = output_dir / "pkt_edges.json"
    with open(edges_path, 'w') as f:
        json.dump(edges, f, indent=2)
    
    # Generate edge list
    edge_list_path = output_dir / "edge_list.txt"
    generate_edge_list_file(edges, edge_list_path)
    
    return {
        "nodes": len(nodes),
        "edges": len(edges),
        "output_dir": str(output_dir)
    }


def main():
    print("=" * 60)
    print("PHEKNOWLATOR ADAPTER FOR LEISHMANIASIS")
    print("=" * 60)
    
    data_root = Path(__file__).parent.parent.parent / "data"
    kg_path = data_root / "leishmaniasis_multimodal" / "leishmaniasis_kg_extended.json"
    output_dir = Path(__file__).parent
    
    if kg_path.exists():
        result = convert_kg_to_pheknowlator(kg_path, output_dir)
        
        print(f"\n✓ Converted {result['nodes']} nodes")
        print(f"✓ Converted {result['edges']} edges")
        print(f"\nOutput files:")
        print(f"  - {output_dir}/pkt_nodes.json")
        print(f"  - {output_dir}/pkt_edges.json")
        print(f"  - {output_dir}/edge_list.txt")
        
        print("\nTo use with PheKnowLator:")
        print("  1. pip install pkt_kg")
        print("  2. Copy edge_list.txt to kg_ref/PheKnowLator/resources/")
        print("  3. python kg_ref/PheKnowLator/Main.py --app instance")
    else:
        print(f"Error: {kg_path} not found")
        print("Run multicare_pipeline first to generate the KG")


if __name__ == "__main__":
    main()
