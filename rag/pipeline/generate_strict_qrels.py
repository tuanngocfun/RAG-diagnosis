"""
Generate Strict Exact-Match QRELs

Creates qrels_strict_exact.json where:
- Grade 3: doc_id == query_case_id (exact self-match)
- Grade 0: all others

This provides a "known-item search" benchmark as recommended by GPT 5.2
for Q1 journal standards.
"""
import json
from pathlib import Path


def generate_strict_qrels(
    test_case_ids: list,
    train_case_ids: list,
    output_path: Path
) -> dict:
    """
    Generate strict exact-match qrels.
    
    Only the exact case_id gets grade 3.
    """
    strict_qrels = {}
    
    train_set = set(train_case_ids)
    
    for test_id in test_case_ids:
        # Only mark as relevant if exact case exists in train
        # (for thesis, this tests "can the system find the source case?")
        strict_qrels[test_id] = {}
        
        if test_id in train_set:
            strict_qrels[test_id][test_id] = 3
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(strict_qrels, f, indent=2)
    
    return strict_qrels


def generate_clinical_similarity_qrels(
    test_case_ids: list,
    train_case_ids: list,
    entity_links_path: Path,
    output_path: Path
) -> dict:
    """
    Generate stricter clinical similarity qrels.
    
    Tightened grading:
    - Grade 3: Disease AND Pathogen AND ≥1 secondary overlap
    - Grade 2: Disease AND Pathogen (both required)
    - Grade 1: Disease OR Pathogen + ≥2 secondary overlaps
    - Grade 0: Otherwise
    
    Excludes generic terms like "leishmaniasis" from disease matching.
    """
    from .qrels_generator import load_entity_links, EntitySet
    
    all_entities = load_entity_links(entity_links_path)
    
    # Generic terms to exclude
    GENERIC_DISEASES = {"leishmaniasis", "leishmania", "infection"}
    
    def clean_entities(entity_set: set) -> set:
        """Remove generic terms."""
        return {e for e in entity_set if e.lower() not in GENERIC_DISEASES}
    
    def compute_strict_grade(test_ent: EntitySet, train_ent: EntitySet) -> int:
        """Stricter grading scheme."""
        # Clean disease sets
        test_diseases = clean_entities(test_ent.diseases)
        train_diseases = clean_entities(train_ent.diseases)
        
        disease_overlap = bool(test_diseases & train_diseases)
        pathogen_overlap = bool(test_ent.pathogens & train_ent.pathogens)
        
        # Count secondary overlaps
        secondary = 0
        if test_ent.symptoms & train_ent.symptoms:
            secondary += len(test_ent.symptoms & train_ent.symptoms)
        if test_ent.anatomy & train_ent.anatomy:
            secondary += len(test_ent.anatomy & train_ent.anatomy)
        if test_ent.drugs & train_ent.drugs:
            secondary += len(test_ent.drugs & train_ent.drugs)
        
        # Grade 3: Disease AND Pathogen AND secondary
        if disease_overlap and pathogen_overlap and secondary >= 1:
            return 3
        
        # Grade 2: Disease AND Pathogen (both required - stricter than OR)
        if disease_overlap and pathogen_overlap:
            return 2
        
        # Grade 1: One of Disease/Pathogen + strong secondary
        if (disease_overlap or pathogen_overlap) and secondary >= 2:
            return 1
        
        return 0
    
    qrels = {}
    for test_id in test_case_ids:
        if test_id not in all_entities:
            continue
        
        test_ent = all_entities[test_id]
        qrels[test_id] = {}
        
        for train_id in train_case_ids:
            if train_id not in all_entities:
                continue
            
            train_ent = all_entities[train_id]
            grade = compute_strict_grade(test_ent, train_ent)
            
            if grade >= 1:
                qrels[test_id][train_id] = grade
    
    # Save
    with open(output_path, "w") as f:
        json.dump(qrels, f, indent=2)
    
    return qrels


if __name__ == "__main__":
    from .config import TRAIN_JSONL, TEST_JSONL, ENTITY_LINKS, SPLIT_DIR
    
    # Load case IDs
    with open(TRAIN_JSONL) as f:
        train_ids = [json.loads(l)["case_id"] for l in f]
    with open(TEST_JSONL) as f:
        test_ids = [json.loads(l)["case_id"] for l in f]
    
    print("Generating strict exact-match qrels...")
    strict_qrels = generate_strict_qrels(
        test_ids, train_ids,
        SPLIT_DIR / "qrels_strict_exact.json"
    )
    print(f"  Queries: {len(strict_qrels)}")
    print(f"  Avg relevant/query: {sum(len(v) for v in strict_qrels.values()) / len(strict_qrels):.1f}")
    
    print("\nGenerating clinical similarity qrels (stricter)...")
    clinical_qrels = generate_clinical_similarity_qrels(
        test_ids, train_ids, ENTITY_LINKS,
        SPLIT_DIR / "qrels_clinical_strict.json"
    )
    
    # Stats
    total = sum(len(v) for v in clinical_qrels.values())
    grades = {}
    for q, docs in clinical_qrels.items():
        for d, g in docs.items():
            grades[g] = grades.get(g, 0) + 1
    
    print(f"  Queries: {len(clinical_qrels)}")
    print(f"  Avg relevant/query: {total / len(clinical_qrels):.1f}")
    print(f"  Grade distribution: {grades}")
