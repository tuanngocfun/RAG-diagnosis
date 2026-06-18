"""
QRELs Generator for RAG Evaluation

Generates graded relevance judgments based on entity overlap:
- Grade 3: Same Disease + Same Pathogen
- Grade 2: Same Disease OR Same Pathogen  
- Grade 1: Overlap in Symptoms/Anatomy/Drugs (≥2)
- Grade 0: No significant overlap
"""
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple
from collections import defaultdict
from dataclasses import dataclass


@dataclass
class EntitySet:
    """Entities for a single case."""
    case_id: str
    diseases: Set[str]
    pathogens: Set[str]
    symptoms: Set[str]
    anatomy: Set[str]
    drugs: Set[str]
    procedures: Set[str]


def load_entity_links(entity_links_path: Path) -> Dict[str, EntitySet]:
    """
    Load entity links and organize by case_id.
    
    Returns:
        Dict mapping case_id to EntitySet
    """
    case_entities = defaultdict(lambda: {
        "diseases": set(),
        "pathogens": set(),
        "symptoms": set(),
        "anatomy": set(),
        "drugs": set(),
        "procedures": set()
    })
    
    with open(entity_links_path) as f:
        for line in f:
            link = json.loads(line)
            case_id = link["case_id"]
            entity_type = link.get("entity_type", "").lower()
            entity_id = link.get("entity_id", "")
            
            if entity_type == "disease":
                case_entities[case_id]["diseases"].add(entity_id)
            elif entity_type == "pathogen":
                case_entities[case_id]["pathogens"].add(entity_id)
            elif entity_type == "symptom":
                case_entities[case_id]["symptoms"].add(entity_id)
            elif entity_type == "anatomy":
                case_entities[case_id]["anatomy"].add(entity_id)
            elif entity_type == "drug":
                case_entities[case_id]["drugs"].add(entity_id)
            elif entity_type == "procedure":
                case_entities[case_id]["procedures"].add(entity_id)
    
    # Convert to EntitySet objects
    result = {}
    for case_id, entities in case_entities.items():
        result[case_id] = EntitySet(
            case_id=case_id,
            diseases=entities["diseases"],
            pathogens=entities["pathogens"],
            symptoms=entities["symptoms"],
            anatomy=entities["anatomy"],
            drugs=entities["drugs"],
            procedures=entities["procedures"]
        )
    
    return result


def compute_relevance_grade(
    test_entities: EntitySet, 
    train_entities: EntitySet,
    exclude_generic_cl: bool = True
) -> int:
    """
    Compute graded relevance between test and train case.
    
    Grading scheme:
    - Grade 3: Same Disease + Same Pathogen
    - Grade 2: Same Disease OR Same Pathogen (excluding generic CL)
    - Grade 1: Overlap in Symptoms/Anatomy/Drugs (≥2 common)
    - Grade 0: No significant overlap
    
    Args:
        test_entities: Entities from test query case
        train_entities: Entities from train corpus case  
        exclude_generic_cl: If True, ignore 'CL' match when more specific
                           diseases are present (fixes CL overmatching)
    """
    # Get diseases for comparison
    test_diseases = test_entities.diseases.copy()
    train_diseases = train_entities.diseases.copy()
    
    # Fix CL overmatching: don't rely on generic 'CL' if specific types exist
    if exclude_generic_cl:
        generic_labels = {'CL', 'VL', 'ML', 'MCL', 'Cutaneous', 'Visceral', 
                         'Mucocutaneous', 'Leishmaniasis'}
        
        # Only remove generic labels if more specific ones exist
        if len(test_diseases - generic_labels) > 0:
            test_diseases = test_diseases - generic_labels
        if len(train_diseases - generic_labels) > 0:
            train_diseases = train_diseases - generic_labels
    
    # Check Disease overlap (after cleaning)
    disease_overlap = bool(test_diseases & train_diseases)
    
    # Check Pathogen overlap (more reliable for Leishmania speciation)
    pathogen_overlap = bool(test_entities.pathogens & train_entities.pathogens)
    
    # Grade 3: Both Disease and Pathogen match
    if disease_overlap and pathogen_overlap:
        return 3
    
    # Grade 2: Either Disease or Pathogen matches
    if disease_overlap or pathogen_overlap:
        return 2
    
    # Count secondary overlaps
    secondary_overlap = 0
    if test_entities.symptoms & train_entities.symptoms:
        secondary_overlap += len(test_entities.symptoms & train_entities.symptoms)
    if test_entities.anatomy & train_entities.anatomy:
        secondary_overlap += len(test_entities.anatomy & train_entities.anatomy)
    if test_entities.drugs & train_entities.drugs:
        secondary_overlap += len(test_entities.drugs & train_entities.drugs)
    
    # Grade 1: At least 2 secondary entity overlaps
    if secondary_overlap >= 2:
        return 1
    
    # Grade 0: No significant overlap
    return 0


def generate_qrels(
    test_case_ids: List[str],
    train_case_ids: List[str],
    entity_links_path: Path,
    min_grade: int = 1,
    top_k: int = None,
    exclude_generic_cl: bool = True
) -> Dict[str, Dict[str, int]]:
    """
    Generate qrels (query relevance judgments) for all test cases.
    
    Args:
        test_case_ids: List of test case IDs (queries)
        train_case_ids: List of train case IDs (corpus)
        entity_links_path: Path to case_entity_links.jsonl
        min_grade: Minimum grade to include (default 1)
        top_k: Maximum relevant docs per query (None = no limit)
        exclude_generic_cl: Exclude generic labels like CL/VL in disease matching
    
    Returns:
        Dict[query_id, Dict[doc_id, grade]]
    """
    # Load all entity links
    all_entities = load_entity_links(entity_links_path)
    
    qrels = {}
    
    for test_id in test_case_ids:
        if test_id not in all_entities:
            continue
        
        test_entities = all_entities[test_id]
        query_docs = []
        
        for train_id in train_case_ids:
            if train_id not in all_entities:
                continue
            
            train_entities = all_entities[train_id]
            grade = compute_relevance_grade(
                test_entities, train_entities, 
                exclude_generic_cl=exclude_generic_cl
            )
            
            if grade >= min_grade:
                query_docs.append((train_id, grade))
        
        # Sort by grade descending and apply top_k cap
        query_docs.sort(key=lambda x: -x[1])
        if top_k is not None:
            query_docs = query_docs[:top_k]
        
        qrels[test_id] = {doc_id: grade for doc_id, grade in query_docs}
    
    return qrels


def save_qrels_trec_format(
    qrels: Dict[str, Dict[str, int]],
    output_path: Path
) -> None:
    """
    Save qrels in TREC format: query_id 0 doc_id grade
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        for query_id, docs in sorted(qrels.items()):
            for doc_id, grade in sorted(docs.items(), key=lambda x: -x[1]):
                f.write(f"{query_id} 0 {doc_id} {grade}\n")


def save_qrels_json(
    qrels: Dict[str, Dict[str, int]],
    output_path: Path
) -> None:
    """Save qrels as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(qrels, f, indent=2)


def get_qrels_stats(qrels: Dict[str, Dict[str, int]]) -> Dict:
    """Get statistics about qrels."""
    grade_counts = defaultdict(int)
    
    for query_id, docs in qrels.items():
        for doc_id, grade in docs.items():
            grade_counts[grade] += 1
    
    total_relevant = sum(grade_counts.values())
    queries_with_relevant = sum(1 for q in qrels.values() if q)
    
    return {
        "total_queries": len(qrels),
        "queries_with_relevant": queries_with_relevant,
        "total_relevant_pairs": total_relevant,
        "avg_relevant_per_query": total_relevant / len(qrels) if qrels else 0,
        "grade_distribution": dict(grade_counts)
    }


if __name__ == "__main__":
    from .config import TRAIN_JSONL, TEST_JSONL, ENTITY_LINKS, DATA_ROOT
    
    # Load case IDs
    with open(TRAIN_JSONL) as f:
        train_ids = [json.loads(l)["case_id"] for l in f]
    with open(TEST_JSONL) as f:
        test_ids = [json.loads(l)["case_id"] for l in f]
    
    # Generate qrels
    qrels = generate_qrels(test_ids, train_ids, ENTITY_LINKS)
    
    # Save
    save_qrels_trec_format(qrels, DATA_ROOT / "qrels.txt")
    save_qrels_json(qrels, DATA_ROOT / "qrels.json")
    
    # Print stats
    stats = get_qrels_stats(qrels)
    print("QRELs Statistics:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
