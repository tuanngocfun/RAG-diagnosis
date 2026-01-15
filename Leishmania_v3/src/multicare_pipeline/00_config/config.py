"""
Configuration and paths for the MultiCaRe pipeline.
"""
from pathlib import Path

# Base paths
# Navigate from src/multicare_pipeline/00_config/ → Leishmania_v3 → data/
DATA_ROOT = Path(__file__).parent.parent.parent.parent / "data"
MULTICARE_ROOT = DATA_ROOT / "whole_multicare_dataset"
OUTPUT_ROOT = DATA_ROOT / "leishmaniasis_multimodal"

# Input files
CASES_PARQUET = MULTICARE_ROOT / "cases.parquet"
METADATA_PARQUET = MULTICARE_ROOT / "metadata.parquet"
ABSTRACTS_PARQUET = MULTICARE_ROOT / "abstracts.parquet"
CASE_IMAGES_PARQUET = MULTICARE_ROOT / "case_images.parquet"
CAPTIONS_LABELS_CSV = MULTICARE_ROOT / "captions_and_labels.csv"
MATCHED_CASES_CSV = MULTICARE_ROOT / "leishmaniasis_matched_cases.csv"

# Output files
MULTIMODAL_JSONL = OUTPUT_ROOT / "leishmaniasis_multimodal.jsonl"
MULTIMODAL_JSON = OUTPUT_ROOT / "leishmaniasis_multimodal.json"
IMAGES_DIR = OUTPUT_ROOT / "images"

# Entity extraction outputs
ENTITIES_OUTPUT = OUTPUT_ROOT / "extracted_entities.jsonl"
MESH_PARSED_OUTPUT = OUTPUT_ROOT / "parsed_mesh_terms.jsonl"

# Knowledge Graph outputs
KG_OUTPUT = OUTPUT_ROOT / "leishmaniasis_kg_extended.json"
CASE_ENTITY_LINKS = OUTPUT_ROOT / "case_entity_links.jsonl"

# RAG outputs
RAG_UNITS_OUTPUT = OUTPUT_ROOT / "rag_units.jsonl"
