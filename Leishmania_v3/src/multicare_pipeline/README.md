# MultiCaRe Data Pipeline for Leishmaniasis Knowledge Graph

This folder contains organized scripts for building a multimodal Knowledge Graph from the MultiCaRe dataset.

## Pipeline Overview

```
multicare_pipeline/
├── 00_config/              # Configuration and schema definitions
│   ├── config.py           # Paths and settings
│   └── entity_schema.py    # KG entity definitions (13 pathogens, 22 drugs, 26 procedures)
│
├── 01_data_extraction/     # Extract data from MultiCaRe
│   └── parse_mesh_keywords.py      # Parse mesh_terms + major_mesh_terms + keywords
│
├── 02_entity_extraction/   # Extract clinical entities via NER
│   └── run_ner_extraction.py       # Pattern-based NER (+ optional scispaCy)
│
├── 03_kg_building/         # Build Knowledge Graph
│   └── build_kg_entities.py        # Create entity nodes and case-entity links
│
└── 04_multimodal_linking/  # Train-test split  
    └── split_train_test.py         # Stratified train-test split
```

## Running the Pipeline (IN ORDER!)

⚠️ **If you edit any script, you must re-run from that step onwards:**

```bash
cd /home/students/Leishmania/Leishmania_v3/data

# STEP 0: Build multimodal dataset (includes major_mesh_terms)
python build_multimodal_dataset.py

# STEP 1: Parse MeSH terms and keywords  
python multicare_pipeline/01_data_extraction/parse_mesh_keywords.py

# STEP 2: Extract clinical entities via NER
python multicare_pipeline/02_entity_extraction/run_ner_extraction.py

# STEP 3: Build Knowledge Graph
python multicare_pipeline/03_kg_building/build_kg_entities.py

# STEP 4: Create train-test split (default 80/20)
python multicare_pipeline/04_multimodal_linking/split_train_test.py --ratio 0.8
```

## Train-Test Split Strategy

Based on Q1 journal standards (JAMIA, Journal of Biomedical Informatics):

1. **Stratified by entity type** - maintains distribution across splits
2. **Rare entities (≤3 cases) → training** - prevents test-only entities
3. **Case-level split** - no data leakage between patients
4. **Verification** - ensures all test entities appear in training

## Data Sources

| Source | Contains | Entity Types |
|--------|----------|--------------|
| `mesh_terms` | "Leishmaniasis, Visceral / diagnosis" | Disease, Procedure |
| `major_mesh_terms` | Primary MeSH headings | Disease, Drug |
| `keywords` | "visceral leishmaniasis", "miltefosine" | Disease, Drug |
| `case_text` | Full clinical narrative | All entities via NER |
