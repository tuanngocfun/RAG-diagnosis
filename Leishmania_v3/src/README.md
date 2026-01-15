# Leishmania_v3 Source Code

This folder contains all source code for the Leishmaniasis Knowledge Graph project.

## Folder Structure

```
src/
├── multicare_pipeline/     # YOUR METHOD (main contribution)
│   ├── 00_config/          # Configuration and entity schema
│   ├── 01_data_extraction/ # MeSH/keyword parsing
│   ├── 02_entity_extraction/ # NER pipeline
│   ├── 03_kg_building/     # Knowledge Graph construction
│   └── 04_multimodal_linking/ # Train/test split
│
├── baselines/              # BASELINE METHODS (no KG)
│   ├── text_only_rag.py    # Text embedding retrieval
│   ├── image_only.py       # Image embedding retrieval
│   └── combined_no_kg.py   # Multimodal without KG
│
├── external_kg/            # EXTERNAL Q1 METHODS
│   ├── autord_adapter.py   # AutoRD (JMIR Med Inform Q1)
│   └── pheknowlator_adapter.py # PheKnowLator (Bioinformatics Q1)
│
└── evaluation/             # EVALUATION FRAMEWORK
    ├── metrics.py          # Precision, Recall, F1, MRR
    └── compare_methods.py  # Cross-method comparison
```

## Data Location

Data files remain in `../data/`:
- `leishmaniasis_multimodal/` - Processed multimodal dataset
- `whole_multicare_dataset/` - Raw MultiCaRe files
- `kg_ref/` - Reference repos (AutoRD, PheKnowLator)

## Quick Start

```bash
# 1. Run your KG pipeline
cd multicare_pipeline
python 01_data_extraction/parse_mesh_keywords.py
python 02_entity_extraction/run_ner_extraction.py
python 03_kg_building/build_kg_entities.py
python 04_multimodal_linking/split_train_test.py

# 2. Run baselines for comparison
cd ../baselines
python text_only_rag.py --data ../../data/leishmaniasis_multimodal/train.jsonl

# 3. Compare methods
cd ../evaluation
python compare_methods.py
```

## Q1 Journal Sources

| Method | Journal | Q1 Status |
|--------|---------|-----------|
| AutoRD | JMIR Med Inform | ✅ Q1 (CiteScore 7.7) |
| PheKnowLator | Bioinformatics | ✅ Q1 |
