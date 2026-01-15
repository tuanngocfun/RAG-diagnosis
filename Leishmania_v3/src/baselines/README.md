# Baseline Methods (No Knowledge Graph)

These baselines do NOT use Knowledge Graph enhancement.
Use for comparison to demonstrate the value of KG-enhanced retrieval.

## Files

| File | Description |
|------|-------------|
| `text_only_rag.py` | Text embeddings only (SentenceTransformers/TF-IDF) |
| `image_only.py` | Image embeddings only (CLIP) |
| `combined_no_kg.py` | Text + Image without KG |

## Usage

```bash
# Text-only baseline
python text_only_rag.py --data ../../data/leishmaniasis_multimodal/train.jsonl

# Image-only baseline
python image_only.py --images ../../data/leishmaniasis_multimodal/images/

# Combined without KG
python combined_no_kg.py --data ../../data/leishmaniasis_multimodal/train.jsonl
```

## Installation

```bash
pip install sentence-transformers scikit-learn numpy
pip install torch torchvision  # For image baselines
pip install git+https://github.com/openai/CLIP.git  # For CLIP
```
