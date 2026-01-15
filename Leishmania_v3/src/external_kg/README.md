# External KG Methods (Q1 Journal Sources)

Adapters for established KG construction methods from Q1 journals.

## Methods

| Method | Journal | Adapter |
|--------|---------|---------|
| AutoRD | JMIR Med Inform (Q1) | `autord_adapter.py` |
| PheKnowLator | Bioinformatics (Q1) | `pheknowlator_adapter.py` |

## Usage

### AutoRD Adapter
```bash
# Convert your data to AutoRD format
python autord_adapter.py

# Outputs:
# - autord_input.jsonl (for running with OpenAI)
# - autord_output.json (your KG in AutoRD triple format)
```

### PheKnowLator Adapter
```bash
# Convert your KG to PheKnowLator RDF format
python pheknowlator_adapter.py

# Outputs:
# - pkt_nodes.json
# - pkt_edges.json
# - edge_list.txt
```

## Running Actual AutoRD (requires OpenAI API)

```bash
# 1. Set API key
echo 'OPENAI_API_KEY = "sk-xxx"' > ../../data/kg_ref/AutoRD/env.py

# 2. Run
cd ../../data/kg_ref/AutoRD && bash run.sh
```

## Running PheKnowLator (requires pkt_kg)

```bash
pip install pkt_kg
cd ../../data/kg_ref/PheKnowLator
python Main.py --app instance --kg full
```
