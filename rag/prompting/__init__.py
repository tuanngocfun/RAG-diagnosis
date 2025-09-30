"""
RAG Prompting Module - Standalone MedGemma-4B-IT Implementation

This module provides standalone multimodal medical analysis using MedGemma-4B-IT
without requiring a RAG pipeline. It directly processes medical images and text
for diagnostic analysis.

Key Components:
- medgemma4b_standalone.py: Core standalone MedGemma implementation
- run_batch_medgemma4b_standalone.py: Batch processing script

Usage Examples:

1. Direct Analysis:
```python
from rag.prompting.medgemma4b_standalone import StandaloneMedGemma4B

analyzer = StandaloneMedGemma4B()
result = analyzer.analyze_images(["/path/to/image.png"], "What diagnosis do you see?")
print(result)
```

2. Case-based Diagnosis:
```python
diagnosis = analyzer.diagnose_case("case_001", "What is the most likely diagnosis?")
print(diagnosis)
```

3. Batch Processing:
```bash
python -m rag.prompting.run_batch_medgemma4b_standalone \
    --manifest /app/kaggle/working2/rag_knowledge_base/questions_manifest.jsonl \
    --out /app/output/answers_standalone.ndjson \
    --images_per_answer 3 \
    --resume
```
"""

from .medgemma4b_standalone import (
    StandaloneMedGemma4B,
    StandaloneCFG,
    create_medgemma_analyzer,
    analyze_medical_images,
    batch_analyze_cases
)

__all__ = [
    'StandaloneMedGemma4B',
    'StandaloneCFG', 
    'create_medgemma_analyzer',
    'analyze_medical_images',
    'batch_analyze_cases'
]