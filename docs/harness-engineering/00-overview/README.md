# Overview

The RAG diagnosis harness governs three linked surfaces:

- Experiment pipeline: retrieval, reranking, generation, parsing, and metrics.
- Live demo: local GPU backend, Flutter UI, retriever audit, and failure modes.
- Defense deck: V12d slides, appendix evidence, raw JSON traces, and QA notes.

Every public claim should map to an artifact: code, config, JSON/JSONL, slide
source, validation output, or documented reviewer synthesis.

## Success Criteria

- Rebuildable code and tests are committed.
- Heavy data and raw archives are represented by manifests, not accidental Git
  blobs.
- V12d remains bounded as thesis evidence, not a clinical product.
- Demo outputs distinguish official replay, fresh GPU audit, and retrieved
  evidence paths.
