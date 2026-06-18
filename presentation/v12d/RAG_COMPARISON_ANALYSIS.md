# RAG Comparison Analysis - V12d

Date: 2026-06-17

## Decision

V12d does not use the demo-backend no-RAG attempt as evidence. That attempt sent
flags intended to disable retrieval, but the backend still returned retrieved
support chunks.

V12d instead uses official Gemma 4 experiment-pipeline artifacts:

- RAG:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`
- No-RAG:
  `p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411`

## Safe Interpretation

The selected cases are useful for Q&A:

- Case 1 changes from Non-Leishmaniasis without retrieval to MCL with retrieval.
- Case 2 remains a subtype-resolution challenge.
- Case 3 is now treated as a label-conflict/evidence-attribution stress test.
  The official comparison artifact returns Non-Leishmaniasis in both
  conditions, but the verified label, pseudolabel, full-text image/text case,
  live demo output, and LLM council reviews conflict. Do not present it as
  specificity proof.

These observations are selected case-level examples, not an aggregate accuracy
estimate. The thesis-level matched metrics remain the benchmark.
