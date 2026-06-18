# Thesis Defense Presentation V12d

## Canonical Artifacts

Use:

- `FINAL_PRESENTATION.pptx` for the editable 31-slide deck.
- `FINAL_PRESENTATION.pdf` for presentation delivery.
- `qa/final_render/contact-sheet.png` for a full-deck visual check.

Spoken route:

- Slides 1-18: thesis story and three illustrative held-out cases.
- Slide 19: close and Q&A bridge.
- Slides 20-22: appendix-only full blinded inputs.
- Slides 23-25: appendix-only Gemma 4 RAG/no-RAG comparison for Q&A.
- Slides 26-31: appendix-only exact official RAG trace and fresh real-GPU
  output audit evidence.

V12d now aligns the active case slides to the locked live-demo recapture under
`re-capture/20260617_165636`. The active case slides use silver/pipeline label
language, the case-review evidence under `cases/`, and fresh local-GPU demo
outputs. Case 3 is explicitly a label-conflict/evidence-attribution stress
test: the verified reference is Non-Leishmaniasis, the pseudolabel is CL, and
the live/demo-review evidence is leishmaniasis-plausible.

## V12d Decision

- Use the v11a/v12b main deck story because it has safer claim boundaries,
  clearer provenance, and a more defensible PKDL case-2 narrative.
- Keep v11b as audit evidence only; its fresh case-2 output changes the story
  to a stronger sensitivity/family-level miss.
- Use the Gemma 4 experiment pipeline for RAG/no-RAG backup evidence, not the
  demo backend no-RAG attempt.
- Do not treat the three selected examples as an aggregate accuracy estimate.
  Aggregate thesis metrics remain the benchmark.

## RAG/No-RAG Appendix

The comparison appendix is generated from:

- RAG run:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`
- No-RAG run:
  `p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411`

The earlier demo-backend no-RAG attempt is superseded because the backend still
returned retrieved support chunks. It is not used as V12d evidence.

## Exact Trace Appendix

Slides 26, 28, and 30 use the official Gemma 4 experiment-pipeline RAG run:

- RAG trace:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`

Those slides show the rerank-enabled final context list used for generation:
`retriever_method=hybrid`, `rerank=true`, and `retrieval_top_k=20`. They do not
claim a separate pre-rerank list.

Slides 27, 29, and 31 use the latest locked live-demo recapture saved under
`re-capture/20260617_165636`. This evidence is Q&A audit material and is also
the source used to align the active case slides.

## Rebuild

From this directory:

```bash
python3 scripts/refresh_preserved_result_provenance.py
python3 scripts/extract_gemma4_rag_norag_comparison.py
python3 scripts/extract_exact_rag_trace_appendix.py
python3 scripts/build_evaluation_slides.py
python3 scripts/standardize_presentations.py \
  source/thesis_defense_slides_v5_with_evaluation.pptx \
  FINAL_PRESENTATION.pptx \
  --logo assets/logos/vgu_hda_lockup.png
libreoffice --headless --convert-to pdf --outdir . FINAL_PRESENTATION.pptx
python3 scripts/validate_presentation.py
```

Use the project venv if system Python lacks `python-pptx`:

```bash
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python3 scripts/validate_presentation.py
```

## Verification

- `docs/qa/FINAL_QA.md`: final V12d QA record.
- `docs/V12D_EXACT_RAG_TRACE_AND_GPU_REVALIDATION.md`: exact trace and fresh
  real-GPU audit record.
- `docs/V12C_REAL_MODEL_REVALIDATION_20260615.md`: fresh non-mutating
  real-model revalidation audit.
- `docs/V12C_FINAL_DECISION_AND_QA_APPENDIX.md`: base selection and comparison
  appendix rationale.
- `docs/HELDOUT_CASE_SELECTION_AND_VALIDATION.md`: case provenance.
- `docs/V12C_30MIN_COLLOQUIUM_RUN_OF_SHOW.md`: timing and speaking route.
- `docs/V12C_SPEAKER_NOTES.md`: slide-level talk track and Q&A defenses.
- `MANIFEST.sha256`: package checksums.

Files in `archive/` and documents with V5/V7/V11A names are retained as
historical lineage, not canonical V12d instructions.
