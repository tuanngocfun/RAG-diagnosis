# Final Presentation QA - V12d

Date: 2026-06-17

## Deliverables

- `FINAL_PRESENTATION.pptx`: 31 slides
- `FINAL_PRESENTATION.pdf`: 31 pages
- `qa/final_render/slide-01.png` through `slide-31.png`
- `qa/final_render/contact-sheet.png`

## Content Checks

- Slides 1-18 form the spoken path; slide 19 closes; slides 20-31 are Q&A
  appendix only.
- Slide 15 labels the cases as illustrative and states that aggregate thesis
  metrics remain the benchmark.
- Slides 16-18 use the locked live-demo recapture and LLM council review
  synthesis: `PMC7516301_01` MCL -> MCL, `PMC7456484_01` PKDL -> MCL subtype
  mismatch, and `PMC10026180_04` verified Non-Leish / pseudolabel CL -> live
  CL/MCL-family label-conflict stress test.
- Slides 20-22 preserve the complete blinded clinical inputs, with slide 21
  using a two-column layout for the long PKDL case.
- Slides 23-25 use official Gemma 4 experiment-pipeline RAG/no-RAG comparison
  artifacts and are marked Q&A backup only.
- Slides 26-31 use exact official RAG trace and fresh real-GPU backend audit
  artifacts and are marked Q&A trace/audit only.
- The official trace slides show the rerank-enabled final context list used for
  generation (`retriever_method=hybrid`, `rerank=true`, `retrieval_top_k=20`);
  they do not claim a separate pre-rerank list.
- The fresh real-GPU audit slides use the locked recapture request IDs prefixed
  `v12d_recapture_live_gpu_` and preserve the exact extracted rank-1 fields
  from the raw backend JSON. The active slide story no longer treats case 3 as
  a clean specificity success; it is a label-conflict/evidence-attribution
  stress test.
- The failed demo-backend no-RAG attempt is superseded and not used as V12d
  evidence because it still returned retrieved support chunks.
- No active slide claims readiness for clinical deployment, image-only
  diagnosis, or a three-case aggregate accuracy estimate.
- A locked live-demo recapture was run on 2026-06-17 and saved under
  `re-capture/20260617_165636`. It is the source of the active case slide
  mapping and the latest fresh-audit appendix fields.

## Structural Checks

- PPTX ZIP integrity passed on 2026-06-15.
- PPTX and PDF both contain 31 pages/slides.
- `FINAL_PRESENTATION.pdf` creation time: 2026-06-15 12:02:55 +07.
- Slide size must remain 10 x 5.625 inches.
- No shape should extend beyond the slide canvas.
- The selected active case IDs must appear once in the held-out test file and
  once in the mixed-56 eval-query file, and zero times in the canonical
  121-case Tier A+B retrieval-corpus artifact.
- `scripts/validate_presentation.py` checks silver-label and label-conflict
  wording, corpus/runtime provenance, appendix text preservation,
  comparison-slide source labels, and exact trace/audit slide fields against
  the official trace plus the locked live recapture.
  Final run passed for the rebuilt V12d deck.

## Visual Checks

- Rendered all 31 PDF pages at 160 DPI after export.
- Inspected slides 15-31 individually for clipping, overlap, stale images,
  unreadable dense text, and page-badge collisions.
- Contact sheet spot-check confirms slides 1-14 are preserved from the
  validated base.

## Inference Provenance

Slides 16-18 use the locked Gemma 4 local-demo recapture and explicitly
distinguish the official 121-case experimental retrieval corpus from the small
local defense demo KB used by those outputs.

Slides 23-25 use separate official Gemma 4 experiment-pipeline artifacts:

- RAG:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`
- No-RAG:
  `p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411`

The comparison appendix supports selected case-level Q&A only. Aggregate thesis
metrics remain the benchmark.

Slides 26-31 are stricter audit evidence:

- Official RAG trace:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`
- Locked live GPU recapture:
  `re-capture/20260617_165636`

The live backend exposes lexical retrieval scores and returned evidence chunks,
but no separate re-ranker contract. The official pipeline trace is therefore
the valid source for the rerank-enabled context-list evidence.
