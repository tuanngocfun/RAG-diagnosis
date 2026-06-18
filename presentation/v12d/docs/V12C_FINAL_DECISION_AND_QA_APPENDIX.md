# V12c Final Decision And Q&A Appendix

## Final Base Choice

V12c uses the v11a/v12b defense-ready deck as the main base. This follows the
`comparison/v11a-v11b` review: v11a is safer for a thesis defense because it has
cleaner silver-label wording, clearer provenance, stronger validation, and a
more defensible case-2 narrative.

V11b remains useful audit evidence, but it is not the main deck base because its
fresh case-2 result changes the PKDL example into a stronger family-level miss.

## Spoken Path

- Slides 1-18 are the 30-minute spoken path.
- Slide 19 is the close and Q&A bridge.
- Slides 20-25 are appendix-only evidence for supervisor questions.

The active case slides remain:

- `PMC7516301_01`: MCL silver reference, model rank 1 MCL.
- `PMC7456484_01`: PKDL silver reference, model rank 1 MCL subtype mismatch.
- `PMC10026180_04`: Non-Leishmaniasis silver reference, model rank 1
  Non-Leishmaniasis.

## RAG/No-RAG Backup

Slides 23-25 add selected case-level RAG/no-RAG backup evidence from official
Gemma 4 experiment-pipeline artifacts:

- RAG:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`
- No-RAG:
  `p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411`

The appendix wording is intentionally bounded: the selected cases illustrate
retrieval behavior, but they are not an aggregate accuracy estimate. Aggregate
thesis metrics remain the benchmark.

## Superseded Attempt

The earlier demo-backend no-RAG attempt is not used as evidence. Its request
flags attempted to disable retrieval, but the backend still returned retrieved
support chunks. V12c therefore uses the experiment pipeline, where the no-RAG
run records zero retrieved contexts.

## Defense Boundary

V12c remains a research evaluation deck. It does not claim clinical validation,
deployment readiness, or diagnosis from images alone.
