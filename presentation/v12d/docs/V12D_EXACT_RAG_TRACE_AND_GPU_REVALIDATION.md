# V12d Exact RAG Trace And GPU Revalidation

Date: 2026-06-17

## Purpose

V12d aligns the active case slides to the locked live-demo recapture and
appends Q&A-only audit slides. Slides 1-18 remain the 30-minute spoken route,
slide 19 is the close, slides 20-25 remain appendix evidence, and slides 26-31
expose exact retriever/rerank and backend-output evidence for supervisor
questions.

## Evidence Sources

Official RAG trace source:

`/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000/runs/p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935`

The V12d extraction reads `retrieval.jsonl`, `answers_latency.jsonl`,
`run_config.json`, and `summary.json`. The run records
`retriever_method=hybrid`, `rerank=true`, and `retrieval_top_k=20`.

Locked live-GPU backend source:

`re-capture/20260617_165636`

The locked recapture used `model_name=google/gemma-4-E4B-it`,
`provider_mode=real_gpu_gemma4`, `device_platform=v12d_recapture_live_demo_alignment`,
and request IDs prefixed with `v12d_recapture_live_gpu_`.

## Source Boundary

The official experiment pipeline is the valid source for rerank-enabled RAG
evidence. Its `retrieval.jsonl` rows are treated as the final context lists used
for generation. V12d does not claim a separate pre-rerank/post-rerank pair,
because the artifact does not expose such a pair.

The live backend provides exact returned evidence chunks and lexical retrieval
scores, but it does not expose a separate re-ranker contract. Those fresh outputs
are therefore used as backend audit evidence, not as the official reranker trace.

## Case-Level Audit Notes

- `PMC7516301_01`: locked live recapture returned rank-1 type `MCL`.
- `PMC7456484_01`: locked live recapture returned rank-1 type `MCL`, supporting
  the PKDL-to-MCL subtype-resolution limitation.
- `PMC10026180_04`: locked live recapture returned rank-1 type `CL` while the
  verified reference label is Non-Leishmaniasis and the pseudolabel is CL. This
  is presented as label-conflict/evidence-attribution stress-test evidence, not
  specificity proof.

The three selected examples are not an aggregate accuracy estimate. Aggregate
thesis metrics remain the benchmark, and the material is not clinical
validation or deployment-readiness evidence.

## Generated Artifacts

- `data/exact_rag_trace_appendix/trace_summary.json`
- `data/exact_rag_trace_appendix/PMC7516301_01_trace.json`
- `data/exact_rag_trace_appendix/PMC7456484_01_trace.json`
- `data/exact_rag_trace_appendix/PMC10026180_04_trace.json`
- Slides 26, 28, 30: official RAG retrieval/rerank trace.
- Slides 27, 29, 31: locked live-GPU output audit.
