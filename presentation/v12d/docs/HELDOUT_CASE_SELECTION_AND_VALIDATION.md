# Held-Out Case Selection And Validation

Date: 2026-06-17

## Selection Rationale

V12d uses three selected cases from the 56-case held-out evaluation set so the
presentation can inspect success, subtype limitation, and label-conflict
behavior without returning to retrieval-corpus examples.

| Slide | Case ID | Reference label | Role in talk |
| ---: | --- | --- | --- |
| 16 | `PMC7516301_01` | Mucocutaneous Leishmaniasis (MCL) | Concordant held-out MCL example |
| 17 | `PMC7456484_01` | Post-Kala-Azar Dermal Leishmaniasis (PKDL) | Subtype-differentiation limitation |
| 18 | `PMC10026180_04` | Verified Non-Leishmaniasis / pseudolabel CL | Label-conflict and evidence-attribution stress test |

Selection constraints:

- Present exactly once in `test_p14_v7_normalized.jsonl`.
- Present exactly once in `eval_queries_p14_v7_mixed56.jsonl`.
- Absent from
  `nonleish_additions/generated/train_phase1b_tierAB.jsonl`, the official
  121-case Tier A+B experimental retrieval corpus.
- At least one image path resolves through `LEISH_IMAGE_ROOT` or a supported
  legacy/shared-mount fallback.

## Locked Live-Demo Outputs

V12d uses the locked live-demo recapture under `re-capture/20260617_165636`
for slides 16-18. The full case-review folder was inspected before remapping
`PMC10026180_04`; it is no longer treated as a clean specificity
success.

Backend summary:

- Model: `google/gemma-4-E4B-it`
- Provider mode: `real_gpu_gemma4`
- GPU: NVIDIA TITAN RTX
- Image tensor count: 1 for each selected case
- Output directory: `re-capture/20260617_165636/`
- Runtime support source: backend-reported small local defense demo KB

Observed rank-1 outputs from the locked recapture:

| Case ID | Reference note | Model rank 1 type | Confidence | Runtime |
| --- | --- | --- | --- | ---: |
| `PMC7516301_01` | MCL silver reference | MCL | Medium | 78.3s |
| `PMC7456484_01` | PKDL silver reference | MCL | Medium | 74.0s |
| `PMC10026180_04` | verified Non-Leish / pseudolabel CL | CL | Medium | 61.5s |

Interpretation for the defense:

- Do not present these three cases as a statistically meaningful accuracy
  estimate.
- Present them as readable, inspectable examples of the same system behavior
  described by the aggregate thesis results.
- State that the official 121-case artifact verifies case exclusion, while the
  preserved outputs retrieved support from a separate local defense demo KB.
- For `PMC7456484_01`, say the model stayed within the leishmaniasis family but
  shifted the subtype toward MCL rather than PKDL.
- For `PMC10026180_04`, say the verified label, pseudolabel, image/text, live
  output, and LLM council reviews conflict. Present it as a label-conflict and
  evidence-attribution stress test, not as specificity proof.

## Visual QA

Rendered slides `15-25` from `FINAL_PRESENTATION.pdf` were inspected after the
rebuild. The active case slides and appendix slides showed no clipping,
overflow, stale corpus-case render, or image/text overlap.

Appendix behavior:

- Slide 20 includes text plus image thumbnail.
- Slide 21 uses a two-column text layout because the blinded PKDL input is long.
- Slide 22 includes text plus image thumbnail.
- Slides 23-25 are Q&A-only Gemma 4 RAG/no-RAG comparison slides from the
  official experiment pipeline, not the demo backend.

## Boundary Language

Use this line if asked why only three cases are shown:

> These three cases are selected held-out examples for inspection. The aggregate
> thesis metrics remain the benchmark. Their labels are silver references, and
> the saved outputs use a local defense demo KB.
