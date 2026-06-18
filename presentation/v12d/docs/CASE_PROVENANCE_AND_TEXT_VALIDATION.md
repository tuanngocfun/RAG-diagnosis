# Case Provenance And Text Validation

Date: 2026-06-13

## Current Active Case Source

The active V12c case slides use selected examples from the **56-case held-out
evaluation set**. They were checked against the official **121-case Tier A+B
experimental retrieval corpus** for split exclusion.

Active case IDs:

- `PMC7516301_01`
- `PMC7456484_01`
- `PMC10026180_04`

Canonical split artifacts:

- Official experimental retrieval corpus:
  `nonleish_additions/generated/train_phase1b_tierAB.jsonl` (121 rows)
- Held-out cases: `test_p14_v7_normalized.jsonl` (56 rows)
- Presentation queries: `eval_queries_p14_v7_mixed56.jsonl`

Validated split status:

| File role | `PMC7516301_01` | `PMC7456484_01` | `PMC10026180_04` |
| --- | ---: | ---: | ---: |
| `test_p14_v7_normalized.jsonl` | 1 | 1 | 1 |
| `eval_queries_p14_v7_mixed56.jsonl` | 1 | 1 | 1 |
| `train_phase1b_tierAB.jsonl` | 0 | 0 | 0 |

The official 121-case artifact is used here to verify split exclusion. It must
not be described as the runtime evidence source for the saved case outputs. The
backend health metadata records a separate small local defense demo KB, which
supplied the retrieved support shown on slides 16-18.

Use this wording in the defense:

> These three cases are selected examples from the 56-case held-out evaluation
> set. They were verified absent from the official 121-case experimental
> retrieval corpus. The saved outputs used a smaller local defense demo KB.
> They are useful for inspecting input, output, retrieved support, and
> failure modes, but they do not replace the thesis aggregate benchmark.

## Historical Case Outputs

The older V7 files under `data/evaluation_results/` are preserved for
traceability. They used clinical retrieval-corpus examples that are no longer
the active slides 15-22. They should be described, if needed, as historical
functional-demo outputs, not as held-out evaluation-set case analysis.

## Appendix Text Preservation

Slides 20-22 are validated against the `clinical_text` fields in:

- `data/heldout_evaluation_results/PMC7516301_01_result.json`
- `data/heldout_evaluation_results/PMC7456484_01_result.json`
- `data/heldout_evaluation_results/PMC10026180_04_result.json`

Expected result:

- Slide 20 equals `PMC7516301_01_result.json["clinical_text"]`
- Slide 21 equals `PMC7456484_01_result.json["clinical_text"]`
- Slide 22 equals `PMC10026180_04_result.json["clinical_text"]`

The validation extracts the body paragraph from each appendix slide's
`Full blinded clinical input` text panel and compares it exactly against the
corresponding JSON `clinical_text` value.

## RAG/No-RAG Appendix Source

Slides 23-25 use normalized artifacts under
`data/gemma4_rag_norag_comparison/`, extracted from the official Gemma 4
experiment-pipeline RAG and no-RAG runs. They are Q&A-only backup evidence and
must not be described as a three-case aggregate accuracy estimate.

The earlier demo-backend no-RAG attempt is superseded because it still returned
retrieved support chunks.

## Safety Boundary

These slides are a blinded functional evaluation and case-analysis aid. They
are not clinical validation, not deployment evidence, and not diagnosis from
image alone. Images are evidence/context for the model input; the reference
labels are weak/silver evaluation labels from the thesis pipeline rather than
clinician-adjudicated gold labels.
