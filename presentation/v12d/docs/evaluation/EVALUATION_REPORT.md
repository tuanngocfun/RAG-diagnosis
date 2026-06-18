# Held-Out Case Evaluation Report

Date: 2026-06-17  
Model: `google/gemma-4-E4B-it` with 4-bit quantization  
Hardware: NVIDIA TITAN RTX  
Presentation: `FINAL_PRESENTATION.pptx`

## Executive Summary

V12d uses three selected examples from the 56-case held-out evaluation set for
slides 15-18 and appendix slides 20-22. These cases are absent from the
official 121-case Tier A+B experimental retrieval corpus. The active outputs
are aligned to the locked live-demo recapture under `re-capture/20260617_165636`,
using blinded text and image input plus a separate small local defense demo KB
for runtime support.

The purpose is case-level inspection, not a new aggregate accuracy claim. The
reference labels are silver labels from the thesis pipeline, not
clinician-adjudicated ground truth.

## Method

- Source text: `eval_queries_p14_v7_mixed56.jsonl["clinical_context"]`
- Split check: one held-out row, one eval-query row, zero train/retrieval-corpus
  rows for each active case
- Official split-exclusion artifact:
  `nonleish_additions/generated/train_phase1b_tierAB.jsonl` (121 rows)
- Blinding: diagnosis, title, abstract, captions, and confirmatory findings
  withheld from the model input
- Image handling: stale `Leishmania_v3` paths normalized to
  `Leishmaniasis_v3`; each selected case used one rendered PNG thumbnail
- Backend: `http://127.0.0.1:8010/v1/chat`
- Runtime support source: backend-reported small local defense demo KB
- Output directory: `re-capture/20260617_165636/`
- Inference status: locked live-demo recapture; raw JSON is preserved

## Locked Live-Demo Results

| Slide | Case ID | Reference note | Model rank 1 type | Confidence | Runtime |
| ---: | --- | --- | --- | --- | ---: |
| 16 | `PMC7516301_01` | MCL silver reference | MCL | Medium | 78.3s |
| 17 | `PMC7456484_01` | PKDL silver reference | MCL | Medium | 74.0s |
| 18 | `PMC10026180_04` | verified Non-Leish / pseudolabel CL | CL | Medium | 61.5s |

## Interpretation

- `PMC7516301_01`: concordant example where the model rank-1 type matches
  the held-out MCL silver reference.
- `PMC7456484_01`: limitation example where the model stays within the
  leishmaniasis family but shifts the subtype from PKDL toward MCL.
- `PMC10026180_04`: label-conflict and evidence-attribution stress test. The
  verified reference label is Non-Leishmaniasis, the pseudolabel is CL, the
  full-text image/text case is leishmaniasis-plausible, and the live output
  remains in the CL/MCL-family. It is not specificity proof.

These examples support the defense narrative that the system is technically
functional and inspectable, but not clinically validated and not
deployment-ready. Images are used as context, not as standalone diagnostic
proof.

## Slide Updates

- Slide 15: held-out case-analysis setup
- Slide 16: `PMC7516301_01`
- Slide 17: `PMC7456484_01`
- Slide 18: `PMC10026180_04`
- Slide 19: Thank You / Questions
- Slide 20: full blinded input for `PMC7516301_01`
- Slide 21: full blinded input for `PMC7456484_01`
- Slide 22: full blinded input for `PMC10026180_04`
- Slide 23: Q&A-only Gemma 4 RAG/no-RAG comparison for `PMC7516301_01`
- Slide 24: Q&A-only Gemma 4 RAG/no-RAG comparison for `PMC7456484_01`
- Slide 25: Q&A-only Gemma 4 RAG/no-RAG comparison for `PMC10026180_04`

Historical retrieval-corpus case outputs remain in `data/evaluation_results/`
for traceability, but they are no longer the active case slides.

The comparison appendix uses the official Gemma 4 experiment-pipeline runs,
not the failed demo-backend no-RAG attempt.

## Safety Boundary

This is a research demonstration of a local multimodal RAG system. It is not
clinical deployment evidence, not clinician adjudication, and not diagnosis from
image alone. The labels are weak/silver reference labels from the thesis
pipeline.
