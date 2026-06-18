# V12c Real-Model Revalidation

Date: 2026-06-15

## Verdict

The slide outputs are not fabricated. Slides 16-18 trace to saved real local-GPU
Gemma 4 backend outputs in `data/heldout_evaluation_results/`. Slides 23-25
trace to official Gemma 4 experiment-pipeline RAG/no-RAG answer files.

A fresh non-mutating real-model rerun was also completed on 2026-06-15 and
saved separately under:

`data/revalidation_real_model_20260615-1023/`

This fresh run confirms that the backend is real:

- Model: `google/gemma-4-E4B-it`
- Provider mode: `real_gpu_gemma4`
- GPU: NVIDIA TITAN RTX
- CUDA available: true
- Backend status: ok

## Main Case Slides

| Case | Deck saved rank-1 type | Fresh rerun rank-1 type | Interpretation |
| --- | --- | --- | --- |
| `PMC7516301_01` | MCL | MCL | Reproduced the concordant MCL story. |
| `PMC7456484_01` | MCL | CL | Still a leishmaniasis-family subtype mismatch, but not exact subtype reproduction. |
| `PMC10026180_04` | Non-Leishmaniasis | Non-Leishmaniasis | Historical June 15 run only; superseded by the June 17 label-conflict remapping. |

The important nuance is case 2. The V12c slide value `PKDL -> MCL` is a real
saved model output, but a fresh rerun returned `PKDL -> CL`. Both support the
same defense-safe interpretation: the model stays in the leishmaniasis family
but does not resolve the PKDL subtype. Do not promise exact deterministic
reproduction for case 2 in a live demo.

## RAG/No-RAG Appendix Slides

Slides 23-25 were checked directly against the original official answer files:

- No-RAG:
  `p14_v7_phase1b_tierAB_official_latency_generation_only_multimodal_norag_gemma4_20260423_183411/answers_norag.jsonl`
- RAG:
  `p14_v7_phase1b_tierAB_official_post_gemini_freeze_base_20260419_031439_seed42_gemma4_20260424_014935/answers_latency.jsonl`

Historical comparison before the June 17 label-conflict remapping:

| Case | Silver reference | No-RAG rank-1 type | RAG rank-1 type | Retrieved contexts |
| --- | --- | --- | --- | --- |
| `PMC7516301_01` | MCL | Non-Leishmaniasis | MCL | 0 / 3 |
| `PMC7456484_01` | PKDL | MCL | MCL | 0 / 1 |
| `PMC10026180_04` | verified Non-Leish / pseudolabel CL | Non-Leishmaniasis | Non-Leishmaniasis | 0 / 0 |

## Defense Guidance

This document is retained as historical revalidation evidence. Current V12d
uses the locked June 17 recapture and presents `PMC10026180_04` as a
label-conflict/evidence-attribution stress test, not specificity proof.

The final claim remains bounded: selected case-level inspection only; aggregate
thesis metrics remain the benchmark; not clinical validation and not deployment
evidence.
