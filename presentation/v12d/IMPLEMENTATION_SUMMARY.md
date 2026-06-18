# V12c Implementation Summary

## Base Selection

V12c uses the safer v11a/v12b defense deck as the spoken base. The
`comparison/v11a-v11b` review favored that path because it has stronger
silver-label wording, clearer corpus/runtime provenance, cleaner validation, and
a more defensible PKDL case narrative.

## Implemented Changes

- Kept slides 1-18 as the 30-minute spoken path and slide 19 as the close.
- Preserved the three active held-out case examples:
  `PMC7516301_01`, `PMC7456484_01`, and `PMC10026180_04`.
- Kept slides 20-22 as full blinded-input Q&A appendix slides, including the
  two-column PKDL text layout.
- Added slides 23-25 as Q&A-only Gemma 4 experiment-pipeline RAG/no-RAG
  comparison slides.
- Added `scripts/extract_gemma4_rag_norag_comparison.py` to normalize the
  official Gemma 4 RAG and no-RAG experiment artifacts.
- Updated `scripts/validate_presentation.py` for 25 slides, paragraph-level
  appendix text preservation, and comparison-source/overclaim guards.
- Marked the earlier demo-backend no-RAG attempt as superseded because it still
  returned retrieved support chunks.

## Verification Target

The final V12c package should pass:

- `unzip -t FINAL_PRESENTATION.pptx`
- `pdfinfo FINAL_PRESENTATION.pdf` with 25 pages
- `scripts/validate_presentation.py`
- visual inspection of rendered slides 15-25

## Defense Boundary

Slides 23-25 support selected case-level Q&A only. They are not a performance
sample and do not replace the thesis aggregate RAG/no-RAG metrics.
