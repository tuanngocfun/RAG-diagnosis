# Historical V5 / V7 Comparison

Date: 2026-06-13

## Historical Decision

This document is retained for lineage only. It explains why V7 superseded V5 at
that point in the presentation history.

The current canonical package is V12c:

- `FINAL_PRESENTATION.pptx`
- `FINAL_PRESENTATION.pdf`

V12c keeps the safer v11a/v12b spoken path and adds Q&A-only Gemma 4
experiment-pipeline RAG/no-RAG comparison slides.

## What V5 Fixed

- Rebuilt the evaluation deck as 21 slides.
- Kept slides 16-18 compact instead of showing truncated clinical paragraphs.
- Added full blinded-input appendix slides so supervisors can inspect the full
  case text when needed.
- Preserved the three historical functional demo/Q&A cases that were later
  superseded by held-out evaluation-set examples.

## What V7 Added Historically

- 22-slide canonical package with `FINAL_PRESENTATION.pptx` and
  `FINAL_PRESENTATION.pdf`.
- Standardized VGU/h_da title and closing-slide branding.
- Consistent top-right page badges on content and appendix slides.
- Slide 19 as an explicit Thank You / Q&A bridge.
- Correct appendix numbering: full blinded inputs are slides 20-22.
- Refined slides 15-18 from historical retrieval-corpus demo cases into three
  selected held-out evaluation-set case-analysis examples.
- Organized `source/`, `scripts/`, `docs/`, `qa/`, `deliverables/`, and
  `archive/` folders.
- Rendered slide QA in `qa/final_render/` and package traceability through
  `MANIFEST.sha256`.

## Current V12c Use Path

- Slides 1-18 are the spoken colloquium deck.
- Slide 19 closes the talk and opens committee questions.
- Slides 20-22 are appendix-only evidence for supervisor questions about the
  blinded case inputs.
- Slides 23-25 are Q&A-only Gemma 4 RAG/no-RAG comparison evidence from the
  official experiment pipeline.
- Slides 15-18 use selected cases from the 56-case held-out evaluation set.
  They should be described as case-analysis examples, not as a replacement for
  the thesis aggregate benchmark.

This follows the practical direction from MIT Communication Lab materials:
slides should support the spoken argument, and technical demonstrations should
be paced, contextualized, visible, and risk-managed. It also matches the oral
exam expectation reflected in Stanford policy: a defense can include a seminar
component and committee questioning, so the main deck should make the research
claim clear while the appendix remains ready for rigorous inspection.

## Safety Boundary

V12c should be described as a functional research evaluation deck for a local
multimodal RAG system. It is not clinical validation, not deployment evidence,
and not an image-only diagnosis workflow. Images are evidence/context for
retrieval and reasoning, while final labels depend on clinical text and
confirmatory case-report information outside the blinded prompt.

## External Reference Notes

- MIT CommLab, Slide Design:
  https://mitcommlab.mit.edu/aeroastro/commkit/slide-design/
- MIT CommLab, Technical Demonstrations:
  https://mitcommlab.mit.edu/aeroastro/commkit/technical-demonstrations/
- Stanford GAP, University Oral Examinations:
  https://gap.stanford.edu/handbooks/gap-handbook/chapter-4/subchapter-7/page-4-7-1
- Stanford GAP, Dissertations:
  https://gap.stanford.edu/handbooks/gap-handbook/chapter-4/subchapter-8/page-4-8-1
- Harvard Griffin GSAS, Dissertation:
  https://gsas.harvard.edu/academics/dissertation
- UCLA Graduate Division, Thesis and Dissertation Filing Requirements:
  https://grad.ucla.edu/academics/graduate-study/thesis-and-dissertation-filing-requirements/

Oxford and UCL pages were not used as primary sources in this local package
because their official pages returned anti-bot challenge pages in this
environment.
