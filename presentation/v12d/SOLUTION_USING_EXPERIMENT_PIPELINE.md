# Solution: Use The Experiment Pipeline For V12c Backup Evidence

Date: 2026-06-15

The demo backend is designed for live retrieval-supported demonstration, not for
clean ablation. The V12c no-RAG comparison therefore uses the experiment
pipeline, where the no-RAG condition is recorded with zero retrieved contexts.

## V12c Path

1. Normalize the official Gemma 4 RAG/no-RAG artifacts with
   `scripts/extract_gemma4_rag_norag_comparison.py`.
2. Build the deck with `scripts/build_evaluation_slides.py`.
3. Present slides 23-25 only if a supervisor asks what retrieval changes on the
   selected held-out examples.

## Boundary

The comparison appendix supports selected case-level discussion. It is not a
replacement for the thesis aggregate RAG/no-RAG evaluation.
