# Validation Report

Date: 2026-06-18

Branch: `codex/v12d-rag-defense-harness`

## Passed

- V12d presentation validator:
  `python3 presentation/v12d/scripts/validate_presentation.py`
  - Result: passed
  - Scope: 31 slides/pages, provenance wording, comparison appendix, exact
    trace appendix, and safety-claim guards.
- PPTX integrity:
  `unzip -t presentation/v12d/FINAL_PRESENTATION.pptx`
  - Result: passed
- PDF integrity:
  `pdfinfo presentation/v12d/FINAL_PRESENTATION.pdf`
  - Result: 31 pages
- RAG tests:
  `PYTHONPATH=rag QDRANT_URL=http://localhost:6333 QDRANT_API_KEY=dummy GOOGLE_API_KEY=dummy pytest rag/tests`
  - Result: 67 passed, 1 skipped
- Flutter backend:
  `PYTHONPATH=demo/flutter/backend pytest demo/flutter/backend/tests`
  - Result: 25 passed
- Flutter app:
  `flutter analyze`
  - Result: no issues
- Flutter app:
  `flutter test`
  - Result: all 6 tests passed
- Flutter app:
  `flutter build web --pwa-strategy=none`
  - Result: built `build/web`

## Notes

- The skipped RAG test requires historical local fixture files outside the
  curated repository:
  `/home/ngocnt/Leishmania_v3/rag/testing/multimodal/v7/out/...`.
- Dummy API values were used only to satisfy import-time configuration during
  offline unit tests. No secrets were committed.
- Generated Flutter build/cache folders remain ignored and should not be
  staged.
