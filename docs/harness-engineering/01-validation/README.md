# Validation Gates

Use layered gates rather than one giant check.

## Pipeline

- Run `pytest rag/tests` for parser, retrieval, metrics, and prompt-wiring
  coverage.
- Preserve split provenance for held-out evaluation cases.
- Keep aggregate metrics separate from selected case examples.

## Presentation

- Run `python3 presentation/v12d/scripts/validate_presentation.py`.
- Confirm `FINAL_PRESENTATION.pdf` has 31 pages.
- Inspect rendered slides when layout or case wording changes.

## Demo

- Backend tests: `pytest demo/flutter/backend/tests`.
- Flutter app checks: `flutter analyze`, `flutter test`, and
  `flutter build web --pwa-strategy=none`.
- Health must show explicit success or error, not silent loading.
