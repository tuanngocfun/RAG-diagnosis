# AGENTS.md - RAG Diagnosis Engineering Guardrails

## Project Purpose

This repository contains a research RAG diagnosis pipeline, thesis-defense demo
artifacts, and reproducibility evidence for leishmaniasis-focused multimodal
case retrieval. It is a research/evaluation system, not a clinical deployment.

## Safety Rules

- Do not claim clinical deployment readiness.
- Do not claim diagnosis from image alone.
- Treat labels as silver/pipeline/reference labels unless a file explicitly
  documents another source.
- Keep raw medical datasets, model weights, venvs, caches, and generated build
  folders out of Git.
- Before `git add`, commit, or push, show `git status --short` and a concise
  diff/file summary.
- Stage explicit paths only. Do not use `git add .`.
- Never stage deletions unless the user explicitly confirms them.
- Never use destructive commands such as `rm -rf`, `git reset --hard`,
  `git checkout --`, or `git clean` unless the user explicitly requests that
  exact operation in the current turn.

## Working Layout

- `rag/`: RAG pipeline, configs, CLI, and tests.
- `demo/flutter/`: local GPU assistant backend and Flutter UI demo.
- `presentation/v12d/`: final thesis defense deck and validation scripts.
- `experiments/structured_cases_v4_2_2_rtx6000/`: selected reproducibility
  evidence, not the full experiment archive.
- `data/whole_multicare_dataset/`: lightweight dataset ledger and small
  derived metadata only.
- `docs/harness-engineering/`: operating rules for agent-assisted work.

## Validation Gates

Run the smallest relevant checks first, then broaden when touching shared
contracts:

- RAG: `pytest rag/tests`
- V12d deck: `python3 presentation/v12d/scripts/validate_presentation.py`
- Flutter backend: `pytest demo/flutter/backend/tests`
- Flutter app: `flutter analyze`, `flutter test`, and
  `flutter build web --pwa-strategy=none`

If dependencies or credentials are missing, report the blocker clearly rather
than weakening the validation claim.
