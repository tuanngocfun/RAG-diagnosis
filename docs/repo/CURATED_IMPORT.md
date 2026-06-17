# Curated Import Record

Date: 2026-06-18

This branch imports a lean, reproducible package from local research workspaces:

- `/home/ngocnt/Leishmaniasis_v3`
- `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000`
- `/home/ngocnt/operating_system/harness-engineering`

The import intentionally excludes raw data archives, model weights, venvs,
tool caches, Flutter build products, and historical presentation backups.

## Included

- RAG pipeline code, configs, CLI, and tests.
- Local Flutter GPU assistant backend/UI source and live-demo docs.
- Canonical V12d thesis-defense package with final deck, validation scripts,
  selected evidence, and manifest.
- Two official Gemma 4 experiment runs used by V12d RAG/no-RAG and exact trace
  appendix slides.
- Lightweight MultiCaRe/leishmaniasis dataset ledger with small derived
  metadata and checksums.
- RAG-specific harness-engineering docs and agent guardrails.

## Excluded

- Raw `PMC*.zip` archives.
- Parquet corpus dumps.
- Full 24 GB local workspace.
- Full 370 MB experiment archive beyond the selected official runs and split
  metadata.
- `venv`, `.dart_tool`, `.pytest_cache`, `build`, and generated frontend cache
  folders.

See `docs/repo/BRANCH_INVENTORY.md` for base-branch selection.
