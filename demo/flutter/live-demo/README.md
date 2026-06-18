# Leishmaniasis Supervisor Live Demo

This folder contains the operating harness for the supervisor-facing demo.

The live app is intentionally reliable and bounded:

- Flutter is the primary demo surface.
- Python backend exposes `GET /health` and `POST /v1/consult`.
- Backend provider mode is deterministic demo logic.
- The demo proves evidence flow, uncertainty gating, response contract, and safe abstention.
- Thesis run provenance comes from `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000`.
- Optional real-model proof runs as a terminal sidecar on the RTX TITAN 24GB.

## Files

| File | Purpose |
|---|---|
| `demo-output-contract.md` | public demo contract and acceptance criteria |
| `demo-runbook.md` | exact commands and rehearsal route |
| `demo-scenarios.md` | paste-ready supervisor cases |
| `supervisor-demo-script.md` | 30-minute talk script |
| `source-ledger.md` | local source and artifact map |
| `readiness-checklist.md` | final preflight checklist |
| `fallback-plan.md` | API and artifact fallback route |
| `real-gpu-sidecar.md` | real Gemma 4 sidecar route for RTX TITAN |
| `run-real-gemma4-onecase.sh` | one-case real RAG generation with GPU telemetry |

## Current Known State

- Backend tests pass with `/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python -m pytest`.
- Flutter Web is the recommended SSH-friendly UI route.
- The Flutter/backend app is deterministic by default.
- Restart the backend with `MEDICAL_DEMO_PROVIDER_MODE=real_gpu_gemma4` when the UI itself should call the RTX TITAN model.
- The real model run is the separate terminal sidecar in `real-gpu-sidecar.md`.
