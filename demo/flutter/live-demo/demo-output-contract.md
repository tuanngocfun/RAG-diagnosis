# Live Demo Output Contract

## Goal

Show a reliable supervisor-facing path through a leishmaniasis decision-support prototype: input capture, local evidence retrieval, uncertainty gating, output shaping, and safe abstention.

## Audience

- Primary audience: thesis supervisors.
- Likely concerns: whether the pipeline is real, whether claims are safe, whether results trace back to thesis artifacts, and whether the demo can run without fragile model latency.

## Demo Surfaces

| Surface | Role | Status |
|---|---|---|
| Flutter app | primary live artifact | existing, updated with demo cases and gate audit display |
| Backend API | required fallback | existing, deterministic provider |
| Thesis artifacts | provenance proof | existing under RTX6000 and thesis v44b paths |

## Source Of Truth

| Source | Path | What it proves |
|---|---|---|
| Backend API | `/home/ngocnt/flutter/backend/medical_demo_backend/api.py` | health and consult endpoint |
| Service contract | `/home/ngocnt/flutter/backend/medical_demo_backend/service.py` | retrieval, gates, response shaping |
| Response types | `/home/ngocnt/flutter/backend/medical_demo_backend/types.py` | public payload fields |
| Flutter app | `/home/ngocnt/flutter/app` | user-facing live flow |
| Demo KB | `/home/ngocnt/flutter/kb/leishmaniasis_demo_pack.json` | small local evidence pack |
| Thesis provenance | `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000` | final run/code provenance folder |

## Public Demo Contract

The demo must show:

- Backend readiness through `/health`.
- A supported input with retrieved evidence and a ranked differential.
- An insufficient input with no ranked differential.
- A low-support provisional input with model-only fallback.
- Visible deterministic-provider and medical decision-support disclaimers.
- Gate details: support status, top score, trigger codes, conflict flag, image usability, timing.

## Risk Rules

- Do not describe deterministic provider output as real model inference.
- Do not call the app a diagnosis system.
- Do not claim clinician-validated accuracy.
- Do not hide abstention; it is part of the safety story.
- Use RTX6000 artifacts for thesis provenance, not the live Flutter app.
