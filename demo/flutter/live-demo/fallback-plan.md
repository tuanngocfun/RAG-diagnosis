# Fallback Plan

## If Flutter SDK Is Missing

Use direct API calls from `demo-runbook.md`. This still demonstrates:

- backend readiness,
- deterministic provider disclosure,
- evidence payload shape,
- gate audit fields,
- supported, provisional, and abstained states.

## If Phone Or Emulator Cannot Reach Backend

- Use Android emulator URL `http://10.0.2.2:8010`.
- Use physical-phone LAN URL `http://HOST_LAN_IP:8010`.
- If networking still fails, use API fallback.

## If Backend Fails To Start

Run backend tests:

```bash
cd /home/ngocnt/flutter/backend
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python -m pytest
```

Then inspect:

- `/home/ngocnt/flutter/backend/medical_demo_backend/api.py`
- `/home/ngocnt/flutter/backend/medical_demo_backend/service.py`
- `/home/ngocnt/flutter/kb/leishmaniasis_demo_pack.json`

## If Slides Or Screen Share Fail

Open:

- `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/presentation_supervisor_demo/supervisor_demo.pdf`
- `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/presentation_supervisor_demo/speaker_notes.md`
- `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/presentation_supervisor_demo/source_ledger.md`

## If Asked For Real Thesis Runs

Show:

- `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000/runs`
- `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/details_analysis/text_only`
- `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/details_analysis/multimodal`
- `/home/ngocnt/flutter/live-demo/real-gpu-sidecar.md`

Say:

"The live app is a deterministic demo harness. The thesis results come from these run artifacts and the v44b thesis tables."

If a real local model proof is needed during the session, run:

```bash
cd /home/ngocnt/flutter/live-demo
./run-real-gemma4-onecase.sh
```

Say:

"This one-case sidecar is a real Gemma 4 RAG generation on the demo GPU. It is
live execution proof, not a new benchmark result."
