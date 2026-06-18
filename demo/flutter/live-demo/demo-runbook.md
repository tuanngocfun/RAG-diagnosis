# Demo Runbook

## Demo Identity

- Demo name: Leishmaniasis supervisor hybrid demo.
- Primary surface: Flutter app.
- Fallback surface: direct backend API.
- Backend mode: deterministic consult gate by default; real GPU Gemma 4 for the GPU Assistant page.
- Thesis provenance folder: `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000`.

Use the GPU Assistant only as research decision support. Do not present it as a
clinically validated diagnosis system or as ground truth.

## Real GPU Preflight

Run this before starting the real-GPU backend. It checks the driver, CUDA,
model cache, bitsandbytes, and free VRAM. It does not stop or kill any process.

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader

cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/backend
HF_HOME=/mnt/data/hf \
TRANSFORMERS_CACHE=/mnt/data/hf/transformers \
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python - <<'PY'
from pathlib import Path
import importlib
import torch

cache = Path("/mnt/data/hf/transformers/models--google--gemma-4-E4B-it")
print("model_cache_exists", cache.exists(), cache)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible; start the backend from a GPU-visible terminal/session.")
free, total = torch.cuda.mem_get_info(0)
free_mib = int(free / (1024 ** 2))
total_mib = int(total / (1024 ** 2))
print("gpu", torch.cuda.get_device_name(0))
print("vram_free_mib", free_mib)
print("vram_total_mib", total_mib)
if free_mib < 12000:
    raise SystemExit("Free VRAM is low. Manually close unneeded GPU workloads; do not auto-kill processes.")
importlib.import_module("bitsandbytes")
print("bitsandbytes", "ok")
PY
```

## Start Backend

Default deterministic backend:

```bash
cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/backend
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python -m medical_demo_backend.api
```

Expected:

```text
medical-demo-backend listening on http://127.0.0.1:8010
```

Real GPU backend for UI-driven Gemma 4 requests:

```bash
cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/backend
MEDICAL_DEMO_PROVIDER_MODE=real_gpu_gemma4 \
HF_HOME=/mnt/data/hf \
TRANSFORMERS_CACHE=/mnt/data/hf/transformers \
HF_HUB_OFFLINE=1 \
TRANSFORMERS_OFFLINE=1 \
GEMMA4_FORCE_4BIT=1 \
GEMMA4_IMAGE_SIZE=768 \
GEMMA4_MAX_QUERY_IMAGES=1 \
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python -m medical_demo_backend.api
```

In this mode, the first supported/provisional Flutter request loads Gemma 4 on
the GPU and can take about one to two minutes. The model stays loaded for later
requests while the backend process is alive.

## Health Check

```bash
curl -s http://127.0.0.1:8010/health
```

Expected fields:

```json
{
  "status": "ok",
  "kb_ready": true,
  "provider_mode": "deterministic_demo",
  "chat_available": false
}
```

For real GPU UI mode, `provider_mode` and `chat_available` should be:

```json
{
  "provider_mode": "real_gpu_gemma4",
  "chat_available": true,
  "cuda_available": true
}
```

## Flutter Launch

Run this only after Flutter SDK is available in `PATH`.

Desktop or host-local:

```bash
cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/app
flutter run --dart-define=MEDICAL_DEMO_BACKEND_URL=http://127.0.0.1:8010
```

Android emulator:

```bash
cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/app
flutter run --dart-define=MEDICAL_DEMO_BACKEND_URL=http://10.0.2.2:8010
```

Physical phone:

```bash
cd /home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/app
flutter run --dart-define=MEDICAL_DEMO_BACKEND_URL=http://HOST_LAN_IP:8010
```

## Demo Route

1. Show `/health`.
2. Open Flutter app or API fallback.
3. Use the Consult Gate tab and select or paste the supported case.
4. Run consult and point out evidence cards, gate audit, and disclaimer.
   - In real GPU mode, also point out the elapsed timer while running, the
     backend terminal Gemma 4 logs, `nvtop` or `nvidia-smi`, and the Runtime
     audit panel after completion.
5. Select or paste the insufficient case.
6. Run consult and point out abstention and needed inputs.
7. Select or paste the provisional case.
8. Run consult and point out model-only fallback.
9. Switch to the GPU Assistant tab only after `/health` shows real GPU readiness.
10. Submit one detailed case and point out provider mode, GPU name, evidence,
    model latency, and the safety disclaimer.
11. Open the `Retriever and reranker audit` panel.
    - The live retriever should be labeled `local_demo_lexical`.
    - The live reranker should say it was not executed by this demo backend.
    - If the request includes a known V12d case ID, the official pipeline
      reference should show `retriever hybrid`, `rerank true`, and `top-k 20`.
12. Mention RTX6000 artifacts as the thesis run provenance.

## API Fallback Commands

Supported:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/consult \
  -H 'Content-Type: application/json' \
  -d '{"patient_text":"Ulcerated plaque on the forearm after sandfly exposure with a smear showing amastigotes."}'
```

Insufficient:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/consult \
  -H 'Content-Type: application/json' \
  -d '{"patient_text":"Rash."}'
```

Provisional:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/consult \
  -H 'Content-Type: application/json' \
  -d '{"patient_text":"Chronic skin lesion with ulcerated border after travel to an endemic region."}'
```

GPU Assistant:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Adult patient with an ulcerated plaque on the forearm after sandfly exposure and a smear showing amastigotes. Please provide cautious decision support and what confirmation is needed."}]}'
```

GPU Assistant with official V12d rerank reference:

```bash
curl -s -X POST http://127.0.0.1:8010/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"client_request_id":"manual-test-PMC7516301_01","messages":[{"role":"user","content":"PMC7516301_01 adult patient with an ulcerated plaque on the forearm after sandfly exposure and a smear showing amastigotes. Please provide cautious decision support and what confirmation is needed."}]}'
```

For the second GPU Assistant request, check that the response includes:

- `retrieval_audit.retrieval_backend=local_demo_lexical`
- `retrieval_audit.live_rerank_executed=false`
- `retrieval_audit.official_rerank_reference.retriever_method=hybrid`
- `retrieval_audit.official_rerank_reference.rerank=true`
- `retrieval_audit.official_rerank_reference.retrieval_top_k=20`

## Stop

- Backend: `Ctrl-C`.
- Flutter: `q` or `Ctrl-C`.
