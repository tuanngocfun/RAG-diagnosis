# Medical Demo Backend

This backend implements a conservative two-stage uncertainty gate for a leishmaniasis-focused diagnosis support demo.

## What is implemented

- Stage 1 evidence gate after retrieval and input validation
- Stage 2 constrained parametric fallback gate
- A pure-Python WSGI app with:
  - `GET /health`
  - `POST /v1/consult`
  - `POST /v1/chat`
- A deterministic generator adapter that preserves the planned response contract without pretending to be a real clinical model
- A real-GPU Gemma 4 chat path for the Flutter GPU Assistant page
- Rich evidence payloads for supervisor walkthroughs: source case, title, diagnosis label, retrieval score, and confirmatory flag
- A GPU Assistant retrieval audit that shows the live local lexical retriever
  trace, returned chunks, and an explicit re-ranker boundary

## Important note

The backend is deterministic by default. The `/v1/chat` endpoint is intentionally blocked in deterministic mode and returns `safety_state: real_gpu_required`; it does not fake live model inference.

For the GPU Assistant page, start the backend with `MEDICAL_DEMO_PROVIDER_MODE=real_gpu_gemma4`. That path uses local Gemma 4 generation through the thesis pipeline and must still be described as research decision support, not clinical deployment or ground truth.

The live backend uses a small local lexical demo retriever. It exposes retrieval
scores and returned evidence chunks, but it does not execute a separate live
re-ranker. When the request names one of the V12d held-out case IDs, the
response can additionally attach reference-only official Gemma 4 pipeline trace
fields from the V12d `trace_summary.json`. That official reference is the
rerank-enabled final context list from the experiment pipeline, not a live
reranker call from the demo backend.

For defense consistency, `/v1/chat` also supports
`response_mode=official_v12d_replay`. That mode returns the saved V12d
experiment-pipeline output for the three selected thesis cases and sets
`fresh_generation_executed=false`. Use `response_mode=live_gpu` for a fresh
Gemma 4 audit run that may legitimately drift from the slide narrative.

## Run locally

```bash
python3 -m medical_demo_backend.api
```

The server listens on `http://127.0.0.1:8010` by default. Set `MEDICAL_DEMO_PORT` to choose another local port.

## Run real GPU mode

Run a preflight before the meeting:

```bash
nvidia-smi --query-gpu=name,memory.total,memory.free,driver_version --format=csv,noheader
HF_HOME=/mnt/data/hf \
TRANSFORMERS_CACHE=/mnt/data/hf/transformers \
/home/ngocnt/Leishmaniasis_v3/data/venv/bin/python - <<'PY'
from pathlib import Path
import importlib
import torch

cache = Path("/mnt/data/hf/transformers/models--google--gemma-4-E4B-it")
print("model_cache_exists", cache.exists(), cache)
print("cuda_available", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not visible in this terminal/session")
free, total = torch.cuda.mem_get_info(0)
print("gpu", torch.cuda.get_device_name(0))
print("vram_free_mib", int(free / (1024 ** 2)))
print("vram_total_mib", int(total / (1024 ** 2)))
importlib.import_module("bitsandbytes")
print("bitsandbytes", "ok")
PY
```

If free VRAM is low, manually close unneeded GPU workloads. Do not auto-kill processes.

Start the real-GPU backend:

```bash
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

Confirm readiness:

```bash
curl -s http://127.0.0.1:8010/health
```

`provider_mode` should be `real_gpu_gemma4`, and `chat_available` should be `true`.
`model_loaded` may be `false` immediately after a clean backend restart; the
first `live_gpu` request loads the model. Official replay does not require model
loading.

## Retriever and reranker audit check

```bash
curl -s -X POST http://127.0.0.1:8010/v1/chat \
  -H 'Content-Type: application/json' \
  -d '{"client_request_id":"manual-test-PMC7516301_01","messages":[{"role":"user","content":"PMC7516301_01 adult patient with an ulcerated plaque on the forearm after sandfly exposure and a smear showing amastigotes. Please provide cautious decision support and what confirmation is needed."}]}'
```

The response should include `retrieval_audit.retrieval_backend` as
`local_demo_lexical`, `retrieval_audit.live_rerank_executed` as `false`, and an
`official_rerank_reference` for `PMC7516301_01` when the V12d trace file exists.

## Test

```bash
pytest
```
