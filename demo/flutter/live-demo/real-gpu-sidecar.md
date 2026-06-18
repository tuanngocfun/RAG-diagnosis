# Real GPU Sidecar Demo

This sidecar is the real-model proof for the supervisor demo. The Flutter app
also has a GPU Assistant page that can call the real-GPU backend directly. Keep
this sidecar as a terminal fallback and as a provenance-rich proof run.

For an interactive UI-driven real model request, start the backend itself in
real GPU mode instead:

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

Then launch the Flutter app, open the GPU Assistant tab, submit a detailed case,
and watch `nvtop` or `nvidia-smi`. The page shows backend readiness, provider
mode, GPU name, quantization, model load time, generation time, image tensor
count, evidence sent to the model, and the safety disclaimer.

## What It Runs

- Codebase: `/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000`
- Retrieval base: `runs/ccx_pmx_ret_base_20260501_142038`
- Model: `google/gemma-4-E4B-it`
- Hardware target: RTX TITAN 24GB
- Mode: one RAG case, seed 42, no judge evaluation

The folder name contains `rtx6000` because it is the thesis provenance codebase.
For this live sidecar, the hardware is the RTX TITAN visible to the terminal.

## Run

```bash
cd /home/ngocnt/flutter/live-demo
./run-real-gemma4-onecase.sh
```

The script stops before generation if CUDA is not visible. It writes logs to:

```text
/home/ngocnt/flutter/live-demo/real-gpu-runs/gemma4_rag_onecase_<timestamp>/
```

Expected files:

- `runtime.log`
- `gpu.csv`
- `answers_rag_real_demo.jsonl`
- `answer_generation_contract.json`
- `summary.txt`
- `environment.txt`

## Quick Checks

Use these before the meeting if you only want to verify wiring:

```bash
cd /home/ngocnt/flutter/live-demo
./run-real-gemma4-onecase.sh --paths-only
./run-real-gemma4-onecase.sh --check-only
```

`--paths-only` does not require GPU access. `--check-only` requires the RTX TITAN
session and confirms CUDA plus the local 4-bit dependencies before stopping.

## OOM Fallback

If the RTX TITAN 24GB run runs out of memory, keep 4-bit quantization and reduce
image load:

```bash
cd /home/ngocnt/flutter/live-demo
GEMMA4_IMAGE_SIZE=768 GEMMA4_MAX_QUERY_IMAGES=1 ./run-real-gemma4-onecase.sh
```

Use this wording if you need the fallback:

> This is still a real Gemma 4 local run, but with smaller image tensors to fit
> the 24GB demo GPU. It is a live proof of pipeline execution, not a new thesis
> benchmark.

## Suggested Demo Script

1. Keep Flutter open and show the Consult Gate as the deterministic
   safety-gated pipeline explainer.
2. Open a terminal beside it and run the sidecar command.
3. Point to `Loading Gemma 4`, `Device: cuda`, `GPU VRAM`, and `Quantization:
   4-bit` in `runtime.log`.
4. Point to `gpu.csv` or `nvidia-smi` to show live RTX TITAN memory/utilization.
5. After generation, open `summary.txt` and show the QID, model name, latency,
   route, prompt context count, and answer preview.
6. For an interactive route, restart the backend in `real_gpu_gemma4` mode and
   use the GPU Assistant tab instead of the sidecar script.

Do not present this one-case run as a thesis result. Present it as live proof
that the thesis pipeline can execute real local generation on the demo GPU.
