# Readiness Checklist

## Backend

- [ ] Start backend from `/home/ngocnt/Leishmaniasis_v3/rag/instructions/process/14/demo/flutter/backend`.
- [ ] `/health` returns `status: ok`.
- [ ] `/health` returns the intended `provider_mode`.
- [ ] Supported case returns `rag_supported`.
- [ ] Insufficient case returns `abstained`.
- [ ] Provisional case returns `provisional_parametric`.
- [ ] Out-of-scope case returns `abstained`.
- [ ] Evidence cards include title, diagnosis label, score, source case, and confirmatory flag.

## Real GPU Assistant

- [ ] `nvidia-smi` sees the TITAN RTX and reports free VRAM.
- [ ] `torch.cuda.is_available()` is true in the backend terminal/session.
- [ ] `/mnt/data/hf/transformers/models--google--gemma-4-E4B-it` exists.
- [ ] `bitsandbytes` imports successfully.
- [ ] If free VRAM is low, manually close unneeded GPU workloads; do not auto-kill processes.
- [ ] Backend is started with `MEDICAL_DEMO_PROVIDER_MODE=real_gpu_gemma4`.
- [ ] `/health` returns `chat_available: true` and `cuda_available: true`.
- [ ] GPU Assistant tab shows real provider mode before submitting.
- [ ] `/v1/chat` returns a decision-support response or a safe blocked state.
- [ ] Runtime audit shows provider, model, generation time, GPU name, and quantization when available.

## Flutter

- [ ] Flutter SDK is available in `PATH`.
- [ ] `flutter test` passes.
- [ ] App starts with `--dart-define=MEDICAL_DEMO_BACKEND_URL=...`.
- [ ] App shows both `Consult Gate` and `GPU Assistant` destinations.
- [ ] Demo-case selector fills text correctly.
- [ ] Gate audit panel shows top score and conflict flag.
- [ ] Disclaimer is visible in every response.

## Presentation

- [ ] `supervisor_demo.pptx` exists.
- [ ] `supervisor_demo.pdf` exists.
- [ ] PDF text contains main results and deterministic-provider boundary.
- [ ] Speaker notes are ready.
- [ ] Source ledger maps claims to paths.

## Safety

- [ ] Do not claim clinical validation.
- [ ] Do not describe deterministic output as real model inference.
- [ ] Do not describe GPU Assistant output as ground truth.
- [ ] Do not claim retrieval is universally beneficial.
- [ ] State that real-model generation is research decision support only.

## Rehearsal

- [ ] Run full slide and demo route once.
- [ ] Run API fallback route once.
- [ ] Keep generated PDF open as last-resort fallback.
