#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: ./run-real-gemma4-onecase.sh [--check-only|--paths-only]

Runs one real Gemma 4 RAG generation case from the thesis provenance codebase
with nvidia-smi telemetry. Intended for the RTX TITAN 24GB supervisor sidecar.

Modes:
  --paths-only   Check local files, cache paths, and environment wiring only.
  --check-only   Check paths plus CUDA/GPU visibility, then stop before loading.

Useful overrides:
  GENERATION_GPU=0
  GEMMA4_IMAGE_SIZE=768 GEMMA4_MAX_QUERY_IMAGES=1   # fallback if 24GB OOMs
  SIDE_LOG_ROOT=/home/ngocnt/flutter/live-demo/real-gpu-runs
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

mode="run"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --paths-only)
      mode="paths"
      shift
      ;;
    --check-only)
      mode="check"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      usage >&2
      die "unknown argument: $1"
      ;;
  esac
done

export PROJECT_ROOT="${PROJECT_ROOT:-/home/ngocnt/experiments/structured_cases_v4_2_2_rtx6000}"
export LEGACY_ROOT="${LEGACY_ROOT:-/home/ngocnt/Leishmaniasis_v3}"

base_run_default="$PROJECT_ROOT/runs/ccx_pmx_ret_base_20260501_142038"
BASE_RUN="${BASE_RUN:-$base_run_default}"
OUTPUT_FILE="${OUTPUT_FILE:-answers_rag_real_demo.jsonl}"
SIDE_LOG_ROOT="${SIDE_LOG_ROOT:-/home/ngocnt/flutter/live-demo/real-gpu-runs}"
MODEL_ID="${MODEL_ID:-google/gemma-4-E4B-it}"
SEED="${SEED:-42}"
SAMPLE_SIZE="${SAMPLE_SIZE:-1}"
CONTEXT_K="${CONTEXT_K:-3}"
PROMPT_MODE="${PROMPT_MODE:-balanced}"
ORDERING_MODE="${ORDERING_MODE:-image_first}"

if [[ -n "${GENERATION_GPU:-}" ]]; then
  gpu_index="$GENERATION_GPU"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "ALL" ]]; then
  gpu_index="${CUDA_VISIBLE_DEVICES%%,*}"
else
  gpu_index="0"
fi
export GENERATION_GPU="$gpu_index"
export CUDA_VISIBLE_DEVICES="$gpu_index"

env_script="$PROJECT_ROOT/runbooks/structured_cases_env.sh"
telemetry_script="$PROJECT_ROOT/runbooks/run_with_gpu_telemetry.sh"

[[ -d "$PROJECT_ROOT" ]] || die "PROJECT_ROOT not found: $PROJECT_ROOT"
[[ -d "$LEGACY_ROOT" ]] || die "LEGACY_ROOT not found: $LEGACY_ROOT"
[[ -f "$env_script" ]] || die "environment script not found: $env_script"
[[ -f "$telemetry_script" ]] || die "telemetry script not found: $telemetry_script"
[[ -d "$BASE_RUN" ]] || die "base retrieval run not found: $BASE_RUN"
[[ -s "$BASE_RUN/retrieval.jsonl" ]] || die "base retrieval file missing: $BASE_RUN/retrieval.jsonl"
[[ -s "$BASE_RUN/queries.json" ]] || die "base queries file missing: $BASE_RUN/queries.json"

# shellcheck disable=SC1090
source "$env_script"

MODEL_CACHE_DIR="${MODEL_CACHE_DIR:-$TRANSFORMERS_CACHE/models--google--gemma-4-E4B-it}"
[[ -d "$MODEL_CACHE_DIR" ]] || die "Gemma 4 cache not found: $MODEL_CACHE_DIR"

echo "Real Gemma 4 sidecar paths are ready."
echo "PROJECT_ROOT=$PROJECT_ROOT"
echo "BASE_RUN=$BASE_RUN"
echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "mode=$mode"

if [[ "$mode" == "paths" ]]; then
  echo "paths-only check passed; no GPU/model run was started."
  exit 0
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  die "nvidia-smi not found; run this in the RTX TITAN terminal"
fi

if ! nvidia-smi -i "$GENERATION_GPU" --query-gpu=index,name,memory.total,memory.free --format=csv,noheader; then
  die "nvidia-smi could not communicate with the NVIDIA driver; run this from the RTX TITAN host/session"
fi

python - <<'PY'
import importlib
import torch

print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_device_count", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA unavailable; refusing to run the real demo on CPU")

name = torch.cuda.get_device_name(0)
props = torch.cuda.get_device_properties(0)
vram_gb = props.total_memory / (1024 ** 3)
print("cuda_device_0", name)
print("cuda_vram_gb", f"{vram_gb:.1f}")
if vram_gb < 20:
    raise SystemExit("Expected roughly 24GB VRAM for this demo; refusing this GPU")

transformers = importlib.import_module("transformers")
print("transformers", getattr(transformers, "__version__", "unknown"))
accelerate = importlib.import_module("accelerate")
print("accelerate", getattr(accelerate, "__version__", "unknown"))
try:
    bitsandbytes = importlib.import_module("bitsandbytes")
    print("bitsandbytes", getattr(bitsandbytes, "__version__", "unknown"))
except Exception as exc:
    raise SystemExit(
        "bitsandbytes is required for this RTX TITAN 24GB 4-bit demo but is unavailable: "
        f"{type(exc).__name__}: {exc}"
    )
PY

if [[ "$mode" == "check" ]]; then
  echo "check-only preflight passed; no model run was started."
  exit 0
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export GEMMA4_FORCE_4BIT="${GEMMA4_FORCE_4BIT:-1}"

timestamp="$(date +%Y%m%d_%H%M%S)"
SIDE_LOG_DIR="${SIDE_LOG_DIR:-$SIDE_LOG_ROOT/gemma4_rag_onecase_$timestamp}"
mkdir -p "$SIDE_LOG_DIR"

runtime_log="$SIDE_LOG_DIR/runtime.log"
gpu_log="$SIDE_LOG_DIR/gpu.csv"
summary_path="$SIDE_LOG_DIR/summary.txt"
env_snapshot="$SIDE_LOG_DIR/environment.txt"

{
  echo "created_at=$(date -Is)"
  echo "PROJECT_ROOT=$PROJECT_ROOT"
  echo "LEGACY_ROOT=$LEGACY_ROOT"
  echo "BASE_RUN=$BASE_RUN"
  echo "MODEL_ID=$MODEL_ID"
  echo "MODEL_CACHE_DIR=$MODEL_CACHE_DIR"
  echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
  echo "GENERATION_GPU=$GENERATION_GPU"
  echo "HF_HOME=$HF_HOME"
  echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
  echo "HF_HUB_OFFLINE=$HF_HUB_OFFLINE"
  echo "TRANSFORMERS_OFFLINE=$TRANSFORMERS_OFFLINE"
  echo "GEMMA4_FORCE_4BIT=$GEMMA4_FORCE_4BIT"
  echo "GEMMA4_IMAGE_SIZE=${GEMMA4_IMAGE_SIZE:-default_896}"
  echo "GEMMA4_MAX_QUERY_IMAGES=${GEMMA4_MAX_QUERY_IMAGES:-default_5}"
} > "$env_snapshot"

cmd=(
  python -m pipeline.run_seed_sweep
  --base-run "$BASE_RUN"
  --seeds "$SEED"
  --generator gemma4
  --model "$MODEL_ID"
  --variant 4b
  --prompt-mode "$PROMPT_MODE"
  --output-file "$OUTPUT_FILE"
  --sample "$SAMPLE_SIZE"
  --context-k "$CONTEXT_K"
  --ordering-mode "$ORDERING_MODE"
  --delay 0
  --no-eval
)

echo "Starting real Gemma 4 RAG sidecar."
echo "Runtime log: $runtime_log"
echo "GPU log: $gpu_log"
printf 'Command:'
printf ' %q' "${cmd[@]}"
printf '\n'

bash "$telemetry_script" \
  --runtime-log "$runtime_log" \
  --gpu-log "$gpu_log" \
  --sample-interval 1 \
  --gpu-index "$GENERATION_GPU" \
  -- "${cmd[@]}"

run_prefix="$(basename "$BASE_RUN")_seed${SEED}_gemma4_"
RUN_DIR="$(
  find "$STRUCTURED_CASES_RUNS_DIR" -maxdepth 1 -type d -name "${run_prefix}*" -printf '%T@ %p\n' \
    | sort -nr \
    | awk 'NR==1{sub(/^[^ ]+ /, ""); print}'
)"
[[ -n "$RUN_DIR" ]] || die "could not locate generated run directory with prefix: $run_prefix"

ANSWER_PATH="$RUN_DIR/$OUTPUT_FILE"
[[ -s "$ANSWER_PATH" ]] || die "answer file missing or empty: $ANSWER_PATH"

cp "$ANSWER_PATH" "$SIDE_LOG_DIR/$OUTPUT_FILE"
if [[ -s "$RUN_DIR/answer_generation_contract.json" ]]; then
  cp "$RUN_DIR/answer_generation_contract.json" "$SIDE_LOG_DIR/answer_generation_contract.json"
fi

export RUN_DIR ANSWER_PATH SIDE_LOG_DIR GPU_LOG="$gpu_log" RUNTIME_LOG="$runtime_log"
python - <<'PY' | tee "$summary_path"
import csv
import json
import os
from pathlib import Path

run_dir = Path(os.environ["RUN_DIR"])
answer_path = Path(os.environ["ANSWER_PATH"])
side_log_dir = Path(os.environ["SIDE_LOG_DIR"])
gpu_log = Path(os.environ["GPU_LOG"])
runtime_log = Path(os.environ["RUNTIME_LOG"])

with answer_path.open(encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

gpu_rows = []
if gpu_log.exists():
    with gpu_log.open(encoding="utf-8") as f:
        gpu_rows = list(csv.DictReader(f))

def as_int(value):
    try:
        return int(str(value).strip())
    except Exception:
        return 0

max_memory = max((as_int(r.get("memory.used")) for r in gpu_rows), default=0)
gpu_name = next((str(r.get("name") or "").strip() for r in gpu_rows if r.get("name")), "")

record = rows[0] if rows else {}
answer = str(record.get("answer") or "").replace("\n", " ").strip()
if len(answer) > 1200:
    answer = answer[:1200] + "..."

print("Real Gemma 4 RAG sidecar summary")
print(f"run_dir: {run_dir}")
print(f"side_log_dir: {side_log_dir}")
print(f"runtime_log: {runtime_log}")
print(f"gpu_log: {gpu_log}")
print(f"answer_file: {answer_path}")
print(f"record_count: {len(rows)}")
print(f"qid: {record.get('qid', '')}")
print(f"query_type: {record.get('query_type', '')}")
print(f"model_name: {record.get('model_name', '')}")
print(f"generation_mode: {record.get('generation_mode', '')}")
print(f"final_route: {record.get('final_route', '')}")
print(f"retrieval_support_status: {record.get('retrieval_support_status', '')}")
print(f"prompt_context_count: {record.get('prompt_context_count', '')}")
print(f"generation_latency_seconds: {record.get('generation_latency_seconds', '')}")
print(f"gpu_name: {gpu_name}")
print(f"max_gpu_memory_used_mib: {max_memory}")
print("answer_preview:")
print(answer)
PY

echo "Summary written to: $summary_path"
echo "Copied answer JSONL to: $SIDE_LOG_DIR/$OUTPUT_FILE"
