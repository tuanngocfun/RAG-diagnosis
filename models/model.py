# model.py — robust local-or-download loader (fixed for new HF cache)
import os
from pathlib import Path
from typing import Optional

# -------------------------------
# 0) Environment & cache layout
# -------------------------------
# You want /data4t/hf. We'll default to that, but allow HF_HOME override.
HF_HOME = Path(os.getenv("HF_HOME", "/data4t/hf")).resolve()
HF_TRANSFORMERS = HF_HOME / "transformers"
HF_HUB_CACHE = HF_HOME / "hub"

# Ensure folders exist and export env so HF/transformers use them.
HF_HUB_CACHE.mkdir(parents=True, exist_ok=True)
HF_TRANSFORMERS.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(HF_HOME)
os.environ["HF_HUB_CACHE"] = str(HF_HUB_CACHE)
os.environ["TRANSFORMERS_CACHE"] = str(HF_TRANSFORMERS)  # still accepted

# Optional: .env support for HF_TOKEN and offline toggle
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

HF_TOKEN = (os.getenv("HF_TOKEN") or "").strip() or None
FORCE_OFFLINE = os.getenv("HF_FORCE_OFFLINE", "0") == "1"
if FORCE_OFFLINE:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
else:
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    os.environ.pop("HF_HUB_OFFLINE", None)

print(f"[INFO] HF_HOME={os.environ.get('HF_HOME')}")
print(f"[INFO] TRANSFORMERS_CACHE={os.environ.get('TRANSFORMERS_CACHE')}")
print(f"[INFO] HF_HUB_CACHE={os.environ.get('HF_HUB_CACHE')}")
print(f"[INFO] HF_FORCE_OFFLINE={FORCE_OFFLINE}")

# -------------------------------
# 1) Snapshot download helper
# -------------------------------
from huggingface_hub import snapshot_download

def download_or_cache(model_id: str, fixed_under_transformers: bool = True) -> Path:
    """
    Ensure model repo is present locally and return the real on-disk directory.
    Works with the new HF cache (no manual 'snapshots/' walking).

    If fixed_under_transformers=True, we always place it at:
        {HF_TRANSFORMERS}/models--<org>--<name>
    so your cache stays under /data4t/hf/transformers even if someone overrides HF_HOME.
    """
    if fixed_under_transformers:
        # mimic classic layout root, but let HF control internal structure
        org, name = model_id.split("/")
        local_dir = HF_TRANSFORMERS / f"models--{org}--{name}"
        local_dir.mkdir(parents=True, exist_ok=True)
        local_dir_str = str(local_dir)
    else:
        local_dir_str = None  # let HF_HOME decide

    # If we’re forced offline, snapshot_download will reuse what’s present or fail clearly.
    path = snapshot_download(
        repo_id=model_id,
        token=HF_TOKEN,                      # optional but avoids throttling
        local_dir=local_dir_str,             # keep all under /data4t/hf/transformers
        local_dir_use_symlinks=False,        # explicit; ignored on new hub, harmless
        resume_download=True,
        # allow/ignore_patterns=None  # download everything
    )
    p = Path(path)
    print(f"[INFO] Ready: {model_id} at {p}")
    return p

# -------------------------------
# 2) Model IDs
# -------------------------------
MEDGEMMA_ID = "google/medgemma-4b-it"
COLQWEN2_ID = "vidore/colqwen2-v1.0-hf"

# -------------------------------
# 3) Resolve or download
# -------------------------------
medgemma_path = download_or_cache(MEDGEMMA_ID, fixed_under_transformers=True)
colqwen2_path = download_or_cache(COLQWEN2_ID, fixed_under_transformers=True)

# -------------------------------
# 4) Load models & processors
# -------------------------------
import torch
from transformers import AutoProcessor

# Dtype: use float16 on GPU; fall back to float32 on CPU
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32

# ---- MedGemma (generator)
# Many VLM repos need trust_remote_code=True. If your local repo defines custom classes, enable it.
from transformers import AutoModelForImageTextToText

processor_medgemma = AutoProcessor.from_pretrained(
    str(medgemma_path),
    trust_remote_code=True,
)

model_medgemma = AutoModelForImageTextToText.from_pretrained(
    str(medgemma_path),
    trust_remote_code=True,
    torch_dtype=DTYPE,
    device_map="auto",
)

print("[INFO] Loaded MedGemma generator.")

# ---- ColQwen2 (retriever)
# Newer transformers expose dedicated classes; otherwise fall back gracefully.
try:
    from transformers import ColQwen2ForRetrieval, ColQwen2Processor  # requires newer transformers
    HAS_COLQWEN2 = True
except Exception:
    HAS_COLQWEN2 = False

if HAS_COLQWEN2:
    processor_colqwen2 = ColQwen2Processor.from_pretrained(
        str(colqwen2_path),
        trust_remote_code=True,
    )
    model_colqwen2 = ColQwen2ForRetrieval.from_pretrained(
        str(colqwen2_path),
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    print("[INFO] Loaded ColQwen2 retriever (native classes).")
else:
    # Fallback: still usable to embed but without the typed class
    from transformers import AutoModel
    processor_colqwen2 = AutoProcessor.from_pretrained(
        str(colqwen2_path),
        trust_remote_code=True,
    )
    model_colqwen2 = AutoModel.from_pretrained(
        str(colqwen2_path),
        trust_remote_code=True,
        torch_dtype=DTYPE,
        device_map="auto",
    )
    print("[WARN] transformers lacks ColQwen2* classes; loaded via AutoModel. Consider upgrading transformers.")

# -------------------------------
# 5) Example tiny smoke test (optional)
# -------------------------------
if __name__ == "__main__":
    print("[INFO] Model bootstrap finished.")
    # If you want, add a minimal processor call here later to validate shapes.
