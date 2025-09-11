# download_mixtral_snapshot.py
import os
from huggingface_hub import snapshot_download

REPO_ID = "mistralai/Mixtral-8x7B-Instruct-v0.1"
DEST = "/data4t/hf/transformers/Mixtral-8x7B-Instruct-v0.1"
TOKEN = os.getenv("HF_TOKEN")

# Optional: pin to a specific commit for reproducibility
# REV = "commit-hash-here"  # e.g., "3b5d1f3..."  (use the repo's commit id)
REV = None

# (Optional) silence warning about symlinks on exFAT
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Only pull what you actually need
allow = ["model.safetensors.index.json", "*.safetensors"]

snapshot_download(
    repo_id=REPO_ID,
    revision=REV,
    local_dir=DEST,
    local_dir_use_symlinks=False,   # <-- key for exFAT
    allow_patterns=allow,
    token=TOKEN,
    resume_download=True,
)
print("✅ Mixtral files are ready in:", DEST)
