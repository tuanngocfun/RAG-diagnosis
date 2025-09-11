import os
from huggingface_hub import snapshot_download

REPO_ID = "ncbi/MedCPT-Query-Encoder"
DEST = "/data4t/hf/transformers/MedCPT-Query-Encoder"
TOKEN = os.getenv("HF_TOKEN")  # set HF_TOKEN in your env if private repo

# Optional: pin a commit hash for reproducibility
REV = None

# Prevent symlink warnings on exFAT
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Download everything needed for inference (model + tokenizer + config)
allow_patterns = [
    "config.json",
    "tokenizer*",
    "vocab*",
    "*.safetensors",
    "*.json"
]

snapshot_download(
    repo_id=REPO_ID,
    revision=REV,
    local_dir=DEST,
    local_dir_use_symlinks=False,  # ✅ important for exFAT
    allow_patterns=allow_patterns,
    token=TOKEN,
    resume_download=True,
)

print(f"✅ MedCPT Query Encoder downloaded to: {DEST}")
