# models/medgemma27b-it_download.py
#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import login, snapshot_download, HfFolder

load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))

hf_token = os.getenv("HF_TOKEN")
if not hf_token:
    raise SystemExit("HF_TOKEN missing in .env")

hf_home = os.getenv("HF_HOME", "/data4t/hf")
cache_dir = os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers")
target_dir = os.path.join(cache_dir, "models--google--medgemma-27b-it")

Path(hf_home).mkdir(parents=True, exist_ok=True)
Path(cache_dir).mkdir(parents=True, exist_ok=True)
Path(target_dir).mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = hf_home
os.environ["TRANSFORMERS_CACHE"] = cache_dir

HfFolder.save_token(hf_token)
login(token=hf_token, add_to_git_credential=True)

local_path = snapshot_download(
    repo_id="google/medgemma-27b-it",
    revision="main",
    local_files_only=False,       # must be online the first time
    local_dir=target_dir,         # <-- point to the subfolder
    resume_download=True,         # will pick up shards where it left off
    token=hf_token,
)

print(f"✓ Downloaded to: {local_path}")
