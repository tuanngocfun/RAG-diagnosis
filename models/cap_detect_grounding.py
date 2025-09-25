#!/usr/bin/env python3
import os
from pathlib import Path
from dotenv import load_dotenv
from huggingface_hub import snapshot_download

def main():
    # 1. Load environment variables
    load_dotenv(dotenv_path=os.getenv("DOTENV_PATH", ".env"))
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        raise SystemExit("❌ HF_TOKEN missing. Put it in .env")

    # 2. Set local HF cache paths (same as your existing setup)
    hf_home = os.getenv("HF_HOME", "/data4t/hf")
    transformers_cache = os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers")
    Path(hf_home).mkdir(parents=True, exist_ok=True)
    Path(transformers_cache).mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = hf_home
    os.environ["TRANSFORMERS_CACHE"] = transformers_cache

    # 3. Download model snapshot
    repo_id = "microsoft/Florence-2-large"
    print(f"⬇️ Downloading {repo_id} to {transformers_cache} ...")
    snapshot_download(
        repo_id=repo_id,
        cache_dir=transformers_cache,
        token=hf_token,
        local_files_only=False,   # set True if offline only
        ignore_patterns=["*.msgpack", "*.h5"],  # optional to skip large unused files
        resume_download=True
    )
    print("✅ Download finished!")

if __name__ == "__main__":
    main()
