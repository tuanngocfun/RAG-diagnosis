# env_bootstrap.py
import os
from pathlib import Path

def init_env(
    dot_env_path: str | None = None,
    hf_root: str = "/media/pc1/Ubuntu/Extend_Data/ngoc/hf",
    offline: bool = True,  # flip to False the first time you need to fetch a new model
) -> dict:
    """Load .env, set HF caches & offline flags, login with HF_TOKEN if present.
    Returns a dict with resolved settings."""
    # 1) Load .env (robust to cwd)
    loaded = False
    try:
        from dotenv import load_dotenv, find_dotenv
        loaded = load_dotenv(dot_env_path or find_dotenv(usecwd=True), override=False)
    except Exception:
        pass

    if not loaded and dot_env_path:
        # ultra-light fallback: KEY=VAL lines (no quotes/escapes)
        p = Path(dot_env_path)
        if p.exists():
            for ln in p.read_text(encoding="utf-8").splitlines():
                ln = ln.strip()
                if not ln or ln.startswith("#") or "=" not in ln:
                    continue
                k, v = ln.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())

    # 2) Resolve token and cache roots
    # Allow .env to specify HF_ROOT; never clobber if already set in the process env.
    env_hf_root = os.getenv("HF_ROOT")
    hf_root = str(Path(env_hf_root or hf_root).expanduser())

    # Only set defaults if not already defined
    os.environ.setdefault("HF_HOME", hf_root)
    os.environ.setdefault("HF_HUB_CACHE", f"{hf_root}/transformers")
    os.environ.setdefault("TRANSFORMERS_CACHE", f"{hf_root}/transformers")

    # Ensure cache dirs exist (no-op if present)
    try:
        Path(os.environ["HF_HOME"]).expanduser().mkdir(parents=True, exist_ok=True)
        Path(os.environ["HF_HUB_CACHE"]).expanduser().mkdir(parents=True, exist_ok=True)
        Path(os.environ["TRANSFORMERS_CACHE"]).expanduser().mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    os.environ["HF_HUB_OFFLINE"] = "1" if offline else "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "1" if offline else "0"

    # Optional but nice: avoid accidental network retries
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")  # faster downloads when online

    hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or ""

    # 3) If we're online and have a token, register it for gated/private repos
    if not offline and hf_token:
        try:
            from huggingface_hub import login, whoami
            login(token=hf_token, add_to_git_credential=False)
            _ = whoami()  # sanity check
        except Exception as e:
            print(f"[WARN] Hugging Face login failed: {e}")

    return {
        "hf_root": hf_root,
        "offline": offline,
        "hf_token": hf_token,
        "dot_env": dot_env_path,
        "resolved": {
            "HF_HOME": os.environ.get("HF_HOME", ""),
            "HF_HUB_CACHE": os.environ.get("HF_HUB_CACHE", ""),
            "TRANSFORMERS_CACHE": os.environ.get("TRANSFORMERS_CACHE", ""),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "")
        }
    }
