#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini API key supervisor & log watcher (hardened)

- Runs your command (e.g., the script that calls Gemini).
- Streams and records stdout/stderr.
- On 429/503 (rate limit / service unavailable) or known Gemini failure phrases,
  gracefully stops the child process group, rotates GOOGLE_API_KEY in your .env,
  and restarts with exponential backoff.
- Tracks restarts and the active key in a state file; writes a pid file.
- Optional: pre-kill stale workers by pattern to avoid duplicates.

Usage example:
python rag/gen/utils/gemini_key_guard.py \
  --env-file /home/students/Leishmania/.env \
  --workdir /home/students/Leishmania \
  --cmd "python -m rag.gen.evaluation_gemini_only --jsonl_dir ... --answers_file ... --out_dir ... --gemini_rpm 5 --strategy all --stream_append" \
  --echo --cycle --max-restarts 100 \
  --prekill_pattern "evaluation_gemini_only|gemini_key_guard.py"
"""

import argparse
import os
import sys
import time
import re
import json
import shutil
import signal
import subprocess
import tempfile
import shlex
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# --- Detection: broadened phrases seen in logs/caches ---
DETECTION_PATTERNS = [
    r"\b429\b",
    r"\b503\b",
    r"Too\s*Many\s*Requests",
    r"RESOURCE[_\s-]?EXHAUSTED",
    r"rate\s*limit(?:ed|ing)?",
    r"exceeded\s+quota",
    r"\bUNAVAILABLE\b",
    r"Retry[-\s]?After",
    r"Error\s*code:\s*429",
    r"Error\s*code:\s*503",
    r"http\s*status\s*429",
    r"http\s*status\s*503",
    r"\bquota\b.*\bexceed",
    r"\bRateLimitError\b",
    r"\bSERVICE\s*UNAVAILABLE\b",
]

# ---------- .env helpers ----------

def load_env_file(env_path: Path) -> Tuple[Dict[str, str], List[str]]:
    """Simple .env parser (keeps ordering & comments). Returns (env_dict, lines)."""
    env_dict: Dict[str, str] = {}
    lines: List[str] = []
    if env_path.exists():
        with env_path.open("r", encoding="utf-8") as f:
            for ln in f.readlines():
                lines.append(ln)
                m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$', ln)
                if m:
                    key = m.group(1)
                    val = m.group(2).strip()
                    # Trim surrounding quotes if present
                    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
                        val = val[1:-1]
                    env_dict[key] = val
    return env_dict, lines

def atomic_write(path: Path, content: str) -> None:
    """Write atomically to avoid partial .env on crashes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent), encoding="utf-8") as tmp:
        tmp.write(content)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)

def save_env_file(env_path: Path, lines: List[str], updates: Dict[str, str]) -> None:
    """
    Writes updates into existing lines when possible; appends new keys if missing.
    Creates a timestamped backup first. Uses atomic write.
    """
    if env_path.exists():
        backup = env_path.with_suffix(env_path.suffix + f".bak_{datetime.now().strftime('%Y%m%dT%H%M%S')}")
        shutil.copy2(env_path, backup)

    remaining = dict(updates)
    out_lines: List[str] = []
    for ln in lines:
        m = re.match(r'\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*)\s*$', ln)
        if m:
            key = m.group(1)
            if key in remaining:
                new_val = remaining.pop(key)
                # Quote if contains spaces or special chars
                if re.search(r'\s|#|=', new_val):
                    new_val_fmt = f'"{new_val}"'
                else:
                    new_val_fmt = new_val
                out_lines.append(f"{key}={new_val_fmt}\n")
            else:
                out_lines.append(ln)
        else:
            out_lines.append(ln)
    # Append leftovers
    for k, v in remaining.items():
        v_fmt = f'"{v}"' if re.search(r'\s|#|=', v) else v
        out_lines.append(f"{k}={v_fmt}\n")

    atomic_write(env_path, "".join(out_lines))

def find_key_rotation_order(env_dict: Dict[str, str], key_base: str) -> List[str]:
    """
    Returns a sorted list of key names: [GOOGLE_API_KEY, GOOGLE_API_KEY0, GOOGLE_API_KEY1, ...]
    present in env_dict, in numeric order (base first if present).
    Supports either 0-indexed or 1-indexed suffixes; sorts numerically.
    """
    keys = []
    if key_base in env_dict:
        keys.append(key_base)

    numbered = []
    for k in env_dict.keys():
        if k.startswith(key_base) and k != key_base:
            suffix = k[len(key_base):]
            if suffix.isdigit():
                numbered.append((int(suffix), k))

    for _, k in sorted(numbered):
        keys.append(k)

    if key_base not in env_dict and numbered:
        keys = [k for _, k in sorted(numbered)]

    return keys

def choose_next_key(current_key_name: str, rotation_order: List[str], cycle: bool) -> Optional[str]:
    if current_key_name not in rotation_order:
        return rotation_order[0] if rotation_order else None
    idx = rotation_order.index(current_key_name)
    if idx + 1 < len(rotation_order):
        return rotation_order[idx + 1]
    return rotation_order[0] if cycle and rotation_order else None

def rotate_google_api_key(env_path: Path, key_base: str, cycle: bool) -> Tuple[Optional[str], Optional[str]]:
    """
    Rotate GOOGLE_API_KEY's VALUE to the next among siblings (GOOGLE_API_KEY{N}).
    Uses VALUE-matching first: if GOOGLE_API_KEY has the same VALUE as some numbered key,
    advance from that numbered key; otherwise advance from base or first found.
    Ensures the new VALUE is different to avoid no-op rotations.
    """
    env_dict, lines = load_env_file(env_path)
    rotation_order = find_key_rotation_order(env_dict, key_base)
    if not rotation_order:
        return None, None

    base_val = env_dict.get(key_base, None)
    numbered = [k for k in rotation_order if k != key_base]

    # 1) Prefer matching by value among numbered keys
    current_key_name = None
    if base_val is not None:
        for name in numbered:
            if env_dict.get(name) == base_val:
                current_key_name = name
                break

    # 2) Fallback to base or first
    if current_key_name is None:
        current_key_name = key_base if key_base in env_dict else rotation_order[0]

    next_key_name = choose_next_key(current_key_name, rotation_order, cycle=cycle)
    if not next_key_name:
        return None, None

    next_val = env_dict.get(next_key_name)
    if not next_val:
        return None, None

    # 3) Avoid no-op rotation
    if base_val == next_val:
        # Try to skip further ahead (rare, but protects against duplicate values)
        idx = rotation_order.index(next_key_name)
        tried = set([next_key_name])
        while idx + 1 < len(rotation_order):
            idx += 1
            cand = rotation_order[idx]
            if cand in tried:
                break
            tried.add(cand)
            v = env_dict.get(cand)
            if v and v != base_val:
                next_key_name, next_val = cand, v
                break
        if base_val == next_val:
            # All values identical; give up.
            return None, None

    save_env_file(env_path, lines, {key_base: next_val})
    return next_key_name, next_val

# ---------- child process helpers ----------

def spawn_child(cmd: str, env_override: Dict[str, str], cwd: Optional[str] = None):
    """
    Spawns child in its own process group so we can signal the whole tree.
    If cmd looks like a raw string, run via shell=False with shlex.split for safety.
    """
    env = os.environ.copy()
    env.update(env_override or {})
    env.setdefault("PYTHONUNBUFFERED", "1")

    use_shell = False
    argv = cmd
    if isinstance(cmd, str):
        argv = shlex.split(cmd)
        use_shell = False

    return subprocess.Popen(
        argv,
        shell=use_shell,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        cwd=cwd,
        env=env,
        preexec_fn=os.setsid if os.name != "nt" else None,  # start a new process group on POSIX
    )

def kill_pgid(pid: int, sig) -> None:
    try:
        os.killpg(os.getpgid(pid), sig)
    except Exception:
        pass

def terminate_child(proc: subprocess.Popen, grace_sec: float) -> None:
    """Graceful SIGTERM to process group, then SIGKILL after grace."""
    if proc.poll() is not None:
        return
    if os.name != "nt":
        kill_pgid(proc.pid, signal.SIGTERM)
    else:
        try: proc.terminate()
        except Exception: pass

    waited = 0.0
    while proc.poll() is None and waited < grace_sec:
        time.sleep(0.2)
        waited += 0.2

    if proc.poll() is None:
        if os.name != "nt":
            kill_pgid(proc.pid, signal.SIGKILL)
        else:
            try: proc.kill()
            except Exception: pass

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(
        description="Gemini API key guard: watch logs, rotate GOOGLE_API_KEY on 429/503, and restart your command."
    )
    ap.add_argument("--env-file", required=True, help="Path to your .env file")
    ap.add_argument("--cmd", required=True, help='Command to run (quote if needed). Example: "python -m rag.gen.evaluation_gemini_only ... --resume"')
    ap.add_argument("--workdir", default=None, help="Working directory to run the command in")
    ap.add_argument("--google-key-base", default="GOOGLE_API_KEY", help="Base name for the key (default: GOOGLE_API_KEY)")
    ap.add_argument("--cycle", action="store_true", help="Cycle to the first key after the last one (round-robin)")
    ap.add_argument("--grace-sec", type=float, default=10.0, help="Seconds to wait after SIGTERM before SIGKILL")
    ap.add_argument("--max-restarts", type=int, default=50, help="Hard cap on total restarts to avoid infinite loops")
    ap.add_argument("--state-file", default="guard_state.json", help="Where to store guard state")
    ap.add_argument("--pid-file", default="guard.pid", help="Where to write the guard PID and last child PID")
    ap.add_argument("--log-file", default=None, help="Optional: combined child log path. Defaults to guard_logs/run_<timestamp>.log")
    ap.add_argument("--echo", action="store_true", help="Echo child output to console")
    ap.add_argument("--prekill_pattern", default=None, help="Optional regex; pre-kill stale processes matching this (via pkill -f) before starting")
    ap.add_argument("--backoff-floor", type=float, default=1.0, help="Initial backoff seconds")
    ap.add_argument("--backoff-cap", type=float, default=60.0, help="Max backoff seconds")
    args = ap.parse_args()

    env_path = Path(args.env_file).expanduser().resolve()
    if not env_path.exists():
        print(f"[ERROR] .env file not found at: {env_path}", file=sys.stderr)
        return 2

    # Optional pre-clean to avoid duplicate runners
    if args.prekill_pattern:
        try:
            subprocess.run(["pkill", "-f", args.prekill_pattern], check=False)
            print(f"[INFO] Pre-killed stale processes matching: {args.prekill_pattern}")
        except Exception as e:
            print(f"[WARN] Pre-kill failed: {e}")

    # Prepare log folder / file
    log_dir = Path("guard_logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.log_file:
        log_path = Path(args.log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = log_dir / f"run_{ts}.log"

    state_path = Path(args.state_file)
    pid_path = Path(args.pid_file)

    state = {"restarts": 0, "last_reason": None, "current_key_name": None,
             "current_key_value_fingerprint": None, "last_update": None}
    if state_path.exists():
        try:
            state.update(json.loads(state_path.read_text(encoding="utf-8")))
        except Exception:
            pass

    # Ensure initial key is loaded (into child env)
    env_dict, _ = load_env_file(env_path)
    key_base = args.google_key_base
    child_env_override = {}
    chosen_name = None
    chosen_val = None

    if key_base in env_dict and env_dict[key_base]:
        chosen_name = key_base
        chosen_val = env_dict[key_base]
    else:
        rotation_order = find_key_rotation_order(env_dict, key_base)
        if rotation_order:
            chosen_name = rotation_order[0]
            chosen_val = env_dict[chosen_name]
        else:
            print(f"[ERROR] No {key_base} or numbered variants found in {env_path}", file=sys.stderr)
            return 2

    child_env_override[key_base] = chosen_val
    state["current_key_name"] = chosen_name
    state["current_key_value_fingerprint"] = f"{hash(chosen_val)}"
    state["last_update"] = datetime.now().isoformat(timespec="seconds")
    state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    # Compile detection regex once
    combined_re = re.compile("|".join(DETECTION_PATTERNS), re.IGNORECASE)

    # Guard PID file
    pid_info = {"guard_pid": os.getpid(), "child_pid": None}
    pid_path.write_text(json.dumps(pid_info), encoding="utf-8")

    # Handle parent termination: ensure child is reaped
    terminating = {"flag": False}
    def _handle_parent_signal(signum, frame):
        if terminating["flag"]:
            return
        terminating["flag"] = True
        print(f"\n[INFO] Guard received signal {signum}. Stopping child…", file=sys.stderr)
        try:
            if proc and proc.poll() is None:
                terminate_child(proc, grace_sec=args.grace_sec)
        except Exception:
            pass
        sys.exit(128 + signum)

    signal.signal(signal.SIGINT, _handle_parent_signal)
    signal.signal(signal.SIGTERM, _handle_parent_signal)

    # Spawn the child
    cmd = args.cmd
    proc = spawn_child(cmd, env_override=child_env_override, cwd=args.workdir)
    pid_info["child_pid"] = proc.pid
    pid_path.write_text(json.dumps(pid_info), encoding="utf-8")

    print(f"[INFO] Spawned child PID={proc.pid}: {cmd}")
    print(f"[INFO] Logging to: {log_path}")

    backoff = max(0.0, float(args.backoff_floor))
    backoff_cap = max(backoff, float(args.backoff_cap))

    with log_path.open("a", encoding="utf-8") as lf:
        while True:
            line = proc.stdout.readline() if proc.stdout else ""
            if not line:
                rc = proc.poll()
                if rc is None:
                    time.sleep(0.1)
                    continue
                print(f"[INFO] Child exited with code {rc}")
                break

            lf.write(line)
            lf.flush()
            if args.echo:
                sys.stdout.write(line)
                sys.stdout.flush()

            if combined_re.search(line):
                reason = line.strip()[:500]
                print(f"[DETECT] Throttling/Service issue detected: {reason}")

                # Stop child process group
                terminate_child(proc, grace_sec=args.grace_sec)

                # Rotate key
                new_key_name, new_key_val = rotate_google_api_key(env_path, key_base, cycle=args.cycle)
                if not new_key_name or not new_key_val:
                    print("[ERROR] Could not rotate GOOGLE_API_KEY (missing/duplicate keys). Stopping.")
                    break

                # Update state
                state["restarts"] = int(state.get("restarts", 0)) + 1
                state["last_reason"] = reason
                state["current_key_name"] = new_key_name
                state["current_key_value_fingerprint"] = f"{hash(new_key_val)}"
                state["last_update"] = datetime.now().isoformat(timespec="seconds")
                state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

                if state["restarts"] > args.max_restarts:
                    print(f"[ERROR] Max restarts exceeded ({args.max_restarts}). Stopping.")
                    break

                # Backoff to avoid hammering
                sleep_for = min(backoff, backoff_cap)
                if sleep_for > 0:
                    print(f"[INFO] Backing off {sleep_for:.1f}s before restart…")
                    time.sleep(sleep_for)
                    backoff = min(backoff * 2.0, backoff_cap) if backoff > 0 else 1.0

                # Restart with new key
                child_env_override[key_base] = new_key_val
                proc = spawn_child(cmd, env_override=child_env_override, cwd=args.workdir)
                pid_info["child_pid"] = proc.pid
                pid_path.write_text(json.dumps(pid_info), encoding="utf-8")
                print(f"[INFO] Restarted child with {new_key_name}. PID={proc.pid}")

    print("[INFO] Guard finished. See state/logs for details.")
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n[INFO] Guard interrupted by user.")
        sys.exit(130)
