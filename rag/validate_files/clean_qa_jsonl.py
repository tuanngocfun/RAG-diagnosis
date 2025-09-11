#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
clean_qa_jsonl.py — Normalize Leishmaniasis QA .jsonl files

What it does (by default):
  - Recursively remove all keys named "_extras"
  - Pretty-print JSON (indent=2, UTF-8, keep non-ASCII)
  - Create a timestamped .bak backup next to each file before writing

Options:
  --root:   directory to scan (default: rag/qa/jsonl)
  --glob:   filename glob (default: *.jsonl)
  --indent: json indent (default: 2; use 0 for minified)
  --no-backup: skip creating .bak_TIMESTAMP backups
  --format-only: only reformat (don’t remove _extras)
  --dry-run: show what would be changed without writing
"""

import argparse
import datetime as dt
import json
from pathlib import Path
from typing import Any

def clean_extras(data: Any) -> Any:
    if isinstance(data, dict):
        return {k: clean_extras(v) for k, v in data.items() if k != '_extras'}
    if isinstance(data, list):
        return [clean_extras(x) for x in data]
    return data

def load_json_text(p: Path) -> str:
    # Read as UTF-8; strip BOM if present
    raw = p.read_text(encoding="utf-8")
    if raw.startswith("\ufeff"):
        raw = raw.lstrip("\ufeff")
    return raw

def parse_json_or_die(text: str, path: Path) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError as e:
        raise SystemExit(f"[ERROR] {path.name}: invalid JSON — {e}")

def main():
    ap = argparse.ArgumentParser(description="Normalize QA JSONL files")
    ap.add_argument("--root", default="kaggle/working2/rag_knowledge_base/qa/jsonl", help="Directory to scan")
    ap.add_argument("--glob", default="*.jsonl", help="Glob to match")
    ap.add_argument("--indent", type=int, default=2, help="JSON indent")
    ap.add_argument("--no-backup", action="store_true", help="Do not write .bak backup")
    ap.add_argument("--format-only", action="store_true", help="Only reformat; keep _extras")
    ap.add_argument("--dry-run", action="store_true", help="Show actions without writing")
    args = ap.parse_args()

    root = Path(args.root)
    files = sorted(root.glob(args.glob))
    if not files:
        print(f"[INFO] No files matched {root}/{args.glob}")
        return

    ts = dt.datetime.now().strftime("%Y%m%dT%H%M%S")

    changed = 0
    for f in files:
        text = load_json_text(f)
        data = parse_json_or_die(text, f)

        # The project uses *single JSON object per file* (even though extension is .jsonl).
        # If you truly have JSON Lines (multiple records), adapt here.
        original = json.dumps(data, ensure_ascii=False, separators=(",", ":"))

        processed = data if args.format_only else clean_extras(data)
        reformatted = json.dumps(processed, ensure_ascii=False, indent=args.indent)

        # Compare normalized (minified) content to avoid false positives from whitespace
        normalized_new = json.dumps(processed, ensure_ascii=False, separators=(",", ":"))
        if normalized_new == original:
            print(f"[SKIP] {f.name}: already normalized")
            continue

        print(f"[CLEAN] {f.name}")
        if not args.dry_run:
            if not args.no_backup:
                backup_path = f.with_suffix(f.suffix + f".bak_{ts}")
                backup_path.write_text(text, encoding="utf-8")
            f.write_text(reformatted + "\n", encoding="utf-8")
        changed += 1

    print(f"[DONE] {changed} file(s) updated out of {len(files)}")

if __name__ == "__main__":
    main()
