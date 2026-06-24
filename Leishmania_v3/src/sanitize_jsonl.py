#!/usr/bin/env python3
import json
import math
import argparse
from pathlib import Path

def sanitize_obj(obj):
    if isinstance(obj, float) and math.isnan(obj):
        return None
    if isinstance(obj, dict):
        return {k: sanitize_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_obj(v) for v in obj]
    return obj

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL (may contain NaN)")
    ap.add_argument("--out", dest="out", required=True, help="Output strict JSONL")
    args = ap.parse_args()

    inp = Path(args.inp)
    out = Path(args.out)

    with inp.open("r", encoding="utf-8") as fin, out.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)  # Python tolerates NaN
            obj = sanitize_obj(obj)
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Wrote: {out}")

if __name__ == "__main__":
    main()
