#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import unicodedata
import re
from pathlib import Path
from difflib import get_close_matches

def normalize_name(name: str) -> str:
    """
    Robust basename normalizer (without extension):
    - Unicode NFKC normalization
    - strip leading/trailing spaces
    - collapse internal whitespace to single space
    - keep case (change to .lower() if you want case-insensitive)
    """
    n = unicodedata.normalize("NFKC", name)
    n = n.strip()
    n = re.sub(r"\s+", " ", n)
    return n

def stem_without_ext(p: Path) -> str:
    """
    Remove the last extension only. For names like 'foo.pdf.jsonl',
    we'll first strip '.jsonl' and then additionally strip trailing '.pdf'
    because users sometimes keep that inside the JSONL filename.
    """
    stem = p.name
    # strip final extension
    if p.suffix:
        stem = stem[: -len(p.suffix)]
    # also strip trailing '.pdf' if present in the remaining stem
    if stem.lower().endswith(".pdf"):
        stem = stem[: -len(".pdf")]
    return normalize_name(stem)

def collect_stems(dir_path: Path, exts: set[str]) -> set[str]:
    out = set()
    for f in dir_path.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in exts:
            out.add(stem_without_ext(f))
    return out

def main():
    ap = argparse.ArgumentParser(description="Compare PDF vs JSONL basenames.")
    ap.add_argument("--pdf_dir", default="/home/students/Leishmania/data/standard",
                    help="Directory containing PDF files")
    ap.add_argument("--jsonl_dir", default="/home/students/Leishmania/kaggle/working2/rag_knowledge_base/qa/jsonl",
                    help="Directory containing JSONL files")
    ap.add_argument("--show_fuzzy", action="store_true",
                    help="Show fuzzy near-matches for missing items")
    ap.add_argument("--topk", type=int, default=3, help="Top-K fuzzy suggestions")
    args = ap.parse_args()

    pdf_dir = Path(args.pdf_dir)
    jsonl_dir = Path(args.jsonl_dir)

    if not pdf_dir.is_dir():
        raise SystemExit(f"PDF dir not found: {pdf_dir}")
    if not jsonl_dir.is_dir():
        raise SystemExit(f"JSONL dir not found: {jsonl_dir}")

    pdf_stems = collect_stems(pdf_dir, exts={".pdf"})
    jsonl_stems = collect_stems(jsonl_dir, exts={".jsonl"})

    pdf_only = sorted(pdf_stems - jsonl_stems)
    jsonl_only = sorted(jsonl_stems - pdf_stems)

    print(f"PDF count     : {len(pdf_stems)} (from {pdf_dir})")
    print(f"JSONL count   : {len(jsonl_stems)} (from {jsonl_dir})")
    print()

    print(f"PDFs WITHOUT matching JSONL: {len(pdf_only)}")
    for s in pdf_only:
        print(f"  - {s}")
        if args.show_fuzzy:
            sug = get_close_matches(s, jsonl_stems, n=args.topk, cutoff=0.6)
            if sug:
                print(f"      ~ maybe: {', '.join(sug)}")
    print()

    print(f"JSONLs WITHOUT matching PDF: {len(jsonl_only)}")
    for s in jsonl_only:
        print(f"  - {s}")
        if args.show_fuzzy:
            sug = get_close_matches(s, pdf_stems, n=args.topk, cutoff=0.6)
            if sug:
                print(f"      ~ maybe: {', '.join(sug)}")

    # Optional: write lists to files for later inspection
    (Path.cwd() / "missing_jsonl_from_pdf.txt").write_text("\n".join(pdf_only) + ("\n" if pdf_only else ""))
    (Path.cwd() / "missing_pdf_from_jsonl.txt").write_text("\n".join(jsonl_only) + ("\n" if jsonl_only else ""))
    print("\nWrote: missing_jsonl_from_pdf.txt, missing_pdf_from_jsonl.txt")

if __name__ == "__main__":
    main()
