#!/usr/bin/env python3
# build_page_maps.py (robust matching against sanitized extract folders)
import json
import re
import unicodedata
from pathlib import Path
import fitz  # PyMuPDF

# --- CONFIG ---
PDF_SRC_DIR   = Path("data/standard")            # adjust if needed
EXTRACT_ROOT  = Path("kaggle/working2/extract")  # adjust if needed
PAGE_GLOB     = "page_*.png"

# --------------------------
# Sanitization & Canonicalization
# --------------------------
SMART_TO_ASCII = {
    "’": "'", "‘": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", "‐": "-", "-": "-",
    "…": "...",
    "¼": "1/4", "½": "1/2", "¾": "3/4",
}
SMART_TRANS = str.maketrans(SMART_TO_ASCII)

def normalize_unicode(s: str) -> str:
    # NFKD → strip combining marks → translate smart punctuation
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))
    s = s.translate(SMART_TRANS)
    return s

def sanitize_like_extractor(name: str) -> str:
    """
    Heuristic to mirror your extractor's folder naming:

    - Unicode normalize + smart punctuation → ASCII-ish
    - Replace any char not [A-Za-z0-9._-] with underscore
    - Collapse multiple underscores/hyphens
    - Trim leading/trailing underscores
    - Keep dots (for suffix parts) but you likely had none in folder names
    """
    s = normalize_unicode(name)
    # Replace filesystem-unfriendly chars with underscore (conservative set)
    s = re.sub(r"[^A-Za-z0-9._\-]+", "_", s)
    # Collapse runs like __ or -- or _-_- etc. into single underscores/hyphens
    s = re.sub(r"[_\-]{2,}", lambda m: "_" if "_" in m.group(0) else "-", s)
    # Sometimes you had patterns like " - " → we already normalized to "_"
    s = s.strip("_.-")
    return s

def ultra_loose_key(name: str) -> str:
    """
    Very loose canonical key: lowercased alnum only.
    Useful when punctuation/underscore placement drifted.
    """
    s = normalize_unicode(name).lower()
    s = re.sub(r"[^a-z0-9]+", "", s)
    return s

# --------------------------
# Case folder index
# --------------------------
def index_case_dirs(root: Path):
    """
    Build multiple lookup keys per existing case folder:
    - exact directory name
    - sanitized directory name
    - ultra-loose alnum-only key
    Returns dicts for fast matching.
    """
    exact = {}
    sanitized = {}
    loose = {}

    if not root.exists():
        return exact, sanitized, loose

    for case_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        name = case_dir.name
        exact[name] = case_dir

        san = sanitize_like_extractor(name)
        sanitized.setdefault(san, case_dir)

        ulk = ultra_loose_key(name)
        loose.setdefault(ulk, case_dir)

    return exact, sanitized, loose

def find_case_dir_for_pdf(pdf_path: Path, idx_exact, idx_sanitized, idx_loose):
    """
    Try to find the matching extract folder for this PDF using several tiers.
    """
    stem = pdf_path.stem

    # Tier 0: exact (if extractor accidentally used raw stem as folder name)
    if stem in idx_exact:
        return idx_exact[stem]

    # Tier 1: sanitized(stem)
    san = sanitize_like_extractor(stem)
    if san in idx_exact:
        return idx_exact[san]
    if san in idx_sanitized:
        return idx_sanitized[san]

    # Tier 2: ultra-loose key (ignore punctuation/underscores entirely)
    ulk = ultra_loose_key(stem)
    if ulk in idx_loose:
        return idx_loose[ulk]

    # Tier 3: near-match search among sanitized keys (prefix/containment)
    # Helps when very long names were truncated by the extractor.
    # We look for best containment overlap.
    candidates = []
    for k, v in idx_exact.items():
        if sanitize_like_extractor(k) == san:
            return v  # perfect sanitized match
        if ultra_loose_key(k) == ulk:
            return v  # perfect loose match
        if san and sanitize_like_extractor(k).startswith(san):
            candidates.append((k, v))
    if len(candidates) == 1:
        return candidates[0][1]

    # Give up
    return None

# --------------------------
# Label extraction
# --------------------------
def page_labels_for_pdf(pdf_path: Path):
    """Return list[str] of display labels for each page; fallback to 1-based numbers."""
    with fitz.open(pdf_path) as doc:
        labels = []
        for i in range(len(doc)):
            try:
                lab = doc[i].get_label()  # PyMuPDF 1.22+
                if not lab:
                    lab = str(i + 1)
            except Exception:
                lab = str(i + 1)
            labels.append(lab)
    return labels

# --------------------------
# Build one page map
# --------------------------
def build_page_map(pdf_path: Path, case_dir: Path):
    pages_dir = case_dir / "pages"
    pngs = sorted(pages_dir.glob(PAGE_GLOB))
    if not pngs:
        raise FileNotFoundError(f"No extracted pages in {pages_dir}")

    labels = page_labels_for_pdf(pdf_path)
    n_pdf = len(labels)
    n_png = len(pngs)
    if n_pdf != n_png:
        print(f"[WARN] Page count mismatch: PDF {n_pdf} vs PNGs {n_png} in {case_dir}")

    n = min(n_pdf, n_png)
    index_to_label = {str(i): labels[i] for i in range(n)}
    label_to_index = {}
    for i in range(n):
        label_to_index.setdefault(labels[i], i)
    index_to_png = {str(i): pngs[i].name for i in range(n)}

    page_map = {
        "n_pages": n,
        "index_to_label": index_to_label,
        "label_to_index": label_to_index,
        "index_to_png": index_to_png,
    }
    (pages_dir / "_page_map.json").write_text(
        json.dumps(page_map, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return page_map

# --------------------------
# Main
# --------------------------
def main():
    pdfs = sorted(PDF_SRC_DIR.glob("*.pdf"))
    print(f"[INFO] Found {len(pdfs)} PDFs")

    idx_exact, idx_sanitized, idx_loose = index_case_dirs(EXTRACT_ROOT)

    built = 0
    for pdf in pdfs:
        case_dir = find_case_dir_for_pdf(pdf, idx_exact, idx_sanitized, idx_loose)
        if not case_dir:
            print(f"[ERR] {pdf}: could not match to any extract folder under {EXTRACT_ROOT}")
            continue
        try:
            build_page_map(pdf, case_dir)
            built += 1
        except Exception as e:
            print(f"[ERR] {pdf}: {e}")

    print(f"[INFO] Built {built} _page_map.json files")

if __name__ == "__main__":
    main()
