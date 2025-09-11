#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extract all PDFs from /home/students/Leishmania/data/fix
into /home/students/Leishmania/kaggle/working2/extract
with structure:

<DEST>/<PDF_STEM_EXACT>/pages/page_0000.png, page_0001.png, ...

- Folder name == PDF filename (without extension), kept AS-IS (Linux-safe).
- Robust rendering with retries and live progress bars.
- No CSV/JSON outputs are created.
"""

from pathlib import Path
from typing import Optional, Tuple, List, Dict
import sys, time
import fitz  # PyMuPDF
from tqdm.auto import tqdm

# -------------------- CONFIG --------------------
PDF_SRC_DIR   = Path("/home/students/Leishmania/data/standard")
EXTRACT_ROOT  = Path("/home/students/Leishmania/kaggle/working2/extract")
DPI           = 200        # you can change this (144/200/300)
OVERWRITE     = False      # set True to re-render existing PNGs
# ------------------------------------------------

# Silence ICC/console noise from MuPDF (best-effort)
try:
    fitz.TOOLS.set_icc(False)
    fitz.TOOLS.mupdf_display_errors(False)
except Exception:
    pass

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def make_case_name_exact(pdf_stem: str, used_names: set) -> str:
    """
    Keep folder name EXACTLY the same as the PDF stem (Linux allows almost all chars except '/').
    If a collision occurs (duplicate stems), append a numeric suffix.
    """
    name = pdf_stem
    # '/' cannot appear in a file name; just in case, replace with underscore.
    name = name.replace("/", "_")
    if name not in used_names:
        used_names.add(name)
        return name
    # De-duplicate if needed
    k = 2
    while True:
        candidate = f"{name}__{k}"
        if candidate not in used_names:
            used_names.add(candidate)
            return candidate
        k += 1

def render_pdf_to_case_folder(
    pdf_path: Path,
    case_dir: Path,
    dpi: int,
    pages_bar: tqdm,
    per_pdf_postfix=None,  # callable(done, total, name)
    overwrite: bool = False,
) -> Tuple[Optional[Path], Optional[str]]:
    """Render pdf_path -> case_dir/pages; update pages_bar per page."""
    try:
        if not pdf_path.exists():
            return None, f"missing_pdf:{pdf_path}"

        pages_dir = case_dir / "pages"
        ensure_dir(pages_dir)

        # Fast-skip if already extracted (has any page_*.png) and not overwriting
        if not overwrite and any(pages_dir.glob("page_*.png")):
            if per_pdf_postfix:
                per_pdf_postfix(0, 0, pdf_path.name)
            return case_dir, None

        try:
            doc = fitz.open(pdf_path)
        except Exception as e:
            return None, f"open_failed:{e}"

        n_pages = len(doc)
        if per_pdf_postfix:
            per_pdf_postfix(0, n_pages, pdf_path.name)

        zoom = dpi / 72.0
        mat = fitz.Matrix(zoom, zoom)
        log_path = case_dir / "extract_warnings.log"  # local log per case

        pad = max(4, len(str(n_pages)))  # zero-padding width

        for i in range(n_pages):
            page = doc[i]
            out = pages_dir / f"page_{i:0{pad}d}.png"

            # If overwriting is False and file exists, just count progress and continue
            if out.exists() and not overwrite:
                pages_bar.update(1)
                if per_pdf_postfix and (i % 5 == 0 or i == n_pages - 1):
                    per_pdf_postfix(i + 1, n_pages, pdf_path.name)
                continue

            ok = False
            for attempt in range(3):
                try:
                    if attempt == 0:
                        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csRGB,  alpha=False)
                    elif attempt == 1:
                        pix = page.get_pixmap(matrix=mat, colorspace=fitz.csGRAY, alpha=False)
                    else:
                        mat_low = fitz.Matrix(max(1, dpi // 2) / 72.0, max(1, dpi // 2) / 72.0)
                        pix = page.get_pixmap(matrix=mat_low, colorspace=fitz.csGRAY, alpha=False)
                    pix.save(str(out))
                    ok = True
                    break
                except Exception as e_try:
                    try:
                        with log_path.open("a", encoding="utf-8") as lf:
                            lf.write(f"page {i} attempt {attempt} failed: {e_try}\n")
                    except Exception:
                        pass

            pages_bar.update(1)
            if per_pdf_postfix and (i % 5 == 0 or i == n_pages - 1):
                per_pdf_postfix(i + 1, n_pages, pdf_path.name)

            if not ok:
                try:
                    with log_path.open("a", encoding="utf-8") as lf:
                        lf.write(f"page {i} permanently failed after 3 attempts\n")
                except Exception:
                    pass

        try:
            doc.close()
        except Exception:
            pass

        if per_pdf_postfix:
            per_pdf_postfix(n_pages, n_pages, pdf_path.name)
        return case_dir, None

    except Exception as e:
        return None, f"case_failed:{e}"

def extract_all_pdfs(src_dir: Path, out_root: Path, dpi: int = 144, overwrite: bool = False) -> List[Path]:
    ensure_dir(out_root)

    # Only direct children (you can switch to rglob("*.pdf") if needed)
    pdfs = sorted([p for p in src_dir.iterdir() if p.suffix.lower() == ".pdf"])
    print(f"[INFO] Found {len(pdfs)} PDFs under {src_dir}")

    # Precompute exact folder names (as-is)
    used_names = set()
    case_name_map: Dict[Path, str] = {p: make_case_name_exact(p.stem, used_names) for p in pdfs}

    # Pre-scan page counts (fast open/close) for global progress
    totals: List[int] = []
    for p in pdfs:
        try:
            with fitz.open(p) as d:
                totals.append(len(d))
        except Exception:
            totals.append(0)
    total_pages = sum(totals)
    if total_pages == 0:
        print("[WARN] Total pages = 0; nothing to extract?")
        return []

    case_dirs: List[Path] = []

    # Two progress bars: pages (position 0) and PDFs (position 1)
    with tqdm(total=total_pages, desc="Pages", position=0, leave=True, file=sys.stdout) as pages_bar, \
         tqdm(total=len(pdfs),       desc="PDFs",  position=1, leave=True, file=sys.stdout) as pdfs_bar:

        def _pdf_postfix(done_pages: int, total_pages_pdf: int, name: str):
            if total_pages_pdf > 0:
                pdfs_bar.set_postfix_str(f"{name}  {done_pages}/{total_pages_pdf}", refresh=True)
            else:
                pdfs_bar.set_postfix_str(f"{name}  (skipped/extracted)", refresh=True)

        for p in pdfs:
            start = time.time()
            case_dir = out_root / case_name_map[p]
            ensure_dir(case_dir)

            cd, err = render_pdf_to_case_folder(
                p, case_dir, dpi=dpi, pages_bar=pages_bar, per_pdf_postfix=_pdf_postfix, overwrite=overwrite
            )
            if err is None:
                case_dirs.append(cd)
            else:
                # Only print errors (no JSON files are written)
                print(f"[WARN] {p.name}: {err}", file=sys.stderr)

            pdfs_bar.update(1)
            pdfs_bar.set_postfix_str(f"Done: {p.name} in {time.time()-start:.1f}s", refresh=True)

    print(f"[DONE] Extracted PNG pages for {len(case_dirs)}/{len(pdfs)} PDFs into {out_root}")
    return case_dirs

if __name__ == "__main__":
    # Ensure source/dest exist
    if not PDF_SRC_DIR.exists():
        print(f"[ERR] Source not found: {PDF_SRC_DIR}", file=sys.stderr)
        sys.exit(1)
    ensure_dir(EXTRACT_ROOT)

    # Run extraction
    CASE_DIRS = extract_all_pdfs(PDF_SRC_DIR, EXTRACT_ROOT, dpi=DPI, overwrite=OVERWRITE)
    print(f"[INFO] Prepared {len(CASE_DIRS)} case folders under {EXTRACT_ROOT}")
