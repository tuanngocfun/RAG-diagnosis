#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MedGemma-4B-IT + LoRA inference — v2.8 (full patched)

Key fixes vs. your file:
- Prevents blue "Abstract" boxes / flat page graphics from being treated as photos
  via a flat-graphics penalty (entropy + edges) used in figure picking & guards.
- Uses np.gradient for edge density (no broadcasting crashes).
- Corrects NameError in refine_crop_within (img -> im) and consistent variable use.
- Safer document-page handling: try figure crop automatically; refine inside crop.
- Keeps your schema/repair logic and intent-aware overrides.
"""

from __future__ import annotations
import os, re, argparse, unicodedata
from pathlib import Path
from typing import List, Optional, Dict, Tuple

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText, BitsAndBytesConfig
from peft import PeftModel

# ----------------------
# HF cache / base model
# ----------------------
HF_CACHE = "/media/pc1/Ubuntu/Extend_Data/ngoc/hf/transformers"
BASE_ID  = "google/medgemma-4b-it"

ALLOWED_TYPES = [
    "clinical photo",
    "light microscopy (H&E or similar)",
    "gel-electrophoresis",
    "TEM",
    "other",
]

ORDERED_FIELDS = [
    "image type",
    "organism visible",
    "morphological form",
    "evidence",
    "species from visible text",
    "likely syndrome from photo alone",
    "HIV/immunosuppression impact (knowledge-based)",
    "clinical clues (knowledge-based)",
    "epidermal pattern",
    "dermal infiltrate",
    "diagnostic sign",
    "cellular location",
    "differential (knowledge-based)",
]

# ======================
# Loading
# ======================
def load_model(adapter_dir: Path, device_map="auto"):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
    )
    base = AutoModelForImageTextToText.from_pretrained(
        BASE_ID,
        dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
        quantization_config=bnb,
        device_map=device_map,
        cache_dir=HF_CACHE,
        local_files_only=True,
        token=os.getenv("HF_TOKEN", None),
    )
    model = PeftModel.from_pretrained(
        base, str(adapter_dir), is_trainable=False, local_files_only=True
    ).eval()

    try:
        lora_tensors = [p for n, p in model.named_parameters() if "lora" in n.lower()]
        if lora_tensors:
            with torch.no_grad():
                print("[LoRA] first tensor norm:", float(lora_tensors[0].norm().cpu()))
        else:
            print("[WARN] No LoRA tensors found – check adapter path.")
    except Exception as e:
        print(f"[WARN] LoRA norm print failed: {e}")
    return model

# ======================
# Helpers
# ======================
Rect = Tuple[int, int, int, int]  # x, y, w, h

def to_ascii(s: str) -> str:
    return s.encode("ascii", "ignore").decode("ascii")

def looks_like_document_page(img: Image.Image) -> bool:
    gray = img.convert('L')
    arr = np.array(gray, dtype=np.uint8)
    white_ratio = float((arr > 240).sum()) / float(arr.size)
    std_dev = float(arr.std())
    w, h = img.size
    if white_ratio > 0.60 and std_dev < 80:
        return True
    if (h > 1.2 * w) and (white_ratio > 0.40):
        return True
    return False

def parse_crops_arg(crops: Optional[str], n_images: int) -> List[Optional[Rect]]:
    if not crops:
        return [None] * n_images
    parts = [p.strip() for p in crops.split(";") if p.strip()]
    out: List[Optional[Rect]] = []
    for p in parts:
        x, y, w, h = [int(v.strip()) for v in p.split(",")]
        out.append((x, y, w, h))
    while len(out) < n_images:
        out.append(None)
    if len(out) > n_images:
        raise SystemExit(f"--crops has {len(out)} entries but only {n_images} image(s) provided")
    return out

def apply_crop(img: Image.Image, rect: Optional[Rect]) -> Image.Image:
    if not rect:
        return img
    x, y, w, h = rect
    return img.crop((x, y, x + w, y + h))

def auto_panel_fallback_crop(img: Image.Image) -> Image.Image:
    w, h = img.size
    if w >= int(1.6 * h):
        size = h
        return img.crop((w - size, 0, w, size))
    if h >= int(1.6 * w):
        size = w
        return img.crop((0, h - size, size, h))
    size = min(w, h)
    x0 = (w - size) // 2
    y0 = (h - size) // 2
    return img.crop((x0, y0, x0 + size, y0 + size))

def propose_two_panel_crops(img: Image.Image) -> List[Rect]:
    w, h = img.size
    def n2i(x0, y0, x1, y1):
        return (int(x0*w), int(y0*h), int((x1-x0)*w), int((y1-y0)*h))
    top    = n2i(0.06, 0.06, 0.50, 0.44)
    bottom = n2i(0.06, 0.52, 0.94, 0.94)
    return [top, bottom]

def auto_enhance_for_histology(img: Image.Image) -> Image.Image:
    w, h = img.size
    if min(w, h) < 800:
        scale = 900.0 / float(min(w, h))
        img = img.resize((int(w*scale), int(h*scale)), Image.Resampling.LANCZOS)
    return img

def save_dbg(img: Image.Image, path: Path, tag: str, idx: Optional[int] = None):
    try:
        base = Path(path).stem
        out = Path.cwd() / f"debug_{base}_{tag}{'' if idx is None else f'_{idx}'}.png"
        img.save(out)
        print("[DEBUG] wrote", out)
    except Exception as e:
        print("[WARN] debug save failed:", e)

# ---------- Scoring utilities ----------
def _edge_density(gray_arr: np.ndarray) -> float:
    """Edge density via gradient magnitude with shapes aligned."""
    arr = gray_arr.astype(np.float32)
    gy, gx = np.gradient(arr)
    mag = np.hypot(gx, gy)
    thresh = 8.0
    return float((mag > thresh).sum()) / float(mag.size)

def _saturation_mean(img: Image.Image) -> float:
    hsv = img.convert("HSV")
    s = np.array(hsv)[:, :, 1].astype(np.float32) / 255.0
    return float(s.mean())

def _textiness_penalty(gray_arr: np.ndarray) -> float:
    """Cheap text detector using horizontal projection + flatness."""
    g = gray_arr
    thr = 220
    binv = (g < thr).astype(np.uint8)  # ink-ish
    proj = binv.sum(axis=1)
    h, w = binv.shape
    if h == 0 or w == 0:
        return 0.0
    lines = ((proj > 0.05 * w) & (proj < 0.35 * w)).sum()
    line_ratio = lines / float(h)
    flat = max(0.0, 1.0 - (float(g.std()) / 64.0))
    return min(1.0, 0.7 * line_ratio + 0.5 * flat)

def _entropy_gray(gray_arr: np.ndarray) -> float:
    """Shannon entropy of an 8-bit grayscale array."""
    hist = np.bincount(gray_arr.ravel(), minlength=256).astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    nz = p[p > 0]
    return float(-(nz * np.log2(nz)).sum())

def _flat_graphics_penalty(gray_arr: np.ndarray) -> float:
    """
    Penalty [0..1] for flat colored boxes / page decorations:
    high when BOTH entropy and edge density are very low.
    """
    ent = _entropy_gray(gray_arr)         # ~0–8 bits
    edg = _edge_density(gray_arr)
    ent_pen = max(0.0, (3.0 - min(ent, 3.0)) / 3.0)   # <=3 bits ~ very flat
    edg_pen = max(0.0, (0.04 - min(edg, 0.04)) / 0.04)
    return float(min(1.0, 0.7 * ent_pen + 0.7 * edg_pen))

def _color_cluster_penalty(img: Image.Image) -> float:
    """
    Penalty for solid-color callouts (like blue "Abstract" boxes).
    Returns 0-1, higher for saturated single-hue regions.
    """
    hsv = img.convert("HSV")
    h = np.array(hsv)[:, :, 0].astype(np.uint8)
    s = np.array(hsv)[:, :, 1].astype(np.float32)/255.0
    hist, _ = np.histogram(h, bins=36, range=(0,256))
    peak_ratio = hist.max() / (hist.sum() + 1e-6)  # 0..1, high if single hue dominates
    sat = s.mean()
    return float(min(1.0, 0.6*peak_ratio + 0.6*max(0.0, sat-0.35)))

try:
    import pytesseract
    HAS_OCR = True
except ImportError:
    HAS_OCR = False
    print("[WARN] pytesseract not available - OCR keyword detection disabled")

DOC_KEYWORDS = ("abstract", "introduction", "references", "dermatology online journal", "keywords", "case discussion")

def has_doc_keywords(img: Image.Image) -> bool:
    """Check if image contains document section keywords via OCR."""
    if not HAS_OCR:
        return False
    try:
        txt = pytesseract.image_to_string(img, config="--psm 6").lower()
        return any(k in txt for k in DOC_KEYWORDS)
    except Exception:
        return False

# ------------ figure extractor using scoring ------------
def extract_likely_figure_crop(img: Image.Image, grid: int = 4) -> Rect:
    """
    Pick the most 'photo-like' region from a document page.
    Penalizes 'textiness', 'flat graphics', and 'color clusters' to avoid abstract boxes.
    """
    w, h = img.size
    gray = np.array(img.convert("L"), dtype=np.uint8)

    def white_ratio(a):
        return float((a > 240).sum()) / float(a.size)

    def edge_density(a):
        return _edge_density(a)

    def local_std(a):
        return float(a.std())

    def textiness(a):
        e = edge_density(a)
        s = local_std(a)
        t = e * max(0.0, 1.0 - min(s / 35.0, 1.0))
        return float(min(max(t, 0.0), 1.0))

    cw, ch = w // grid, h // grid
    cells = []
    for gy in range(grid):
        for gx in range(grid):
            x0 = gx * cw
            y0 = gy * ch
            x1 = (gx + 1) * cw if gx < grid - 1 else w
            y1 = (gy + 1) * ch if gy < grid - 1 else h
            tile = gray[y0:y1, x0:x1]
            
            # Get corresponding color tile for hue penalty
            tile_color = img.crop((x0, y0, x1, y1))

            wr = white_ratio(tile)
            ed = edge_density(tile)
            ls = local_std(tile)
            tx = textiness(tile)
            fg = _flat_graphics_penalty(tile)
            hue_pen = _color_cluster_penalty(tile_color)  # NEW

            score = (1.0 - wr) * 0.45 + ed * 0.30 + (min(ls / 50.0, 1.0)) * 0.20
            score -= tx * 0.25
            score -= fg * 0.60
            score -= hue_pen * 0.50  # NEW: penalize single-hue regions

            if wr > 0.90 or ls < 10.0:
                score -= 0.5

            cells.append((score, gx, gy, x0, y0, x1, y1))

    cells.sort(reverse=True, key=lambda x: x[0])
    if not cells:
        return (0, 0, w, h)

    # Rest of function remains the same...
    best_score, bx, by, bx0, by0, bx1, by1 = cells[0]
    taken = {(bx, by)}
    tol = max(0.12, best_score * 0.25)
    for sc, gx, gy, x0, y0, x1, y1 in cells[1:]:
        if sc < best_score - tol:
            continue
        if abs(gx - bx) + abs(gy - by) == 1:
            taken.add((gx, gy))

    xs0, ys0, xs1, ys1 = w, h, 0, 0
    for gx, gy in taken:
        x0 = gx * cw
        y0 = gy * ch
        x1 = (gx + 1) * cw if gx < grid - 1 else w
        y1 = (gy + 1) * ch if gy < grid - 1 else h
        xs0, ys0 = min(xs0, x0), min(ys0, y0)
        xs1, ys1 = max(xs1, x1), max(ys1, y1)

    bw, bh = xs1 - xs0, ys1 - ys0
    side = min(max(bw, bh), min(w, h))
    cx, cy = xs0 + bw // 2, ys0 + bh // 2
    sx0 = max(0, min(w - side, cx - side // 2))
    sy0 = max(0, min(h - side, cy - side // 2))
    return (sx0, sy0, side, side)

def refine_crop_within(im: Image.Image) -> Image.Image:
    """Re-run the figure picker inside a crop to dodge nearby text/graphics."""
    if not looks_like_document_page(im):
        return im
    rect = extract_likely_figure_crop(im, grid=4)
    return apply_crop(im, rect)

def looks_like_text_region(img: Image.Image) -> bool:
    g = np.array(img.convert("L"), dtype=np.uint8)
    pen_text = _textiness_penalty(g)
    pen_flat = _flat_graphics_penalty(g)
    sat = _saturation_mean(img)
    hue_pen = _color_cluster_penalty(img)  # NEW
    
    # Any of these conditions should flag as text/graphics
    return (pen_text > 0.25) or (pen_flat > 0.45) or (sat < 0.06) or (hue_pen > 0.40)

# ======================
# VLM classification
# ======================
def classify_image_types(model, processor, ims: List[Image.Image]) -> List[str]:
    """Always send images through classifier; no early 'other' short-circuit."""
    if not ims:
        return []
    sys_txt = (
        "Classify each image strictly into exactly one of: "
        + "; ".join(ALLOWED_TYPES)
        + ". Output one line per image in the form: Image i: <type>. Do not explain."
    )
    msgs = [
        {"role": "system", "content": [{"type": "text", "text": sys_txt}]},
        {"role": "user", "content": [{"type": "text", "text": "Classify the images."}] + [{"type": "image", "image": im} for im in ims]},
    ]
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    batch = processor(text=[text], images=[ims], return_tensors="pt").to(model.device)
    tok = processor.tokenizer
    eos = tok.eos_token_id
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = [eos] + ([eot] if isinstance(eot, int) and eot >= 0 else [])
    with torch.inference_mode():
        out = model.generate(
            **batch,
            max_new_tokens=96,
            do_sample=False,
            num_beams=1,
            eos_token_id=stop_ids,
            pad_token_id=(eos or 0),
        )
    ans = processor.decode(out[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True)

    classified_types: List[str] = []
    for i in range(1, len(ims) + 1):
        t = "other"
        for line in ans.splitlines():
            if line.strip().lower().startswith(f"image {i}:"):
                t = line.split(":", 1)[1].strip().lower(); break
        match = None
        for opt in ALLOWED_TYPES:
            if opt.lower() in t:
                match = opt; break
        classified_types.append(match or "other")
    return classified_types

# ======================
# Intent / overrides
# ======================
_HE_PAT       = re.compile(r"\b(h&e|hematoxylin|eosin|histolog|biopsy|microscop|epidermal|dermal|reaction\s*pattern|pseudoepitheliomatous|\bpeh\b|marquee|granulomatous|giemsa|wright)\b", re.I)
_CLIN_PAT     = re.compile(r"\b(photo|photograph|picture|image|from\s+photo\s+alone|clinical\s+features?|lesion|nodule|ulcer)\b", re.I)
_GEL_PAT      = re.compile(r"\b(pcr|gel|electrophoresis|amplicon|band)\b", re.I)
_TEM_PAT      = re.compile(r"\b(tem|ultrastruct)\b", re.I)
_TISSUE_PAT   = re.compile(r"\b(tissue|biopsy|section|slide|histopath|bone\s*marrow|marrow|aspirate|smear)\b", re.I)
_SYNDROME_PAT = re.compile(r"\b(syndrome|\bCL\b|\bDCL\b|\bMCL\b|\bVL\b)\b", re.I)
_IMMUNO_PAT   = re.compile(r"\b(HIV|AIDS|immunosuppress|immuno[- ]?comprom|CD4)\b", re.I)
_CLIN_SIGNS_PAT = re.compile(r"\b(clinical\s*signs?|signs|symptom|hepatosplenomegaly|splenomegaly|hepatomegaly|fever|pallor|weight\s*loss)\b", re.I)
_HISTO_DETAILS_PAT = re.compile(r"\b(epidermal|dermal|reaction\s*pattern|pseudoepitheliomatous|\bpeh\b|marquee|infiltrate|histology|histopath)\b", re.I)
_DIFFERENTIAL_PAT = re.compile(r"\b(mimic|scc|differential|alternative|fits\s*better|versus|vs\.?|confound)\b", re.I)

def override_types_from_question(question: str, base_types: List[str], doc_like_mask: List[bool] = None) -> List[str]:
    """Override types based on question intent, but respect document/text flags."""
    q = question or ""
    want_gel    = bool(_GEL_PAT.search(q))
    want_he     = bool(_HE_PAT.search(q))
    want_tem    = bool(_TEM_PAT.search(q))
    want_cln    = bool(_CLIN_PAT.search(q))
    want_tissue = bool(_TISSUE_PAT.search(q))
    
    if doc_like_mask is None:
        doc_like_mask = [False] * len(base_types)
    
    out: List[str] = []
    for i, t in enumerate(base_types):
        tl = (t or "other").lower()
        if tl in ("light microscopy (h&e or similar)", "tem", "gel-electrophoresis"):
            out.append(t)
            continue
        if tl == "other":
            # CRITICAL FIX: Don't upcast to clinical if flagged as document-like
            if doc_like_mask[i]:
                out.append("other")
                continue
            if want_gel: out.append("gel-electrophoresis"); continue
            if want_tem: out.append("TEM"); continue
            if want_he or want_tissue: out.append("light microscopy (H&E or similar)"); continue
            if want_cln: out.append("clinical photo"); continue
        out.append(t)
    return out

def syndrome_intent(question: str) -> bool: return bool(_SYNDROME_PAT.search(question or ""))
def immuno_intent(question: str) -> bool:   return bool(_IMMUNO_PAT.search(question or ""))
def clinical_signs_intent(question: str) -> bool: return bool(_CLIN_SIGNS_PAT.search(question or ""))
def histology_details_intent(question: str) -> bool: return bool(_HISTO_DETAILS_PAT.search(question or ""))
def differential_intent(question: str) -> bool: return bool(_DIFFERENTIAL_PAT.search(question or ""))

# ======================
# Knowledge helpers
# ======================
ALLOWED_SYNDROMES = {"CL", "DCL", "MCL", "VL", "uncertain"}
_MCL_HINTS   = ["mucosal", "nasal", "septum", "palate", "lip", "oral", "oropharynx"]
_DCL_STRONG  = ["disseminated", "widespread", "generalized", "diffuse"]
_DCL_REGIONS = ["face", "extremities", "trunk", "upper limb", "lower limb", "arms", "legs"]
_DCL_COUNT   = ["multiple", "numerous", "countless", "many"]

def suggest_syndrome(desired_type: str, evidence: str, question: str) -> str:
    if desired_type != "clinical photo":
        return "uncertain"
    ev = (evidence or "").lower()
    if any(k in ev for k in _MCL_HINTS): return "MCL"
    if any(k in ev for k in _DCL_STRONG): return "DCL"
    regions = sum(1 for r in _DCL_REGIONS if r in ev)
    if regions >= 2 and any(k in ev for k in _DCL_COUNT): return "DCL"
    return "uncertain"

def choose_clinical_clues(question: str) -> str:
    q = (question or "").upper()
    if "VL" in q or "VISCERAL" in q:
        return "fever; weight loss; marked hepatosplenomegaly; pallor; pancytopenia"
    if "MCL" in q:
        return "mucosal involvement (nasal/oral); septal perforation; chronic destructive lesions"
    if "DCL" in q:
        return "numerous widespread papules/nodules; diffuse infiltration; anergy to leishmanin test"
    return "localized ulcer/plaques at sandfly bite site; raised borders; satellite papules"

def choose_differential(question: str) -> str:
    q = (question or "").lower()
    if "scc" in q or "squamous" in q or "mimic" in q or "fits" in q:
        return ("SCC vs CL: CL fits better with pseudoepitheliomatous hyperplasia on histology, "
                "dermal lymphoplasmacytic/histiocytic infiltrate with parasitized histiocytes, "
                "and confirmation by parasitic forms/PCR; clinically non-healing indurated plaque/ulcer with raised borders")
    return "Consider CL in SCC-like lesions; confirm with histology (amastigotes) or PCR."

# ======================
# Prompt builder
# ======================
def build_msgs(question: str, ims: List[Image.Image], types: List[str],
               want_clinical_signs: bool, want_histo_details: bool, want_differential: bool):
    n = len(ims)
    ask_syndrome = syndrome_intent(question)
    ask_immuno   = immuno_intent(question)

    base_rubric = (
        "You are a careful medical assistant. Base your answer ONLY on the provided image(s). "
        "Answer in ENGLISH only (ASCII). For each image i = 1..{n}, follow EXACTLY this schema:\n"
        "Image i:\n"
        "- image type: <clinical photo / light microscopy (H&E or similar) / gel-electrophoresis / TEM / other>\n"
        "- organism visible: <yes/no>\n"
        "- morphological form: <amastigote/promastigote/none>\n"
        "- evidence: <brief visual cues you SEE>\n"
        "- species from visible text: <exact text or 'none visible'>\n"
        "Rules:\n"
        "A) If image type is 'clinical photo' or 'gel-electrophoresis', set organism visible=no and morphological form=none.\n"
        "B) 'amastigote' only if you SEE intracellular oval bodies in host cells on histology/TEM; 'promastigote' only if you SEE a free flagellated parasite.\n"
        "C) Never infer species unless the species NAME is legible as TEXT in the pixels; otherwise write 'none visible'.\n"
        "D) Do NOT mention PCR/gel unless a gel figure is actually shown."
    ).format(n=n)

    if ask_syndrome:
        base_rubric += (
            "\nAdditionally, append this line for each image:\n"
            "- likely syndrome from photo alone: <CL / DCL / MCL / VL / uncertain>\n"
            "Syndrome rules:\n"
            "• Default to 'uncertain' unless the image clearly supports a choice.\n"
            "• Prefer DCL only if lesions are numerous and widespread; consider MCL only if mucosal involvement is directly visible.\n"
            "• Do NOT infer VL from a photo.\n"
        )
    if ask_immuno:
        base_rubric += (
            "\nAlso append this knowledge-only line (not inferred from the image pixels):\n"
            "- HIV/immunosuppression impact (knowledge-based): "
            "<heavier parasite burden; more disseminated/diffuse cutaneous disease; higher risk of mucosal/visceral involvement; greater relapse risk>\n"
        )
    if want_clinical_signs:
        base_rubric += (
            "\nAlso append this knowledge-only line (not inferred from the image pixels):\n"
            "- clinical clues (knowledge-based): <succinct signs differentiating VL/CL/MCL when requested>\n"
        )
    if want_histo_details:
        base_rubric += (
            "\nIf the image is histology/TEM and details are requested, also append pixel-grounded lines:\n"
            "- epidermal pattern: <e.g., acanthosis, pseudoepitheliomatous hyperplasia, ulceration>\n"
            "- dermal infiltrate: <e.g., lymphocytes/plasma cells/histiocytes/granulomas>\n"
            "- diagnostic sign: <e.g., marquee sign; other named sign if clearly visible; else 'none'>\n"
            "If amastigotes are visible, append:\n"
            "- cellular location: within macrophage cytoplasm (parasitophorous vacuoles)\n"
        )
    if want_differential:
        base_rubric += (
            "\nIf the question mentions mimicry/differential/alternative (e.g., SCC vs CL), append:\n"
            "- differential (knowledge-based): <concise differential; why infectious dx (e.g., CL) fits better>\n"
        )

    type_lines = "\n".join([f"Image {i+1} is classified as: {types[i]}" for i in range(n)])
    return [
        {"role": "system", "content": [{"type": "text", "text": base_rubric}]},
        {"role": "user", "content": ([{"type": "text", "text": f"{question}\nThere are {n} image(s). Use these fixed types (do not change):\n{type_lines}\nReturn the schema in order."}] + [{"type": "image", "image": im} for im in ims])},
    ]

# ======================
# Parsing / reconstruction
# ======================
_FIELD_CANON = [
    (r"image\s*type", "image type"),
    (r"organism\s*visible", "organism visible"),
    (r"morphological\s*form", "morphological form"),
    (r"species\s*from\s*visible\s*text", "species from visible text"),
    (r"evidence", "evidence"),
    (r"likely\s*syndrome(?:\s*from\s*photo\s*alone)?", "likely syndrome from photo alone"),
    (r"hiv\s*\/?\s*immuno\s*-?\s*suppression\s*impact.*", "HIV/immunosuppression impact (knowledge-based)"),
    (r"clinical\s*clues.*", "clinical clues (knowledge-based)"),
    (r"epidermal\s*pattern", "epidermal pattern"),
    (r"dermal\s*infiltrate", "dermal infiltrate"),
    (r"diagnostic\s*sign", "diagnostic sign"),
    (r"cellular\s*location", "cellular location"),
    (r"differential.*", "differential (knowledge-based)"),
]

def canon_label(s: str) -> str:
    s0 = s.strip().lower()
    for pat, repl in _FIELD_CANON:
        if re.fullmatch(pat, s0, flags=re.I):
            return repl
    return s0

def parse_fields(raw_body: str) -> Dict[str, str]:
    body = unicodedata.normalize("NFKC", raw_body).replace("：", ":")
    fields: Dict[str, str] = {}
    for line in body.splitlines():
        m = re.match(r"^\s*-\s*([^:]+)\s*:\s*(.*)\s*$", line)
        if not m: continue
        lab, val = m.group(1), m.group(2)
        lab = canon_label(lab)
        if lab in ORDERED_FIELDS:
            fields[lab] = to_ascii(val.strip())
    return fields

def reconstruct_block(idx: int, fields: Dict[str, str], ensure_order: List[str]) -> str:
    lines = [f"Image {idx}:"]
    for key in ensure_order:
        if key in fields:
            lines.append(f"- {key}: {fields[key]}")
    return "\n".join(lines)

# ======================
# Post-repair
# ======================
_AMAST_EVID_PAT = re.compile(
    r"(intra(?:cellular)?|inside\s+of\s+cells|within\s+(?:histiocytes|macrophages)|parasitized\s+histiocytes|leishman-?donovan\s+bodies|marquee\s+sign).*(round|oval|dot|amastigote|parasite|organism)",
    re.I
)

def add_histology_guardrails(fields: Dict[str, str], desired_type: str, 
                             no_guardrails: bool = False, soft_mode: bool = True) -> Tuple[Dict[str, str], Dict[str, str]]:
    """
    Add guardrails for histology when amastigotes are detected.
    Returns (updated_fields, changes_made_dict)
    """
    original_fields = fields.copy()
    changes_made = {}
    
    if no_guardrails or desired_type.lower() != "light microscopy (h&e or similar)":
        return fields, changes_made
    
    organism_visible = fields.get("organism visible", "").lower()
    morph_form = fields.get("morphological form", "").lower()
    
    # If amastigotes are detected, provide reasonable defaults for missing histo details
    if organism_visible == "yes" and "amastigote" in morph_form:
        # Dermal infiltrate
        if not fields.get("dermal infiltrate") or fields.get("dermal infiltrate").lower() in ("none", "na", "n/a"):
            old_val = fields.get("dermal infiltrate", "none")
            if soft_mode:
                fields["dermal infiltrate"] = "suggestive of lymphocytes, plasma cells, and histiocytes"
            else:
                fields["dermal infiltrate"] = "lymphocytes, plasma cells, and histiocytes"
            changes_made["dermal infiltrate"] = {"old": old_val, "new": fields["dermal infiltrate"]}
        
        # Diagnostic sign  
        if not fields.get("diagnostic sign") or fields.get("diagnostic sign").lower() in ("none", "na", "n/a"):
            old_val = fields.get("diagnostic sign", "none")
            if soft_mode:
                fields["diagnostic sign"] = "likely marquee sign"
            else:
                fields["diagnostic sign"] = "marquee sign"
            changes_made["diagnostic sign"] = {"old": old_val, "new": fields["diagnostic sign"]}
        
        # Epidermal pattern - only suggest PEH if completely missing
        if not fields.get("epidermal pattern") or fields.get("epidermal pattern").lower() in ("none", "na", "n/a"):
            old_val = fields.get("epidermal pattern", "none")
            if soft_mode:
                fields["epidermal pattern"] = "suggestive of pseudoepitheliomatous hyperplasia"
            else:
                fields["epidermal pattern"] = "pseudoepitheliomatous hyperplasia"
            changes_made["epidermal pattern"] = {"old": old_val, "new": fields["epidermal pattern"]}
    
    return fields, changes_made

def repair_schema(text: str, known_types: List[str],
                  ask_syndrome: bool, ask_immuno: bool, ask_clinical_signs: bool,
                  ask_histo_details: bool, ask_differential: bool, question: str,
                  no_guardrails: bool = False, no_knowledge_lines: bool = False,
                  emit_diff: bool = False) -> Tuple[str, Optional[Dict]]:
    """
    Repair schema with optional diff tracking.
    Returns (repaired_text, diff_dict_or_None)
    """
    text = to_ascii(unicodedata.normalize("NFKC", text))
    parts = re.split(r"(?=\bImage\s+\d+\s*[::])", text, flags=re.I)
    blocks = [p.strip() for p in parts if p.strip() and re.match(r"^Image\s+\d+\s*[::]", p, flags=re.I)]
    if not blocks:
        blocks = [f"Image 1:\n{text.strip()}" if text.strip() else "Image 1:"]

    fixed = []
    all_diffs = {} if emit_diff else None
    
    for idx, block in enumerate(blocks, 1):
        body = re.sub(r"^\s*Image\s+\d+\s*[::]\s*", "", block, flags=re.I)
        raw_fields = parse_fields(body)
        fields = raw_fields.copy()
        block_changes = {}

        desired_type = known_types[min(idx - 1, len(known_types) - 1)] if known_types else "other"
        fields["image type"] = desired_type
        t = desired_type.lower()

        # Standard field repairs (same as before)
        if t == "other":
            fields["organism visible"] = "no"
            fields["morphological form"] = "none"
            fields["evidence"] = fields.get("evidence") or "document page; not a clinical image or microscopy"
            fields["species from visible text"] = "none visible"

        elif t in ("clinical photo", "gel-electrophoresis"):
            fields["organism visible"] = "no"
            fields["morphological form"] = "none"
            ev = (fields.get("evidence") or "").lower()
            if any(k in ev for k in ["introduction", "abstract", "keywords", "case discussion", "references"]):
                fields["evidence"] = "document text/graphics; organisms are not visible"
            elif not ev or ev in {"none", "na", "n/a"}:
                fields["evidence"] = ("clinical photograph; organisms are not visible"
                                      if t == "clinical photo" else
                                      "DNA bands on gel; organisms are not visible on gels")
            else:
                if any(w in ev for w in ["amastigote", "promastigote", "intracellular", "parasite", "organism"]):
                    fields["evidence"] = ("clinical photograph; organisms are not visible"
                                          if t == "clinical photo" else
                                          "DNA bands on gel; organisms are not visible on gels")
            fields["species from visible text"] = "none visible"

        else:
            ev = fields.get("evidence", "").strip()
            if not ev or ev.lower() in {"none", "na", "n/a"}:
                fields["evidence"] = "histological section; dermal infiltrate with histiocytes; look for intracellular round dots"
            ev_low = fields.get("evidence", "").lower()

            tokens = ["amastigote", "parasitized histiocytes", "leishman-donovan", "marquee", "intracellular", "within macrophage", "inside of cells"]
            if _AMAST_EVID_PAT.search(ev_low) or any(tok in ev_low for tok in tokens):
                fields["organism visible"] = "yes"
                fields["morphological form"] = "amastigote"
                fields["cellular location"] = "within macrophage cytoplasm (parasitophorous vacuoles)"
            else:
                fields.setdefault("organism visible", "no")
                fields.setdefault("morphological form", "none")
            fields["species from visible text"] = "none visible"

        # Knowledge-based fields (conditional on no_knowledge_lines)
        if not no_knowledge_lines:
            if ask_syndrome and t != "other":
                fields["likely syndrome from photo alone"] = suggest_syndrome(desired_type, fields.get("evidence", ""), question)
            else:
                fields.pop("likely syndrome from photo alone", None)

            if ask_immuno:
                fields["HIV/immunosuppression impact (knowledge-based)"] = (
                    "heavier parasite burden; more disseminated/diffuse cutaneous disease; "
                    "higher risk of mucosal/visceral involvement; greater relapse risk"
                )
            else:
                fields.pop("HIV/immunosuppression impact (knowledge-based)", None)

            if ask_clinical_signs:
                fields["clinical clues (knowledge-based)"] = choose_clinical_clues(question)
            else:
                fields.pop("clinical clues (knowledge-based)", None)

            if ask_differential and t != "other":
                fields["differential (knowledge-based)"] = choose_differential(question)
            else:
                fields.pop("differential (knowledge-based)", None)
        else:
            # Remove all knowledge-based fields
            fields.pop("likely syndrome from photo alone", None)
            fields.pop("HIV/immunosuppression impact (knowledge-based)", None)
            fields.pop("clinical clues (knowledge-based)", None)
            fields.pop("differential (knowledge-based)", None)

        # Clean up histology fields if not requested
        if not (ask_histo_details and t in ("light microscopy (h&e or similar)", "tem")):
            fields.pop("epidermal pattern", None)
            fields.pop("dermal infiltrate", None)
            fields.pop("diagnostic sign", None)
            fields.pop("cellular location", None)
        
        # Apply histology guardrails and track changes
        fields, guardrail_changes = add_histology_guardrails(fields, desired_type, no_guardrails)
        block_changes.update(guardrail_changes)
        
        # Track all changes for diff
        if emit_diff:
            for field, value in fields.items():
                if field not in raw_fields or raw_fields[field] != value:
                    if field not in block_changes:
                        block_changes[field] = {"old": raw_fields.get(field, "missing"), "new": value}
            
            if block_changes:
                all_diffs[f"Image {idx}"] = block_changes

        order = [f for f in ORDERED_FIELDS if f in fields]
        fixed.append(reconstruct_block(idx, fields, order))
    
    final_text = "\n".join(fixed)
    return (final_text, all_diffs) if emit_diff else (final_text, None)

# ======================
# Generation
# ======================
def _generate(model, processor, msgs, images_batch):
    text = processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    batch = processor(text=[text], images=images_batch, return_tensors="pt").to(model.device)
    tok = processor.tokenizer
    eos = tok.eos_token_id
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    stop_ids = [eos] + ([eot] if isinstance(eot, int) and eot >= 0 else [])
    with torch.inference_mode():
        gen = model.generate(
            **batch,
            max_new_tokens=256,
            do_sample=False,
            num_beams=1,
            eos_token_id=stop_ids,
            pad_token_id=(eos or 0),
            no_repeat_ngram_size=6,
            repetition_penalty=1.15,
            use_cache=True,
        )
    out = processor.decode(gen[0][batch["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
    return unicodedata.normalize("NFKC", out)

# ======================
# Answer helpers
# ======================
def apply_document_guards(im: Image.Image, doc_strict: bool = False, debug: bool = False, 
                         img_path: Optional[Path] = None, idx: Optional[int] = None) -> Tuple[Image.Image, bool]:
    """
    Apply document/text guards to an image.
    Returns (processed_image, is_forced_other)
    """
    is_doc_like = looks_like_document_page(im) or looks_like_text_region(im) or has_doc_keywords(im)
    
    if doc_strict and is_doc_like:
        # Force to "other" immediately in strict mode
        return im, True
    
    if is_doc_like:
        # Try to refine the crop to find actual content
        im_refined = refine_crop_within(im)
        if not looks_like_text_region(im_refined) and not has_doc_keywords(im_refined):
            if debug and img_path:
                save_dbg(im_refined, img_path, "refined", idx)
            return im_refined, False
        else:
            # Still looks like text/graphics after refinement
            return im, True
    
    return im, False

def answer_one(model, processor, img_path: Path, question: str,
               forced_type: Optional[str] = None, crop: Optional[Rect] = None,
               want_clinical_signs: bool = False, want_histo_details: bool = False,
               want_differential: bool = False, auto_panel: bool = False, debug: bool = False,
               no_guardrails: bool = False, no_knowledge_lines: bool = False,
               doc_strict: bool = False, emit_diff: bool = False) -> str:

    im_raw = Image.open(str(img_path)).convert("RGB")
    if debug:
        print(f"[INFO] image size: {im_raw.size[0]}x{im_raw.size[1]} — {img_path}")
        save_dbg(im_raw, img_path, "orig")

    im_for_detect = apply_crop(im_raw, crop) if crop else im_raw
    if looks_like_document_page(im_for_detect):
        rect = extract_likely_figure_crop(im_for_detect)
        im = apply_crop(im_for_detect, rect)
        im = refine_crop_within(im)
        if debug: save_dbg(im, img_path, "crop_docfig")
    else:
        im = apply_crop(im_for_detect, crop)
        if auto_panel and crop is None:
            im = auto_panel_fallback_crop(im)
            if debug: save_dbg(im, img_path, "crop_0")

    # Document/text guard
    forced = None
    is_doc_like = looks_like_document_page(im) or looks_like_text_region(im) or has_doc_keywords(im)
    if doc_strict and is_doc_like:
        forced = ["other"]
    elif is_doc_like:
        im2 = refine_crop_within(im)
        if not looks_like_text_region(im2) and not has_doc_keywords(im2):
            im = im2
        else:
            forced = ["other"]

    # classify (or force)
    if forced is not None:
        types = forced
    else:
        types = [forced_type] if forced_type else classify_image_types(model, processor, [im])
    types = override_types_from_question(question, types, [False])  # mask optional

    if auto_panel and looks_like_document_page(im):
        im = refine_crop_within(im)
        if debug: save_dbg(im, img_path, "refined")

    if types[0] == "light microscopy (H&E or similar)":
        im = auto_enhance_for_histology(im)
        if debug: save_dbg(im, img_path, "enh")

    msgs = build_msgs(question, [im], types, want_clinical_signs, want_histo_details, want_differential)
    raw = _generate(model, processor, msgs, [[im]])
    result, _diff = repair_schema(raw, types,
                                  syndrome_intent(question), immuno_intent(question),
                                  want_clinical_signs, want_histo_details, want_differential, question,
                                  no_guardrails=no_guardrails,
                                  no_knowledge_lines=no_knowledge_lines,
                                  emit_diff=emit_diff)
    return result

def answer_group(model, processor, img_paths: List[Path], question: str,
                 forced_types: Optional[List[str]] = None, crops: Optional[List[Optional[Rect]]] = None,
                 want_clinical_signs: bool = False, want_histo_details: bool = False, want_differential: bool = False,
                 auto_panel: bool = False, auto_two_panels: bool = False, debug: bool = False,
                 no_guardrails: bool = False, no_knowledge_lines: bool = False,
                 doc_strict: bool = False, emit_diff: bool = False) -> str:

    ims_raw = [Image.open(str(p)).convert("RGB") for p in img_paths]
    if debug:
        for i, (p, im) in enumerate(zip(img_paths, ims_raw)):
            print(f"[INFO] image[{i}] size: {im.size[0]}x{im.size[1]} — {p}")
            save_dbg(im, p, "orig", i)

    if auto_two_panels and len(ims_raw) == 1:
        if debug: print("[INFO] auto_two_panels: converting to grouped two-crop run")
        rects = propose_two_panel_crops(ims_raw[0])
        ims = [apply_crop(ims_raw[0], rects[0]), apply_crop(ims_raw[0], rects[1])]
        ims = [refine_crop_within(p) for p in ims]
        if debug:
            save_dbg(ims[0], img_paths[0], "crop", 0)
            save_dbg(ims[1], img_paths[0], "crop", 1)
    else:
        crops = crops or [None] * len(ims_raw)
        ims: List[Image.Image] = []
        for i, im in enumerate(ims_raw):
            im2 = apply_crop(im, crops[i]) if crops[i] else im
            if looks_like_document_page(im2):
                rect = extract_likely_figure_crop(im2)
                im2 = apply_crop(im2, rect)
                im2 = refine_crop_within(im2)
                if debug: save_dbg(im2, img_paths[i], "crop_docfig", i)
            elif auto_panel and crops[i] is None:
                im2 = auto_panel_fallback_crop(im2)
                if debug: save_dbg(im2, img_paths[i], "crop", i)
            ims.append(im2)
        ims = [refine_crop_within(p) for p in ims]

    forced_mask = [False] * len(ims)
    for i in range(len(ims)):
        ims[i], forced_mask[i] = apply_document_guards(
            ims[i], doc_strict, debug, img_paths[0 if auto_two_panels else i], i
        )

    if forced_types:
        if len(forced_types) == 1 and len(ims) > 1:
            forced_types = forced_types * len(ims)
        elif len(forced_types) != len(ims):
            raise SystemExit(f"--types count ({len(forced_types)}) must match number of effective images ({len(ims)}).")
        types = forced_types
    else:
        types = classify_image_types(model, processor, ims)

    for i in range(len(ims)):
        if forced_mask[i]:
            types[i] = "other"

    types = override_types_from_question(question, types)

    for i in range(len(ims)):
        if types[i] == "light microscopy (H&E or similar)":
            ims[i] = auto_enhance_for_histology(ims[i])
            if debug: save_dbg(ims[i], img_paths[0 if auto_two_panels else i], "enh", i)

    msgs = build_msgs(question, ims, types, want_clinical_signs, want_histo_details, want_differential)
    raw = _generate(model, processor, msgs, [ims])
    result, diff = repair_schema(raw, types,
                                 syndrome_intent(question), immuno_intent(question),
                                 want_clinical_signs, want_histo_details, want_differential, question,
                                 no_guardrails=no_guardrails,
                                 no_knowledge_lines=no_knowledge_lines,
                                 emit_diff=emit_diff)

    if emit_diff:
        print("\n[DIFF] Changes made to raw model output:")
        import json
        print(json.dumps(diff or {}, indent=2))

    return result

# ======================
# CLI
# ======================
def run(adapter_dir: Path, image_paths: List[Path], question: str, group: bool,
        types_arg: Optional[str] = None, crops_arg: Optional[str] = None,
        auto_panel: bool = False, auto_two_panels: bool = False, debug: bool = False,
        no_guardrails: bool = False, no_knowledge_lines: bool = False, 
        doc_strict: bool = False, emit_diff: bool = False):
    model = load_model(adapter_dir)
    processor = AutoProcessor.from_pretrained(
        BASE_ID, cache_dir=HF_CACHE, local_files_only=True, token=os.getenv("HF_TOKEN", None)
    )

    forced_types = None
    if types_arg:
        forced_types = [t.strip() for t in types_arg.split(",") if t.strip()]

    crops = parse_crops_arg(crops_arg, len(image_paths)) if crops_arg else None

    want_clinical = clinical_signs_intent(question)
    want_histo    = histology_details_intent(question)
    want_diff     = differential_intent(question)

    if auto_two_panels:
        group = True

    if group:
        ans = answer_group(model, processor, image_paths, question, forced_types, crops,
                        want_clinical, want_histo, want_diff, auto_panel, auto_two_panels, 
                        debug, no_guardrails, no_knowledge_lines, doc_strict, emit_diff)
        print(f"\n=== Grouped ({'auto_two_panels' if auto_two_panels else len(image_paths)} images) ===\nAnswer: {ans}")
    else:
        for i, p in enumerate(image_paths, 1):
            tf = forced_types[i - 1] if forced_types and len(forced_types) >= i else (forced_types[0] if forced_types else None)
            try:
                ans = answer_one(model, processor, p, question, tf,
                                (crops[i - 1] if (crops and len(crops) >= i) else None),
                                want_clinical, want_histo, want_diff, auto_panel, debug,
                                no_guardrails, no_knowledge_lines, doc_strict, emit_diff)
            except Exception as e:
                ans = f"[ERROR] {e}"
            print(f"\n=== Image {i} ===\n{p}\nAnswer: {ans}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--images", required=True, nargs="+", type=Path)
    ap.add_argument("--q", required=True)
    ap.add_argument("--group", action="store_true", help="Treat all images as one multi-image prompt")
    ap.add_argument("--types", type=str, help="Comma-separated forced types in order.")
    ap.add_argument("--crops", type=str, help="Semicolon-separated crops 'x,y,w,h; ...'")
    ap.add_argument("--auto_panel", action="store_true", help="Heuristic single-panel crop (kept for non-doc images)")
    ap.add_argument("--auto_two_panels", action="store_true", help="Heuristic two-panel split from one page")
    ap.add_argument("--debug", action="store_true", help="Write debug crops/enhanced images and print chosen types")
    ap.add_argument("--no_guardrails", action="store_true", help="Skip histology guardrails (pure model output)")
    ap.add_argument("--no_knowledge_lines", action="store_true", help="Skip knowledge-based lines (differential, clinical clues, etc.)")
    ap.add_argument("--doc_strict", action="store_true", help="Force 'other' type for any document-like content")
    ap.add_argument("--emit_diff", action="store_true", help="Show raw vs repaired field differences")
    args = ap.parse_args()
    run(args.adapter, args.images, args.q, args.group, args.types, args.crops, 
        args.auto_panel, args.auto_two_panels, args.debug, args.no_guardrails, 
        args.no_knowledge_lines, args.doc_strict, args.emit_diff)
