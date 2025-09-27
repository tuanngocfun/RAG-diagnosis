#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qdrant_rag.py — ColQwen2 + MedGemma-4B-IT indexing & retrieval (v4.1, cloud-ready)
- Reads page maps (_page_map.json) to align PDF labels with page_####.png
- Exposes helpers used by the evaluator:
    * load_page_map(case_dir)
    * map_page_id_to_indices(page_id, page_map)
    * page_indices_to_paths(case_dir, indices)
    * find_case_dir(case_id, extract_root)
- Adds text_excerpt to search results for downstream RAGAS eval.
- Cloud-friendly: supports Qdrant Cloud via CLI flags or env vars; auto-creates collection.
"""

from __future__ import annotations
import os, re, json, logging, uuid, string, shutil, hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Force offline behavior for HF (optional, harmless if online)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

import torch
from transformers import (
    AutoProcessor, AutoModelForImageTextToText, AutoModelForVision2Seq,
    ColQwen2ForRetrieval, AutoTokenizer, AutoModelForSequenceClassification
)
# Fallback in case ColQwen2Processor isn't exposed in this transformers build
try:
    from transformers import ColQwen2Processor  # type: ignore
except Exception:
    ColQwen2Processor = AutoProcessor  # type: ignore

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm
from qdrant_client.http.models import (
    Distance, VectorParams, PointStruct, NamedVector,
    OptimizersConfigDiff, ScalarQuantization, ScalarQuantizationConfig, ScalarType,
    Filter, FieldCondition, MatchAny, MatchValue
)

# ------------------------------
# Config
# ------------------------------
@dataclass
class CFG:
    ROOT: Path = Path(os.getenv("RAG_ROOT", "/home/students/Leishmania"))
    EXTRACT_ROOT: Path = Path(os.getenv("RAG_EXTRACT_ROOT", str(ROOT / "kaggle" / "working2" / "extract")))  # case_dir/pages/page_*.png

    # Models
    RET_MODEL_ID: str = "vidore/colqwen2-v1.0-hf"
    GEN_MODEL_ID: str = "google/medgemma-4b-it"

    # Re-ranker — can be disabled from caller
    # Reranker: MedCPT Cross-Encoder (true cross-attention scorer)
    RERANKER_MODEL_ID: str = os.getenv("RERANKER_MODEL_ID", "ncbi/MedCPT-Cross-Encoder")
    RERANKER_DOC_MODEL_ID: Optional[str] = None  # unused with cross-encoder
    RERANK_MIN_EXCERPT_CHARS: int = 30   # gate influence if excerpt too short/missing # 40
    RERANK_ALPHA: float = 0.65            # weight for text re-ranker vs. ColQwen2 sim # 0.6
    RERANK_FALLBACK_ALPHA: float = 0.25   # downweight when excerpt is short/missing # 0.2

    # HF cache / device
    HF_CACHE: Path = Path(os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers"))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Qdrant (allow env overrides)
    COLLECTION: str = os.getenv("QDRANT_COLLECTION", "leish_cases_pages")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None

    # Indexing & generation
    BATCH_EMBED: int = 16 # 8
    LANGUAGE: str = "en"
    MAX_NEW_TOKENS: int = 1024 # 768
    TOP_K: int = 15 # 12

    # Retrieval scoring (None => no score_threshold)
    SCORE_THRESHOLD: Optional[float] = None

    # Payload limits
    MAX_TEXT_EXCERPT: int = 1500 # 1200
    MAX_KEYWORDS: int = 35 # 25

    # Processor stability
    USE_FAST_PROCESSORS: bool = True

    PDF_SEARCH_DIRS: Tuple[Path, ...] = tuple(
        Path(p) for p in os.getenv("RAG_PDF_DIRS", f"{ROOT}/data/standard").split(":")
    )

# ------------------------------
# HF local cache helpers (offline-robust)
# ------------------------------

def resolve_local_model_dir(model_id: str, cache_dir: Path) -> str:
    """
    Return a usable path for from_pretrained(..., local_files_only=True).
    Supports:
      1) HF cache layout:   <cache>/models--org--repo/snapshots/<rev>/
      2) Flat local folder: <cache>/<repo>/
      3) Absolute path passed as model_id
    """
    def usable(d: Path) -> bool:
        if not d or not d.exists():
            return False
        has_cfg = (d / "config.json").exists()
        has_proc = (d / "preprocessor_config.json").exists() or (d / "processor_config.json").exists()
        has_tok = (d / "tokenizer.json").exists() or (d / "tokenizer_config.json").exists()
        # Accept either processor (for vision/multimodal) or tokenizer (for text models like BGE)
        return has_cfg and (has_proc or has_tok)

    # 0) Absolute or relative explicit path provided
    p = Path(model_id)
    if p.exists() and usable(p):
        return str(p)

    # 1) Hugging Face cache layout
    repo_dir = cache_dir / f"models--{model_id.replace('/', '--')}"
    snaps_root = repo_dir / "snapshots"
    if snaps_root.is_dir():
        # Prefer "offline-materialized"
        offline = snaps_root / "offline-materialized"
        if usable(offline):
            return str(offline)
        # Otherwise newest usable snapshot
        snaps = sorted((x for x in snaps_root.iterdir() if x.is_dir()),
                       key=lambda x: x.stat().st_mtime, reverse=True)
        for s in snaps:
            if usable(s):
                return str(s)
    # Sometimes files are placed directly under the repo dir
    if usable(repo_dir):
        return str(repo_dir)

    # 2) Flat local folder (your case: /data4t/hf/transformers/bge-reranker-v2-m3)
    flat_dir = cache_dir / model_id.split("/")[-1]
    if usable(flat_dir):
        return str(flat_dir)

    # 3) Fall back to original HF id (will work online; offline will raise)
    return model_id

# ------------------------------
# Utilities (page maps, case dirs, text extraction)
# ------------------------------
def load_page_map(case_dir: Path) -> Optional[Dict[str, Any]]:
    """Load pages/_page_map.json if present; return dict or None."""
    p = case_dir / "pages" / "_page_map.json"
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None

_ROMAN = {
    "i":1,"ii":2,"iii":3,"iv":4,"v":5,"vi":6,"vii":7,"viii":8,"ix":9,"x":10,
    "xi":11,"xii":12,"xiii":13,"xiv":14,"xv":15,"xvi":16,"xvii":17,"xviii":18,"xix":19,"xx":20
}

def map_page_id_to_indices(page_id: str, page_map: Optional[Dict[str,Any]]) -> List[int]:
    """Map JSONL gold_evidence page_id -> list of 0-based indices using page_map labels when available.
       Supports ranges like '44-45' and roman numerals like 'xi'.
    """
    if not page_id:
        return []
    page_id = str(page_id).strip()

    # Range like '44-45'
    m = re.match(r"^\s*(\d+)\s*-\s*(\d+)\s*$", page_id)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        lo, hi = (a, b) if a <= b else (b, a)
        return [i-1 for i in range(lo, hi+1)]

    # Integer
    if page_id.isdigit():
        i = int(page_id) - 1
        return [i] if i >= 0 else []

    # Roman numerals
    low = page_id.lower()
    if low in _ROMAN:
        return [_ROMAN[low] - 1]

    # Fallback via label_to_index (e202, S12, etc.)
    if page_map and "label_to_index" in page_map:
        idx = page_map["label_to_index"].get(page_id)
        if isinstance(idx, int):
            return [idx]

    return []

def page_indices_to_paths(case_dir: Path, indices: List[int]) -> List[Path]:
    """Return image paths for 0-based page indices using _page_map.json if available."""
    pages_dir = case_dir / "pages"
    pm = load_page_map(case_dir)
    out: List[Path] = []
    for i in indices:
        if pm and "index_to_png" in pm and str(i) in pm["index_to_png"]:
            p = pages_dir / pm["index_to_png"][str(i)]
        else:
            p = pages_dir / f"page_{i+1:04d}.png"
        if p.exists():
            out.append(p)
    return out

def sanitize_name(name: str) -> str:
    s = re.sub(r"[^\w\-.]", "_", name)
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def find_case_dir(case_id: str, extract_root: Path) -> Optional[Path]:
    """Locate a case folder under EXTRACT_ROOT from a human case_id (JSONL)."""
    # exact match
    cand = extract_root / case_id
    if cand.is_dir():
        return cand
    # sanitized
    cand2 = extract_root / sanitize_name(case_id)
    if cand2.is_dir():
        return cand2
    # loose (contains)
    for p in extract_root.iterdir():
        if p.is_dir() and sanitize_name(case_id).lower() in p.name.lower():
            return p
    return None

# ------------------------------
# Text helpers (for keywords/entities)
# ------------------------------
_STOP = {
    "the","a","an","and","or","of","on","in","to","for","from","with","without","by","at","as","is","are",
    "this","that","these","those","into","about","it","its","be","been","being","we","our","you","your",
    "may","can","could","should","would","will","also","than","then","thus","such","not","no","yes",
    "case","cases","study","report","figure","fig","table","image","images","page","pages"
}
def tokenize_words(s: str) -> List[str]:
    s = s.lower().translate(str.maketrans("", "", string.punctuation))
    return [w for w in s.split() if w.isalpha() and 2 < len(w) < 30]

def extract_keywords(title: str, text: str, top_k: int = CFG.MAX_KEYWORDS) -> List[str]:
    words = tokenize_words((title or "") + " " + (text or ""))
    counts: Dict[str,int] = {}
    for w in words:
        if w in _STOP: continue
        counts[w] = counts.get(w, 0) + 1
    
    # EXPANDED boost dictionary with more clinical terms
    boosts = {
        "leishmaniasis":4, "leishmania":4, "amastigotes":5, "promastigotes":4, 
        "auricular":4, "pinna":3, "hiv":4, "mucocutaneous":4, "visceral":4, 
        "cutaneous":3, "martiniquensis":4, "squamous":3, "carcinoma":3, 
        "lupus":3, "vulgaris":3, "recidivans":4, "mimicking":3,
        # New high-value terms
        "kinetoplast":5, "giemsa":5, "wright":4, "histopathology":4,
        "biopsy":4, "pcr":5, "its2":5, "culture":4, "diagnosis":4,
        "treatment":3, "amphotericin":4, "antimony":4, "miltefosine":4,
        "immunocompromised":4, "transplant":4, "lesion":3, "ulcer":3,
        "nodule":3, "papule":3, "plaque":3, "erythematous":3,
        "granuloma":4, "histiocytes":4, "macrophages":4, "lymphocytes":3,
        "epithelioid":4, "necrosis":3, "pseudoepitheliomatous":4,
        "hyperplasia":4, "acanthosis":3, "spongiosis":3
    }
    
    # Apply n-gram extraction for compound terms
    bigrams = []
    for i in range(len(words)-1):
        if words[i] not in _STOP and words[i+1] not in _STOP:
            bigram = f"{words[i]}_{words[i+1]}"
            if any(term in bigram for term in ["leishman", "donovan", "wright", "giemsa"]):
                bigrams.append(bigram)
                counts[bigram] = counts.get(bigram, 0) + 5  # High boost for diagnostic terms
    
    for k,b in boosts.items():
        if k in counts: counts[k] *= b
    
    return [w for w,_ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]

def guess_case_type(title: str, text: str) -> str:
    s = f"{title} {text}".lower()
    if "visceral" in s or "kala-azar" in s: return "visceral"
    if "mucocutaneous" in s or "mucosal" in s: return "mucocutaneous"
    if "cutaneous" in s: return "cutaneous"
    return "unknown"

def guess_case_type_document(title: str, all_pages_text: List[str]) -> str:
    """Determine case type at document level to avoid mislabeling."""
    # Title takes precedence
    title_lower = title.lower()
    if "cutaneous" in title_lower:
        return "cutaneous"
    if "mucocutaneous" in title_lower or "mucosal" in title_lower:
        return "mucocutaneous"
    if "visceral" in title_lower or "kala-azar" in title_lower:
        return "visceral"
    
    # Vote across all pages
    votes = {"cutaneous": 0, "mucocutaneous": 0, "visceral": 0}
    for text in all_pages_text:
        text_lower = text.lower()
        if "cutaneous" in text_lower:
            votes["cutaneous"] += 1
        if "mucocutaneous" in text_lower or "mucosal" in text_lower:
            votes["mucocutaneous"] += 1
        if "visceral" in text_lower or "kala-azar" in text_lower:
            votes["visceral"] += 1
    
    # Return the majority vote
    if votes["cutaneous"] > votes["visceral"] and votes["cutaneous"] > votes["mucocutaneous"]:
        return "cutaneous"
    elif votes["visceral"] > votes["cutaneous"] and votes["visceral"] > votes["mucocutaneous"]:
        return "visceral"
    elif votes["mucocutaneous"] > 0:
        return "mucocutaneous"
    
    return "unknown"

def extract_entities(title: str, text: str) -> List[str]:
    s = f"{title} {text}"
    ents = set()
    for pat in [
        r"Leishmania\s+[A-Z][a-z]+", r"L\.\s*[a-z]+",
        r"martiniquensis", r"donovani", r"infantum", r"tropica", r"major", r"braziliensis", r"mexicana",
        r"amastigote[s]?", r"promastigote[s]?", r"macrophage[s]?", r"histiocyte[s]?", r"auricular", r"pinna",
        r"HIV", r"immunosuppression", r"immunocompromis(ed|ed)", r"squamous\s+cell\s+carcinoma", r"lupus\s+vulgaris",
        r"recidivans", r"\bViannia\b", r"\bpanamensis\b", r"\bL\.\s*(?:panamensis|braziliensis|guyanensis)\b",
        r"\bPCR\b", r"\bITS2\b", r"\bNovy[-–]Mac(?:Neal|Neill)[-–]Nicolle\b",
        r"\bpentavalent\s+antimony\b", r"\bamphotericin(?:\s+B)?\b", r"\bliposomal amphotericin\b",
        r"\bepitrochlear\b", r"\bsporotrichoid\b"
    ]:
        for m in re.findall(pat, s, flags=re.I):
            ents.add(m.strip())
    return sorted(list(ents))[:20]

# ------------------------------
# Evidence extraction shared helpers
# ------------------------------
# Expanded clinical terms for better evidence coverage (module-level for reuse)
KEEP_TERMS_REGEX = re.compile(
    r"\b(amastigote|Leishman[-\s]?Donovan|macrophage|histiocyte|granuloma|"
    r"pseudoepitheliomatous|hyperplasia|suppurative|ulcer|nodule|nodular|plaque|papule|"
    r"crust(?:ed|ing)?|induration|border|erythema(?:tous)?|cheek|face|facial|pinna|auricle|ear|"
    r"Leishmania|leishmaniasis|cutaneous|mucocutaneous|visceral|promastigote|kinetoplast|"
    r"sandfly|biopsy|PCR|ITS2|culture|Novy[-–]Mac(?:Neal|Neill)[-–]Nicolle|tissue\s+cultures?|"
    r"immunocompromised|HIV|lesion|size|\bcm\b|\bmm\b|course|onset|month(?:s)?|evolved|progress(?:ed|ion)|"
    r"left|right|lateral|medial|anterior|posterior|lymphadenopathy|epitrochlear|sporotrichoid|Peru|Amazon|"
    r"antimony|pentavalent|amphotericin|liposomal|miltefosine|dose|dosage|\bmg\b|\bkg\b|treatment|regimen)\b",
    re.I,
)

def safe_trim(text: str, max_len: int) -> str:
    """Trim text safely to complete sentences without ellipses."""
    if not text or len(text) <= max_len:
        return text
    
    # Try to find a sentence boundary within reasonable range
    search_window = min(len(text), max_len + 200)
    search_text = text[:search_window]
    
    # Find last complete sentence within max_len
    import re
    # Look for sentence endings followed by space or end of string
    sentence_ends = list(re.finditer(r'[.!?](?:\s+|$)', search_text))
    
    if sentence_ends:
        # Find the last sentence that fits
        for match in reversed(sentence_ends):
            if match.start() <= max_len:
                cut_point = match.end() - 1 if match.group().endswith(' ') else match.end()
                result = text[:cut_point].strip()
                # Ensure proper ending punctuation
                if result and result[-1] not in '.!?':
                    result += '.'
                return result
    
    # Fallback: cut at max_len and add period
    result = text[:max_len].strip()
    # Remove incomplete brackets/citations at the end
    result = re.sub(r'\[[^\]]*$', '', result).rstrip()
    result = re.sub(r'\([^\)]*$', '', result).rstrip()
    if result and result[-1] not in '.!?':
        result += '.'
    return result

# ------------------------------
# Image hashing & metrics
# ------------------------------
def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    im = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32)
    med = np.median(arr)
    bits = (arr > med).astype(np.uint8)
    return "".join("01"[b] for b in bits.flatten())

def image_metrics(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    arr_hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    sat_mean = float(arr_hsv[...,1].mean())

    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:,1:-1] = (g[:,2:] - g[:,:-2]) * 0.5
    gy[1:-1,:] = (g[2:,:] - g[:-2,:]) * 0.5
    grad = np.sqrt(gx*gx + gy*gy)
    edge_density = float((grad > 0.2).mean())

    hist = np.histogram((g*255).astype(np.uint8), bins=256, range=(0,255))[0].astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    if edge_density < 0.02 and sat_mean < 25:
        page_kind = "mostly_text"
    elif edge_density > 0.045 and sat_mean < 60 and entropy > 4.9:
        page_kind = "figure_or_micrograph"
    else:
        page_kind = "mixed"

    return dict(
        width=w, height=h,
        sat_mean=sat_mean,
        edge_density=edge_density,
        entropy=entropy,
        page_kind=page_kind,
        micrograph_like=(page_kind == "figure_or_micrograph"),
        ahash=average_hash(img)
    )

def file_info(p: Path) -> Dict[str, Any]:
    try:
        st = p.stat()
        return {"bytes": int(st.st_size), "mtime": int(st.st_mtime)}
    except Exception:
        return {"bytes": None, "mtime": None}

# ------------------------------
# ColQwen2 retriever
# ------------------------------
class CQ2:
    def __init__(self, model_id: str):
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self._dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)

        self.processor = ColQwen2Processor.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            use_fast=CFG.USE_FAST_PROCESSORS
        )
        self.model = ColQwen2ForRetrieval.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            device_map="auto", torch_dtype=self._dtype
        ).eval()
        try: self.model.to(self._dtype)
        except Exception: pass

    @torch.inference_mode()
    def _pool(self, seq: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.float()
        return (seq * m.unsqueeze(-1)).sum(1) / (m.sum(1, keepdim=True) + 1e-6)

    @torch.inference_mode()
    def embed_images(self, pil_images: List[Image.Image]) -> np.ndarray:
        batch = self.processor(images=pil_images, return_tensors="pt")
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        feats = None
        # 1) API ưu tiên nếu có
        if hasattr(self.model, "get_image_features"):
            try:
                feats = self.model.get_image_features(**batch)
            except Exception:
                feats = None
        # 2) Fallback: gọi forward rồi lấy trường phù hợp
        if feats is None:
            out = self.model(**batch)
            # out có thể là tuple/list hoặc ModelOutput
            if isinstance(out, (tuple, list)):
                feats = out[0] if len(out) > 0 else None
            else:
                for name in ("image_embeds", "embeddings", "last_hidden_state", "pooler_output"):
                    val = getattr(out, name, None)
                    if val is not None:
                        feats = val
                        break
        if feats is None:
            raise RuntimeError("No image embeddings found on model output")
        # If sequence-like, average over sequence dim
        if hasattr(feats, "ndim") and feats.ndim == 3:
            pooled = feats.mean(dim=1)
        else:
            pooled = feats
        
        # Ensure proper dtype conversion before CPU transfer
        if pooled.dtype == torch.bfloat16:
            pooled = pooled.float()
        
        return pooled.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_texts(self, queries: List[str]) -> np.ndarray:
        batch = self.processor(text=queries, return_tensors="pt")
        batch = {k: v.to(self.model.device) for k, v in batch.items()}

        feats = None
        # 1) API ưu tiên nếu có
        if hasattr(self.model, "get_text_features"):
            try:
                feats = self.model.get_text_features(**batch)
            except Exception:
                feats = None
        # 2) Fallback: gọi forward rồi lấy trường phù hợp
        if feats is None:
            out = self.model(**batch)
            if isinstance(out, (tuple, list)):
                feats = out[0] if len(out) > 0 else None
            else:
                for name in ("text_embeds", "embeddings", "last_hidden_state", "pooler_output"):
                    val = getattr(out, name, None)
                    if val is not None:
                        feats = val
                        break
        if feats is None:
            raise RuntimeError("No text embeddings found on model output")
        # If sequence-like, mean-pool with attention mask when available
        if hasattr(feats, "ndim") and feats.ndim == 3:
            mask = batch.get("attention_mask", None)
            pooled = feats.mean(dim=1) if mask is None else self._pool(feats, mask)
        else:
            pooled = feats
        
        # Ensure proper dtype conversion before CPU transfer
        if pooled.dtype == torch.bfloat16:
            pooled = pooled.float()
            
        return pooled.detach().cpu().numpy().astype(np.float32)
    
# MedGemma-4B-IT generator (HF, offline)
# ------------------------------
class MedGemma4B:
    """
    Local multimodal generator using google/medgemma-4b-it via HF Transformers.
    Mirrors the Gemini25.answer(...) API so callers remain unchanged.
    """
    def __init__(self, model_id: str = None):
        model_id = model_id or CFG.GEN_MODEL_ID  # default: "google/medgemma-4b-it"
        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)

        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            self._dtype = torch.bfloat16
        elif torch.cuda.is_available():
            self._dtype = torch.float16
        else:
            self._dtype = torch.float32

        # Processor & model (local only)
        self.processor = AutoProcessor.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            use_fast=CFG.USE_FAST_PROCESSORS
        )
        # Prefer Vision2Seq when available; fallback to ImageTextToText
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
                device_map="auto", torch_dtype=self._dtype
            ).eval()
        except Exception:
            self.model = AutoModelForImageTextToText.from_pretrained(
                name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
                device_map="auto", torch_dtype=self._dtype
            ).eval()

        # Some higher-level code expects a small config-like object on the generator.
        # Provide a minimal shim to avoid attribute errors (mirrors Gemini wrapper usage).
        try:
            from types import SimpleNamespace
        except Exception:  # extremely unlikely
            class SimpleNamespace:  # fallback shim
                def __init__(self, **kwargs):
                    for k, v in kwargs.items():
                        setattr(self, k, v)

        # Limits used by prompt/continuation heuristics
        self.cfg_cls = SimpleNamespace(
            max_text_context=16000,                # chars of supporting context to pass
            max_new_tokens=int(CFG.MAX_NEW_TOKENS),# generation cap
            images_per_answer_default=3            # default number of images per answer
        )

        # Tokenizer (if exposed by processor)
        self.tokenizer = getattr(self.processor, "tokenizer", None)
        if self.tokenizer is None:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(
                    name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True
                )
            except Exception:
                self.tokenizer = None

    def _build_prompt_text(
        self,
        q: str,
        spans: List[Tuple[str, str]],
        context_text: str,
    ) -> str:
        """
        FIXED: Simplified prompt builder that prevents garbage generation
        """
        # Detect question type for focused instruction
        q_lower = q.lower()
        is_treatment = any(term in q_lower for term in ["cure", "treat", "therapy", "management"])
        is_diagnostic = any(term in q_lower for term in ["diagnosis", "identify", "what is", "disease"])
        
        # Use simple, focused instructions
        if is_treatment:
            instruction = "Provide a concise treatment answer (2-3 sentences maximum):"
        elif is_diagnostic:
            instruction = "Provide a clear diagnostic assessment (2-3 sentences maximum):"
        else:
            instruction = "Provide a focused medical answer (2-3 sentences maximum):"
        
        # Build concise evidence (LIMIT TO PREVENT OVERFLOW)
        evidence_text = ""
        if spans:
            # Only use top 3 most relevant spans to prevent information overload
            relevant_spans = []
            for span_text, citation in spans[:3]:
                if len(span_text.strip()) > 20:  # Only substantial spans
                    relevant_spans.append(span_text.strip()[:200])  # Truncate long spans
            
            if relevant_spans:
                evidence_text = "Evidence: " + " ".join(relevant_spans)
        
        # Add minimal context (HEAVILY TRUNCATED)
        if context_text and len(context_text) > 500:
            context_text = context_text[:500] + "..."
        
        # Create SIMPLE, focused prompt (NO COMPLEX INSTRUCTIONS)
        prompt = f"""Medical Question: {q}

{instruction}

{evidence_text}

{context_text}

Answer:"""
        
        return prompt

    @staticmethod
    def _encode_image(pil_img: Image.Image) -> dict:
        """Return an inline binary image part compatible with google-genai 0.6.0, with downsizing."""
        import io
        img = pil_img.copy()
        
        # Remove ICC profile if present
        if "icc_profile" in img.info:
            try:
                del img.info["icc_profile"]
            except Exception:
                pass
        
        # Downscale to max 1280px on longest side to reduce token usage
        max_side = 1280
        w, h = img.size
        scale = min(1.0, max_side / max(w, h))
        if scale < 1.0:
            img = img.resize((int(w*scale), int(h*scale)), Image.LANCZOS)
        
        # Encode as JPEG instead of PNG (much smaller)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True, progressive=True)
        data = buf.getvalue()
        
        return {"inline_data": {"mime_type": "image/jpeg", "data": data}}

    def _build_system_and_user(
        self,
        q: str,
        ims: List[Image.Image],
        spans: List[Tuple[str, str]],
        context_text: str = ""
    ):
        """
        Enhanced prompt building with medical focus
        """
        # Detect question type for tailored instructions
        is_diagnostic = any(term in q.lower() for term in ["diagnosis", "identify", "stain", "microscopy"])
        is_clinical = any(term in q.lower() for term in ["clinical", "presentation", "treatment"])
        
        # Limit images based on question type
        if is_diagnostic:
            ims = ims[:3]  # Allow more images for diagnostic questions
        else:
            ims = ims[:2]  # Standard limit

        if spans:
            # Enhanced system instruction based on question type
            if is_diagnostic:
                sys_instr = (
                    "You are a medical diagnostician analyzing clinical case images and documentation. "
                    "Your task is to provide a systematic diagnostic assessment based on the available evidence. "
                    "Structure your response as follows: "
                    "1) Key clinical findings from images and text, "
                    "2) Differential diagnosis considerations, "
                    "3) Most likely diagnosis with reasoning. "
                    "For parasitic infections, specifically mention: morphological features, tissue reactions, "
                    "geographic/epidemiologic factors, and diagnostic methods used. "
                    "Avoid repeating case titles or descriptive headers. Focus on medical analysis. "
                    "Cite evidence as [1], [2], etc."
                )
            elif is_clinical:
                sys_instr = (
                    "You are a clinical medicine specialist analyzing patient presentation and disease course. "
                    "Provide a concise clinical summary focusing on: "
                    "1) Patient demographics and presentation, "
                    "2) Lesion characteristics and evolution, "
                    "3) Geographic/travel history relevance, "
                    "4) Treatment approach and outcomes. "
                    "Avoid repeating case numbers or titles. Present information in a flowing narrative. "
                    "Only include details supported by evidence [1], [2], etc. "
                    "If key details are missing, briefly note: 'Additional [detail] not documented.'"
                )
            else:
                sys_instr = (
                    "You are a medical expert providing evidence-based analysis of clinical cases. "
                    "Answer the specific question asked using the provided evidence and images. "
                    "Structure your response logically and avoid repeating case identifiers or titles. "
                    "Focus on medical content and reasoning. Cite sources as [1], [2], etc. "
                    "Provide complete, precise answers without unnecessary repetition."
                )
            
            # Format evidence with medical context
            ev_lines = []
            for i, (s, c) in enumerate(spans):
                # Clean up citation format
                clean_cite = c.replace("unknown:", "").replace("p-1", "p0")
                ev_lines.append(f"[{i+1}] (Page {clean_cite}): {s}")
            
            ev_text = "\n".join(ev_lines)
            user_text = f"Medical Question: {q}\n\nEvidence Sources:\n{ev_text}\n"
            
            # Add focused instructions for common question types
            if "species" in q.lower() or "identification" in q.lower():
                user_text += "\nNote: Species identification requires molecular/culture evidence, not morphology alone.\n"
            
        else:
            sys_instr = (
                "You are a medical assistant analyzing clinical images. "
                "Describe only what is directly visible in the images. "
                "Avoid speculation about diagnosis without supporting evidence."
            )
            user_text = f"Medical Question: {q}"

        if context_text:
            # Smart context truncation preserving medical terms
            context_limit = 8000  # Increased from 6000
            if len(context_text) > context_limit:
                # Try to preserve complete sentences with medical terms
                sentences = context_text.split('. ')
                medical_sentences = [s for s in sentences if any(
                    term in s.lower() for term in [
                        "leishmania", "diagnosis", "treatment", "clinical",
                        "histopathology", "microscopy", "culture", "pcr"
                    ]
                )]
                other_sentences = [s for s in sentences if s not in medical_sentences]
                
                rebuilt = ". ".join(medical_sentences[:50] + other_sentences[:30])
                if len(rebuilt) > context_limit:
                    context_text = rebuilt[:context_limit] + "..."
                else:
                    context_text = rebuilt
            
            user_text += f"\n\nAdditional Medical Context:\n{context_text}\n"

        parts = [{"text": user_text}]
        for im in ims:
            parts.append(self._encode_image(im))

        contents = [{"role": "user", "parts": parts}]
        return sys_instr, contents

    def answer(
        self,
        q: str,
        image_paths: List[Path],
        spans: List[Tuple[str, str]] = None,
        max_output_tokens: int = 512,
        context_text: str = "",
        images_per_answer: int = 3  # ADD THIS PARAMETER
    ) -> str:
        spans = spans or []

        # FIXED: Better image validation and limiting
        img_paths = []
        for path in image_paths[:min(2, int(images_per_answer or 1))]:  # Hard limit to 2 images
            try:
                if Path(path).exists() and Path(path).stat().st_size > 0:
                    img_paths.append(path)
            except Exception:
                continue
        
        print(f"[DEBUG] Validated {len(img_paths)} images from {len(image_paths)} provided")

        # Build prompt
        prompt = self._build_prompt_text(q, spans, context_text)

        # Load images safely
        ims: List[Image.Image] = []
        for p in img_paths:
            try:
                im = Image.open(p).convert("RGB")
                ims.append(im)
            except Exception:
                pass

        def _close_all():
            for _im in ims:
                try:
                    _im.close()
                except Exception:
                    pass

        try:
            # FIXED: Use proper MedGemma 4B-IT message format instead of manual tokens
            if ims:
                print(f"[DEBUG] Preparing multimodal input with {len(ims)} images")
                # Create proper message structure as shown in MedGemma notebooks
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt}
                        ] + [{"type": "image", "image": img} for img in ims]
                    }
                ]
            else:
                print(f"[DEBUG] Preparing text-only input")
                # Text-only message structure
                messages = [
                    {
                        "role": "user", 
                        "content": [{"type": "text", "text": prompt}]
                    }
                ]

            # Use processor.apply_chat_template() as shown in notebooks
            chat_prompt = None
            if self.processor is not None and hasattr(self.processor, "apply_chat_template"):
                try:
                    chat_prompt = self.processor.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True
                    )
                    print(f"[DEBUG] Processor chat template applied successfully")
                    print(f"[DEBUG] Chat prompt length: {len(chat_prompt)} chars")
                except Exception as e:
                    print(f"[DEBUG] Processor chat template failed: {e}")
                    chat_prompt = None

            # Fallback to tokenizer if processor fails
            if chat_prompt is None and self.tokenizer is not None and hasattr(self.tokenizer, "apply_chat_template"):
                try:
                    # Flatten content for tokenizer (it may not support structured content)
                    simple_messages = [{"role": "user", "content": prompt}]
                    chat_prompt = self.tokenizer.apply_chat_template(
                        simple_messages, tokenize=False, add_generation_prompt=True
                    )
                    print(f"[DEBUG] Tokenizer chat template applied as fallback")
                except Exception as e:
                    print(f"[DEBUG] Tokenizer chat template also failed: {e}")
                    chat_prompt = None

            # Final fallback to raw prompt
            text_for_gen = chat_prompt or prompt
            print(f"[DEBUG] Final text_for_gen length: {len(text_for_gen)} chars")
            print(f"[DEBUG] Images to process: {len(ims)}")
            
            try:
                # FIXED: Use proper MedGemma 4B-IT processor call as shown in notebooks
                if ims:
                    print(f"[INFO] Processing multimodal input with {len(ims)} images")
                    # Method 1: Try with structured messages (preferred)
                    try:
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        print(f"[DEBUG] Structured message processing succeeded")
                    except Exception as e:
                        print(f"[DEBUG] Structured processing failed: {e}")
                        # Method 2: Fallback to text+images format
                        inputs = self.processor(
                            text=text_for_gen,
                            images=ims,
                            return_tensors="pt",
                        )
                        print(f"[DEBUG] Text+images fallback succeeded")
                else:
                    print(f"[INFO] Processing text-only input")
                    # Text-only processing - try structured first, then simple
                    try:
                        inputs = self.processor.apply_chat_template(
                            messages,
                            add_generation_prompt=True,
                            tokenize=True,
                            return_dict=True,
                            return_tensors="pt",
                        )
                        print(f"[DEBUG] Text-only structured processing succeeded")
                    except Exception as e:
                        print(f"[DEBUG] Text-only structured failed: {e}")
                        inputs = self.processor(text=text_for_gen, return_tensors="pt")
                        print(f"[DEBUG] Simple text processing succeeded")
                    
                print(f"[DEBUG] Final inputs keys: {list(inputs.keys())}")
                if "input_ids" in inputs:
                    print(f"[DEBUG] Input IDs shape: {inputs['input_ids'].shape}")
                
            except ValueError as e:
                error_msg = str(e).lower()
                if "image tokens" in error_msg or "image" in error_msg:
                    print(f"[WARN] Multimodal processing failed, using text-only: {e}")
                    # Complete fallback to text-only
                    try:
                        inputs = self.processor(text=text_for_gen, return_tensors="pt")
                        ims = []  # Clear images since we're falling back
                        print(f"[DEBUG] Text-only fallback succeeded")
                    except Exception as fallback_e:
                        print(f"[ERROR] Even text-only fallback failed: {fallback_e}")
                        raise
                else:
                    print(f"[ERROR] Processing failed: {type(e).__name__}: {e}")
                    raise
            except Exception as e:
                print(f"[ERROR] Processing failed: {type(e).__name__}: {e}")
                print(f"[DEBUG] Processor type: {type(self.processor)}")
                print(f"[DEBUG] Text sample: {repr(text_for_gen[:200])}")
                print(f"[DEBUG] Image types: {[type(img) for img in ims]}")
                print(f"[DEBUG] Messages structure: {messages}")
                import traceback
                print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
                raise

            # Move to device
            inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            eos_id = self.tokenizer.eos_token_id if self.tokenizer is not None else None
            pad_id = None
            if self.tokenizer is not None:
                pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else eos_id

            # Adjust parameters based on question type for better responses
            is_diagnostic_q = any(term in prompt.lower() for term in ["diagnose", "diagnosis", "identify", "disease"])
            
            # Add entropy to avoid cached responses - use question hash as seed
            import hashlib
            import random
            question_hash = int(hashlib.md5(q.encode()).hexdigest()[:8], 16)
            random.seed(question_hash)
            temperature_offset = random.uniform(-0.1, 0.1)
            
            # FIXED: Conservative generation parameters to prevent garbage output
            safe_max_tokens = min(int(max_output_tokens or 512), 256)  # Much more conservative
            
            gen_kwargs = dict(
                max_new_tokens=safe_max_tokens,
                min_new_tokens=20,  # Minimum reasonable response
                do_sample=True,
                num_beams=1,
                top_p=0.9,  # More focused sampling
                temperature=0.7,  # Stable temperature
                repetition_penalty=1.2,  # Strong penalty for repetition
                length_penalty=0.9,  # Slight preference for shorter answers
                no_repeat_ngram_size=4,  # Prevent repetitive phrases
                eos_token_id=eos_id,
                pad_token_id=pad_id,
                early_stopping=True,  # Stop when complete
            )
            # FIXED: Remove the else branch - use single conservative parameter set

            # DEBUG: Log key generation parameters
            print(f"[DEBUG] Generation parameters: max_new_tokens={gen_kwargs.get('max_new_tokens')}, temperature={gen_kwargs.get('temperature')}")
            print(f"[DEBUG] Prompt text (first 200 chars): {repr(text_for_gen[:200])}")  
            print(f"[DEBUG] Number of images: {len(ims)}")
            
            try:
                # Get input length for proper decoding (following notebook pattern)
                input_len = inputs["input_ids"].shape[-1] if "input_ids" in inputs else 0
                print(f"[DEBUG] Input length: {input_len}")
                
                with torch.inference_mode():
                    generated = self.model.generate(**inputs, **{k: v for k, v in gen_kwargs.items() if v is not None})
                    
                # Extract only the new tokens (following notebook pattern)
                if input_len > 0 and len(generated.shape) > 1:
                    new_tokens = generated[0][input_len:]
                    print(f"[DEBUG] Generated {len(new_tokens)} new tokens")
                else:
                    new_tokens = generated[0] if len(generated.shape) > 1 else generated
                    print(f"[DEBUG] Using full generation ({len(new_tokens)} tokens)")
                    
            except Exception as e:
                print(f"[ERROR] Model generation failed: {type(e).__name__}: {e}")
                print(f"[DEBUG] Inputs keys: {list(inputs.keys())}")
                print(f"[DEBUG] Input shapes: {[(k, getattr(v, 'shape', 'no shape')) for k, v in inputs.items()]}")
                print(f"[DEBUG] Gen kwargs: {gen_kwargs}")
                import traceback
                print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
                raise  # Re-raise the exception to trigger fallback logic

            # Decode using processor (preferred) or tokenizer fallback
            try:
                if self.processor is not None:
                    text = self.processor.decode(new_tokens, skip_special_tokens=True)
                    print(f"[DEBUG] Used processor.decode()")
                elif self.tokenizer is not None:
                    text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
                    print(f"[DEBUG] Used tokenizer.decode()")
                else:
                    text = ""
                    print(f"[WARN] No decoder available")
            except Exception as e:
                print(f"[WARN] Decode failed: {e}, trying batch_decode")
                # Fallback to batch decode
                if self.tokenizer is not None:
                    out = self.tokenizer.batch_decode([new_tokens], skip_special_tokens=True)
                    text = out[0] if out else ""
                else:
                    text = ""

            text = (text or "").strip()
            print(f"[DEBUG] Decoded output: {repr(text[:300])}")
            def _normalize_gen_out(txt: str) -> str:
                import re as _re
                if not txt:
                    return txt
                t = txt
                
                # Remove echoed prompts and prefixes
                t = _re.sub(r"^\s*(?:Question|QUESTION|MEDICAL QUESTION)\s*:\s*.*?(?:Answer|ANSWER|MEDICAL ANALYSIS)\s*:\s*", "", t, flags=_re.I | _re.S)
                t = _re.sub(r"^\s*(?:Answer|ANSWER|MEDICAL ANALYSIS)\s*:\s*", "", t, flags=_re.I)
                
                # Remove case title repetitions early
                t = _re.sub(r"\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+[A-Z\s]+\s*[.M]*\s*", "", t, flags=_re.I | _re.M)
                t = _re.sub(r"The lesion progressed quickly from a sore to eat through[^.]*\.\s*(?=The lesion progressed quickly)", "", t, flags=_re.I)
                
                # Keep last 'Final Answer:' block if repeated
                idx = t.lower().rfind("final answer:")
                if idx != -1:
                    tail = t[idx + len("final answer:"):].strip()
                    if tail:
                        t = tail
                t = _re.sub(r"^final\s+answer:\s*", "", t, flags=_re.I).strip()
                
                # Remove LaTeX wrappers
                t = _re.sub(r"\\boxed\\{([^}]*)\\}", r"\1", t)
                t = _re.sub(r"\$\$?(.*?)\$\$?", r"\1", t, flags=_re.S)
                
                # Improve sentence structure
                t = _re.sub(r"\s+", " ", t).strip()
                t = _re.sub(r"\s*(?:…|\.{3,})\s*$", ".", t)
                
                # Ensure proper ending
                if t and not t.endswith((".", "!", "?")):
                    t += "."
                
                return t

            if text:
                print(f"[DEBUG] Before normalization: {repr(text[:300])}")
                norm = _normalize_gen_out(text)
                print(f"[DEBUG] After local normalization: {repr(norm[:300])}")
                if norm:
                    return norm

            # Fallback: text-only generation if image path failed silently
            try:
                # Remove image tokens for text-only fallback
                text_only_prompt = prompt.replace("<image>\n", "").replace("<image>", "")
                inputs2 = self.processor(text=text_only_prompt, return_tensors="pt")
                inputs2 = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs2.items()}
                with torch.inference_mode():
                    generated2 = self.model.generate(**inputs2, **{k: v for k, v in gen_kwargs.items() if v is not None})
            except Exception as e:
                print(f"[ERROR] Fallback text-only generation failed: {type(e).__name__}: {e}")
                print(f"[DEBUG] Fallback inputs keys: {list(inputs2.keys()) if 'inputs2' in locals() else 'N/A'}")
                raise  # Re-raise to trigger the final retry
            if self.tokenizer is not None:
                out2 = self.tokenizer.batch_decode(generated2, skip_special_tokens=True)
            else:
                decode = getattr(self.processor, "batch_decode", None)
                out2 = decode(generated2, skip_special_tokens=True) if callable(decode) else None
            text2 = (out2[0] if isinstance(out2, list) and out2 else "").strip()
            norm2 = _normalize_gen_out(text2)
            if norm2:
                return norm2

            # Force a final retry with minimal constraints instead of returning evidence
            try:
                # Attempt simple direct generation without complex prompting
                simple_prompt = f"Medical Question: {q}\n\nProvide a concise medical answer based on the available information."
                # Ensure no image tokens in simple text-only generation
                simple_prompt = simple_prompt.replace("<image>\n", "").replace("<image>", "")
                simple_inputs = self.processor(text=simple_prompt, return_tensors="pt")
                simple_inputs = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in simple_inputs.items()}
                
                try:
                    with torch.inference_mode():
                        simple_gen = self.model.generate(
                            **simple_inputs,
                            max_new_tokens=256,
                            min_new_tokens=30,
                            do_sample=True,
                            temperature=0.8,
                            top_p=0.9,
                            repetition_penalty=1.3,
                            no_repeat_ngram_size=4,
                            eos_token_id=eos_id,
                            pad_token_id=pad_id,
                            early_stopping=True
                        )
                except Exception as e:
                    print(f"[ERROR] Simple retry generation failed: {type(e).__name__}: {e}")
                    print(f"[DEBUG] Simple inputs keys: {list(simple_inputs.keys())}")
                    raise  # Re-raise to trigger the final fallback
                
                if self.tokenizer is not None:
                    simple_out = self.tokenizer.batch_decode(simple_gen, skip_special_tokens=True)
                else:
                    decode = getattr(self.processor, "batch_decode", None)
                    simple_out = decode(simple_gen, skip_special_tokens=True) if callable(decode) else None
                
                simple_text = (simple_out[0] if isinstance(simple_out, list) and simple_out else "").strip()
                simple_norm = _normalize_gen_out(simple_text)
                
                # Only return if we get a substantive answer (not evidence regurgitation)
                if simple_norm and len(simple_norm) > 50 and not simple_norm.lower().startswith(("insufficient", "no evidence", "from the evidence")):
                    return simple_norm
            except Exception:
                pass
            
            # Final fallback: clear failure message instead of evidence dump
            return "Unable to generate a complete medical answer from the available information."
        except Exception:
            # Last resort: try a minimal text-only prompt without spans/context
            try:
                minimal = f"Question: {q}\nAnswer:"
                inputs3 = self.processor(text=minimal, return_tensors="pt")
                inputs3 = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in inputs3.items()}
                with torch.inference_mode():
                    gen3 = self.model.generate(**inputs3, max_new_tokens=int(max_output_tokens or 1024), 
                                             do_sample=True, temperature=0.7, top_p=0.92, repetition_penalty=1.1)
                if self.tokenizer is not None:
                    out3 = self.tokenizer.batch_decode(gen3, skip_special_tokens=True)
                else:
                    decode = getattr(self.processor, "batch_decode", None)
                    out3 = decode(gen3, skip_special_tokens=True) if callable(decode) else None
                txt3 = (out3[0] if isinstance(out3, list) and out3 else "").strip()
                norm3 = _normalize_gen_out(txt3)
                if norm3:
                    return norm3
                # Don't return evidence as answer - signal clear failure
                return "Model generation failed - unable to produce a medical answer."
            except Exception:
                # Don't return evidence snippets as answers
                # OCR should not be used as a medical answer fallback
                pass
                return "Medical answer generation failed due to processing errors."
        finally:
            _close_all()

class TextCrossReranker:
    """
    Cross-encoder reranker using MedCPT-Cross-Encoder.
    .score(query, docs) -> list of logits (higher = more relevant).
    """
    def __init__(self, model_id: str, doc_model_id: Optional[str] = None):
        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)
        dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else \
                (torch.float16 if torch.cuda.is_available() else torch.float32)
        self.tok = AutoTokenizer.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            device_map="auto", torch_dtype=dtype
        ).eval()

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], max_length: int = 512) -> List[float]:
        if not docs:
            return []
        pairs = [[query, (d or "")] for d in docs]
        enc = self.tok(
            pairs, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
        )
        enc = {k: (v.to(self.model.device) if hasattr(v, "to") else v) for k, v in enc.items()}
        logits = self.model(**enc).logits
        # Handle shapes: [N,1] or [N,2]
        if logits.ndim == 2 and logits.size(-1) == 1:
            scores = logits.squeeze(-1)
        elif logits.ndim == 2 and logits.size(-1) == 2:
            scores = logits[:, 1]  # positive class
        else:
            scores = logits.mean(dim=-1)
        return scores.detach().float().cpu().tolist()

# ------------------------------
# Reranker helpers
# ------------------------------
def z_score_normalize(scores: List[float]) -> List[float]:
    """Z-score normalization with stability checks."""
    if not scores:
        return []
    
    arr = np.array(scores, dtype=np.float32)
    mean = np.mean(arr)
    std = np.std(arr)
    
    # Handle degenerate case (all scores identical)
    if std < 1e-6:
        return [0.0] * len(scores)
    
    z_scores = (arr - mean) / (std + 1e-6)
    return z_scores.tolist()

def rerank_with_case_level_cross_encoder(
    question: str,
    hits: List[Dict[str, Any]],
    reranker: Optional[TextCrossReranker] = None,
    top_cases: int = 5,
    top_pages_per_case: int = 3
) -> List[Dict[str, Any]]:
    """
    Case-level reranking using cross-encoder on {query, case_id} pairs.
    
    1. Group hits by case_id (doc_id)
    2. Create case-level documents by concatenating all pages for each case
    3. Use cross-encoder to rank {query, case_content} pairs
    4. Return top_cases with top_pages_per_case each
    5. Apply question-specific filtering to avoid duplicate answers
    """
    if not hits:
        return []
    if not reranker:
        return hits[:top_cases * top_pages_per_case]
    
    # Group hits by case_id (doc_id)
    case_groups = {}
    for hit in hits:
        case_id = hit.get("doc_id", "unknown")
        if case_id not in case_groups:
            case_groups[case_id] = []
        case_groups[case_id].append(hit)
    
    # Detect question type for smart page selection
    question_lower = question.lower()
    clinical_keywords = ["lesion", "clinical", "course", "onset", "features", "history", "size", "location", "referral", "evolution", "appearance"]
    diagnostic_keywords = ["diagnosis", "identify", "stain", "microscopy", "species", "PCR", "culture", "molecular", "sequencing", "identification", "specimens", "methods"]
    treatment_keywords = ["treatment", "therapy", "dose", "dosage", "regimen", "outcome", "follow-up", "response"]
    histopath_keywords = ["histopathology", "histopathologic", "findings", "biopsy", "microscopy", "amastigotes", "inflammatory"]
    
    is_clinical = any(kw in question_lower for kw in clinical_keywords)
    is_diagnostic = any(kw in question_lower for kw in diagnostic_keywords)
    is_treatment = any(kw in question_lower for kw in treatment_keywords)
    is_histopath = any(kw in question_lower for kw in histopath_keywords)
    
    # Create case-level documents and score them
    case_scores = []
    case_documents = []
    
    for case_id, case_hits in case_groups.items():
        # Sort pages by page_index for proper ordering
        case_hits_sorted = sorted(case_hits, key=lambda h: h.get("page_index", 999))
        
        # Filter and prioritize content based on question type
        relevant_hits = []
        for hit in case_hits_sorted:
            excerpt = (hit.get("text_excerpt") or "").strip()
            if not excerpt:
                continue
                
            excerpt_lower = excerpt.lower()
            relevance_score = 0
            
            # Score based on question type
            if is_clinical:
                clinical_terms = ["patient", "lesion", "nodule", "ulcer", "rash", "size", "location", "onset", "course", "month", "year", "clinical", "presentation"]
                relevance_score += sum(2 for term in clinical_terms if term in excerpt_lower)
                # Prefer earlier pages for clinical info
                relevance_score += max(0, 3 - hit.get("page_index", 0))
            
            if is_diagnostic:
                diagnostic_terms = ["pcr", "culture", "sequencing", "blast", "genbank", "dna", "specimens", "biopsy", "saliva", "blood", "identification", "species"]
                relevance_score += sum(3 for term in diagnostic_terms if term in excerpt_lower)
            
            if is_treatment:
                treatment_terms = ["treatment", "therapy", "dose", "mg", "kg", "amphotericin", "antimony", "antimoniate", "outcome", "response", "follow", "weeks"]
                relevance_score += sum(3 for term in treatment_terms if term in excerpt_lower)
                # Prefer later pages for treatment info
                relevance_score += max(0, hit.get("page_index", 0) - 1)
            
            if is_histopath:
                histopath_terms = ["histopathology", "biopsy", "inflammatory", "amastigotes", "macrophages", "lymphocytes", "granuloma", "infiltrate"]
                relevance_score += sum(3 for term in histopath_terms if term in excerpt_lower)
            
            hit_with_score = hit.copy()
            hit_with_score["question_relevance"] = relevance_score
            relevant_hits.append(hit_with_score)
        
        # Sort by question relevance, then by original score
        relevant_hits.sort(key=lambda h: (h.get("question_relevance", 0), h.get("score", 0)), reverse=True)
        
        # Create focused case document using top relevant content
        case_content_parts = []
        case_title = relevant_hits[0].get("case_title", case_id) if relevant_hits else case_id
        
        # Add case title/metadata first
        case_content_parts.append(f"Case: {case_title}")
        
        # Add question-specific focused content
        content_added = 0
        for hit in relevant_hits:
            if content_added >= 5:  # Limit to most relevant content
                break
            excerpt = (hit.get("text_excerpt") or "").strip()
            if excerpt and hit.get("question_relevance", 0) > 0:
                page_idx = hit.get("page_index", 0)
                case_content_parts.append(f"Page {page_idx + 1}: {excerpt}")
                content_added += 1
        
        # If no specifically relevant content, add some general content
        if content_added == 0:
            for hit in relevant_hits[:3]:
                excerpt = (hit.get("text_excerpt") or "").strip()
                if excerpt:
                    page_idx = hit.get("page_index", 0)
                    case_content_parts.append(f"Page {page_idx + 1}: {excerpt}")
        
        # Combine into case document (limit length to avoid truncation issues)
        case_document = " ".join(case_content_parts)[:4000]  # Limit to reasonable length
        case_documents.append(case_document)
        
        # Keep reference to case data
        case_scores.append({
            "case_id": case_id,
            "case_document": case_document,
            "hits": relevant_hits  # Already sorted by relevance
        })
    
    # Use cross-encoder to score {query, case_document} pairs
    if case_documents:
        relevance_scores = reranker.score(question, case_documents)
        
        # Pair scores with case data
        for i, score in enumerate(relevance_scores):
            case_scores[i]["relevance_score"] = float(score)
    
    # Sort cases by relevance score (descending)
    case_scores.sort(key=lambda x: x.get("relevance_score", 0.0), reverse=True)
    
    # Select top cases and return their top pages
    final_hits = []
    rank = 1
    
    for case_data in case_scores[:top_cases]:
        case_hits = case_data["hits"]
        
        # For each case, take the top pages based on question relevance
        case_hits_to_include = case_hits[:top_pages_per_case]
        
        # Add case-level score to each hit and update rank
        for hit in case_hits_to_include:
            hit_copy = hit.copy()
            hit_copy["case_relevance_score"] = case_data.get("relevance_score", 0.0)
            hit_copy["rank"] = rank
            final_hits.append(hit_copy)
            rank += 1
    
    # Safety fallback: if no final hits, return original hits
    if not final_hits:
        print(f"[WARN] Case-level reranking produced no final hits, returning original hits")
        return hits[:top_cases * top_pages_per_case]
    
    return final_hits

def rerank_with_text(
    question: str,
    hits: List[Dict[str, Any]],
    reranker: Optional[TextCrossReranker] = None,
    alpha: float = CFG.RERANK_ALPHA,
    min_excerpt_chars: int = CFG.RERANK_MIN_EXCERPT_CHARS,
    fallback_alpha: float = CFG.RERANK_FALLBACK_ALPHA
) -> List[Dict[str, Any]]:
    """
    Enhanced reranking with medical relevance scoring
    """
    if not hits:
        return []
    if not reranker:
        return hits
    
    # Detect medical question type for adaptive alpha
    medical_diagnostic = any(term in question.lower() for term in [
        "diagnosis", "histopathology", "microscopy", "stain", "culture",
        "pcr", "identification", "confirm", "differential"
    ])
    
    medical_clinical = any(term in question.lower() for term in [
        "clinical", "presentation", "symptoms", "signs", "course",
        "progression", "treatment", "therapy", "outcome"
    ])
    
    # Adjust alpha based on question type
    if medical_diagnostic:
        alpha = min(0.75, alpha + 0.15)  # Boost text reranker for diagnostic Qs
    elif medical_clinical:
        alpha = min(0.7, alpha + 0.1)   # Moderate boost for clinical Qs
    
    # Extract and validate excerpts
    excerpts = []
    valid_mask = []
    medical_relevance = []
    
    for h in hits:
        excerpt = (h.get("text_excerpt") or "").strip()
        excerpts.append(excerpt)
        
        # Check both length and medical content
        has_medical = any(term in excerpt.lower() for term in [
            "leishmania", "amastigote", "histopathology", "diagnosis",
            "treatment", "clinical", "patient", "lesion", "biopsy"
        ])
        
        is_valid = len(excerpt) >= min_excerpt_chars
        valid_mask.append(is_valid)
        medical_relevance.append(has_medical)
    
    # Prioritize medical-relevant excerpts for scoring
    valid_excerpts = []
    for i, (ex, valid, medical) in enumerate(zip(excerpts, valid_mask, medical_relevance)):
        if valid or (medical and len(ex) >= 20):  # Lower threshold for medical content
            valid_excerpts.append(ex)
    
    # Get cross-encoder scores
    rerank_scores = [None] * len(hits)
    if valid_excerpts:
        try:
            scores = reranker.score(question, valid_excerpts)
            score_iter = iter(scores)
            for i, (valid, medical) in enumerate(zip(valid_mask, medical_relevance)):
                if valid or (medical and len(excerpts[i]) >= 20):
                    rerank_scores[i] = next(score_iter)
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
            return hits
    
    # Extract original scores
    sim_scores = [float(h.get("score", 0.0)) for h in hits]
    
    # Medical relevance boost
    for i, h in enumerate(hits):
        if medical_relevance[i] and rerank_scores[i] is not None:
            rerank_scores[i] *= 1.15  # Boost medical-relevant content
    
    # Z-score normalize
    z_sim = z_score_normalize(sim_scores)
    z_rerank = z_score_normalize([s if s is not None else 0.0 for s in rerank_scores])
    
    # Late fusion with adaptive weights
    fused_results = []
    for i, h in enumerate(hits):
        # Dynamic alpha based on excerpt quality
        if medical_relevance[i] and valid_mask[i]:
            effective_alpha = alpha
        elif valid_mask[i]:
            effective_alpha = alpha * 0.9
        else:
            effective_alpha = fallback_alpha
        
        # Compute fused score
        fused_score = (1.0 - effective_alpha) * z_sim[i]
        if rerank_scores[i] is not None:
            fused_score += effective_alpha * z_rerank[i]
        
        # Page position boost for clinical questions
        if medical_clinical and h.get("page_index", 999) <= 2:
            fused_score *= 1.1
        
        # Create enriched hit
        enriched_hit = dict(h)
        enriched_hit.update({
            "fused_score": fused_score,
            "original_rank": i + 1,
            "rerank_logit": rerank_scores[i],
            "z_sim": z_sim[i],
            "z_rerank": z_rerank[i] if rerank_scores[i] is not None else None,
            "text_valid": valid_mask[i],
            "medical_relevant": medical_relevance[i]
        })
        fused_results.append((fused_score, enriched_hit))
    
    # Sort by fused score
    fused_results.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in fused_results]

def rebuild_spans_from_full_pages(hits, per_doc=6, max_chars=500, question=""):
    from collections import defaultdict
    out = []
    by_doc = defaultdict(list)
    for h in hits[:8]:
        by_doc[h.get("doc_id")].append(h)
    for doc_id, hs in by_doc.items():
        case_dir = find_case_dir(doc_id, CFG.EXTRACT_ROOT)
        pdf = find_case_pdf(case_dir) if case_dir else None
        for h in hs[:3]:
            idx0 = int(h.get("page_index", 0))
            full = read_pdf_page_text(pdf, idx0) if pdf else ""
            if not full:
                # light OCR fallback just for this page
                full = ocr_png_fallback(case_dir, idx0) if case_dir else ""
            if not full:
                continue
            # reuse your sentence/keep_terms logic
            sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", full) if len(s.strip()) >= 25]
            picked = []
            for s in sents:
                if len(picked) >= per_doc: break
                if KEEP_TERMS_REGEX.search(s):
                    picked.append(s)
            if len(picked) < per_doc:
                extra = sorted([s for s in sents if s not in picked], key=len, reverse=True)
                picked.extend(extra[:per_doc-len(picked)])
            cite = f"{doc_id}:p{idx0+1}"
            for s in picked[:per_doc]:
                out.append((safe_trim(s, max_chars), cite))
            if len(out) >= 20:
                return out[:20]
    return out[:20]

def extractive_spans(hits: List[Dict[str, Any]], per_doc: int = 6, max_chars: int = 500, question: str = "") -> List[Tuple[str,str]]:
    """
    Enhanced evidence extraction with question-aware span selection.
    """
    # Question-specific keywords for better span selection
    question_lower = question.lower()
    imaging_terms = {"neuroimaging","mri","ct","computed tomography","magnetic resonance",
                     "flair","dwi","adc","t1","t2","contrast","enhancement","brain","cns","encephal","lesion"}
    diagnostic_terms = {"diagnosis", "histopathology", "microscopy", "stain", "amastigotes", 
                       "promastigotes", "pcr", "culture", "species", "identify"}
    clinical_terms = {"clinical", "presentation", "lesion", "treatment", "therapy", 
                     "outcome", "size", "location", "course", "response"}
    treatment_terms = {"treatment", "dose", "regimen", "therapy", "antimony", 
                       "amphotericin", "miltefosine", "outcome", "follow"}
    
    is_imaging_q = any(term in question_lower for term in imaging_terms)
    is_diagnostic_q = any(term in question_lower for term in diagnostic_terms)
    is_clinical_q = any(term in question_lower for term in clinical_terms) 
    is_treatment_q = any(term in question_lower for term in treatment_terms)
    table_like = re.compile(r"\b(table\s*\d+|laborator(?:y|ial)\s+findings?)\b", re.I)
    
    out: List[Tuple[str, str]] = []
    seen_texts: set = set()  # Prevent duplicate spans
    
    for h in hits:
        te = (h.get("text_excerpt") or "").strip()
        if not te or len(te) < 50:
            continue
        
        # Check for duplicates early
        te_key = te.lower().replace(" ", "")[:100]
        if te_key in seen_texts:
            continue
        seen_texts.add(te_key)
        
        # Split into sentences
        import re as _re
        sents = _re.split(r'(?<=[.!?])\s+', te)
        span_candidates = []
        
        for sent in sents:
            if len(sent.strip()) < 30:
                continue
                
            # Question-specific scoring
            score = 0
            sent_lower = sent.lower()
            
            # Base clinical relevance
            clinical_score = len(_re.findall(KEEP_TERMS_REGEX, sent))
            score += clinical_score * 2
            
            # Question-specific bonuses
            if is_imaging_q:
                if any(t in sent_lower for t in imaging_terms):
                    score += 5
                # Penalty for table/lab findings when asking about imaging
                if "table" in sent_lower or "laboratorial findings" in sent_lower:
                    score -= 3
            elif is_diagnostic_q:
                if any(term in sent_lower for term in diagnostic_terms):
                    score += 5
            elif is_clinical_q:
                if any(term in sent_lower for term in clinical_terms):
                    score += 5
            elif is_treatment_q:
                if any(term in sent_lower for term in treatment_terms):
                    score += 5
                    
            # Penalize very generic sentences
            if any(generic in sent_lower for generic in [
                "leishmaniasis is", "cutaneous leishmaniasis (cl) is", 
                "this diagnosis", "is a parasitic disease"
            ]):
                score -= 3
            
            # NEW: keep tables out unless question is treatment/lab related
            if table_like.search(sent_lower) and not (is_treatment_q or "lab" in question_lower or "laborator" in question_lower):
                score -= 6  # strong penalty so it rarely survives

            if score >= 2:  # Only include sentences with good relevance
                span_candidates.append((score, sent.strip()))
                
        # Sort by score and take best spans
        span_candidates.sort(reverse=True, key=lambda x: x[0])
        for score, sent in span_candidates[:per_doc]:
            trimmed = safe_trim(sent, max_chars)
            if trimmed and trimmed not in [s[0] for s in out]:  # Avoid exact duplicates
                doc_id = h.get("doc_id", "unknown")
                pg = h.get("page_index", 0)
                cite = f"{doc_id}:p{pg+1}"
                out.append((trimmed, cite))
                if len(out) >= 20:  # Global limit
                    break
            
        if len(out) >= 20:
            break
    
    return out[:20]

# ------------------------------
# Qdrant helpers
# ------------------------------
def qdrant() -> QdrantClient:
    return QdrantClient(
        url=CFG.QDRANT_URL,
        api_key=CFG.QDRANT_API_KEY,
        timeout=120.0,
        prefer_grpc=False,
    )

def recreate_collection(client: QdrantClient, dim: int):
    vectors = {
        "image": VectorParams(size=dim, distance=Distance.COSINE),
        "text":  VectorParams(size=dim, distance=Distance.COSINE),
    }
    kwargs = dict(
        collection_name=CFG.COLLECTION,
        vectors_config=vectors,
        on_disk_payload=True,
        optimizers_config=OptimizersConfigDiff(default_segment_number=2),
        quantization_config=ScalarQuantization(
            scalar=ScalarQuantizationConfig(type=ScalarType.INT8, always_ram=False)
        ),
    )
    try:
        client.recreate_collection(**kwargs)
    except TypeError:
        kwargs.pop("on_disk_payload", None)
        kwargs.pop("quantization_config", None)
        client.recreate_collection(**kwargs)

def ensure_collection_exists(client: QdrantClient, dim: int):
    """Create the collection if it does not exist; no destructive action if it exists."""
    try:
        infos = client.get_collections().collections
        exists = any(getattr(ci, "name", None) == CFG.COLLECTION for ci in infos)
    except Exception:
        try:
            client.get_collection(CFG.COLLECTION)
            exists = True
        except Exception:
            exists = False
    if not exists:
        recreate_collection(client, dim)

def create_payload_indexes(client: QdrantClient):
    for field, schema in [
        ("doc_id",          qm.PayloadSchemaType.KEYWORD),   # ← ADD THIS LINE
        ("keywords",        qm.PayloadSchemaType.KEYWORD),
        ("entities",        qm.PayloadSchemaType.KEYWORD),
        ("case_type",       qm.PayloadSchemaType.KEYWORD),
        ("page_kind",       qm.PayloadSchemaType.KEYWORD),
        ("micrograph_like", qm.PayloadSchemaType.BOOL),
        ("page_index",      qm.PayloadSchemaType.INTEGER),
        ("file_mtime",      qm.PayloadSchemaType.INTEGER),
        ("ahash",           qm.PayloadSchemaType.KEYWORD),
    ]:
        try:
            client.create_payload_index(
                collection_name=CFG.COLLECTION,
                field_name=field,
                field_schema=schema
            )
            logging.info(f"Created payload index: {field} ({schema})")
        except Exception as e:
            logging.info(f"Index {field}: {e}")

# ------------------------------
# Indexing (pages already rendered)
# ------------------------------
def sha24(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()[:24]

def discover_cases(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "pages").is_dir()]) if root.is_dir() else []

def list_pages_full(case_dir: Path) -> List[Path]:
    return sorted((case_dir / "pages").glob("page_*.png"))

def build_uid(doc_id: str, page_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}__p{page_idx:04d}"))

def _tokens(s: str) -> set:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    toks = [t for t in s.split() if len(t) >= 3]
    return set(toks)

def _pdf_semantic_match_ok(pdf_path: Path, case_dir: Path, min_overlap: int = 2) -> bool:
    """Heuristic: the first couple of pages must share ≥min_overlap tokens with case_dir name."""
    try:
        name_tokens = _tokens(case_dir.name.replace("_", " "))
        if not name_tokens:
            return True  # nothing to check
        text_blob = ""
        if _HAVE_PYMUPDF:
            with fitz.open(pdf_path) as doc:
                for i in range(min(3, len(doc))):  # read first up to 3 pages
                    text_blob += " " + (doc.load_page(i).get_text("text") or "")
        if not text_blob:
            # Try pdfminer fallback (first page)
            from pdfminer.high_level import extract_text
            text_blob = extract_text(str(pdf_path), page_numbers=[0]) or ""
        page_tokens = _tokens(text_blob)
        overlap = len(name_tokens & page_tokens)
        return overlap >= min_overlap
    except Exception:
        # Be conservative: if we cannot verify, do not accept
        return False

def find_case_pdf(case_dir: Path) -> Optional[Path]:
    """Resolve the correct PDF for this case directory, conservatively."""
    # 0) page_map hint (preferred)
    pm = load_page_map(case_dir)
    if pm:
        for key in ("source_pdf", "pdf", "pdf_basename"):
            v = pm.get(key)
            if v:
                cand = Path(v)
                if not cand.is_absolute():
                    # try relative to likely roots
                    for root in (case_dir, case_dir.parent, *CFG.PDF_SEARCH_DIRS):
                        p = (root / v).resolve()
                        if p.exists():
                            cand = p; break
                if cand.exists() and _pdf_semantic_match_ok(cand, case_dir):
                    return cand

    # 1) inside the case_dir
    locals_ = list(case_dir.glob("*.pdf")) + list(case_dir.glob("*.PDF"))
    for p in sorted(locals_, key=lambda x: -x.stat().st_size):
        if _pdf_semantic_match_ok(p, case_dir):
            return p

    # 2) exact basename in configured search roots only
    stem = case_dir.name
    for root in CFG.PDF_SEARCH_DIRS:
        cand = (root / f"{stem}.pdf")
        if cand.exists() and _pdf_semantic_match_ok(cand, case_dir):
            return cand.resolve()
        # accept ONE safe near-exact: whitespace/underscore variants only
        near = []
        for p in root.glob("*.pdf"):
            name = p.stem.lower().replace(" ", "_")
            tgt  = stem.lower().replace(" ", "_")
            if name == tgt and _pdf_semantic_match_ok(p, case_dir):
                near.append(p)
        if near:
            # if multiple “equal”, pick the newest
            return max(near, key=lambda x: x.stat().st_mtime).resolve()

    # 3) refuse wildcard/first-match guessing — safer to return None
    return None

def read_pdf_page_text(pdf_path: Path, page_index: int) -> str:
    """
    Enhanced page-text extractor with better error handling and caching
    """
    if not pdf_path or not pdf_path.exists():
        return ""
    
    # Try cache first (add simple in-memory cache)
    cache_key = f"{pdf_path}:{page_index}"
    if not hasattr(read_pdf_page_text, "_cache"):
        read_pdf_page_text._cache = {}
    
    if cache_key in read_pdf_page_text._cache:
        return read_pdf_page_text._cache[cache_key]

    text = ""
    
    # --- 1) PyMuPDF with better extraction ---
    if _HAVE_PYMUPDF:
        try:
            import contextlib, io
            with contextlib.redirect_stderr(io.StringIO()):
                with fitz.open(pdf_path) as doc:
                    if 0 <= page_index < len(doc):
                        page = doc.load_page(page_index)
                        # Use multiple extraction methods
                        txt = page.get_text("text") or ""
                        if len(txt) < 50:  # If text is too short, try blocks
                            blocks = page.get_text("blocks")
                            txt = " ".join([b[4] for b in blocks if len(b) > 4])
                        txt = txt.strip()
                        if txt:
                            text = txt
                    
                    # Smart neighbor fallback - check both directions
                    if not text:
                        for delta in [-1, 1, -2, 2]:
                            idx = page_index + delta
                            if 0 <= idx < len(doc):
                                t = doc.load_page(idx).get_text("text") or ""
                                if len(t) > 100:  # Only use if substantial
                                    text = t
                                    break
        except Exception:
            pass

    # --- 2) pdfminer.six with better handling ---
    if not text:
        try:
            from pdfminer.high_level import extract_text
            from pdfminer.layout import LAParams
            # Use layout analysis for better extraction
            laparams = LAParams(detect_vertical=True, all_texts=True)
            txt = extract_text(
                str(pdf_path), 
                page_numbers=[page_index], 
                laparams=laparams,
                maxpages=1
            ) or ""
            text = txt.strip()
        except Exception:
            pass

    # --- 3) MANDATORY OCR fallback when text is insufficient ---
    if not text or len(text) < 50:
        try:
            case_dir = pdf_path.parent
            ocr_text = ocr_png_fallback(case_dir, page_index)
            if ocr_text and len(ocr_text) > len(text):
                text = ocr_text
        except Exception:
            pass
    
    # Cache the result
    read_pdf_page_text._cache[cache_key] = text
    
    # Keep cache size reasonable
    if len(read_pdf_page_text._cache) > 100:
        # Remove oldest entries
        keys = list(read_pdf_page_text._cache.keys())
        for k in keys[:20]:
            del read_pdf_page_text._cache[k]

    # Drop common publishing boilerplate that pollutes evidence
    if text:
        text = re.sub(r"\bISSN\b.*?(?:\n|$)", "", text, flags=re.I)
        text = re.sub(r"\bDOI\b.*?(?:\n|$)", "", text, flags=re.I)
        text = re.sub(r"\b(Received|Accepted|Published)\b.*?(?:\n|$)", "", text, flags=re.I)
        text = re.sub(r"\bCreative Commons\b.*?(?:\n|$)", "", text, flags=re.I)

    return text

def qdrant_init():
    logging.info("Bootstrapping ColQwen2 to infer embedding dimension…")
    retr = CQ2(CFG.RET_MODEL_ID)
    dim = int(retr.embed_images([Image.new("RGB", (64, 64))]).shape[-1])
    logging.info(f"Pooled embedding dim = {dim}")
    client = qdrant()
    recreate_collection(client, dim)
    logging.info(f"Recreated collection '{CFG.COLLECTION}' on {CFG.QDRANT_URL}")

def build_payload(case_dir: Path, page_path: Path, page_idx: int, img: Image.Image) -> Dict[str, Any]:
    title = case_dir.name.replace("_", " ")
    pdf = find_case_pdf(case_dir)
    text = read_pdf_page_text(pdf, page_idx) if pdf else ""
    
    # Fallback to OCR if no PDF or insufficient text
    if not text or len(text) < 30:
        try:
            ocr_text = ocr_png_fallback(case_dir, page_idx)
            if ocr_text and len(ocr_text) > len(text or ""):
                text = ocr_text
        except Exception:
            pass
    
    # If still no text, use title as fallback
    if not text:
        text = f"Case: {title} (page {page_idx + 1})"
    
    # Sentence-safe truncation (prevents dangling "…" and mid-sentence cuts)
    text_excerpt = safe_trim(text, CFG.MAX_TEXT_EXCERPT)

    kw = extract_keywords(title, text)
    ctype = guess_case_type(title, text)
    ents = extract_entities(title, text)

    metrics = image_metrics(img)
    finfo = file_info(page_path)

    return {
        "uid": build_uid(case_dir.name, page_idx),
        "doc_id": case_dir.name,
        "case_title": title,
        "page_index": page_idx,
        "image_path": str(page_path),
        "language": CFG.LANGUAGE,

        # provenance / semantics
        "source_pdf": (str(pdf) if pdf else ""),
        "text_excerpt": text_excerpt if text_excerpt else "",
        "keywords": kw,
        "case_type": ctype,
        "entities": ents,

        # image stats
        "width": metrics["width"],
        "height": metrics["height"],
        "sat_mean": metrics["sat_mean"],
        "edge_density": metrics["edge_density"],
        "entropy": metrics["entropy"],
        "page_kind": metrics["page_kind"],
        "micrograph_like": metrics["micrograph_like"],
        "ahash": metrics["ahash"],

        # file info
        "file_bytes": finfo["bytes"],
        "file_mtime": finfo["mtime"],
    }

def _page_text_for_embedding(case_dir: Path, pdf: Optional[Path], page_idx: int, title_fallback: str) -> str:
    """
    Get best-effort text to embed for a page:
      - Try PDF text (read_pdf_page_text)
      - Fallback to OCR of the rendered PNG
      - Finally fallback to document title so the text vector isn't empty
    """
    txt = ""
    if pdf and pdf.exists():
        try:
            txt = read_pdf_page_text(pdf, page_idx) or ""
        except Exception:
            txt = ""
    
    # Always try OCR fallback if no substantial text from PDF
    if not txt or len(txt.strip()) < 50:
        try:
            ocr_text = ocr_png_fallback(case_dir, page_idx) or ""
            if ocr_text and len(ocr_text.strip()) > len(txt.strip()):
                txt = ocr_text
        except Exception:
            pass
    
    txt = txt.strip()
    if not txt:
        txt = title_fallback
    # keep it reasonable in length for embedding
    return txt[:4000]

def qdrant_index():
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)

    # Both ColQwen2 text & image live in the same space/dim
    dim = int(retr.embed_images([Image.new("RGB", (64, 64))]).shape[-1])
    ensure_collection_exists(client, dim)
    logging.info("Using Qdrant at %s, collection '%s'.", CFG.QDRANT_URL, CFG.COLLECTION)

    cases = discover_cases(CFG.EXTRACT_ROOT)
    logging.info("Found %d case(s).", len(cases))

    for c in cases:
        pages = list_pages_full(c)
        if not pages:
            continue
        title = c.name.replace("_", " ")
        logging.info("Indexing %s (%d pages)…", c.name, len(pages))
        
        # PRE-SCAN: Collect all page texts to determine document-level case_type
        pdf = find_case_pdf(c)
        all_texts = []
        for pth in pages:
            m = re.search(r"page_(\d+)\.png$", pth.name)
            num = int(m.group(1)) if m else 1
            page_idx = max(0, num - 1)
            t = read_pdf_page_text(pdf, page_idx) if pdf else ""
            if not t:
                # Don't pay the OCR cost here (it’s expensive); case_type needs only rough signal
                t = ""
            all_texts.append(t)
        
        # Determine case type once for entire document
        doc_case_type = guess_case_type_document(title, all_texts)
        
        # Now index with consistent case_type
        for i in range(0, len(pages), CFG.BATCH_EMBED):
            chunk = pages[i:i+CFG.BATCH_EMBED]
            ims = [Image.open(pth).convert("RGB") for pth in chunk]
            try:
                # ---- image vectors (as before)
                img_vecs = retr.embed_images(ims)

                # ---- text to embed (batch)
                page_idx_batch = []
                text_batch = []
                for pth in chunk:
                    m = re.search(r"page_(\d+)\.png$", pth.name)
                    num = int(m.group(1)) if m else 1
                    page_idx = max(0, num - 1)
                    page_idx_batch.append(page_idx)
                    text_batch.append(_page_text_for_embedding(c, pdf, page_idx, title))

                txt_vecs = retr.embed_texts(text_batch)

                # ---- upsert points with BOTH vectors
                points: List[PointStruct] = []
                for j, pth in enumerate(chunk):
                    page_idx = page_idx_batch[j]
                    payload = build_payload(c, pth, page_idx, ims[j])  # builds text_excerpt, keywords, etc.
                    payload["case_type"] = doc_case_type  # enforce doc-level label
                    points.append(
                        PointStruct(
                            id=payload["uid"],
                            payload=payload,
                            vector={"image": img_vecs[j], "text": txt_vecs[j]}
                        )
                    )
                client.upsert(collection_name=CFG.COLLECTION, points=points, wait=True)
            finally:
                for im in ims:
                    try: im.close()
                    except Exception: pass

        logging.info("Done: %s", c.name)
    logging.info("✅ Indexing complete.")

# ------------------------------
# Query + (optional) generate
# ------------------------------
def _qdrant_search(client: QdrantClient,
                   qv: np.ndarray,
                   top_k: int,
                   filt: Optional[Filter],
                   score_th: Optional[float],
                   using: str = "image") -> List:
    """
    Search a specific named vector ('image' or 'text').
    For query_points API: Qdrant returns distance; we convert to similarity (1 - d).
    For legacy search(): returns 'score' (usually similarity) already.
    """
    assert using in ("image", "text"), f"unknown vector name: {using}"
    vec_list = qv.tolist()
    sim_th = None if score_th is None else max(0.0, min(1.0, float(score_th)))

    # Try new API first
    try:
        from qdrant_client.http.models import NearVector
        qobj = NearVector(vector=vec_list, using=using)
        kwargs = dict(
            collection_name=CFG.COLLECTION,
            query=qobj,
            limit=top_k,
            filter=filt,
            with_payload=True,
        )
        if sim_th is not None:
            kwargs["score_threshold"] = 1.0 - sim_th  # distance cutoff
        res = client.query_points(**kwargs)
        points = getattr(res, "points", []) or []
        for p in points:
            # Convert distance -> similarity
            try:
                d = float(getattr(p, "score", 0.0))
            except Exception:
                d = 0.0
            setattr(p, "score", 1.0 - d)
        return points
    except Exception:
        # Legacy fallback
        pts = client.search(
            collection_name=CFG.COLLECTION,
            query_vector=NamedVector(name=using, vector=vec_list),
            limit=top_k,
            query_filter=filt,
            with_payload=True,
            score_threshold=(sim_th if sim_th is not None else None),
        )
        return pts

def qdrant_ask_text(question: str, top_k: int = CFG.TOP_K,
                    case_type: Optional[str] = None,
                    keyword: Optional[str] = None,
                    any_keywords: Optional[str] = None,
                    micrograph_only: bool = False,
                    micrograph_strict: bool = False,
                    doc_id: Optional[str] = None,
                    use_reranker: bool = True) -> Dict[str, Any]:
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)
    qv_text = retr.embed_texts([question])[0]

    # Build filter (default to cutaneous if implied by question)
    if case_type is None and re.search(r"\bcutaneous\b", question, re.I):
        case_type = "cutaneous"
    musts = []
    if doc_id:
        musts.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    if case_type:
        musts.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))

    or_conditions = None
    if keyword:
        or_conditions = [FieldCondition(key="keywords", match=MatchAny(any=[keyword]))]
    if any_keywords:
        toks = [t.strip() for t in any_keywords.split(",") if t.strip()]
        if toks:
            extra = []
            for t in list(toks):
                low = t.lower()
                extra.append(low.replace("-", ""))
                extra.append(low.replace("-", " "))
            toks = sorted(set(list(toks) + extra))
            or_conditions = [
                FieldCondition(key="keywords", match=MatchAny(any=toks)),
                FieldCondition(key="entities", match=MatchAny(any=toks)),
            ]

    base_filter = Filter(must=musts, should=or_conditions) if (musts or or_conditions) else None

    pool_multiplier = 3
    pool_size = max(top_k * pool_multiplier, 24) if use_reranker else max(top_k * 2, 16)

    # --- Dual-modality recall: text-vector + cross-modal (text->image)
    hits_text = _qdrant_search(client, qv_text, pool_size, base_filter, CFG.SCORE_THRESHOLD, using="text")
    # For cross-modal, re-use the same qv_text against 'image' vectors (works well with ColQwen2)
    # Apply micrograph constraint here only if requested strictly; otherwise we'll prefer later.
    if micrograph_only and micrograph_strict:
        strict_filter = Filter(
            must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))],
            should=or_conditions
        )
        hits_img = _qdrant_search(client, qv_text, pool_size, strict_filter, CFG.SCORE_THRESHOLD, using="image")
        if not hits_img and or_conditions:
            strict_filter = Filter(must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))])
            hits_img = _qdrant_search(client, qv_text, pool_size, strict_filter, CFG.SCORE_THRESHOLD, using="image")
    else:
        hits_img = _qdrant_search(client, qv_text, pool_size, base_filter, CFG.SCORE_THRESHOLD, using="image")

    # --- Merge & dedupe (keep max similarity across modalities)
    merged = {}  # key -> (score, payload, modality)
    def _key(pt):
        pl = getattr(pt, "payload", {}) or {}
        # Prefer stable UID if present; else fallback (doc_id, page_index)
        uid = pl.get("uid")
        if uid:
            return ("uid", uid)
        return ("dpi", pl.get("doc_id"), pl.get("page_index"))
    def _add(points, tag):
        for pt in points or []:
            k = _key(pt)
            sc = float(getattr(pt, "score", 0.0))
            if (k not in merged) or (sc > merged[k][0]):
                merged[k] = (sc, getattr(pt, "payload", {}) or {}, tag)

    _add(hits_text, "text")
    _add(hits_img,  "image")

    # --- Convert to raw_items (dicts) for reranking
    raw_items = []
    for _, (score, pl, tag) in merged.items():
        raw_items.append({
            "rank": 0,  # will be set by reranker/ordering
            "score": float(score),
            "doc_id": pl.get("doc_id"),
            "page_index": pl.get("page_index"),
            "image_path": pl.get("image_path"),
            "case_type": pl.get("case_type"),
            "page_kind": pl.get("page_kind"),
            "micrograph_like": pl.get("micrograph_like"),
            "keywords": (pl.get("keywords") or [])[:6],
            "text_excerpt": pl.get("text_excerpt"),
            "via": tag,  # keep provenance of which modality recalled it
        })

    # Sort by score first (pre-rerank), then cut to a manageable pool
    raw_items.sort(key=lambda r: r["score"], reverse=True)
    raw_items = raw_items[:pool_size]

    # --- Optional micrograph preference (soft) BEFORE rerank
    if micrograph_only and not micrograph_strict and raw_items:
        prefer = [r for r in raw_items if r.get("micrograph_like") is True]
        others = [r for r in raw_items if not r.get("micrograph_like")]
        raw_items = (prefer + others)[:pool_size]
    
    # Store pre-rerank pool for diagnostics
    pre_rerank_pool = [dict(r) for r in raw_items[:min(pool_size, 30)]]  # Deep copy top 30

    # --- Apply reranking if enabled
    if use_reranker and raw_items:
        try:
            reranker = TextCrossReranker(CFG.RERANKER_MODEL_ID)
            raw_items = rerank_with_text(
                question, raw_items, reranker,
                alpha=CFG.RERANK_ALPHA,
                min_excerpt_chars=CFG.RERANK_MIN_EXCERPT_CHARS,
                fallback_alpha=CFG.RERANK_FALLBACK_ALPHA
            )
        except Exception as e:
            logging.warning(f"Reranking failed, falling back to original scores: {e}")

    # Final selection
    selected = raw_items[:top_k]
    # fix ranks
    for i, r in enumerate(selected, 1):
        r["rank"] = i

    return {"mode": "text", "question": question, "hits": selected, "pre_rerank_pool": pre_rerank_pool}

def qdrant_ask_image(image_path: str, top_k: int = CFG.TOP_K,
                     micrograph_only: bool = False, micrograph_strict: bool = False) -> Dict[str, Any]:
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)
    im = Image.open(image_path).convert("RGB")
    try:  # ← Add try block
        qv = retr.embed_images([im])[0]
    finally:  # ← Add finally block
        try: im.close()
        except Exception: pass

    base_filter = None
    if micrograph_only and micrograph_strict:
        base_filter = Filter(must=[FieldCondition(key="micrograph_like", match=MatchValue(value=True))])

    hits = _qdrant_search(client, qv, top_k * (3 if micrograph_only and not micrograph_strict else 1),
                          base_filter, CFG.SCORE_THRESHOLD)

    raw_items = []
    for i, h in enumerate(hits):
        pl = getattr(h, "payload", {}) or {}  # ← Same robust fix as above
        raw_items.append({
            "rank": i+1,
            "score": float(getattr(h, "score", 0.0)),
            "doc_id": pl.get("doc_id"),
            "page_index": pl.get("page_index"),
            "image_path": pl.get("image_path"),
            "case_type": pl.get("case_type"),
            "page_kind": pl.get("page_kind"),
            "micrograph_like": pl.get("micrograph_like"),
            "keywords": (pl.get("keywords") or [])[:6],
            "text_excerpt": pl.get("text_excerpt"),
        })

    if micrograph_only and not micrograph_strict and raw_items:
        prefer = [r for r in raw_items if r.get("micrograph_like") is True]
        others = [r for r in raw_items if not r.get("micrograph_like")]
        selected = (prefer + others)[:top_k] if prefer else raw_items[:top_k]
    else:
        selected = raw_items[:top_k]

    return {"mode": "image", "image": image_path, "hits": selected}

def answer_with_gemini(question: str, hits: List[Dict[str, Any]], take: int = 2) -> Dict[str, Any]:
    if not hits:
        raise SystemExit("No hits to answer from. Relax filters or lower SCORE_THRESHOLD.")
    
    # Reuse your span extractor (text evidence)
    spans = extractive_spans(hits, per_doc=2)
    
    # Images to feed Gemini
    imgs = [Path(h["image_path"]) for h in hits[:take]]
    
    # Init local MedGemma-4B-IT (offline)
    g = MedGemma4B(model_id=CFG.GEN_MODEL_ID)
    
    ans = g.answer(
        q=question,
        image_paths=imgs,
        spans=spans,
        max_output_tokens=CFG.MAX_NEW_TOKENS
    )
    
    return {"answer": ans, "used_images": [str(p) for p in imgs], "evidence": spans}

def _preprocess_for_ocr(pil: Image.Image) -> Image.Image:
    """Light, robust preproc for OCR: grayscale + contrast + binarize."""
    import numpy as np
    from PIL import ImageOps, ImageFilter

    g = pil.convert("L")                     # grayscale
    g = ImageOps.autocontrast(g)            # stretch
    g = g.filter(ImageFilter.MedianFilter(3))
    arr = np.array(g, dtype="uint8")
    # simple adaptive-ish threshold
    thr = max(100, min(175, int(arr.mean() + 0.5*arr.std())))
    arr_bin = (arr > thr).astype("uint8") * 255
    return Image.fromarray(arr_bin, mode="L")


def ocr_png_tesseract(case_dir: Path, page_index: int) -> str:
    """OCR a rendered PNG page with Tesseract (eng, LSTM-only, psm=6)."""
    try:
        import pytesseract
    except Exception:
        return ""
    pm = load_page_map(case_dir)
    paths = page_indices_to_paths(case_dir, [page_index])
    if not paths:
        return ""
    p = paths[0]
    try:
        with Image.open(p).convert("RGB") as im:
            pim = _preprocess_for_ocr(im)
            cfg = "--oem 1 --psm 6 -l eng"
            txt = pytesseract.image_to_string(pim, config=cfg) or ""
            return txt.strip()
    except Exception:
        return ""


def _merge_ocr_text(primary: str, fallback: str) -> str:
    """Prefer primary if reasonably long; else append fallback (dedup-ish)."""
    import re as _re
    A = (primary or "").strip()
    B = (fallback or "").strip()
    if len(A) >= 80:
        return A
    if not B:
        return A
    # If A is short, try to enrich it with unseen lines from B
    a_lines = {ln.strip() for ln in A.splitlines() if ln.strip()}
    b_lines = [ln.strip() for ln in B.splitlines() if ln.strip()]
    extra = [ln for ln in b_lines if ln not in a_lines]
    merged = (A + ("\n" if A and extra else "") + "\n".join(extra)).strip()
    # Light cleanup: collapse spaces
    merged = _re.sub(r"[ \t]+", " ", merged)
    return merged

def ocr_png_fallback(case_dir: Path, page_index: int) -> str:
    """
    Hybrid OCR:
      1) EasyOCR (fast, GPU if available)
      2) Tesseract (good on small captions / serif text)
      Merge results to maximize recall for captions, figure legends, and tiny clinical text.
    """
    easy = ""
    try:
        import easyocr
        pm = load_page_map(case_dir)
        paths = page_indices_to_paths(case_dir, [page_index])
        if paths and paths[0].exists():
            # Use CPU for stability if GPU fails
            for gpu_flag in [True, False]:
                try:
                    reader = easyocr.Reader(['en'], gpu=gpu_flag)
                    res = reader.readtext(str(paths[0]), detail=0, paragraph=True)
                    easy = "\n".join([s for s in res if s]).strip()
                    break  # Success, exit the retry loop
                except Exception as e:
                    if not gpu_flag:  # Last attempt failed
                        print(f"[WARN] EasyOCR failed on {paths[0]}: {e}")
                    continue
    except ImportError:
        print("[WARN] EasyOCR not available, falling back to Tesseract only")
    except Exception as e:
        print(f"[WARN] EasyOCR setup failed: {e}")

    tess = ""
    try:
        tess = ocr_png_tesseract(case_dir, page_index)
    except Exception as e:
        print(f"[WARN] Tesseract OCR failed: {e}")

    merged = _merge_ocr_text(easy, tess)
    
    # If OCR failed entirely, return a minimal description based on case title
    if not merged:
        title = case_dir.name.replace("_", " ")
        return f"Medical case: {title} (page {page_index + 1}) - OCR extraction not available"
    
    return merged

def backfill_text_excerpts(batch_size: int = 512):
    """Scan existing points and backfill missing text_excerpt fields."""
    client = qdrant()
    next_offset = None
    total_scanned = 0
    total_updated = 0

    while True:
        res = client.scroll(
            collection_name=CFG.COLLECTION,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=next_offset
        )

        # Normalize return type: tuple (points, next_offset) vs object
        if isinstance(res, tuple):
            points, next_offset = res
        else:
            points = getattr(res, "points", []) or []
            next_offset = getattr(res, "next_page_offset", None)

        if not points:
            break

        updates = {}
        for pt in points:
            total_scanned += 1
            pl = getattr(pt, "payload", {}) or {}
            cur = pl.get("text_excerpt")
            if isinstance(cur, str) and cur.strip():
                # already has non-empty text; skip
                continue

            doc_id = pl.get("doc_id")
            img_path = pl.get("image_path")
            page_index = pl.get("page_index")

            if not doc_id:
                continue

            # Prefer payload page_index; fall back to parsing filename
            if isinstance(page_index, int):
                pdf_idx0 = max(0, int(page_index))  # already 0-based in your payload
            else:
                m = re.search(r"page_(\d+)\.png", img_path or "")
                if not m:
                    continue
                pdf_idx0 = max(0, int(m.group(1)) - 1)

            case_dir = CFG.EXTRACT_ROOT / doc_id
            pdf_path = find_case_pdf(case_dir)
            if not pdf_path:
                continue

            text = read_pdf_page_text(pdf_path, pdf_idx0) or ""
            if not text:
                text = ocr_png_fallback(case_dir, pdf_idx0)
            excerpt = (text[:CFG.MAX_TEXT_EXCERPT] + "…") if len(text) > CFG.MAX_TEXT_EXCERPT else text
            updates[getattr(pt, "id")] = {"text_excerpt": excerpt}

        # Apply updates one-by-one (cloud-safe)
        if updates:
            for pid, payload in updates.items():
                client.set_payload(
                    collection_name=CFG.COLLECTION,
                    payload=payload,
                    points=[pid]
                )
            total_updated += len(updates)
            logging.info(f"Updated {len(updates)} points")

        if not next_offset:
            break

    logging.info(f"Backfill complete: scanned={total_scanned}, updated={total_updated}")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(
        description="Qdrant RAG for ColQwen2 + MedGemma-4B-IT (rich attrs v4.1)",
        allow_abbrev=False
    )
    # Global overrides (cloud-friendly)
    ap.add_argument("--qdrant_url", help="Override CFG.QDRANT_URL (e.g. https://<cluster>.<region>.cloud.qdrant.io)")
    ap.add_argument("--qdrant_api_key", help="Override CFG.QDRANT_API_KEY")
    ap.add_argument("--collection", help="Override CFG.COLLECTION")

    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("qdrant-init", help="(Re)create collection with correct dim")
    sub.add_parser("qdrant-index", help="Index all pages under EXTRACT_ROOT/*/pages")
    sub.add_parser("qdrant-create-indexes", help="Create payload indexes for faster filtered queries")

    ask = sub.add_parser("qdrant-ask", help="Retrieve then answer with MedGemma")
    ask.add_argument("-q", "--question", help="Text question")
    ask.add_argument("--image", help="Image path for image query")
    ask.add_argument("--topk", type=int, default=CFG.TOP_K)
    ask.add_argument("--case_type", choices=["cutaneous","mucocutaneous","visceral","unknown"])
    ask.add_argument("--keyword")
    ask.add_argument("--any_keywords", help="Comma-separated keywords to OR-match in payload (keywords/entities)")
    ask.add_argument("--micrograph_only", action="store_true", help="Prefer micrograph pages (soft).")
    ask.add_argument("--micrograph_strict", action="store_true", help="Require micrograph pages (hard filter).")
    ask.add_argument("--score_threshold", type=float, help="Override CFG.SCORE_THRESHOLD for this query (None => no threshold).")
    sub.add_parser("qdrant-backfill-text", help="Backfill missing text_excerpt from PDFs")

    args = ap.parse_args()

    # Apply CLI overrides
    if getattr(args, "qdrant_url", None):
        CFG.QDRANT_URL = args.qdrant_url
    if getattr(args, "qdrant_api_key", None):
        CFG.QDRANT_API_KEY = args.qdrant_api_key
    if getattr(args, "collection", None):
        CFG.COLLECTION = args.collection

    if args.cmd == "qdrant-init":
        qdrant_init()
    elif args.cmd == "qdrant-index":
        qdrant_index()
    elif args.cmd == "qdrant-create-indexes":
        create_payload_indexes(qdrant())
    elif args.cmd == "qdrant-backfill-text":
        backfill_text_excerpts()
    elif args.cmd == "qdrant-ask":
        if args.score_threshold is not None:
            CFG.SCORE_THRESHOLD = args.score_threshold
        if args.question:
            res = qdrant_ask_text(args.question, args.topk, args.case_type, args.keyword,
                                  args.any_keywords, args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
            if res["hits"]:
                out = answer_with_gemini(args.question, res["hits"])
                print(json.dumps(out, indent=2, ensure_ascii=False))
            else:
                logging.warning("No hits after filters/threshold. Try relaxing filters or lowering SCORE_THRESHOLD.")
        elif args.image:
            res = qdrant_ask_image(args.image, args.topk, args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
        else:
            raise SystemExit("Provide --q or --image")
