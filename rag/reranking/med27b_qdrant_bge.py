#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
qdrant_rag.py — ColQwen2 + MedGemma-27B-IT indexing & retrieval (v4.1, cloud-ready)
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
    AutoProcessor, AutoModelForImageTextToText,
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
    ROOT: Path = Path("/home/students/Leishmania")
    EXTRACT_ROOT: Path = ROOT / "kaggle" / "working2" / "extract"  # case_dir/pages/page_*.png

    # Models
    RET_MODEL_ID: str = "vidore/colqwen2-v1.0-hf"
    GEN_MODEL_ID: str = "google/medgemma-27b-it"

    # Re-ranker (text cross-encoder) — can be disabled from caller
    RERANKER_MODEL_ID: str = "BAAI/bge-reranker-v2-m3"
    RERANK_MIN_EXCERPT_CHARS: int = 80   # gate influence if excerpt too short/missing
    RERANK_ALPHA: float = 0.6            # weight for text re-ranker vs. ColQwen2 sim
    RERANK_FALLBACK_ALPHA: float = 0.2   # downweight when excerpt is short/missing

    # HF cache / device
    HF_CACHE: Path = Path(os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers"))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Qdrant (allow env overrides)
    COLLECTION: str = os.getenv("QDRANT_COLLECTION", "leish_cases_pages")
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None

    # Indexing & generation
    BATCH_EMBED: int = 8
    LANGUAGE: str = "en"
    MAX_NEW_TOKENS: int = 256
    TOP_K: int = 8

    # Retrieval scoring (None => no score_threshold)
    SCORE_THRESHOLD: Optional[float] = 0.25

    # Payload limits
    MAX_TEXT_EXCERPT: int = 800
    MAX_KEYWORDS: int = 20

    # Processor stability
    USE_FAST_PROCESSORS: bool = True

    PDF_SEARCH_DIRS: Tuple[Path, ...] = (
        Path("/home/students/Leishmania/data/standard"),
        # Path("/home/students/Leishmania/data/fix"),
    )

# ------------------------------
# HF local cache helpers (offline-robust)
# ------------------------------
def _cache_repo_dir(model_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"models--{model_id.replace('/', '--')}"

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
    boosts = {"leishmaniasis":3, "leishmania":3, "amastigotes":4, "promastigotes":3, "auricular":3,
              "pinna":2, "hiv":3, "mucocutaneous":3, "visceral":3, "cutaneous":2, "martiniquensis":3,
              "squamous":2, "carcinoma":2, "lupus":2, "vulgaris":2, "recidivans":3, "mimicking":2}
    for k,b in boosts.items():
        if k in counts: counts[k] *= b
    return [w for w,_ in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]]

def guess_case_type(title: str, text: str) -> str:
    s = f"{title} {text}".lower()
    if "visceral" in s or "kala-azar" in s: return "visceral"
    if "mucocutaneous" in s or "mucosal" in s: return "mucocutaneous"
    if "cutaneous" in s: return "cutaneous"
    return "unknown"

def extract_entities(title: str, text: str) -> List[str]:
    s = f"{title} {text}"
    ents = set()
    for pat in [
        r"Leishmania\s+[A-Z][a-z]+", r"L\.\s*[a-z]+",
        r"martiniquensis", r"donovani", r"infantum", r"tropica", r"major", r"braziliensis", r"mexicana",
        r"amastigote[s]?", r"promastigote[s]?", r"macrophage[s]?", r"histiocyte[s]?", r"auricular", r"pinna",
        r"HIV", r"immunosuppression", r"immunocompromis(ed|ed)", r"squamous\s+cell\s+carcinoma", r"lupus\s+vulgaris",
        r"recidivans"
    ]:
        for m in re.findall(pat, s, flags=re.I):
            ents.add(m.strip())
    return sorted(list(ents))[:20]

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
    elif edge_density > 0.06 and sat_mean < 50 and entropy > 5.2:
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
        batch = {k: v.to(self.model.device) for k,v in batch.items()}
        out = self.model(**batch)
        if hasattr(self.model, "get_image_features"):
            pooled = self.model.get_image_features(**batch)
        else:
            pooled = self._pool(out.embeddings, batch["attention_mask"])
        return pooled.detach().cpu().numpy().astype(np.float32)

    @torch.inference_mode()
    def embed_texts(self, queries: List[str]) -> np.ndarray:
        batch = self.processor(text=queries, return_tensors="pt")
        batch = {k: v.to(self.model.device) for k,v in batch.items()}
        out = self.model(**batch)
        if hasattr(self.model, "get_text_features"):
            pooled = self.model.get_text_features(**batch)
        else:
            pooled = self._pool(out.embeddings, batch["attention_mask"])
        return pooled.detach().cpu().numpy().astype(np.float32)

# ------------------------------
# MedGemma generator
# ------------------------------
class MedGemma:
    """
    MedGemma loader with 4-bit NF4 quantization (bitsandbytes) and graceful fallbacks.
    Designed for RTX 4090 (24GB). Requires local cache availability.
    Env overrides:
      GEN_MODEL_ID, GEN_4BIT(=1/0), GEN_COMPUTE_DTYPE(bf16/fp16), GEN_OFFLOAD_DIR, GEN_MAX_CPU_MEM
    """
    def __init__(self, model_id: str):
        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)

        # --- runtime knobs (can be changed via env without code edits) ---
        use_4bit = os.getenv("GEN_4BIT", "1") != "0"       # default: 4-bit ON
        comp = os.getenv("GEN_COMPUTE_DTYPE", "bf16").lower()
        compute_dtype = torch.bfloat16 if comp in ("bf16","bfloat16") else torch.float16
        offload_dir = Path(os.getenv("GEN_OFFLOAD_DIR", "/tmp/medgemma_offload"))
        offload_dir.mkdir(parents=True, exist_ok=True)
        cpu_mem = os.getenv("GEN_MAX_CPU_MEM", "48GiB")    # RAM budget for offload
        cuda_mem = os.getenv("GEN_MAX_CUDA_MEM", "24GiB")  # VRAM budget per GPU

        # Make CUDA friendlier
        try:
            torch.backends.cuda.matmul.allow_tf32 = True   # on Ampere+ improves perf
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

        device_map = "auto"
        
        # FIXED: Build max_memory with correct key types (integers for GPU devices)
        def _build_max_memory(cuda_mem, cpu_mem):
            mm = {"cpu": cpu_mem}
            # CUDA devices use integer keys, not "cuda:0"
            if hasattr(torch, "cuda") and torch.cuda.is_available():
                for i in range(torch.cuda.device_count()):
                    mm[i] = cuda_mem
            # Intel XPU support (optional)
            if hasattr(torch, "xpu") and torch.xpu.is_available():
                for i in range(torch.xpu.device_count()):
                    mm[i] = cuda_mem
            return mm

        max_memory = _build_max_memory(cuda_mem, cpu_mem)

        self.proc = AutoProcessor.from_pretrained(
            name_or_path,
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
            use_fast=CFG.USE_FAST_PROCESSORS
        )

        model_kwargs = dict(
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
            device_map=device_map,
            max_memory=max_memory,               # <-- now uses correct integer keys
            trust_remote_code=True,
            offload_folder=str(offload_dir),
            low_cpu_mem_usage=True,
        )

        # Try 4-bit first (fits 24GB comfortably with CPU offload)
        loaded = False
        if use_4bit:
            try:
                from transformers import BitsAndBytesConfig
                bnb_cfg = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=compute_dtype,
                )
                self.model = AutoModelForImageTextToText.from_pretrained(
                    name_or_path,
                    quantization_config=bnb_cfg,
                    **model_kwargs,
                ).eval()
                loaded = True
            except Exception as e:
                logging.warning(f"4-bit load failed ({type(e).__name__}): {e}. Falling back…")

        if not loaded:
            # Fallback: no quantization (will be slower, may OOM at larger settings)
            # Tip: keep CFG.MAX_NEW_TOKENS modest and images_per_answer ≤ 4
            self.model = AutoModelForImageTextToText.from_pretrained(
                name_or_path,
                torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
                **model_kwargs,
            ).eval()

        # Optional: try FlashAttention2 if available (safe no-op if not installed)
        try:
            self.model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

        # Some Med* tokenizers lack pad; set a sane default if missing
        try:
            if getattr(self.proc.tokenizer, "pad_token_id", None) is None and getattr(self.proc.tokenizer, "eos_token_id", None) is not None:
                self.proc.tokenizer.pad_token_id = self.proc.tokenizer.eos_token_id
        except Exception:
            pass

    def _msgs(self, q: str, ims: List[Image.Image]) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type":"text","text":
                "You are a careful medical assistant. Use only the provided images to answer. "
                "If unsupported by pixels, say 'Insufficient evidence.'"}]},
            {"role": "user", "content": ([{"type":"text","text": q}]
                                         + [{"type":"image","image": im} for im in ims])},
        ]

    @torch.inference_mode()
    def answer(self, q: str, image_paths: List[Path]) -> str:
        # Keep image batch small to reduce peak VRAM; you already limit caller to ~4
        ims = [Image.open(p).convert("RGB") for p in image_paths]
        msgs = self._msgs(q, ims)

        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        
        # FIX 1: Pass images as flat list, not list-of-list
        batch = self.proc(text=[text], images=ims, return_tensors="pt")

        # Move inputs to model device without exploding VRAM
        dev = next(self.model.parameters()).device
        batch = {k: (v.to(dev) if hasattr(v, "to") else v) for k, v in batch.items()}

        in_len = batch["input_ids"].shape[-1]
        
        # FIX 2: Use only the model's EOS token, not <end_of_turn>
        eos_id = self.proc.tokenizer.eos_token_id
        
        # FIX 3: Ensure pad token exists
        pad_id = self.proc.tokenizer.pad_token_id or self.proc.tokenizer.eos_token_id

        out = self.model.generate(
            **batch,
            max_new_tokens=CFG.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            eos_token_id=eos_id,      # Single int, not list
            pad_token_id=pad_id,      # Ensure pad token exists
            use_cache=True,
        )
        ids = out[0][in_len:]
        return self.proc.decode(ids, skip_special_tokens=True)
    
class TextCrossReranker:
    """
    BAAI/bge-reranker-v2-m3 cross-encoder. Returns raw logits (higher = more relevant).
    """
    def __init__(self, model_id: str):
        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)
        self.tok = AutoTokenizer.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True, use_fast=True
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
            device_map="auto",
        ).eval()

    @torch.inference_mode()
    def score(self, query: str, docs: List[str], max_length: int = 512) -> List[float]:
        pairs = [(query, d) for d in docs]
        enc = self.tok(
            text=[p[0] for p in pairs],
            text_pair=[p[1] for p in pairs],
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)
        out = self.model(**enc)
        # Handle shape [B,1] or [B] robustly
        logits = out.logits
        if logits.ndim == 2 and logits.shape[1] == 1:
            logits = logits.squeeze(1)
        return logits.detach().float().cpu().tolist()

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

def rerank_with_text(
    question: str,
    hits: List[Dict[str, Any]],
    reranker: Optional[TextCrossReranker] = None,
    alpha: float = CFG.RERANK_ALPHA,
    min_excerpt_chars: int = CFG.RERANK_MIN_EXCERPT_CHARS,
    fallback_alpha: float = CFG.RERANK_FALLBACK_ALPHA
) -> List[Dict[str, Any]]:
    """
    Rerank hits using late fusion of ColQwen2 similarity and text cross-encoder scores.
    Falls back gracefully when text excerpts are missing or too short.
    """
    if not hits or not reranker:
        return hits
    
    # Extract text excerpts and check validity
    excerpts = []
    valid_mask = []
    for h in hits:
        excerpt = (h.get("text_excerpt") or "").strip()
        excerpts.append(excerpt)
        valid_mask.append(len(excerpt) >= min_excerpt_chars)
    
    # Only score valid excerpts to save compute
    valid_excerpts = [ex for ex, valid in zip(excerpts, valid_mask) if valid]
    
    # Get cross-encoder scores for valid excerpts
    rerank_scores = [None] * len(hits)
    if valid_excerpts:
        try:
            valid_scores = reranker.score(question, valid_excerpts)
            score_iter = iter(valid_scores)
            for i, valid in enumerate(valid_mask):
                if valid:
                    rerank_scores[i] = next(score_iter)
        except Exception as e:
            logging.warning(f"Reranking failed: {e}")
            return hits
    
    # Extract original similarity scores
    sim_scores = [float(h.get("score", 0.0)) for h in hits]
    
    # Z-score normalize both score sets
    z_sim = z_score_normalize(sim_scores)
    z_rerank = z_score_normalize([s if s is not None else 0.0 for s in rerank_scores])
    
    # Late fusion with adaptive alpha
    fused_results = []
    for i, h in enumerate(hits):
        # Use fallback alpha if text excerpt is invalid
        effective_alpha = alpha if valid_mask[i] else fallback_alpha
        
        # Compute fused score
        fused_score = (1.0 - effective_alpha) * z_sim[i]
        if rerank_scores[i] is not None:
            fused_score += effective_alpha * z_rerank[i]
        
        # Create enriched hit with debug info
        enriched_hit = dict(h)
        enriched_hit.update({
            "fused_score": fused_score,
            "original_rank": i + 1,
            "rerank_logit": rerank_scores[i],
            "z_sim": z_sim[i],
            "z_rerank": z_rerank[i] if rerank_scores[i] is not None else None,
            "text_valid": valid_mask[i]
        })
        fused_results.append((fused_score, enriched_hit))
    
    # Sort by fused score and return
    fused_results.sort(key=lambda x: x[0], reverse=True)
    return [hit for _, hit in fused_results]

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
    vectors = {"image": VectorParams(size=dim, distance=Distance.COSINE)}
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

# update find_case_pdf
def find_case_pdf(case_dir: Path) -> Optional[Path]:
    """Find the matching PDF for this case directory."""
    # 1) current behavior
    pdfs = list(case_dir.glob("*.pdf")) + list(case_dir.glob("*.PDF"))
    if not pdfs:
        parent = case_dir.parent
        pdfs = list(parent.glob(f"{case_dir.name}*.pdf")) + list(parent.glob(f"{case_dir.name}*.PDF"))
    if pdfs:
        try:
            return max(pdfs, key=lambda p: p.stat().st_size)
        except Exception:
            return pdfs[0]

    # 2) NEW: search external roots by exact stem
    stem = case_dir.name
    for root in CFG.PDF_SEARCH_DIRS:
        cand = root / f"{stem}.pdf"
        if cand.exists():
            return cand
        # be tolerant to benign whitespace diffs
        for p in root.glob(f"{stem}*.pdf"):
            return p

    return None

def read_pdf_page_text(pdf_path: Path, page_index: int) -> str:
    if not _HAVE_PYMUPDF or not pdf_path or not pdf_path.exists():
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            if 0 <= page_index < len(doc):
                return (doc.load_page(page_index).get_text("text") or "").strip()
            # relaxed fallback: neighbor page
            for delta in (-1, 1):
                idx = page_index + delta
                if 0 <= idx < len(doc):
                    t = (doc.load_page(idx).get_text("text") or "").strip()
                    if t: return t
    except Exception:
        pass
    return ""

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
    text_excerpt = (text[:CFG.MAX_TEXT_EXCERPT] + "…") if len(text) > CFG.MAX_TEXT_EXCERPT else text

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

def qdrant_index():
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)

    # Infer embedding dimension and ensure the collection exists on the target endpoint
    dim = int(retr.embed_images([Image.new("RGB", (64, 64))]).shape[-1])
    ensure_collection_exists(client, dim)
    logging.info("Using Qdrant at %s, collection '%s'.", CFG.QDRANT_URL, CFG.COLLECTION)

    cases = discover_cases(CFG.EXTRACT_ROOT)
    logging.info("Found %d case(s).", len(cases))

    for c in cases:
        pages = list_pages_full(c)
        if not pages:
            continue
        logging.info("Indexing %s (%d pages)…", c.name, len(pages))

        for i in range(0, len(pages), CFG.BATCH_EMBED):
            chunk = pages[i:i+CFG.BATCH_EMBED]
            ims = [Image.open(p).convert("RGB") for p in chunk]
            vecs = retr.embed_images(ims)
            points: List[PointStruct] = []

            for j, p in enumerate(chunk):
                m = re.search(r"page_(\d+)\.png$", p.name)
                num = int(m.group(1)) if m else 1
                page_idx = max(0, num - 1)  # Convert to 0-based for PyMuPDF
                payload = build_payload(c, p, page_idx, ims[j])
                points.append(PointStruct(id=payload["uid"], payload=payload, vector={"image": vecs[j]}))

            client.upsert(collection_name=CFG.COLLECTION, points=points, wait=True)

        logging.info("Done: %s", c.name)
    logging.info("✅ Indexing complete.")

# ------------------------------
# Query + (optional) generate
# ------------------------------
def _qdrant_search(client: QdrantClient,
                   qv: np.ndarray,
                   top_k: int,
                   filt: Optional[Filter],
                   score_th: Optional[float]):
    """
    Unified search that prefers `query_points` (new API) with named vector,
    and falls back to deprecated `search()` when needed.
    NOTE: We interpret score_th as a *similarity* threshold in [0,1].
          For query_points (distance-based), we convert to distance cutoff.
    """
    vec_list = qv.tolist()
    # Clamp and convert similarity threshold to distance if using query_points + cosine
    sim_th = None if score_th is None else max(0.0, min(1.0, float(score_th)))

    # Try new API first
    try:
        from qdrant_client.http.models import NearVector
        qobj = NearVector(vector=vec_list, using="image")
        kwargs = dict(
            collection_name=CFG.COLLECTION,
            query=qobj,
            limit=top_k,
            filter=filt,            # <-- correct kwarg name for query_points
            with_payload=True,
        )
        if sim_th is not None:
            # cosine distance = 1 - cosine_sim
            kwargs["score_threshold"] = 1.0 - sim_th
        res = client.query_points(**kwargs)
        # Normalize Qdrant distance -> similarity so downstream assumes higher=better
        for p in getattr(res, "points", []) or []:
            try:
                d = float(getattr(p, "score", 0.0))
            except Exception:
                d = 0.0
            setattr(p, "score", 1.0 - d)
        return res
    except Exception:
        # Fall back to legacy search() with NamedVector semantics (similarity threshold)
        return client.search(
            collection_name=CFG.COLLECTION,
            query_vector=NamedVector(name="image", vector=vec_list),
            limit=top_k,
            query_filter=filt,      # legacy kwarg name
            with_payload=True,
            score_threshold=(sim_th if sim_th is not None else None),
        )

def qdrant_ask_text(question: str, top_k: int = CFG.TOP_K,
                    case_type: Optional[str] = None,
                    keyword: Optional[str] = None,
                    micrograph_only: bool = False,
                    micrograph_strict: bool = False,
                    use_reranker: bool = True) -> Dict[str, Any]:
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)
    qv = retr.embed_texts([question])[0]

    # Build filter
    musts = []
    if case_type:
        musts.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
    if keyword:
        musts.append(FieldCondition(key="keywords", match=MatchAny(any=[keyword])))
    base_filter = Filter(must=musts) if musts else None

    # Adjust retrieval size for reranking pool
    pool_size = top_k * 3 if use_reranker else top_k

    # Handle strict micrograph filtering
    if micrograph_only and micrograph_strict:
        strict_filter = Filter(must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))])
        hits = _qdrant_search(client, qv, pool_size, strict_filter, CFG.SCORE_THRESHOLD)
    else:
        hits = _qdrant_search(client, qv, pool_size, base_filter, CFG.SCORE_THRESHOLD)

    # Convert to dict format
    items = getattr(hits, "points", hits) or []

    raw_items = [{
        "rank": i+1,
        "score": float(getattr(h, "score", 0.0)),
        "doc_id": (getattr(h, "payload", {}) or {}).get("doc_id"),
        "page_index": (getattr(h, "payload", {}) or {}).get("page_index"),
        "image_path": (getattr(h, "payload", {}) or {}).get("image_path"),
        "case_type": (getattr(h, "payload", {}) or {}).get("case_type"),
        "page_kind": (getattr(h, "payload", {}) or {}).get("page_kind"),
        "micrograph_like": (getattr(h, "payload", {}) or {}).get("micrograph_like"),
        "keywords": ((getattr(h, "payload", {}) or {}).get("keywords", [])[:6]),
        "text_excerpt": (getattr(h, "payload", {}) or {}).get("text_excerpt"),
    } for i, h in enumerate(items)]

    # Apply reranking if enabled
    if use_reranker and raw_items:
        try:
            reranker = TextCrossReranker(CFG.RERANKER_MODEL_ID)
            raw_items = rerank_with_text(question, raw_items, reranker,
                                        alpha=CFG.RERANK_ALPHA,
                                        min_excerpt_chars=CFG.RERANK_MIN_EXCERPT_CHARS,
                                        fallback_alpha=CFG.RERANK_FALLBACK_ALPHA)
        except Exception as e:
            logging.warning(f"Reranking failed, falling back to original scores: {e}")

    # Apply micrograph preference if requested (after reranking)
    if micrograph_only and not micrograph_strict and raw_items:
        prefer = [r for r in raw_items if r.get("micrograph_like") is True]
        others = [r for r in raw_items if not r.get("micrograph_like")]
        selected = (prefer + others)[:top_k] if prefer else raw_items[:top_k]
    else:
        selected = raw_items[:top_k]

    return {"mode": "text", "question": question, "hits": selected}

def qdrant_ask_image(image_path: str, top_k: int = CFG.TOP_K,
                     micrograph_only: bool = False, micrograph_strict: bool = False) -> Dict[str, Any]:
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)
    im = Image.open(image_path).convert("RGB")
    qv = retr.embed_images([im])[0]

    base_filter = None
    if micrograph_only and micrograph_strict:
        base_filter = Filter(must=[FieldCondition(key="micrograph_like", match=MatchValue(value=True))])

    hits = _qdrant_search(client, qv, top_k * (3 if micrograph_only and not micrograph_strict else 1),
                          base_filter, CFG.SCORE_THRESHOLD)

    items = getattr(hits, "points", hits) or []

    raw_items = [{
        "rank": i+1,
        "score": float(getattr(h, "score", 0.0)),
        "doc_id": (getattr(h, "payload", {}) or {}).get("doc_id"),
        "page_index": (getattr(h, "payload", {}) or {}).get("page_index"),
        "image_path": (getattr(h, "payload", {}) or {}).get("image_path"),
        "case_type": (getattr(h, "payload", {}) or {}).get("case_type"),
        "page_kind": (getattr(h, "payload", {}) or {}).get("page_kind"),
        "micrograph_like": (getattr(h, "payload", {}) or {}).get("micrograph_like"),
        "keywords": ((getattr(h, "payload", {}) or {}).get("keywords", [])[:6]),
        "text_excerpt": (getattr(h, "payload", {}) or {}).get("text_excerpt"),
    } for i, h in enumerate(items)]

    if micrograph_only and not micrograph_strict and raw_items:
        prefer = [r for r in raw_items if r.get("micrograph_like") is True]
        others = [r for r in raw_items if not r.get("micrograph_like")]
        selected = (prefer + others)[:top_k] if prefer else raw_items[:top_k]
    else:
        selected = raw_items[:top_k]

    return {"mode": "image", "image": image_path, "hits": selected}

def answer_with_mdg(question: str, hits: List[Dict[str, Any]], take: int = 4) -> Dict[str, Any]:
    if not hits:
        raise SystemExit("No hits to answer from. Relax filters or lower SCORE_THRESHOLD.")
    mdg = MedGemma(CFG.GEN_MODEL_ID)
    imgs = [Path(h["image_path"]) for h in hits[:take]]
    ans = mdg.answer(question, imgs)
    return {"answer": ans, "used_images": [str(p) for p in imgs]}

def ocr_png_fallback(case_dir: Path, page_index: int) -> str:
    try:
        import pytesseract
        from PIL import Image
    except Exception:
        return ""
    pm = load_page_map(case_dir)
    paths = page_indices_to_paths(case_dir, [page_index])
    if not paths:
        return ""
    try:
        img = Image.open(paths[0]).convert("RGB")
        return pytesseract.image_to_string(img) or ""
    except Exception:
        return ""

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
        description="Qdrant RAG for ColQwen2 + MedGemma-27B-IT (rich attrs v4.1)",
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
                                  args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
            if res["hits"]:
                out = answer_with_mdg(args.question, res["hits"])
                print(json.dumps(out, indent=2, ensure_ascii=False))
            else:
                logging.warning("No hits after filters/threshold. Try relaxing filters or lowering SCORE_THRESHOLD.")
        elif args.image:
            res = qdrant_ask_image(args.image, args.topk, args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
        else:
            raise SystemExit("Provide --q or --image")
