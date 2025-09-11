#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Qdrant + ColQwen2 + MedGemma-4B-IT — Rich payload v4 (soft micrograph filter + offline robust)

- Soft preference for micrograph pages (no empty results): --micrograph_only
- Hard filter only when requested: --micrograph_strict
- score_threshold only applied when provided (None => no threshold)
- Offline-friendly HF cache loading with snapshot materialization

Tested with:
- qdrant-client==1.9.x
- transformers >= 4.43
"""

from __future__ import annotations
import os, re, json, time, hashlib, logging, uuid, string, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Optional

# --- Force offline behavior for HF ---
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
from PIL import Image, ImageStat
from dotenv import load_dotenv; load_dotenv()

# Optional PDF page text extraction
try:
    import fitz  # PyMuPDF
    _HAVE_PYMUPDF = True
except Exception:
    _HAVE_PYMUPDF = False

import torch
from transformers import (
    AutoProcessor, AutoModelForImageTextToText,
    ColQwen2ForRetrieval, ColQwen2Processor
)

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
    GEN_MODEL_ID: str = "google/medgemma-4b-it"

    # HF cache / device
    HF_CACHE: Path = Path(os.getenv("TRANSFORMERS_CACHE", "/data4t/hf/transformers"))
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Qdrant Cloud
    COLLECTION: str = "leish_cases_pages"
    QDRANT_URL: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    QDRANT_API_KEY: Optional[str] = os.getenv("QDRANT_API_KEY") or None

    # Indexing & generation
    BATCH_EMBED: int = 8
    BATCH_UPSERT: int = 32
    LANGUAGE: str = "en"
    MAX_NEW_TOKENS: int = 512
    TOP_K: int = 8

    # Retrieval scoring (None => do not pass to Qdrant)
    SCORE_THRESHOLD: Optional[float] = 0.25

    # Rich-attr controls
    MAX_TEXT_EXCERPT: int = 600
    MAX_KEYWORDS: int = 20

    # Processor stability / warnings
    USE_FAST_PROCESSORS: bool = True  # set False to mirror old slow behavior

# ------------------------------
# HF local cache helpers (offline)
# ------------------------------
REQUIRED_FILES = [
    "config.json",
    "model.safetensors.index.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
    "special_tokens_map.json",
    "generation_config.json",
    "processor_config.json",
    "preprocessor_config.json",
    "chat_template.jinja",
    "added_tokens.json",
]

def _cache_repo_dir(model_id: str, cache_dir: Path) -> Path:
    return cache_dir / f"models--{model_id.replace('/', '--')}"

def _list_shards(repo_dir: Path) -> List[str]:
    idx = repo_dir / "model.safetensors.index.json"
    shards = []
    if idx.exists():
        try:
            import json as _json
            j = _json.loads(idx.read_text())
            shards = sorted(set(j.get("weight_map", {}).values()))
        except Exception:
            pass
    if not shards:
        # common fallbacks
        for k in (1,2):
            cand = f"model-0000{k}-of-00002.safetensors"
            if (repo_dir / cand).exists():
                shards.append(cand)
    return shards

def _ensure_snapshot_complete(repo_dir: Path) -> Path:
    snaps_root = repo_dir / "snapshots"
    shards = _list_shards(repo_dir)
    has_all = all((repo_dir / f).exists() for f in REQUIRED_FILES) and all((repo_dir / s).exists() for s in shards)
    if has_all:
        return repo_dir
    snaps_root.mkdir(parents=True, exist_ok=True)
    rev_dir = snaps_root / "offline-materialized"
    rev_dir.mkdir(exist_ok=True)
    for f in REQUIRED_FILES:
        src, dst = repo_dir / f, rev_dir / f
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    for s in shards:
        src, dst = repo_dir / s, rev_dir / s
        if src.exists() and not dst.exists():
            try: shutil.copy2(src, dst)
            except Exception: pass
    crit = ["config.json", "model.safetensors.index.json"]
    if all((rev_dir / c).exists() for c in crit) and any((rev_dir / s).exists() for s in shards):
        return rev_dir
    return repo_dir

def resolve_local_model_dir(model_id: str, cache_dir: Path) -> str:
    repo_dir = _cache_repo_dir(model_id, cache_dir)
    if not repo_dir.exists():
        return model_id
    usable = _ensure_snapshot_complete(repo_dir)
    return str(usable)

# ------------------------------
# Small helpers
# ------------------------------
def sha24(s: str) -> str: return hashlib.sha256(s.encode()).hexdigest()[:24]

def discover_cases(root: Path) -> List[Path]:
    return sorted([p for p in root.iterdir() if p.is_dir() and (p / "pages").is_dir()]) if root.is_dir() else []

def list_pages_full(case_dir: Path) -> List[Path]:
    return sorted((case_dir / "pages").glob("page_*.png"))

def build_uid(doc_id: str, page_idx: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}__p{page_idx:04d}"))

def find_case_pdf(case_dir: Path) -> Optional[Path]:
    pdfs = [p for p in case_dir.glob("*.pdf")]
    return pdfs[0] if pdfs else None

def read_pdf_page_text(pdf_path: Path, page_index: int) -> str:
    if not _HAVE_PYMUPDF or not pdf_path or not pdf_path.exists():
        return ""
    try:
        with fitz.open(pdf_path) as doc:
            if 0 <= page_index < len(doc):
                return (doc.load_page(page_index).get_text("text") or "").strip()
            for delta in (-1, 1):
                idx = page_index + delta
                if 0 <= idx < len(doc):
                    t = (doc.load_page(idx).get_text("text") or "").strip()
                    if t: return t
    except Exception:
        pass
    return ""

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
    gx[:,1:-1] = (g[:,2:] - g[:,-2]) * 0.5
    gy[1:-1,:] = (g[2:,:] - g[:-2,:]) * 0.5
    grad = np.sqrt(gx*gx + gy*gy)
    edge_density = float((grad > 0.2).mean())

    hist = np.histogram((g*255).astype(np.uint8), bins=256, range=(0,255))[0].astype(np.float32)
    p = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(p * np.log2(p + 1e-12)))

    if edge_density < 0.03 and sat_mean < 25:
        page_kind = "mostly_text"
    elif edge_density > 0.10 and sat_mean < 50 and entropy > 5.5:
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
# ColQwen2 retriever (pooled vectors)
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
# MedGemma generator (offline-robust)
# ------------------------------
class MedGemma:
    def __init__(self, model_id: str):
        name_or_path = resolve_local_model_dir(model_id, CFG.HF_CACHE)

        self.model = AutoModelForImageTextToText.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
            device_map="auto",
        ).eval()
        self.proc = AutoProcessor.from_pretrained(
            name_or_path, cache_dir=str(CFG.HF_CACHE), local_files_only=True,
            use_fast=CFG.USE_FAST_PROCESSORS
        )

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
        ims = [Image.open(p).convert("RGB") for p in image_paths]
        msgs = self._msgs(q, ims)
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        batch = self.proc(text=[text], images=[ims], return_tensors="pt").to(self.model.device)
        in_len = batch["input_ids"].shape[-1]
        eos = [self.proc.tokenizer.eos_token_id,
               self.proc.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        out = self.model.generate(**batch, max_new_tokens=CFG.MAX_NEW_TOKENS,
                                  do_sample=False, temperature=0.0, num_beams=1,
                                  eos_token_id=eos)
        ids = out[0][in_len:]
        return self.proc.decode(ids, skip_special_tokens=True)

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

def create_payload_indexes(client: QdrantClient):
    for field, schema in [
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
        "source_pdf": (str(pdf) if pdf else None),
        "text_excerpt": text_excerpt if text_excerpt else None,
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
                page_idx = int(m.group(1)) if m else 0
                payload = build_payload(c, p, page_idx, ims[j])
                points.append(PointStruct(id=payload["uid"], payload=payload, vector={"image": vecs[j]}))

            client.upsert(collection_name=CFG.COLLECTION, points=points, wait=True)

        logging.info("Done: %s", c.name)
    logging.info("✅ Indexing complete.")

# ------------------------------
# Query + (optional) generate
# ------------------------------
def _qdrant_search(client: QdrantClient, qv: np.ndarray, top_k: int, filt: Optional[Filter], score_th: Optional[float]):
    kwargs = dict(
        collection_name=CFG.COLLECTION,
        query_vector=NamedVector(name="image", vector=qv.tolist()),
        limit=top_k,
        query_filter=filt
    )
    if score_th is not None:
        kwargs["score_threshold"] = score_th
    return client.search(**kwargs)

def qdrant_ask_text(question: str, top_k: int = CFG.TOP_K,
                    case_type: Optional[str] = None,
                    keyword: Optional[str] = None,
                    micrograph_only: bool = False,
                    micrograph_strict: bool = False) -> Dict[str, Any]:
    client = qdrant()
    retr = CQ2(CFG.RET_MODEL_ID)
    qv = retr.embed_texts([question])[0]

    # Build strict filter only for case_type/keyword here
    musts = []
    if case_type:
        musts.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))
    if keyword:
        musts.append(FieldCondition(key="keywords", match=MatchAny(any=[keyword])))
    base_filter = Filter(must=musts) if musts else None

    # Strict micrograph path (explicit request)
    if micrograph_only and micrograph_strict:
        strict_filter = Filter(must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))])
        hits = _qdrant_search(client, qv, top_k, strict_filter, CFG.SCORE_THRESHOLD)
    else:
        # Soft path: search without micrograph constraint
        hits = _qdrant_search(client, qv, top_k * 3, base_filter, CFG.SCORE_THRESHOLD)  # widen pool for re-ranking

    # Build output dicts from raw hits
    raw_items = [{
        "rank": i+1,
        "score": float(h.score),
        "doc_id": h.payload.get("doc_id"),
        "page_index": h.payload.get("page_index"),
        "image_path": h.payload.get("image_path"),
        "case_type": h.payload.get("case_type"),
        "page_kind": h.payload.get("page_kind"),
        "micrograph_like": h.payload.get("micrograph_like"),
        "keywords": h.payload.get("keywords", [])[:6],
    } for i, h in enumerate(hits)]

    # Soft preference: if requested, keep micrograph pages first; if none exist, keep original (non-empty)
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

    raw_items = [{
        "rank": i+1,
        "score": float(h.score),
        "doc_id": h.payload.get("doc_id"),
        "page_index": h.payload.get("page_index"),
        "image_path": h.payload.get("image_path"),
        "case_type": h.payload.get("case_type"),
        "page_kind": h.payload.get("page_kind"),
        "micrograph_like": h.payload.get("micrograph_like"),
        "keywords": h.payload.get("keywords", [])[:6],
    } for i, h in enumerate(hits)]

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

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Qdrant RAG for ColQwen2 + MedGemma-4B-IT (rich attrs v4)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    sub.add_parser("qdrant-init", help="(Re)create collection with correct dim")
    sub.add_parser("qdrant-index", help="Index all pages under EXTRACT_ROOT/*/pages")
    sub.add_parser("qdrant-create-indexes", help="Create payload indexes for faster filtered queries")

    ask = sub.add_parser("qdrant-ask", help="Retrieve then answer with MedGemma")
    ask.add_argument("--q", help="Text question")
    ask.add_argument("--image", help="Image path for image query")
    ask.add_argument("--topk", type=int, default=CFG.TOP_K)
    ask.add_argument("--case_type", choices=["cutaneous","mucocutaneous","visceral","unknown"])
    ask.add_argument("--keyword")
    ask.add_argument("--micrograph_only", action="store_true", help="Prefer micrograph pages (soft).")
    ask.add_argument("--micrograph_strict", action="store_true", help="Require micrograph pages (hard filter).")
    ask.add_argument("--score_threshold", type=float, help="Override CFG.SCORE_THRESHOLD for this query (None => no threshold).")

    args = ap.parse_args()

    if args.cmd == "qdrant-init":
        qdrant_init()
    elif args.cmd == "qdrant-index":
        qdrant_index()
    elif args.cmd == "qdrant-create-indexes":
        create_payload_indexes(qdrant())
    elif args.cmd == "qdrant-ask":
        # allow per-run threshold override
        if args.score_threshold is not None:
            CFG.SCORE_THRESHOLD = args.score_threshold
        if args.q:
            res = qdrant_ask_text(args.q, args.topk, args.case_type, args.keyword,
                                  args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
            if res["hits"]:
                out = answer_with_mdg(args.q, res["hits"])
                print(json.dumps(out, indent=2, ensure_ascii=False))
            else:
                logging.warning("No hits after filters/threshold. Try relaxing filters or lowering SCORE_THRESHOLD.")
        elif args.image:
            res = qdrant_ask_image(args.image, args.topk, args.micrograph_only, args.micrograph_strict)
            print(json.dumps(res, indent=2))
        else:
            raise SystemExit("Provide --q or --image")
