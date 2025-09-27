#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Batch answer questions with doc_id-constrained retrieval + robust resume/dedupe.
Medgemma-4b-it version — loads encoder/reranker once, uses MedGemma4B for answering.
"""

import argparse
import json
import os
import signal
import sys
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
import re as _re
import math
from collections import Counter
import numpy as np
import torch
from PIL import Image

# Import from the Gemini-enabled module
from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import (
    CFG, qdrant, CQ2, MedGemma4B, TextCrossReranker,
    rerank_with_text, rerank_with_case_level_cross_encoder, find_case_dir, page_indices_to_paths,
    _qdrant_search, extractive_spans, find_case_pdf, read_pdf_page_text
)
from qdrant_client.http.models import Filter, FieldCondition, MatchValue, MatchAny

# -------------------- small utils --------------------
def _compute_dcg(relevances: List[float]) -> float:
    """Discounted Cumulative Gain."""
    score = 0.0
    for i, rel in enumerate(relevances, 1):
        score += (2.0**rel - 1.0) / math.log2(i + 1)
    return score

def _compute_ndcg_at_k(relevances: List[float], k: int) -> float:
    """Normalized DCG at k."""
    relevances_k = relevances[:k]
    dcg_k = _compute_dcg(relevances_k)
    ideal = sorted(relevances_k, reverse=True)
    idcg_k = _compute_dcg(ideal)
    return 0.0 if idcg_k <= 0.0 else float(dcg_k / idcg_k)

def _compute_graded_relevance(doc: str, page: Optional[int], gold_doc: str, gold_pages: Optional[List[int]]) -> int:
    """Graded relevance: exact page=3, ±1 page=2, same doc=1, else=0."""
    if doc != gold_doc:
        return 0
    if not gold_pages or page is None:
        return 1  # Same doc but no page info
    if page in gold_pages:
        return 3  # Exact page match
    if any(abs(page - gp) <= 1 for gp in gold_pages):
        return 2  # Adjacent page
    return 1  # Same doc, different page

def _compute_mrr_at_k(predictions: List[Tuple[str, int]], gold_doc: str, gold_pages: Optional[List[int]], k: int) -> float:
    """Mean Reciprocal Rank at k."""
    for i, (doc, page) in enumerate(predictions[:k], 1):
        if doc != gold_doc:
            continue
        if not gold_pages or page in gold_pages or any(abs(page - gp) <= 1 for gp in gold_pages):
            return 1.0 / i
    return 0.0

def _compute_recall_at_k(predictions: List[Tuple[str, int]], gold_doc: str, gold_pages: Optional[List[int]], k: int) -> float:
    """Recall at k."""
    if not gold_pages:
        return 1.0 if any(doc == gold_doc for doc, _ in predictions[:k]) else 0.0
    for doc, page in predictions[:k]:
        if doc == gold_doc and (page in gold_pages or any(abs(page - gp) <= 1 for gp in gold_pages)):
            return 1.0
    return 0.0

def _compute_precision_at_k(predictions: List[Tuple[str, int]], gold_doc: str, gold_pages: Optional[List[int]], k: int) -> float:
    """Precision at k."""
    if k <= 0:
        return 0.0
    relevant_count = 0
    for doc, page in predictions[:k]:
        if doc != gold_doc:
            continue
        if not gold_pages or page in gold_pages or any(abs(page - gp) <= 1 for gp in gold_pages):
            relevant_count += 1
    return float(relevant_count) / float(k)

def _extract_doc_page_pairs(hits: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
    """Extract (doc_id, page_index) pairs from hits."""
    pairs = []
    for hit in hits or []:
        doc = hit.get("doc_id")
        page = hit.get("page_index")
        
        # Try to recover from image_path if needed
        if (doc is None or page is None) and hit.get("image_path"):
            path_str = str(hit["image_path"])
            if doc is None and "/extract/" in path_str:
                parts = path_str.split("/")
                try:
                    idx = parts.index("extract")
                    if idx + 1 < len(parts):
                        doc = parts[idx + 1]
                except (ValueError, IndexError):
                    pass
            if page is None:
                match = _re.search(r"page_(\d+)\.png", path_str)
                if match:
                    page = int(match.group(1)) - 1  # Convert to 0-based
        
        if doc is not None and page is not None:
            pairs.append((str(doc), int(page)))
    return pairs

def _infer_gold_pages_from_seeds(row: Dict[str, Any]) -> Optional[List[int]]:
    """Infer gold pages from seed_image_paths in the row."""
    seeds = row.get("seed_image_paths") or []
    pages = []
    for path in seeds:
        if isinstance(path, str):
            match = _re.search(r"page_(\d+)\.png", path)
            if match:
                pages.append(int(match.group(1)) - 1)  # Convert to 0-based
    return sorted(set(pages)) if pages else None

def _compute_retrieval_metrics(hits: List[Dict[str, Any]], doc_id: str, gold_pages: Optional[List[int]], k: int = 10) -> Dict[str, Any]:
    """Compute comprehensive retrieval metrics for a single query."""
    predictions = _extract_doc_page_pairs(hits)
    
    # Compute graded relevances
    relevances = [_compute_graded_relevance(doc, page, doc_id, gold_pages) for doc, page in predictions]
    
    # Core metrics
    metrics = {
        "k": k,
        "gold_pages": gold_pages,
        "predictions_count": len(predictions)
    }
    
    # Page-level metrics (only if we have gold pages)
    if gold_pages:
        metrics.update({
            "MRR@k": round(_compute_mrr_at_k(predictions, doc_id, gold_pages, k), 6),
            "nDCG@k": round(_compute_ndcg_at_k(relevances, k), 6),
            "Recall@k": round(_compute_recall_at_k(predictions, doc_id, gold_pages, k), 6),
            "Precision@k": round(_compute_precision_at_k(predictions, doc_id, gold_pages, k), 6),
        })
    
    # Document-level recall (always computable)
    doc_recall = 1.0 if any(doc == doc_id for doc, _ in predictions[:k]) else 0.0
    metrics["doc_recall@k"] = round(doc_recall, 6)
    
    # Proxy metrics
    text_coverage = sum(1 for h in hits[:k] if (h.get("text_excerpt") or "").strip()) / max(1, k)
    pages_in_topk = [h.get("page_index") for h in hits[:k] if isinstance(h.get("page_index"), int)]
    dup_ratio = 1.0 - (len(set(pages_in_topk)) / max(1, len(pages_in_topk)))
    
    metrics.update({
        "text_coverage@k": round(text_coverage, 6),
        "duplicate_ratio@k": round(dup_ratio, 6),
        "unique_pages@k": len(set(pages_in_topk)),
    })
    
    # Modality distribution
    modality_counts = Counter(h.get("via", "unknown") for h in hits[:k])
    metrics["modality_dist@k"] = dict(modality_counts)
    
    return metrics

def _check_retrieval_quality(metrics: Dict[str, Any], thresholds: Dict[str, float]) -> Tuple[bool, List[str]]:
    """Check if retrieval meets quality thresholds and return warnings."""
    warnings = []
    
    # Check document recall
    if metrics.get("doc_recall@k", 0) < thresholds.get("min_doc_recall", 0.8):
        warnings.append(f"Low doc recall: {metrics['doc_recall@k']:.2f}")
    
    # Check text coverage
    if metrics.get("text_coverage@k", 0) < thresholds.get("min_text_coverage", 0.5):
        warnings.append(f"Low text coverage: {metrics['text_coverage@k']:.2f}")
    
    # Check duplicates
    if metrics.get("duplicate_ratio@k", 0) > thresholds.get("max_dup_ratio", 0.3):
        warnings.append(f"High duplicates: {metrics['duplicate_ratio@k']:.2f}")
    
    # Check unique pages
    if metrics.get("unique_pages@k", 0) < thresholds.get("min_unique_pages", 3):
        warnings.append(f"Few unique pages: {metrics['unique_pages@k']}")
    
    return len(warnings) == 0, warnings

def _stable_key(row: Dict[str, Any]) -> str:
    """Prefer question_id; else derive a stable key from (case_id, question)."""
    qid = row.get("question_id")
    if isinstance(qid, str) and qid.strip():
        return f"qid:{qid.strip()}"
    case_id = (row.get("case_id") or row.get("doc_id") or row.get("caseId") or "").strip()
    qtxt = (row.get("question") or "").strip()
    h = hashlib.sha1((case_id + "\n" + qtxt).encode("utf-8")).hexdigest()
    return f"hk:{h}"

def _read_existing_out(path: Path) -> Tuple[Set[str], Set[str], Set[str]]:
    """Returns (done_keys, error_keys, insuf_keys) from an NDJSON output file."""
    done, err, insuf = set(), set(), set()
    if not path.exists():
        return done, err, insuf
    with path.open("r", encoding="utf-8", errors="replace") as fin:
        for ln in fin:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            k = _stable_key(rec)
            ans = (rec.get("answer") or "").strip()
            if "error" in rec:
                err.add(k)
            elif ans:
                # treat "Insufficient evidence" as its own bucket
                if ans.lower().startswith("insufficient evidence"):
                    insuf.add(k)
                else:
                    done.add(k)
    return done, err, insuf

def repair_image_path(hit: Dict[str, Any]) -> Optional[Path]:
    """Repair image paths using extraction dir + page index when payload path is stale/missing."""
    raw = hit.get("image_path")
    if raw:
        p = Path(raw)
        if p.exists():
            return p

    doc_id = hit.get("doc_id")
    pi = hit.get("page_index")
    if not (doc_id and isinstance(pi, int)):
        # NEW: derive doc_id from image_path as a fallback
        raw = hit.get("image_path")
        if raw:
            try:
                p = Path(raw)
                if "pages" in p.parts:
                    i = p.parts.index("pages")
                    if i >= 1:
                        doc_id = p.parts[i-1]
            except Exception:
                pass
    if not (doc_id and isinstance(pi, int)):
        return None

    case_dir = find_case_dir(doc_id, CFG.EXTRACT_ROOT)
    if not case_dir:
        return None

    paths = page_indices_to_paths(case_dir, [pi])
    if paths and paths[0].exists():
        return paths[0]

    p1 = case_dir / "pages" / f"page_{pi+1:04d}.png"  # 1-based filename
    if p1.exists():
        return p1
    p2 = case_dir / "pages" / f"page_{pi:04d}.png"    # 0-based filename (rare)
    if p2.exists():
        return p2

    return None

def select_images(seed: List[str], hits: List[Dict[str, Any]], take: int, target_doc_id: Optional[str] = None) -> List[str]:
    """
    Return up to `take` image paths WITH doc_id enforcement.
    """
    out, seen = [], set()

    # 1) seed paths first
    for p in seed or []:
        if p and Path(p).exists() and p not in seen:
            out.append(p); seen.add(p)
            if len(out) >= take:
                return out

    # 2) repair hit paths, but only from target_doc_id if given
    for h in hits:
        if target_doc_id and h.get("doc_id") != target_doc_id:
            continue  # Skip hits from other documents
        rp = repair_image_path(h)
        if rp and rp.exists():
            sp = str(rp)
            if sp not in seen:
                out.append(sp); seen.add(sp)
                if len(out) >= take:
                    return out

    # 3) strict fallback: only from target_doc_id
    fallback_doc = target_doc_id
    if fallback_doc:
        case_dir = find_case_dir(fallback_doc, CFG.EXTRACT_ROOT)
        if case_dir:
            # Use retrieval-ranked pages first
            ranked_pages = []
            for h in hits:
                if h.get("doc_id") == fallback_doc:
                    pi = h.get("page_index")
                    if isinstance(pi, int) and pi not in ranked_pages:
                        ranked_pages.append(pi)
            if not ranked_pages:
                ranked_pages = [0, 1, 2]
            
            for p in page_indices_to_paths(case_dir, ranked_pages):
                if p.exists():
                    sp = str(p)
                    if sp not in seen:
                        out.append(sp); seen.add(sp)
                        if len(out) >= take:
                            return out

    return out

# -------------------- batch processor --------------------
def _normalize_answer(text: str) -> str:
    """
    FIXED: Better normalization that handles garbage output, repetition, and prompt leakage
    """
    if not text:
        return text
    
    t = text.strip()
    
    # Early garbage detection - truncate very long outputs
    if len(t) > 4000:
        print("⚠️ [FIX] Truncating very long answer (likely repetitive)")
        t = t[:2000] + "..."
    
    # Remove prompt leakage patterns first (CRITICAL FIX)
    t = _re.sub(r'^\s*(?:user\s+)?(?:You are a medical expert[^.]*\.)?', '', t, flags=_re.I)
    t = _re.sub(r'Evidence Sources:\s*\[?\d+\]?[^.]*\.', '', t, flags=_re.I)
    t = _re.sub(r'User has uploaded \d+ files:[^.]*\.', '', t, flags=_re.I)
    t = _re.sub(r'Please analyze both the uploaded content[^.]*\.', '', t, flags=_re.I)
    
    # ENHANCED: Remove diagnostic instruction echoing (NEW FIX)
    t = _re.sub(r'Medical Question:\s*Provide a systematic medical diagnosis for:[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Please structure your diagnostic analysis as follows:[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Key clinical findings from the evidence[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Relevant differential diagnoses to consider[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Most likely diagnosis with supporting reasoning[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Diagnostic methods or confirmatory tests mentioned[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Focus on medical analysis rather than case descriptions[^.]*\.?', '', t, flags=_re.I)
    t = _re.sub(r'Continue the diagnostic analysis:[^.]*\.?', '', t, flags=_re.I)
    
    # Remove answer prefixes and labels  
    t = _re.sub(r'^\s*(?:Question\s*:\s*.*?)?\b(?:Answer|ANSWER|MEDICAL ANALYSIS)\s*:\s*', '', t, flags=_re.I | _re.S)
    
    # Handle Final Answer sections
    idx = t.lower().rfind("final answer:")
    if idx != -1:
        tail = t[idx + len("final answer:"):].strip()
        if tail and len(tail) < len(t) * 0.8:
            t = tail
    
    # Remove repetitive case descriptions (ENHANCED)
    t = _re.sub(r'(?:\b\d+\s+\d+\s+A\s+\d+-YEAR-OLD\s+[A-Z\s]+\s+WITH\s+A\s+LESION[^.]*\.?\s*){2,}', 
               '', t, flags=_re.I)
    t = _re.sub(r'(?:The lesion progressed quickly from a sore to eat through[^.]*\.?\s*){2,}', 
               'The lesion progressed quickly.', t, flags=_re.I)
    
    # Remove LaTeX and formatting
    t = _re.sub(r'\\boxed\\{([^}]*)\\}', r'\1', t)
    t = _re.sub(r'\$\$?(.*?)\$\$?', r'\1', t, flags=_re.S)
    
    # Normalize whitespace
    t = _re.sub(r'\s+', ' ', t).strip()
    
    # Enhanced sentence deduplication (IMPROVED)
    sentences = _re.split(r'(?<=[.!?])(?:["\')\]\}]+)?(?:\s+)', t)
    seen = set()
    unique_sentences = []
    
    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence) < 15:
            continue
        
        # Create deduplication key
        key = _re.sub(r'\W+', '', sentence.lower())[:50]  # First 50 chars only
        
        # Skip obvious repetitive patterns (ENHANCED)
        if any(pattern in sentence.lower() for pattern in [
            "year-old boy from laos", "year-old man from cambodia",
            "progressed quickly from", "medical expert", "evidence sources",
            "you are a medical", "user has uploaded"
        ]):
            continue
        
        if key and key not in seen:
            seen.add(key)
            unique_sentences.append(sentence)
            
            # Limit sentences to prevent runaway (CRITICAL FIX)
            if len(unique_sentences) >= 6:
                break
    
    result = " ".join(unique_sentences).strip()
    
    # Ensure proper ending
    if result and result[-1] not in '.!?':
        last_punct = max(result.rfind('.'), result.rfind('!'), result.rfind('?'))
        if last_punct > len(result) * 0.7:
            result = result[:last_punct + 1]
        else:
            result += "."
    
    # Final sanity check (CRITICAL FIX)
    if len(result) < 30 or result.count(' ') < 4:
        return "Unable to generate a clear medical answer from the available information."
    
    return result
class BatchProcessor:
    """Optimized batch processor that loads models once and reuses them."""
    
    def __init__(self, use_reranker: bool = True):
        print("[INFO] Loading models (one-time initialization)...")

        # Load once
        self.client = qdrant()
        self.encoder = CQ2(CFG.RET_MODEL_ID)
        self.gem = MedGemma4B(model_id=CFG.GEN_MODEL_ID)
        
        # Optional reranker
        self.reranker = None
        if use_reranker:
            try:
                self.reranker = TextCrossReranker(CFG.RERANKER_MODEL_ID)
                print(f"[INFO] Reranker loaded: {CFG.RERANKER_MODEL_ID}")
            except Exception as e:
                print(f"[WARN] Reranker unavailable: {e}")
                self.reranker = None
        
        # Cross-modal dimension guard
        self.use_image_search = False
        try:
            text_dim = getattr(self.encoder, "output_dim", None)
            # If you know your image vector dimension from collection schema:
            image_dim = getattr(CFG, "IMAGE_VECTOR_DIM", None)
            self.use_image_search = (text_dim is not None and image_dim is not None and text_dim == image_dim)
            if self.use_image_search:
                print(f"[INFO] Cross-modal search enabled: text_dim={text_dim}, image_dim={image_dim}")
            else:
                print(f"[INFO] Cross-modal search disabled: text_dim={text_dim}, image_dim={image_dim}")
        except Exception:
            self.use_image_search = False
            print("[WARN] Could not verify vector dimensions, disabling cross-modal search")
        
        print("[INFO] All models loaded successfully!")
    
    def enhanced_qdrant_ask_text(self, question: str, doc_id: Optional[str] = None, 
                            top_k: int = CFG.TOP_K,
                            case_type: Optional[str] = None,
                            keyword: Optional[str] = None,
                            any_keywords: Optional[str] = None,
                            micrograph_only: bool = False,
                            micrograph_strict: bool = False,
                            pool_multiplier: int = 3,
                            uploaded_images: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Enhanced with imaging question detection and multimodal retrieval support
        """
        from PIL import Image
        import torch
        import numpy as np
        
        # Text embedding (always present)
        qv_text = self.encoder.embed_texts([question])[0]
        print(f"[DEBUG] Text embedding shape: {qv_text.shape}, first 5 values: {qv_text[:5]}")
        
        # Image embeddings from uploaded images (NEW MULTIMODAL FEATURE)
        uploaded_image_embeddings = []
        if uploaded_images:
            print(f"[INFO] Processing {len(uploaded_images)} uploaded images for multimodal retrieval")
            valid_images = []
            for img_path in uploaded_images:
                try:
                    if Path(img_path).exists():
                        with Image.open(img_path).convert("RGB") as img:
                            valid_images.append(img.copy())
                except Exception as e:
                    print(f"[WARN] Failed to load uploaded image {img_path}: {e}")
            
            if valid_images:
                try:
                    raw_embeddings = self.encoder.embed_images(valid_images)
                    
                    # Handle different tensor types and formats with robust type checking
                    # Convert to list of numpy arrays to avoid boolean evaluation issues
                    if hasattr(raw_embeddings, 'dtype'):  # Check for tensor-like objects first
                        print(f"[DEBUG] Raw embedding tensor dtype: {raw_embeddings.dtype}, shape: {getattr(raw_embeddings, 'shape', 'no shape')}")
                        # Convert any tensor type to Float32 for compatibility
                        if hasattr(raw_embeddings, 'float'):
                            embedding_array = raw_embeddings.float().cpu().numpy()
                            uploaded_image_embeddings = [embedding_array[i] for i in range(len(embedding_array))]
                            print(f"[INFO] Converted tensor embeddings to Float32 for {len(uploaded_image_embeddings)} uploaded images")
                        else:
                            embedding_array = np.array(raw_embeddings, dtype=np.float32)
                            uploaded_image_embeddings = [embedding_array[i] for i in range(len(embedding_array))]
                            print(f"[INFO] Converted tensor-like embeddings to numpy for {len(uploaded_image_embeddings)} uploaded images")
                    elif isinstance(raw_embeddings, np.ndarray):
                        embedding_array = raw_embeddings.astype(np.float32)
                        uploaded_image_embeddings = [embedding_array[i] for i in range(len(embedding_array))]
                        print(f"[INFO] Generated embeddings (numpy) for {len(uploaded_image_embeddings)} uploaded images")
                    else:
                        # Fallback: try to convert to numpy array
                        embedding_array = np.array(raw_embeddings, dtype=np.float32)
                        uploaded_image_embeddings = [embedding_array[i] for i in range(len(embedding_array))]
                        print(f"[INFO] Generated embeddings (converted) for {len(uploaded_image_embeddings)} uploaded images")
                        
                except Exception as e:
                    print(f"[WARN] Failed to embed uploaded images: {e}")
                    print(f"[DEBUG] Exception type: {type(e)}, args: {e.args}")
                    uploaded_image_embeddings = []

        # Detect question types
        ql = question.lower()
        print(f"[DEBUG] Processing question: {repr(question)}")
        imaging_terms = ["neuroimaging","mri","ct","computed tomography","magnetic resonance",
                        "flair","dwi","adc","t1","t2","contrast","enhancement","lesion",
                        "brain","encephal","cns","cranial"]
        clinical_terms = ["lesion", "clinical", "course", "onset", "features", "history", 
                        "size", "location", "referral", "evolution", "appearance"]
        treatment_terms = ["treatment", "dose", "dosage", "regimen", "therapy", "outcome", 
                        "follow-up", "response", "mg", "kg", "administered", "cure", "procedure"]
        diagnostic_terms = ["diagnosis", "identify", "stain", "microscopy", "species", 
                        "pcr", "culture", "molecular", "sequencing", "identification", "having", "tell me what"]

        imaging_question = any(t in ql for t in imaging_terms)
        clinical_question = any(t in ql for t in clinical_terms)
        treatment_question = any(t in ql for t in treatment_terms)
        diagnostic_question = any(t in ql for t in diagnostic_terms)

        # Adjust pool size based on question type and uploaded images
        if uploaded_image_embeddings:
            pool_multiplier = max(pool_multiplier, 4)  # Expand search when we have uploaded images
            top_k = max(top_k, 10)  # Increase result count for better fusion
            print(f"[INFO] Multimodal mode: expanding search pool to {pool_multiplier}x, top_k={top_k}")
        elif imaging_question:
            pool_multiplier = max(pool_multiplier, 4)
            top_k = max(top_k, 12)
        elif treatment_question or diagnostic_question:
            pool_multiplier = max(pool_multiplier, 4)
            top_k = max(top_k, 10)
        elif clinical_question:
            pool_multiplier = max(pool_multiplier, 4)
            top_k = max(top_k, 8)

        # Build filter with imaging boosts
        musts, should = [], []

        if doc_id:
            musts.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
        if case_type:
            musts.append(FieldCondition(key="case_type", match=MatchValue(value=case_type)))

        if imaging_question or uploaded_image_embeddings:  # Boost visual content when we have uploaded images
            should += [
                FieldCondition(key="micrograph_like", match=MatchValue(value=True)),
                FieldCondition(key="page_kind", match=MatchValue(value="figure_or_micrograph")),
            ]

        if clinical_question and not any_keywords:
            clinical_terms = ["lesion","nodule","clinical","history","onset","course"]
            # extend, don't replace:
            should += [FieldCondition(key="keywords", match=MatchAny(any=clinical_terms))]

        if keyword:
            should += [FieldCondition(key="keywords", match=MatchAny(any=[keyword]))]

        if any_keywords:
            toks = [t.strip() for t in any_keywords.split(",") if t.strip()]
            extra = []
            for t in list(toks):
                low = t.lower()
                extra.extend([low.replace("-", ""), low.replace("-", " ")])
            toks = sorted(set(toks + extra))
            should += [
                FieldCondition(key="keywords", match=MatchAny(any=toks)),
                FieldCondition(key="entities", match=MatchAny(any=toks)),
            ]

        base_filter = Filter(must=musts, should=should) if (musts or should) else None
        if doc_id:
            assert any(isinstance(c, FieldCondition) and c.key == "doc_id" for c in (musts or [])), \
                "Safety check: doc_id constraint missing from filter"
        pool_size = top_k * pool_multiplier if self.reranker else top_k * pool_multiplier

        # Use the cross-modal dimension guard set in __init__

        # --- Dual retrieval: text-vector + cross-modal (text->image)
        # text vector search
        hits_text = _qdrant_search(self.client, qv_text, pool_size, base_filter, CFG.SCORE_THRESHOLD, using="text")

        # image vector search (strict micrograph filter if requested)
        if self.use_image_search:
            if micrograph_only and micrograph_strict:
                strict_filter = Filter(
                    must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))],
                    should=should
                )
                hits_img = _qdrant_search(self.client, qv_text, pool_size, strict_filter, CFG.SCORE_THRESHOLD, using="image")
                if not hits_img and should:
                    strict_filter = Filter(must=musts + [FieldCondition(key="micrograph_like", match=MatchValue(value=True))])
                    hits_img = _qdrant_search(self.client, qv_text, pool_size, strict_filter, CFG.SCORE_THRESHOLD, using="image")
            else:
                hits_img = _qdrant_search(self.client, qv_text, pool_size, base_filter, CFG.SCORE_THRESHOLD, using="image")
        else:
            hits_img = []

        # --- Merge & dedupe (keep highest similarity per page)
        merged = {}  # key -> (score, payload, via)
        def _derive_docid_from_image_path(pl: dict):
            raw = pl.get("image_path")
            if raw:
                try:
                    p = Path(raw)
                    if "pages" in p.parts:
                        i = p.parts.index("pages")
                        if i >= 1:
                            return p.parts[i-1]
                except Exception:
                    pass
            return None

        def _safe_docid(pl: dict):
            return pl.get("doc_id") or _derive_docid_from_image_path(pl) or pl.get("uid")

        def _key(pt):
            pl = getattr(pt, "payload", {}) or {}
            uid = pl.get("uid")
            if uid:
                return ("uid", uid)
            doc = _safe_docid(pl)
            pi = pl.get("page_index")
            if doc is not None and pi is not None:
                return ("dpi", doc, pi)
            # ultra-safe fallback
            import json, hashlib
            h = hashlib.sha1(json.dumps(pl, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
            return ("h", h)
        def _add(points, tag):
            for pt in points or []:
                k = _key(pt)
                sc = float(getattr(pt, "score", 0.0))
                pl = getattr(pt, "payload", {}) or {}
                if (k not in merged) or (sc > merged[k][0]):
                    merged[k] = (sc, pl, tag)

        _add(hits_text, "text")
        _add(hits_img,  "db_image")
        
        # NEW: Uploaded image-based retrieval (the key fix!)
        if uploaded_image_embeddings:
            print(f"[INFO] Performing uploaded image-based retrieval...")
            uploaded_hits_all = []
            for i, img_embedding in enumerate(uploaded_image_embeddings):
                try:
                    # Ensure embedding is the right format for Qdrant search (numpy array)
                    if isinstance(img_embedding, np.ndarray):
                        search_embedding = img_embedding.flatten().astype(np.float32)
                    else:
                        search_embedding = np.array(img_embedding, dtype=np.float32)
                    
                    print(f"[DEBUG] Using embedding of type {type(search_embedding)} with shape {search_embedding.shape}")
                    
                    # Search using uploaded image embedding
                    hits_uploaded = _qdrant_search(self.client, search_embedding, pool_size, base_filter, CFG.SCORE_THRESHOLD, using="image")
                    uploaded_hits_all.extend(hits_uploaded)
                    print(f"[INFO] Uploaded image {i+1} retrieval: {len(hits_uploaded)} results")
                except Exception as e:
                    print(f"[WARN] Failed to search with uploaded image {i+1}: {e}")
                    print(f"[DEBUG] Embedding details - type: {type(img_embedding)}, shape: {getattr(img_embedding, 'shape', 'no shape')}")
                    import traceback
                    print(f"[DEBUG] Full traceback: {traceback.format_exc()}")
            
            # Add uploaded image results with higher weight since they're user-provided
            def _add_weighted(points, tag, weight=1.0):
                for pt in points or []:
                    k = _key(pt)
                    sc = float(getattr(pt, "score", 0.0)) * weight
                    pl = getattr(pt, "payload", {}) or {}
                    if (k not in merged) or (sc > merged[k][0]):
                        merged[k] = (sc, pl, tag)
            
            _add_weighted(uploaded_hits_all, "uploaded_image", weight=1.3)
            print(f"[INFO] Total uploaded image results: {len(uploaded_hits_all)}")

        # Convert to dicts
        raw_items: List[Dict[str, Any]] = []
        for _, (score, pl, via) in merged.items():
            doc_fixed = pl.get("doc_id") or _derive_docid_from_image_path(pl)
            raw_items.append({
                "rank": 0,
                "score": float(score),
                "doc_id": doc_fixed,
                "page_index": pl.get("page_index"),
                "image_path": pl.get("image_path"),
                "case_type": pl.get("case_type"),
                "page_kind": pl.get("page_kind"),
                "micrograph_like": pl.get("micrograph_like"),
                "keywords": (pl.get("keywords") or [])[:6],
                "text_excerpt": pl.get("text_excerpt"),
                "via": via,
            })
        
        for item in raw_items:
            page_idx = item.get("page_index", 999)
            base_score = item["score"]
            
            # NEW: Boost results from uploaded image searches (key multimodal enhancement)
            if item.get("via") == "uploaded_image":
                item["score"] = base_score * 1.25  # Significant boost for user-uploaded image matches
                print(f"[DEBUG] Boosted uploaded image match: {item.get('doc_id')} page {page_idx}")
            
            # Apply question-aware page boosting and content-based boosting
            if treatment_question or diagnostic_question:
                # Boost later pages for treatment/diagnostic info
                if page_idx >= 3:
                    item["score"] = base_score * 1.15
                elif page_idx == 0:
                    item["score"] = base_score * 0.85  # Slight penalty for first page
            elif clinical_question:
                # Keep early page boost for clinical presentation
                if page_idx <= 2:
                    item["score"] = base_score * 1.1
            
            # Post-retrieval content boosting (replaces some Filter.should functionality)
            if imaging_question and item.get("micrograph_like") is True:
                item["score"] *= 1.08
            if clinical_question and "clinical" in (item.get("keywords") or []):
                item["score"] *= 1.05
            if diagnostic_question and any(term in (item.get("keywords") or []) for term in ["pcr", "culture", "molecular"]):
                item["score"] *= 1.12

        # Pre-sort by similarity, trim pool
        raw_items.sort(key=lambda r: r["score"], reverse=True)
        raw_items = raw_items[:pool_size]

        # Optional soft micrograph preference
        if micrograph_only and not micrograph_strict and raw_items:
            prefer = [r for r in raw_items if r.get("micrograph_like") is True]
            others = [r for r in raw_items if not r.get("micrograph_like")]
            raw_items = (prefer + others)[:pool_size]

        # Clinical early-page fallback if doc-scoped recall is thin
        if clinical_question and doc_id and len(raw_items) < max(1, top_k // 2):
            early_filter = Filter(
                must=musts + [FieldCondition(key="page_index", match=MatchAny(any=[0, 1]))]
            )
            early_hits = _qdrant_search(self.client, qv_text, 4, early_filter, None, using="text")
            seen = {((r.get("doc_id") or _derive_docid_from_image_path(r)), r.get("page_index")) for r in raw_items}
            for eh in early_hits or []:
                pl = getattr(eh, "payload", {}) or {}
                key = (pl.get("doc_id"), pl.get("page_index"))
                if key not in seen:
                    raw_items.append({
                        "rank": 0,
                        "score": float(getattr(eh, "score", 0.0)),
                        "doc_id": pl.get("doc_id"),
                        "page_index": pl.get("page_index"),
                        "image_path": pl.get("image_path"),
                        "case_type": pl.get("case_type"),
                        "page_kind": pl.get("page_kind"),
                        "micrograph_like": pl.get("micrograph_like"),
                        "keywords": (pl.get("keywords") or [])[:6],
                        "text_excerpt": pl.get("text_excerpt"),
                        "via": "early_boost",
                    })
                    seen.add(key)

        # --- Apply appropriate reranking strategy based on constraints
        print(f"[DEBUG] After merge: {len(raw_items)} items")
        for i, item in enumerate(raw_items[:5]):
            print(f"[DEBUG] Item {i+1}: {item.get('doc_id', 'N/A')} page {item.get('page_index', 'N/A')} score={item.get('score', 0):.4f}")
        
        pre_rerank_pool = [dict(r) for r in raw_items[:20]]  # Always store pre-rerank state
        post_rerank_pool = []  # Initialize to avoid undefined variable issues
        if self.reranker and raw_items:
            try:
                # Store pre-rerank for metrics
                # pre_rerank_pool already set above
                
                if doc_id:
                    # Single document: use page-level reranking (case-level would be no-op)
                    print(f"[INFO] Using page-level reranking for single doc: {doc_id}")
                    reranked_items = rerank_with_text(
                        question, raw_items, self.reranker,
                        alpha=0.5 if clinical_question else CFG.RERANK_ALPHA,
                        min_excerpt_chars=CFG.RERANK_MIN_EXCERPT_CHARS,
                        fallback_alpha=CFG.RERANK_FALLBACK_ALPHA
                    )
                else:
                    # Multiple documents: use case-level reranking
                    print(f"[INFO] Using case-level reranking across multiple docs")
                    reranked_items = rerank_with_case_level_cross_encoder(
                        question, raw_items, self.reranker,
                        top_cases=5,
                        top_pages_per_case=3
                    )
                
                # Verify reranking actually changed something
                if reranked_items:
                    # Check if order changed
                    pre_ids = [r.get("page_index") for r in raw_items[:10]]
                    post_ids = [r.get("page_index") for r in reranked_items[:10]]
                    if pre_ids != post_ids:
                        print(f"[INFO] Reranking changed order: {pre_ids[:5]} -> {post_ids[:5]}")
                    # Store immediate reranker output for metrics
                    post_rerank_pool = [dict(r) for r in reranked_items[:20]]
                    raw_items = reranked_items
                else:
                    print(f"[WARN] Reranking returned empty, keeping original")
                    post_rerank_pool = [dict(r) for r in raw_items[:20]]
                    
            except Exception as e:
                print(f"[WARN] Reranking failed: {e}, keeping original hits")
                post_rerank_pool = [dict(r) for r in raw_items[:20]]  # Ensure post_rerank_pool is always set
                
        else:
            # No reranker used, set post_rerank_pool to raw_items for consistent metrics
            post_rerank_pool = [dict(r) for r in raw_items[:20]]
                
            
        # --- Final selection with clinical bias and diversity enforcement
        # Use stable preference sorting that preserves reranker order within groups
        if clinical_question:
            # Use stable sort keys that group desired items first, without reordering within groups
            def bias_tuple(r):
                page_ok = 0 if isinstance(r.get("page_index"), int) and r["page_index"] <= 2 else 1
                text_ok = 0 if len((r.get("text_excerpt") or "")) > 100 else 1
                return (page_ok, text_ok)

            candidates = sorted(raw_items, key=bias_tuple)
        else:
            candidates = raw_items
        
        # Enforce diversity to prevent same-page repetition and similar content
        selected = []
        seen_pages = set()
        seen_excerpts = set()
        
        for candidate in candidates:
            if len(selected) >= top_k:
                break
                
            # Diversity checks
            page_key = (candidate.get("doc_id"), candidate.get("page_index"))
            excerpt = (candidate.get("text_excerpt") or "").strip()[:200]  # First 200 chars for similarity
            
            # Skip if we've seen this exact page
            if page_key in seen_pages:
                continue
                
            # Skip if we've seen very similar content (helps prevent duplicate evidence)
            excerpt_norm = _re.sub(r'\W+', '', excerpt.lower())
            if len(excerpt_norm) > 50:  # Only check substantial excerpts
                is_duplicate = False
                for seen_excerpt in seen_excerpts:
                    if len(seen_excerpt) > 50:
                        # Check for high similarity (>80% overlap)
                        import difflib
                        similarity = difflib.SequenceMatcher(None, excerpt_norm, seen_excerpt).ratio()
                        if similarity > 0.8:
                            is_duplicate = True
                            break
                
                if is_duplicate:
                    print(f"[DEBUG] Skipping duplicate content from {candidate.get('doc_id')} page {candidate.get('page_index')}")
                    continue
                    
                seen_excerpts.add(excerpt_norm)
            
            selected.append(candidate)
            seen_pages.add(page_key)

        # If we don't have enough diverse results, fill with best remaining (relaxed diversity)
        if len(selected) < top_k:
            for candidate in candidates:
                if candidate not in selected and len(selected) < top_k:
                    selected.append(candidate)

        # set ranks
        for i, r in enumerate(selected, 1):
            r["rank"] = i

        def _compute_rerank_impact(pre_hits: List[Dict], post_hits: List[Dict], k: int = 5) -> Dict[str, float]:
            """Compute metrics showing reranker impact"""
            pre_pages = [(h.get("doc_id"), h.get("page_index")) for h in pre_hits[:k]]
            post_pages = [(h.get("doc_id"), h.get("page_index")) for h in post_hits[:k]]
            
            # Position changes
            position_changes = []
            for i, page in enumerate(post_pages):
                if page in pre_pages:
                    old_pos = pre_pages.index(page)
                    position_changes.append(abs(old_pos - i))
            
            # Set overlap
            pre_set = set(pre_pages)
            post_set = set(post_pages)
            overlap = len(pre_set & post_set) / max(1, len(pre_set))
            
            return {
                "avg_position_change": sum(position_changes) / max(1, len(position_changes)),
                "overlap_ratio": overlap,
                "new_pages_ratio": 1.0 - overlap
            }

        result = {
            "mode": "text", 
            "question": question, 
            "hits": selected,
            "retrieval_diagnostics": {
                "pool_size": pool_size,
                "pre_filter_count": len(merged),
                "post_rerank_count": len(selected),
                "imaging_question": imaging_question,
                "clinical_question": clinical_question,
                "treatment_question": treatment_question,
                "diagnostic_question": diagnostic_question,
                "used_reranker": self.reranker is not None,
                "reranker_type": "page-level" if doc_id else "case-level",
                "doc_constraint": doc_id is not None,
                "rerank_impact": _compute_rerank_impact(
                    pre_rerank_pool,
                    post_rerank_pool[:top_k],
                    k=min(5, len(post_rerank_pool))
                )
            }
        }
        
        # Add pre-rerank pool if available
        # Store for metrics
        result["pre_rerank_pool"] = pre_rerank_pool
        
        return result
    
    def answer_with_images(self, question: str, image_paths: List[str], 
                           hits: List[Dict[str, Any]],
                           images_per_answer: int = 3) -> str:
        """
        Answer using Gemini with enhanced context, evidence spans, and clinical detail validation.
        Now with quiet PyMuPDF + pdfminer fallback to avoid MuPDF ICC crashes/noise.
        Enhanced with question-specific context building to reduce duplication.
        """
        paths = [Path(p) for p in image_paths if Path(p).exists()]
        if not paths:
            return "No valid images provided."

        # Add question hash to context for uniqueness across different questions
        question_hash = hashlib.md5(question.encode()).hexdigest()[:8]
        print(f"[DEBUG] Question hash: {question_hash}")
        
        # Evidence spans (more per doc for coverage) - now question-aware
        spans = extractive_spans(hits, per_doc=6, max_chars=500, question=question) # Enhanced with question context
        print(f"[DEBUG] Extracted {len(spans)} spans for question analysis")

        # Build augmented context: PDF early pages first, then ranked excerpts
        context_parts: List[str] = []

        # Detect question types for context strategy
        treatment_question = any(term in question.lower() for term in [
            "treatment", "dose", "dosage", "regimen", "therapy", "outcome", 
            "follow-up", "response", "mg", "kg", "administered"
        ])
        diagnostic_question = any(term in question.lower() for term in [
            "diagnosis", "identify", "stain", "microscopy", "species", 
            "pcr", "culture", "molecular", "sequencing", "identification", "specimens", "methods"
        ])
        clinical_question = any(term in question.lower() for term in [
            "lesion", "clinical", "course", "onset", "features",
            "history", "size", "location", "evolution", "appearance", "referral"
        ])
        histopath_question = any(term in question.lower() for term in [
            "histopathology", "histopathologic", "findings", "biopsy", "inflammatory", "amastigotes"
        ])

        # Enhanced question type detection for better context building - DEFINE BEFORE USE
        diagnostic_all = any(term in question.lower() for term in [
            "diagnosis", "diagnose", "identify", "disease", "condition", "what is", "which disease", "having"
        ])
        procedure_question = any(term in question.lower() for term in [
            "procedure", "cure", "how to treat", "treatment steps", "therapy", "management"
        ])

        # Add question-specific context prefix with enhanced discrimination
        if procedure_question:
            context_parts.append(f"TREATMENT/PROCEDURE ANALYSIS for: {question}")
            context_parts.append("Focus on therapeutic interventions, treatment protocols, and management strategies.")
        elif diagnostic_all:
            context_parts.append(f"DIAGNOSTIC ANALYSIS for: {question}")
            context_parts.append("Focus on diagnostic criteria, identification methods, and differential diagnoses.")
        elif clinical_question:
            context_parts.append(f"CLINICAL PRESENTATION ANALYSIS for: {question}")
            context_parts.append("Focus on clinical features, symptoms, and patient presentation.")
        else:
            context_parts.append(f"MEDICAL ANALYSIS for: {question}")
        
        context_parts.append(f"Question ID: {question_hash}")
        print(f"[DEBUG] Question type - Diagnostic: {diagnostic_all}, Treatment: {treatment_question}, Procedure: {procedure_question}, Clinical: {clinical_question}, Histopath: {histopath_question}")
        
        # Filter hits based on question relevance to build focused context
        filtered_hits = []
        for hit in hits[:15]:  # Increase pool for better selection
            excerpt = (hit.get("text_excerpt") or "").strip()
            if not excerpt or len(excerpt) < 30:  # Skip very short excerpts
                continue
            
            excerpt_lower = excerpt.lower()
            relevance_score = 0
            
            # Enhanced relevance scoring for different question types
            if diagnostic_all:
                # For diagnostic questions, prioritize medical findings and methods
                diagnostic_terms = [
                    "diagnosis", "identified", "species", "microscopy", "pcr", "culture", 
                    "biopsy", "histopathology", "staining", "morphology", "amastigotes",
                    "leishmania", "cutaneous", "visceral", "mucocutaneous", "parasites"
                ]
                relevance_score += sum(3 for term in diagnostic_terms if term in excerpt_lower)
                # Penalize case descriptions for diagnostic questions
                if any(phrase in excerpt_lower for phrase in [
                    "year-old", "from laos", "from cambodia", "progressed quickly"
                ]):
                    relevance_score -= 2
            
            if clinical_question:
                clinical_terms = ["patient", "lesion", "nodule", "ulcer", "clinical", "presentation", "onset", "course", "size", "location", "symptoms"]
                relevance_score += sum(2 for term in clinical_terms if term in excerpt_lower)
            
            if treatment_question:
                treatment_terms = ["treatment", "therapy", "dose", "antimony", "amphotericin", "outcome", "response", "cure", "healing"]
                relevance_score += sum(3 for term in treatment_terms if term in excerpt_lower)
            
            if histopath_question:
                histopath_terms = ["histopathology", "biopsy", "inflammatory", "amastigotes", "macrophages", "granuloma", "tissue", "cells"]
                relevance_score += sum(3 for term in histopath_terms if term in excerpt_lower)
            
            # Bonus for medical terminology and scientific content
            medical_bonus_terms = ["microscopic", "molecular", "endemic", "epidemiological", "pathogenesis", "immunology"]
            relevance_score += sum(1 for term in medical_bonus_terms if term in excerpt_lower)
            
            if relevance_score > 0:
                hit_copy = hit.copy()
                hit_copy["context_relevance"] = relevance_score
                filtered_hits.append(hit_copy)
        
        # Sort by relevance and take the most relevant
        filtered_hits.sort(key=lambda h: h.get("context_relevance", 0), reverse=True)
        selected_hits = filtered_hits[:8] if filtered_hits else hits[:8]

        # 1) Early-page PDF context (p.1–5), suppressing MuPDF stderr
        def _strip_biblio_noise(txt: str) -> str:
            import re as _re
            # Drop common headers: ISSN/DOI/Volume/Received/Accepted/Published/Correspondence/Open Access/CC license
            lines = []
            for line in txt.splitlines():
                l = line.strip()
                if _re.search(r"^(issn|doi|volume|issue|received|accepted|published|open access|creative commons|copyright|correspondence)[:\s]", l, _re.I):
                    continue
                lines.append(line)
            return " ".join(lines)
        doc_id_for_pdf = next((h.get("doc_id") for h in hits if h.get("doc_id")), None)
        if doc_id_for_pdf:
            case_dir = find_case_dir(doc_id_for_pdf, CFG.EXTRACT_ROOT)
            if case_dir:
                try:
                    import contextlib, io
                    pdf = find_case_pdf(case_dir)
                    if pdf:
                        buf = io.StringIO()
                        with contextlib.redirect_stderr(buf):
                            first_pages = []
                            for i in range(5):
                                t = read_pdf_page_text(pdf, i)
                                if t: first_pages.append(t)
                        early_text = _strip_biblio_noise(" ".join(first_pages).strip())
                        if early_text:
                            context_parts.append(f"[{doc_id_for_pdf} p.1-5] {early_text[:3000]}")
                except Exception as e:
                    print(f"[WARN] PDF context extraction failed: {e}")

        # 1b) Targeted later-page PDF context for treatment/diagnostic
        if (treatment_question or diagnostic_question) and doc_id_for_pdf:
            case_dir = find_case_dir(doc_id_for_pdf, CFG.EXTRACT_ROOT)
            if case_dir:
                try:
                    import contextlib, io
                    pdf = find_case_pdf(case_dir)
                    if pdf:
                        buf = io.StringIO()
                        with contextlib.redirect_stderr(buf):
                            # choose up to 3 ranked later pages (>= page 3 / index >=2) from selected hits
                            later_idxs = []
                            for h in selected_hits:
                                pi = h.get("page_index")
                                if isinstance(pi, int) and pi >= 2 and pi not in later_idxs:
                                    later_idxs.append(pi)
                                if len(later_idxs) >= 3:
                                    break
                            for idx in later_idxs:
                                t = read_pdf_page_text(pdf, idx)
                                if t:
                                    context_parts.append(f"[{doc_id_for_pdf} p.{idx+1}] {t[:1200]}")
                except Exception as e:
                    print(f"[WARN] PDF later-page context extraction failed: {e}")

        # 2) Use question-specific selected hits for context
        early_hits = [h for h in selected_hits if h.get("page_index", 999) <= 2]
        other_hits = [h for h in selected_hits if h.get("page_index", 999) > 2]

        def _truncate_preserving_medical(excerpt: str, limit: int = 1000) -> str:
            """Enhanced truncation that preserves important medical content"""
            if len(excerpt) <= limit:
                return excerpt
            import re as _re  # Ensure _re is available
            
            # For diagnostic questions, prioritize diagnostic content
            if diagnostic_all:
                diagnostic_patterns = [
                    r'[^.]*(?:diagnosis|identified|species|amastigotes|leishmania|pcr|culture|microscopy)[^.]*\.',
                    r'[^.]*(?:biopsy|histopathology|staining|morphology|parasites)[^.]*\.',
                    r'[^.]*(?:cutaneous|visceral|mucocutaneous)[^.]*\.'
                ]
                for pattern in diagnostic_patterns:
                    m = _re.search(pattern, excerpt, _re.I)
                    if m:
                        start_pos = max(0, m.start() - 300)
                        return excerpt[start_pos:start_pos + limit] + "..."
            
            # For clinical questions, preserve specific findings
            clinical_match = _re.search(r'[^.]*(?:lesion|nodule|ulcer|crust|onset|month|cm|mm|diameter|clinical|symptoms)[^.]*\.', excerpt, _re.I)
            if clinical_match:
                start_pos = max(0, clinical_match.start() - 200)
                return excerpt[start_pos:start_pos + limit] + "..."
            
            # Default truncation from beginning
            return excerpt[:limit] + "..."

        # Prioritize hits based on question relevance score and avoid repetitive content
        all_selected = sorted(early_hits + other_hits, key=lambda h: h.get("context_relevance", 0), reverse=True)
        seen_excerpts = set()
        
        for h in all_selected[:10]:  # Limit to top 10 most relevant
            excerpt = (h.get("text_excerpt") or "").strip()
            if not excerpt:
                continue
                
            # Skip if we've seen very similar content
            import re as _re  # Ensure _re is available in this scope
            excerpt_key = _re.sub(r"\W+", "", excerpt.lower())[:100]
            if excerpt_key in seen_excerpts:
                continue
            seen_excerpts.add(excerpt_key)
            
            # Skip excerpts that are mostly case descriptions for diagnostic questions
            if diagnostic_all and any(phrase in excerpt.lower() for phrase in [
                "year-old boy from laos", "year-old man from cambodia", 
                "progressed quickly from a sore to eat through"
            ]):
                continue
            
            page_num = h.get("page_index", -1) + 1
            # Remove relevance note for cleaner context
            truncated_excerpt = _truncate_preserving_medical(excerpt)
            context_parts.append(f"[{h.get('doc_id','unknown')} p.{page_num}] {truncated_excerpt}")

        # 3) Assemble focused medical context (larger for diagnostic questions)
        max_context_length = 18000 if diagnostic_all else 15000
        ctx = " ".join(context_parts)[:max_context_length]
        
        # For diagnostic questions, add a brief medical context primer
        if diagnostic_all and ctx:
            ctx = "Medical diagnostic context - focus on identifying disease characteristics, diagnostic methods, and clinical findings:\n\n" + ctx

        # Clinical QA heuristics (optional safety checks)
        clinical_question = any(term in question.lower() for term in [
            "lesion", "clinical", "course", "onset", "features",
            "history", "size", "location", "evolution", "appearance"
        ])

        try:
            # FIXED: Simple question formatting - no complex instructions
            enhanced_question = question
            if diagnostic_all:
                # Simple diagnostic question - let the model's training handle the format
                enhanced_question = f"What is the diagnosis for this medical condition? {question}"
            
            # Add question uniqueness markers to avoid cached responses
            answer = self.gem.answer(
                enhanced_question, [str(p) for p in paths], spans=spans, context_text=ctx,
                max_output_tokens=max(1024, CFG.MAX_NEW_TOKENS), images_per_answer=images_per_answer
            )
            print(f"[DEBUG] Original question: {repr(question[:100])}")
            print(f"[DEBUG] Enhanced question: {repr(enhanced_question[:200])}")
            print(f"[DEBUG] MedGemma answer before normalize: {repr(answer[:500])}")
            answer = _normalize_answer(answer)
            print(f"[DEBUG] Answer after normalize: {repr(answer[:500])}")
            
            # ENHANCED: Handle "insufficient details" responses when image processing fails
            if answer and ("insufficient details" in answer.lower() or 
                          "not certain due to lack" in answer.lower() or
                          "consider performing other relevant test" in answer.lower()):
                print(f"[INFO] Detected insufficient details response, attempting evidence-based answer")
                
                # Build a better response using available evidence
                evidence_text = ""
                if spans:
                    # Use the best evidence spans to provide a helpful response
                    relevant_evidence = []
                    for span_text, citation in spans[:3]:
                        if len(span_text.strip()) > 30:
                            relevant_evidence.append(span_text.strip())
                    
                    if relevant_evidence:
                        evidence_text = " ".join(relevant_evidence)[:800]  # Limit length
                        
                        # Try to generate a better answer using the evidence
                        try:
                            better_answer = self.gem.answer(
                                f"Based on this medical evidence, answer: {question}\n\nEvidence: {evidence_text}",
                                [str(p) for p in paths[:1]],  # Use just one image to reduce complexity
                                spans=[],
                                context_text="",
                                max_output_tokens=200,
                                images_per_answer=1
                            )
                            
                            better_normalized = _normalize_answer(better_answer)
                            
                            # Use the better answer if it's actually better
                            if (better_normalized and 
                                len(better_normalized) > 50 and
                                "insufficient details" not in better_normalized.lower() and
                                better_normalized != answer):
                                answer = better_normalized
                                print(f"[INFO] Used evidence-based fallback answer: {repr(answer[:100])}")
                        except Exception as e:
                            print(f"[WARN] Evidence-based fallback failed: {e}")
                
                # If we still have insufficient details and no good evidence, provide helpful guidance
                if "insufficient details" in answer.lower() and not evidence_text:
                    answer = ("The uploaded image quality may be insufficient for detailed medical analysis. "
                             "For accurate diagnosis, please provide: 1) Higher resolution images, "
                             "2) Multiple views of the affected area, 3) Clinical history and symptoms.")
            
            # Enhanced evidence-as-answer detection and prevention
            if answer and spans:
                # Check if answer is just a concatenation of our spans
                first_spans_text = " ".join([s for s, _ in spans[:3]]).strip().lower()
                answer_lower = answer.lower()
                
                # Multiple checks for evidence regurgitation
                is_evidence_dump = False
                
                # Check 1: Direct span concatenation
                if len(first_spans_text) >= 80 and (
                    answer_lower.startswith(first_spans_text[:80]) or
                    first_spans_text[:80] in answer_lower[:100]
                ):
                    is_evidence_dump = True
                
                # Check 2: Answer starts with evidence-like patterns
                evidence_patterns = [
                    r"^(?:insufficient evidence|key evidence|from the evidence|evidence summary)",
                    r"^\[?\d+\]?\s*[A-Z][^.]{20,}\.?\s*\[?\d+\]?",  # Looks like cited evidence
                    r"^(?:The|This)\s+(?:case|patient|lesion).{20,}(?:from|in)\s+(?:Laos|Cambodia|Peru)"  # Case descriptions
                ]
                
                for pattern in evidence_patterns:
                    if _re.search(pattern, answer, _re.I):
                        is_evidence_dump = True
                        break
                
                # Check 3: Answer is mostly just reformatted spans
                import difflib
                answer_words = set(_re.findall(r'\w+', answer_lower))
                span_words = set(_re.findall(r'\w+', first_spans_text))
                if len(answer_words) > 10 and len(span_words) > 10:
                    overlap_ratio = len(answer_words & span_words) / len(answer_words)
                    if overlap_ratio > 0.8:  # 80% word overlap suggests evidence regurgitation
                        is_evidence_dump = True
                
                if is_evidence_dump:
                    print(f"[WARN] Detected evidence regurgitation, attempting simplified retry")
                    # Retry with a much simpler, more direct prompt
                    try:
                        retry_answer = self.gem.answer(
                            f"Provide a direct medical diagnosis or answer for: {question}",
                            [str(p) for p in paths[:2]],  # Fewer images to reduce confusion
                            spans=[],  # No spans to avoid regurgitation
                            context_text="",  # Minimal context
                            max_output_tokens=300,
                            images_per_answer=min(2, images_per_answer)
                        )
                        retry_normalized = _normalize_answer(retry_answer)
                        
                        # Check if retry was successful (not just more evidence)
                        if (retry_normalized and 
                            len(retry_normalized) > 30 and
                            not _re.search(r"^(?:insufficient|from the evidence|key evidence)", retry_normalized.lower()) and
                            retry_normalized.lower() not in first_spans_text):
                            answer = retry_normalized
                            print(f"[INFO] Successful retry produced: {repr(answer[:100])}")
                        else:
                            # If retry also fails, provide a clear failure message
                            answer = "Unable to generate a reliable medical answer from the available case information."
                            print(f"[WARN] Retry also produced evidence regurgitation, using failure message")
                    except Exception as e:
                        print(f"[ERROR] Retry failed: {e}")
                        answer = "Medical answer generation failed - unable to process case information effectively."

            if clinical_question and answer:
                import re as _re
                evidence_lower = " ".join([span[0] for span in spans]).lower()
                unsupported_patterns = [
                    r'\b(?:October|November|December|January|February|March|April|May|June|July|August|September)\s+\d{4}\b',
                    r'\bleft\s+cheek\b|\bright\s+cheek\b',
                    r'\d+\s*(?:mm|cm)\s+(?:diameter|size|lesion)',
                ]
                for pattern in unsupported_patterns:
                    if _re.search(pattern, answer, _re.I) and not _re.search(pattern, evidence_lower, _re.I):
                        print(f"[WARN] Potential unsupported detail in answer: {pattern}")

            # Heuristic: incomplete-or-truncated answer detection
            def _runner_looks_incomplete(txt: str) -> bool:
                """Enhanced detection of truncated answers."""
                if not txt:
                    return True
                import re as _re
                s = txt.strip()
                
                # No terminal punctuation
                if not _re.search(r"[.!?]\"?$", s):
                    return True
                
                # Bad endings
                if _re.search(r"\b(by|with|including|include|such as|that|because|due to|via|as|which|where|when|to|of|for|in|on|and|or)\.$", s.lower()):
                    return True
                
                # Ends with colon
                if s.lower().endswith(":"):
                    return True
                
                # Ellipsis or etc.
                if _re.search(r'(?:\.\.\.|…|etc\.?)\s*$', s.lower()):
                    return True
                
                # NEW: Detect mid-word truncation (e.g., "se.")
                last_token = _re.findall(r"([A-Za-z]{1,3})[.!?]\"?$", s)
                if last_token:
                    tok = last_token[-1].lower()
                    # Exclude valid abbreviations
                    if tok not in {"mg","ml","kg","iv","im","po","cl","vl","dl","ct","mr","cm","mm","dr","mr","ms","mrs"}:
                        return True
                
                return False

            # Retry once with expanded context if insufficient
            if answer.strip().lower().startswith("insufficient evidence"):
                # For diagnostic questions, try a different approach
                if diagnostic_all:
                    # Retry with more targeted diagnostic prompting
                    diagnostic_retry_question = (
                        f"Based on the available medical evidence and images, what can you determine about: {question}\n\n"
                        "Even with limited information, please provide:\n"
                        "- Any visible clinical findings\n"
                        "- Possible diagnostic considerations\n"
                        "- What additional information would be helpful\n\n"
                        "Use only the evidence provided."
                    )
                else:
                    diagnostic_retry_question = question
                
                # Expand context using all evidence spans and more excerpts
                more_ctx_parts = list(context_parts)
                if spans:
                    span_text = " ".join([s for s, _ in spans])
                    more_ctx_parts.append("Key Evidence Summary: " + span_text[:6000])
                
                # Add more relevant excerpts (avoid repetitive ones)
                added_excerpts = 0
                for h in selected_hits[:8]:
                    if added_excerpts >= 4:  # Limit additional context
                        break
                    ex = (h.get("text_excerpt") or "").strip()
                    if ex and len(ex) > 100:  # Only substantial excerpts
                        # Skip if it's mainly case description
                        if not any(desc in ex.lower() for desc in [
                            "year-old", "progressed quickly", "from laos", "from cambodia"
                        ]):
                            page_num = h.get("page_index", -1) + 1
                            more_ctx_parts.append(f"[{h.get('doc_id','unknown')} p.{page_num}] {ex[:1500]}")
                            added_excerpts += 1
                
                ctx2 = " ".join(more_ctx_parts)[:18000]
                answer2 = self.gem.answer(
                    diagnostic_retry_question, [str(p) for p in paths], spans=spans, context_text=ctx2,
                    max_output_tokens=CFG.MAX_NEW_TOKENS, images_per_answer=images_per_answer
                )
                answer2 = _normalize_answer(answer2)
                if answer2 and not answer2.strip().lower().startswith("insufficient evidence"):
                    answer = answer2

            # Retry once with continuation if incomplete
            if _runner_looks_incomplete(answer):
                if diagnostic_all:
                    # For diagnostic questions, try to complete the analysis
                    cont_prompt = (
                        f"Continue the diagnostic analysis: {answer.rstrip()}\n\n"
                        "Please complete the medical assessment by providing:\n"
                        "- Final diagnostic conclusion\n"
                        "- Supporting evidence from the case\n"
                        "- Any relevant clinical considerations"
                    )
                else:
                    # Use a continuation prompt that seeds with the incomplete part
                    cont_prompt = answer.rstrip() + " ..."
                
                continuation = self.gem.answer(
                    cont_prompt,
                    [str(p) for p in paths],
                    spans=spans,
                    context_text=ctx,
                    max_output_tokens=min(512, CFG.MAX_NEW_TOKENS//2),  # Allow more tokens for completion
                    images_per_answer=images_per_answer
                )
                continuation = _normalize_answer(continuation)
                
                if continuation and len(continuation) > 10:
                    # Smart merge without duplication
                    if diagnostic_all:
                        # For diagnostic questions, append as separate analysis
                        answer = (answer.rstrip(".") + " " + continuation).strip()
                    else:
                        # Standard continuation logic
                        if answer.endswith(".") and continuation.startswith(answer[-50:]):
                            answer = answer[:-1] + continuation[len(answer[-50:]):]
                        else:
                            answer = (answer.rstrip(".") + " " + continuation).strip()
                    
                    # Final normalization
                    answer = _normalize_answer(answer)

            return answer
        except Exception as e:
            print(f"[ERROR] MedGemma generation failed: {e}")
            return "Unable to generate answer due to processing error."

def derive_doc_id_from_row(row_in: Dict[str, Any]) -> Optional[str]:
    """
    Prefer explicit doc_id; else derive from case_id -> case_dir.name; else parse from seed_image_paths.
    Fail-closed (return None) if nothing resolvable.
    """
    doc_id = (row_in.get("doc_id") or "").strip()
    if doc_id:
        return doc_id

    # 1) Try case_id → case_dir.name (with fuzzy/diacritics-insensitive fallback)
    case_id = (row_in.get("case_id") or "").strip()
    if case_id:
        try:
            case_dir = find_case_dir(case_id, CFG.EXTRACT_ROOT)
            if case_dir:
                return case_dir.name
        except Exception:
            pass
        # NEW: fuzzy fallback across EXTRACT_ROOT dirs
        try:
            import unicodedata, difflib
            def _norm(s: str) -> str:
                s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
                return _re.sub(r"\W+", " ", s).strip().lower()
            want = _norm(case_id)
            candidates = []
            for d in Path(CFG.EXTRACT_ROOT).iterdir():
                if d.is_dir():
                    score = difflib.SequenceMatcher(None, want, _norm(d.name)).ratio()
                    candidates.append((score, d.name))
            if candidates:
                candidates.sort(reverse=True)
                if candidates[0][0] >= 0.72:
                    return candidates[0][1]
        except Exception:
            pass

    # 2) Parse from seed_image_paths if available
    seeds = row_in.get("seed_image_paths") or []
    for sp in seeds:
        try:
            p = Path(sp)
            # Expect: <EXTRACT_ROOT>/<DOC_ID>/pages/page_XXXX.png
            parts = p.parts
            if "pages" in parts:
                i = parts.index("pages")
                if i >= 1:
                    cand = parts[i-1]  # folder just before 'pages' is doc_id
                    if cand:
                        return cand
        except Exception:
            continue

    return None

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Batch answer questions with case-constrained retrieval (resume-safe).")
    ap.add_argument("--manifest", required=True, help="questions_manifest.jsonl")
    ap.add_argument("--out", required=True, help="Output NDJSON path (append-safe with --resume)")
    ap.add_argument("--topk", type=int, default=CFG.TOP_K, help="Top-K retrieval")
    ap.add_argument("--images_per_answer", type=int, default=4, help="Images to feed to Gemini")
    ap.add_argument("--micrograph_only", action="store_true", help="Soft preference for micrograph-like pages")
    ap.add_argument("--micrograph_strict", action="store_true", help="Hard filter to micrograph-like pages")
    ap.add_argument("--score_threshold", type=float, default=None, help="Override CFG.SCORE_THRESHOLD (None => no threshold)")
    ap.add_argument("--case_type", choices=["cutaneous", "mucocutaneous", "visceral", "unknown"])
    ap.add_argument("--keyword", help="Optional keyword payload filter")
    ap.add_argument("--any_keywords", help="Comma-separated OR keywords (payload keywords & entities)")

    # Retrieval evaluation metrics (for analysis/debug)
    ap.add_argument("--compute_metrics", action="store_true", help="Compute retrieval metrics inline")
    ap.add_argument("--metrics_k", type=int, default=10, help="K value for metrics computation")
    ap.add_argument("--min_doc_recall", type=float, default=0.8, help="Minimum acceptable doc recall")
    ap.add_argument("--min_text_coverage", type=float, default=0.5, help="Minimum acceptable text coverage")
    ap.add_argument("--max_dup_ratio", type=float, default=0.3, help="Maximum acceptable duplicate ratio")
    ap.add_argument("--abort_on_low_quality", action="store_true", help="Abort if quality thresholds not met")
    
    # Reranking arguments
    ap.add_argument("--use_reranker", action="store_true", help="Enable BGE cross-encoder reranking")
    ap.add_argument("--pool_mult", type=int, default=3, help="Multiplier for retrieval pool before reranking")

    # Resume/durability arguments
    ap.add_argument("--resume", action="store_true", help="Scan existing --out and skip already answered items")
    ap.add_argument("--retry_errors", action="store_true", help="When used with --resume, reattempt items that previously had 'error'")
    ap.add_argument("--fsync_interval", type=int, default=25, help="Call os.fsync after this many lines")
    ap.add_argument("--retry_insufficient", action="store_true", help="When used with --resume, retry rows whose answer was 'Insufficient evidence.'")

    args = ap.parse_args()

    if args.score_threshold is not None:
        CFG.SCORE_THRESHOLD = args.score_threshold
    
    if args.compute_metrics:
        metrics_accumulator = {
            "mrr": [], "ndcg": [], "recall": [], "precision": [],
            "doc_recall": [], "text_coverage": [], "dup_ratio": [],
            "warnings_count": 0, "low_quality_count": 0
        }
        quality_thresholds = {
            "min_doc_recall": args.min_doc_recall,
            "min_text_coverage": args.min_text_coverage,
            "max_dup_ratio": args.max_dup_ratio,
            "min_unique_pages": 3
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Read existing progress (if any)
    done_keys, err_keys, insuf_keys = _read_existing_out(out_path)
    want_retry_errors = args.resume and args.retry_errors
    want_retry_insuf  = args.resume and args.retry_insufficient

    # Open OUT in append mode if resuming, else write mode.
    mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(mode, encoding="utf-8")
    need_fsync_every = max(1, int(args.fsync_interval))

    # Signal handling: ensure flush on SIGINT/SIGTERM
    exiting = {"flag": False}
    def _graceful(signum, frame):
        exiting["flag"] = True
        try:
            fout.flush()
            os.fsync(fout.fileno())
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _graceful)

    # Initialize the batch processor (loads all models once)
    processor = BatchProcessor(use_reranker=args.use_reranker)

    total = ok = skipped = retried = written_since_fsync = 0

    def write_row(row: Dict[str, Any]) -> None:
        nonlocal written_since_fsync
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        written_since_fsync += 1
        if written_since_fsync >= need_fsync_every:
            os.fsync(fout.fileno())
            written_since_fsync = 0

    print(f"[INFO] Starting batch processing...")

    with open(args.manifest, "r", encoding="utf-8") as fin:
        for ln in fin:
            if exiting["flag"]:
                break
            total += 1
            try:
                row_in = json.loads(ln)
            except Exception as e:
                write_row({"error": "bad_manifest_line", "detail": str(e), "line": ln[:200]})
                continue

            key = _stable_key(row_in)

            # Resume logic
            if args.resume:
                if key in done_keys:
                    skipped += 1
                    continue
                if (not want_retry_errors) and key in err_keys:
                    skipped += 1
                    continue
                if (not want_retry_insuf) and key in insuf_keys:
                    skipped += 1
                    continue
                if want_retry_errors and key in err_keys:
                    retried += 1
                if want_retry_insuf and key in insuf_keys:
                    retried += 1

            qtext = (row_in.get("question") or "").strip()
            doc_id = derive_doc_id_from_row(row_in)
            if not doc_id:
                write_row({"error": "doc_id_unresolvable", **row_in})
                skipped += 1
                continue
            if not qtext:
                write_row({"error": "empty_question", **row_in})
                continue

            try:
                doc_id = derive_doc_id_from_row(row_in)
                seed_images: List[str] = row_in.get("seed_image_paths") or []

                result = processor.enhanced_qdrant_ask_text(
                    question=qtext,
                    doc_id=doc_id,
                    top_k=args.topk,
                    case_type=args.case_type,
                    keyword=args.keyword,
                    any_keywords=args.any_keywords,
                    micrograph_only=args.micrograph_only,
                    micrograph_strict=args.micrograph_strict,
                    pool_multiplier=args.pool_mult
                )
                hits = result["hits"]

                # Compute and store metrics if enabled
                if args.compute_metrics:
                    # Infer gold pages
                    gold_pages = _infer_gold_pages_from_seeds(row_in)
                    
                    # Compute metrics
                    metrics = _compute_retrieval_metrics(
                        hits, doc_id, gold_pages, args.metrics_k
                    )
                    
                    # Check quality
                    quality_ok, warnings = _check_retrieval_quality(metrics, quality_thresholds)
                    
                    # Store metrics in the output record
                    metrics["quality_warnings"] = warnings
                    metrics["quality_pass"] = quality_ok
                    
                    # Add pre/post rerank comparison if available
                    if "pre_rerank_pool" in result:
                        pre_metrics = _compute_retrieval_metrics(
                            result["pre_rerank_pool"], doc_id, gold_pages, args.metrics_k
                        )
                        metrics["pre_rerank_recall@k"] = pre_metrics.get("Recall@k")
                        metrics["rerank_recall_delta"] = round(
                            metrics.get("Recall@k", 0) - pre_metrics.get("Recall@k", 0), 6
                        ) if gold_pages else None
                    
                    # Accumulate for summary
                    if gold_pages:
                        for key in ["MRR@k", "nDCG@k", "Recall@k", "Precision@k"]:
                            if key in metrics:
                                short_key = key.split("@")[0].lower()
                                metrics_accumulator[short_key].append(metrics[key])
                    
                    metrics_accumulator["doc_recall"].append(metrics["doc_recall@k"])
                    metrics_accumulator["text_coverage"].append(metrics["text_coverage@k"])
                    metrics_accumulator["dup_ratio"].append(metrics["duplicate_ratio@k"])
                    
                    if warnings:
                        metrics_accumulator["warnings_count"] += 1
                        print(f"[WARN] Query {total}: {'; '.join(warnings)}")
                    
                    if not quality_ok:
                        metrics_accumulator["low_quality_count"] += 1
                        if args.abort_on_low_quality and metrics_accumulator["low_quality_count"] > 10:
                            print(f"[ABORT] Too many low-quality retrievals ({metrics_accumulator['low_quality_count']})")
                            break

                # Verify doc_id constraint was respected
                if doc_id and any(h.get("doc_id") != doc_id for h in hits):
                    write_row({"error": "retriever_leak_other_doc",
                               "expected_doc_id": doc_id,
                               "got_doc_ids": sorted({h.get('doc_id') for h in hits if h.get('doc_id')}),
                               **row_in})
                    continue

                if not hits:
                    write_row({"error": "no_hits_in_doc", **row_in})
                    continue

                used_images = select_images(seed_images, hits, args.images_per_answer, target_doc_id=doc_id)
                if not used_images:
                    write_row({"error": "no_images_for_doc", "target_doc_id": doc_id, **row_in, "hits": hits})
                    continue

                answer_text = processor.answer_with_images(
                    qtext, used_images, hits, 
                    images_per_answer=args.images_per_answer
                )

                out_rec = {
                    "question_id": row_in.get("question_id"),
                    "case_id": row_in.get("case_id"),
                    "doc_id": doc_id,
                    "question": qtext,
                    "retrieve_mode": row_in.get("retrieve_mode", "pure"),
                    "used_images": used_images,
                    "answer": answer_text,
                    "retrieval_hits": hits,
                }
                if args.compute_metrics and 'metrics' in locals():
                    out_rec["retrieval_metrics"] = metrics
                    out_rec["retrieval_diagnostics"] = result.get("retrieval_diagnostics", {})
                write_row(out_rec)
                ok += 1

                if total % 10 == 0:
                    print(f"[PROGRESS] Processed {total} questions, answered {ok}")

            except Exception as e:
                sys.stderr.write(f"[WARN] failed row {total}: {e}\n")
                try:
                    write_row({"error": f"exception:{type(e).__name__}", "detail": str(e), **row_in})
                except Exception:
                    pass

    try:
        fout.flush()
        os.fsync(fout.fileno())
    except Exception:
        pass
    fout.close()
    if args.compute_metrics and 'metrics_accumulator' in locals():
        def safe_avg(lst):
            return round(sum(lst) / len(lst), 6) if lst else 0.0
        
        summary = {
            "total_queries": total,
            "successful_queries": ok,
            "queries_with_gold_pages": len(metrics_accumulator["mrr"]),
            "metrics": {
                "MRR@k": safe_avg(metrics_accumulator["mrr"]),
                "nDCG@k": safe_avg(metrics_accumulator["ndcg"]),
                "Recall@k": safe_avg(metrics_accumulator["recall"]),
                "Precision@k": safe_avg(metrics_accumulator["precision"]),
                "doc_recall@k": safe_avg(metrics_accumulator["doc_recall"]),
                "text_coverage@k": safe_avg(metrics_accumulator["text_coverage"]),
                "duplicate_ratio@k": safe_avg(metrics_accumulator["dup_ratio"])
            },
            "quality": {
                "warnings_count": metrics_accumulator["warnings_count"],
                "low_quality_count": metrics_accumulator["low_quality_count"],
                "warning_rate": round(metrics_accumulator["warnings_count"] / max(1, total), 4),
                "pass_rate": round(1 - (metrics_accumulator["low_quality_count"] / max(1, total)), 4)
            },
            "thresholds_used": quality_thresholds
        }
        
        # Write summary to file
        summary_path = Path(str(out_path).replace(".ndjson", "") + ".metrics_summary.json")
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(f"\n[METRICS SUMMARY]\n{json.dumps(summary, indent=2)}")

    print(f"[FINAL] Answered {ok}/{total} (skipped={skipped}, retried={retried}) -> {out_path}")

if __name__ == "__main__":
    main()
