#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gemini-only evaluation — streams per-question; tagged outputs per run.

What it does
------------
• Reads gold QA JSONL files from JSONL_DIR (one JSON object per file)
• Reads generated answers from an NDJSON (one JSON object per question)
• Judges each Q with Gemini 2.5 Pro
• Streams each judged question to stream_*.ndjson immediately
• Respects --strategy all|missing (missing reuses ONLY successful cache hits)
• Writes per-file JSON, per-question CSV, and an aggregate JSON
• Folders & cache names are parameterized by a run tag (auto or --run_tag)

Env
---
GOOGLE_API_KEY must be set.
"""

import os, re, json, time, logging, hashlib, unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Any, Tuple, Iterator, Optional

from importlib_metadata import files
import numpy as np
import pandas as pd

from dotenv import load_dotenv
load_dotenv(dotenv_path=Path(".env"), override=True)

# ========================
# Utilities / parsing
# ========================
def load_env_file(path=".env"):
    try:
        for ln in Path(path).read_text(encoding="utf-8").splitlines():
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, v = ln.split("=", 1)
            k, v = k.strip(), v.strip()
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]
            os.environ.setdefault(k, v)
    except FileNotFoundError:
        pass

# call this early:
load_env_file(".env")

def norm_q(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip().lower())

def sha_key(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update((p or "").encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:24]

def parse_case_jsonl(fp: Path) -> Dict[str, Any]:
    """Load a single JSON object either as a whole file or first line JSONL."""
    txt = fp.read_text(encoding="utf-8").strip()
    try:
        o = json.loads(txt)
        if isinstance(o, dict) and "case_id" in o:
            return o
    except Exception:
        pass
    for line in txt.splitlines():
        try:
            o = json.loads(line)
            if isinstance(o, dict) and "case_id" in o:
                return o
        except Exception:
            continue
    return {}

def read_answers_ndjson(path: Path) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """
    Map (case_id, normalized_question) -> record {answer, used_images, retrieval_hits...}
    """
    lut: Dict[Tuple[str, str], Dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except Exception:
                continue
            k = (o.get("case_id"), norm_q(o.get("question")))
            if k[0] and k[1]:
                lut[k] = o
    return lut

# ========================
# Run-tag helpers
# ========================
def _slugify(s: str) -> str:
    """
    Filesystem-safe slug: keep alnum and '_', replace separators with '-', collapse dashes.
    Preserves underscores for values like '0_10'.
    """
    s = unicodedata.normalize("NFKD", str(s))
    out = []
    prev_dash = False
    for ch in s:
        if ch.isalnum() or ch == "_":
            out.append(ch)
            prev_dash = False
        elif ch in "./:\\":
            if not prev_dash:
                out.append("-")
                prev_dash = True
        else:
            if not prev_dash:
                out.append("-")
                prev_dash = True
    slug = "".join(out).strip("-")
    return slug or "x"

def _fmt_val(v) -> str:
    """Stable numeric formatting: 0.10 -> 0_10; general strings -> slug."""
    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        s = f"{v:.4f}".rstrip("0").rstrip(".") or "0"
        return s.replace(".", "_")
    return _slugify(str(v))

def make_run_tag(pairs: Dict[str, object]) -> str:
    """Create a stable 'k1-v1-k2-v2-...' tag (sorted by key)."""
    parts = []
    for k in sorted(pairs.keys()):
        parts.append(_slugify(k))
        parts.append(_fmt_val(pairs[k]))
    return "-".join(parts)

# ========================
# Gemini client with token bucket + persistent cache
# ========================
class TokenBucket:
    def __init__(self, rate_per_min: int):
        self.rate = max(1, int(rate_per_min))
        self.tokens = 1.0  # CHANGED: was float(self.rate) - avoid initial burst
        self.last = time.time()

    def consume(self):
        import random
        now = time.time()
        elapsed = now - self.last
        self.last = now
        self.tokens = min(self.rate, self.tokens + elapsed * (self.rate / 60.0))
        if self.tokens < 1.0:
            sleep_s = (1.0 - self.tokens) * (60.0 / self.rate)
            # ADDED: tiny jitter to avoid thundering herd
            time.sleep(max(0.0, sleep_s) + random.uniform(0, 0.25))
            self.tokens = 0.0
            self.last = time.time()
        self.tokens -= 1.0

class Gemini:
    def __init__(self, api_key: str, model: str, rpm: int, cache_path: Path):
        from google import genai
        from google.genai import types as genai_types
        self.genai_types = genai_types
        self.client = genai.Client(api_key=api_key)
        self.model = model
        self.bucket = TokenBucket(rpm)
        self.cache_path = cache_path
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Normalize legacy cache rows (no 'ok' key) on load
        def _normalize(v: Dict[str, Any]) -> Dict[str, Any]:
            if not isinstance(v, dict):
                return {"ok": False, "error": "bad_cache_row"}
            if "ok" in v:
                return v
            # Heuristic: if we have numeric scores >= 0 and no explicit error, assume ok
            fa = v.get("faithfulness", -1)
            co = v.get("correctness", -1)
            cp = v.get("completeness", -1)
            rat = str(v.get("rationale", ""))
            if rat.startswith("error:") or (fa < 0 and co < 0 and cp < 0):
                return {"ok": False, **v}
            return {"ok": True, **v}

        if cache_path.exists():
            with cache_path.open("r", encoding="utf-8") as f:
                for ln in f:
                    try:
                        o = json.loads(ln)
                        if "key" in o and "val" in o:
                            self._cache[o["key"]] = _normalize(o["val"])
                    except Exception:
                        pass

    def _persist(self, k: str, v: Dict[str, Any]):
        self._cache[k] = v
        try:
            with self.cache_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps({"key": k, "val": v}, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _is_transient_error(e: Exception) -> bool:
        """Enhanced transient error detection with structured fields + text fallback"""
        # First try structured error fields if available
        code = getattr(e, "code", None) or getattr(e, "status", None)
        if isinstance(code, (int, str)):
            code_s = str(code).lower()
            if (code_s in ("resourceexhausted", "resource_exhausted", "unavailable") or 
                code_s.startswith("429") or code_s.startswith("503")):
                return True
        
        # Check for structured retry info (some API clients expose this)
        retry_info = getattr(e, "retry_info", None) or getattr(e, "retryInfo", None)
        if retry_info:
            return True
        
        # Fallback to text parsing - FIXED: include underscored variant and more patterns
        err_txt = str(e).lower()
        transient_patterns = [
            "429", "503", "resource_exhausted", "resourceexhausted", 
            "unavailable", "retryinfo", "retrydelay", "quota exceeded"
        ]
        # Also check for rate limit phrases
        rate_limit_phrases = ["rate limit", "quota exceeded", "too many requests"]
        
        return (any(pattern in err_txt for pattern in transient_patterns) or 
                any(phrase in err_txt for phrase in rate_limit_phrases))
    
    @staticmethod
    def _extract_retry_delay_seconds(e: Exception, err_txt: str = None) -> int:
        """Extract retry delay from structured fields first, then text parsing"""
        if err_txt is None:
            err_txt = str(e)
            
        # Try structured retry info first (if API client exposes it)
        retry_info = getattr(e, "retry_info", None) or getattr(e, "retryInfo", None)
        if retry_info and hasattr(retry_info, "retry_delay"):
            delay = getattr(retry_info.retry_delay, "seconds", None)
            if isinstance(delay, (int, float)) and delay > 0:
                return int(delay)
        
        # Fallback to text parsing
        import re
        # Look for patterns like "retryDelay": "6s" or retryDelay=6s
        m = re.search(r"retrydelay[\"']?\s*[:=]\s*[\"']?(\d+)s", err_txt, re.I)
        delay = int(m.group(1)) if m else 6  # default small backoff
        return delay if delay > 0 else 6

    @staticmethod
    def legacy_key(question: str, answer: str, gold: str, evidence_text: str) -> str:
        # old scheme (unstable): depended on evidence snippet
        return sha_key("gem-only", question, answer, gold, evidence_text[:512])

    def judge(
    self,
    question: str,
    answer: str,
    gold: str,
    evidence_text: str,
    *,
    strategy: str = "all",
    cache_id: str = None,
    ) -> Dict[str, Any]:
        """
        Single-shot call with ONE polite retry for transient errors.
        """
        if cache_id is None:
            # Use answer (not duplicated question) for fallback cache key stability
            cache_id = sha_key("fallback", question, gold, answer)

        # Fast-skip on stable id
        if strategy == "missing":
            cached = self._cache.get(cache_id)
            if cached and cached.get("ok") is True:
                return {**cached, "cache_status": "hit_ok", "transient": False}
            # try legacy
            leg_k = self.legacy_key(question, answer, gold, evidence_text)
            leg_v = self._cache.get(leg_k)
            if leg_v and leg_v.get("ok") is True:
                self._persist(cache_id, leg_v)
                return {**leg_v, "cache_status": "hit_ok", "transient": False}

        prompt = (
            "You are a strict medical evaluator for Leishmaniasis.\n"
            "Judge ONLY using the provided text evidence.\n"
            'Return JSON: {"faithfulness":0-1,"correctness":0-1,"completeness":0-1,"rationale":"<=60 words"}.\n\n'
            f"QUESTION: {question}\n"
            f"ANSWER: {answer}\n"
            f"GOLD: {gold}\n"
            f"EVIDENCE: {evidence_text[:3000]}\n"
            "JSON:"
        )
        
        # NEW: One try + one retry for transient errors
        for attempt in range(2):  # 0=first try, 1=retry
            try:
                self.bucket.consume()
                resp = self.client.models.generate_content(
                    model=self.model,
                    contents=[prompt],
                    config=self.genai_types.GenerateContentConfig(
                        temperature=0.0,
                        response_mime_type="application/json"
                    ),
                )
                txt = (resp.text or "").strip()
                try:
                    parsed = json.loads(txt)
                except Exception:
                    m = re.search(r"\{.*\}", txt, re.DOTALL)
                    parsed = json.loads(m.group(0)) if m else {
                        "faithfulness": -1,
                        "correctness": -1,
                        "completeness": -1,
                        "rationale": "parse_error"
                    }
                out = {"ok": True, **parsed, "cache_status": "miss", "transient": False}
                self._persist(cache_id, out)
                return out
                
            except Exception as e:
                err_txt = str(e)
                is_transient = self._is_transient_error(e)
                
                # If transient and first attempt, retry after delay
                if is_transient and attempt == 0:
                    delay = self._extract_retry_delay_seconds(e, err_txt)
                    logging.info(f"Transient error (attempt {attempt+1}), retrying after {delay}s: {err_txt[:100]}...")
                    time.sleep(delay)
                    continue  # retry once
                
                # Final failure (either non-transient or second attempt)
                out = {
                    "ok": False,
                    "faithfulness": -1,
                    "correctness": -1,
                    "completeness": -1,
                    "rationale": f"error:{err_txt}",
                    "error": err_txt,
                    "cache_status": "miss",
                    "transient": is_transient,
                }
                self._persist(cache_id, out)
                return out

def _safe_replace(path: Path, new_text_iter):
    """Safely replace file content using atomic write"""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for row in new_text_iter:
            f.write(row)
            if not row.endswith("\n"):
                f.write("\n")
    os.replace(tmp, path)

def compact_cache(cache_path: Path):
    """Compact cache: keep one record per key, prefer OK, then non-transient failures"""
    if not cache_path or not cache_path.exists():
        return
    latest = {}   # key -> best row (dict)
    order = []    # keep insertion order of keys

    with cache_path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
            except Exception:
                continue
            k, v = o.get("key"), o.get("val") or {}
            if not k:
                continue
            # if first time seeing key, note order
            if k not in latest:
                order.append(k)
                latest[k] = v
            else:
                # prefer OK; else prefer non-transient failure; else prefer the newest
                cur = latest[k]
                if v.get("ok") is True:
                    latest[k] = v
                elif cur.get("ok") is True:
                    pass  # keep success
                else:
                    # both failing; prefer non-transient; else prefer newest
                    cur_tr = bool(cur.get("transient"))
                    v_tr = bool(v.get("transient"))
                    if cur_tr and not v_tr:
                        latest[k] = v
                    elif cur_tr == v_tr:
                        latest[k] = v  # newer wins

    def rows():
        for k in order:
            v = latest[k]
            yield json.dumps({"key": k, "val": v}, ensure_ascii=False)

    _safe_replace(cache_path, rows())

def dedupe_stream(stream_path: Path):
    """Deduplicate stream: keep one row per question, prefer OK results"""
    if (not stream_path) or (not stream_path.exists()):
        return
    # load chronologically
    items = []
    with stream_path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
                items.append(o)
            except Exception:
                continue

    # compute stable id (matches judge cache id)
    def norm_q(s): 
        import re
        return re.sub(r"\s+", " ", (s or "").strip().lower())
    import hashlib
    def sid(case_id, q):
        h = hashlib.sha256()
        for p in ("case", str(case_id), norm_q(q)):
            h.update((p or "").encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:24]

    # last-wins, but prefer any OK over fails
    best = {}
    order = []
    for o in items:
        s = sid(o.get("case_id"), o.get("question") or "")
        if s not in best:
            order.append(s)
            best[s] = o
        else:
            cur = best[s]
            cur_ok = bool(((cur.get("gemini_audit") or {}).get("ok")))
            new_ok = bool(((o.get("gemini_audit") or {}).get("ok")))
            if new_ok and not cur_ok:
                best[s] = o
            elif new_ok == cur_ok:
                best[s] = o  # later wins

    def rows():
        for s in order:
            yield json.dumps(best[s], ensure_ascii=False)

    _safe_replace(stream_path, rows())

def repair_stream_from_cache(stream_path: Path, cache_path: Path):
    if (not stream_path) or (not stream_path.exists()) or (not cache_path.exists()):
        logging.warning("repair: missing stream or cache")
        return

    # load stream
    items = []
    with stream_path.open("r", encoding="utf-8") as f:
        for ln in f:
            try: items.append(json.loads(ln))
            except: pass

    # group by stable id
    def _norm_q(s): return re.sub(r"\s+", " ", (s or "").strip().lower())
    def _sid(case_id, q):
        h = hashlib.sha256()
        for p in ("case", str(case_id), _norm_q(q)):
            h.update((p or "").encode("utf-8")); h.update(b"\x00")
        return h.hexdigest()[:24]

    by_sid = {}
    for it in items:
        s = _sid(it.get("case_id"), it.get("question") or "")
        by_sid.setdefault(s, []).append(it)

    # load cache to dict
    cache = {}
    with cache_path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
                k, v = o.get("key"), o.get("val")
                if isinstance(k, str) and isinstance(v, dict):
                    cache[k] = v
            except: pass

    # append a repaired row (copy last stream row, swap gemini_audit) for fail-only SIDs with OK cache
    appended = 0
    with stream_path.open("a", encoding="utf-8") as out:
        for s, rows in by_sid.items():
            if not rows: continue
            has_ok = any((r.get("gemini_audit") or {}).get("ok") for r in rows)
            if has_ok: continue
            cv = cache.get(s)
            if not (cv and cv.get("ok") is True): continue
            base = dict(rows[-1])  # copy last
            base["gemini_audit"] = {**cv, "cache_status": "hit_ok"}
            out.write(json.dumps(base, ensure_ascii=False) + "\n")
            appended += 1

    total_sids = len(by_sid)
    ok_sids = sum(1 for rows in by_sid.values() if any((r.get("gemini_audit") or {}).get("ok") for r in rows))
    logging.info(f"repair_stream_from_cache: {total_sids} total SIDs, {ok_sids} already OK, {appended} repaired")
    # now dedupe
    dedupe_stream(stream_path)

def prune_cache_to_stable_ids(cache_path: Path, stable_ids: set):
    """
    Prune cache to keep only stable IDs with best results.
    stable_ids: set of stable cache keys to keep
    """
    if not cache_path.exists():
        return
    
    kept = {}  # stable_id -> best_record
    
    with cache_path.open("r", encoding="utf-8") as f:
        for ln in f:
            try:
                o = json.loads(ln)
                k, v = o.get("key"), o.get("val", {})
                
                # Only keep stable IDs
                if k not in stable_ids:
                    continue
                    
                if k not in kept:
                    kept[k] = v
                else:
                    # Prefer OK results, then non-transient failures
                    cur = kept[k]
                    if v.get("ok") is True:
                        kept[k] = v
                    elif cur.get("ok") is True:
                        pass  # keep current success
                    else:
                        # Both failing; prefer non-transient
                        if cur.get("transient", False) and not v.get("transient", False):
                            kept[k] = v
                        elif cur.get("transient", False) == v.get("transient", False):
                            kept[k] = v  # newer wins
                            
            except Exception:
                continue
    
    # Write back pruned cache
    def rows():
        for k in sorted(kept.keys()):
            yield json.dumps({"key": k, "val": kept[k]}, ensure_ascii=False)
    
    _safe_replace(cache_path, rows())
    logging.info(f"Cache pruned: kept {len(kept)} stable IDs, removed legacy entries")

# ========================
# Metrics (basic)
# ========================
def compute_exact_match(pred: str, gold: str) -> float:
    return 1.0 if (pred or "").strip().lower() == (gold or "").strip().lower() else 0.0

def compute_f1(pred: str, gold: str) -> float:
    def toks(s: str) -> List[str]:
        return re.findall(r"\w+", (s or "").lower())
    from collections import Counter
    p, g = Counter(toks(pred)), Counter(toks(gold))
    if not p and not g: return 1.0
    if not p or not g:  return 0.0
    overlap = sum((p & g).values())
    precision = overlap / sum(p.values())
    recall = overlap / sum(g.values())
    return 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)

def compute_retrieval_recall(retrieved_indices: List[int], gold_indices: List[int]) -> Optional[float]:
    """Return None when gold_indices is empty to avoid misleading 1.0 scores."""
    if not gold_indices: 
        return None  # Don't claim perfect recall when there's no ground truth
    if not retrieved_indices:
        return 0.0
    return len(set(retrieved_indices) & set(gold_indices)) / float(len(set(gold_indices)))

# ========================
# Config
# ========================
@dataclass
class Config:
    JSONL_DIR: Path
    OUTPUT_DIR: Path
    ANSWERS_FILE: Path
    GOOGLE_API_KEY: str
    GEMINI_MODEL: str = "gemini-2.5-pro"
    GEMINI_RPM: int = 5
    GEMINI_CACHE: Path = None  # set later

# ========================
# Core evaluation
# ========================
def evaluate(cfg: Config, *, strategy: str, stream_write, files: List[Path]) -> List[Dict[str, Any]]:
    gem = Gemini(cfg.GOOGLE_API_KEY, cfg.GEMINI_MODEL, cfg.GEMINI_RPM, cfg.GEMINI_CACHE)
    answers_lut = read_answers_ndjson(cfg.ANSWERS_FILE)

    # Initialize counters for visibility
    counters = {"hit_ok": 0, "miss": 0, "miss_transient": 0, "total": 0}

    # coverage log
    all_q, found = 0, 0
    for fp in files:
        qa = parse_case_jsonl(fp)
        if not qa:
            continue
        case_id = qa.get("case_id")
        for qd in (qa.get("questions") or []):
            all_q += 1
            qtext = (qd.get("question") or "").strip()
            if (case_id, norm_q(qtext)) in answers_lut:
                found += 1
    if all_q > 0:
        coverage = found / all_q
        logging.info(f"Answer coverage: {found}/{all_q} ({coverage:.1%})")
        if coverage < 0.5:
            logging.warning("Low answer coverage - many questions will use fallback 'No answer available'")
    else:
        logging.warning("No questions found in JSONL files")

    summaries = []
    for fp in files:
        logging.info(f"Evaluating {fp.name}")
        qa = parse_case_jsonl(fp)
        if not qa:
            logging.error(f"Parse fail: {fp}")
            continue

        case_id = qa.get("case_id")
        questions = qa.get("questions") or []
        results = []

        for i, qd in enumerate(questions, 1):
            qtext = (qd.get("question") or "").strip()
            gold = (qd.get("gold_answer") or "").strip()
            gold_ev = qd.get("gold_evidence") or []

            rec = answers_lut.get((case_id, norm_q(qtext))) or {}
            answer = (rec.get("answer") or rec.get("generated_answer") or "").strip()
            used_imgs = rec.get("used_images") or []
            hits = rec.get("retrieval_hits") or []
            retrieved_indices = [h.get("page_index") for h in hits if isinstance(h.get("page_index"), int)]

            # Basic metrics
            em = compute_exact_match(answer, gold)
            f1 = compute_f1(answer, gold)
            rrecall = compute_retrieval_recall(
                retrieved_indices,
                [e.get("page_index") for e in gold_ev if isinstance(e.get("page_index"), int)]
            )

            # Evidence text for judge (prefer gold text spans; else short concat of hit text)
            ev_text = " ".join([ev.get("text_span", "") for ev in gold_ev if isinstance(ev, dict) and ev.get("text_span")])

            # Enhanced fallback with better excerpt extraction
            if not ev_text or len(ev_text) < 50:  # Too short, try hit excerpts
                ctxs = []
                for h in hits[:4]:
                    # Try multiple fields for text content
                    snippet = (h.get("text_excerpt") or h.get("text") or 
                            h.get("page_text") or h.get("content") or "").strip()
                    
                    if snippet and len(snippet) > 20:  # Only use meaningful excerpts
                        lab = f"{h.get('doc_id','?')}#p{h.get('page_index','?')}"
                        ctxs.append(f"{lab}: {snippet[:600]}")
                
                if ctxs:
                    ev_text = " ".join(ctxs)[:3000]
                
                # Last resort: try OCR on first used image if available
                if (not ev_text or len(ev_text) < 50) and used_imgs:
                    try:
                        from pathlib import Path
                        img_path = Path(used_imgs[0])
                        if img_path.exists():
                            # Import OCR helper from qdrant_bge
                            import sys
                            sys.path.append(str(Path(__file__).parent.parent))
                            from rag.reranking.med4b_qdrant_bge import ocr_png_fallback, find_case_dir, CFG
                            
                            # Extract case_id and try OCR
                            case_dir = find_case_dir(case_id, CFG.EXTRACT_ROOT)
                            if case_dir:
                                # Parse page index from image name
                                import re
                                m = re.search(r"page_(\d+)\.png", img_path.name)
                                if m:
                                    page_idx = int(m.group(1)) - 1
                                    ocr_text = ocr_png_fallback(case_dir, page_idx)
                                    if ocr_text:
                                        ev_text = f"[OCR fallback]: {ocr_text[:800]}"
                    except Exception as e:
                        logging.debug(f"OCR fallback failed: {e}")

            # Ensure we have at least something for the judge
            if not ev_text:
                ev_text = "[No text evidence available - image-only retrieval]"

            # NEW: stable per-question cache id
            stable_id = sha_key("case", str(case_id), norm_q(qtext))

            res = gem.judge(
                qtext, answer, gold, ev_text,
                strategy=strategy,
                cache_id=stable_id
            )

            # Update counters
            counters["total"] += 1
            if res.get("cache_status") == "hit_ok":
                counters["hit_ok"] += 1
            elif res.get("cache_status") == "miss":
                if res.get("transient"):
                    counters["miss_transient"] += 1
                else:
                    counters["miss"] += 1

            # Prepare trimmed hits for streaming (include text excerpts)
            trimmed_hits = []
            for h in hits[:8]:  # Keep top 8 for stream
                trimmed_hits.append({
                    "doc_id": h.get("doc_id"),
                    "page_index": h.get("page_index"),
                    "score": h.get("score"),
                    "score_kind": "similarity",
                    "text_excerpt": (h.get("text_excerpt") or "")[:200],  # Cap at 200 chars
                    "page_kind": h.get("page_kind"),
                    "micrograph_like": h.get("micrograph_like")
                })

            item = {
                "case_id": case_id,
                "qid": f"q{i:03d}",
                "question": qtext,
                "gold_answer": gold,
                "generated_answer": answer or "No answer available (not found in answers file).",
                "metrics": {"exact_match": em, "f1": f1, "retrieval_recall": rrecall},
                "gemini_audit": res,
                "used_images": used_imgs,
                "retrieved_indices": retrieved_indices[:8],
                "retrieval_hits": trimmed_hits,  # ✅ Add this line
                "gold_indices": [e.get("page_index") for e in gold_ev if isinstance(e.get("page_index"), int)],
                "source_jsonl": str(fp),
                "stable_id": stable_id,
            }
            results.append(item)

            # STREAM this question immediately
            # When resuming (strategy=='missing'), do NOT stream cache-ok hits again.
            should_stream = True
            if strategy == "missing" and (res.get("cache_status") == "hit_ok"):
                should_stream = False

            if should_stream:
                try:
                    stream_write(item)
                except Exception as e:
                    logging.warning(f"stream_write failed: {e}")

        # Summaries with improved CSV handling
        ems = [r["metrics"]["exact_match"] for r in results]
        f1s = [r["metrics"]["f1"] for r in results]
        # Retrieval recall: compute only over defined rows and track coverage
        rr_defined_vals = [r["metrics"]["retrieval_recall"] for r in results if r["metrics"]["retrieval_recall"] is not None]
        avg_rr_defined = float(np.mean(rr_defined_vals)) if rr_defined_vals else float("nan")
        rr_coverage = (len(rr_defined_vals) / len(results)) if results else 0.0

        def avg_gem(k):
            vals = []
            for r in results:
                ga = r.get("gemini_audit") or {}
                v = ga.get(k)
                if isinstance(v, (int, float)) and v >= 0:  # Exclude -1 and None
                    vals.append(v)
            return float(np.mean(vals)) if vals else -1.0

        summary = {
            "case_id": case_id,
            "n_questions": len(results),
            "metrics": {
                "avg_exact_match": float(np.mean(ems)) if ems else 0.0,
                "avg_f1": float(np.mean(f1s)) if f1s else 0.0,
                "avg_retrieval_recall_defined": avg_rr_defined,
                "retrieval_recall_coverage": rr_coverage,
            },
            "gemini_audit": {
                "avg_faithfulness": avg_gem("faithfulness"),
                "avg_correctness":  avg_gem("correctness"),
                "avg_completeness": avg_gem("completeness"),
                "n_audited": len(results),
            },
            "detailed_results": results,
        }
        summaries.append(summary)

    # Log counters for visibility
    logging.info(f"Evaluation counters: {counters}")
    hit_rate = (counters["hit_ok"] / counters["total"]) * 100 if counters["total"] > 0 else 0
    logging.info(f"Cache hit rate: {hit_rate:.1f}% ({counters['hit_ok']}/{counters['total']})")

    return summaries

# ========================
# CLI
# ========================
def main():
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    ap = argparse.ArgumentParser(description="Gemini-only evaluation (streams per-question results)")

    ap.add_argument("--jsonl_dir", type=str, default=None)
    ap.add_argument("--answers_file", type=str, default=None)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--gemini_rpm", type=int, default=5)
    ap.add_argument("--strategy", choices=["all", "missing"], default="all",
                    help="all: judge all Qs; missing: reuse cached successful judgements only")
    ap.add_argument("--run_tag", type=str, default=None,
                    help="Optional suffix tag for outputs (overrides auto tag)")
    ap.add_argument("--cache_path", type=str, default=None,
                    help="Path to NDJSON cache to reuse/append. If omitted, uses OUT_DIR/gemini_cache-<run_tag>.ndjson")
    
    # STREAM HANDLING ARGUMENTS
    ap.add_argument(
        "--stream_path",
        type=str,
        default=None,
        help="If set, append results to this NDJSON file (created if missing)."
    )
    ap.add_argument(
        "--stream_append",
        action="store_true",
        help="Append to the latest existing stream_*.ndjson in OUT_DIR; if none, create a new one."
    )
    
    # COMPACTION ARGUMENTS
    ap.add_argument("--no_compact_cache", action="store_true",
                    help="Disable cache compaction after run (default: compact).")
    ap.add_argument("--no_dedupe_stream", action="store_true",
                    help="Disable stream de-duplication after run (default: dedupe).")
    ap.add_argument("--prune_cache", action="store_true",
                help="Prune cache to keep only stable IDs (removes legacy keys)")
    ap.add_argument("--repair_stream_from_cache", action="store_true",
                help="Append OK cache rows for fail-only SIDs, then dedupe stream (no Gemini calls).")
    
    # MAINTENANCE MODE ARGUMENTS
    ap.add_argument("--compact_only", action="store_true",
                    help="Only compact cache and exit (no evaluation)")
    ap.add_argument("--dedupe_only", action="store_true",
                    help="Only dedupe stream and exit (no evaluation)")
    ap.add_argument("--case_glob", type=str,
                    help="Only evaluate case files whose name matches this glob pattern")

    args = ap.parse_args()

    # --- Build auto run tag ---
    auto_pairs = {
        "strategy": args.strategy,
        "gemini-rpm": args.gemini_rpm,
    }
    run_tag = _slugify(args.run_tag) if args.run_tag else make_run_tag(auto_pairs)

    # Tagged output dir + cache
    out_dir = Path(f"{args.out_dir}-{run_tag}")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.cache_path:
        gemini_cache = Path(args.cache_path)
        gemini_cache.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Using explicit cache_path: {gemini_cache}")
    else:
        gemini_cache = out_dir / f"gemini_cache-{run_tag}.ndjson"
        logging.info(f"Using default cache_path: {gemini_cache}")

    # Check if running maintenance-only mode
    maintenance_only = (args.compact_only or args.dedupe_only or args.prune_cache or args.repair_stream_from_cache)

    # Validate required args based on mode
    if not maintenance_only:
        if not args.jsonl_dir or not args.answers_file:
            ap.error("--jsonl_dir and --answers_file are required unless running a maintenance mode")
    else:
        # prune_cache DOES need jsonl_dir
        if args.prune_cache and not args.jsonl_dir:
            ap.error("--prune_cache requires --jsonl_dir")

    # MAINTENANCE MODES (handle before Path construction to avoid Path(None))
    if args.compact_only:
        logging.info("Compact-only mode: cleaning cache and exiting")
        compact_cache(gemini_cache)
        return

    if args.dedupe_only:
        logging.info("Dedupe-only mode: cleaning stream and exiting")
        if args.stream_path:
            sp = Path(args.stream_path)
        else:
            existing_streams = sorted(out_dir.glob("stream_*.ndjson"))
            sp = existing_streams[-1] if existing_streams else None
        if sp and sp.exists():
            dedupe_stream(sp)
            logging.info(f"Deduplicated: {sp}")
        else:
            logging.warning("No stream file found to deduplicate")
        return

    if args.prune_cache:
        logging.info("Prune-cache mode: removing legacy keys and keeping best per stable ID")
        # Generate stable IDs from JSONL files
        stable_ids = set()
        case_files = sorted(Path(args.jsonl_dir).rglob("*.jsonl"))
        for fp in case_files:
            qa = parse_case_jsonl(fp)
            if not qa:
                continue
            case_id = qa.get("case_id")
            for qd in (qa.get("questions") or []):
                qtext = (qd.get("question") or "").strip()
                stable_id = sha_key("case", str(case_id), norm_q(qtext))
                stable_ids.add(stable_id)
        
        prune_cache_to_stable_ids(gemini_cache, stable_ids)
        logging.info(f"Cache pruned to {len(stable_ids)} stable question IDs")
        return

    if args.repair_stream_from_cache:
        logging.info("Repair-stream mode: appending OK cache rows for fail-only SIDs")
        if not args.stream_path:
            logging.error("--repair_stream_from_cache needs --stream_path")
            return
        repair_stream_from_cache(Path(args.stream_path), gemini_cache)
        return

    # NOW safely construct paths for evaluation mode
    jsonl_dir = Path(args.jsonl_dir)
    answers = Path(args.answers_file)

    key = os.getenv("GOOGLE_API_KEY", "")
    if not key:
        logging.warning("GOOGLE_API_KEY is not set; Gemini calls will fail.")

    cfg = Config(
        JSONL_DIR=jsonl_dir,
        OUTPUT_DIR=out_dir,
        ANSWERS_FILE=answers,
        GOOGLE_API_KEY=key,
        GEMINI_MODEL="gemini-2.5-pro",
        GEMINI_RPM=int(args.gemini_rpm),
        GEMINI_CACHE=gemini_cache
    )

    # STREAM PATH SELECTION LOGIC
    stamp = time.strftime("%Y%m%d_%H%M%S")

    # Choose/derive the stream path
    if args.stream_path:
        stream_path = Path(args.stream_path)
        stream_path.parent.mkdir(parents=True, exist_ok=True)
        logging.info(f"Streaming to explicit path (append/create): {stream_path}")
    elif args.stream_append:
        existing_streams = sorted(out_dir.glob("stream_*.ndjson"))
        if existing_streams:
            stream_path = existing_streams[-1]
            logging.info(f"Appending to latest existing stream: {stream_path}")
        else:
            stream_path = out_dir / f"stream_{stamp}.ndjson"
            logging.info(f"No existing streams found; creating new: {stream_path}")
    else:
        stream_path = out_dir / f"stream_{stamp}.ndjson"
        logging.info(f"Streaming per-question results to new file: {stream_path}")

    def stream_write(obj: dict):
        with stream_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(obj, ensure_ascii=False) + "\n")
            fh.flush()
            os.fsync(fh.fileno())

    # Apply case filtering if specified
    if args.case_glob:
        import fnmatch
        all_files = sorted(Path(args.jsonl_dir).rglob("*.jsonl"))
        files = [fp for fp in all_files if fnmatch.fnmatch(fp.name, args.case_glob)]
        logging.info(f"Filtered to {len(files)} files matching: {args.case_glob}")
    else:
        files = sorted(Path(args.jsonl_dir).rglob("*.jsonl"))

    # Add graceful handling for no files found
    if not files:
        logging.error("No .jsonl files matched (check --jsonl_dir / --case_glob). Exiting.")
        return

    # Evaluate (streams as it goes)
    summaries = evaluate(cfg, strategy=args.strategy, stream_write=stream_write, files=files)

    # Save per-case and aggregate with improved CSV handling
    for summ in summaries:
        case_id = summ.get("case_id", "unknown")
        out_json = out_dir / f"{case_id}__eval_{stamp}.json"
        out_json.write_text(json.dumps(summ, indent=2, ensure_ascii=False), encoding="utf-8")

        # per-question CSV with NaN handling
        rows = []
        for r in summ["detailed_results"]:
            ga = r.get("gemini_audit") or {}
            rows.append({
                "case_id": summ["case_id"],
                "qid": r["qid"],
                "question": r["question"],
                "gold_answer": r["gold_answer"],
                "generated_answer": r["generated_answer"],
                "em": r["metrics"]["exact_match"],
                "f1": r["metrics"]["f1"],
                "retrieval_recall": r["metrics"]["retrieval_recall"],
                "gem_ok": ga.get("ok", False),
                "gem_error": ga.get("error", ""),
                "gem_faithfulness": ga.get("faithfulness", -1),
                "gem_correctness": ga.get("correctness", -1),
                "gem_completeness": ga.get("completeness", -1),
            })
        df = pd.DataFrame(rows).fillna({
            "gem_faithfulness": -1,
            "gem_correctness": -1,
            "gem_completeness": -1
        })
        (out_dir / f"{case_id}__per_q_{stamp}.csv").write_text(df.to_csv(index=False), encoding="utf-8")

    if summaries:
        agg = {
            "timestamp": stamp,
            "n_files": len(summaries),
            "avg_exact_match": float(np.mean([s["metrics"]["avg_exact_match"] for s in summaries])),
            "avg_f1": float(np.mean([s["metrics"]["avg_f1"] for s in summaries])),
            "avg_retrieval_recall_defined": float(np.nanmean([s["metrics"]["avg_retrieval_recall_defined"] for s in summaries])),
            "retrieval_recall_coverage": float(np.mean([s["metrics"]["retrieval_recall_coverage"] for s in summaries])),
            "gemini_audit": {
                "files_audited": int(sum(1 for s in summaries if s["gemini_audit"]["n_audited"] > 0)),
                "avg_faithfulness": float(np.mean([s["gemini_audit"]["avg_faithfulness"] for s in summaries if s["gemini_audit"]["avg_faithfulness"] >= 0])),
                "avg_correctness":  float(np.mean([s["gemini_audit"]["avg_correctness"]  for s in summaries if s["gemini_audit"]["avg_correctness"]  >= 0])),
                "avg_completeness": float(np.mean([s["gemini_audit"]["avg_completeness"] for s in summaries if s["gemini_audit"]["avg_completeness"] >= 0])),
            }
        }
        (out_dir / f"aggregate_eval_{stamp}.json").write_text(json.dumps(agg, indent=2, ensure_ascii=False), encoding="utf-8")
    else:
        logging.warning("No summaries produced; check your inputs.")

    # POST-RUN HYGIENE
    try:
        if not args.no_compact_cache:
            compact_cache(gemini_cache)
            logging.info("Cache compacted (old 429/503 rows removed when superseded).")
        if not args.no_dedupe_stream:
            dedupe_stream(stream_path)
            logging.info("Stream de-duplicated (one row per question; success preferred).")
    except Exception as e:
        logging.warning(f"Post-run hygiene failed: {e}")

    # Optional: manifest
    try:
        manifest = {
            "timestamp": stamp,
            "run_tag": run_tag,
            "args": {
                "answers_file": str(answers),
                "jsonl_dir": str(jsonl_dir),
                "out_dir": str(out_dir),
                "gemini_rpm": int(args.gemini_rpm),
                "strategy": args.strategy,
                "stream_path": str(stream_path),
                "stream_append": args.stream_append,
                "case_glob": args.case_glob,
            },
            "env": {"GOOGLE_API_KEY_SET": bool(key)}
        }
        (out_dir / f"run_manifest_{stamp}.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        logging.warning(f"Failed to write run manifest: {e}")

if __name__ == "__main__":
    main()