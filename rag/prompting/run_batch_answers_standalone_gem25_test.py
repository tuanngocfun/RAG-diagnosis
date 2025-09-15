# Patch: add --pool_mult and --use_reranker to the standalone Gemini script.
# - pool_mult: expands the candidate image pool size (pool_mult * images_per_answer)
# - use_reranker: optional Gemini-based *light* reranker that scores image relevance in one call
#                 (no BGE, still "Gemini-only").
# - Backward compatible with previous flags; retrieval flags are still accepted but ignored.

from pathlib import Path
import os, sys, json, signal, hashlib, argparse, logging, re
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass
import numpy as np
from PIL import Image

# -------------------- Config --------------------

@dataclass
class CFG:
    EXTRACT_ROOT: Path = Path(os.getenv("EXTRACT_ROOT", "/home/students/Leishmania/kaggle/working2/extract"))
    MAX_NEW_TOKENS: int = int(os.getenv("GEM_MAX_NEW_TOKENS", "768"))
    LANGUAGE: str = "en"

# -------------------- Utilities --------------------

def _stable_key(row: Dict[str, Any]) -> str:
    qid = row.get("question_id")
    if isinstance(qid, str) and qid.strip():
        return f"qid:{qid.strip()}"
    case_id = (row.get("case_id") or "").strip()
    qtxt = (row.get("question") or "").strip()
    h = hashlib.sha1((case_id + "\n" + qtxt).encode("utf-8")).hexdigest()
    return f"hk:{h}"

def _read_existing_out(path: Path):
    done, err = set(), set()
    if not path.exists():
        return done, err
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
            if isinstance(rec.get("answer"), str) and rec["answer"].strip():
                done.add(k)
            elif "error" in rec:
                err.add(k)
    return done, err

def sanitize_name(name: str) -> str:
    import re
    s = re.sub(r"[^\w\-.]", "_", str(name))
    s = re.sub(r"_+", "_", s)
    return s.strip("_")

def find_case_dir(case_id: str, extract_root: Path) -> Optional[Path]:
    if not case_id:
        return None
    cand = extract_root / case_id
    if cand.is_dir():
        return cand
    cand2 = extract_root / sanitize_name(case_id)
    if cand2.is_dir():
        return cand2
    low = sanitize_name(case_id).lower()
    for p in extract_root.iterdir():
        if p.is_dir() and low in p.name.lower():
            return p
    return None

# -------------------- Image heuristics --------------------

def average_hash(img: Image.Image, hash_size: int = 8) -> str:
    im = img.convert("L").resize((hash_size, hash_size), Image.BILINEAR)
    arr = np.array(im, dtype=np.float32)
    med = np.median(arr)
    bits = (arr > med).astype(np.uint8)
    return "".join("01"[b] for b in bits.flatten())

def image_metrics(img: Image.Image) -> Dict[str, Any]:
    w, h = img.size
    arr_hsv = np.asarray(img.convert("HSV"), dtype=np.uint8)
    sat_mean = float(arr_hsv[..., 1].mean())

    g = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gx = np.zeros_like(g); gy = np.zeros_like(g)
    gx[:, 1:-1] = (g[:, 2:] - g[:, :-2]) * 0.5
    gy[1:-1, :] = (g[2:, :] - g[:-2, :]) * 0.5
    grad = np.sqrt(gx * gx + gy * gy)
    edge_density = float((grad > 0.2).mean())

    hist = np.histogram((g * 255).astype(np.uint8), bins=256, range=(0, 255))[0].astype(np.float32)
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

def rank_images_by_micrograph(paths: List[Path]) -> List[Tuple[float, Path]]:
    scored: List[Tuple[float, Path]] = []
    for p in paths:
        try:
            im = Image.open(p).convert("RGB")
            try:
                m = image_metrics(im)
            finally:
                im.close()
            score = (2.0 if m["micrograph_like"] else 0.0) + 1.5 * m["edge_density"] + 0.5 * m["entropy"]/8.0
            scored.append((float(score), p))
        except Exception:
            continue
    scored.sort(key=lambda t: t[0], reverse=True)
    return scored

# -------------------- Gemini wrapper --------------------

class Gemini25:
    def __init__(self, model_id: str = "gemini-2.5-pro"):
        try:
            from google import genai
            from google.genai.types import GenerateContentConfig
        except Exception as e:
            raise SystemExit(
                "google-genai is required. Install with:\n"
                "  pip install google-genai==0.6.0\n"
                f"Import error: {e}"
            )
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise SystemExit("Missing GOOGLE_API_KEY in environment for Gemini 2.5 Pro.")
        self._genai = genai
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.cfg_cls = GenerateContentConfig

    @staticmethod
    def _encode_image(pil_img: Image.Image) -> dict:
        import io
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        data = buf.getvalue()
        return {"inline_data": {"mime_type": "image/png", "data": data}}

    def _build_system_and_user(self, q: str, ims: List[Image.Image], context_text: str = ""):
        sys_instr = (
            "You are a careful medical assistant.\n"
            "Use only the provided images and short context.\n"
            "If evidence is insufficient for a precise answer, reply exactly: 'Insufficient evidence.'"
        )
        user_text = f"Question: {q}"
        if context_text:
            user_text += f"\nContext (verbatim excerpts):\n{context_text}\n"

        parts = [{"text": user_text}]
        for im in ims:
            parts.append(self._encode_image(im))

        contents = [{"role": "user", "parts": parts}]
        return sys_instr, contents

    def answer(self, q: str, image_paths: List[Path], max_output_tokens: int = 768, context_text: str = "") -> str:
        ims_open = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
        if not ims_open:
            return "No valid images provided."
        try:
            sys_instr, contents = self._build_system_and_user(q, ims_open, context_text=context_text)

            cfg = self.cfg_cls(
                temperature=0.0, top_p=1.0, top_k=40,
                max_output_tokens=max_output_tokens,
                system_instruction=sys_instr[:400],
                response_mime_type="text/plain",
            )

            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=contents,
                config=cfg
            )

            txt = ""
            try:
                if hasattr(resp, "text") and resp.text:
                    txt = resp.text
                elif hasattr(resp, "candidates") and resp.candidates:
                    for cand in resp.candidates:
                        content = getattr(cand, "content", None)
                        parts = getattr(content, "parts", []) if content is not None else []
                        for part in parts:
                            t = getattr(part, "text", None)
                            if t:
                                txt += t
                txt = (txt or "").strip()
            except Exception:
                txt = ""

            if not txt:
                try:
                    reasons = [getattr(c, "finish_reason", None) for c in (getattr(resp, "candidates", []) or [])]
                    logging.warning(f"Gemini returned no text. finish_reasons={reasons}")
                except Exception:
                    pass
                txt = "Insufficient evidence."
            return txt
        finally:
            for im in ims_open:
                try:
                    im.close()
                except Exception:
                    pass

    def score_images(self, q: str, image_paths: List[Path]) -> List[float]:
        """
        Lightweight Gemini-based reranker: single call that asks for JSON scores 0..1 per image.
        Returns a list of floats aligned with image_paths. Fallback to uniform if parsing fails.
        """
        if not image_paths:
            return []
        ims_open = [Image.open(p).convert("RGB") for p in image_paths if p.exists()]
        if not ims_open:
            return []

        # Build contents that label each image.
        parts = [{"text": (
            "You will see a question and N images. "
            "For each image i, return a JSON object with a 'scores' array of length N, "
            "with values in [0,1] indicating how relevant image i is to answering the question. "
            "Return ONLY JSON, no extra text. Example: {\"scores\": [0.12, 0.77, 0.03]}"
            f"\nQuestion: {q}\n"
            "Now the images follow, labeled.\n"
        )}]
        for idx, im in enumerate(ims_open, start=1):
            parts.append({"text": f"Image {idx}:"})
            parts.append(self._encode_image(im))

        cfg = self.cfg_cls(
            temperature=0.0, top_p=1.0, top_k=1,
            max_output_tokens=256,
            response_mime_type="text/plain",
            system_instruction="Return JSON only."
        )

        try:
            resp = self.client.models.generate_content(
                model=self.model_id,
                contents=[{"role": "user", "parts": parts}],
                config=cfg
            )
            raw = ""
            if hasattr(resp, "text") and resp.text:
                raw = resp.text
            else:
                # fallback: concatenate parts text
                if hasattr(resp, "candidates") and resp.candidates:
                    for cand in resp.candidates:
                        content = getattr(cand, "content", None)
                        for part in (getattr(content, "parts", []) or []):
                            t = getattr(part, "text", None)
                            if t:
                                raw += t
            raw = (raw or "").strip()

            # Extract JSON
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if m:
                obj = json.loads(m.group(0))
                scores = obj.get("scores")
                if isinstance(scores, list) and len(scores) == len(ims_open):
                    vals = [float(x) if isinstance(x, (int, float)) else 0.0 for x in scores]
                else:
                    vals = [1.0/len(ims_open)] * len(ims_open)
            else:
                vals = [1.0/len(ims_open)] * len(ims_open)
        except Exception:
            vals = [1.0/len(ims_open)] * len(ims_open)
        finally:
            for im in ims_open:
                try: im.close()
                except Exception: pass

        return vals

# -------------------- Image selection --------------------

def collect_candidate_images(seed: List[str], doc_id: Optional[str], images_per_answer: int,
                             micrograph_only: bool, pool_mult: int) -> List[str]:
    seen = set()
    out: List[Path] = []
    target_pool = max(1, pool_mult) * max(1, images_per_answer)

    # 1) Seed images first
    for p in seed or []:
        P = Path(p)
        if P.exists() and P.suffix.lower() in {".png", ".jpg", ".jpeg"} and str(P) not in seen:
            out.append(P); seen.add(str(P))
            if len(out) >= target_pool:
                return [str(x) for x in out]

    # 2) Scan case dir
    if doc_id:
        case_dir = find_case_dir(doc_id, CFG.EXTRACT_ROOT)
        if case_dir and (case_dir / "pages").is_dir():
            pages = sorted((case_dir / "pages").glob("page_*.png"))
            if micrograph_only:
                ranked = rank_images_by_micrograph(pages)
                for _, p in ranked:
                    if str(p) in seen: continue
                    out.append(p); seen.add(str(p))
                    if len(out) >= target_pool: break
            else:
                for p in pages:
                    if str(p) in seen: continue
                    out.append(p); seen.add(str(p))
                    if len(out) >= target_pool: break

    return [str(x) for x in out]

# -------------------- Batch processor --------------------

class BatchProcessor:
    def __init__(self, use_reranker: bool):
        print("[INFO] Loading Gemini (one-time init)…")
        self.gem = Gemini25(model_id=os.getenv("GEMINI_MODEL_ID", "gemini-2.5-pro"))
        self.use_reranker = use_reranker
        if use_reranker:
            print("[INFO] Gemini-based reranker enabled (JSON scoring per batch).")
        print("[INFO] Gemini ready.")

    def answer_one(self, row: Dict[str, Any], images_per_answer: int,
                   micrograph_only: bool, pool_mult: int) -> Dict[str, Any]:
        qtext = (row.get("question") or "").strip()
        if not qtext:
            return {"error": "empty_question", **row}

        doc_id = row.get("doc_id")
        seed_images: List[str] = row.get("seed_image_paths") or []
        pool = collect_candidate_images(seed_images, doc_id, images_per_answer, micrograph_only, pool_mult)

        if not pool:
            return {"error": "no_images_on_disk", **row}

        chosen: List[str]
        if self.use_reranker and len(pool) > images_per_answer:
            # Single-call scoring of the pool
            scores = self.gem.score_images(qtext, [Path(p) for p in pool])
            order = sorted(range(len(pool)), key=lambda i: scores[i] if i < len(scores) else 0.0, reverse=True)
            chosen = [pool[i] for i in order[:images_per_answer]]
        else:
            # Fall back to top-N by heuristic order already provided by collect_candidate_images()
            chosen = pool[:images_per_answer]

        ans = self.gem.answer(qtext, [Path(p) for p in chosen], max_output_tokens=CFG.MAX_NEW_TOKENS)
        return {
            "question_id": row.get("question_id"),
            "case_id": row.get("case_id"),
            "doc_id": doc_id,
            "question": qtext,
            "retrieve_mode": "standalone_gemini_only",
            "used_images": chosen,
            "pool_size": len(pool),
            "answer": ans,
        }

# -------------------- CLI --------------------

def main():
    ap = argparse.ArgumentParser(description="Batch answer with Gemini 2.5 Pro only (no retrieval).")
    ap.add_argument("--manifest", required=True, help="questions_manifest.jsonl")
    ap.add_argument("--out", required=True, help="Output NDJSON (append-safe with --resume)")
    ap.add_argument("--images_per_answer", type=int, default=4, help="Images to feed to Gemini")
    ap.add_argument("--micrograph_only", action="store_true", help="Prefer micrograph-like pages (heuristic)")

    # New knobs:
    ap.add_argument("--pool_mult", type=int, default=1, help="Expand candidate image pool by this multiplier.")
    ap.add_argument("--use_reranker", action="store_true",
                    help="Use a Gemini-only lightweight reranker to score images in one JSON-returning call.")

    # Accept familiar retrieval flags but ignore (compat only)
    ap.add_argument("--topk", type=int, default=0)  # ignored
    ap.add_argument("--score_threshold", type=float, default=None)  # ignored
    ap.add_argument("--case_type")  # ignored
    ap.add_argument("--keyword")  # ignored
    ap.add_argument("--any_keywords")  # ignored

    # Durability
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--retry_errors", action="store_true")
    ap.add_argument("--fsync_interval", type=int, default=25)
    args = ap.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    done_keys, err_keys = _read_existing_out(out_path)
    want_retry_errors = args.resume and args.retry_errors

    mode = "a" if args.resume and out_path.exists() else "w"
    fout = out_path.open(mode, encoding="utf-8")
    need_fsync_every = max(1, int(args.fsync_interval))

    exiting = {"flag": False}
    def _graceful(signum, frame):
        exiting["flag"] = True
        try:
            fout.flush(); os.fsync(fout.fileno())
        except Exception:
            pass
        signal.signal(signum, signal.SIG_DFL)
    for s in (signal.SIGINT, signal.SIGTERM):
        signal.signal(s, _graceful)

    # One-time init
    proc = BatchProcessor(use_reranker=args.use_reranker)

    def write_row(row: Dict[str, Any]) -> None:
        nonlocal need_fsync_every, fout
        fout.write(json.dumps(row, ensure_ascii=False) + "\n")
        fout.flush()
        write_row.count += 1
        if write_row.count >= need_fsync_every:
            os.fsync(fout.fileno())
            write_row.count = 0
    write_row.count = 0

    # Warn about ignored retrieval flags
    if any([args.topk, args.score_threshold is not None, args.case_type, args.keyword, args.any_keywords]):
        print("[WARN] Retrieval-related flags were provided but are ignored in standalone Gemini mode.", file=sys.stderr)

    print("[INFO] Starting batch processing…")
    total = ok = skipped = retried = 0

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
            if args.resume:
                if key in done_keys:
                    skipped += 1; continue
                if (not want_retry_errors) and key in err_keys:
                    skipped += 1; continue
                if want_retry_errors and key in err_keys:
                    retried += 1

            try:
                out_rec = proc.answer_one(row_in, args.images_per_answer, args.micrograph_only, args.pool_mult)
                write_row(out_rec)
                if "answer" in out_rec and out_rec["answer"]:
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
        fout.flush(); os.fsync(fout.fileno())
    except Exception:
        pass
    fout.close()

    print(f"[FINAL] Answered {ok}/{total} (skipped={skipped}, retried={retried}) -> {out_path}")

if __name__ == "__main__":
    main()
