#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Multi-Case Local-Cache Edition with Selective Training
========================================================
Enhanced with finetune-some command to train only specific cases
"""
from __future__ import annotations
import os, json, math, time, logging, pickle, re, unicodedata, random, csv, shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional

import torch
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    ColQwen2ForRetrieval,
    ColQwen2Processor,
)
from transformers.utils.import_utils import is_flash_attn_2_available
from datasets import Dataset
from random import randint
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datetime import datetime, timezone
from dotenv import load_dotenv
load_dotenv() 
# ------------------------------
# Config (edit ROOT if needed)
# ------------------------------
@dataclass
class Config:
    ROOT: Path = Path("/media/pc1/Ubuntu/Extend_Data/ngoc")
    EXTRACT_ROOT: Path = ROOT / "kaggle" / "working2" / "extract"

    # Models
    GEN_MODEL_ID: str = "google/medgemma-4b-it"     # generator (multimodal)
    RET_MODEL_ID: str = "vidore/colqwen2-v1.0-hf"   # retriever

    # CSV names produced by your filter
    KEEP_CSV: str = "case_keep.csv"
    UNCERTAIN_CSV: str = "case_uncertain.csv"

    # Strict local cache
    HF_CACHE: Path = ROOT / "hf" / "transformers"

    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    SEED: int = 42

    CACHE_DIR: Path = EXTRACT_ROOT / "_cache_global"
    BATCH_IMAGES: int = 8      # ColQwen2 embedding batch (tune if OOM)
    TOP_K: int = 4
    MAX_NEW_TOKENS: int = 512

    # QLoRA (tiny defaults)
    OUTPUT_DIR: Path = EXTRACT_ROOT / "_adapters_global"
    LORA_R: int = 8  # Reduced from 16 for projector stability
    LORA_ALPHA: int = 16  # Reduced from 32
    LORA_DROPOUT: float = 0.05
    LEARNING_RATE: float = 5e-5  # Reduced from 8e-5
    EPOCHS: int = 2  # Increased from 1
    PER_DEVICE_BATCH: int = 1
    GRAD_ACCUM: int = 4
    MAX_SEQ_LEN: int = 1536
    GRADIENT_CHECKPOINTING: bool = True
    
    # Add eval config
    EVAL_STEPS: int = 50  # Evaluate every N steps if you add validation

CFG = Config()

# Environment
HF_TOKEN = os.getenv("HF_TOKEN")
os.environ.setdefault("TRANSFORMERS_CACHE", str(CFG.HF_CACHE))
os.environ.setdefault("HF_HOME", str(CFG.HF_CACHE))
CFG.CACHE_DIR.mkdir(parents=True, exist_ok=True)
CFG.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(CFG.SEED)
torch.manual_seed(CFG.SEED)

def discover_projector_modules(model_id: str = CFG.GEN_MODEL_ID) -> List[str]:
    """Discover vision projector module names in the model."""
    from transformers import AutoModelForImageTextToText
    
    # Load model structure (meta only)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        token=HF_TOKEN,
        torch_dtype=torch.float16,
        device_map="meta",  # Just load structure, not weights
        cache_dir=str(CFG.HF_CACHE),
        local_files_only=True,
    )
    
    projector_keywords = ["project", "vision", "visual", "mm_", "multi_modal"]
    projector_modules = []
    
    for name, module in model.named_modules():
        name_lower = name.lower()
        # Look for linear/conv layers in vision-related modules
        if any(kw in name_lower for kw in projector_keywords):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Extract the parameter name (e.g., "vision_tower.projection.weight" -> "projection")
                parts = name.split('.')
                if len(parts) > 1:
                    projector_modules.append(parts[-1])
    
    # Deduplicate
    projector_modules = list(set(projector_modules))
    print(f"[INFO] Found potential projector modules: {projector_modules}")
    return projector_modules

# ------------------------------
# Discovery & utilities
# ------------------------------
def _slug(s: str) -> str:
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r"[^a-zA-Z0-9._-]+", "_", s).strip("_")
    return s or "case"

def discover_cases() -> List[Path]:
    """Find case dirs that have a pages/ subdir."""
    roots = []
    if CFG.EXTRACT_ROOT.is_dir():
        for p in CFG.EXTRACT_ROOT.iterdir():
            if p.is_dir() and (p / "pages").is_dir():
                roots.append(p)
    return sorted(roots)

def find_cases_by_fragment(fragment: str) -> List[Path]:
    frag = fragment.strip().lower()
    return [c for c in discover_cases() if frag in c.name.lower()]

def list_pages_full(case_dir: Path) -> List[Path]:
    return sorted((case_dir / "pages").glob("page_*.png"))

def _read_filter_csv(csv_path: Path, pages_dir: Path) -> List[Path]:
    if not csv_path.exists(): return []
    out: List[Path] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        cols = [c.lower() for c in (reader.fieldnames or [])]
        for r in reader:
            # prefer absolute 'path', else 'page'/'filename' (basename)
            if "path" in cols and r.get("path"):
                p = Path(r["path"])
                if p.exists(): out.append(p); continue
            name = (r.get("page") or r.get("filename") or "").strip()
            if name:
                p = pages_dir / name
                if p.exists(): out.append(p)
    # de-dup preserving order
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p); uniq.append(p)
    return uniq

def list_pages_for_index(case_dir: Path, mode: str) -> List[Path]:
    """mode ∈ {full, keep, keep+uncertain}"""
    pages_dir = case_dir / "pages"
    if mode == "full":
        return list_pages_full(case_dir)
    keep = _read_filter_csv(case_dir / CFG.KEEP_CSV, pages_dir)
    if mode == "keep":
        return keep or list_pages_full(case_dir)
    if mode == "keep+uncertain":
        un = _read_filter_csv(case_dir / CFG.UNCERTAIN_CSV, pages_dir)
        merged = keep + [p for p in un if p not in set(keep)]
        return merged or list_pages_full(case_dir)
    return list_pages_full(case_dir)

def _case_cache_paths(case_dir: Path, mode: str) -> Tuple[Path, Path]:
    """Return (embed_cache_file, manifest_file) for this case+mode."""
    idx_dir = case_dir / f"_index_{mode}"
    idx_dir.mkdir(parents=True, exist_ok=True)
    emb = idx_dir / "embeds.pkl"
    man = idx_dir / "manifest.json"
    return emb, man

def _build_manifest(paths: List[Path]) -> Dict[str, Any]:
    items = []
    for p in paths:
        try:
            st = p.stat()
            items.append({"path": str(p), "mtime": st.st_mtime_ns, "size": st.st_size})
        except FileNotFoundError:
            pass
    return {"count": len(items), "items": items}

def _manifests_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    if a.get("count") != b.get("count"): return False
    A, B = a.get("items", []), b.get("items", [])
    if len(A) != len(B): return False
    for i in range(len(A)):
        if A[i] != B[i]: return False
    return True

def _save_ask_result(case_dir: Optional[Path], index_mode: str, payload: Dict[str, Any]) -> Path:
    """Save ask result to JSON file, either in case dir or global dir."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    if case_dir is not None:
        out_dir = case_dir / "_asks"
    else:
        # Khi ask across all cases
        out_dir = CFG.EXTRACT_ROOT / "_asks_global"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{ts}_{index_mode}.json"
    out_file.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return out_file

# ------------------------------
# ColQwen2 Indexer (per-case)
# ------------------------------
class ColQwen2Indexer:
    def __init__(self, index_mode: str = "keep"):
        assert index_mode in {"full","keep","keep+uncertain"}
        self.index_mode = index_mode
        self.model = ColQwen2ForRetrieval.from_pretrained(
            CFG.RET_MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
            device_map="auto",
            attn_implementation=("flash_attention_2" if is_flash_attn_2_available() else "sdpa"),
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        ).eval()
        self.processor = ColQwen2Processor.from_pretrained(
            CFG.RET_MODEL_ID,
            token=HF_TOKEN,
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        )

    @torch.inference_mode()
    def build_or_load_case(self, case_dir: Path) -> Tuple[List[Path], List[torch.Tensor]]:
        pages = list_pages_for_index(case_dir, self.index_mode)
        manifest_now = _build_manifest(pages)
        emb_file, man_file = _case_cache_paths(case_dir, self.index_mode)

        # try cache
        if emb_file.exists() and man_file.exists():
            try:
                past = json.loads(man_file.read_text())
                if _manifests_equal(past, manifest_now):
                    with open(emb_file, "rb") as f:
                        embs = pickle.load(f)
                    if len(embs) == len(pages):
                        logging.info("[CACHE] %s (%s): %d pages", case_dir.name, self.index_mode, len(embs))
                        return pages, embs
            except Exception as e:
                logging.warning("[CACHE] ignore stale cache for %s: %s", case_dir.name, e)

        # rebuild
        logging.info("[BUILD] %s (%s): embedding %d pages", case_dir.name, self.index_mode, len(pages))
        embs: List[torch.Tensor] = []
        device = self.model.device
        bs = max(1, int(CFG.BATCH_IMAGES))
        for i in range(0, len(pages), bs):
            chunk_paths = pages[i:i+bs]
            images = [Image.open(p).convert("RGB") for p in chunk_paths]
            batch = self.processor(images=images, return_tensors="pt").to(device)
            out = self.model(**batch)
            e = out.embeddings
            if isinstance(e, torch.Tensor):
                seq_lens = batch["attention_mask"].sum(dim=1).tolist()
                per = [e[j, :seq_lens[j]].to("cpu") for j in range(e.shape[0])]
            else:
                per = [x.to("cpu") for x in e]
            embs.extend(per)
            del batch, out, e, images
            if torch.cuda.is_available(): torch.cuda.empty_cache()

        with open(emb_file, "wb") as f: pickle.dump(embs, f)
        man_file.write_text(json.dumps(manifest_now))
        logging.info("[BUILD] %s: cached %d embeddings → %s", case_dir.name, len(embs), emb_file)
        return pages, embs

    @torch.inference_mode()
    def search_single(self, query: str, pages: List[Path], embs: List[torch.Tensor]) -> List[Tuple[int, float]]:
        q_inputs = self.processor(text=[query], return_tensors="pt").to(self.model.device)
        q_out = self.model(**q_inputs)
        q_embs = q_out.embeddings
        scores = self.processor.score_retrieval(q_embs, embs)  # list-of-seqs supported
        s = scores[0].tolist()
        top = sorted(list(enumerate(s)), key=lambda x: x[1], reverse=True)[:CFG.TOP_K]
        return [(idx, float(score)) for idx, score in top]

# ------------------------------
# Generator (MedGemma-4B-IT)
# ------------------------------
class MedGemmaGen:
    def __init__(self):
        self.model = AutoModelForImageTextToText.from_pretrained(
            CFG.GEN_MODEL_ID,
            token=HF_TOKEN,
            torch_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
            device_map="auto",
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            CFG.GEN_MODEL_ID,
            token=HF_TOKEN,
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        )

    def _build_messages(self, question: str, pil_images: List[Image.Image]) -> List[Dict[str, Any]]:
        return [
            {"role": "system", "content": [{"type":"text","text":
                "You are a careful medical assistant. Use only the provided images to answer. "
                "If the answer is not supported by the images, say 'Insufficient evidence.'"}]},
            {"role": "user", "content": ([{"type":"text","text": question}]
                                         + [{"type":"image","image": im} for im in pil_images])}
        ]

    @torch.inference_mode()
    def generate(self, question: str, page_paths: List[Path]) -> Dict[str, Any]:
        images = [Image.open(p).convert("RGB") for p in page_paths]
        msgs = self._build_messages(question, images)
        text = self.processor.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        batch = self.processor(text=[text], images=[images], return_tensors="pt").to(self.model.device)
        in_len = batch["input_ids"].shape[-1]
        stop_ids = [self.processor.tokenizer.eos_token_id,
                    self.processor.tokenizer.convert_tokens_to_ids("<end_of_turn>")]
        gen = self.model.generate(**batch, max_new_tokens=CFG.MAX_NEW_TOKENS, do_sample=False, temperature=0.0,top_p=0.1, num_beams=1, eos_token_id=stop_ids)
        out_ids = gen[0][in_len:]
        decoded = self.processor.decode(out_ids, skip_special_tokens=True)
        return {"answer": decoded}

# ------------------------------
# Datasets (multi-images per sample)
# ------------------------------
class MultiImageSFTDataset:
    """
    Each JSONL line:
      {"question": str, "answer": str, "pages": [int,...]}
    -> ONE training sample whose user turn contains ALL referenced images.
    """
    def __init__(self, jsonl_path: Path, full_page_paths: List[Path], max_per_qa: Optional[int] = None):
        rows = []
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line: rows.append(json.loads(line))

        self.samples: List[Dict[str, Any]] = []
        for r in rows:
            q = (r.get("question") or "").strip()
            a = (r.get("answer") or "").strip()
            pages = r.get("pages") or r.get("page_ids") or []
            if not q or not a or not pages:
                continue

            # If max_per_qa is set, trim the list but keep it multi-image
            use_pages = pages if (max_per_qa is None) else pages[:max_per_qa]
            image_paths = []
            for pidx in use_pages:
                if isinstance(pidx, int) and 0 <= pidx < len(full_page_paths):
                    image_paths.append(str(full_page_paths[pidx]))

            if not image_paths:
                continue

            msgs = [
                {"role":"system","content":[{"type":"text","text":
                    "You are a careful medical assistant. Base your answer ONLY on the pixels of the provided images. "
                    "If the answer is not supported by the visible content, reply exactly: 'Insufficient evidence.' "
                    "Do NOT infer species unless its name is explicitly readable as text in the image(s)."}]},
                {"role":"user","content":[
                    {"type":"text","text": q},
                    *[{"type":"image","image_path": p} for p in image_paths]
                ]},
                {"role":"assistant","content":[{"type":"text","text": a}]}
            ]
            self.samples.append({"messages": msgs})

    def to_hfds(self) -> Dataset:
        return Dataset.from_list(self.samples)

class MultiCaseSFTDataset:
    def __init__(self, cases: List[Path], max_per_qa: Optional[int] = None):
        samples = []
        for c in cases:
            full_pages = list_pages_full(c)
            js = sorted(c.glob("*case*_questions*.jsonl"))
            for j in js:
                ds = MultiImageSFTDataset(j, full_pages, max_per_qa=max_per_qa).to_hfds()
                samples.extend(ds)
        self.ds = Dataset.from_list(list(samples)) if samples else Dataset.from_list([])

    def to_hfds(self) -> Dataset:
        return self.ds


# ------------------------------
# Collator & QLoRA trainer
# ------------------------------
class VLMCollator:
    def __init__(self, processor: AutoProcessor, max_length: Optional[int]):
        self.processor = processor
        self.max_length = max_length
        assert getattr(self.processor.tokenizer, "is_fast", False), "Need a fast tokenizer"

    def _extract_assistant_indices(self, text: str):
        spans = []
        pattern = r'<start_of_turn>(?:model|assistant)\n(.*?)<end_of_turn>'
        for m in re.finditer(pattern, text, re.DOTALL):
            spans.append(m.span(1))
        return spans

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        texts, images_per_example = [], []
        assistant_char_spans = []

        for ex in features:
            msgs = ex["messages"]
            # collect ALL user images
            user_imgs = []
            for c in msgs[1]["content"]:
                if isinstance(c, dict) and c.get("type") in ("image","image_path"):
                    if "image" in c and isinstance(c["image"], Image.Image):
                        user_imgs.append(c["image"].convert("RGB"))
                    elif "image_path" in c and c["image_path"]:
                        user_imgs.append(Image.open(c["image_path"]).convert("RGB"))

            assert len(user_imgs) >= 1, "Each sample must include >=1 images"
            images_per_example.append(user_imgs)

            text = self.processor.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=False
            )
            texts.append(text)
            assistant_char_spans.append(self._extract_assistant_indices(text))

        # processor expects list-of-text and list-of-list-of-images
        batch = self.processor(
            text=texts,
            images=images_per_example,     # <-- list of lists
            padding=True,
            truncation=(self.max_length is not None),
            max_length=self.max_length,
            return_tensors="pt",
            return_offsets_mapping=True
        )

        labels = batch["input_ids"].clone()
        offset_mapping = batch.pop("offset_mapping", None)
        if offset_mapping is not None:
            for bi, spans in enumerate(assistant_char_spans):
                labels[bi] = -100
                for cs, ce in spans:
                    for ti, (ts, te) in enumerate(offset_mapping[bi]):
                        if te > cs and ts < ce:
                            labels[bi, ti] = batch["input_ids"][bi, ti]
        batch["labels"] = labels
        return batch

class MedGemmaQLoRA:
    def __init__(self):
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=(torch.bfloat16 if torch.cuda.is_available() else torch.float16),
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            CFG.GEN_MODEL_ID,
            quantization_config=bnb,
            device_map="auto",
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        )
        self.processor = AutoProcessor.from_pretrained(
            CFG.GEN_MODEL_ID,
            cache_dir=str(CFG.HF_CACHE),
            local_files_only=True,
        )
        
        # Base text modules
        base_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]
        
        # Runtime discovery of projector modules (safer than meta-device)
        try:
            projector_mods = self._find_projector_linear_names()
            if projector_mods and len(projector_mods) < 10:  # Sanity check
                print(f"[INFO] Adding projector modules to LoRA: {projector_mods}")
                target_modules = base_modules + projector_mods
                lora_r = min(CFG.LORA_R, 8)  # Use smaller rank for projector
            else:
                target_modules = base_modules
                lora_r = CFG.LORA_R
        except Exception as e:
            print(f"[WARNING] Could not discover projector modules: {e}")
            target_modules = base_modules
            lora_r = CFG.LORA_R
        
        self.peft = LoraConfig(
            r=lora_r,
            lora_alpha=CFG.LORA_ALPHA,
            lora_dropout=CFG.LORA_DROPOUT,
            bias="none",
            target_modules=target_modules,
            task_type="CAUSAL_LM",
        )

    def _find_projector_linear_names(self):
        """Runtime discovery of projector Linear layers"""
        import torch.nn as nn
        candidates = []
        for name, module in self.model.named_modules():
            name_lower = name.lower()
            # Look for Linear layers in vision-related modules
            if any(kw in name_lower for kw in ["vision", "project", "mm_", "multi_modal", "siglip"]):
                if isinstance(module, nn.Linear):
                    candidates.append(name)  # Use full path
        return candidates

    def train(self, ds: Dataset, out_dir: Path):
        collator = VLMCollator(self.processor, None)  # Pass None for max_length
        cfg = SFTConfig(
            output_dir=str(out_dir),
            per_device_train_batch_size=CFG.PER_DEVICE_BATCH,
            gradient_accumulation_steps=CFG.GRAD_ACCUM,
            learning_rate=CFG.LEARNING_RATE,
            num_train_epochs=CFG.EPOCHS,
            logging_steps=10,
            save_steps=200,
            save_total_limit=2,
            bf16=torch.cuda.is_available(),
            fp16=not torch.cuda.is_available(),
            packing=False,
            remove_unused_columns=True,
            # ⬇️ in TRL 0.22.x use max_length (NOT max_seq_length)
            max_length=None,  # Critical for VLMs - don't truncate image tokens
            dataset_kwargs={"skip_prepare_dataset": True},
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},  # Reduce warnings
            warmup_ratio=0.03,  # Add warmup
            seed=CFG.SEED,
        )
        
        if len(ds) == 0:
            raise RuntimeError("Empty training dataset.")
            
        # Validation check
        probe = ds[randint(0, len(ds)-1)]
        assert isinstance(probe, dict) and "messages" in probe
        user_turn = probe["messages"][1]["content"]
        has_path = any(isinstance(c, dict) and c.get("image_path") for c in user_turn)
        assert has_path, "Dataset row missing image_path; did you update to_hfds()?"
        
        trainer = SFTTrainer(
            model=self.model,
            train_dataset=ds,
            data_collator=collator,
            peft_config=self.peft,
            args=cfg,
            # ⛔️ don’t pass max_seq_length here on 0.22.x
        )

        trainer.train()
        peft_model = trainer.model               # this is a PeftModel
        peft_model.save_pretrained(out_dir)      # writes adapter_model.safetensors + adapter_config.json
        self.processor.save_pretrained(out_dir)

# ------------------------------
# High-level ops
# ------------------------------
def embed_cases(cases: List[Path], index_mode: str):
    idx = ColQwen2Indexer(index_mode=index_mode)
    total_pages = 0
    for c in cases:
        pages, embs = idx.build_or_load_case(c)
        total_pages += len(pages)
    print(f"[INFO] Embedded {len(cases)} case(s), total pages={total_pages}, mode={index_mode}")

def ask_in_case(query: str, case_dir: Path, index_mode: str) -> Dict[str, Any]:
    idx = ColQwen2Indexer(index_mode=index_mode)
    pages, embs = idx.build_or_load_case(case_dir)
    top = idx.search_single(query, pages, embs)
    top_paths = [pages[i] for i,_ in top]
    gen = MedGemmaGen()
    out = gen.generate(query, top_paths)
    out.update({
        "index_mode": index_mode,
        "case": case_dir.name,
        "hits": [{"rank": r+1, "path": str(pages[i]), "score": float(s)} for r,(i,s) in enumerate(top)]
    })
    out_path = _save_ask_result(case_dir, index_mode, out)
    out["saved_to"] = str(out_path)
    return out

def ask_across_all(query: str, cases: List[Path], index_mode: str) -> Dict[str, Any]:
    idx = ColQwen2Indexer(index_mode=index_mode)
    # concatenate per-case embs and keep backrefs
    backrefs, all_pages, all_embs = [], [], []
    for ci, c in enumerate(cases):
        pages, embs = idx.build_or_load_case(c)
        start = len(all_pages)
        all_pages.extend([(ci, p) for p in pages])
        all_embs.extend(embs)
        backrefs.append((start, len(all_pages)))  # not used further but handy

    # search once over all_embs
    q_inputs = idx.processor(text=[query], return_tensors="pt").to(idx.model.device)
    q_out = idx.model(**q_inputs)
    q_embs = q_out.embeddings
    scores = idx.processor.score_retrieval(q_embs, all_embs)[0].tolist()
    order = sorted(list(enumerate(scores)), key=lambda x: x[1], reverse=True)[:CFG.TOP_K]

    # gather top page paths
    page_paths = [all_pages[i][1] for i,_ in order]
    gen = MedGemmaGen()
    out = gen.generate(query, page_paths)
    out.update({
        "index_mode": index_mode,
        "hits": [{
            "rank": r+1,
            "case": cases[all_pages[i][0]].name,
            "path": str(all_pages[i][1]),
            "score": float(s)
        } for r,(i,s) in enumerate(order)]
    })
    # LƯU FILE (global)
    out_path = _save_ask_result(None, index_mode, out)
    out["saved_to"] = str(out_path)
    return out

def finetune_case(case_dir: Path, max_per_qa: Optional[int] = 1):
    # pick JSONLs that match your naming style
    jsonls = sorted(case_dir.glob("*case*_questions*.jsonl"))
    if not jsonls:
        raise FileNotFoundError(f"No JSONL in {case_dir}")
    items = []
    for j in jsonls:
        ds = MultiImageSFTDataset(j, list_pages_full(case_dir), max_per_qa=max_per_qa).to_hfds()
        items.extend(ds)
    ds_all = Dataset.from_list(list(items))
    qlora = MedGemmaQLoRA()
    out_dir = case_dir / "_adapters" / f"medgemma4b_{_slug(case_dir.name)}_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    qlora.train(ds_all, out_dir)
    print(f"[INFO] Saved LoRA adapter → {out_dir}  (samples={len(ds_all)})")

def finetune_all(cases: List[Path], max_per_qa: Optional[int] = 1):
    ds = MultiCaseSFTDataset(cases, max_per_qa=max_per_qa).to_hfds()
    if len(ds) == 0:
        raise RuntimeError("No samples found across cases.")
    qlora = MedGemmaQLoRA()
    out_dir = CFG.OUTPUT_DIR / f"medgemma4b_ALL_{len(cases)}cases_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    qlora.train(ds, out_dir)
    print(f"[INFO] Saved LoRA adapter → {out_dir}  (samples={len(ds)})")

def finetune_some(case_frags: List[str], max_per_qa: Optional[int] = 1, dry_run: bool = False):
    """
    Train on a subset of cases specified by fragments.
    If dry_run=True, only show what would be trained without actually training.
    """
    # Map each fragment to exactly one case directory
    picked: List[Path] = []
    for frag in case_frags:
        hits = find_cases_by_fragment(frag)
        if len(hits) == 0:
            raise SystemExit(f"Fragment '{frag}' matches no cases.")
        elif len(hits) > 1:
            print(f"Fragment '{frag}' matches {len(hits)} cases:")
            for h in hits[:5]:  # Show first 5 matches
                print(f"  - {h.name}")
            raise SystemExit(f"Fragment '{frag}' must resolve to exactly one case. Please be more specific.")
        picked.append(hits[0])
    
    # Show what we found
    print(f"\n[INFO] Found {len(picked)} case(s) to process:")
    for i, p in enumerate(picked, 1):
        jsonls = sorted(p.glob("*case*_questions*.jsonl"))
        print(f"  {i}. {p.name}")
        if jsonls:
            for j in jsonls:
                print(f"      - {j.name}")
        else:
            print(f"      - NO JSONL FILES FOUND!")
    
    if dry_run:
        print("\n[DRY RUN] Checking dataset size...")
        # Count samples that would be created
        total_samples = 0
        for c in picked:
            full_pages = list_pages_full(c)
            js = sorted(c.glob("*case*_questions*.jsonl"))
            for j in js:
                try:
                    with j.open("r") as f:
                        qa_count = sum(1 for line in f if line.strip())
                    # Each Q/A can generate up to max_per_qa samples
                    samples_from_file = qa_count * (1 if max_per_qa else 3)  # estimate
                    total_samples += samples_from_file
                    print(f"    {j.name}: ~{samples_from_file} samples from {qa_count} Q/A pairs")
                except Exception as e:
                    print(f"    Error reading {j.name}: {e}")
        
        print(f"\n[DRY RUN] Would create approximately {total_samples} training samples")
        print("[DRY RUN] No actual training performed. Remove --dry-run to train.")
        return

    # Build dataset from selected cases only
    ds = MultiCaseSFTDataset(picked, max_per_qa=max_per_qa).to_hfds()
    if len(ds) == 0:
        raise RuntimeError("No samples found across selected cases.")
    
    print(f"\n[INFO] Created dataset with {len(ds)} samples")
    
    # Train
    qlora = MedGemmaQLoRA()
    tag = "_".join([_slug(p.name)[:16] for p in picked])
    out_dir = CFG.OUTPUT_DIR / f"medgemma4b_SOME_{len(picked)}cases_{tag}_lora"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Starting training...")
    qlora.train(ds, out_dir)
    print(f"[INFO] Saved LoRA adapter → {out_dir}  (samples={len(ds)})")

# ------------------------------
# CLI
# ------------------------------
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
    p = argparse.ArgumentParser(description="MedGemma-4B + ColQwen2 (Multi-Case, Local Cache)")
    sub = p.add_subparsers(dest="cmd", required=True)

    # list
    sub.add_parser("list-cases", help="List discovered cases under EXTRACT_ROOT")
    sub.add_parser("list-jsonl", help="List JSONLs per case")

    # embed
    e = sub.add_parser("embed", help="Build/load embeddings for a case or all cases")
    e.add_argument("--case", type=str, help="Case name fragment to match (optional if --all)")
    e.add_argument("--all", action="store_true", help="Process all cases")
    e.add_argument("--index", choices=["full","keep","keep+uncertain"], default="keep")

    # ask
    a = sub.add_parser("ask", help="Retrieve top-k and answer with MedGemma")
    a.add_argument("--q", required=True, help="Question")
    a.add_argument("--case", type=str, help="Case name fragment (omit to use --all)")
    a.add_argument("--all", action="store_true", help="Search across all cases")
    a.add_argument("--index", choices=["full","keep","keep+uncertain"], default="keep")

    # finetune
    f1 = sub.add_parser("finetune", help="Finetune on one case JSONL(s)")
    f1.add_argument("--case", required=True, type=str, help="Case name fragment")
    f1.add_argument("--max-per-qa", type=int, default=1, help="Images per Q/A to explode into samples (None for all)")
    
    f2 = sub.add_parser("finetune-all", help="Finetune on all cases' JSONL(s)")
    f2.add_argument("--max-per-qa", type=int, default=1)
    
    # NEW: finetune-some command
    fs = sub.add_parser("finetune-some", help="Finetune on specific cases (safer than finetune-all)")
    fs.add_argument("--cases", required=True, type=str, 
                    help="Semicolon-separated case fragments, e.g. 'orientalis;martiniquensis;Cambodia'")
    fs.add_argument("--max-per-qa", type=int, default=1, 
                    help="Images per Q/A to use (default=1, use -1 for all)")
    fs.add_argument("--dry-run", action="store_true", 
                    help="Show what would be trained without actually training")

    args = p.parse_args()

    if args.cmd == "list-cases":
        for c in discover_cases(): print(c)
    elif args.cmd == "list-jsonl":
        for c in discover_cases():
            js = sorted(c.glob("*case*_questions*.jsonl"))
            print(f"{c.name}: {len(js)} JSONL")
            for j in js: print("  -", j.name)
    elif args.cmd == "embed":
        cases = discover_cases() if args.all else (find_cases_by_fragment(args.case) if args.case else [])
        if not cases: raise SystemExit("No cases matched. Use --all or --case '<fragment>'.")
        embed_cases(cases, args.index)
    elif args.cmd == "ask":
        if args.all:
            cases = discover_cases()
            if not cases: raise SystemExit("No cases found under EXTRACT_ROOT.")
            print(json.dumps(ask_across_all(args.q, cases, args.index), indent=2, ensure_ascii=False))
        else:
            cases = find_cases_by_fragment(args.case) if args.case else []
            if len(cases) != 1:
                raise SystemExit("Please specify a --case fragment that resolves to exactly one case.")
            print(json.dumps(ask_in_case(args.q, cases[0], args.index), indent=2, ensure_ascii=False))
    elif args.cmd == "finetune":
        cases = find_cases_by_fragment(args.case)
        if len(cases) != 1:
            raise SystemExit("Provide --case that resolves to exactly one case.")
        finetune_case(cases[0], max_per_qa=None if args.max_per_qa < 0 else args.max_per_qa)
    elif args.cmd == "finetune-all":
        cases = discover_cases()
        if not cases: raise SystemExit("No cases found under EXTRACT_ROOT.")
        finetune_all(cases, max_per_qa=None if args.max_per_qa < 0 else args.max_per_qa)
    elif args.cmd == "finetune-some":
        # Parse semicolon-separated fragments
        frags = [s.strip() for s in args.cases.split(";") if s.strip()]
        if not frags:
            raise SystemExit("Provide at least one case fragment via --cases.")
        finetune_some(frags, 
                     max_per_qa=None if args.max_per_qa < 0 else args.max_per_qa,
                     dry_run=args.dry_run)