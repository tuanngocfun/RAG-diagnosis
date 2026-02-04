"""
Multimodal Evaluation Runner - Handles Q3 Image-Only Queries

This module extends run_evaluation.py to support TRUE multimodal evaluation:
- Q1/Q2: Text queries → text retrieval (existing behavior)
- Q3: Image queries → image/caption retrieval (NEW)

Aligns with MMed-RAG (2024/2025) evaluation protocol for medical VQA.

BUG FIXES per GPT 5.2 review:
- qid now uses case_id::query_type to avoid collision
- Expanded qrels mapping for unique qids
- Added image path existence check with skip reason tracking
- Fixed mutable default args antipattern
- Fixed double JSON parse
- Added auto catalog update
"""
import json
import csv
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

from .config import (
    DATA_ROOT, TRAIN_JSONL, RUNS_DIR,
    EVAL_CONFIG, CHUNK_CONFIG
)
from .evaluator import evaluate_retrieval, RetrievalResults
from .retriever import Lane1Retriever, Lane2Retriever, rrf_fusion
from .reranker import get_medcpt_reranker


def extract_query_focused_snippet(case_text: str, query: str, max_chars: int = 2500) -> str:
    """
    Extract query-relevant portion of case text with diagnosis coverage guarantee.
    
    Enhanced version per GPT 5.2 "head+tail+must-include" strategy:
    1. MUST-INCLUDE: Sentences with diagnosis-bearing keywords (forced)
    2. TAIL: Last 800 chars (diagnosis often at end of case reports)
    3. QUERY-FOCUSED: BM25-style term overlap for remaining budget
    
    Args:
        case_text: Full case report text
        query: Query text to match against
        max_chars: Maximum characters to return (default increased to 2500)
    
    Returns:
        Query-focused excerpt guaranteeing diagnosis coverage
    """
    if not case_text:
        return ""
    
    if not query:
        return case_text[:max_chars]
    
    # Split into sentences
    sentences = re.split(r'(?<=[.!?])\s+', case_text)
    if not sentences:
        return case_text[:max_chars]
    
    # 1. MUST-INCLUDE: Sentences with diagnosis-bearing keywords
    diagnosis_patterns = [
        r'leishman\w*', r'kala-azar', r'rk39', r'amastigot', 
        r'donovan\s*bod', r'confirmed', r'diagnos(?:ed|is)', 
        r'positive\s+(?:for|result)', r'visceral', r'cutaneous',
        r'bone\s*marrow', r'PCR\s+(?:test|positive|showed)'
    ]
    must_include = set()
    for sent in sentences:
        if any(re.search(p, sent, re.IGNORECASE) for p in diagnosis_patterns):
            must_include.add(sent)
    
    # 2. TAIL: Last portion of text (diagnosis often at end)
    tail_chars = 800
    tail_text = case_text[-tail_chars:] if len(case_text) > tail_chars else ""
    tail_sentences = set()
    if tail_text:
        # Find sentences that appear in the tail
        for sent in sentences:
            if sent in tail_text or tail_text.find(sent[:50]) >= 0:
                tail_sentences.add(sent)
    
    # 3. QUERY-FOCUSED: Standard BM25-style overlap
    query_terms = set(re.findall(r'\b\w{3,}\b', query.lower()))
    scored = []
    for sent in sentences:
        if sent in must_include or sent in tail_sentences:
            continue  # Already included via other mechanisms
        sent_terms = set(re.findall(r'\b\w{3,}\b', sent.lower()))
        overlap = len(query_terms & sent_terms)
        scored.append((overlap, sent))
    scored.sort(key=lambda x: -x[0])
    
    # 4. COMBINE: must-include first, then tail, then query-focused
    combined = set(must_include) | tail_sentences
    remaining_chars = max_chars - sum(len(s) + 1 for s in combined)
    
    # Add query-focused sentences within remaining budget
    for _, sent in scored:
        if remaining_chars <= len(sent) + 1:
            break
        combined.add(sent)
        remaining_chars -= len(sent) + 1
    
    # Return in ORIGINAL document order for coherence
    original_order = [s for s in sentences if s in combined]
    result = ' '.join(original_order)
    
    # Final safety: truncate if still over budget
    return result[:max_chars] if len(result) > max_chars else result


# ==============================================================================
# TYPE-AWARE SOFT RERANK (per GPT 5.2 Level B)
# Purpose: Reduce harm from wrong-subtype contexts
# WARNING: Uses query keywords to infer subtype, NOT ground truth (data leakage)
# ==============================================================================

def infer_query_subtype(query_text: str) -> tuple:
    """
    Infer Leishmaniasis subtype from query symptoms.
    
    Per Gemini 3 Pro: Do NOT use ground truth labels - that's data leakage.
    Instead, use high-precision keyword patterns from the query text.
    
    Returns:
        (subtype: str, confidence: float)
        subtype is one of: 'Visceral', 'Cutaneous', 'Mucocutaneous', 'PKDL', 'Unknown'
    """
    if not query_text:
        return ('Unknown', 0.0)
    
    q = query_text.lower()
    
    # High-confidence patterns (per Gemini 3 Pro recommendation)
    # These are explicit mentions or pathognomonic symptoms
    
    # PKDL: Very specific patterns
    if any(k in q for k in ['post-kala', 'pkdl', 'post kala-azar', 'dermal leishmaniasis']):
        return ('PKDL', 0.95)
    
    # Mucocutaneous: Mucosal involvement is distinctive
    if any(k in q for k in ['mucosal', 'mucocutaneous', 'mcl', 'nasal septum', 
                             'nasal mucosa', 'palate ulcer', 'oronasal', 'espundia']):
        return ('Mucocutaneous', 0.9)
    
    # Visceral: Systemic symptoms with organ involvement
    visceral_patterns = ['splenomegaly', 'hepatomegaly', 'pancytopenia', 'kala-azar',
                        'visceral leishmaniasis', 'bone marrow aspirate', 'ld bodies',
                        'leishman-donovan', 'fever.*weight loss.*spleen']
    if any(k in q for k in visceral_patterns[:7]):  # Explicit mentions
        return ('Visceral', 0.85)
    if re.search(r'fever.{0,50}(spleen|hepato)', q):  # Pattern-based
        return ('Visceral', 0.7)
    
    # Cutaneous: Skin lesions without mucosal involvement
    cutaneous_patterns = ['cutaneous leishmaniasis', 'skin ulcer', 'skin nodule',
                          'papule', 'oriental sore', 'skin lesion', 'delhi boil']
    if any(k in q for k in cutaneous_patterns):
        # Check NOT mucosal (to differentiate from MCL)
        if not any(m in q for m in ['mucosa', 'nasal', 'palate']):
            return ('Cutaneous', 0.7)
    
    # Low confidence: Generic skin mention (could be CL or MCL)
    if 'ulcer' in q or 'nodule' in q or 'lesion' in q:
        return ('Cutaneous', 0.4)  # Low confidence - no filtering
    
    # Unknown: No clear signal - don't filter
    return ('Unknown', 0.0)


def infer_doc_subtype(doc_text: str) -> str:
    """
    Infer subtype from document/context text.
    Uses same patterns but returns just the type (for corpus annotation).
    """
    subtype, conf = infer_query_subtype(doc_text)
    return subtype if conf > 0.5 else 'Unknown'


def soft_rerank_by_subtype(
    contexts: List[Dict], 
    query_text: str,
    train_cases: Dict = None
) -> List[Dict]:
    """
    Soft rerank contexts by subtype match/mismatch.
    
    Per GPT 5.2 Level B approach:
    - If query subtype inferred with high confidence:
      - Boost matching subtype docs (+0.005)
      - Penalize mismatching docs (-0.003 * confidence)
    - If low confidence: return unchanged (don't risk filtering correct docs)
    
    Args:
        contexts: List of context dicts with 'doc_id', 'score', 'text'
        query_text: Query text to infer subtype from
        train_cases: Optional dict of train cases for additional metadata
    
    Returns:
        Reranked contexts list
    """
    query_subtype, query_conf = infer_query_subtype(query_text)
    
    # Don't rerank if confidence too low
    if query_subtype == 'Unknown' or query_conf < 0.5:
        return contexts
    
    reranked = []
    for ctx in contexts:
        original_score = ctx.get('score', 0.0)
        adjustment = 0.0
        
        # Infer doc subtype from context text
        doc_subtype = infer_doc_subtype(ctx.get('text', ''))
        
        # Also check train_cases metadata if available
        if train_cases and ctx.get('doc_id') in train_cases:
            case_meta = train_cases[ctx['doc_id']]
            # Use explicit label if available
            if case_meta.get('diagnosis_type'):
                doc_subtype = case_meta['diagnosis_type']
        
        # Apply adjustment
        if doc_subtype != 'Unknown':
            if query_subtype.lower() in doc_subtype.lower() or doc_subtype.lower() in query_subtype.lower():
                adjustment = +0.005  # Small boost for match
            else:
                adjustment = -0.003 * query_conf  # Penalty scales with confidence
        
        reranked.append({
            **ctx,
            'score': original_score + adjustment,
            '_subtype_match': f'{query_subtype}({query_conf:.2f}) vs {doc_subtype}'
        })
    
    # Sort by adjusted score
    reranked.sort(key=lambda x: x['score'], reverse=True)
    
    return reranked


@dataclass
class MultimodalRunConfig:
    """Configuration for multimodal evaluation run."""
    run_id: str
    qrels_file: str
    query_types: List[str]
    retriever_method: str
    rerank: bool
    k_values: List[int]
    created_at: str
    image_search_mode: str


def run_multimodal_evaluation(
    qrels_file: str = "qrels.json",
    query_types: Optional[List[str]] = None,
    method: str = "hybrid",
    rerank: bool = False,
    k_values: Optional[List[int]] = None,
    run_id: Optional[str] = None,
    image_search_mode: str = "captions"
) -> Path:
    """
    Run multimodal evaluation supporting both text and image queries.
    
    Args:
        qrels_file: QRELs filename in DATA_ROOT
        query_types: List of query types to evaluate (default: Q1 + Q3)
        method: Retrieval method for text queries
        rerank: Whether to use MedCPT reranking (text only)
        k_values: K values for metrics (default: [5, 10])
        run_id: Optional run ID (auto-generated if None)
        image_search_mode: "captions" or "images" for Q3 retrieval
    
    Returns:
        Path to run directory
    """
    # Handle mutable default args properly (per GPT 5.2)
    # Added Q1_Q3_multimodal_diagnosis per Claude 4.5 + Grok 4.1 review
    if query_types is None:
        query_types = ["Q1_symptom_only", "Q3_image_only", "Q1_Q3_multimodal_diagnosis"]
    if k_values is None:
        k_values = [5, 10]
    
    # Setup
    if run_id is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"multimodal_{timestamp}"
    
    run_dir = RUNS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Load qrels (keyed by case_id) - from SPLIT_DIR (verified dataset)
    from .config import SPLIT_DIR
    with open(SPLIT_DIR / qrels_file) as f:
        original_qrels = json.load(f)
    
    # Load versioned queries from SPLIT_DIR (verified dataset)
    from .config import DATASET_VERSION
    queries_file = SPLIT_DIR / f"eval_queries_{DATASET_VERSION}.jsonl"
    if not queries_file.exists():
        # Fallback to legacy path
        queries_file = SPLIT_DIR / "eval_queries.jsonl"
        print(f"Warning: Using legacy queries file: {queries_file}")
    
    with open(queries_file) as f:
        all_queries = [json.loads(l) for l in f]
    
    # Filter queries by type AND existence in qrels

    queries = [
        q for q in all_queries 
        if q["query_type"] in query_types and q["case_id"] in original_qrels
    ]
    
    # CRITICAL FIX: Expand qrels to use unique qid (case_id::query_type)
    # This prevents Q1/Q3 from overwriting each other (per GPT 5.2)
    expanded_qrels = {}
    for q in queries:
        case_id = q["case_id"]
        query_type = q["query_type"]
        unique_qid = f"{case_id}::{query_type}"
        # Map the same relevant docs to the unique qid
        expanded_qrels[unique_qid] = original_qrels.get(case_id, [])
    
    # Load train cases (fixed: single JSON parse per line)
    with open(TRAIN_JSONL) as f:
        train_cases = {}
        for line in f:
            obj = json.loads(line)
            train_cases[obj["case_id"]] = obj
    
    # Load test cases for ground truth and query images (per GPT 5.2)
    from . import TEST_JSONL, DATA_ROOT as DR
    IMAGES_DIR = DR / "images"
    with open(TEST_JSONL) as f:
        test_cases = {}
        for line in f:
            obj = json.loads(line)
            test_cases[obj["case_id"]] = obj
    
    # Save config
    config = MultimodalRunConfig(
        run_id=run_id,
        qrels_file=qrels_file,
        query_types=query_types,
        retriever_method=method,
        rerank=rerank,
        k_values=k_values,
        created_at=datetime.now().isoformat(),
        image_search_mode=image_search_mode
    )
    with open(run_dir / "run_config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    
    # Save queries
    with open(run_dir / "queries.json", "w") as f:
        json.dump(queries, f, indent=2)
    
    # Initialize retrievers
    lane1 = Lane1Retriever()
    lane2 = Lane2Retriever()
    reranker_model = get_medcpt_reranker() if rerank else None
    
    # Run retrieval
    all_retrieved = {}
    retrieval_records = []
    
    # Enhanced statistics with skip reasons (per GPT 5.2)
    stats = {
        "Q1": {"attempted": 0, "success": 0},
        "Q2": {"attempted": 0, "success": 0},
        "Q3": {"attempted": 0, "success": 0},
        "skip_reasons": {
            "no_text": 0,
            "no_image_path": 0,
            "image_not_found": 0,
            "no_results": 0
        }
    }
    
    for q in queries:
        case_id = q["case_id"]
        query_type = q["query_type"]
        # Support both legacy (text) and new (clinical_context) field naming
        query_text = q.get("text") or q.get("clinical_context") or q.get("formatted_query", "")
        # Support both legacy (image_path) and new (query_images) field naming
        image_path = q.get("image_path")
        if not image_path:
            query_images_list = q.get("query_images", [])
            image_path = query_images_list[0] if query_images_list else None
        
        # CRITICAL FIX: Use unique qid to prevent collision (per GPT 5.2)
        qid = f"{case_id}::{query_type}"
        
        results = []
        stage = "unknown"
        skip_reason = None
        
        # === Q3: Image-only query (TRUE MULTIMODAL) ===
        # Support both legacy (Q3_image_only) and new (Q3_image_diagnosis) naming
        if query_type in ["Q3_image_only", "Q3_image_diagnosis"]:
            stats["Q3"]["attempted"] += 1
            
            if not image_path:
                stats["skip_reasons"]["no_image_path"] += 1
                skip_reason = "no_image_path"
            elif not Path(image_path).exists():
                stats["skip_reasons"]["image_not_found"] += 1
                skip_reason = f"image_not_found: {image_path}"
            else:
                # Use image encoder to retrieve
                search_captions = (image_search_mode == "captions")
                raw_results = lane2.retrieve_by_image_path(
                    image_path, 
                    top_k=20,
                    search_captions=search_captions
                )
                results = [(cid, score) for cid, score, _ in raw_results]
                stage = f"biomedclip_image→{image_search_mode}"
                
                if results:
                    stats["Q3"]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # === Q1/Q2: Text query ===
        # Support both legacy (Q1_symptom_only) and new (Q1_diagnosis) naming
        elif query_type in ["Q1_symptom_only", "Q2_symptom_exposure", "Q1_diagnosis", "Q2_diagnosis_exposure"]:
            stat_key = "Q1" if "Q1" in query_type else "Q2"
            stats[stat_key]["attempted"] += 1
            
            if not query_text:
                stats["skip_reasons"]["no_text"] += 1
                skip_reason = "no_text"
            else:
                if method == "bm25":
                    results = lane1.retrieve_bm25(query_text, top_k=20)
                elif method == "e5":
                    results = lane1.retrieve_e5(query_text, top_k=20)
                elif method == "hybrid":
                    results = lane1.retrieve_hybrid(query_text, top_k=20)
                elif method == "2lane":
                    l1_results = lane1.retrieve_hybrid(query_text, top_k=20)
                    l2_results = lane2.retrieve_by_caption(query_text, top_k=20)
                    l2_results = [(c, s) for c, s, _ in l2_results]
                    results = rrf_fusion(l1_results, l2_results)[:20]
                else:
                    results = lane1.retrieve_hybrid(query_text, top_k=20)
                
                stage = method
                
                # Rerank if requested (text queries only)
                if rerank and reranker_model and results:
                    candidates = []
                    for cid, score in results:
                        if cid in train_cases:
                            doc_text = train_cases[cid].get("case_text", "")[:512]
                            candidates.append((cid, doc_text, score))
                    
                    if candidates:
                        reranked = reranker_model.rerank(query_text, candidates, top_k=20)
                        results = [(c, s) for c, _, s in reranked]
                        stage = f"{method}+rerank"
                
                if results:
                    stats[stat_key]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # === Q1_Q3: Combined Multimodal Query (per Claude 4.5 + Grok 4.1) ===
        elif query_type == "Q1_Q3_multimodal_diagnosis":
            # Initialize multimodal stats if not present
            if "MULTIMODAL" not in stats:
                stats["MULTIMODAL"] = {"attempted": 0, "success": 0}
            stats["MULTIMODAL"]["attempted"] += 1
            
            # Get image path from query if available
            query_images_list = q.get("query_images", [])
            first_image = query_images_list[0] if query_images_list else None
            
            if not query_text and not first_image:
                stats["skip_reasons"]["no_text"] += 1
                skip_reason = "no_text_or_image"
            else:
                # Combine text and image retrieval using 2-lane fusion
                text_results = []
                image_results = []
                
                # Lane 1: Text retrieval
                if query_text:
                    text_results = lane1.retrieve_hybrid(query_text, top_k=20)
                
                # Lane 2: Image retrieval (if image available)
                if first_image and Path(first_image).exists():
                    raw_img_results = lane2.retrieve_by_image_path(
                        first_image, 
                        top_k=20,
                        search_captions=(image_search_mode == "captions")
                    )
                    image_results = [(cid, score) for cid, score, _ in raw_img_results]
                
                # Combine with RRF fusion
                if text_results and image_results:
                    results = rrf_fusion(text_results, image_results)[:20]
                    stage = "hybrid+biomedclip_fusion"
                elif text_results:
                    results = text_results[:20]
                    stage = "hybrid_text_only"
                elif image_results:
                    results = image_results[:20]
                    stage = "biomedclip_image_only"
                else:
                    results = []
                
                if results:
                    stats["MULTIMODAL"]["success"] += 1
                else:
                    stats["skip_reasons"]["no_results"] += 1
                    skip_reason = "no_results"
        
        # Skip if no results or error
        if not results:
            if skip_reason:
                retrieval_records.append({
                    "qid": qid,
                    "query_type": query_type,
                    "query": (query_text[:100] if query_text else f"[IMAGE]"),
                    "skip_reason": skip_reason,
                    "contexts": [],
                    "stage": "skipped"
                })
            continue
        
        # Store results with unique qid
        all_retrieved[qid] = [c for c, _ in results]
        
        # Build retrieval record with SEPARATE query vs context images (per GPT 5.2)
        contexts = []
        context_images = []
        for cid, score in results[:10]:
            if cid in train_cases:
                ctx_case = train_cases[cid]
                # Use query-focused snippet extraction (per GPT 5.2)
                snippet = extract_query_focused_snippet(
                    ctx_case.get("case_text", ""),
                    query_text or "",
                    max_chars=1500  # Increased from 500
                )
                contexts.append({
                    "doc_id": cid,
                    "score": float(score),
                    "text": snippet,
                    "meta": {"has_images": len(ctx_case.get("images", [])) > 0}
                })
                # Collect context images (from TRAIN cases)
                for img in ctx_case.get("images", []):
                    filename = img.get("file") or img.get("file_name", "")
                    if filename:
                        context_images.append(str(IMAGES_DIR / cid / filename))
        
        # APPLY TYPE-AWARE SOFT RERANK (per GPT 5.2 Level B)
        # Reduces harm from wrong-subtype contexts without hard filtering
        # DISABLED: Investigation shows this may hurt more than help
        # TODO: Fix subtype inference before re-enabling
        # if contexts and query_text:
        #     contexts = soft_rerank_by_subtype(contexts, query_text, train_cases)
        
        # Extract query images and ground truth from TEST case (per GPT 5.2)
        query_images = []
        ground_truth = None
        if case_id in test_cases:
            test_case = test_cases[case_id]
            # Query images from TEST case
            for img in test_case.get("images", []):
                filename = img.get("file") or img.get("file_name", "")
                if filename:
                    img_path = IMAGES_DIR / case_id / filename
                    if img_path.exists():
                        query_images.append(str(img_path))
            # Ground truth
            ground_truth = {
                "diagnosis": test_case.get("diagnosis", ""),
                "diagnosis_type": test_case.get("diagnosis_type", ""),
                "species": test_case.get("species", "")
            }
        
        # Build proper query for RAGAS: question + clinical context
        # FIXED: Previously only had clinical_context, RAGAS needs actual question to evaluate relevance
        question = q.get("question", "What is the diagnosis?")
        full_query_for_ragas = f"{question}\n\nClinical Context: {query_text[:300]}" if query_text else question
        
        retrieval_records.append({
            "qid": qid,
            "query_type": query_type,
            "query": full_query_for_ragas,  # FIXED: Now includes question for RAGAS
            "clinical_context": query_text[:200] if query_text else "",  # Preserved for reference
            "contexts": contexts,
            "query_images": query_images,        # NEW: from TEST case
            "context_images": context_images[:5], # NEW: from TRAIN cases
            "ground_truth": ground_truth,         # NEW: for diagnosis accuracy
            "stage": stage
        })
    
    # Save retrieval.jsonl
    with open(run_dir / "retrieval.jsonl", "w") as f:
        for rec in retrieval_records:
            f.write(json.dumps(rec) + "\n")
    
    # Evaluate using expanded qrels (with unique qids)
    eval_results = evaluate_retrieval(all_retrieved, expanded_qrels, k_values, return_per_query=True)
    
    # Save per-query metrics
    with open(run_dir / "metrics_per_query.csv", "w", newline="") as f:
        writer = csv.writer(f)
        header = ["qid", "query_type"] + [f"recall@{k}" for k in k_values] + \
                 [f"ndcg@{k}" for k in k_values] + \
                 [f"precision@{k}" for k in k_values] + ["mrr", "ap"]
        writer.writerow(header)
        
        for qid_full, metrics in (eval_results.per_query_metrics or {}).items():
            # Extract query_type from qid
            parts = qid_full.split("::")
            qtype = parts[1] if len(parts) > 1 else "unknown"
            row = [qid_full, qtype]
            for k in k_values:
                row.append(f"{metrics.get(f'recall@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'ndcg@{k}', 0):.4f}")
            for k in k_values:
                row.append(f"{metrics.get(f'precision@{k}', 0):.4f}")
            row.append(f"{metrics.get('mrr', 0):.4f}")
            row.append(f"{metrics.get('ap', 0):.4f}")
            writer.writerow(row)
    
    # Save summary (grounded_accuracy added per Grok 4.1 recommendation)
    summary = {
        "run_id": run_id,
        "query_types": query_types,
        "query_stats": stats,
        "n_queries": len(queries),
        "n_retrieved": len(all_retrieved),
        "metrics": {
            "recall": {f"@{k}": eval_results.recall_at_k.get(k, 0) for k in k_values},
            "ndcg": {f"@{k}": eval_results.ndcg_at_k.get(k, 0) for k in k_values},
            "precision": {f"@{k}": eval_results.precision_at_k.get(k, 0) for k in k_values},
            "mrr": eval_results.mrr,
            "map": eval_results.map_score,
            "grounded_accuracy": None  # Populated after RAGAS eval
        }
    }
    
    # NOTE: RAGAS metrics are added separately via rag/update_summary_ragas.py
    # This keeps the pipeline modular - run retrieval first, then RAGAS evaluation separately
    
    with open(run_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Print results (with safe key access)
    print(f"\n{'='*60}")
    print(f"MULTIMODAL EVALUATION: {run_id}")
    print(f"{'='*60}")
    print(f"Query Types: {query_types}")
    print(f"Query Stats:")
    for qt in ["Q1", "Q2", "Q3", "MULTIMODAL"]:
        if qt in stats:
            print(f"  {qt}: {stats[qt]['success']}/{stats[qt]['attempted']} successful")
    print(f"Skip Reasons: {stats['skip_reasons']}")
    print(f"Method: {method} (rerank={rerank})")
    print(f"\nMetrics:")
    for k in k_values:
        print(f"  nDCG@{k}:     {eval_results.ndcg_at_k.get(k, 0):.4f}")
    print(f"  MRR:        {eval_results.mrr:.4f}")
    print(f"\n✓ Run saved to {run_dir}")
    
    # Auto-update catalog (per GPT 5.2)
    try:
        from .run_catalog import update_catalog
        update_catalog()
    except Exception as e:
        print(f"Note: Could not update catalog: {e}")
    
    return run_dir


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Multimodal RAG Evaluation")
    parser.add_argument("--qrels", default="qrels.json", help="QRELs file")
    parser.add_argument("--query-types", nargs="+", 
                        default=None,
                        help="Query types to evaluate (default: Q1 + Q3)")
    parser.add_argument("--method", default="hybrid", 
                        choices=["bm25", "e5", "hybrid", "2lane"],
                        help="Retrieval method for text queries")
    parser.add_argument("--rerank", action="store_true", help="Use MedCPT reranking")
    parser.add_argument("--run-id", default=None, help="Custom run ID")
    parser.add_argument("--image-search", default="captions", 
                        choices=["captions", "images"],
                        help="Collection to search for Q3 image queries")
    
    args = parser.parse_args()
    
    run_multimodal_evaluation(
        qrels_file=args.qrels,
        query_types=args.query_types,
        method=args.method,
        rerank=args.rerank,
        run_id=args.run_id,
        image_search_mode=args.image_search
    )
