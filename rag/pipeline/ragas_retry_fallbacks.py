"""
Retry RAGAS evaluation for samples affected by Gemini 429 errors.

KEY FEATURE: Rotates across ALL Google API keys in .env to maximize
available quota. Each key has its own daily limit, so using 30 keys
gives ~30x more quota.

Identifies samples needing re-evaluation by TWO criteria:
1. diagnosis_method == "string_match_fallback" (diagnosis LLM call failed)
2. ALL 3 multimodal metrics == 0.0 (faithfulness, relevance, context_relevance)
   indicating the LLM metric calls failed due to rate limiting

Usage:
  python -m rag.pipeline.ragas_retry_fallbacks rag/runs/gemma3_4b_rag_v163 --delay 2.0
  python -m rag.pipeline.ragas_retry_fallbacks rag/runs/gemma3_4b_norag_v163 --delay 2.0
  # Use flash model for higher quota:
  python -m rag.pipeline.ragas_retry_fallbacks rag/runs/gemma3_4b_rag_v163 --model gemini-2.0-flash
"""
import json
import asyncio
import os
import sys
import re
from pathlib import Path
from datetime import datetime, timezone

# Add RAGAS source to path
RAGAS_SRC = Path(__file__).parent.parent.parent / "ragas" / "src"
if RAGAS_SRC.exists() and str(RAGAS_SRC) not in sys.path:
    sys.path.insert(0, str(RAGAS_SRC))

from pipeline.ragas_evaluator import RAGAsLibraryEvaluator
from dataclasses import asdict


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_retry_state(state_path: Path) -> dict:
    """Load persistent retry state from disk."""
    if not state_path.exists():
        return {"version": 1, "runs": 0, "qids": {}}
    try:
        with open(state_path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"version": 1, "runs": 0, "qids": {}}
        data.setdefault("version", 1)
        data.setdefault("runs", 0)
        data.setdefault("qids", {})
        return data
    except Exception:
        # Corrupted state should not block evaluation.
        return {"version": 1, "runs": 0, "qids": {}}


def _save_retry_state(state_path: Path, state: dict) -> None:
    """Persist retry state atomically."""
    tmp_path = state_path.with_suffix(state_path.suffix + ".tmp")
    with open(tmp_path, "w") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp_path.replace(state_path)


def load_all_google_api_keys(only_primary_key: bool = False) -> list:
    """
    Load Google API keys from .env file.

    If only_primary_key=True, return only GOOGLE_API_KEY.
    """
    env_path = Path(__file__).parent.parent.parent / ".env"
    keys = []
    
    if env_path.exists():
        with open(env_path) as f:
            for line in f:
                line = line.strip()
                if line.startswith("#") or "=" not in line:
                    continue
                # Match GOOGLE_API_KEY, GOOGLE_API_KEY0, GOOGLE_API_KEY1, etc.
                match = re.match(r'^GOOGLE_API_KEY\d*\s*=\s*"?([^"]+)"?$', line)
                if match:
                    key = match.group(1).strip()
                    if key and key.startswith("AIza"):
                        keys.append(key)

    if only_primary_key:
        # Prefer the dedicated purchased key entry and ignore rotation pool.
        primary = os.getenv("GOOGLE_API_KEY", "").strip()
        if primary and primary.startswith("AIza"):
            return [primary]

        if env_path.exists():
            with open(env_path) as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("#") or "=" not in line:
                        continue
                    match_primary = re.match(r'^GOOGLE_API_KEY\s*=\s*"?([^"]+)"?$', line)
                    if match_primary:
                        key = match_primary.group(1).strip()
                        if key and key.startswith("AIza"):
                            return [key]
        return []
    
    # Deduplicate while preserving order
    seen = set()
    unique_keys = []
    for k in keys:
        if k not in seen:
            seen.add(k)
            unique_keys.append(k)
    
    return unique_keys


def find_samples_needing_retry(results):
    """
    Find all samples that need re-evaluation due to 429 errors.
    
    Returns:
        dict mapping qid -> set of reasons why re-eval is needed
    """
    needs_retry = {}
    
    for r in results:
        qid = r["qid"]
        reasons = set()
        
        # Check 1: Diagnosis fell back to string matching
        if r.get("diagnosis_method") == "string_match_fallback":
            reasons.add("diagnosis_fallback")
        
        # Check 2: For RAG runs (where multimodal metrics should be non-None),
        # all 3 multimodal metrics are 0.0
        f = r.get("multimodal_faithfulness")
        rv = r.get("multimodal_relevance")
        c = r.get("context_relevance")
        
        if f is not None and rv is not None:
            if f == 0.0 and rv == 0.0 and (c == 0.0 or c is None):
                reasons.add("all_multimodal_zero")
        
        if reasons:
            needs_retry[qid] = reasons
    
    return needs_retry


def retry_fallback_samples(
    run_dir: Path,
    answers_file: str = None,
    delay_seconds: float = 2.0,
    judge_model: str = None,
    max_attempts_per_qid: int = 1,
    reset_retry_state: bool = False,
    only_primary_key: bool = False,
):
    """Re-evaluate samples that were affected by 429 errors, rotating API keys."""
    
    ragas_path = run_dir / "ragas.jsonl"
    
    # Auto-detect answers file
    if answers_file:
        answers_path = run_dir / answers_file
    else:
        candidates = ["answers.jsonl", "answers_norag.jsonl", "answers_gemini.jsonl"]
        answers_path = None
        for candidate in candidates:
            p = run_dir / candidate
            if p.exists():
                answers_path = p
                break
        if answers_path is None:
            print(f"ERROR: No answers file found in {run_dir}")
            print(f"  Looked for: {', '.join(candidates)}")
            return
    
    if not ragas_path.exists():
        print(f"ERROR: {ragas_path} not found")
        return
    if not answers_path.exists():
        print(f"ERROR: {answers_path} not found")
        return
    
    # Load all API keys
    all_keys = load_all_google_api_keys(only_primary_key=only_primary_key)
    print(f"Loaded {len(all_keys)} Google API keys for rotation")
    
    if not all_keys:
        print("ERROR: No Google API keys found in .env")
        return
    
    print(f"Using answers file: {answers_path.name}")
    if judge_model:
        print(f"Using judge model: {judge_model}")
    if only_primary_key:
        print("Single-key mode enabled: using only GOOGLE_API_KEY")
    
    # Load existing results
    with open(ragas_path) as f:
        results = [json.loads(l) for l in f]
    
    # Load answers
    with open(answers_path) as f:
        answers = {}
        for line in f:
            sample = json.loads(line)
            answers[sample["qid"]] = sample
    
    # Find all samples needing retry
    needs_retry = find_samples_needing_retry(results)

    state_path = run_dir / "ragas_retry_state.json"
    if reset_retry_state and state_path.exists():
        state_path.unlink()
        print(f"Reset retry state: {state_path.name}")
    retry_state = _load_retry_state(state_path)
    
    # Categorize
    diag_only = [qid for qid, reasons in needs_retry.items() if reasons == {"diagnosis_fallback"}]
    mm_only = [qid for qid, reasons in needs_retry.items() if reasons == {"all_multimodal_zero"}]
    both = [qid for qid, reasons in needs_retry.items() if len(reasons) > 1]
    
    print(f"\nTotal samples: {len(results)}")
    print(f"Samples needing re-eval: {len(needs_retry)}")
    print(f"  - Diagnosis fallback only: {len(diag_only)}")
    print(f"  - All multimodal zero only: {len(mm_only)}")
    print(f"  - Both issues: {len(both)}")
    print()
    
    if not needs_retry:
        print("Nothing to retry!")
        return
    
    # Build index for fast lookup
    result_index = {r["qid"]: i for i, r in enumerate(results)}
    retry_qids = list(needs_retry.keys())

    # Skip qids that already reached retry attempt cap in previous runs.
    capped_qids = []
    filtered_qids = []
    for qid in retry_qids:
        qstate = retry_state.get("qids", {}).get(qid, {})
        attempts = int(qstate.get("attempts", 0))
        if attempts >= max_attempts_per_qid:
            capped_qids.append(qid)
            continue
        filtered_qids.append(qid)
    retry_qids = filtered_qids

    if capped_qids:
        print(f"Skipped by attempt cap (>= {max_attempts_per_qid}): {len(capped_qids)}")
    if not retry_qids:
        print("No samples left after applying retry attempt cap.")
        return
    
    # Key rotation state
    key_index = [0]  # mutable for closure
    consecutive_429 = [0]
    
    def get_next_evaluator():
        """Create a new evaluator with the next API key."""
        key = all_keys[key_index[0] % len(all_keys)]
        key_prefix = key[:10] + "..."
        evaluator = RAGAsLibraryEvaluator(
            model=judge_model,
            api_key=key,
        )
        return evaluator, key_prefix
    
    async def retry_all():
        retried = 0
        still_failed = 0
        rate_limited_failures = 0
        processed = 0
        early_stopped = False
        evaluator, key_prefix = get_next_evaluator()
        
        for j, qid in enumerate(retry_qids):
            if qid not in answers:
                print(f"  WARNING: {qid} not found in answers, skipping")
                continue
            
            sample = answers[qid]
            query = sample["query"]
            answer = sample["answer"]
            contexts = [c.get("text", "") for c in sample.get("contexts", [])]
            query_images = sample.get("query_images", []) or sample.get("image_paths", [])
            context_images = sample.get("context_images", [])
            ground_truth = sample.get("ground_truth", None)
            
            is_norag = not contexts or len(contexts) == 0
            reasons = needs_retry[qid]
            
            try:
                result = await evaluator.evaluate_sample(
                    qid=qid,
                    query=query,
                    answer=answer,
                    contexts=contexts,
                    query_images=query_images,
                    context_images=context_images,
                    ground_truth=ground_truth,
                )
                result_dict = asdict(result)
                processed += 1
                
                # Check if this retry actually improved things
                improved = False
                is_rate_limited = False
                
                if "diagnosis_fallback" in reasons:
                    if result_dict.get("diagnosis_method") == "llm_judge":
                        improved = True
                
                if "all_multimodal_zero" in reasons:
                    new_f = result_dict.get("multimodal_faithfulness", 0)
                    new_r = result_dict.get("multimodal_relevance", 0)
                    new_c = result_dict.get("context_relevance", 0)
                    if new_f is not None and new_r is not None:
                        if not (new_f == 0.0 and new_r == 0.0 and (new_c == 0.0 or new_c is None)):
                            improved = True
                
                if improved:
                    results[result_index[qid]] = result_dict
                    retried += 1
                    status = "✓ improved"
                    consecutive_429[0] = 0
                else:
                    # Check if it was a 429 that caused the failure
                    method = result_dict.get("diagnosis_method", "")
                    reasoning = (result_dict.get("diagnosis_reasoning") or "").upper()
                    if ("429" in reasoning) or ("RESOURCE_EXHAUSTED" in reasoning):
                        is_rate_limited = True
                    if method == "string_match_fallback":
                        consecutive_429[0] += 1
                    else:
                        consecutive_429[0] = 0
                    still_failed += 1
                    status = "✗ same"

                if is_rate_limited:
                    rate_limited_failures += 1

                qstate = retry_state.setdefault("qids", {}).setdefault(qid, {})
                qstate["attempts"] = int(qstate.get("attempts", 0)) + 1
                qstate["last_attempt_at"] = _utc_now_iso()
                qstate["last_status"] = "improved" if improved else "same"
                qstate["last_reason_codes"] = sorted(list(reasons))
                qstate["last_method"] = result_dict.get("diagnosis_method")
                qstate["last_rate_limited"] = bool(is_rate_limited)
                
                diag = result_dict.get("diagnosis_accuracy", "?")
                method = result_dict.get("diagnosis_method", "?")
                f_val = result_dict.get("multimodal_faithfulness", "N/A")
                r_val = result_dict.get("multimodal_relevance", "N/A")
                c_val = result_dict.get("context_relevance", "N/A")
                norag_str = " [NO-RAG]" if is_norag else ""
                print(f"  [{j+1}/{len(retry_qids)}] {qid}: {status} f={f_val}, r={r_val}, c={c_val}, diag={diag}, method={method} [key={key_prefix}]{norag_str}")
                
            except Exception as e:
                err_str = str(e)
                processed += 1
                still_failed += 1
                is_rate_limited = False
                
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    consecutive_429[0] += 1
                    is_rate_limited = True
                    rate_limited_failures += 1
                    print(f"  [{j+1}/{len(retry_qids)}] {qid}: 429 on key={key_prefix}")
                else:
                    consecutive_429[0] = 0
                    print(f"  [{j+1}/{len(retry_qids)}] {qid}: ERROR {err_str[:100]}")

                qstate = retry_state.setdefault("qids", {}).setdefault(qid, {})
                qstate["attempts"] = int(qstate.get("attempts", 0)) + 1
                qstate["last_attempt_at"] = _utc_now_iso()
                qstate["last_status"] = "error"
                qstate["last_reason_codes"] = sorted(list(reasons))
                qstate["last_method"] = "exception"
                qstate["last_rate_limited"] = bool(is_rate_limited)
                qstate["last_error"] = err_str[:200]
            
            # Rotate key after every 429 or every 3 samples (spread load)
            if consecutive_429[0] >= 1 or (j + 1) % 3 == 0:
                key_index[0] += 1
                evaluator, key_prefix = get_next_evaluator()
                if consecutive_429[0] >= 1:
                    print(f"  >> Rotated to key {key_index[0] % len(all_keys) + 1}/{len(all_keys)}: {key_prefix}")
                    consecutive_429[0] = 0
            
            # Rate limit delay
            if delay_seconds > 0 and j < len(retry_qids) - 1:
                await asyncio.sleep(delay_seconds)

            # Early stop when every processed sample in this run is rate-limited and no improvement.
            # This prevents repeated budget burn when all keys are exhausted.
            warmup = min(max(len(all_keys), 3), 10)
            if processed >= warmup and retried == 0 and rate_limited_failures == processed:
                early_stopped = True
                print("\nEarly stop: all processed samples failed with rate limits; likely quota exhausted across keys.")
                break
        
        print(f"\nImproved: {retried}")
        print(f"Still failed: {still_failed}")
        print(f"Rate-limited failures: {rate_limited_failures}")
        if early_stopped:
            print("Stopped early to avoid redundant API usage.")

        retry_state["runs"] = int(retry_state.get("runs", 0)) + 1
        retry_state["last_run_at"] = _utc_now_iso()
        retry_state["last_run_summary"] = {
            "processed": processed,
            "improved": retried,
            "still_failed": still_failed,
            "rate_limited_failures": rate_limited_failures,
            "early_stopped": early_stopped,
            "max_attempts_per_qid": max_attempts_per_qid,
        }
    
    asyncio.run(retry_all())
    
    # Write updated results
    with open(ragas_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    _save_retry_state(state_path, retry_state)
    
    print(f"\n✓ Updated {ragas_path}")
    
    # Print updated aggregates
    n = len(results)
    metrics = ["multimodal_faithfulness", "multimodal_relevance", "context_relevance",
               "diagnosis_accuracy", "diagnosis_type_accuracy"]
    agg = {}
    for m in metrics:
        scores = [r[m] for r in results if r.get(m) is not None]
        agg[m] = sum(scores) / len(scores) if scores else 0.0
    
    llm_count = sum(1 for r in results if r.get("diagnosis_method") == "llm_judge")
    fallback_count = sum(1 for r in results if r.get("diagnosis_method") == "string_match_fallback")
    
    remaining = find_samples_needing_retry(results)
    
    print(f"\n=== Updated Metrics ===")
    print(f"LLM judge: {llm_count}, String match fallback: {fallback_count}")
    print(f"Faithfulness: {agg['multimodal_faithfulness']:.4f}")
    print(f"Relevance: {agg['multimodal_relevance']:.4f}")
    print(f"Context Relevance: {agg['context_relevance']:.4f}")
    print(f"Diagnosis Accuracy: {agg['diagnosis_accuracy']:.4f}")
    print(f"Diagnosis Type Accuracy: {agg['diagnosis_type_accuracy']:.4f}")
    print(f"Remaining samples needing retry: {len(remaining)}")
    print(f"Retry state saved: {state_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Retry 429-errored RAGAS evaluations with API key rotation")
    parser.add_argument("run_dir", type=Path, help="Path to run directory")
    parser.add_argument("--delay", type=float, default=2.0, help="Delay between API calls in seconds")
    parser.add_argument("--model", type=str, default=None, help="Judge model (default: from config, e.g. gemini-2.0-flash for higher quota)")
    parser.add_argument("--max-attempts-per-qid", type=int, default=1,
                        help="Skip qids that already reached this retry count in prior runs (default: 1)")
    parser.add_argument("--reset-retry-state", action="store_true",
                        help="Reset persisted retry state and retry counters for this run")
    parser.add_argument("--only-primary-key", action="store_true",
                        help="Use only GOOGLE_API_KEY from environment/.env (disable key rotation pool)")
    args = parser.parse_args()
    retry_fallback_samples(
        args.run_dir,
        delay_seconds=args.delay,
        judge_model=args.model,
        max_attempts_per_qid=args.max_attempts_per_qid,
        reset_retry_state=args.reset_retry_state,
        only_primary_key=args.only_primary_key,
    )
