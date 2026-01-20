"""
RAGAs Evaluator using Official RAGAS Library

Integrates RAGAS metrics directly from Leishmania_v3/ragas/src for Q1 journal quality.
Following claude45opus_guide.md for correct Collections API usage.

Metrics:
- Generation: MultiModalFaithfulness, MultiModalRelevance
- Retrieval: ContextRelevance

Uses Gemini 2.5 Pro as the vision-capable LLM judge.
"""
import os
import sys
import json
import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict, field

# Add RAGAS source to path
RAGAS_SRC = Path(__file__).parent.parent.parent / "ragas" / "src"
if RAGAS_SRC.exists() and str(RAGAS_SRC) not in sys.path:
    sys.path.insert(0, str(RAGAS_SRC))

# Ensure HF cache is set before imports
os.environ.setdefault("TRANSFORMERS_CACHE", "/data4t/hf/transformers")

from .config import GOOGLE_API_KEY, JUDGE_MODEL


@dataclass
class RAGAsResult:
    """Result from RAGAs evaluation using official RAGAS library."""
    qid: str
    # Generation metrics
    multimodal_faithfulness: Optional[float] = None
    multimodal_relevance: Optional[float] = None
    # Retrieval metrics
    context_relevance: Optional[float] = None
    # Diagnosis accuracy metrics (NEW: per GPT 5.2 recommendation)
    diagnosis_accuracy: Optional[float] = None
    diagnosis_type_accuracy: Optional[float] = None
    diagnosis_reasoning: Optional[str] = None  # LLM explanation
    diagnosis_method: str = "llm_judge"  # "llm_judge" or "string_match_fallback"
    # Metadata
    judge_model: str = ""
    traces: Dict = field(default_factory=dict)
    error: Optional[str] = None


# =============================================================================
# DIAGNOSIS ACCURACY (per GPT 5.2: main metric for diagnostic RAG)
# =============================================================================

import re

def normalize_diagnosis(diagnosis: str) -> str:
    """
    Normalize diagnosis string for comparison.
    
    - Lowercase
    - Remove punctuation
    - Normalize common variants (leishmaniasis -> leishmania)
    """
    if not diagnosis:
        return ""
    
    normalized = diagnosis.lower().strip()
    # Remove punctuation except hyphens
    normalized = re.sub(r'[^\w\s-]', '', normalized)
    # Normalize whitespace
    normalized = ' '.join(normalized.split())
    
    return normalized


# Legacy function - kept for backward compatibility but deprecated
def calculate_diagnosis_accuracy(
    predicted_answer: str,
    ground_truth: Optional[Dict]
) -> Dict[str, float]:
    """
    DEPRECATED: Use evaluate_diagnosis_equivalence_llm() for Q1-standard evaluation.
    
    This string-matching approach is NOT clinically acceptable per:
    - Dinc et al. (2025): LLM judges achieve Kappa=0.852 with specialists
    - ASTRID framework: requires semantic understanding
    
    Kept for backward compatibility only.
    """
    result = {
        "diagnosis_accuracy": 0.0,
        "diagnosis_type_accuracy": 0.0
    }
    
    if not ground_truth:
        return result
    
    gt_diagnosis = ground_truth.get("diagnosis", "")
    gt_type = ground_truth.get("diagnosis_type", "")
    
    if not gt_diagnosis and not gt_type:
        return result
    
    # Normalize predicted answer
    pred_normalized = normalize_diagnosis(predicted_answer)
    gt_diagnosis_normalized = normalize_diagnosis(gt_diagnosis)
    gt_type_normalized = normalize_diagnosis(gt_type)
    
    # Check if ground truth diagnosis appears in prediction
    if gt_diagnosis_normalized and gt_diagnosis_normalized in pred_normalized:
        result["diagnosis_accuracy"] = 1.0
    # Also check for leishmaniasis mentions
    elif "leishmania" in pred_normalized and "leishmania" in gt_diagnosis_normalized:
        result["diagnosis_accuracy"] = 0.5  # Partial match
    
    # Check diagnosis type (CL, VL, MCL, PKDL)
    type_keywords = {
        "cl": ["cutaneous", "cl ", " cl"],
        "vl": ["visceral", "vl ", " vl", "kala-azar", "kala azar"],
        "mcl": ["mucocutaneous", "mcl ", " mcl"],
        "pkdl": ["post-kala-azar", "post kala-azar", "pkdl"],
    }
    
    if gt_type_normalized:
        gt_type_key = gt_type_normalized.lower()
        if gt_type_key in type_keywords:
            for keyword in type_keywords[gt_type_key]:
                if keyword in pred_normalized:
                    result["diagnosis_type_accuracy"] = 1.0
                    break
        # Direct match
        elif gt_type_normalized in pred_normalized:
            result["diagnosis_type_accuracy"] = 1.0
    
    return result


# =============================================================================
# LLM-BASED DIAGNOSIS EQUIVALENCE (Q1 Standard - Dinc et al., 2025)
# =============================================================================

# Diagnosis equivalence prompt following Q1 paper methodology
# Updated per Claude 4.5 + Grok 4.1 review: stricter differential scoring, hedge penalty, species matching
DIAGNOSIS_EQUIVALENCE_PROMPT = """You are a specialist medical evaluator assessing diagnostic accuracy.

## Ground Truth Diagnosis
- **Primary Diagnosis**: {gt_diagnosis}
- **Diagnosis Type**: {gt_type}
- **Species** (if applicable): {gt_species}

## Predicted Answer
{prediction}

## Evaluation Criteria (per Dinc et al., 2025 + Grok 4.1 review)

Assess if the predicted answer contains the CORRECT diagnosis based on:

1. **EXACT MATCH (1.0)**: The prediction explicitly states the ground truth diagnosis as PRIMARY
   - Example: GT="Visceral Leishmaniasis", Pred="Primary Diagnosis: Visceral Leishmaniasis" → 1.0

2. **SYNONYM/CLINICALLY EQUIVALENT (1.0)**: Different terms for the same condition as PRIMARY
   - "Kala-azar" = "Visceral Leishmaniasis" → 1.0
   - "Cutaneous Leishmaniasis" = "Oriental Sore" = "Baghdad Boil" → 1.0
   - "L. donovani infection" = "Visceral Leishmaniasis" → 1.0

3. **CORRECT IN DIFFERENTIAL (ranked score)**: If prediction lists differentials:
   - GT is #1 in differential list (but not stated as primary) → 0.75
   - GT is #2 or #3 in differential list → 0.5
   - GT is #4 or lower in differential list → 0.25

4. **HEDGE WITH CORRECT MENTION (0.5)**: Model hedges but mentions correct answer
   - Prediction says "insufficient evidence" or "cannot determine" or "unclear"
   - BUT mentions the correct diagnosis somewhere in the answer
   - This indicates possible parametric knowledge usage

5. **PARTIAL CREDIT (0.5)**: Correct disease family but wrong subtype
   - GT="Cutaneous Leishmaniasis", Pred="Leishmaniasis (type unspecified)" → 0.5
   - GT="Visceral Leishmaniasis", Pred="Mucocutaneous Leishmaniasis" → 0.5

6. **INCORRECT (0.0)**: Wrong diagnosis or unrelated condition
   - GT="Leishmaniasis", Pred="Malaria" → 0.0
   - GT="Cutaneous Leishmaniasis", Pred="Psoriasis" → 0.0

## Type-Specific Matching
- CL = Cutaneous Leishmaniasis
- VL = Visceral Leishmaniasis = Kala-azar
- MCL = Mucocutaneous Leishmaniasis = Espundia
- PKDL = Post-Kala-azar Dermal Leishmaniasis

## Species Matching (for diagnosis_type_score adjustment)
- Exact species match or equivalent (L. infantum ≈ L. donovani for VL): no penalty
- Species not mentioned but type correct: multiply type_score by 0.9
- Wrong species mentioned: multiply type_score by 0.5

## Output (JSON only)
Respond with ONLY valid JSON:
{{
    "diagnosis_score": <0.0 | 0.25 | 0.5 | 0.75 | 1.0>,
    "diagnosis_type_score": <0.0 | 0.25 | 0.5 | 0.75 | 1.0>,
    "reasoning": "<brief clinical explanation including species assessment>"
}}"""


@dataclass
class DiagnosisEquivalenceResult:
    """Result from LLM-based diagnosis equivalence evaluation."""
    diagnosis_score: float
    diagnosis_type_score: float
    reasoning: str
    method: str = "llm_judge"  # "llm_judge" or "string_match" (fallback)


class RAGAsLibraryEvaluator:
    """
    RAGAs evaluator using official RAGAS library from Leishmania_v3/ragas/src.
    Following claude45opus_guide.md for Collections API usage.
    
    Generation Metrics:
    - MultiModalFaithfulness: Binary (0/1) - Is response grounded in multimodal context?
    - MultiModalRelevance: Binary (0/1) - Is response relevant to query and context?
    
    Retrieval Metrics:
    - ContextRelevance: Continuous (0-1) - Are contexts relevant to query?
    """
    
    def __init__(
        self,
        model: str = None,
        api_key: str = None
    ):
        """
        Initialize RAGAS evaluator with Gemini LLM.
        
        Args:
            model: Model name (default: from config)
            api_key: Google API key (default: from config)
        """
        self.model_name = model or JUDGE_MODEL
        self.api_key = api_key or GOOGLE_API_KEY
        
        # Lazy initialization of RAGAS components
        self._llm = None
        self._metrics_initialized = False
        
        # Metric instances (created on first use)
        self._multimodal_faithfulness = None
        self._multimodal_relevance = None
        self._context_relevance = None
    
    def _init_llm(self):
        """Initialize RAGAS LLM wrapper for Gemini."""
        if self._llm is not None:
            return
        
        try:
            from google import genai
            from ragas.llms import llm_factory
            
            # Create Gemini client using new google-genai SDK
            client = genai.Client(api_key=self.api_key)
            
            # Create RAGAS LLM wrapper using llm_factory as per guide
            self._llm = llm_factory(
                model=self.model_name,
                provider="google",
                client=client,
                temperature=0.0,  # Deterministic for evaluation
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import RAGAS or google-genai. "
                f"Ensure RAGAS is available at {RAGAS_SRC}.\n"
                f"Error: {e}"
            )
    
    def _init_metrics(self):
        """Initialize RAGAS metric instances using Collections API."""
        if self._metrics_initialized:
            return
        
        self._init_llm()
        
        try:
            # Import from Collections API as per claude45opus_guide.md
            from ragas.metrics.collections import (
                MultiModalFaithfulness,
                MultiModalRelevance,
                ContextRelevance,
            )
            
            # Initialize metrics with the LLM
            self._multimodal_faithfulness = MultiModalFaithfulness(llm=self._llm)
            self._multimodal_relevance = MultiModalRelevance(llm=self._llm)
            self._context_relevance = ContextRelevance(llm=self._llm)
            
            self._metrics_initialized = True
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import RAGAS metrics. Check RAGAS installation.\n"
                f"Error: {e}"
            )
    
    async def evaluate_multimodal_faithfulness(
        self,
        response: str,
        retrieved_contexts: List[str]
    ) -> float:
        """
        Evaluate multimodal faithfulness using RAGAS Collections API.
        
        Args:
            response: The generated response to evaluate
            retrieved_contexts: List of text contexts or image paths
        
        Returns: 1.0 if faithful, 0.0 if not (Binary)
        """
        self._init_metrics()
        
        try:
            result = await self._multimodal_faithfulness.ascore(
                response=response,
                retrieved_contexts=retrieved_contexts
            )
            return result.value
        except Exception as e:
            print(f"MultiModalFaithfulness error: {e}")
            return 0.0
    
    async def evaluate_multimodal_relevance(
        self,
        user_input: str,
        response: str,
        retrieved_contexts: List[str]
    ) -> float:
        """
        Evaluate multimodal relevance using RAGAS Collections API.
        
        Args:
            user_input: The user's question/query
            response: The generated response to evaluate
            retrieved_contexts: List of text contexts or image paths
        
        Returns: 1.0 if relevant, 0.0 if not (Binary)
        """
        self._init_metrics()
        
        try:
            result = await self._multimodal_relevance.ascore(
                user_input=user_input,
                response=response,
                retrieved_contexts=retrieved_contexts
            )
            return result.value
        except Exception as e:
            print(f"MultiModalRelevance error: {e}")
            return 0.0
    
    async def evaluate_context_relevance(
        self,
        user_input: str,
        retrieved_contexts: List[str]
    ) -> float:
        """
        Evaluate context relevance using RAGAS Collections API.
        
        Args:
            user_input: The user's question/query
            retrieved_contexts: List of retrieved text contexts
        
        Returns: Continuous score 0.0-1.0
        """
        self._init_metrics()
        
        try:
            result = await self._context_relevance.ascore(
                user_input=user_input,
                retrieved_contexts=retrieved_contexts
            )
            return result.value
        except Exception as e:
            print(f"ContextRelevance error: {e}")
            return 0.0
    
    async def evaluate_diagnosis_equivalence(
        self,
        prediction: str,
        ground_truth: Dict,
        use_llm_judge: bool = True
    ) -> DiagnosisEquivalenceResult:
        """
        Evaluate diagnosis accuracy using LLM-as-Judge (Q1 Standard).
        
        This method follows the methodology from:
        - Dinc et al. (2025) "Comparative Analysis of LLMs" - Kappa=0.852 with specialists
        - ASTRID framework - semantic understanding for clinical terms
        
        The LLM judge can recognize:
        - Synonyms (Kala-azar = Visceral Leishmaniasis)
        - Clinical equivalence (L. donovani infection = VL)
        - Differential diagnosis (GT in top-3 list)
        - Partial credit (correct family, wrong subtype)
        
        Args:
            prediction: The model's generated answer
            ground_truth: Dict with 'diagnosis', 'diagnosis_type', 'species'
            use_llm_judge: If True, use LLM; if False, fall back to string matching
        
        Returns:
            DiagnosisEquivalenceResult with scores and reasoning
        """
        # Fallback to legacy string matching if requested
        if not use_llm_judge:
            legacy_result = calculate_diagnosis_accuracy(prediction, ground_truth)
            return DiagnosisEquivalenceResult(
                diagnosis_score=legacy_result["diagnosis_accuracy"],
                diagnosis_type_score=legacy_result["diagnosis_type_accuracy"],
                reasoning="[Legacy string matching]",
                method="string_match"
            )
        
        self._init_llm()
        
        # Format the prompt
        prompt = DIAGNOSIS_EQUIVALENCE_PROMPT.format(
            gt_diagnosis=ground_truth.get("diagnosis", "Unknown"),
            gt_type=ground_truth.get("diagnosis_type", "Unknown"),
            gt_species=ground_truth.get("species", "Not specified"),
            prediction=prediction[:2000]  # Truncate very long answers
        )
        
        try:
            # Use the google genai client directly for this call
            from google import genai
            
            client = genai.Client(api_key=self.api_key)
            
            response = client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config={
                    "temperature": 0.0,  # Deterministic
                    "response_mime_type": "application/json",
                }
            )
            
            # Parse JSON response
            import json
            result_text = response.text.strip()
            
            # Handle potential JSON parsing issues
            try:
                result = json.loads(result_text)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                import re
                json_match = re.search(r'\{[^{}]*\}', result_text, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                else:
                    raise ValueError(f"Could not parse JSON: {result_text[:200]}")
            
            return DiagnosisEquivalenceResult(
                diagnosis_score=float(result.get("diagnosis_score", 0.0)),
                diagnosis_type_score=float(result.get("diagnosis_type_score", 0.0)),
                reasoning=result.get("reasoning", "No reasoning provided"),
                method="llm_judge"
            )
            
        except Exception as e:
            print(f"LLM diagnosis evaluation error: {e}")
            # Fallback to string matching
            legacy_result = calculate_diagnosis_accuracy(prediction, ground_truth)
            return DiagnosisEquivalenceResult(
                diagnosis_score=legacy_result["diagnosis_accuracy"],
                diagnosis_type_score=legacy_result["diagnosis_type_accuracy"],
                reasoning=f"[Fallback to string matching due to: {str(e)[:100]}]",
                method="string_match_fallback"
            )
    
    async def evaluate_sample(
        self,
        qid: str,
        query: str,
        answer: str,
        contexts: List[str],
        query_images: Optional[List[str]] = None,
        context_images: Optional[List[str]] = None,
        ground_truth: Optional[Dict] = None,
        # Legacy parameter - use query_images instead
        image_paths: Optional[List[str]] = None
    ) -> RAGAsResult:
        """
        Evaluate a single sample with all RAGAS metrics.
        
        CRITICAL FIX (per GPT 5.2):
        - query_images: Images from TEST case (for visual query understanding)
        - context_images: Images from TRAIN cases (retrieved contexts)
        - For multimodal metrics, use query_images (the actual test case images)
        
        Args:
            qid: Query ID
            query: User query (user_input)
            answer: Generated answer (response)
            contexts: Retrieved text contexts
            query_images: Images from TEST case (NEW - use this)
            context_images: Images from retrieved TRAIN cases
            ground_truth: Dict with diagnosis, diagnosis_type, species
            image_paths: DEPRECATED - use query_images
        
        Returns:
            RAGAsResult with all metric scores including diagnosis accuracy
        """
        # Handle legacy image_paths parameter
        if query_images is None and image_paths is not None:
            query_images = image_paths
        
        result = RAGAsResult(
            qid=qid,
            judge_model=self.model_name,
            traces={
                "has_query_images": bool(query_images),
                "has_context_images": bool(context_images),
                "has_ground_truth": bool(ground_truth)
            }
        )
        
        # For multimodal metrics: Use query images (from TEST case) + text contexts
        # NOT context_images (from TRAIN cases) per GPT 5.2 trap detection
        retrieved_contexts = contexts.copy()
        if query_images:
            retrieved_contexts.extend(query_images)
        
        # Run metrics SEQUENTIALLY to avoid rate limiting (GPT 5.2 advice)
        try:
            result.multimodal_faithfulness = await self.evaluate_multimodal_faithfulness(
                response=answer,
                retrieved_contexts=retrieved_contexts
            )
        except Exception as e:
            print(f"MultiModalFaithfulness error: {e}")
        
        try:
            result.multimodal_relevance = await self.evaluate_multimodal_relevance(
                user_input=query,
                response=answer,
                retrieved_contexts=retrieved_contexts
            )
        except Exception as e:
            print(f"MultiModalRelevance error: {e}")
        
        try:
            result.context_relevance = await self.evaluate_context_relevance(
                user_input=query,
                retrieved_contexts=contexts  # Only text for context relevance
            )
        except Exception as e:
            print(f"ContextRelevance error: {e}")
        
        # Calculate diagnosis accuracy using LLM-as-Judge (Q1 Standard)
        # Per Dinc et al. (2025): LLM judges achieve Kappa=0.852 with specialists
        if ground_truth:
            try:
                diag_result = await self.evaluate_diagnosis_equivalence(
                    prediction=answer,
                    ground_truth=ground_truth,
                    use_llm_judge=True
                )
                result.diagnosis_accuracy = diag_result.diagnosis_score
                result.diagnosis_type_accuracy = diag_result.diagnosis_type_score
                result.diagnosis_reasoning = diag_result.reasoning
                result.diagnosis_method = diag_result.method
            except Exception as e:
                print(f"Diagnosis equivalence error: {e}")
                # Fallback to legacy string matching
                legacy_scores = calculate_diagnosis_accuracy(answer, ground_truth)
                result.diagnosis_accuracy = legacy_scores["diagnosis_accuracy"]
                result.diagnosis_type_accuracy = legacy_scores["diagnosis_type_accuracy"]
                result.diagnosis_method = "string_match_fallback"
        
        # Flag parametric knowledge usage (per Claude 4.5 + Grok 4.1 analysis)
        # If context is irrelevant but diagnosis is correct, LLM is using pre-trained knowledge
        if result.context_relevance is not None and result.diagnosis_accuracy is not None:
            if result.context_relevance < 0.2 and result.diagnosis_accuracy >= 0.8:
                result.traces["parametric_knowledge_suspected"] = True
                result.traces["grounded_accuracy"] = 0.0  # Not grounded in retrieval
            else:
                result.traces["parametric_knowledge_suspected"] = False
                result.traces["grounded_accuracy"] = result.diagnosis_accuracy
        
        return result


def run_ragas_evaluation(
    run_dir: Path,
    answers_file: str = "answers.jsonl",
    judge_model: str = None,
    max_samples: int = None,
    delay_seconds: float = 1.0,
    resume: bool = True,
) -> Path:
    """
    Run RAGAS evaluation on generated answers using official RAGAS library.
    
    Implements GPT 5.2's rate limit handling recommendations:
    - Resume: Skip already evaluated samples (saves quota)
    - Sequential: Run metrics one at a time (avoid concurrent 429s)
    - Delay: Sleep between samples (1-2s recommended)
    - Judge model: Use flash model for lower quota usage
    
    Args:
        run_dir: Path to run directory
        answers_file: Filename of answers JSONL
        judge_model: Optional judge model (default: gemini-2.5-pro, use 'gemini-1.5-flash' for lower quota)
        max_samples: Optional limit on samples to evaluate (for debugging)
        delay_seconds: Delay between samples in seconds (default: 1.0)
        resume: If True, skip samples already in ragas.jsonl
    
    Returns:
        Path to ragas.jsonl output
    """
    evaluator = RAGAsLibraryEvaluator(model=judge_model)
    
    # Load answers
    answers_path = run_dir / answers_file
    if not answers_path.exists():
        raise FileNotFoundError(f"No {answers_file} in {run_dir}")
    
    with open(answers_path) as f:
        samples = [json.loads(l) for l in f]
    
    # Resume: Load existing results and skip already evaluated samples
    output_path = run_dir / "ragas.jsonl"
    existing_results: Dict[str, dict] = {}
    completed_qids: Set[str] = set()
    
    if resume and output_path.exists():
        with open(output_path) as f:
            for line in f:
                r = json.loads(line)
                existing_results[r["qid"]] = r
                # Only skip if all metrics are non-null (fully evaluated)
                if (r.get("multimodal_faithfulness") is not None and
                    r.get("multimodal_relevance") is not None and
                    r.get("context_relevance") is not None):
                    completed_qids.add(r["qid"])
        if completed_qids:
            print(f"  Resume: Skipping {len(completed_qids)} already completed samples")
    
    # Limit samples for debugging
    if max_samples:
        samples = samples[:max_samples]
        print(f"  Limiting to {max_samples} samples (debug mode)")
    
    async def evaluate_all():
        results = []
        evaluated_count = 0
        skipped_count = 0
        
        for i, sample in enumerate(samples):
            qid = sample["qid"]
            
            # Resume: Skip completed samples
            if qid in completed_qids:
                results.append(existing_results[qid])
                skipped_count += 1
                continue
            
            query = sample["query"]
            answer = sample["answer"]
            contexts = [c.get("text", "") for c in sample.get("contexts", [])]
            
            # NEW: Use query_images (TEST case) not image_paths (TRAIN cases)
            query_images = sample.get("query_images", [])
            context_images = sample.get("context_images", [])
            ground_truth = sample.get("ground_truth", None)
            
            # Legacy support: fall back to image_paths if query_images not present
            if not query_images:
                query_images = sample.get("image_paths", [])
            
            result = await evaluator.evaluate_sample(
                qid=qid,
                query=query,
                answer=answer,
                contexts=contexts,
                query_images=query_images,
                context_images=context_images,
                ground_truth=ground_truth
            )
            results.append(asdict(result))
            evaluated_count += 1
            
            # Progress (include diagnosis accuracy if available)
            diag_str = ""
            if result.diagnosis_accuracy is not None:
                diag_str = f", diag={result.diagnosis_accuracy:.1f}"
            print(f"  [{i + 1}/{len(samples)}] {qid}: f={result.multimodal_faithfulness}, r={result.multimodal_relevance}, c={result.context_relevance}{diag_str}")
            
            # Rate limit delay (GPT 5.2 recommendation)
            if delay_seconds > 0 and i < len(samples) - 1:
                await asyncio.sleep(delay_seconds)
        
        print(f"\n  Evaluated: {evaluated_count}, Skipped: {skipped_count}")
        return results
    
    # Run async evaluation
    results = asyncio.run(evaluate_all())
    
    # Save results
    output_path = run_dir / "ragas.jsonl"
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    
    # Compute and print aggregates
    n = len(results)
    metrics = ["multimodal_faithfulness", "multimodal_relevance", "context_relevance",
               "diagnosis_accuracy", "diagnosis_type_accuracy"]
    
    agg = {}
    for m in metrics:
        scores = [r[m] for r in results if r.get(m) is not None]
        agg[m] = sum(scores) / len(scores) if scores else 0.0
    
    print(f"\n✓ RAGAS evaluation saved to {output_path}")
    print(f"  Using: Official RAGAS library (Collections API)")
    print(f"\n  --- Generation Metrics ---")
    print(f"  Multimodal Faithfulness: {agg['multimodal_faithfulness']:.4f}")
    print(f"  Multimodal Relevance: {agg['multimodal_relevance']:.4f}")
    print(f"\n  --- Retrieval Metrics ---")
    print(f"  Context Relevance: {agg['context_relevance']:.4f}")
    print(f"\n  --- Diagnosis Accuracy (MAIN METRIC) ---")
    print(f"  Diagnosis Accuracy: {agg['diagnosis_accuracy']:.4f}")
    print(f"  Diagnosis Type Accuracy: {agg['diagnosis_type_accuracy']:.4f}")
    
    return output_path


if __name__ == "__main__":
    # Test initialization
    print("Testing RAGAS Library Evaluator...")
    evaluator = RAGAsLibraryEvaluator()
    print(f"✓ Evaluator created with model: {evaluator.model_name}")
    print(f"✓ RAGAS source path: {RAGAS_SRC}")
    
    # Test metric initialization
    try:
        evaluator._init_metrics()
        print("✓ All RAGAS metrics initialized successfully")
        print("  - MultiModalFaithfulness (Binary 0/1)")
        print("  - MultiModalRelevance (Binary 0/1)")
        print("  - ContextRelevance (Continuous 0-1)")
    except Exception as e:
        print(f"✗ Failed to initialize metrics: {e}")
