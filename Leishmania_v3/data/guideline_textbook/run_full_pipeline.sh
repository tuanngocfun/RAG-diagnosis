#!/bin/bash
# =============================================================================
# Full RAG Pipeline with Extended Training Set (Guidelines + Articles + PubMed)
# =============================================================================
# 
# This script runs the COMPLETE RAG evaluation pipeline:
# 1. Preprocess guidelines → guideline_entries.jsonl
# 2. Integrate → train_extended.jsonl (114 PubMed + 4 articles + 8 guidelines = 126)
# 3. Update config → use extended training set
# 4. Reindex Qdrant → BM25 + E5 (Lane 1) + BiomedCLIP (Lane 2)
# 5. Run multimodal eval → creates retrieval.jsonl with ALL query types
# 6. Generate answers → with specified generator and PromptMode.BALANCED
# 7. Run RAGAS evaluation → MultimodalFaithfulness, Relevance, DiagnosisAccuracy
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh [OPTIONS]
#
# Options:
#   --skip-preprocess   Skip guideline preprocessing (if already done)
#   --skip-index        Skip reindexing (if corpus unchanged)
#   --generator TYPE    Generator type:
#                         gemini  - Gemini 2.5 Pro (default, cloud API)
#                         gemma3  - Gemma3 12B/27B (local)
#                         medgemma - MedGemma 4B IT (local, medical-tuned)
#   --variant SIZE      Model variant:
#                         For gemma3: 12b (default), 27b
#                         For medgemma: 4b (default)
#                         For gemini: ignored
#   --topk K            Top-K for retrieval: 3, 5, 10 (default: 5)
#   --rerank            Enable cross-encoder reranking
#   --run-id ID         Custom run ID (default: auto-generated)
#
# Examples:
#   ./run_full_pipeline.sh --generator gemini           # Gemini 2.5 Pro
#   ./run_full_pipeline.sh --generator gemma3 --variant 12b
#   ./run_full_pipeline.sh --generator medgemma         # MedGemma 4B IT
#
# Prerequisites:
#   pip install pdfplumber sentence-transformers qdrant-client google-genai ragas
# =============================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Default options
SKIP_PREPROCESS=false
SKIP_INDEX=false
GENERATOR="gemini"  # gemini | gemma3 | medgemma
VARIANT=""          # Auto-set based on generator
TOPK=5
RERANK=""
RUN_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            head -40 "$0" | tail -35
            exit 0
            ;;
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --skip-index) SKIP_INDEX=true; shift ;;
        --generator) GENERATOR="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --topk) TOPK="$2"; shift 2 ;;
        --rerank) RERANK="True"; shift ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

# Set default variant based on generator if not specified
if [ -z "$VARIANT" ]; then
    case $GENERATOR in
        gemini) VARIANT="2.5-pro" ;;
        gemma3) VARIANT="12b" ;;
        medgemma) VARIANT="4b" ;;
        *) VARIANT="default" ;;
    esac
fi

# Auto-generate run ID if not provided
if [ -z "$RUN_ID" ]; then
    DATE=$(date +%Y%m%d)
    RUN_ID="${GENERATOR}_${VARIANT}_topk${TOPK}_extended_${DATE}"
fi

# Display generator info
case $GENERATOR in
    gemini) GEN_INFO="Gemini 2.5 Pro (Cloud API)" ;;
    gemma3) GEN_INFO="Gemma3 ${VARIANT} (Local)" ;;
    medgemma) GEN_INFO="MedGemma 4B IT (Local, Medical)" ;;
    *) GEN_INFO="$GENERATOR" ;;
esac

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  RAG Pipeline with Extended Training Set (126 entries)    ${NC}"
echo -e "${GREEN}  PubMed Cases + Articles + CDC/WHO Guidelines             ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Configuration:"
echo "  Run ID:          $RUN_ID"
echo "  Skip preprocess: $SKIP_PREPROCESS"
echo "  Skip indexing:   $SKIP_INDEX"
echo "  Generator:       $GEN_INFO"
echo "  Top-K:           $TOPK"
echo "  Rerank:          ${RERANK:-False}"
echo ""

# Set working directory
cd /home/students/Leishmania/Leishmania_v3
source data/venv/bin/activate

# =============================================================================
# STEP 1: Preprocess Guidelines (if needed)
# =============================================================================
if [ "$SKIP_PREPROCESS" = false ]; then
    echo -e "\n${CYAN}[STEP 1/6] Preprocessing guideline PDFs...${NC}"
    cd data/guideline_textbook
    python3 guideline_preprocessor.py
    python3 integrate_guidelines.py
    cd ../..
else
    echo -e "\n${CYAN}[STEP 1/6] Skipping preprocessing (--skip-preprocess)${NC}"
fi

# =============================================================================
# STEP 2: Update Config to Use Extended Training Set
# =============================================================================
echo -e "\n${CYAN}[STEP 2/6] Configuring training set...${NC}"

if grep -q 'train_extended.jsonl' rag/pipeline/config.py; then
    echo "  Config already uses train_extended.jsonl"
else
    cp rag/pipeline/config.py rag/pipeline/config.py.bak
    sed -i 's|TRAIN_JSONL = SPLIT_DIR / "train.jsonl"|TRAIN_JSONL = SPLIT_DIR / "train_extended.jsonl"|g' rag/pipeline/config.py
    echo "  Updated: TRAIN_JSONL → train_extended.jsonl"
fi

# =============================================================================
# STEP 3: Reindex Qdrant (if needed)
# =============================================================================
if [ "$SKIP_INDEX" = false ]; then
    echo -e "\n${CYAN}[STEP 3/6] Reindexing Qdrant vector database...${NC}"
    python3 -m rag.run_indexing --strategy fixed
else
    echo -e "\n${CYAN}[STEP 3/6] Skipping reindex (--skip-index)${NC}"
fi

# =============================================================================
# STEP 4-6: Run Full Pipeline (Retrieval → Answer → RAGAS)
# =============================================================================
# Following the EXACT workflow from cmd4.txt:
# 1. run_multimodal_evaluation() - creates retrieval.jsonl with ALL query types together
# 2. generate_answers() - generates answers with specified generator
# 3. run_ragas_evaluation() - runs RAGAS evaluation

echo -e "\n${CYAN}[STEP 4-6/6] Running full multimodal pipeline...${NC}"

python3 -c "
from rag.pipeline.run_multimodal_eval import run_multimodal_evaluation
from rag.pipeline.answer_generator import generate_answers
from rag.pipeline.ragas_evaluator import run_ragas_evaluation
from rag.configs.prompt_mode import PromptMode

# Step 4: Run multimodal evaluation (retrieval)
# This handles ALL query types together in one call
run_dir = run_multimodal_evaluation(
    qrels_file='qrels_v143.json',
    run_id='${RUN_ID}',
    method='hybrid',
    query_types=['Q1_diagnosis', 'Q3_image_diagnosis', 'Q1_Q3_multimodal_diagnosis'],
    k_values=[${TOPK}],
    image_search_mode='captions',
    rerank=${RERANK:-False}
)

# Step 5: Generate answers with specified generator
generate_answers(
    run_dir, 
    generator_type='${GENERATOR}',
    model_variant='${VARIANT}',
    prompt_mode=PromptMode.BALANCED
)

# Step 6: Run RAGAS evaluation
run_ragas_evaluation(
    run_dir, 
    answers_file='answers.jsonl', 
    delay_seconds=1.5
)
" 2>&1 | tee rag/instructions/process/14/${RUN_ID}.txt

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  Pipeline Complete!                                        ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Run ID: ${RUN_ID}"
echo "Results: rag/runs/${RUN_ID}/"
echo "Log:     rag/instructions/process/14/${RUN_ID}.txt"
echo ""
echo "Output files:"
echo "  - retrieval.jsonl: Retrieved contexts for all 55 queries"
echo "  - answers.jsonl:   Generated answers"
echo "  - ragas.jsonl:     RAGAS evaluation scores"
echo ""
echo "To restore original config:"
echo "  cp rag/pipeline/config.py.bak rag/pipeline/config.py"
