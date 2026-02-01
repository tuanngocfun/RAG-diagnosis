#!/bin/bash
# =============================================================================
# No-RAG Baseline Pipeline (For Comparison with RAG)
# =============================================================================
#
# This script runs the NO-RAG baseline evaluation:
# 1. Generate answers using LLM parametric knowledge ONLY (no retrieval)
# 2. Run RAGAS evaluation (Diagnosis Accuracy, etc.)
#
# Purpose: Compare against RAG results to measure RAG value
#
# Usage:
#   chmod +x run_norag_pipeline.sh
#   ./run_norag_pipeline.sh [OPTIONS]
#
# Options:
#   --generator TYPE    Generator type:
#                         gemini   - Gemini 2.5 Pro (default, cloud API)
#                         gemma3   - Gemma3 12B/27B (local)
#                         medgemma - MedGemma 4B IT (local, medical-tuned)
#   --variant SIZE      Model variant:
#                         For gemma3: 12b (default), 27b
#                         For others: ignored
#   --run-id ID         Custom run ID (default: auto-generated)
#
# Examples:
#   ./run_norag_pipeline.sh --generator gemini         # Gemini 2.5 Pro No-RAG
#   ./run_norag_pipeline.sh --generator medgemma       # MedGemma No-RAG
#   ./run_norag_pipeline.sh --generator gemma3 --variant 12b
#
# =============================================================================

set -e

# Colors
GREEN='\033[0;32m'
CYAN='\033[0;36m'
NC='\033[0m'

# Default options
GENERATOR="gemini"
VARIANT=""
RUN_ID=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            head -35 "$0" | tail -30
            exit 0
            ;;
        --generator) GENERATOR="$2"; shift 2 ;;
        --variant) VARIANT="$2"; shift 2 ;;
        --run-id) RUN_ID="$2"; shift 2 ;;
        *) echo "Unknown option: $1. Use --help for usage."; exit 1 ;;
    esac
done

# Set default variant
if [ -z "$VARIANT" ]; then
    case $GENERATOR in
        gemini) VARIANT="2.5-pro" ;;
        gemma3) VARIANT="12b" ;;
        medgemma) VARIANT="4b" ;;
    esac
fi

# Auto-generate run ID
if [ -z "$RUN_ID" ]; then
    DATE=$(date +%Y%m%d_%H%M%S)
    RUN_ID="${GENERATOR}_norag_${DATE}"
fi

# Display generator info
case $GENERATOR in
    gemini) GEN_INFO="Gemini 2.5 Pro (Cloud API)" ;;
    gemma3) GEN_INFO="Gemma3 ${VARIANT} (Local)" ;;
    medgemma) GEN_INFO="MedGemma 4B IT (Local, Medical)" ;;
esac

echo -e "${GREEN}============================================================${NC}"
echo -e "${GREEN}  NO-RAG BASELINE PIPELINE                                  ${NC}"
echo -e "${GREEN}  Generate answers WITHOUT retrieval (parametric only)      ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Configuration:"
echo "  Run ID:     $RUN_ID"
echo "  Generator:  $GEN_INFO"
echo "  Mode:       NO-RAG (no retrieval, no contexts)"
echo ""

# Set working directory
cd "$(dirname "$0")/../.."

# Activate venv if exists
if [ -f "data/venv/bin/activate" ]; then
    source data/venv/bin/activate
fi

# =============================================================================
# Run No-RAG Pipeline
# =============================================================================
echo -e "\n${CYAN}[STEP 1/2] Generating answers (NO-RAG)...${NC}"

python3 -m rag.pipeline.run_baseline_norag \
    --run-id "$RUN_ID" \
    --generator "$GENERATOR" \
    --variant "$VARIANT" \
    --query-types Q1_diagnosis Q3_image_diagnosis Q1_Q3_multimodal_diagnosis \
    2>&1 | tee "rag/instructions/process/14/${RUN_ID}.txt"

# =============================================================================
# SUMMARY
# =============================================================================
echo -e "\n${GREEN}============================================================${NC}"
echo -e "${GREEN}  No-RAG Baseline Complete!                                ${NC}"
echo -e "${GREEN}============================================================${NC}"
echo ""
echo "Run ID: ${RUN_ID}"
echo "Results: rag/runs/${RUN_ID}/"
echo "Log:     rag/instructions/process/14/${RUN_ID}.txt"
echo ""
echo "Output files:"
echo "  - answers_norag.jsonl:  Generated answers (no retrieval)"
echo "  - answers_gemini.jsonl: RAGAS-compatible format"
echo "  - ragas.jsonl:          RAGAS evaluation scores"
echo ""
echo "Compare with RAG run:"
echo "  ./run_full_pipeline.sh --generator $GENERATOR --run-id ${GENERATOR}_rag_compare"
