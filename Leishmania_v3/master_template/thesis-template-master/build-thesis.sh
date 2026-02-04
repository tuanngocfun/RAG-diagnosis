#!/bin/bash
# ============================================================================
# Thesis LaTeX Compilation Script using Docker
# ============================================================================
# This script builds and runs a Docker container to compile thesis.tex
# without requiring sudo access to install LaTeX packages locally.
#
# Usage:
#   ./build-thesis.sh          # Build image + compile thesis
#   ./build-thesis.sh compile  # Just compile (image must exist)
#   ./build-thesis.sh clean    # Clean auxiliary files
#   ./build-thesis.sh shell    # Open interactive shell in container
#   ./build-thesis.sh help     # Show help
#
# Author: Auto-generated for Leishmania RAG Thesis Project
# ============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE_NAME="thesis-latex"
DOCKERFILE="Dockerfile.thesis"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

show_help() {
    echo "============================================================"
    echo "Thesis LaTeX Compilation Script using Docker"
    echo "============================================================"
    echo ""
    echo "Usage: ./build-thesis.sh [command]"
    echo ""
    echo "Commands:"
    echo "  build     Build Docker image (required first time)"
    echo "  compile   Compile thesis.tex using latexmk"
    echo "  quick     Quick compile with pdflatex (no bibliography)"
    echo "  full      Full compile with pdflatex + biber/bibtex"
    echo "  clean     Clean auxiliary files"
    echo "  shell     Open interactive bash shell in container"
    echo "  help      Show this help message"
    echo ""
    echo "Default (no args): build + compile"
    echo ""
    echo "Examples:"
    echo "  ./build-thesis.sh              # First time: build image + compile"
    echo "  ./build-thesis.sh compile      # Subsequent runs: just compile"
    echo "  ./build-thesis.sh clean        # Remove *.aux, *.log, etc."
    echo ""
}

build_image() {
    print_info "Building Docker image '$IMAGE_NAME'..."
    docker build -t "$IMAGE_NAME" -f "$DOCKERFILE" .
    print_success "Docker image built successfully!"
}

check_image_exists() {
    if ! docker image inspect "$IMAGE_NAME" > /dev/null 2>&1; then
        print_warning "Docker image '$IMAGE_NAME' not found. Building..."
        build_image
    fi
}

compile_thesis() {
    check_image_exists
    print_info "Compiling thesis.tex with latexmk..."
    docker run --rm -v "$(pwd)":/thesis "$IMAGE_NAME" \
        latexmk -pdf -interaction=nonstopmode -file-line-error thesis.tex
    
    if [ -f "thesis.pdf" ]; then
        print_success "Compilation complete! Output: thesis.pdf"
        ls -lh thesis.pdf
    else
        print_error "Compilation failed - thesis.pdf not created"
        exit 1
    fi
}

quick_compile() {
    check_image_exists
    print_info "Quick compile with pdflatex..."
    docker run --rm -v "$(pwd)":/thesis "$IMAGE_NAME" \
        pdflatex -interaction=nonstopmode thesis.tex
    print_success "Quick compile done!"
}

full_compile() {
    check_image_exists
    print_info "Full compilation with bibliography..."
    docker run --rm -v "$(pwd)":/thesis "$IMAGE_NAME" \
        sh -c "
            echo '=== Pass 1: pdflatex ===' &&
            pdflatex -interaction=nonstopmode thesis.tex &&
            echo '=== Pass 2: biber/bibtex ===' &&
            (biber thesis 2>/dev/null || bibtex thesis) &&
            echo '=== Pass 3: pdflatex ===' &&
            pdflatex -interaction=nonstopmode thesis.tex &&
            echo '=== Pass 4: pdflatex ===' &&
            pdflatex -interaction=nonstopmode thesis.tex
        "
    print_success "Full compilation complete!"
}

clean_files() {
    check_image_exists
    print_info "Cleaning auxiliary files..."
    docker run --rm -v "$(pwd)":/thesis "$IMAGE_NAME" \
        latexmk -C
    # Also clean additional files that latexmk might miss
    rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot *.lol *.fls *.fdb_latexmk
    rm -f *.run.xml *-blx.bib *.bcf *.synctex.gz
    print_success "Cleaned auxiliary files!"
}

open_shell() {
    check_image_exists
    print_info "Opening interactive shell..."
    docker run --rm -it -v "$(pwd)":/thesis "$IMAGE_NAME" /bin/bash
}

# Main logic
case "${1:-}" in
    build)
        build_image
        ;;
    compile)
        compile_thesis
        ;;
    quick)
        quick_compile
        ;;
    full)
        full_compile
        ;;
    clean)
        clean_files
        ;;
    shell)
        open_shell
        ;;
    help|--help|-h)
        show_help
        ;;
    "")
        # Default: build + compile
        print_info "Running build + compile (default)..."
        build_image
        compile_thesis
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
