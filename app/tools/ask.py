#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CLI tool for Medical RAG Chatbot
Provides command-line interface for asking questions to the RAG system
Supports both single questions and batch processing of JSONL files
"""

import argparse
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import logging

# Load environment variables from parent directory
try:
    from dotenv import load_dotenv
    parent_env = Path(__file__).parent.parent / '.env'
    if parent_env.exists():
        load_dotenv(parent_env)
except ImportError:
    pass  # Silent fallback

# Add the parent directory to Python path to import the RAG module
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import (
        qdrant_ask_text, answer_with_gemini
    )
except ImportError as e:
    logging.error(f"Failed to import RAG module: {e}")
    print("Error: Could not import the RAG module. Please ensure you're running from the correct directory.")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def process_question(
    question: str,
    top_k: int = 8,
    case_type: Optional[str] = None,
    keyword: Optional[str] = None,
    any_keywords: Optional[str] = None,
    micrograph_only: bool = False,
    micrograph_strict: bool = False,
    images_per_answer: int = 2
) -> Dict[str, Any]:
    """
    Process a single question through the RAG pipeline
    
    Args:
        question: The medical question to ask
        top_k: Number of top results to retrieve
        case_type: Case type filter
        keyword: Keyword filter
        any_keywords: Comma-separated OR keywords
        micrograph_only: Prefer micrograph-like pages
        micrograph_strict: Hard filter to micrograph-like pages only
        images_per_answer: Number of images to include in answer
    
    Returns:
        Dict containing the answer, hits, evidence, and used images
    """
    try:
        logger.info(f"🔍 Processing question: {question[:100]}...")
        
        # Step 1: Retrieve relevant documents
        retrieval_result = qdrant_ask_text(
            question=question,
            top_k=top_k,
            case_type=case_type,
            keyword=keyword,
            any_keywords=any_keywords,
            micrograph_only=micrograph_only,
            micrograph_strict=micrograph_strict,
            use_reranker=True
        )
        
        hits = retrieval_result.get("hits", [])
        
        if not hits:
            logger.warning("⚠️ No relevant documents found")
            return {
                "question": question,
                "answer": "",
                "hits": [],
                "evidence": [],
                "used_images": [],
                "note": "No relevant documents found for your question."
            }
        
        logger.info(f"📄 Retrieved {len(hits)} documents")
        
        # Step 2: Generate answer using retrieved context
        generation_result = answer_with_gemini(
            question=question,
            hits=hits,
            take=images_per_answer
        )
        
        # Format the response
        response = {
            "question": question,
            "answer": generation_result.get("answer", ""),
            "hits": hits,
            "evidence": generation_result.get("evidence", []),
            "used_images": generation_result.get("used_images", [])
        }
        
        logger.info(f"✅ Generated answer: {len(response['answer'])} chars")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        return {
            "question": question,
            "answer": "",
            "hits": [],
            "evidence": [],
            "used_images": [],
            "note": f"Error processing question: {str(e)}"
        }


def process_batch_file(
    input_file: Path,
    output_file: Optional[Path] = None
) -> None:
    """
    Process a batch file of questions in JSONL format
    
    Args:
        input_file: Path to input JSONL file with questions
        output_file: Path to output JSONL file (defaults to stdout)
    """
    if not input_file.exists():
        logger.error(f"❌ Input file not found: {input_file}")
        sys.exit(1)
    
    output_handle = open(output_file, 'w', encoding='utf-8') if output_file else sys.stdout
    
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                
                try:
                    # Parse the question data
                    data = json.loads(line)
                    question = data.get("question", "")
                    
                    if not question:
                        logger.warning(f"⚠️ Line {line_num}: No question found")
                        continue
                    
                    # Extract parameters (with defaults)
                    params = {
                        "top_k": data.get("top_k", 8),
                        "case_type": data.get("case_type"),
                        "keyword": data.get("keyword"),
                        "any_keywords": data.get("any_keywords"),
                        "micrograph_only": data.get("micrograph_only", False),
                        "micrograph_strict": data.get("micrograph_strict", False),
                        "images_per_answer": data.get("images_per_answer", 2)
                    }
                    
                    # Process the question
                    result = process_question(question, **params)
                    
                    # Write the result
                    output_handle.write(json.dumps(result, ensure_ascii=False) + '\n')
                    output_handle.flush()
                    
                except json.JSONDecodeError as e:
                    logger.error(f"❌ Line {line_num}: Invalid JSON - {e}")
                    continue
                except Exception as e:
                    logger.error(f"❌ Line {line_num}: Error processing - {e}")
                    continue
    
    finally:
        if output_file:
            output_handle.close()


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        description="CLI tool for Medical RAG Chatbot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Ask a single question
  python ask.py --question "What are the diagnostic features on biopsy?"
  
  # Ask with specific case type and settings
  python ask.py --question "Show me microscopy findings" --case-type cutaneous --micrograph-only
  
  # Process a batch file
  python ask.py --input questions.jsonl --output answers.jsonl
  
  # Process batch file to stdout
  python ask.py --input questions.jsonl
  
JSONL Input Format:
  {"question": "What are the symptoms?", "top_k": 10, "case_type": "cutaneous"}
  {"question": "Show me treatment options", "micrograph_only": true}
  
JSONL Output Format:
  {"question": "...", "answer": "...", "hits": [...], "evidence": [...], "used_images": [...]}
        """
    )
    
    # Input options
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--question", "-q",
        help="Single question to ask"
    )
    group.add_argument(
        "--input", "-i",
        type=Path,
        help="Input JSONL file with questions"
    )
    
    # Output options
    parser.add_argument(
        "--output", "-o",
        type=Path,
        help="Output JSONL file (default: stdout)"
    )
    
    # RAG parameters
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=8,
        help="Number of top results to retrieve (1-15, default: 8)"
    )
    parser.add_argument(
        "--case-type", "-t",
        choices=["cutaneous", "mucocutaneous", "visceral", "unknown"],
        help="Case type filter"
    )
    parser.add_argument(
        "--keyword",
        help="Keyword filter"
    )
    parser.add_argument(
        "--any-keywords",
        help="Comma-separated OR keywords"
    )
    parser.add_argument(
        "--micrograph-only",
        action="store_true",
        help="Prefer micrograph-like pages"
    )
    parser.add_argument(
        "--micrograph-strict",
        action="store_true",
        help="Hard filter to micrograph-like pages only"
    )
    parser.add_argument(
        "--images-per-answer",
        type=int,
        default=2,
        choices=range(1, 6),
        help="Number of images to include in answer (1-5, default: 2)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.top_k < 1 or args.top_k > 15:
        parser.error("--top-k must be between 1 and 15")
    
    try:
        if args.question:
            # Process single question
            result = process_question(
                question=args.question,
                top_k=args.top_k,
                case_type=args.case_type,
                keyword=args.keyword,
                any_keywords=args.any_keywords,
                micrograph_only=args.micrograph_only,
                micrograph_strict=args.micrograph_strict,
                images_per_answer=args.images_per_answer
            )
            
            # Output result
            if args.output:
                with open(args.output, 'w', encoding='utf-8') as f:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
                logger.info(f"✅ Result written to {args.output}")
            else:
                print(json.dumps(result, ensure_ascii=False, indent=2))
        
        elif args.input:
            # Process batch file
            logger.info(f"📂 Processing batch file: {args.input}")
            process_batch_file(args.input, args.output)
            if args.output:
                logger.info(f"✅ Results written to {args.output}")
            else:
                logger.info("✅ Results written to stdout")
    
    except KeyboardInterrupt:
        logger.info("🛑 Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()