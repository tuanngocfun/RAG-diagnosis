#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RAG API Service Layer
Integrates with the existing medgemma4b_qdrant_crossencoder_medcpt module
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from pydantic import BaseModel, Field, validator
import logging
from dotenv import load_dotenv
import tempfile
import shutil
from fastapi import UploadFile
import fitz  # PyMuPDF for PDF processing
from PIL import Image
import uuid

# Set up environment variables for the RAG system
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("RAG_ROOT", "/app")
os.environ.setdefault("RAG_EXTRACT_ROOT", "/app/kaggle/working2/extract")
os.environ.setdefault("QDRANT_COLLECTION", "leish_cases_pages")

# Load environment variables from multiple locations
project_root = Path(__file__).parent.parent.parent
app_env = Path(__file__).parent.parent / '.env'
project_env = project_root / '.env'

# Load in priority order
if project_env.exists():
    load_dotenv(project_env)
if app_env.exists():
    load_dotenv(app_env)
load_dotenv()  # Load from current directory/environment

# Add the project root to Python path for RAG module imports
sys.path.insert(0, str(project_root))

# Import from the existing RAG module - use BatchProcessor for enhanced functionality
try:
    from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import (
        CFG, find_case_dir
    )
    from rag.retriever.run_batch_answers_medgemma4b_test_medcpt import (
        BatchProcessor, select_images, derive_doc_id_from_row
    )
except ImportError as e:
    logging.error(f"Failed to import RAG module: {e}")
    logging.error(f"Python path: {sys.path}")
    logging.error(f"Project root: {project_root}")
    raise

logger = logging.getLogger(__name__)

# Pydantic models for API request/response
class AskRequest(BaseModel):
    """Request model for /api/ask endpoint"""
    question: str = Field(..., min_length=1, max_length=1000, description="The medical question to ask")
    top_k: int = Field(default=8, ge=1, le=15, description="Number of top results to retrieve")
    case_type: Optional[str] = Field(default=None, description="Case type filter")
    keyword: Optional[str] = Field(default=None, max_length=100, description="Keyword filter")
    any_keywords: Optional[str] = Field(default=None, max_length=200, description="Comma-separated OR keywords")
    micrograph_only: bool = Field(default=False, description="Prefer micrograph-like pages")
    micrograph_strict: bool = Field(default=False, description="Hard filter to micrograph-like pages only")
    images_per_answer: int = Field(default=2, ge=1, le=5, description="Number of images to include in answer")
    
    @validator('case_type')
    def validate_case_type(cls, v):
        if v is not None and v not in ["cutaneous", "mucocutaneous", "visceral", "unknown"]:
            raise ValueError("case_type must be one of: cutaneous, mucocutaneous, visceral, unknown")
        return v
    
    @validator('question')
    def validate_question(cls, v):
        if not v or not v.strip():
            raise ValueError("question cannot be empty")
        return v.strip()

class UploadedFileInfo(BaseModel):
    """Information about an uploaded file"""
    filename: str
    file_type: str  # 'pdf' or 'image'
    pages_extracted: Optional[int] = None
    processed_images: List[str] = []
    error: Optional[str] = None

class HitInfo(BaseModel):
    """Information about a retrieved hit/document"""
    rank: int
    score: float
    doc_id: str
    page_index: int
    image_path: Optional[str] = None
    page_kind: Optional[str] = None
    micrograph_like: bool = False
    keywords: List[str] = []
    text_excerpt: Optional[str] = None

class AskResponse(BaseModel):
    """Response model for /api/ask endpoint"""
    answer: str
    hits: List[HitInfo]
    evidence: List[Tuple[str, str]]  # [(span_text, citation), ...]
    used_images: List[str]
    note: Optional[str] = None
    # File upload related responses
    uploaded_file_info: Optional[List[UploadedFileInfo]] = None
    processing_status: Optional[str] = None

class FileProcessor:
    """Service for processing uploaded files (PDFs and images)"""
    
    def __init__(self, upload_dir: Path):
        self.upload_dir = upload_dir
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 File processor initialized with upload dir: {upload_dir}")
    
    async def process_uploaded_files(self, files: List[UploadFile]) -> List[UploadedFileInfo]:
        """Process uploaded files and return information about processed files"""
        processed_files = []
        
        for file in files:
            try:
                file_info = await self._process_single_file(file)
                processed_files.append(file_info)
            except Exception as e:
                logger.error(f"❌ Error processing file {file.filename}: {e}")
                processed_files.append(UploadedFileInfo(
                    filename=file.filename or "unknown",
                    file_type="unknown",
                    error=str(e)
                ))
        
        return processed_files
    
    async def _process_single_file(self, file: UploadFile) -> UploadedFileInfo:
        """Process a single uploaded file"""
        filename = file.filename or f"upload_{uuid.uuid4().hex}"
        file_extension = Path(filename).suffix.lower()
        
        # Create unique directory for this upload session
        session_id = uuid.uuid4().hex[:8]
        session_dir = self.upload_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        
        # Save uploaded file
        temp_file_path = session_dir / filename
        
        try:
            # Read and save file content
            content = await file.read()
            with open(temp_file_path, 'wb') as f:
                f.write(content)
            
            if file_extension == '.pdf':
                return await self._process_pdf(temp_file_path, session_dir)
            elif file_extension in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
                return await self._process_image(temp_file_path, session_dir)
            else:
                raise ValueError(f"Unsupported file type: {file_extension}")
                
        except Exception as e:
            # Clean up on error
            if temp_file_path.exists():
                temp_file_path.unlink()
            raise e
    
    async def _process_pdf(self, pdf_path: Path, session_dir: Path) -> UploadedFileInfo:
        """Extract pages from PDF as images"""
        try:
            doc = fitz.open(pdf_path)
            processed_images = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Render page as image (300 DPI for medical documents)
                mat = fitz.Matrix(300/72, 300/72)  # 300 DPI scaling
                pix = page.get_pixmap(matrix=mat)
                
                # Save as PNG
                image_filename = f"page_{page_num + 1:04d}.png"
                image_path = session_dir / image_filename
                pix.save(image_path)
                
                # Store relative path from session directory
                processed_images.append(str(image_path))
                pix = None  # Free memory
            
            doc.close()
            
            return UploadedFileInfo(
                filename=pdf_path.name,
                file_type='pdf',
                pages_extracted=len(processed_images),
                processed_images=processed_images
            )
            
        except Exception as e:
            raise Exception(f"Failed to process PDF: {e}")
    
    async def _process_image(self, image_path: Path, session_dir: Path) -> UploadedFileInfo:
        """Process uploaded image (validate and potentially convert)"""
        try:
            # Open and validate image
            with Image.open(image_path) as img:
                # Convert to PNG for consistency
                output_filename = f"{image_path.stem}.png"
                output_path = session_dir / output_filename
                
                # Convert to RGB if needed
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                
                # Save as PNG
                img.save(output_path, 'PNG', optimize=True)
                
                return UploadedFileInfo(
                    filename=image_path.name,
                    file_type='image',
                    pages_extracted=1,
                    processed_images=[str(output_path)]
                )
                
        except Exception as e:
            raise Exception(f"Failed to process image: {e}")

class RAGService:
    """
    Service class that wraps the RAG functionality
    Uses BatchProcessor for enhanced retrieval with reranking and multimodal support
    """
    
    def __init__(self):
        """Initialize the RAG service with BatchProcessor"""
        logger.info("🔄 Initializing RAG service with BatchProcessor...")
        
        # Load environment variables
        self._load_env_config()
        
        # Initialize the BatchProcessor (loads models once)
        try:
            self.batch_processor = BatchProcessor(use_reranker=True)
            logger.info("✅ BatchProcessor initialized with reranker support")
        except Exception as e:
            logger.error(f"❌ Failed to initialize BatchProcessor: {e}")
            raise
        
        # Initialize file processor
        upload_dir = Path("/tmp/rag_uploads")
        self.file_processor = FileProcessor(upload_dir)
        
        logger.info("✅ RAG service initialized")
    
    def _load_env_config(self):
        """Load and validate environment configuration"""
        # The RAG module reads from CFG and environment variables
        # Ensure critical env vars are set
        required_vars = ['QDRANT_URL', 'QDRANT_API_KEY']
        for var in required_vars:
            if not os.getenv(var):
                raise ValueError(f"Required environment variable {var} not set")
        
        logger.info(f"📍 Qdrant URL: {os.getenv('QDRANT_URL')}")
        logger.info(f"🗂️ Extract root: {getattr(CFG, 'EXTRACT_ROOT', 'Not configured')}")
    
    def get_extract_root(self) -> Optional[Path]:
        """Get the extraction root directory for file serving"""
        extract_root = getattr(CFG, 'EXTRACT_ROOT', None)
        if extract_root:
            return Path(extract_root)
        return None
    
    async def ask(self, request: AskRequest, uploaded_files: List[UploadFile] = None) -> AskResponse:
        """
        Process a question and return an answer with evidence
        Uses BatchProcessor for enhanced retrieval and generation
        Supports both database retrieval and user-uploaded files
        """
        try:
            # Step 0: Process uploaded files if any
            uploaded_file_info = []
            user_images = []
            
            if uploaded_files:
                logger.info(f"📤 Processing {len(uploaded_files)} uploaded files...")
                uploaded_file_info = await self.file_processor.process_uploaded_files(uploaded_files)
                
                # Collect all processed images from uploads
                for file_info in uploaded_file_info:
                    if not file_info.error:
                        user_images.extend(file_info.processed_images)
                
                logger.info(f"📷 Extracted {len(user_images)} images from uploads")
            
            # Step 1: Enhanced retrieval using BatchProcessor (database content)
            retrieval_mode = "multimodal" if user_images else "text-only"
            logger.info(f"🔍 Retrieving documents for: {request.question[:50]}... (mode: {retrieval_mode})")
            
            retrieval_result = self.batch_processor.enhanced_qdrant_ask_text(
                question=request.question,
                top_k=request.top_k,
                case_type=request.case_type,
                keyword=request.keyword,
                any_keywords=request.any_keywords,
                micrograph_only=request.micrograph_only,
                micrograph_strict=request.micrograph_strict,
                pool_multiplier=3,  # Use enhanced retrieval pool
                uploaded_images=user_images  # NEW: Pass uploaded images for multimodal retrieval
            )
            
            hits = retrieval_result.get("hits", [])
            logger.info(f"📄 Retrieved {len(hits)} documents from database")
            
            # Step 2: Combine database hits with user-uploaded content
            all_images = []
            
            # Add user-uploaded images first (higher priority)
            if user_images:
                all_images.extend(user_images[:request.images_per_answer])
                logger.info(f"📸 Using {len(all_images)} user-uploaded images")
            
            # Fill remaining slots with database images
            if len(all_images) < request.images_per_answer and hits:
                db_images = select_images(
                    seed=[],
                    hits=hits,
                    take=request.images_per_answer - len(all_images),
                    target_doc_id=None
                )
                all_images.extend(db_images)
                logger.info(f"📚 Added {len(db_images)} database images")
            
            # If no hits and no user images, return informative response
            if not hits and not user_images:
                logger.warning("⚠️ No relevant documents or uploaded files found")
                return AskResponse(
                    answer="I don't have access to relevant medical documents in my database to answer your question. Please upload some medical images or PDFs to help me provide a more informed response.",
                    hits=[],
                    evidence=[],
                    used_images=[],
                    uploaded_file_info=uploaded_file_info,
                    note="No relevant documents found. Consider uploading medical files for analysis."
                )
            
            # Step 3: Generate answer using hybrid approach
            logger.info("🤖 Generating answer with multimodal support...")
            
            # Create context from user uploads for enhanced answer generation
            upload_context = ""
            if uploaded_file_info:
                upload_context = f"\n\nUser has uploaded {len(uploaded_file_info)} files: "
                upload_context += ", ".join([
                    f"{info.filename} ({info.file_type}, {info.pages_extracted or 1} pages)"
                    for info in uploaded_file_info if not info.error
                ])
                upload_context += "\nPlease analyze both the uploaded content and database evidence to provide a comprehensive answer."
            
            # Enhanced question with upload context
            enhanced_question = request.question
            if upload_context:
                enhanced_question += upload_context
            
            # Generate answer using BatchProcessor
            answer_text = self.batch_processor.answer_with_images(
                question=enhanced_question,
                image_paths=all_images,
                hits=hits,
                images_per_answer=len(all_images)
            )
            
            # Step 3: Format response
            formatted_hits = []
            for rank, hit in enumerate(hits, 1):
                # Convert image path to relative path for secure serving
                image_path = hit.get("image_path")
                if image_path and self.get_extract_root():
                    try:
                        # Convert absolute path to relative path for API serving
                        abs_path = Path(image_path).resolve()
                        extract_root = Path(self.get_extract_root()).resolve()
                        if str(abs_path).startswith(str(extract_root)):
                            image_path = str(abs_path.relative_to(extract_root))
                        else:
                            # If path is not under extract root, keep original but log warning
                            logger.warning(f"⚠️ Image path outside extract root: {image_path}")
                    except Exception as e:
                        logger.warning(f"⚠️ Could not convert image path: {e}")
                        image_path = None
                
                # Truncate text excerpt
                text_excerpt = hit.get("text_excerpt", "")
                if text_excerpt and len(text_excerpt) > 300:
                    text_excerpt = text_excerpt[:297] + "..."
                
                formatted_hit = HitInfo(
                    rank=rank,
                    score=round(float(hit.get("score", 0.0)), 3),
                    doc_id=hit.get("doc_id", ""),
                    page_index=int(hit.get("page_index", 0)),
                    image_path=image_path,
                    page_kind=hit.get("page_kind", ""),
                    micrograph_like=bool(hit.get("micrograph_like", False)),
                    keywords=hit.get("keywords", []) if isinstance(hit.get("keywords"), list) else [],
                    text_excerpt=text_excerpt
                )
                formatted_hits.append(formatted_hit)
            
            # Convert image paths to relative paths for secure serving
            relative_all_images = []
            for img_path in all_images:
                try:
                    abs_path = Path(img_path).resolve()
                    
                    # Check if it's from extract root (database images)
                    if self.get_extract_root():
                        extract_root = Path(self.get_extract_root()).resolve()
                        if str(abs_path).startswith(str(extract_root)):
                            relative_all_images.append(str(abs_path.relative_to(extract_root)))
                            continue
                    
                    # Check if it's from upload directory (user images)
                    upload_root = self.file_processor.upload_dir.resolve()
                    if str(abs_path).startswith(str(upload_root)):
                        relative_all_images.append(f"uploads/{abs_path.relative_to(upload_root)}")
                        continue
                    
                    # Fallback: keep original path
                    relative_all_images.append(img_path)
                    
                except Exception as e:
                    logger.warning(f"Path conversion error for {img_path}: {e}")
                    relative_all_images.append(img_path)
            
            all_images = relative_all_images
            
            # Extract evidence spans for better citations
            from rag.retriever.medgemma4b_qdrant_crossencoder_medcpt import extractive_spans
            evidence_spans = extractive_spans(hits, per_doc=4, max_chars=400, question=request.question)
            
            response = AskResponse(
                answer=answer_text,
                hits=formatted_hits,
                evidence=evidence_spans,
                used_images=all_images,
                uploaded_file_info=uploaded_file_info,
                processing_status="completed" if uploaded_file_info else None
            )
            
            logger.info(f"✅ Generated answer: {len(response.answer)} chars, {len(response.hits)} hits")
            return response
            
        except Exception as e:
            error_msg = str(e)
            logger.error(f"❌ Error in RAG pipeline: {e}")
            
            # Provide specific error messages for common issues
            if "Name or service not known" in error_msg:
                error_note = "❌ Cannot connect to Qdrant database. This is likely a network/DNS issue. Please check:\n1. Internet connectivity\n2. Qdrant URL configuration\n3. Docker network settings\n\nTry running with host network mode: docker run --network host ..."
            elif "Connection refused" in error_msg:
                error_note = "❌ Qdrant database connection refused. Please verify the Qdrant service is running and accessible."
            elif "Authentication failed" in error_msg or "Unauthorized" in error_msg:
                error_note = "❌ Qdrant authentication failed. Please check your QDRANT_API_KEY configuration."
            else:
                error_note = f"❌ Error processing question: {error_msg}"
            
            # Return partial response with error note and uploaded file info if available
            return AskResponse(
                answer="I encountered an error while processing your question. Please check the system configuration and try again.",
                hits=[],
                evidence=[],
                used_images=[],
                uploaded_file_info=uploaded_file_info if 'uploaded_file_info' in locals() else None,
                note=error_note
            )