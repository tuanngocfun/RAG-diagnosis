#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FastAPI server for Medical RAG Chatbot
Serves the Leishmania multimodal RAG pipeline with ColQwen2 + MedGemma-4B-IT
"""

import os
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any
import uvicorn
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from contextlib import asynccontextmanager
import logging
from dotenv import load_dotenv

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

# Add the project root to Python path to import the RAG module
sys.path.insert(0, str(project_root))

from app.server.rag_api import RAGService, AskRequest, AskResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global RAG service instance
rag_service: Optional[RAGService] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize and cleanup the RAG service"""
    global rag_service
    
    logger.info("🚀 Starting RAG service initialization...")
    try:
        rag_service = RAGService()
        logger.info("✅ RAG service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Failed to initialize RAG service: {e}")
        raise
    
    yield
    
    logger.info("🔄 Shutting down RAG service...")
    rag_service = None

# Create FastAPI app
app = FastAPI(
    title="Medical RAG Chatbot API",
    description="Leishmania multimodal RAG system with ColQwen2 retriever and MedGemma-4B generator",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware - allow frontend connections
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/healthz")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "medical-rag-chatbot",
        "rag_initialized": rag_service is not None
    }

# Main RAG endpoint (JSON-only, for backward compatibility)
@app.post("/api/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """
    Main RAG endpoint that processes questions and returns answers with evidence
    (JSON-only version for backward compatibility)
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        logger.info(f"🔍 Processing question: {request.question[:100]}...")
        response = await rag_service.ask(request)
        logger.info(f"✅ Generated response with {len(response.hits)} hits")
        return response
    
    except Exception as e:
        logger.error(f"❌ Error processing question: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question: {str(e)}"
        )

# Enhanced RAG endpoint with file upload support
@app.post("/api/ask-with-files", response_model=AskResponse)
async def ask_question_with_files(
    question: str = Form(..., description="The medical question to ask"),
    top_k: int = Form(8, description="Number of top results to retrieve"),
    case_type: Optional[str] = Form(None, description="Case type filter"),
    keyword: Optional[str] = Form(None, description="Keyword filter"),
    any_keywords: Optional[str] = Form(None, description="Comma-separated OR keywords"),
    micrograph_only: bool = Form(False, description="Prefer micrograph-like pages"),
    micrograph_strict: bool = Form(False, description="Hard filter to micrograph-like pages only"),
    images_per_answer: int = Form(2, description="Number of images to include in answer"),
    files: List[UploadFile] = File(default=[], description="Uploaded medical files (PDFs or images)")
):
    """
    Enhanced RAG endpoint that supports file uploads (PDFs and images)
    Processes user-uploaded medical content alongside database retrieval
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        # Validate uploaded files
        MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB per file
        ALLOWED_EXTENSIONS = {'.pdf', '.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'}
        
        for file in files:
            if not file.filename:
                continue
                
            # Check file size
            file_content = await file.read()
            if len(file_content) > MAX_FILE_SIZE:
                raise HTTPException(
                    status_code=413, 
                    detail=f"File {file.filename} exceeds maximum size of 50MB"
                )
            
            # Check file extension
            file_ext = Path(file.filename).suffix.lower()
            if file_ext not in ALLOWED_EXTENSIONS:
                raise HTTPException(
                    status_code=400,
                    detail=f"File {file.filename} has unsupported extension. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
                )
            
            # Reset file position for processing
            await file.seek(0)
        
        # Create request object
        request = AskRequest(
            question=question,
            top_k=top_k,
            case_type=case_type,
            keyword=keyword,
            any_keywords=any_keywords,
            micrograph_only=micrograph_only,
            micrograph_strict=micrograph_strict,
            images_per_answer=images_per_answer
        )
        
        logger.info(f"🔍 Processing question with {len(files)} uploaded files: {question[:100]}...")
        response = await rag_service.ask(request, uploaded_files=files if files else None)
        logger.info(f"✅ Generated response with {len(response.hits)} hits and {len(response.uploaded_file_info or [])} processed files")
        return response
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error processing question with files: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing question with files: {str(e)}"
        )

# Static file serving for images (secure path-based serving)
@app.get("/files/{file_path:path}")
async def serve_file(file_path: str, request: Request):
    """
    Serve image files securely from extract root or upload directory
    Supports both database images and user-uploaded files
    """
    if not rag_service:
        raise HTTPException(status_code=503, detail="RAG service not available")
    
    try:
        full_path = None
        
        # Handle uploaded files (prefixed with "uploads/")
        if file_path.startswith("uploads/"):
            upload_path = file_path[8:]  # Remove "uploads/" prefix
            upload_root = rag_service.file_processor.upload_dir.resolve()
            full_path = upload_root / upload_path
            
            # Security check: ensure path is within upload root
            if not str(full_path.resolve()).startswith(str(upload_root)):
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Handle database files (extract root)
        else:
            extract_root = rag_service.get_extract_root()
            if not extract_root:
                raise HTTPException(status_code=500, detail="Extract root not configured")
            
            full_path = Path(extract_root) / file_path
            full_path = full_path.resolve()
            
            # Security check: ensure path is within extract root
            if not str(full_path).startswith(str(Path(extract_root).resolve())):
                raise HTTPException(status_code=403, detail="Access denied")
        
        # Check if file exists and is a file
        if not full_path.exists() or not full_path.is_file():
            raise HTTPException(status_code=404, detail="File not found")
        
        # Check if it's an image file
        if full_path.suffix.lower() not in ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']:
            raise HTTPException(status_code=400, detail="Not an image file")
        
        return FileResponse(full_path)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Error serving file {file_path}: {e}")
        raise HTTPException(status_code=500, detail="Error serving file")

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Medical RAG Chatbot API",
        "docs": "/docs",
        "health": "/healthz",
        "api": "/api/ask"
    }

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    try:
        from uvicorn_config import get_config
        config = get_config()
        uvicorn.run(**config)
    except ImportError:
        # Fallback to direct configuration
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
            timeout_keep_alive=120,
            reload_excludes=["*.log", "*.tmp", "__pycache__", "*.pyc"],
            reload_includes=["*.py"],
            access_log=False
        )