#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Uvicorn configuration for Medical RAG Chatbot
Optimized for stability and handling long model loading times
"""

import os
import logging

# Determine if we're in development or production
IS_DEV = os.getenv("ENVIRONMENT", "development") == "development"
LOG_LEVEL = os.getenv("LOG_LEVEL", "info")

# Base configuration
UVICORN_CONFIG = {
    "app": "main:app",
    "host": "0.0.0.0",
    "port": 8000,
    "log_level": LOG_LEVEL,
    
    # Timeout settings for model loading
    "timeout_keep_alive": int(os.getenv("UVICORN_TIMEOUT_KEEP_ALIVE", "120")),
    
    # Performance settings
    "backlog": 2048,
    "limit_concurrency": 10,  # Limit concurrent requests
    "limit_max_requests": 1000,
    
    # Access log settings
    "access_log": IS_DEV,
}

# Development-specific settings
if IS_DEV:
    UVICORN_CONFIG.update({
        "reload": True,
        "reload_excludes": [
            "*.log", "*.tmp", "__pycache__", "*.pyc", "*.pyo",
            "*.orig", "*.swp", "*.swo", "*~", ".DS_Store",
            "*.git*", "node_modules", "*.env*"
        ],
        "reload_includes": ["*.py"],
        "reload_delay": 0.25,  # Debounce file changes
    })
else:
    # Production settings
    UVICORN_CONFIG.update({
        "reload": False,
    })

def get_config():
    """Get uvicorn configuration"""
    return UVICORN_CONFIG

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    logging.info(f"🚀 Starting uvicorn with config: {config}")
    uvicorn.run(**config)