"""
RAG CLI - Command Line Interface for RAG Pipeline

CRITICAL: This module imports from pipeline FIRST to ensure HF cache paths are set.
All CLI scripts should be run via: PYTHONPATH="$PROJECT_ROOT/codes" python -m cli.script_name
"""
# Import pipeline first to set HF paths before any model imports
from pipeline import log_env_summary

__all__ = ["log_env_summary"]
