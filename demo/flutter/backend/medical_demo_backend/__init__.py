"""Medical demo backend package."""

from .api import create_app
from .service import build_default_service

__all__ = ["build_default_service", "create_app"]
