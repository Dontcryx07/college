"""Vercel ASGI entrypoint — auto-detected by the @vercel/python builder.

Vercel scans for ``asgi.py`` at the project root and looks for an ``app``
variable (the ASGI callable). This file re-exports the FastAPI application
from ``api/app.py`` so the entire pipeline remains importable.

Deploying via ``vercel --prod`` auto-detects this file with zero config.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the project root is on sys.path so pipeline modules resolve.
_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from api.app import app

__all__ = ["app"]
