"""Vercel Python runtime entry point for the Streamable HTTP MCP server."""

import sys
from pathlib import Path

# Vercel imports api/index.py directly from /var/task. The project uses a src layout,
# so make /var/task/src importable before loading the ASGI application.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from riskcube_mcp.server import app

__all__ = ["app"]
