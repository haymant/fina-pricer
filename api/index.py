"""Vercel Python runtime entry point for the Streamable HTTP MCP server."""

import sys
from pathlib import Path
from typing import Any

# Vercel imports api/index.py directly from /var/task. The project uses a src layout,
# so make /var/task/src importable before loading the ASGI application.
_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from riskcube_mcp.server import app as _mcp_app


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    """Adapt Vercel's function path to FastMCP's `/mcp` route."""
    if scope.get("type") == "http":
        path = scope.get("path", "")
        if path == "/api" or path == "/api/":
            scope = {**scope, "path": "/mcp", "raw_path": b"/mcp"}
        elif path.startswith("/api/"):
            suffix = path.removeprefix("/api")
            scope = {**scope, "path": "/mcp" + suffix, "raw_path": ("/mcp" + suffix).encode()}
    await _mcp_app(scope, receive, send)


__all__ = ["app"]
