import inspect

from mcp.server.fastmcp import FastMCP

print("FastMCP", inspect.signature(FastMCP))
print("streamable_http_app", inspect.signature(FastMCP.streamable_http_app))
