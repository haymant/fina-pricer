# AAD integration findings

As of 2026-09-03, the official QuantLib extensions page states that AAD can be enabled with the open-source XAD tool and an XAD/QuantLib integration module. It describes tape-based AAD and optional JIT execution, including record-once/replay-many workflows for repetitive Monte Carlo inner loops. It lists this as an external project rather than part of the standard SWIG Python QuantLib wrapper.

The QuantLibAAD repository describes itself as “QuantLib with XAD Automatic Differentiation in C++”. Its repository is C++/CMake-oriented and contains `ql`, `Examples`, `cmake`, and `test-suite`; it does not expose a ready-made Python wheel or direct Python bindings in the repository root. Therefore, a Python MCP service cannot honestly claim to use XAD/QuantLibAAD merely by installing the ordinary `QuantLib` PyPI wheel. A true XAD path requires building the C++ QuantLibAAD/XAD stack and exposing a narrow native bridge (for example, pybind11 or a subprocess RPC boundary).

The MCP Python SDK supports stdio, Streamable HTTP, and SSE. FastMCP exposes `streamable_http_app()` as a Starlette ASGI application, and the app requires lifespan startup. The Vercel Python runtime supports ASGI applications, detects Python entry points, and supports streaming responses.

Sources:
- https://www.quantlib.org/extensions.shtml
- https://github.com/auto-differentiation/QuantLibAAD
- https://github.com/modelcontextprotocol/python-sdk
- https://vercel.com/docs/functions/runtimes/python
