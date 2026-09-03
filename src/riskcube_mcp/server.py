from __future__ import annotations

import os
from typing import Any

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings
from starlette.responses import JSONResponse

from .core import PricingRequest, sensitivity

allowed_hosts = [
    host.strip()
    for host in os.getenv(
        "ALLOWED_HOSTS", "localhost,127.0.0.1,[::1],localhost:*,127.0.0.1:*,[::1]:*,fina-pricer.vercel.app,fina-pricer.vercel.app:*"
    ).split(",")
    if host.strip()
]
mcp = FastMCP(
    "riskcube-pricing",
    stateless_http=True,
    transport_security=TransportSecuritySettings(allowed_hosts=allowed_hosts),
)


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(_request: Any) -> JSONResponse:
    return JSONResponse({"status": "ok", "service": "riskcube-pricing"})


app = mcp.streamable_http_app()


@mcp.tool()
def pricing_and_sensitivity(request: dict[str, Any]) -> dict[str, Any]:
    """Price a structured product and return an explainable RiskCube.

    The response reports PV, Monte Carlo error, event-state diagnostics, and per-RFK
    AAD or finite-difference sensitivities. Barriers and memory coupons are handled as explicit
    path events, so each result can be traced back to model inputs and lifecycle state.
    """
    parsed = PricingRequest.model_validate(request)
    return sensitivity(parsed)


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
