import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import httpx
from asgi_lifespan import LifespanManager

from api.index import app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with LifespanManager(app), httpx.AsyncClient(transport=transport, base_url="http://localhost") as client:
        health = await client.get("/healthz", headers={"host": "localhost"})
        print("health", health.status_code, health.json())
        response = await client.post(
            "/mcp",
            headers={"host": "localhost", "content-type": "application/json", "accept": "application/json, text/event-stream"},
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "smoke", "version": "1.0"}}},
        )
        print("mcp_initialize_status", response.status_code)
        print("mcp_content_type", response.headers.get("content-type"))
        print("mcp_body_prefix", response.text[:300])
        assert health.status_code == 200
        assert response.status_code == 200
        assert "text/event-stream" in response.headers.get("content-type", "")
        assert "serverInfo" in response.text


if __name__ == "__main__":
    asyncio.run(main())
