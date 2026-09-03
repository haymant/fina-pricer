import asyncio
import json
import sys
from pathlib import Path

import httpx
from asgi_lifespan import LifespanManager

sys.path.insert(0, str(Path(__file__).parents[1]))
from api.index import app


async def main() -> None:
    transport = httpx.ASGITransport(app=app)
    async with (
        LifespanManager(app),
        httpx.AsyncClient(transport=transport, base_url="http://localhost") as client,
    ):
        response = await client.post(
            "/mcp",
            headers={
                "content-type": "application/json",
                "accept": "application/json, text/event-stream",
            },
            content=json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {
                        "protocolVersion": "2025-03-26",
                        "capabilities": {},
                        "clientInfo": {"name": "smoke", "version": "1.0"},
                    },
                }
            ),
        )
        print(response.status_code)
        print(response.headers.get("content-type"))
        print(response.text[:500])
        response.raise_for_status()


asyncio.run(main())
