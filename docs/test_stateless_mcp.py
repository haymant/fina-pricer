import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1]))

import httpx
from asgi_lifespan import LifespanManager

from api.index import app


def event_json(response: httpx.Response) -> dict:
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise RuntimeError(response.text[:500])


async def main() -> None:
    payload = json.loads((Path(__file__).parents[1] / "data" / "attachment_sample.json").read_text())
    headers = {"host": "localhost", "content-type": "application/json", "accept": "application/json, text/event-stream"}
    async with LifespanManager(app), httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://localhost") as client:
        initialize = await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "stateless-smoke", "version": "1"}}})
        tool_call = await client.post("/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "pricing_and_sensitivity", "arguments": {"request": payload}}})
        assert initialize.status_code == 200, initialize.text
        assert tool_call.status_code == 200, tool_call.text
        result = event_json(tool_call)
        assert result["result"]["content"][0]["type"] == "text"
        print("initialize", initialize.status_code, initialize.headers.get("content-type"))
        print("tools_call", tool_call.status_code, tool_call.headers.get("content-type"))
        print("result_prefix", result["result"]["content"][0]["text"][:160])


if __name__ == "__main__":
    asyncio.run(main())
