from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import requests

LIVE_URL = os.getenv("MCP_URL", "https://fina-pricer.vercel.app/api")
ROOT = Path(__file__).parents[1]
BASE = json.loads((ROOT / "data" / "attachment_sample.json").read_text())


def sse_json(response: requests.Response) -> dict:
    for line in response.text.splitlines():
        if line.startswith("data: "):
            return json.loads(line[6:])
    raise RuntimeError(f"No SSE data in response: {response.status_code} {response.text[:500]}")


def main() -> int:
    session = requests.Session()
    headers = {"content-type": "application/json", "accept": "application/json, text/event-stream"}
    init = session.post(LIVE_URL, headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "fina-pricer-live-matrix", "version": "1.0"}}}, timeout=90)
    init.raise_for_status()
    init_result = sse_json(init)
    if "error" in init_result:
        raise RuntimeError(f"MCP initialize failed: {init_result['error']}")
    rows = []
    strike = BASE["UnwindMapRaw"]["underlyings"][0]["strikePrice"]
    scenarios = [("none", [])]
    scenarios += [("up_KO_global", [{"direction": "up", "event": "KO", "level": strike * 1.20, "monitoring": "global", "observation_dates": [], "rebate": 0.01}])]
    scenarios += [("up_KO_local_mid", [{"direction": "up", "event": "KO", "level": strike * 1.20, "monitoring": "local", "observation_dates": ["2027-09-03"], "rebate": 0.01}])]
    scenarios += [("up_KO_local_expiry", [{"direction": "up", "event": "KO", "level": strike * 1.20, "monitoring": "local", "observation_dates": ["2028-05-25"], "rebate": 0.01}])]
    scenarios += [("down_KI_global", [{"direction": "down", "event": "KI", "level": strike * 0.80, "monitoring": "global", "observation_dates": [], "rebate": 0.0}])]
    scenarios += [("down_KI_local_mid", [{"direction": "down", "event": "KI", "level": strike * 0.80, "monitoring": "local", "observation_dates": ["2027-09-03"], "rebate": 0.0}])]
    for moneyness, multiple in (("OTM", 0.90), ("ATM", 1.00), ("ITM", 1.15)):
        for memory in (False, True):
            for label, barriers in scenarios:
                payload = copy.deepcopy(BASE)
                u = payload["UnwindMapRaw"]["underlyings"][0]
                u["spot"] = strike * multiple
                payload["MarketDataSnapshot"]["spot_data"][0]["value"] = u["spot"]
                payload["parameters"]["payoff_type"] = "fcn"
                payload["parameters"]["accrual"]["memory"] = memory
                payload["parameters"]["barriers"] = barriers
                case_id = f"{moneyness}_{label}_memory_{memory}"
                call = session.post(LIVE_URL, headers=headers, json={"jsonrpc": "2.0", "id": len(rows) + 2, "method": "tools/call", "params": {"name": "pricing_and_sensitivity", "arguments": {"request": payload}}}, timeout=180)
                call.raise_for_status()
                message = sse_json(call)
                if "error" in message:
                    raise RuntimeError(f"{case_id}: {message['error']}")
                result = json.loads(message["result"]["content"][0]["text"])
                spot = next(cell for cell in result["RiskCube"]["cells"] if cell["rfk"]["type"] == "Spot")
                rows.append({"case_id": case_id, "moneyness": moneyness, "barrier": label, "memory": memory, "PV": result["PV"], "PV_stderr": result["PV_stderr"], "spot_greeks": spot["sensitivities"], "method": spot["method"], "explainability": result["explainability"]})
                print(case_id, result["PV"], flush=True)
    output = ROOT / "data" / "live_permutation_report.json"
    output.write_text(json.dumps({"endpoint": LIVE_URL, "cases": rows}, indent=2) + "\n")
    print(json.dumps({"cases": len(rows), "output": str(output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
