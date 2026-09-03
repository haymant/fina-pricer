from __future__ import annotations

import json
from pathlib import Path

import requests

ROOT = Path(__file__).parents[1]
payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
payload["RiskFactorKeys"] = [
    {"type": "Spot", "underlying": "UND_A HK", "temporal_role": "ValuationDate", "date": "2027-05-18"},
    {"type": "Volatility", "underlying": "UND_A HK", "expiry": "2028-05-25", "strike": 5.5858},
    {"type": "InterestRate", "underlying": "USD", "tenor": "5Y", "date": "2027-05-18"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "rho"},
]
payload["parameters"] = {**payload["parameters"], "payoff_type": "vanilla", "accrual": None, "svi": {"a": 0.04, "b": 0.12, "rho": -0.35, "m": 0.0, "sigma": 0.20}}
headers = {"host": "fina-pricer.vercel.app", "content-type": "application/json", "accept": "application/json, text/event-stream"}
base = "https://fina-pricer.vercel.app"
health = requests.get(base + "/healthz", timeout=30)
health.raise_for_status()
init = requests.post(base + "/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {"protocolVersion": "2025-03-26", "capabilities": {}, "clientInfo": {"name": "svi-live-verify", "version": "1"}}}, timeout=60)
init.raise_for_status()
call = requests.post(base + "/mcp", headers=headers, json={"jsonrpc": "2.0", "id": 2, "method": "tools/call", "params": {"name": "pricing_and_sensitivity", "arguments": {"request": payload}}}, timeout=180)
call.raise_for_status()
result = next(json.loads(line[6:]) for line in call.text.splitlines() if line.startswith("data: "))
if "error" in result:
    raise RuntimeError(result["error"])
report = json.loads(result["result"]["content"][0]["text"])
assert report["explainability"]["model"].startswith("SVI")
assert len(report["RiskCube"]["cells"]) == 4
print(json.dumps({"health": health.json(), "initialize_status": init.status_code, "pricing_status": call.status_code, "PV": report["PV"], "model": report["explainability"]["model"], "aad_backend": report["explainability"]["aad_backend"], "cells": len(report["RiskCube"]["cells"])}, indent=2))
