from __future__ import annotations

import copy
import json
from pathlib import Path

from riskcube_mcp import PricingRequest, sensitivity

ROOT = Path(__file__).parents[1]
payload = json.loads((ROOT / "data" / "attachment_sample.json").read_text())
payload["RiskFactorKeys"] = [
    {"type": "Spot", "underlying": "UND_A HK", "temporal_role": "ValuationDate", "date": "2027-05-18"},
    {"type": "Volatility", "underlying": "UND_A HK", "expiry": "2028-05-25", "strike": 5.5858},
    {"type": "InterestRate", "underlying": "USD", "tenor": "5Y", "date": "2027-05-18"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "a"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "b"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "rho"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "m"},
    {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "sigma"},
]
payload["parameters"] = {
    **payload["parameters"],
    "payoff_type": "vanilla",
    "accrual": None,
    "svi": {"a": 0.04, "b": 0.12, "rho": -0.35, "m": 0.0, "sigma": 0.20},
}
result = sensitivity(PricingRequest.model_validate(copy.deepcopy(payload)))
(ROOT / "data" / "svi_aad_sample_result.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps({"PV": result["PV"], "model": result["explainability"]["model"], "backend": result["explainability"]["aad_backend"], "cells": len(result["RiskCube"]["cells"])}, indent=2))
for cell in result["RiskCube"]["cells"]:
    rfk = cell["rfk"]
    print(rfk.get("type"), rfk.get("surface_parameter", ""), cell["sensitivities"].get("skew_sensitivity"), cell["sensitivities"].get("delta_vega"), cell["sensitivities"].get("gamma_vega"))
