import copy
import json
from math import isfinite
from pathlib import Path

from riskcube_mcp import PricingRequest, price_request, sensitivity

root = Path(__file__).parents[1]
case = json.loads((root / "data" / "attachment_sample.json").read_text())
case["parameters"] = {**case["parameters"], "payoff_type": "barrier", "accrual": None}
case["parameters"]["barriers"] = [{
    "direction": "down",
    "event": "KI",
    "level": 0.8 * case["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
    "monitoring": "global",
    "observation_dates": [],
    "rebate": 0.0,
}]
request = PricingRequest.model_validate(case)
exact = price_request(request)
print(json.dumps({"exact_pv": exact.pv, "exact_model": exact.diagnostics["model"]}, indent=2))
for width in (0.02, 0.01, 0.005):
    smooth_case = copy.deepcopy(case)
    smooth_case["parameters"]["smooth_barrier"] = True
    smooth_case["parameters"]["barrier_smoothing_width"] = width
    smooth_request = PricingRequest.model_validate(smooth_case)
    output = sensitivity(smooth_request)
    spot = next(c for c in output["RiskCube"]["cells"] if c["rfk"]["type"] == "Spot")
    values = spot["sensitivities"]
    assert all(isfinite(v) for v in values.values() if isinstance(v, (float, int)))
    print(json.dumps({"width": width, "pv": output["PV"], "pv_bias": output["PV"] - exact.pv, "delta": values["delta"], "gamma": values["gamma"], "method": spot["method"]}, indent=2))
