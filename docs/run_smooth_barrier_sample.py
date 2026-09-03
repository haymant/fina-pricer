import copy
import json
from pathlib import Path

from riskcube_mcp import PricingRequest, sensitivity

root = Path(__file__).parents[1]
payload = json.loads((root / "data" / "attachment_sample.json").read_text())
payload["parameters"]["payoff_type"] = "barrier"
payload["parameters"]["accrual"] = None
payload["parameters"]["smooth_barrier"] = True
payload["parameters"]["barrier_smoothing_width"] = 0.01
payload["parameters"]["barriers"] = [{
    "direction": "down",
    "event": "KI",
    "level": 0.8 * payload["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
    "monitoring": "global",
    "observation_dates": [],
    "rebate": 0.0,
}]
result = sensitivity(PricingRequest.model_validate(copy.deepcopy(payload)))
spot_cell = next(cell for cell in result["RiskCube"]["cells"] if cell["rfk"]["type"] == "Spot")
print(json.dumps({
    "PV": result["PV"],
    "model": result["explainability"]["model"],
    "backend": result["explainability"]["aad_backend"],
    "delta": spot_cell["sensitivities"]["delta"],
    "gamma": spot_cell["sensitivities"]["gamma"],
    "smoothing_width": result["explainability"]["barrier_smoothing_width"],
}, indent=2))
