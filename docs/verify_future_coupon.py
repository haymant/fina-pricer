from __future__ import annotations

import json
from pathlib import Path

from riskcube_mcp.core import PricingRequest, price_request

root = Path(__file__).parents[1]
case = json.loads((root / "data" / "attachment_sample.json").read_text())
case["parameters"] = {
    **case["parameters"],
    "payoff_type": "fcn",
    "paths": 3000,
    "steps": 100,
    "accrual": {
        "coupon_rate": 0.086778,
        "memory": True,
        "observations": 10,
        "accruals": [0.0, 0.009642, 0.009642, 0.009642, 0.009642, 0.009642, 0.009642, 0.009642, 0.009642, 0.009642],
        "payment_dates": [
            "2027-05-22", "2027-05-23", "2027-06-21", "2027-07-24", "2027-08-22",
            "2027-09-21", "2027-10-23", "2027-11-21", "2027-12-25", "2028-01-21",
        ],
        "n1": [20, 1, 21, 22, 14, 0, 0, 0, 0, 0],
        "n2": [20, 1, 21, 22, 21, 21, 22, 20, 22, 19],
        "fixed_n1_periods": 5,
        "range_lower": 0.99,
        "range_upper": 1.01,
        "range_level_type": "relative_initial",
    },
}
request = PricingRequest.model_validate(case)
result = price_request(request)
schedule = result.diagnostics["coupon_state"]["coupon_schedule"]
periods = schedule["periods"]
future = periods[5:]
assert schedule["future_n1_stochastic"] is True
assert all(period["realized"] is False for period in future)
assert all(period["n2"] > 0 for period in future)
assert any(abs(period["n1_expected"] - period["n2"]) > 1e-9 for period in future), future
assert all(period["discounted_forward_total"] >= 0 for period in future)
print(json.dumps({
    "coupon_paid": result.diagnostics["coupon_state"]["coupon_paid"],
    "coupon_forward": result.diagnostics["coupon_state"]["coupon_forward"],
    "future_periods": future,
}, indent=2))
