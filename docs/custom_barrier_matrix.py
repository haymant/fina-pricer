import copy
import json
from pathlib import Path

from riskcube_mcp import PricingRequest, sensitivity

root = Path(__file__).parents[1]
base = json.loads((root / "data" / "attachment_sample.json").read_text())
rows = []
for moneyness, multiple in (("OTM", 0.90), ("ATM", 1.00), ("ITM", 1.15)):
    for memory in (False, True):
        for label, barrier in (
            ("none", []),
            (
                "up_KO_global",
                [
                    {
                        "direction": "up",
                        "event": "KO",
                        "level": 1.20
                        * base["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
                        "monitoring": "global",
                        "observation_dates": [],
                        "rebate": 0.01,
                    }
                ],
            ),
            (
                "up_KO_local_mid",
                [
                    {
                        "direction": "up",
                        "event": "KO",
                        "level": 1.20
                        * base["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
                        "monitoring": "local",
                        "observation_dates": ["2027-09-03"],
                        "rebate": 0.01,
                    }
                ],
            ),
            (
                "up_KO_local_expiry",
                [
                    {
                        "direction": "up",
                        "event": "KO",
                        "level": 1.20
                        * base["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
                        "monitoring": "local",
                        "observation_dates": ["2028-05-25"],
                        "rebate": 0.01,
                    }
                ],
            ),
            (
                "down_KI_global",
                [
                    {
                        "direction": "down",
                        "event": "KI",
                        "level": 0.80
                        * base["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
                        "monitoring": "global",
                        "observation_dates": [],
                        "rebate": 0.0,
                    }
                ],
            ),
            (
                "down_KI_local_mid",
                [
                    {
                        "direction": "down",
                        "event": "KI",
                        "level": 0.80
                        * base["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
                        "monitoring": "local",
                        "observation_dates": ["2027-09-03"],
                        "rebate": 0.0,
                    }
                ],
            ),
        ):
            payload = copy.deepcopy(base)
            u = payload["UnwindMapRaw"]["underlyings"][0]
            u["spot"] = u["strikePrice"] * multiple
            payload["MarketDataSnapshot"]["spot_data"][0]["value"] = u["spot"]
            payload["parameters"]["barriers"] = barrier
            payload["parameters"]["payoff_type"] = "fcn"
            payload["parameters"]["accrual"]["memory"] = memory
            result = sensitivity(PricingRequest.model_validate(payload))
            spot = next(
                c for c in result["RiskCube"]["cells"] if c["rfk"]["type"] == "Spot"
            )
            rows.append(
                {
                    "case_id": f"{moneyness}_{label}_memory_{memory}",
                    "moneyness": moneyness,
                    "barrier": label,
                    "memory": memory,
                    "pv": result["PV"],
                    "spot_greeks": spot["sensitivities"],
                    "methods": [c["method"] for c in result["RiskCube"]["cells"]],
                    "explainability": result["explainability"],
                }
            )
(root / "data" / "custom_barrier_matrix_report.json").write_text(
    json.dumps({"cases": rows}, indent=2) + "\n"
)
print(
    json.dumps(
        {
            "cases": len(rows),
            "output": str(root / "data" / "custom_barrier_matrix_report.json"),
        },
        indent=2,
    )
)
