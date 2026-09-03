from __future__ import annotations

import itertools
import json
from datetime import date
from pathlib import Path

OUT = Path(__file__).with_name("scenarios.jsonl")


def make_case(
    moneyness: str,
    tenor: str,
    barrier_mode: str,
    memory: bool,
    accrual: bool,
    index: int,
) -> dict:
    spot = {"OTM": 90.0, "ATM": 100.0, "ITM": 115.0}[moneyness]
    expiry = date(2026, 10, 3) if tenor == "near" else date(2028, 9, 3)
    barrier = None
    barriers: list[dict] = []
    if barrier_mode != "none":
        direction = "up" if barrier_mode.startswith("up") else "down"
        event = "KI" if barrier_mode.endswith("ki") else "KO"
        barrier = 120.0 if direction == "up" else 80.0
        barriers = [
            {
                "direction": direction,
                "event": event,
                "level": barrier,
                "monitoring": "global" if "global" in barrier_mode else "local",
                "observation_dates": ["2027-09-03"],
                "rebate": 0.01,
            }
        ]
    payoff_type = "fcn" if accrual else "barrier" if barriers else "vanilla"
    req = {
        "InstrumentKey": {
            "isin": f"XS{index:010d}",
            "name": f"RISKCASE_{index:03d}",
            "strategy_id": "STRAT_FCN",
            "product_type": "FCN" if accrual else "OPTION",
            "family": "EQD",
            "group": "OPT",
            "leg_id": 1,
            "leg_name": "PUT",
            "notional": 1_000_000.0,
            "payment_currency": "USD",
            "status": "LIVE",
        },
        "UnwindMapRaw": {
            "underlyings": [
                {
                    "name": "UND_A",
                    "currency": "USD",
                    "spot": spot,
                    "strikePrice": 100.0,
                    "barrierPrice": barrier,
                    "fx_pair": "EURUSD",
                    "calendar": "NYSE",
                    "time": "1600",
                    "time_zone": "America/New_York",
                }
            ]
        },
        "RiskFactorKeys": [
            {
                "type": "Spot",
                "underlying": "UND_A",
                "temporal_role": "ValuationDate",
                "date": "2026-09-03",
            },
            {
                "type": "Volatility",
                "underlying": "UND_A",
                "expiry": expiry.isoformat(),
                "strike": 100.0,
            },
            {"type": "InterestRate", "tenor": "5Y", "date": "2026-09-03"},
            {"type": "FXSpot", "currency_pair": "EURUSD", "date": "2026-09-03"},
        ],
        "MarketDataSnapshot": {
            "spot_data": [{"rfk": {"underlying": "UND_A"}, "value": spot}],
            "vol_data": [
                {
                    "rfk": {
                        "underlying": "UND_A",
                        "expiry": expiry.isoformat(),
                        "strike": 100.0,
                    },
                    "value": 0.25,
                }
            ],
            "ir_data": [{"rfk": {"currency": "USD", "tenor": "5Y"}, "value": 0.03}],
            "fx_data": [{"rfk": {"currency_pair": "EURUSD"}, "value": 1.08}],
        },
        "UpdatedLifecycle": {
            "instrument_state": "LIVE",
            "applied_fixings": ["2026-09-03"],
            "adjusted_underlyings": [{"name": "UND_A", "adjustment_factor": 1.0}],
        },
        "parameters": {
            "eval_datetime": "2026-09-03",
            "expiry": expiry.isoformat(),
            "option_type": "put",
            "payoff_type": payoff_type,
            "barriers": barriers,
            "accrual": {
                "coupon_rate": 0.12,
                "memory": memory,
                "observation_frequency": "monthly",
                "observations": 12,
                "pay_if_ki": True,
            }
            if accrual
            else None,
            "risk_free_rate": 0.03,
            "dividend_yield": 0.01,
            "volatility": 0.25,
            "paths": 2500,
            "steps": 32,
            "seed": 100 + index,
            "bump_size": 0.0001,
            "bump_mode": "relative",
            "currency_conversion": 1.0,
        },
    }
    return {
        "case_id": f"{moneyness}_{tenor}_{barrier_mode}_{'memory' if memory else 'nomemory'}_{'accrual' if accrual else 'noaccrual'}",
        "request": req,
    }


cases = []
index = 1
for combo in itertools.product(
    ["OTM", "ATM", "ITM"],
    ["near", "far"],
    ["none", "up_global_ko", "down_local_ki", "up_local_ko"],
    [False, True],
    [False, True],
):
    cases.append(make_case(*combo, index))
    index += 1
OUT.write_text("\n".join(json.dumps(case) for case in cases) + "\n")
print(f"wrote {len(cases)} cases to {OUT}")
