#!/usr/bin/env python3
"""Demonstrate the fina-pricer FCN pipeline with deterministic sample permutations."""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


def project_root() -> Path:
    repo_root = Path(__file__).parents[3]
    if (repo_root / "src" / "riskcube_mcp").exists():
        return repo_root
    return Path(__file__).parents[4] / "riskcube-mcp"


def load_engine(root: Path):
    sys.path.insert(0, str(root / "src"))
    from riskcube_mcp import PricingRequest, sensitivity

    return PricingRequest, sensitivity


def make_variant(base: dict, moneyness: str, tenor: str, barrier: str, memory: bool) -> dict:
    request = copy.deepcopy(base)
    ratio = {"OTM": 0.90, "ATM": 1.00, "ITM": 1.15}[moneyness]
    expiry = "2027-12-03" if tenor == "near" else "2028-09-03"
    for underlying, spot_point in zip(request["UnwindMapRaw"]["underlyings"], request["MarketDataSnapshot"]["spot_data"]):
        underlying["spot"] = underlying["strikePrice"] * ratio
        spot_point["value"] = underlying["spot"]
    request["parameters"]["expiry"] = expiry
    request["parameters"]["accrual"]["memory"] = memory
    smooth_component = barrier == "none" and not memory
    request["parameters"]["payoff_type"] = "vanilla" if smooth_component else "fcn"
    if smooth_component:
        request["parameters"]["accrual"] = None
    if barrier == "none":
        request["parameters"]["barriers"] = []
        for underlying in request["UnwindMapRaw"]["underlyings"]:
            underlying["barriers"] = []
    else:
        direction, event, monitoring = barrier.split("_")
        request["parameters"]["barriers"] = []
        for index, underlying in enumerate(request["UnwindMapRaw"]["underlyings"]):
            level = (1.20 if direction == "up" else 0.80) - (0.05 * index if direction == "down" else 0.0)
            underlying["barriers"] = [{"direction": direction, "event": event, "level": level, "level_type": "relative_initial", "monitoring": monitoring, "observation_dates": ["2027-09-03"], "rebate": 0.01}]
    return request


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, help="FCN JSON payload; defaults to the repository attachment fixture")
    parser.add_argument("--output", type=Path, default=Path("fina_pricer_demo_report.json"))
    parser.add_argument("--project", type=Path, default=project_root())
    parser.add_argument("--aad-only", action="store_true", help="Fail if any result uses the discontinuous-payoff fallback")
    args = parser.parse_args()
    PricingRequest, sensitivity = load_engine(args.project)
    fixture = args.input or args.project / "data" / "basket_aapl_tsla.json"
    base = json.loads(fixture.read_text())
    variants = []
    for moneyness in ("OTM", "ATM", "ITM"):
        for tenor in ("near", "far"):
            for barrier in ("none", "up_KO_global", "down_KI_local"):
                for memory in (False, True):
                    variants.append((moneyness, tenor, barrier, memory))
    report = {"backend_policy": "AAD-first", "fixture": str(fixture), "results": []}
    for index, (moneyness, tenor, barrier, memory) in enumerate(variants, start=1):
        payload = make_variant(base, moneyness, tenor, barrier, memory)
        result = sensitivity(PricingRequest.model_validate(payload))
        methods = sorted({cell["method"] for cell in result["RiskCube"]["cells"]})
        if args.aad_only and any("finite-difference" in method for method in methods):
            raise SystemExit(f"AAD-only rejected case {index}: discontinuous event-state sensitivity requires a native XAD/QuantLib bridge")
        report["results"].append({"case_id": f"{moneyness}_{tenor}_{barrier}_{memory}", "pv": result["PV"], "pv_stderr": result["PV_stderr"], "methods": methods, "explainability": result["explainability"], "risk_cube": result["RiskCube"]})
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    aad_cases = sum(any("AAD" in method for method in row["methods"]) for row in report["results"])
    fd_cases = len(report["results"]) - aad_cases
    print(json.dumps({"cases": len(report["results"]), "aad_cases": aad_cases, "finite_difference_cases": fd_cases, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
