from __future__ import annotations

import json
from pathlib import Path

import pytest

from riskcube_mcp.core import PricingRequest, price_request, sensitivity
from riskcube_mcp.server import app

ROOT = Path(__file__).parents[1]
DATA = ROOT / "data" / "scenarios.jsonl"
ATTACHMENT_SAMPLE = ROOT / "data" / "attachment_sample.json"


def load_cases() -> list[dict]:
    return [json.loads(line) for line in DATA.read_text().splitlines() if line.strip()]


@pytest.mark.parametrize("case", load_cases(), ids=lambda c: c["case_id"])
def test_scenario_is_valid_and_explainable(case: dict) -> None:
    request = PricingRequest.model_validate(case["request"])
    output = sensitivity(request)
    assert output["PV"] >= 0
    assert output["PV_stderr"] >= 0
    assert output["RiskCube"]["cells"]
    explanation = output["explainability"]
    assert explanation["model"] in {
        "risk-neutral GBM Monte Carlo",
        "QuantLib-Risks analytic Black-Scholes-Merton",
    }
    assert explanation["moneyness"] > 0
    assert "barrier_events" in explanation
    assert "coupon_state" in explanation


def test_attachment_sample_fixture_prices_and_generates_riskcube() -> None:
    request = PricingRequest.model_validate(json.loads(ATTACHMENT_SAMPLE.read_text()))
    output = sensitivity(request)
    assert output["PV"] >= 0
    assert len(output["RiskCube"]["cells"]) == 4
    assert output["explainability"]["lifecycle_state"] == "LIVE"
    assert output["explainability"]["coupon_state"]["memory_carry"] >= 0
    assert all(
        "finite-difference" in cell["method"] for cell in output["RiskCube"]["cells"]
    )


def test_put_call_intrinsic_direction() -> None:
    case = load_cases()[0]["request"]
    call = case.copy()
    call["parameters"] = {
        **case["parameters"],
        "option_type": "call",
        "payoff_type": "vanilla",
    }
    put = case.copy()
    put["parameters"] = {
        **case["parameters"],
        "option_type": "put",
        "payoff_type": "vanilla",
    }
    call_pv = price_request(PricingRequest.model_validate(call)).pv
    put_pv = price_request(PricingRequest.model_validate(put)).pv
    assert call_pv >= 0 and put_pv >= 0


def test_knock_out_rebate_is_reflected_in_state() -> None:
    request = PricingRequest.model_validate(load_cases()[0]["request"])
    request.parameters.barriers = [
        {
            "direction": "down",
            "event": "KO",
            "level": 1000.0,
            "monitoring": "global",
            "rebate": 0.01,
        }
    ]
    request.parameters.payoff_type = "barrier"
    result = price_request(request)
    assert result.diagnostics["barrier_events"]
    assert result.pv > 0


def test_smooth_vanilla_uses_aad_and_emits_full_greek_vector() -> None:
    request = PricingRequest.model_validate(load_cases()[0]["request"])
    request.parameters.payoff_type = "vanilla"
    request.parameters.barriers = []
    request.parameters.accrual = None
    output = sensitivity(request)
    assert all("AAD" in cell["method"] for cell in output["RiskCube"]["cells"])
    assert {"delta", "gamma", "vega", "rho", "theta", "vanna", "volga", "charm"} <= set(
        output["RiskCube"]["cells"][0]["sensitivities"]
    )


def test_vercel_asgi_app_is_exported() -> None:
    assert app is not None
    assert hasattr(app, "routes")


def test_memory_coupon_has_carry_state() -> None:
    case = load_cases()[0]["request"]
    case["parameters"] = {
        **case["parameters"],
        "payoff_type": "fcn",
        "accrual": {
            "coupon_rate": 0.12,
            "memory": True,
            "observation_frequency": "monthly",
            "observations": 12,
        },
    }
    result = price_request(PricingRequest.model_validate(case))
    assert "memory_carry" in result.diagnostics["coupon_state"]
