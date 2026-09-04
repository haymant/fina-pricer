from __future__ import annotations

import json
from math import isfinite
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
    assert output["PV_amount"] == output["PV"]
    assert output["PV_currency"] == "USD"
    assert output["price_pct_of_notional"] == pytest.approx(output["PV"] / output["notional"] * 100.0)
    assert output["RiskCube"]["valuation"]["price_pct_of_notional"] == pytest.approx(output["price_pct_of_notional"])
    assert all(cell["pv_amount"] == output["PV"] for cell in output["RiskCube"]["cells"])
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


def test_svi_aad_emits_skew_and_cross_greeks() -> None:
    case = json.loads(ATTACHMENT_SAMPLE.read_text())
    case["RiskFactorKeys"] = [
        {"type": "Spot", "underlying": "UND_A HK", "temporal_role": "ValuationDate", "date": "2027-05-18"},
        {"type": "Volatility", "underlying": "UND_A HK", "expiry": "2028-05-25", "strike": 5.5858},
        {"type": "InterestRate", "underlying": "USD", "tenor": "5Y", "date": "2027-05-18"},
        {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "rho"},
        {"type": "SVIParameter", "underlying": "UND_A HK", "expiry": "2028-05-25", "surface_parameter": "b"},
    ]
    case["parameters"] = {
        **case["parameters"],
        "payoff_type": "vanilla",
        "accrual": None,
        "svi": {"a": 0.04, "b": 0.12, "rho": -0.35, "m": 0.0, "sigma": 0.20},
    }
    output = sensitivity(PricingRequest.model_validate(case))
    assert "SVI" in output["explainability"]["model"]
    assert "reverse-mode SVI" in output["RiskCube"]["cells"][0]["method"]
    for cell in output["RiskCube"]["cells"]:
        sensitivities = cell["sensitivities"]
        assert "skew_sensitivity" in sensitivities
        assert "delta_vega" in sensitivities
        assert "gamma_vega" in sensitivities
        assert all(isfinite(value) for value in sensitivities.values() if isinstance(value, (int, float)))
    rho_cell = next(cell for cell in output["RiskCube"]["cells"] if cell["rfk"].get("surface_parameter") == "rho")
    assert rho_cell["sensitivities"]["svi_rho"] != 0


def test_sigmoid_smoothed_barrier_uses_aad_and_finite_greeks() -> None:
    case = json.loads(ATTACHMENT_SAMPLE.read_text())
    case["RiskFactorKeys"] = [
        {"type": "Spot", "underlying": "UND_A HK"},
        {"type": "Volatility", "underlying": "UND_A HK"},
    ]
    case["parameters"] = {
        **case["parameters"],
        "payoff_type": "barrier",
        "accrual": None,
        "smooth_barrier": True,
        "barrier_smoothing_width": 0.01,
        "barriers": [{
            "direction": "down",
            "event": "KI",
            "level": 0.8 * case["UnwindMapRaw"]["underlyings"][0]["strikePrice"],
            "monitoring": "global",
            "observation_dates": [],
            "rebate": 0.0,
        }],
    }
    output = sensitivity(PricingRequest.model_validate(case))
    assert "sigmoid-smoothed" in output["explainability"]["model"]
    assert all("sigmoid-smoothed barrier" in cell["method"] for cell in output["RiskCube"]["cells"])
    for cell in output["RiskCube"]["cells"]:
        assert all(isfinite(value) for value in cell["sensitivities"].values() if isinstance(value, (int, float)))


def test_aapl_tsla_worst_of_basket_has_per_underlying_risk() -> None:
    request = PricingRequest.model_validate(json.loads((ROOT / "data" / "basket_aapl_tsla.json").read_text()))
    output = sensitivity(request)
    assert output["explainability"]["underlyings"] == ["AAPL US", "TSLA US"]
    assert output["explainability"]["basket_method"] == "worst_of"
    assert output["price_pct_of_notional"] == pytest.approx(output["PV"] / output["notional"] * 100.0)
    spot_cells = [c for c in output["RiskCube"]["cells"] if c["rfk"]["type"] == "Spot"]
    assert {c["rfk"]["underlying"] for c in spot_cells} == {"AAPL US", "TSLA US"}
    assert all(c["bump"] == pytest.approx(c["rfk"]["underlying"] == "AAPL US" and 2.0 or 2.5) for c in spot_cells)
    assert {event["underlying"] for event in output["explainability"]["barrier_events"]} == {"AAPL US", "TSLA US"}
    levels = {event["underlying"]: event["level"] for event in output["explainability"]["barrier_events"]}
    assert levels["AAPL US"] == pytest.approx(0.80)
    assert levels["TSLA US"] == pytest.approx(0.70)
    assert all(event["level_type"] == "relative_initial" for event in output["explainability"]["barrier_events"])


def test_basket_is_capped_at_three_underlyings() -> None:
    case = json.loads((ROOT / "data/basket_aapl_tsla.json").read_text())
    case["UnwindMapRaw"]["underlyings"].extend([
        {**case["UnwindMapRaw"]["underlyings"][0], "name": "NVDA US"},
        {**case["UnwindMapRaw"]["underlyings"][0], "name": "AMZN US"},
    ])
    case["MarketDataSnapshot"]["spot_data"].extend([
        {"rfk": {"underlying": "NVDA US"}, "value": 200.0},
        {"rfk": {"underlying": "AMZN US"}, "value": 200.0},
    ])
    with pytest.raises(ValueError, match="at most 3"):
        PricingRequest.model_validate(case)


def test_default_bump_is_one_percent() -> None:
    case = json.loads((ROOT / "data/attachment_sample.json").read_text())
    case["parameters"].pop("bump_size", None)
    request = PricingRequest.model_validate(case)
    assert request.parameters.bump_size == pytest.approx(0.01)


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
