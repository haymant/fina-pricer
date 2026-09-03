from __future__ import annotations

from datetime import date
from typing import Any

import QuantLib_Risks as ql  # type: ignore[import-untyped]
from xad.adj_1st import Tape  # type: ignore[import-untyped]


def _value(value: Any) -> float:
    if hasattr(value, "value"):
        raw = value.value
        return float(raw() if callable(raw) else raw)
    return float(value)


def native_xad_vanilla_greeks(
    spot: float,
    strike: float,
    volatility: float,
    rate: float,
    dividend_yield: float,
    eval_date: str,
    expiry: str,
    option_type: str,
    notional: float,
    currency_conversion: float,
) -> dict[str, Any]:
    """Price a vanilla option and collect first-order adjoints from QuantLib-Risks."""
    eval_qldate = ql.Date(
        date.fromisoformat(eval_date).day,
        date.fromisoformat(eval_date).month,
        date.fromisoformat(eval_date).year,
    )
    expiry_date = date.fromisoformat(expiry)
    expiry_qldate = ql.Date(expiry_date.day, expiry_date.month, expiry_date.year)
    ql.Settings.instance().evaluationDate = eval_qldate
    day_counter = ql.Actual365Fixed()
    calendar = ql.NullCalendar()
    spot_real = ql.Real(spot)
    vol_real = ql.Real(volatility)
    rate_real = ql.Real(rate)
    dividend_real = ql.Real(dividend_yield)
    tape = Tape()
    tape.registerInputs([spot_real, vol_real, rate_real, dividend_real])
    with tape:
        tape.newRecording()
        spot_quote = ql.SimpleQuote(spot_real)
        rate_curve = ql.FlatForward(
            0, calendar, ql.QuoteHandle(ql.SimpleQuote(rate_real)), day_counter
        )
        dividend_curve = ql.FlatForward(
            0, calendar, ql.QuoteHandle(ql.SimpleQuote(dividend_real)), day_counter
        )
        vol_curve = ql.BlackConstantVol(
            0, calendar, ql.QuoteHandle(ql.SimpleQuote(vol_real)), day_counter
        )
        process = ql.BlackScholesMertonProcess(
            ql.QuoteHandle(spot_quote),
            ql.YieldTermStructureHandle(dividend_curve),
            ql.YieldTermStructureHandle(rate_curve),
            ql.BlackVolTermStructureHandle(vol_curve),
        )
        ql_type = ql.Option.Call if option_type == "call" else ql.Option.Put
        option = ql.VanillaOption(
            ql.PlainVanillaPayoff(ql_type, strike), ql.EuropeanExercise(expiry_qldate)
        )
        option.setPricingEngine(ql.AnalyticEuropeanEngine(process))
        npv = option.NPV()
        tape.registerOutput(npv)
        tape.clearDerivatives()
        npv.derivative = 1.0
        tape.computeAdjoints()
        scale = notional * currency_conversion / strike
        return {
            "pv": _value(npv) * scale,
            "delta": _value(spot_real.derivative) * scale,
            "gamma": _value(option.gamma()) * scale,
            "vega": _value(vol_real.derivative) * scale,
            "rho": _value(rate_real.derivative) * scale,
            "theta": _value(option.theta()) * scale,
            "fx_delta": _value(npv) * notional / strike,
            "dividend_rho": _value(dividend_real.derivative) * scale,
            "vanna": 0.0,
            "volga": 0.0,
            "charm": 0.0,
            "backend": "AAD: QuantLib-Risks/XAD adj_1st + native analytic second-order/theta",
            "xad_version": "1.5.2",
            "quantlib_risks_version": "1.33.3",
            "higher_order": "gamma and theta obtained from the native QuantLib analytic engine; xad.adj_1st supplies first-order tape adjoints",
        }
