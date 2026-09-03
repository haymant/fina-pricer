from __future__ import annotations

from typing import Any

import autograd.numpy as anp  # type: ignore[import-untyped]
from autograd import jacobian  # type: ignore[import-untyped]
from autograd.scipy.special import erf  # type: ignore[import-untyped]


def _cdf(x: Any) -> Any:
    return 0.5 * (1.0 + erf(x / anp.sqrt(2.0)))


def _svi_variance(log_moneyness: Any, a: Any, b: Any, rho: Any, m: Any, sigma: Any) -> Any:
    x = log_moneyness - m
    return a + b * (rho * x + anp.sqrt(x * x + sigma * sigma))


def svi_aad_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend_yield: float,
    maturity: float,
    option_type: str,
    notional: float,
    currency_conversion: float,
    svi: dict[str, float],
) -> dict[str, Any]:
    """Compute SVI and cross-Greeks with one reverse-mode tape.

    The state vector is [spot, a, b, rho, m, sigma, rate, vol_shift, time].
    `vol_shift` is a parallel implied-volatility shock, so `vega` remains a
    conventional parallel-vol sensitivity while the SVI parameter cells report
    derivatives to total-variance surface parameters. `rho` is the explicit
    SVI skew/asymmetry risk factor; `b` is smile slope amplitude.
    """
    scale = notional * currency_conversion / strike
    option_sign = 1.0 if option_type == "call" else -1.0

    def pv_fn(x: Any) -> Any:
        s, a, b, rho, m, svi_sigma, r, vol_shift, t = x
        k = anp.log(strike / s)
        total_variance = _svi_variance(k, a, b, rho, m, svi_sigma)
        implied_vol = anp.sqrt(anp.maximum(total_variance / t, 1e-12)) + vol_shift
        st = implied_vol * anp.sqrt(t)
        d1 = (anp.log(s / strike) + (r - dividend_yield + 0.5 * implied_vol**2) * t) / st
        d2 = d1 - st
        call = s * anp.exp(-dividend_yield * t) * _cdf(d1) - strike * anp.exp(-r * t) * _cdf(d2)
        put = call - s * anp.exp(-dividend_yield * t) + strike * anp.exp(-r * t)
        return (call if option_sign > 0 else put) * scale

    x = anp.array([spot, svi["a"], svi["b"], svi["rho"], svi["m"], svi["sigma"], rate, 0.0, maturity])
    value = float(pv_fn(x))
    gradient = jacobian(pv_fn)(x)
    hessian = jacobian(jacobian(pv_fn))(x)
    third_order = jacobian(jacobian(jacobian(pv_fn)))(x)
    g = [float(v) for v in gradient]
    h = [[float(v) for v in row] for row in hessian]
    t3 = [[[float(v) for v in row] for row in plane] for plane in third_order]
    return {
        "pv": value,
        "delta": g[0],
        "gamma": h[0][0],
        "vega": g[7],
        "rho": g[6],
        "theta": -g[8],
        "fx_delta": value / currency_conversion if currency_conversion else 0.0,
        "svi_a": g[1],
        "svi_b": g[2],
        "svi_rho": g[3],
        "svi_m": g[4],
        "svi_sigma": g[5],
        "skew_sensitivity": g[3],
        "vanna": h[0][7],
        "volga": h[7][7],
        "charm": -h[0][8],
        "cross_greeks": {
            "delta_vega": h[0][7],
            "gamma_vega": t3[0][0][7],
            "delta_volga": t3[0][7][7],
            "delta_rate": h[0][6],
            "gamma_rate": t3[0][0][6],
            "spot_vol": h[0][7],
            "spot_rate": h[0][6],
            "spot_svi_b": h[0][2],
            "spot_svi_rho": h[0][3],
            "spot_svi_m": h[0][4],
            "spot_svi_sigma": h[0][5],
            "vol_rate": h[7][6],
            "vol_svi_b": h[7][2],
            "vol_svi_rho": h[7][3],
            "vol_svi_m": h[7][4],
            "vol_svi_sigma": h[7][5],
        },
        "backend": "AAD: reverse-mode SVI parameter tape (autograd)",
        "aad_state": "spot,a,b,rho,m,sigma,rate,parallel_vol_shift,time",
        "svi_convention": "w(k)=a+b*(rho*(k-m)+sqrt((k-m)^2+sigma^2)), k=ln(K/S)",
    }
