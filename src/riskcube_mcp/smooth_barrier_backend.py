from __future__ import annotations

from datetime import date
from typing import Any

import autograd.numpy as anp  # type: ignore[import-untyped]
import numpy as np
from autograd import grad  # type: ignore[import-untyped]


def _sigmoid(x: Any) -> Any:
    """Numerically stable logistic function, compatible with autograd."""
    raw = 0.5 * (anp.tanh(0.5 * x) + 1.0)
    eps = 1e-12
    return eps + (1.0 - 2.0 * eps) * raw


def _indices(barrier: dict[str, Any], eval_date: str, expiry_date: str, steps: int) -> list[int]:
    if barrier.get("monitoring", "global") == "global" or not barrier.get("observation_dates"):
        return list(range(steps + 1))
    total_days = (date.fromisoformat(expiry_date) - date.fromisoformat(eval_date)).days
    indices: list[int] = []
    for raw in barrier["observation_dates"]:
        offset = (date.fromisoformat(raw) - date.fromisoformat(eval_date)).days
        if total_days > 0 and 0 <= offset <= total_days:
            indices.append(round(offset / total_days * steps))
    return sorted(set(indices))


def smooth_barrier_aad_greeks(
    spot: float,
    strike: float,
    rate: float,
    dividend: float,
    volatility: float,
    maturity: float,
    eval_date: str,
    expiry_date: str,
    option_type: str,
    payoff_type: str,
    notional: float,
    currency_conversion: float,
    barriers: list[dict[str, Any]],
    steps: int,
    paths: int,
    seed: int,
    width: float,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((paths, steps))
    scale = notional / strike * currency_conversion
    z_ad = anp.asarray(z)

    payoff_eps = 1e-5 * strike

    def _softplus_zero(x: Any) -> Any:
        return 0.5 * (x + anp.sqrt(x * x + payoff_eps * payoff_eps))

    def _smooth_min(x: Any, y: Any) -> Any:
        return 0.5 * (x + y - anp.sqrt((x - y) * (x - y) + payoff_eps * payoff_eps))

    def estimator(spot_: float, vol_: float, rate_: float, maturity_: float) -> Any:
        dt = maturity_ / steps
        increments = (rate_ - dividend - 0.5 * vol_ * vol_) * dt + vol_ * anp.sqrt(dt) * z_ad
        path = anp.concatenate(
            [anp.full((paths, 1), spot_), spot_ * anp.exp(anp.cumsum(increments, axis=1))], axis=1
        )
        terminal = path[:, -1]
        intrinsic = _softplus_zero(terminal - strike) if option_type == "call" else _softplus_zero(strike - terminal)
        payoff = intrinsic
        for barrier in barriers:
            idx = _indices(barrier, eval_date, expiry_date, steps)
            monitored = path[:, idx]
            level = float(barrier["level"])
            barrier_scale = width * level
            if barrier["direction"] == "up":
                probs = _sigmoid((monitored - level) / barrier_scale)
            else:
                probs = _sigmoid((level - monitored) / barrier_scale)
            no_hit = anp.prod(1.0 - probs, axis=1)
            hit_prob = 1.0 - no_hit
            if barrier["event"] == "KO":
                settlement = strike if payoff_type in {"fcn", "autocall"} else float(barrier.get("rebate", 0.0))
                payoff = (1.0 - hit_prob) * payoff + hit_prob * settlement
            else:
                if payoff_type in {"barrier", "vanilla"}:
                    payoff = hit_prob * payoff
                else:
                    ki_redemption = _smooth_min(terminal, strike)
                    payoff = (1.0 - hit_prob) * payoff + hit_prob * ki_redemption
        return anp.mean(anp.exp(-rate_ * maturity_) * payoff) * scale

    d_spot = float(grad(estimator, 0)(spot, volatility, rate, maturity))
    d2_spot = float(grad(grad(estimator, 0), 0)(spot, volatility, rate, maturity))
    d_vol = float(grad(estimator, 1)(spot, volatility, rate, maturity))
    d_rate = float(grad(estimator, 2)(spot, volatility, rate, maturity))
    d_time = float(grad(estimator, 3)(spot, volatility, rate, maturity))
    pv = float(estimator(spot, volatility, rate, maturity))
    return {
        "pv": pv,
        "delta": d_spot,
        "gamma": d2_spot,
        "vega": d_vol,
        "rho": d_rate,
        "theta": -d_time,
        "fx_delta": pv / currency_conversion,
        "vanna": float(grad(grad(estimator, 0), 1)(spot, volatility, rate, maturity)),
        "volga": float(grad(grad(estimator, 1), 1)(spot, volatility, rate, maturity)),
        "charm": float(-grad(grad(estimator, 0), 3)(spot, volatility, rate, maturity)),
        "backend": "AAD: reverse-mode sigmoid-smoothed barrier tape (autograd)",
        "smoothing_convention": "barrier hit probability = 1 - product(1 - sigmoid(signed distance / (width * level)))",
    }
