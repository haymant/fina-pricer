from pathlib import Path

path = Path('/home/ubuntu/fina-pricer-push/src/riskcube_mcp/core.py')
s = path.read_text()
start = s.index('def price_request(')
end = s.index('\ndef _aad_greeks(', start)
new = '''def _spot_for(request: PricingRequest, underlying: Underlying, overrides: dict[str, float]) -> float:
    value = overrides.get(
        f"spot:{underlying.name}",
        _market_value(request.market_data.spot_data, underlying.name, underlying.spot),
    )
    factor = next(
        (x.adjustment_factor for x in request.lifecycle.adjusted_underlyings if x.name == underlying.name),
        1.0,
    )
    return value * factor


def _market_value(points: list[MarketPoint], key: str, fallback: float) -> float:
    for point in points:
        if point.rfk.get(key) is not None:
            return point.value
    return fallback


def _market_value_for(points: list[MarketPoint], key: str, value: str, fallback: float) -> float:
    for point in points:
        if point.rfk.get(key) == value:
            return point.value
    return fallback


def _correlation_matrix(request: PricingRequest, n: int) -> np.ndarray:
    raw = request.parameters.correlation
    if raw is None:
        corr = np.full((n, n), 0.5)
        np.fill_diagonal(corr, 1.0)
    else:
        corr = np.asarray(raw, dtype=float)
    if corr.shape != (n, n):
        raise ValueError(f"correlation must be a {n}x{n} matrix for this basket")
    if not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1.0):
        raise ValueError("correlation must be symmetric with unit diagonal")
    try:
        np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as exc:
        raise ValueError("correlation must be positive definite") from exc
    return corr


def _relative_barrier(barrier: BarrierSpec, initial: float) -> BarrierSpec:
    if barrier.level_type == "relative_initial":
        return barrier
    return barrier.model_copy(update={"level": barrier.level / initial, "level_type": "relative_initial"})


def price_request(
    request: PricingRequest, overrides: dict[str, float] | None = None
) -> PriceResult:
    """Price a single underlying or a capped worst-of/best-of basket with GBM paths."""
    p = request.parameters
    underlyings = request.unwind_map.underlyings
    n = len(underlyings)
    if n > 3:
        raise ValueError("at most three underlyings are supported")
    overrides = overrides or {}
    spots = np.array([_spot_for(request, u, overrides) for u in underlyings], dtype=float)
    strike = underlyings[0].strikePrice
    rng = np.random.default_rng(p.seed)
    t = (date.fromisoformat(p.expiry) - date.fromisoformat(p.eval_datetime)).days / 365.0
    dt = t / p.steps
    corr = _correlation_matrix(request, n)
    z = rng.standard_normal((p.paths, p.steps, n)) @ np.linalg.cholesky(corr).T
    vol = overrides.get("vol", _market_value(request.market_data.vol_data, "underlying", p.volatility))
    rate = overrides.get("rate", _market_value(request.market_data.ir_data, "currency", p.risk_free_rate))
    increments = (rate - p.dividend_yield - 0.5 * vol * vol) * dt + vol * sqrt(dt) * z
    asset_paths = spots[None, None, :] * np.exp(np.cumsum(increments, axis=1))
    asset_paths = np.concatenate([np.broadcast_to(spots, (p.paths, 1, n)), asset_paths], axis=1)
    performance = asset_paths / spots[None, None, :]
    basket_paths = np.min(performance, axis=2) if p.basket_method == "worst_of" else np.max(performance, axis=2)
    terminal = strike * basket_paths[:, -1]
    intrinsic = np.maximum(terminal - strike, 0.0) if p.option_type == "call" else np.maximum(strike - terminal, 0.0)
    payoff = intrinsic.copy()
    state: dict[str, Any] = {"knock_in": False, "knock_out": False, "coupon_paid": 0.0, "memory_carry": 0.0}
    knock_in_mask = np.zeros(p.paths, dtype=bool)
    knock_out_mask = np.zeros(p.paths, dtype=bool)
    barrier_events: list[dict[str, Any]] = []
    coupon_paid_path = np.zeros(p.paths)
    barriers = [b if isinstance(b, BarrierSpec) else BarrierSpec.model_validate(b) for b in p.barriers]
    for original in barriers:
        b = _relative_barrier(original, strike if original.level_type == "absolute" else 1.0)
        hit_mask = _barrier_mask(basket_paths, b, p.eval_datetime, p.expiry)
        if np.any(hit_mask):
            barrier_events.append({"event": b.event, "direction": b.direction, "level": original.level, "level_type": original.level_type, "monitoring": b.monitoring, "observation_dates": b.observation_dates, "hit_probability": float(np.mean(hit_mask))})
            if b.event == "KI":
                knock_in_mask |= hit_mask
                state["knock_in"] = True
            else:
                knock_out_mask |= hit_mask
                state["knock_out"] = True
    if p.payoff_type in {"autocall", "fcn"} and p.accrual:
        accrual = p.accrual
        obs_idx = np.linspace(1, p.steps, accrual.observations, dtype=int)
        memory = np.zeros(p.paths)
        coupon = np.full(p.paths, accrual.coupon_rate / accrual.observations)
        for idx in obs_idx:
            eligible = basket_paths[:, idx] >= 1.0
            paid = np.where(eligible, coupon + (memory if accrual.memory else 0.0), 0.0)
            memory = np.where(eligible, 0.0, memory + coupon)
            coupon_paid_path += paid
        state["coupon_paid"] = float(np.mean(coupon_paid_path))
        state["memory_carry"] = float(np.mean(memory))
        payoff = payoff + coupon_paid_path * strike
    if p.payoff_type == "fcn":
        redemption = np.where(knock_in_mask, np.minimum(terminal, strike), strike)
        payoff = coupon_paid_path * strike + redemption
    if p.payoff_type == "barrier" and any(b.event == "KI" for b in barriers):
        payoff = np.where(knock_in_mask, intrinsic, 0.0)
    if np.any(knock_out_mask):
        rebate = max((b.rebate for b in barriers if b.event == "KO"), default=0.0)
        ko_settlement = strike if p.payoff_type in {"fcn", "autocall"} else rebate
        payoff = np.where(knock_out_mask, ko_settlement, payoff)
    if p.payoff_type == "autocall":
        payoff = np.where(knock_out_mask, np.full(p.paths, strike), payoff)
    discounted = exp(-rate * t) * payoff * p.currency_conversion * request.instrument.notional / strike
    return PriceResult(float(np.mean(discounted)), float(np.std(discounted, ddof=1) / sqrt(p.paths)), {"model": "risk-neutral GBM Monte Carlo", "paths": p.paths, "steps": p.steps, "time_to_expiry_years": t, "underlyings": [u.name for u in underlyings], "basket_method": p.basket_method, "correlation": corr.tolist(), "moneyness": float(basket_paths[0, 0]), "spot_state": "ATM" if abs(basket_paths[0, 0] - 1.0) < 1e-12 else ("OTM" if basket_paths[0, 0] < 1.0 else "ITM"), "barrier_events": barrier_events, "lifecycle_state": request.lifecycle.instrument_state, "applied_fixings": request.lifecycle.applied_fixings, "coupon_state": state})

'''
path.write_text(s[:start] + new + s[end:])
