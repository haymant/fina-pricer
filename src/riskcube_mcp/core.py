from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from math import erf, exp, log, sqrt
from typing import Any, Literal

import autograd.numpy as anp  # type: ignore[import-untyped]
import numpy as np
from autograd import grad  # type: ignore[import-untyped]
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .smooth_barrier_backend import smooth_barrier_aad_greeks
from .svi_backend import svi_aad_greeks
from .xad_backend import native_xad_vanilla_greeks


class InstrumentKey(BaseModel):
    model_config = ConfigDict(extra="allow")
    isin: str
    name: str
    strategy_id: str
    product_type: str
    family: str
    group: str
    leg_id: int = 1
    leg_name: str = "OPTION"
    notional: float = Field(gt=0)
    payment_currency: str
    status: str = "LIVE"


class Underlying(BaseModel):
    name: str
    currency: str = "USD"
    spot: float = Field(gt=0)
    strikePrice: float = Field(gt=0)
    barrierPrice: float | None = Field(default=None, gt=0)
    barriers: list[dict[str, Any]] = Field(default_factory=list)
    fx_pair: str | None = None
    calendar: str | None = None
    time: str | None = None
    time_zone: str | None = None


class UnwindMapRaw(BaseModel):
    underlyings: list[Underlying] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def unique_underlyings(self) -> UnwindMapRaw:
        names = [u.name for u in self.underlyings]
        if len(names) != len(set(names)):
            raise ValueError("basket underlying names must be unique")
        return self


class RiskFactorKey(BaseModel):
    type: Literal["Spot", "Volatility", "InterestRate", "FXSpot", "SVIParameter"]
    underlying: str | None = None
    currency_pair: str | None = None
    expiry: str | None = None
    strike: float | None = None
    tenor: str | None = None
    temporal_role: str | None = None
    date: str | None = None
    surface_parameter: Literal["a", "b", "rho", "m", "sigma"] | None = None


class MarketPoint(BaseModel):
    rfk: dict[str, Any] = Field(default_factory=dict)
    value: float


class MarketDataSnapshot(BaseModel):
    spot_data: list[MarketPoint] = Field(default_factory=list)
    vol_data: list[MarketPoint] = Field(default_factory=list)
    ir_data: list[MarketPoint] = Field(default_factory=list)
    fx_data: list[MarketPoint] = Field(default_factory=list)


class AdjustedUnderlying(BaseModel):
    name: str
    adjustment_factor: float = Field(gt=0)


class UpdatedLifecycle(BaseModel):
    instrument_state: str = "LIVE"
    applied_fixings: list[str] = Field(default_factory=list)
    adjusted_underlyings: list[AdjustedUnderlying] = Field(default_factory=list)


class BarrierSpec(BaseModel):
    direction: Literal["up", "down"]
    event: Literal["KI", "KO"]
    level: float = Field(gt=0)
    level_type: Literal["relative_initial", "absolute"] = "relative_initial"
    monitoring: Literal["global", "local"] = "global"
    observation_dates: list[str] = Field(default_factory=list)
    rebate: float = 0.0


class AccrualSpec(BaseModel):
    coupon_rate: float = 0.0
    memory: bool = False
    observation_frequency: Literal["daily", "weekly", "monthly"] = "monthly"
    observations: int = Field(default=12, ge=1)
    pay_if_ki: bool = True


class SVIParameters(BaseModel):
    a: float = Field(gt=0)
    b: float = Field(gt=0)
    rho: float = Field(gt=-1, lt=1)
    m: float
    sigma: float = Field(gt=0)


class PricingParameters(BaseModel):
    eval_datetime: str
    expiry: str
    option_type: Literal["call", "put"] = "put"
    payoff_type: Literal["vanilla", "barrier", "autocall", "fcn"] = "vanilla"
    barriers: list[BarrierSpec] = Field(default_factory=list)
    accrual: AccrualSpec | None = None
    risk_free_rate: float = 0.03
    dividend_yield: float = 0.0
    volatility: float = Field(default=0.25, gt=0)
    svi: SVIParameters | None = None
    paths: int = Field(default=20000, ge=1000, le=500000)
    steps: int = Field(default=64, ge=2, le=512)
    seed: int = 7
    bump_size: float = Field(default=0.01, gt=0)
    bump_mode: Literal["relative", "absolute"] = "relative"
    currency_conversion: float = Field(default=1.0, gt=0)
    smooth_barrier: bool = False
    barrier_smoothing_width: float = Field(default=0.01, gt=0, le=0.25)
    basket_method: Literal["worst_of", "best_of"] = "worst_of"
    correlation: list[list[float]] | None = None

    @model_validator(mode="after")
    def dates_are_ordered(self) -> PricingParameters:
        if date.fromisoformat(self.expiry) <= date.fromisoformat(self.eval_datetime):
            raise ValueError("expiry must be after eval_datetime")
        return self


class CommonEconomics(BaseModel):
    """Economics shared by all decomposed legs.

    These fields are optional so legacy requests remain wire-compatible. When
    supplied, they are the authoritative common notional/currency/discounting
    inputs used by every leg.
    """

    notional: float | None = Field(default=None, gt=0)
    payment_currency: str | None = None
    discount_rate: float | None = None
    currency_conversion: float | None = Field(default=None, gt=0)


class LegDefinition(BaseModel):
    """Economic definition of one simulation leg."""

    leg_id: int
    name: str
    leg_type: Literal["intrinsic_option", "funding", "coupon"]
    notional: float | None = Field(default=None, gt=0)
    sign: Literal["long", "short"] = "long"
    option_type: Literal["call", "put"] | None = None
    strike: float | None = Field(default=None, gt=0)
    coupon_rate: float | None = None
    memory: bool | None = None
    observations: int | None = Field(default=None, ge=1)
    pay_if_ki: bool | None = None


class PricingRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    instrument: InstrumentKey = Field(alias="InstrumentKey")
    unwind_map: UnwindMapRaw = Field(alias="UnwindMapRaw")
    risk_factor_keys: list[RiskFactorKey] = Field(alias="RiskFactorKeys", min_length=1)
    market_data: MarketDataSnapshot = Field(
        alias="MarketDataSnapshot", default_factory=MarketDataSnapshot
    )
    lifecycle: UpdatedLifecycle = Field(
        alias="UpdatedLifecycle", default_factory=UpdatedLifecycle
    )
    parameters: PricingParameters
    common_economics: CommonEconomics | None = Field(
        default=None, alias="CommonEconomics"
    )
    legs: list[LegDefinition] | None = Field(default=None, alias="Legs")

    @model_validator(mode="after")
    def unique_leg_ids_and_types(self) -> PricingRequest:
        if self.legs:
            ids = [leg.leg_id for leg in self.legs]
            if len(ids) != len(set(ids)):
                raise ValueError("leg ids must be unique")
            types = [leg.leg_type for leg in self.legs]
            if len(types) != len(set(types)):
                raise ValueError("each leg type may be defined only once")
        return self


@dataclass(frozen=True)
class PriceResult:
    pv: float
    stderr: float
    diagnostics: dict[str, Any]
    leg_values: dict[str, Any]
    leg_stderrs: dict[str, float]


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _black_scholes(
    spot: float,
    strike: float,
    t: float,
    rate: float,
    div: float,
    vol: float,
    option_type: str,
) -> float:
    if t <= 0:
        return max((spot - strike) if option_type == "call" else (strike - spot), 0.0)
    st = vol * sqrt(t)
    d1 = (log(spot / strike) + (rate - div + 0.5 * vol * vol) * t) / st
    d2 = d1 - st
    if option_type == "call":
        return spot * exp(-div * t) * _norm_cdf(d1) - strike * exp(
            -rate * t
        ) * _norm_cdf(d2)
    return strike * exp(-rate * t) * _norm_cdf(-d2) - spot * exp(-div * t) * _norm_cdf(
        -d1
    )


def _barrier_mask(
    path: np.ndarray, barrier: BarrierSpec, eval_date: str, expiry_date: str
) -> np.ndarray:
    monitored = path
    if barrier.monitoring == "local" and barrier.observation_dates:
        start = date.fromisoformat(eval_date)
        total_days = (date.fromisoformat(expiry_date) - start).days
        indices = []
        for raw in barrier.observation_dates:
            offset = (date.fromisoformat(raw) - start).days
            if total_days > 0 and 0 <= offset <= total_days:
                indices.append(round(offset / total_days * (path.shape[1] - 1)))
        monitored = path[:, sorted(set(indices))] if indices else path[:, :0]
    if barrier.direction == "up":
        return (
            np.max(monitored, axis=1) >= barrier.level
            if monitored.size
            else np.zeros(path.shape[0], dtype=bool)
        )
    return (
        np.min(monitored, axis=1) <= barrier.level
        if monitored.size
        else np.zeros(path.shape[0], dtype=bool)
    )


def _barrier_hit(
    path: np.ndarray, barrier: BarrierSpec, eval_date: str, expiry_date: str
) -> bool:
    return bool(np.any(_barrier_mask(path, barrier, eval_date, expiry_date)))


def _spot_for(request: PricingRequest, underlying: Underlying, overrides: dict[str, float]) -> float:
    value = overrides.get(
        f"spot:{underlying.name}",
        _market_value_for(request.market_data.spot_data, "underlying", underlying.name, underlying.spot),
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


def _default_legs(request: PricingRequest) -> list[LegDefinition]:
    if request.legs is not None:
        return request.legs
    legs = [
        LegDefinition(leg_id=1, name="INTRINSIC_OPTION", leg_type="intrinsic_option"),
    ]
    if request.parameters.payoff_type in {"fcn", "autocall"}:
        legs.append(LegDefinition(leg_id=2, name="FUNDING", leg_type="funding"))
    if request.parameters.accrual is not None:
        legs.append(LegDefinition(leg_id=3, name="COUPON", leg_type="coupon"))
    return legs


def _leg_economics(request: PricingRequest, leg: LegDefinition) -> tuple[float, float]:
    common = request.common_economics
    notional = leg.notional or (common.notional if common and common.notional else request.instrument.notional)
    conversion = common.currency_conversion if common and common.currency_conversion else request.parameters.currency_conversion
    sign = 1.0 if leg.sign == "long" else -1.0
    return notional * sign, conversion


def price_request(
    request: PricingRequest,
    overrides: dict[str, float] | None = None,
    component: str | None = None,
) -> PriceResult:
    """Price a single underlying or a capped worst-of/best-of basket with GBM paths."""
    p = request.parameters
    underlyings = request.unwind_map.underlyings
    n = len(underlyings)
    if n > 3:
        raise ValueError("at most three underlyings are supported")
    overrides = overrides or {}
    spots = np.array([_spot_for(request, u, overrides) for u in underlyings], dtype=float)
    strikes = np.array([u.strikePrice for u in underlyings], dtype=float)
    rng = np.random.default_rng(p.seed)
    t = (date.fromisoformat(p.expiry) - date.fromisoformat(p.eval_datetime)).days / 365.0
    dt = t / p.steps
    corr = _correlation_matrix(request, n)
    z = rng.standard_normal((p.paths, p.steps, n)) @ np.linalg.cholesky(corr).T
    vols = np.array([overrides.get(f"vol:{u.name}", _market_value_for(request.market_data.vol_data, "underlying", u.name, p.volatility)) for u in underlyings], dtype=float)
    common_rate = request.common_economics.discount_rate if request.common_economics and request.common_economics.discount_rate is not None else p.risk_free_rate
    conversion = request.common_economics.currency_conversion if request.common_economics and request.common_economics.currency_conversion is not None else p.currency_conversion
    rate = overrides.get("rate", _market_value_for(request.market_data.ir_data, "currency", request.instrument.payment_currency, common_rate))
    increments = (rate - p.dividend_yield - 0.5 * vols[None, None, :] ** 2) * dt + vols[None, None, :] * sqrt(dt) * z
    asset_paths = spots[None, None, :] * np.exp(np.cumsum(increments, axis=1))
    asset_paths = np.concatenate([np.broadcast_to(spots, (p.paths, 1, n)), asset_paths], axis=1)
    performance = asset_paths / spots[None, None, :]
    normalized_levels = asset_paths / strikes[None, None, :]
    basket_paths = np.min(performance, axis=2) if p.basket_method == "worst_of" else np.max(performance, axis=2)
    basket_ratios = np.min(normalized_levels, axis=2) if p.basket_method == "worst_of" else np.max(normalized_levels, axis=2)
    terminal_ratio = basket_ratios[:, -1]
    intrinsic_ratio = np.maximum(terminal_ratio - 1.0, 0.0) if p.option_type == "call" else np.maximum(1.0 - terminal_ratio, 0.0)
    default_notional = request.common_economics.notional if request.common_economics and request.common_economics.notional else request.instrument.notional
    intrinsic_payoff = intrinsic_ratio * default_notional
    funding_payoff = np.zeros(p.paths)
    coupon_payoff = np.zeros(p.paths)
    state: dict[str, Any] = {"knock_in": False, "knock_out": False, "coupon_paid": 0.0, "memory_carry": 0.0}
    knock_in_mask = np.zeros(p.paths, dtype=bool)
    knock_out_mask = np.zeros(p.paths, dtype=bool)
    barrier_events: list[dict[str, Any]] = []
    coupon_paid_path = np.zeros(p.paths)
    barriers = [b if isinstance(b, BarrierSpec) else BarrierSpec.model_validate(b) for b in p.barriers]
    barrier_specs: list[tuple[str | None, BarrierSpec, np.ndarray, float]] = [(None, b, basket_paths, 1.0) for b in barriers]
    for index, underlying in enumerate(underlyings):
        for raw in underlying.barriers:
            barrier_specs.append((underlying.name, raw if isinstance(raw, BarrierSpec) else BarrierSpec.model_validate(raw), performance[:, :, index], spots[index]))
    for underlying_name, original, barrier_path, reference in barrier_specs:
        b = _relative_barrier(original, reference if original.level_type == "absolute" else 1.0)
        hit_mask = _barrier_mask(barrier_path, b, p.eval_datetime, p.expiry)
        if np.any(hit_mask):
            barrier_events.append({"underlying": underlying_name or "basket", "event": b.event, "direction": b.direction, "level": original.level, "level_type": original.level_type, "monitoring": b.monitoring, "observation_dates": b.observation_dates, "hit_probability": float(np.mean(hit_mask))})
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
            eligible = basket_ratios[:, idx] >= 1.0
            paid = np.where(eligible, coupon + (memory if accrual.memory else 0.0), 0.0)
            if not accrual.pay_if_ki:
                paid = np.where(knock_in_mask, 0.0, paid)
            memory = np.where(eligible, 0.0, memory + coupon)
            coupon_paid_path += paid
        state["coupon_paid"] = float(np.mean(coupon_paid_path))
        state["memory_carry"] = float(np.mean(memory))
        coupon_payoff = coupon_paid_path * default_notional
    if p.payoff_type == "fcn":
        funding_payoff = np.full(p.paths, default_notional)
        # FCN redemption is par funding plus a short downside intrinsic option
        # after KI. This makes the three legs add back exactly to redemption.
        intrinsic_payoff = np.where(
            knock_in_mask, -np.maximum(1.0 - terminal_ratio, 0.0) * default_notional, 0.0
        )
        payoff = funding_payoff + intrinsic_payoff + coupon_payoff
    else:
        payoff = intrinsic_payoff
    if p.payoff_type == "barrier" and any(original.event == "KI" for _, original, _, _ in barrier_specs):
        intrinsic_payoff = np.where(knock_in_mask, intrinsic_payoff, 0.0)
        payoff = intrinsic_payoff
    if np.any(knock_out_mask):
        rebate = max((original.rebate for _, original, _, _ in barrier_specs if original.event == "KO"), default=0.0)
        ko_settlement = default_notional if p.payoff_type in {"fcn", "autocall"} else rebate * default_notional
        payoff = np.where(knock_out_mask, ko_settlement, payoff)
        if p.payoff_type == "fcn":
            funding_payoff = np.where(knock_out_mask, ko_settlement, funding_payoff)
            intrinsic_payoff = np.where(knock_out_mask, 0.0, intrinsic_payoff)
            coupon_payoff = np.where(knock_out_mask, 0.0, coupon_payoff)
        else:
            intrinsic_payoff = np.where(knock_out_mask, ko_settlement, intrinsic_payoff)
    if p.payoff_type == "autocall":
        payoff = np.where(knock_out_mask, np.full(p.paths, default_notional), payoff)
    leg_values = {
        "intrinsic_option": intrinsic_payoff,
        "funding": funding_payoff,
        "coupon": coupon_payoff,
    }
    if request.legs is not None:
        for leg in request.legs:
            amount, _ = _leg_economics(request, leg)
            if leg.leg_type == "funding":
                leg_values[leg.leg_type] = funding_payoff * amount / default_notional
            elif leg.leg_type == "coupon":
                leg_values[leg.leg_type] = coupon_payoff * amount / default_notional
            elif leg.leg_type == "intrinsic_option":
                leg_values[leg.leg_type] = intrinsic_payoff * amount / default_notional
    selected = payoff if component is None else leg_values.get(component, np.zeros(p.paths))
    discounted_legs = {name: exp(-rate * t) * values * conversion for name, values in leg_values.items()}
    return PriceResult(float(np.mean(exp(-rate * t) * selected * conversion)), float(np.std(exp(-rate * t) * selected * conversion, ddof=1) / sqrt(p.paths)), {"model": "risk-neutral GBM Monte Carlo", "paths": p.paths, "steps": p.steps, "time_to_expiry_years": t, "underlyings": [u.name for u in underlyings], "basket_method": p.basket_method, "correlation": corr.tolist(), "moneyness": float(basket_ratios[0, 0]), "spot_state": "ATM" if abs(basket_ratios[0, 0] - 1.0) < 1e-12 else ("OTM" if basket_ratios[0, 0] < 1.0 else "ITM"), "barrier_events": barrier_events, "lifecycle_state": request.lifecycle.instrument_state, "applied_fixings": request.lifecycle.applied_fixings, "coupon_state": state, "legs": [leg.model_dump(exclude_none=True) for leg in _default_legs(request)], "leg_decomposition": {name: float(np.mean(value)) for name, value in discounted_legs.items()}}, leg_values, {name: float(np.std(value, ddof=1) / sqrt(p.paths)) for name, value in discounted_legs.items()})


def _aad_greeks(request: PricingRequest, z: np.ndarray) -> dict[str, float]:
    """Compute smooth-payoff Greeks with reverse-mode algorithmic differentiation.

    The random-number matrix is held fixed, making the derivative of the same Monte
    Carlo estimator comparable with bump-and-revalue. Event indicators, max/min,
    and memory state are deliberately excluded because they are discontinuous; those
    products use the finite-difference path below.
    """
    p = request.parameters
    u = request.unwind_map.underlyings[0]
    spot0 = request.unwind_map.underlyings[0].spot
    vol0 = p.volatility
    rate0 = p.risk_free_rate
    t0 = (
        date.fromisoformat(p.expiry) - date.fromisoformat(p.eval_datetime)
    ).days / 365.0
    scale = request.instrument.notional / u.strikePrice * p.currency_conversion
    z_ad = anp.asarray(z)

    def estimator(spot: float, vol: float, rate: float, maturity: float) -> Any:
        dt = maturity / p.steps
        increments = (rate - p.dividend_yield - 0.5 * vol * vol) * dt + vol * anp.sqrt(
            dt
        ) * z_ad
        terminal = spot * anp.exp(anp.sum(increments, axis=1))
        payoff = (
            anp.maximum(terminal - u.strikePrice, 0.0)
            if p.option_type == "call"
            else anp.maximum(u.strikePrice - terminal, 0.0)
        )
        return anp.mean(anp.exp(-rate * maturity) * payoff) * scale

    d_spot = float(grad(estimator, 0)(spot0, vol0, rate0, t0))
    d2_spot = float(grad(grad(estimator, 0), 0)(spot0, vol0, rate0, t0))
    d_vol = float(grad(estimator, 1)(spot0, vol0, rate0, t0))
    d_rate = float(grad(estimator, 2)(spot0, vol0, rate0, t0))
    d_time = float(grad(estimator, 3)(spot0, vol0, rate0, t0))
    return {
        "delta": d_spot,
        "gamma": d2_spot,
        "vega": d_vol,
        "rho": d_rate,
        "theta": -d_time,
    }


def sensitivity(request: PricingRequest) -> dict[str, Any]:
    base = price_request(request)
    p = request.parameters
    basket = len(request.unwind_map.underlyings) > 1
    smooth_vanilla = not basket and p.payoff_type == "vanilla" and not p.barriers and p.accrual is None
    smooth_barrier = not basket and p.smooth_barrier and bool(p.barriers) and p.accrual is None
    smooth = smooth_vanilla or smooth_barrier
    aad: dict[str, Any] | None = None
    if smooth:
        u = request.unwind_map.underlyings[0]
        if smooth_barrier:
            u = request.unwind_map.underlyings[0]
            aad = smooth_barrier_aad_greeks(
                u.spot,
                u.strikePrice,
                p.risk_free_rate,
                p.dividend_yield,
                p.volatility,
                (date.fromisoformat(p.expiry) - date.fromisoformat(p.eval_datetime)).days / 365.0,
                p.eval_datetime,
                p.expiry,
                p.option_type,
                p.payoff_type,
                request.instrument.notional,
                p.currency_conversion,
                [b.model_dump() for b in p.barriers],
                p.steps,
                p.paths,
                p.seed,
                p.barrier_smoothing_width,
            )
            model_name = "sigmoid-smoothed barrier Monte Carlo with reverse-mode AAD"
        elif p.svi is not None:
            aad = svi_aad_greeks(
                u.spot,
                u.strikePrice,
                p.risk_free_rate,
                p.dividend_yield,
                (date.fromisoformat(p.expiry) - date.fromisoformat(p.eval_datetime)).days / 365.0,
                p.option_type,
                request.instrument.notional,
                p.currency_conversion,
                p.svi.model_dump(),
            )
            model_name = "SVI implied-volatility surface with reverse-mode AAD"
        else:
            aad = native_xad_vanilla_greeks(
                u.spot,
                u.strikePrice,
                p.volatility,
                p.risk_free_rate,
                p.dividend_yield,
                p.eval_datetime,
                p.expiry,
                p.option_type,
                request.instrument.notional,
                p.currency_conversion,
            )
            model_name = "QuantLib-Risks analytic Black-Scholes-Merton"
        base = PriceResult(
            aad["pv"],
            base.stderr,
            {
                **base.diagnostics,
                "model": model_name,
                "aad_backend": aad["backend"],
                **({"svi_parameters": p.svi.model_dump(), "svi_convention": aad["svi_convention"]} if p.svi is not None else ({"smooth_barrier": True, "barrier_smoothing_width": p.barrier_smoothing_width, "smoothing_convention": aad["smoothing_convention"]} if smooth_barrier else {"xad_version": aad["xad_version"]})),
            },
            base.leg_values,
            base.leg_stderrs,
        )
    h = p.bump_size
    theta_request = request.model_copy(deep=True)
    theta_request.parameters.eval_datetime = (
        date.fromisoformat(p.eval_datetime) + timedelta(days=1)
    ).isoformat()
    theta_fd = (price_request(theta_request).pv - base.pv) / (1.0 / 365.0)
    cells = []
    for rfk in request.risk_factor_keys:
        kind = rfk.type
        base_val = {
            "Spot": next((u.spot for u in request.unwind_map.underlyings if u.name == (rfk.underlying or request.unwind_map.underlyings[0].name)), request.unwind_map.underlyings[0].spot),
            "Volatility": next((uvol for uvol in [_market_value_for(request.market_data.vol_data, "underlying", rfk.underlying or request.unwind_map.underlyings[0].name, p.volatility)] if uvol is not None), p.volatility),
            "InterestRate": p.risk_free_rate,
            "FXSpot": p.currency_conversion,
            "SVIParameter": getattr(p.svi, rfk.surface_parameter) if p.svi is not None and rfk.surface_parameter else 0.0,
        }[kind]
        bump = h * max(abs(base_val), 1.0) if p.bump_mode == "relative" else h
        key = {
            "Spot": f"spot:{rfk.underlying or request.unwind_map.underlyings[0].name}",
            "Volatility": f"vol:{rfk.underlying}" if rfk.underlying else "vol",
            "InterestRate": "rate",
            "FXSpot": "fx",
            "SVIParameter": "svi",
        }[kind]
        if aad is not None:
            parameter = rfk.surface_parameter
            parameter_value = aad.get(f"svi_{parameter}", 0.0) if kind == "SVIParameter" and parameter else 0.0
            greeks = {
                "delta": aad["delta"] if kind == "Spot" else 0.0,
                "gamma": aad["gamma"] if kind == "Spot" else 0.0,
                "vega": aad["vega"] if kind == "Volatility" else 0.0,
                "rho": aad["rho"] if kind == "InterestRate" else 0.0,
                "theta": aad["theta"],
                "fx_delta": aad["fx_delta"] if kind == "FXSpot" else 0.0,
                "vanna": aad["vanna"],
                "volga": aad["volga"],
                "charm": aad["charm"],
                "skew_sensitivity": aad.get("skew_sensitivity", 0.0),
                "svi_a": aad.get("svi_a", 0.0) if parameter == "a" else parameter_value,
                "svi_b": aad.get("svi_b", 0.0) if parameter == "b" else parameter_value,
                "svi_rho": aad.get("svi_rho", 0.0) if parameter == "rho" else parameter_value,
                "svi_m": aad.get("svi_m", 0.0) if parameter == "m" else parameter_value,
                "svi_sigma": aad.get("svi_sigma", 0.0) if parameter == "sigma" else parameter_value,
                **aad.get("cross_greeks", {}),
            }
            method = aad["backend"]
        else:
            if kind == "SVIParameter":
                raise ValueError("SVIParameter sensitivities require a smooth vanilla payoff with parameters.svi configured")
            if key == "fx":
                up = base.pv * (base_val + bump) / base_val
                down = base.pv * (base_val - bump) / base_val
            else:
                up = price_request(request, {key: base_val + bump}).pv
                down = price_request(request, {key: base_val - bump}).pv
            first = (up - down) / (2 * bump)
            greeks = {
                "delta": first if kind == "Spot" else 0.0,
                "gamma": (up - 2 * base.pv + down) / (bump * bump)
                if kind == "Spot"
                else 0.0,
                "vega": first if kind == "Volatility" else 0.0,
                "rho": first if kind == "InterestRate" else 0.0,
                "theta": theta_fd,
                "fx_delta": first if kind == "FXSpot" else 0.0,
                "vanna": 0.0,
                "volga": 0.0,
                "charm": 0.0,
            }
            method = (
                "common-random-number finite-difference (discontinuous payoff fallback)"
            )
        cells.append(
            {
                "rfk": rfk.model_dump(exclude_none=True),
                "sensitivities": greeks,
                "method": method,
                "bump": bump,
            }
        )
    notional = request.common_economics.notional if request.common_economics and request.common_economics.notional else request.instrument.notional
    price_pct = base.pv / notional * 100.0
    stderr_pct = base.stderr / notional * 100.0
    valuation = {
        "pv_amount": base.pv,
        "pv_currency": request.instrument.payment_currency,
        "price_pct_of_notional": price_pct,
        "price_convention": "100.0 means par; price_pct_of_notional = pv_amount / notional * 100",
        "notional": notional,
        "notional_currency": request.instrument.payment_currency,
        "pv_stderr_amount": base.stderr,
        "pv_stderr_pct_of_notional": stderr_pct,
    }
    for cell in cells:
        cell["pv_amount"] = base.pv
        cell["price_pct_of_notional"] = price_pct

    # Leg risks are deliberately calculated from the same common-random-number
    # estimator as the leg price. This keeps decomposition additive and makes
    # event-state sensitivities explicit rather than attributing them to the
    # aggregate product by assumption.
    leg_results: dict[str, Any] = {}
    for leg in _default_legs(request):
        leg_base = price_request(request, component=leg.leg_type)
        leg_cells: list[dict[str, Any]] = []
        leg_theta_request = request.model_copy(deep=True)
        leg_theta_request.parameters.eval_datetime = (
            date.fromisoformat(p.eval_datetime) + timedelta(days=1)
        ).isoformat()
        leg_theta = (
            price_request(leg_theta_request, component=leg.leg_type).pv - leg_base.pv
        ) / (1.0 / 365.0)
        for rfk in request.risk_factor_keys:
            kind = rfk.type
            base_val = {
                "Spot": next((u.spot for u in request.unwind_map.underlyings if u.name == (rfk.underlying or request.unwind_map.underlyings[0].name)), request.unwind_map.underlyings[0].spot),
                "Volatility": _market_value_for(request.market_data.vol_data, "underlying", rfk.underlying or request.unwind_map.underlyings[0].name, p.volatility),
                "InterestRate": p.risk_free_rate,
                "FXSpot": p.currency_conversion,
                "SVIParameter": 0.0,
            }[kind]
            bump = h * max(abs(base_val), 1.0) if p.bump_mode == "relative" else h
            key = {
                "Spot": f"spot:{rfk.underlying or request.unwind_map.underlyings[0].name}",
                "Volatility": f"vol:{rfk.underlying}" if rfk.underlying else "vol",
                "InterestRate": "rate",
                "FXSpot": "fx",
                "SVIParameter": "svi",
            }[kind]
            if kind == "SVIParameter":
                up = down = leg_base.pv
                bump = 0.0
            elif key == "fx":
                up = leg_base.pv * (base_val + bump) / base_val
                down = leg_base.pv * (base_val - bump) / base_val
            else:
                up = price_request(request, {key: base_val + bump}, leg.leg_type).pv
                down = price_request(request, {key: base_val - bump}, leg.leg_type).pv
            first = (up - down) / (2.0 * bump) if bump else 0.0
            leg_cells.append({
                "rfk": rfk.model_dump(exclude_none=True),
                "sensitivities": {
                    "delta": first if kind == "Spot" else 0.0,
                    "gamma": (up - 2.0 * leg_base.pv + down) / (bump * bump) if kind == "Spot" else 0.0,
                    "vega": first if kind == "Volatility" else 0.0,
                    "rho": first if kind == "InterestRate" else 0.0,
                    "theta": -leg_theta,
                    "fx_delta": first if kind == "FXSpot" else 0.0,
                    "vanna": 0.0,
                    "volga": 0.0,
                    "charm": 0.0,
                },
                "method": "common-random-number finite-difference leg decomposition",
                "bump": bump,
            })
        leg_notional, _ = _leg_economics(request, leg)
        leg_results[leg.leg_type] = {
            "leg_id": leg.leg_id,
            "name": leg.name,
            "leg_type": leg.leg_type,
            "pv_amount": leg_base.pv,
            "pv_currency": request.instrument.payment_currency,
            "price_pct_of_notional": leg_base.pv / abs(leg_notional) * 100.0,
            "price": leg_base.pv / abs(leg_notional) * 100.0,
            "price_convention": "100.0 means the leg notional; signed PV carries long/short direction",
            "notional": abs(leg_notional),
            "sign": leg.sign,
            "pv_stderr_amount": leg_base.stderr,
            "sensitivities": leg_cells,
        }

    cube_types = {
        "Spot": "SpotCube",
        "Volatility": "VolCube",
        "InterestRate": "IRCube",
        "FXSpot": "FXCube",
        "SVIParameter": "SVICube",
    }
    return {
        "PV": base.pv,
        "PV_stderr": base.stderr,
        "PV_amount": base.pv,
        "PV_currency": request.instrument.payment_currency,
        "price_pct_of_notional": price_pct,
        "price_convention": valuation["price_convention"],
        "notional": notional,
        "notional_currency": request.instrument.payment_currency,
        "PV_stderr_pct_of_notional": stderr_pct,
        "LegResults": leg_results,
        "explainability": base.diagnostics,
        "SensitivityResults": cells,
        "RiskCube": {
            "cube_type": cube_types.get(request.risk_factor_keys[0].type, "RiskCube"),
            "axes": ["Underlying", "Expiry", "Strike", "RiskFactorType"],
            "valuation": valuation,
            "cells": cells,
        },
    }
