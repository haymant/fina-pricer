---
name: fina-pricer
description: AAD-first pricing and risk generation for FCN and structured equity-linked instruments. Use when configuring, pricing, testing, or explaining FCN products from InstrumentKey, UnwindMapRaw, RiskFactorKeys, MarketDataSnapshot, UpdatedLifecycle, and RiskCube metadata; use the bundled schemas and demo pipeline.
---

# Fina Pricer

Use this skill to turn a structured FCN request into a validated price, explainable lifecycle state, comprehensive sensitivities, and a RiskCube. Treat the attached contract as the canonical wire format. Read the relevant files in `schema/` before constructing or modifying a payload.

## AAD-first policy

Prefer **true reverse-mode AAD** for every differentiable calculation. Use the XAD/QuantLibAAD C++ stack when it is available and compiled for the runtime. The ordinary Python `QuantLib` wheel is not sufficient evidence of XAD support: use `QuantLib-Risks` plus `xad`, which are available on PyPI and expose the documented `Tape`/`ql.Real` workflow. QuantLib’s official extensions page separately documents the C++ XAD/QuantLibAAD integration. Do not label ordinary QuantLib analytic Greeks or finite differences as XAD AAD.

Configure the AAD backend explicitly in the pipeline. The supported backend order is:

1. `quantlib_risks_xad`: use the PyPI `QuantLib-Risks` package with `xad.adj_1st.Tape`; this is the primary Python AAD backend for supported QuantLib pricing paths.
2. `xad_quantlib`: use a separately built XAD/QuantLibAAD C++ integration only when a project requires features not exposed by the Python package.
3. `finite_difference`: permitted only for discontinuous barrier, digital, callability, and memory-state transitions, and only when the request explicitly allows a fallback. Report the fallback method per cell. For barrier event risk, requests may instead set `parameters.smooth_barrier=true` to use sigmoid-smoothed reverse-mode AAD; report the smoothing width and treat the result as a differentiable approximation, not exact event-state risk.

If the user says “AAD only,” use `QuantLib-Risks`/XAD for every supported smooth QuantLib path and fail closed for unsupported discontinuous sensitivities instead of silently substituting finite differences. Return the unsupported RFKs and the missing AAD capability configuration. If sigmoid smoothing is explicitly requested, it is an AAD approximation and must be labeled with `smooth_barrier`, `barrier_smoothing_width`, and the smoothing convention.
 For mixed mode, use AAD for smooth components and clearly isolate event-state risk as a separate discontinuity or scenario risk rather than misrepresenting it as AAD.

## Standard FCN workflow

1. Load and validate `schema/instrument_key.schema.json`, `schema/unwind_map_raw.schema.json`, `schema/risk_factor_key.schema.json`, `schema/market_data_snapshot.schema.json`, `schema/updated_lifecycle.schema.json`, and `schema/riskcube.schema.json`.
2. Normalize dates, currencies, calendars, notionals, underlyings, strikes, barriers, observation dates, and applied fixings. Reject missing expiry, inconsistent valuation dates, non-positive prices, and ambiguous RFKs. Support one to three underlyings only, with an independent `strikePrice` for each underlying. Each underlying may define its own `barriers` list; these are evaluated on that underlying’s normalized path. Top-level `parameters.barriers` remains a shared basket barrier for backward compatibility.
3. Build market handles. Map each underlying’s spot RFK to its own market value, vol RFK to the calibrated volatility surface, interest-rate RFK to the discount curve, and FX RFK to FX spot handles. Keep each RFK’s identity attached to the variable. Basket paths use correlated GBM and report one RiskCube cell per requested underlying RFK.
4. Apply lifecycle first. Incorporate fixings, adjustment factors, current instrument state, past KI/KO events, coupon memory, accrual carry, and observation calendars before generating future paths. For an FCN, a triggered KI changes redemption to the lower of the terminal basket ratio and 100% of notional; it is a cash-equivalent representation of share delivery and must never create redemption above par.
5. Price the FCN. Model the equity-linked put/call, coupon accrual, memory release, global or local KI/KO barriers, redemption/autocall rules, currency conversion, and discounting. Use QuantLibAAD/XAD for the native path; use the project’s reverse-mode fixed-tape reference only when the native bridge is unavailable and the payoff is smooth.
6. Generate the full Greek vector for every requested RFK: `delta`, `gamma`, `vega`, `rho`, `theta`, `fx_delta`, `vanna`, `volga`, and `charm`. When an SVI surface is configured, also generate `svi_a`, `svi_b`, `svi_rho`, `svi_m`, `svi_sigma`, and `skew_sensitivity`, plus `delta_vega`, `gamma_vega`, `delta_volga`, `delta_rate`, `gamma_rate`, and surface cross-Greeks. Define the SVI convention as total variance `w(k)=a+b*(rho*(k-m)+sqrt((k-m)^2+sigma^2))`, with `k=ln(K/S)`, and state whether skew means `dV/d rho` or a quoted-vol slope. Preserve zero values when a Greek is structurally irrelevant, and explain why. Include method, backend, bump details if applicable, tape/path counts, and error estimates.
7. Construct the RiskCube with canonical axes matching the RFK coordinate fields: `type`, `underlying`, `currency_pair`, `expiry`, `strike`, `tenor`, `temporal_role`, `date`, and `surface_parameter`. Every cell must include both the original `rfk` object and a flat `coordinates` object containing the populated axis values. Keep one cell per RFK and include explainability diagnostics: moneyness, time to expiry, barrier events, lifecycle state, applied fixings, coupon-memory carry, and model/backend. This contract is intentionally long-form and can be flattened directly into DuckDB or Parquet columns.
8. Validate financial sanity. Check PV non-negativity where applicable, finite PV and Greeks, put/call direction, barrier state transitions, coupon-memory release, FX scaling, and convergence across seeds or path counts. Compare AAD against finite differences only as a validation diagnostic, never as the production method when AAD is required. Validate SVI skew direction by perturbing `rho` and compare cross-Greeks against a controlled finite-difference check.

## Demo and testing

Run the bundled demo with the repository’s uv environment:

```bash
uv run --project /path/to/riskcube-mcp python /home/ubuntu/skills/fina-pricer/scripts/demo.py
```

The demo loads a representative FCN payload, creates OTM/ATM/ITM and near/far scenarios, toggles global/local KI/KO barriers and memory accrual, requests a full RFK set, and writes a JSON report. It can be pointed at another payload with `--input` and can enforce AAD-only behavior with `--aad-only`.

Use `schema/` as the source of truth when adding fields. Keep schemas JSON Schema Draft 2020-12 and preserve the original PascalCase top-level field names because they are part of the integration contract.

## Resources

The `schema/` directory contains the formal wire schemas. The `scripts/demo.py` file is the executable sample pipeline. Load `references/xad_quantlib_notes.md` when deciding whether a native XAD/QuantLibAAD bridge is installed or when documenting an AAD limitation.

## Barrier permutation and report workflow

When the task asks for barrier, observation-date, memory, or moneyness analysis, execute the bundled scripts in this order:

```bash
cd /path/to/riskcube-mcp
uv run python /home/ubuntu/skills/fina-pricer/scripts/custom_barrier_matrix.py
uv run python /home/ubuntu/skills/fina-pricer/scripts/inspect_custom_report.py
```

The matrix runner keeps the payoff family fixed as FCN and varies one contract condition at a time. Include at least OTM, ATM, and ITM spot states; memory off/on; no barrier; global KI/KO; local KI/KO with a mid-life observation; and local KI/KO with an expiry observation. Barrier `level` is a multiplier of initial spot by default (`level_type="relative_initial"`), such as `0.80` for down-KI and `1.20` for up-KO; use `level_type="absolute"` only for legacy contracts.

Inspect the resulting JSON rather than relying only on console output. For each case, compare `PV`, the spot RiskCube cell, `delta`, `gamma`, `theta`, `method`, `coupon_state`, and `barrier_events[].hit_probability`. Explain that `hit_probability` is the simulated fraction of paths crossing the event condition under the specified monitoring rule, not a standalone market-implied probability.

Explain financial effects using like-for-like comparisons. Global monitoring observes every simulated step and normally produces more barrier hits than a single local observation. A local mid-life observation and a local expiry observation are different contracts; expiry observation can have a wider terminal distribution, while mid-life observation can be less exposed depending on the barrier and drift. Down-KI generally lowers value when it converts protected principal into downside-linked redemption; up-KO changes the remaining optionality and can create event-state delta and gamma. For FCNs, validate that post-KI redemption is capped at 100% of notional even if the underlying later recovers above strike. Memory increases expected coupon carry but creates state-dependent jumps around coupon eligibility and release dates.

Treat large or sign-changing barrier gamma as discontinuity/event-state risk, not smooth local curvature. State the model convention for KO settlement, KI redemption, coupon memory, calendars, and fixings. If the report compares vanilla and FCN rows, disclose that the payoff-family change confounds the barrier effect; prefer the bundled apples-to-apples FCN matrix.

Read `references/custom_barrier_financial_analysis.md` when preparing the narrative interpretation. Preserve the generated JSON report as an audit artifact and record the exact seed, path count, step count, valuation date, expiry, barrier levels, barrier level type, observation dates, basket method/correlation if applicable, smoothing width if applicable, and AAD/fallback method.

## Sigmoid-smoothed barrier mode

Use `smooth_barrier=true` only when a differentiable approximation is desired for barrier-event Greeks. The engine replaces the hard hit indicator with `1 - product(1 - sigmoid(signed_distance / (width * level)))`, and uses smooth approximations for intrinsic max and KI redemption min. Smaller widths track the hard barrier more closely but can create steep, numerically sensitive Greeks; larger widths improve stability but introduce more PV and risk bias. Run a width-convergence table, for example `0.02`, `0.01`, and `0.005`, and compare against the exact event-state price. Do not use sigmoid smoothing to claim exact barrier prices or exact discontinuity Greeks.

## Basket and bump conventions

The default sensitivity bump is `1%` in relative mode (`bump_size=0.01`). For a basket, request one Spot RFK per underlying, identify it with the exact underlying name, and use `correlation` as an `n × n` positive-definite matrix for `n ≤ 3`. The default basket payoff is worst-of: for each path, the basket level is the minimum of the underlyings’ individually strike-normalized levels. Underlyings may have different strikes; redemption and coupon eligibility use each underlying’s own strike before the worst-of/best-of aggregation. **Independent barriers are supported and are not merged into one common barrier.** Barriers can be assigned independently under each underlying, allowing different KI/KO directions, levels, monitoring schedules, observation dates, and rebates. A KI on any configured underlying activates the FCN adverse redemption state; KO state and rebate handling are aggregated according to the basket contract. The bundled sample is `data/basket_aapl_tsla.json`, containing AAPL US with a 200 strike and 0.80 relative-initial down-KI, and TSLA US with a 220 strike and a 0.70 relative-initial down-KI. The sample demo defaults to this fixture and varies both AAPL and TSLA together for OTM/ATM/ITM scenarios. Explainability reports the owning underlying for each barrier event and its simulated hit probability.

## Common economics and explicit legs

Accept optional `CommonEconomics` for shared `notional`, `payment_currency`, `discount_rate`, and `currency_conversion`, and optional `Legs` for explicit economic definitions. A leg must have a stable `leg_id`, `name`, and `leg_type`: `intrinsic_option`, `funding`, or `coupon`. It may also specify its own notional, sign, strike, option type, coupon rate, memory, observation count, and `pay_if_ki` convention. When no legs are supplied, derive the applicable leg set from the payoff and accrual configuration.

Return `LegResults` alongside aggregate `PV` and `SensitivityResults`. Each leg result must include currency PV, normalized `price` (percentage of leg notional), standard error, and per-RFK sensitivities. For FCN, use the common decomposition **funding at par + intrinsic downside option after KI + coupon cashflows**; allocate KO settlement to funding and suppress coupons on KO paths. Ensure leg PVs add back to the aggregate cashflow PV. Label leg sensitivities as common-random-number leg decomposition risks, and do not misrepresent event-state leg risks as smooth AAD.

For range-accrual coupons, accept `n1` and `n2` arrays with one entry per coupon period. Treat past `n1` values as fixed realized accruing-day counts and future `n1` values as pathwise stochastic counts. Prefer the `accruals` per-period rate array when the source provides it; retain `period_coupon_rates` as an alias. This preserves zero-rate periods such as period 0. For future periods, count simulated daily/step observations inside `range_lower`/`range_upper`, scale the in-range fraction to scheduled `N2`, and calculate `accrual[i] × N1_path / N2`. Apply memory to the period shortfall and discount expected future coupon cashflows. When `payment_dates` are supplied, report only paid historical coupons in `coupon_paid`; keep forward coupons in the coupon-leg PV. Record per-period rate, N1, N2, realized/forward status, coupon amount, discounted forward total, fixed-period count, range convention, and whether future `N1` was stochastic in explainability diagnostics. Reject mismatched schedule lengths and do not assume `N1=N2` for unfixed future periods.
For verification, include a scenario with a non-trivial future range and assert at least one future period has simulated expected `N1` below `N2`; a broad range may legitimately produce values close to full accrual but must still use the simulation path.
