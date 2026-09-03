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
3. `finite_difference`: permitted only for discontinuous barrier, digital, callability, and memory-state transitions, and only when the request explicitly allows a fallback. Report the fallback method per cell.

If the user says “AAD only,” use `QuantLib-Risks`/XAD for every supported smooth QuantLib path and fail closed for unsupported discontinuous sensitivities instead of silently substituting finite differences. Return the unsupported RFKs and the missing AAD capability configuration.
 For mixed mode, use AAD for smooth components and clearly isolate event-state risk as a separate discontinuity or scenario risk rather than misrepresenting it as AAD.

## Standard FCN workflow

1. Load and validate `schema/instrument_key.schema.json`, `schema/unwind_map_raw.schema.json`, `schema/risk_factor_key.schema.json`, `schema/market_data_snapshot.schema.json`, `schema/updated_lifecycle.schema.json`, and `schema/riskcube.schema.json`.
2. Normalize dates, currencies, calendars, notionals, underlyings, strikes, barriers, observation dates, and applied fixings. Reject missing expiry, inconsistent valuation dates, non-positive prices, and ambiguous RFKs.
3. Build market handles. Map spot RFKs to `SimpleQuote` or native AAD variables, vol RFKs to the calibrated volatility surface, interest-rate RFKs to the discount curve, and FX RFKs to FX spot handles. Keep each RFK’s identity attached to the variable on the AAD tape.
4. Apply lifecycle first. Incorporate fixings, adjustment factors, current instrument state, past KI/KO events, coupon memory, accrual carry, and observation calendars before generating future paths.
5. Price the FCN. Model the equity-linked put/call, coupon accrual, memory release, global or local KI/KO barriers, redemption/autocall rules, currency conversion, and discounting. Use QuantLibAAD/XAD for the native path; use the project’s reverse-mode fixed-tape reference only when the native bridge is unavailable and the payoff is smooth.
6. Generate the full Greek vector for every requested RFK: `delta`, `gamma`, `vega`, `rho`, `theta`, `fx_delta`, `vanna`, `volga`, and `charm`. When an SVI surface is configured, also generate `svi_a`, `svi_b`, `svi_rho`, `svi_m`, `svi_sigma`, and `skew_sensitivity`, plus `delta_vega`, `gamma_vega`, `delta_volga`, `delta_rate`, `gamma_rate`, and surface cross-Greeks. Define the SVI convention as total variance `w(k)=a+b*(rho*(k-m)+sqrt((k-m)^2+sigma^2))`, with `k=ln(K/S)`, and state whether skew means `dV/d rho` or a quoted-vol slope. Preserve zero values when a Greek is structurally irrelevant, and explain why. Include method, backend, bump details if applicable, tape/path counts, and error estimates.
7. Construct the RiskCube using stable axes `Underlying`, `Expiry`, `Strike`, and `RiskFactorType`. Use `SVIParameter` RFKs with `surface_parameter` equal to `a`, `b`, `rho`, `m`, or `sigma` for surface-risk cells. Keep one cell per RFK and include explainability diagnostics: moneyness, time to expiry, barrier events, lifecycle state, applied fixings, coupon-memory carry, and model/backend.
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

The matrix runner keeps the payoff family fixed as FCN and varies one contract condition at a time. Include at least OTM, ATM, and ITM spot states; memory off/on; no barrier; global KI/KO; local KI/KO with a mid-life observation; and local KI/KO with an expiry observation. Use strike-relative barrier levels, such as 80% for down-KI and 120% for up-KO, unless the contract specifies otherwise.

Inspect the resulting JSON rather than relying only on console output. For each case, compare `PV`, the spot RiskCube cell, `delta`, `gamma`, `theta`, `method`, `coupon_state`, and `barrier_events[].hit_probability`. Explain that `hit_probability` is the simulated fraction of paths crossing the event condition under the specified monitoring rule, not a standalone market-implied probability.

Explain financial effects using like-for-like comparisons. Global monitoring observes every simulated step and normally produces more barrier hits than a single local observation. A local mid-life observation and a local expiry observation are different contracts; expiry observation can have a wider terminal distribution, while mid-life observation can be less exposed depending on the barrier and drift. Down-KI generally lowers value when it converts protected principal into downside-linked redemption; up-KO changes the remaining optionality and can create event-state delta and gamma. Memory increases expected coupon carry but creates state-dependent jumps around coupon eligibility and release dates.

Treat large or sign-changing barrier gamma as discontinuity/event-state risk, not smooth local curvature. State the model convention for KO settlement, KI redemption, coupon memory, calendars, and fixings. If the report compares vanilla and FCN rows, disclose that the payoff-family change confounds the barrier effect; prefer the bundled apples-to-apples FCN matrix.

Read `references/custom_barrier_financial_analysis.md` when preparing the narrative interpretation. Preserve the generated JSON report as an audit artifact and record the exact seed, path count, step count, valuation date, expiry, barrier levels, observation dates, and AAD/fallback method.
