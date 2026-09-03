# Custom FCN barrier permutation analysis

## Test design

The custom matrix contains 36 cases: OTM, ATM, and ITM spot states; memory off/on; no barrier; global up-and-out; local up-and-out observed mid-life; local up-and-out observed only at expiry; global down-and-in; and local down-and-in observed mid-life. The base FCN is held constant across each comparison: same notional, coupon schedule, volatility, discount rate, expiry, and random-number seed. Barrier levels are strike-relative: 120% for up-KO and 80% for down-KI.

The report is located at `data/custom_barrier_matrix_report.json`. Each barrier event includes `monitoring`, `observation_dates`, and a pathwise `hit_probability`. This probability is the fraction of simulated paths that meet the barrier condition under the configured monitoring rule; it is not a market-implied probability outside the model assumptions.

## Financial interpretation

| Condition | OTM example | ATM example | ITM example | Interpretation |
|---|---:|---:|---:|---|
| No barrier, memory off | PV 768,062 | PV 791,148 | PV 819,528 | Baseline FCN value with scheduled coupon and par redemption in the simplified contract. |
| Global up-KO | Hit 22.35% / PV 755,061 | Hit 41.28% / PV 760,989 | Hit 80.98% / PV 754,346 | Continuous monitoring makes KO materially more likely, especially ITM because the spot starts closer to the 120% upper level. KO truncates future optionality and changes spot delta toward a more event-driven profile. |
| Local up-KO, mid-life observation | Hit 1.75% / PV 766,780 | Hit 8.80% / PV 783,892 | Hit 37.40% / PV 787,325 | Restricting observation to one mid-life date sharply reduces KO probability. The PV is therefore closer to the no-barrier baseline than under global monitoring. |
| Local up-KO, expiry observation | Hit 13.00% / PV 760,397 | Hit 23.63% / PV 773,235 | Hit 42.68% / PV 782,017 | A single expiry observation has more exposure than the selected mid-life date in this seed/scenario set, but less than global monitoring. The result is sensitive to the observation date because the spot distribution widens with time. |
| Global down-KI | Hit 59.20% / PV 672,574 | Hit 33.50% / PV 736,992 | Hit 12.65% / PV 798,330 | Continuous monitoring gives the greatest chance of activating downside principal loss. The PV reduction is largest in OTM cases where KI changes a mostly protected payoff into a loss-sensitive redemption. |
| Local down-KI, mid-life observation | Hit 19.53% / PV 730,786 | Hit 5.12% / PV 781,935 | Hit 0.35% / PV 818,767 | A single local observation substantially reduces KI activation, so PV remains near the protected baseline. The effect is strongest for ITM, where the spot is far above the 80% KI level. |
| Memory on, no barrier | PV 789,162 | PV 808,099 | PV 825,097 | Memory increases coupon value by carrying unpaid coupons forward. In the report, the coupon-paid and memory-carry diagnostics rise relative to memory-off cases. |

## Sensitivity interpretation

The sensitivity results are deliberately discontinuous around barrier boundaries. A global up-KO changes the spot exposure because paths that cross 120% are settled at par rather than retaining the pre-event payoff. This creates large, unstable-looking gamma concentrations around the barrier: the large gamma values are event-state curvature from common-random-number finite differences, not smooth Black–Scholes gamma and should not be interpreted as a continuous local derivative.

Global down-KI creates the strongest downside state exposure. Its spot delta can become positive in the report because a higher spot reduces the chance of breaching the lower barrier and therefore reduces expected loss. This can coexist with a put-like terminal component after KI. In contrast, local down-KI has lower hit probability and generally keeps delta closer to the protected FCN profile.

Memory changes both PV and spot sensitivity. When coupons are missed, the memory balance increases; when a later observation is eligible, the accumulated balance is released. This introduces a state-dependent positive carry to PV but also creates jump-like sensitivity around coupon eligibility dates. The report’s `coupon_paid` and `memory_carry` fields should be read together with the spot delta rather than treated as independent additive Greeks.

Global versus local monitoring is economically important. Global monitoring observes every simulated step, so it detects intraday/path excursions that local monitoring intentionally ignores. Local monitoring is only sensitive to the supplied observation dates. Consequently, changing a local observation from mid-life to expiry changes both hit probability and the event-state sensitivity even when all market inputs remain unchanged.

## Important model caveats

The implementation is an explainable Monte Carlo reference pricer, not a production legal-document payoff engine. The custom matrix uses a simplified FCN settlement convention: KI paths redeem at the lower of terminal spot and strike, KO paths redeem at strike, and memory coupons are carried pathwise. Product-specific details such as worst-of baskets, issuer call schedules, settlement lags, business-day adjustments, coupon barriers, and exact KI/KO precedence should be added before production use.

Smooth no-barrier vanilla cases use the tested QuantLib-Risks/XAD AAD backend. Barrier and memory cases use common-random-number finite differences because their payoff contains discontinuous event-state transitions. The JSON report records this method per RiskCube cell rather than presenting those event-state estimates as smooth AAD Greeks.
