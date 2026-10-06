# Listed SPX RR-spread translation protocol

Frozen 2026-10-06, against forecasting baseline commit `973809c`, before any 2025 option-chain selection or strategy P&L was inspected. Forecasting and RR25 robustness results were already known, so this is not a pristine new economic hypothesis. It tests whether the existing forecast translates into listed-option P&L after bid/ask and delta hedging. Rules must not be changed after opening locked P&L merely to improve results. The forecasting protocol and conclusions remain separate.

## Signal and clock

Use the existing QC-masked state $X_t=RR^{30}_t-RR^{60}_t$, where downside RR25 is 25-delta put IV minus 25-delta call IV. Fit $X_{t+5}-X_t=\alpha+\beta z_t+\varepsilon_t$ once using only 2023–2024 labels matured by 2024-12-31. The expanding z-score uses the sample standard deviation of earlier valid states, requires 60 prior valid observations, and excludes the current state from its reference moments. Earlier observed 2025 states may update these moments; coefficients stay fixed and no 2025 targets enter fitting.

Use sign only: positive forecasts buy the 30D downside RR and sell the 60D downside RR; negative forecasts reverse all four signs. Missing/nonfinite forecasts or exactly zero produce no trade. No magnitude sizing, thresholds, model alternatives, or recursive fitting.

Signal is known at EOD $t$, entry is EOD $t+1$, and scheduled exit is EOD $t+5$. These are authoritative SPX trading-session offsets, not calendar-day increments. The executable holding period is four session intervals. Original forecast `session_index` begins at 2023-01-03 and is never compressed for missing observations; report all five `session_index % 5` offsets. A one-session delayed exit can meet the next same-offset entry at a common close; close the old basket before opening the next.

## Entry selection

Validate the historical SPX identities and retain `am_settlement == 0`. Use the existing cleaning, call/put matching, unfiltered parity-OLS forward/discount inference, and midpoint IV solvers. Invalid expiry estimates make that expiry unavailable. Valid inputs need positive finite $F,D,K,\tau,\sigma$; zero-IV observations cannot supply vega-normalised positions. No vendor IV/delta/vega is substituted.

Selection quotes require finite bid/ask, bid strictly positive, and ask at least bid. Locked quotes are executable under this rule. Do not add the exploratory SVI relative-spread screen, volume/OI cutoffs, or an IV cap to trading selection. Zero-bid quotes may still contribute to the existing parity estimator but cannot be selected for execution. Contract identities must be unique within each date.

Choose two distinct expiries minimising $|DTE_{30}-30|+|DTE_{60}-60|$, with $DTE_{30}<DTE_{60}$ and each absolute error at most seven calendar days. Each expiry must contain executable, valid-IV OTM put and call rows. Break pair ties by earliest 30D expiry, then earliest 60D expiry. Apply this expiry choice before the delta-distance check; do not switch to a farther pair to rescue missing 25-delta wings.

Within each selected expiry, minimise distance to signed unadjusted Black forward delta $-0.25$ for puts and $+0.25$ for calls. Require distance at most 0.05, inclusive. Keep the repo's OTM convention: put $K<F$, call $K>F$. Break contract ties by lower strike, then lower option ID. Selection consumes only entry-date rows. Future exit availability never changes entry eligibility.

## Greeks, quantities and units

The existing pricing model is discounted Black on an inferred forward. With $d_1$ from those same inputs, forward delta is $N(d_1)$ for a call and $N(d_1)-1$ for a put. Use this unadjusted convention for 25-delta selection, matching RR25. For the SPX hedge use spot sensitivity $\Delta_S=(DF/S)\Delta_F$, holding the inferred carry ratio $F/S$ fixed for the instantaneous spot sensitivity. Forward delta itself is not a spot-price hedge coefficient.

Vega is $DF\varphi(d_1)\sqrt{\tau}$ per unit decimal volatility. Apply the OptionMetrics contract multiplier 100 to both vega and spot delta; execution premiums are also multiplied by 100. Quantities are fractional theoretical contracts, not integer rounded orders. Results will be dollars per unit of entry gross vega per decimal volatility, not returns on deployed capital.

Long downside RR has raw quantities $+1/v_{put}$ and $-1/v_{call}$; short RR reverses them. Scale all four together so $\sum_i|q_iv_i|=1$. Thus each wing has absolute entry vega 0.25 and net entry vega is zero. Freeze quantities until exit. Record identities, expiry/DTE, strikes/types, quotes/midpoint, IV, forward delta, contract spot delta/vega, quantity, per-leg and portfolio delta, gross/net vega and initial hedge.

## Execution, exit and hedge accounting

Longs enter at ask and exit at bid; shorts enter at bid and exit at ask. Midpoint P&L is diagnostic only. No discretionary slippage.

Keep the original four contracts. Exit the entire basket together at $t+5$ if all four have executable quotes. Otherwise try $t+6$ once, requiring all four executable quotes on that common date. If unavailable, record `unevaluable`; do not roll, replace contracts, stagger exits or search later dates. Report contract-level and basket coverage separately. Do not retrospectively filter entries by future availability.

Set entry hedge $H=-\sum_iq_i\Delta_{S,i}$. At each later EOD earn $H_{previous}(S_{today}-S_{previous})$ first. Recompute deltas from today's quotes/IV and inferred $F,D$ only, then set the next hedge. At exit unwind the previous hedge to zero without opening a new one. Record positions, turnover and hedge P&L. Every intervening session needs valid hedge inputs and SPX close; missing inputs make accounting unevaluable, with no stale-delta or missing-session bridge. Exit requires prices, not a new IV inversion.

Intermediate hedge marks can have a zero bid if bid/ask are finite, nonnegative and noncrossed and midpoint IV remains valid; they are not executable entry/exit quotes. Missing quotes or IV still make hedge accounting unevaluable. This distinction permits a zero-bid scheduled exit to use the frozen next-session fallback without inventing a stale delta.

The SPX level is a frictionless hedge proxy; it is not a futures execution model. Option bid/ask costs are included, but hedge transaction costs, cash financing, dividends and option-premium funding are zero in v1. No capital-return or profitability claim follows from validating this machinery.

## Development feasibility and release gate

Use only 2023 and 2024 annual option files. Test expiry coverage, four-wing delta availability, distances, quote widths, vega/delta/quantities, same-contract quote coverage and one-session fallback. For feasibility, a canonical positive direction displays position properties on every entry date independently of signal availability; it does not simulate development trades or P&L. The final development RR25 signal fit can be checked separately. Select examples as the first valid entry in each calendar quarter.

Report every session on the original SPX calendar, including absent-chain dates. Assess lifecycle only where origin, entry, scheduled exit and the allowed fallback are all inside development; classify boundary observations separately. Report complete intermediate quote/hedge-input coverage as well as exit availability. Never calculate real-data option-price changes, hedge P&L, aggregate returns, Sharpe or win rates in this notebook.

Poor development expiry/delta coverage, widespread missing exits, or absent required columns stop the release for review. Do not loosen seven-day expiry tolerance, 0.05 delta distance or fallback limits automatically. 2025 selection and P&L require a separate explicit instruction after reviewing these results.

## Frozen decisions

| Decision | Choice |
| --- | --- |
| Signal | Existing static development-fitted M2 for 5-session RR30–RR60 change; sign only |
| Timing | EOD origin; next-session entry; original fifth-session exit |
| Expiries | Distinct ordered pair, minimum total DTE error, each within seven days |
| Contracts | Executable OTM signed forward 25-delta wings, distance ≤ 0.05; deterministic ties |
| Quantities | Equal absolute wing vega; unit total entry gross vega; fixed fractional contracts |
| Pricing | Observed bid/ask; 100 multiplier; no fitted-surface execution price |
| Missing exit | Whole-basket next-session fallback once; otherwise unevaluable |
| Hedge | Previous SPX hedge earns next interval; today's spot delta sets following hedge |
| Costs | Option bid/ask; frictionless index hedge and zero financing in v1 |
| Offsets | All five original origin-session offsets; none selected |
| Current scope | Synthetic accounting tests and development-only feasibility; no real strategy P&L |
