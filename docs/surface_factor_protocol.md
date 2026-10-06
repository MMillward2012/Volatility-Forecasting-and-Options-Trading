# Constant-factor diagnostic protocol

Specified 2026-10-06 against repository commit `e75a73e`, before this experiment's 2025 outputs were generated or inspected. **2025 was previously examined for forecasting, fixed-contract trading and attribution. This is a blinded post-result re-analysis**, with an implementation freeze before opening its own outputs. It provides an explanatory mapping test; it cannot replace the frozen forecasting or listed-options findings.

## Signal and timing

Retain the previously audited development-fitted RR25 M2 coefficients: intercept $0.0006537436032636668$, slope $-0.0017253059632106394$. No coefficient estimation occurs in this experiment. The source model used 426 eligible development observations with labels matured by 2024-12-31. Its state is $X_t=RR^{30}_t-RR^{60}_t$, with downside RR25 equal to put IV minus call IV at signed unadjusted Black forward deltas $-0.25,+0.25$.

Use the existing expanding past-only sample-mean/sample-standard-deviation z-score with 60 prior valid states. Earlier observed 2025 states can enter later reference moments; the current state enters only the numerator. Forecast outcomes never enter feature construction or fitting. Use forecast sign only. Positive direction buys the 30D put and 60D call, and sells the 30D call and 60D put; negative reverses all four signs.

On the original SPX security-price session calendar, observe the signal at EOD $t$, enter at EOD $t+1$ and exit at EOD $t+5$. The analysis period is 2025-01-02–2025-08-29. Final origins without a complete fifth-session endpoint remain unavailable. Preserve `session_index` from 2023-01-03 and all five `% 5` offsets; missing observations never compress the clock.

## Primary: direct surface factor

Use only the existing metric-specific QC-masked RR25 states. Report in decimal implied-volatility units:

$$
p_t=\operatorname{sign}(\hat y_t)(X_{t+5}-X_{t+1}),\qquad
p_t^{\rm full}=\operatorname{sign}(\hat y_t)(X_{t+5}-X_t),
$$
$$
p_t^{\rm missed}=\operatorname{sign}(\hat y_t)(X_{t+1}-X_t),\qquad
p_t^{\rm full}=p_t^{\rm missed}+p_t.
$$

Missing entry/exit states produce unavailable delayed outcomes. Full-horizon and first-session outcomes can separately be available on different dates. Report their native coverage and also all three contributions on identical complete origins. Do not sum unequal samples to claim a decomposition. No filling, extra state interpolation, scaling, execution-cost adjustment or retrospective subset selection.

## Secondary: ideal daily-reset synthetic RR

Reconstruct the production surface from PM options using the existing cleaning, matched parity OLS, IV inversion, arbitrage-aware multi-start Raw SVI, 14–180 DTE window, ATM sanity rejection, support-aware PAVA and calendar-repair guard. `process_quote_date(..., retain_surface=True)` returns the same grid and expiry-level forwards alongside its original metrics; default callers and mathematics are unchanged.

At every opening/rebalance close, require both production 30D/60D RR25 QC flags. Solve the same unique supported OTM forward-delta roots as `skew_metrics.py` at exactly $30/365$ and $60/365$. No listed identifiers or nearest-expiry/nearest-delta approximation is used. Each newly opened wing has absolute vega $0.25$, total gross vega one and net vega zero, using existing Black vega and contract multiplier 100. Quantities are reset at every daily roll with the original trade direction; magnitude does not size positions.

For forwards and discounts at any synthetic maturity, use the same surrounding accepted SVI expiries as the total-variance surface. Interpolate $F$ linearly and $\log D$ linearly in ACT/365 maturity. No extrapolation; bracket gap must be at most 30 calendar days. Apply the existing maximum 0.05 raw-versus-repaired ATM-IV adjustment guard at requested maturities. Reprice held strikes only where both surrounding maturities support their current $\log(K/F)$. No additional butterfly optimiser or strike interpolation is introduced.

**Age held options by actual elapsed calendar days under ACT/365**, including weekends and holidays. A Friday 30D option has 27D remaining on Monday. Its strike, expiry and quantity stay fixed over that interval. Today's fresh replacement resets to 30D/60D and current 25-delta strikes. This distinguishes ageing options from the daily constant-tenor surface state.

## Self-financing roll and hedge

Price model mids with the existing discounted Black functions. Hedge spot sensitivity is the existing $(DF/S)\Delta_F$, with the multiplier 100. At entry borrow/lend the signed option and hedge purchase amounts so initial cash plus assets is zero. At each later close:

1. Reprice the previous basket at today's surface, spot and aged maturity; earn its price difference using previous quantities.
2. Earn the previous close's hedge times today's SPX price change.
3. Sell the old basket, buy today's replacement at model mid and exchange the hedge difference at today's spot. Opening the replacement is a cash/asset transfer, never income.
4. Today's replacement delta sets the next interval's hedge. At exit liquidate options and hedge to cash without opening a replacement.

Require interval wealth change to equal option plus hedge P&L and final cash to equal accumulated P&L within $10^{-10}$. Report the same option path without hedge as the unhedged diagnostic. Cash earns zero interest; no financing, transaction costs, bid/ask, capital denominator or annualised Sharpe. Units match the previous unit-gross-entry-vega accounting, but risk is refreshed each day. **Idealised model-mid constant-factor tracking portfolio; not an executable trading result.**

Any missing session surface, failed fresh RR25 QC, unsupported held strike/maturity or missing SPX close makes the synthetic path unavailable. Entry never depends on future availability. Preserve the attempted origin and first failure reason; no fallback, stale marks or substitution. Report direct-factor and synthetic coverage separately and their common origins as a labelled mapping diagnostic. Cumulative pooled all-origin results are overlapping/descriptive; report every original offset without choosing one.

For explanation, retain a fixed sequential interval repricing attribution: change maturity holding yesterday's $F,D,\sigma$; change $F,D$ holding yesterday's IV; then update held-strike IV. These three option components reconcile exactly. Retain the entry-vega linear approximation to the held IV change and the actual previous-close hedge. Allocation depends on the declared order; forward/discount plus hedge is not uniquely gamma or economic carry. No inference of a causal mechanism follows from aggregate allocation alone.

## Freeze and release gates

Phase A reads no existing 2025 trading/result audits or attribution outputs. Synthetic tests validate timing, sign, delta/vega, support, cash conservation, calendar ageing and absence of fitting/outcome inputs. Development checks use only 2023/2024 annual option files and development states/prices: report full-period state availability, then reconstruct the first QC-valid date of each calendar quarter plus its next session. Display only availability, sizing/numerical stability, state reproduction errors and accounting residuals; no aggregate development profitability.

Run the full pytest suite. Commit and push the protocol, evaluator, tests and executed development notebook before opening 2025. Record that full evaluator hash in the 2025 notebook and an exclusive local run manifest. An existing manifest prevents automatic retry. A technical failure stops Phase B without editing or rerunning the evaluator. Expected missing/QC outcomes are coverage observations, not technical retries.

Phase B runs once. Save direct-factor and synthetic outputs under gitignored `data/processed/surface_factor_2025/` before reading the old fixed-contract audit for comparison. Public notebook outputs show aggregate diagnostics only. Report direct full/delayed/missed outcomes, ideal option/hedge/total P&L, all offsets, cumulative paths, missingness, and the fixed-contract midpoint and bid/ask totals with clearly different units/coverage. Update progress and commit the new result separately.

No forecast refit, threshold, VIX rule, magnitude sizing, tenor/delta/horizon/rebalance search, additional controls, cost assumptions, strategy tuning or replacement headline. The statistical and economic conclusions remain separate. Neither a positive factor outcome nor a model-mid synthetic outcome establishes tradability.
