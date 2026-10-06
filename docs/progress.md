# Project progress

This is a chronological research record; earlier "next step" and snapshot statements describe their stage of the project. For the completed v1 outcome, see the forecasting sections below and the README. The frozen protocol remains in `docs/forecasts.md`.

## `src/pricing.py`

### Implemented

- `black_scholes_call_price(...)` prices a European call using the Black formula.
- `black_scholes_put_price(...)` prices the corresponding European put.
- `black_scholes_greeks(...)` adds signed unadjusted forward delta, carry-adjusted SPX spot delta, and vega using the same Black inputs; existing price functions are unchanged.
- Inputs are converted to finite, strictly positive NumPy values.

### Mathematical logic

For a forward price $F$, strike $K$, discount factor $D$, time to expiry $\tau = T-t$, and volatility $\sigma$:

$$
d_1 = \frac{\ln(F/K) + \frac{1}{2}\sigma^2\tau}{\sigma\sqrt{\tau}},
\qquad
d_2 = d_1 - \sigma\sqrt{\tau}.
$$

The call and put prices are:

$$
C = D\left(FN(d_1) - KN(d_2)\right),
\qquad
P = D\left(KN(-d_2) - FN(-d_1)\right).
$$

### Tests

`tests/test_pricing.py` covers call-put parity, call-price monotonicity with respect to strike, and rejection of non-positive or non-finite inputs. `tests/test_trade_selection.py` checks the new Greeks against pricing finite differences and the existing RR25 forward-delta convention.

## `src/forward.py`

### Implemented

- `infer_forward_and_discount_factor(...)` fits call-minus-put values against strikes to infer the forward price and discount factor.
- Inputs must be one-dimensional, finite, aligned arrays with at least two distinct positive strikes.
- Call and put prices must be non-negative. The fitted discount factor and forward must be finite and strictly positive; discount factors above one are allowed.

### Mathematical logic

For matched European call and put prices under forward pricing:

$$
C - P = D(F - K) = DF - DK.
$$

Therefore, fitting $C-P$ as a linear function of strike gives a slope of $-D$ and an intercept of $DF$. The implementation recovers:

$$
D = -\text{slope},
\qquad
F = \frac{\text{intercept}}{D}.
$$

### Tests

`tests/test_forward.py` covers synthetic recovery (including discount factors above one), zero prices, invalid input lengths and strikes, non-finite or negative prices, and invalid fitted results.

## `src/forward_inference.py`

### Implemented

- `infer_expiry_forwards(...)` excludes expiry-day rows and applies the forward estimator separately to each `(security_id, quote_date, expiry_date)` group.
- It records the inferred forward, discount factor, number of strikes, and parity RMSE for each expiry.
- The script entry point reads the matched CSV and saves the expiry-level estimates to `data/processed/spx_forward_estimates_2025-08-29.csv`.

### Mathematical logic

For each expiry, the estimator fits:

$$
C_K-P_K=D(F-K)=DF-DK.
$$

The slope of the fitted line is $-D$ and the intercept is $DF$, so:

$$
D=-\text{slope},
\qquad
F=\frac{\text{intercept}}{D}.
$$

The parity RMSE is calculated from the midpoint residuals:

$$
\varepsilon_K=(C_{K,\mathrm{mid}}-P_{K,\mathrm{mid}})-D(F-K),
\qquad
\mathrm{RMSE}=\sqrt{\frac{1}{n}\sum_K\varepsilon_K^2}.
$$

### Validation

The saved estimates reproduce the results from `infer_forward_and_discount_factor(...)` to floating-point precision. Expiry-day rows are excluded because their time to expiry is zero.

### Tests

`tests/test_forward_inference.py` covers synthetic expiry-level recovery, exclusion of expiry-day rows, and CSV output from the script entry point.

## `notebooks/forward_inference_diagnostics.ipynb`

### Implemented

The notebook compares the baseline OLS estimates with a spread-weighted fit and diagnoses quote quality without applying production filters. It includes:

- observed $C-P$ against strike with the fitted parity line;
- parity residuals against strike and combined spread;
- exploratory zero-bid and widest-5%-spread buckets;
- a spread-weighted fit using objective weights proportional to $1/\text{combined spread}^2$;
- normalized residuals $R_K$ against strike, with a threshold at $R_K=1$ and a vertical line at the inferred forward;
- a per-expiry feasibility check for one common $F,D$ across all executable quote intervals.

### Mathematical logic

The executable parity interval for a matched call-put pair is:

$$
[C_{\mathrm{bid}}-P_{\mathrm{ask}},\quad C_{\mathrm{ask}}-P_{\mathrm{bid}}].
$$

The notebook tests whether a common pair $(F,D)$ satisfies this interval for every strike in an expiry:

$$
C_{\mathrm{bid}}-P_{\mathrm{ask}}
\leq D(F-K)
\leq C_{\mathrm{ask}}-P_{\mathrm{bid}}.
$$

With $A=DF$, this becomes a linear feasibility problem in $(A,D)$ because $D(F-K)=A-DK$. The check is separate from the OLS estimator and does not select a unique preferred pair.

### Current result

For the current SPX sample, all 38 positive-DTE expiries admit a feasible common $F,D$ within the executable intervals. The OLS diagnostics contain 1,006 individual normalized residual breaches, so those breaches do not prevent an expiry-level common fit.

## `src/options_chain.py`

### Implemented

- `calculate_mid_price(bid, ask)` calculates the midpoint between bid and ask prices.
- `calculate_time_to_expiry(quote_date, expiry_date)` calculates time to expiry in years using the ACT/365 convention.
- Both functions support scalar values and NumPy/pandas-friendly inputs where natural.

### Mathematical logic

The mid price is the average of the bid and ask:

$$
M = \frac{\mathrm{bid} + \mathrm{ask}}{2}.
$$

Using ACT/365, time to expiry is the actual number of calendar days between the quote and expiry dates divided by 365:

$$
\tau = \frac{\mathrm{expiry\ date} - \mathrm{quote\ date}}{365}.
$$

### Tests

`tests/test_options_chain.py` covers scalar and array mid-price calculations, plus scalar and pandas Series time-to-expiry calculations.

## `src/implied_vol.py`

### Implemented

- `implied_volatility_call(...)` solves for call implied volatility using `scipy.optimize.brentq`.
- `implied_volatility_put(...)` solves for put implied volatility using the same approach.
- Both solvers validate finite market prices against Black price bounds and return zero volatility at discounted intrinsic value.
- Each solver starts with an upper volatility of $5$, doubles it while the relevant Black option price remains below the market price, and stops at the current safety cap of $20$.

### Mathematical logic

For positive forward, strike, discount factor, and time to expiry, accepted market prices satisfy:

$$
D\max(F-K,0) \leq C_{\mathrm{market}} < DF,
\qquad
D\max(K-F,0) \leq P_{\mathrm{market}} < DK.
$$

Equality at the lower price bound returns $\sigma=0$ directly. Equality at the upper price bound is rejected because no finite volatility reaches it. Model inputs are validated even when returning zero volatility.

For calls, the solver defines:

$$
f(\sigma)
=
C_{\mathrm{BS}}(\sigma)
-
C_{\mathrm{market}}.
$$

For puts, the equivalent objective is:

$$
f(\sigma)
=
P_{\mathrm{BS}}(\sigma)
-
P_{\mathrm{market}}.
$$

The relevant Black option price is strictly increasing in volatility, so
$f(\sigma)$ is also increasing. Each solver begins with the bracket

$$
\sigma_{\mathrm{lower}} = 10^{-8},
\qquad
\sigma_{\mathrm{upper}} = 5.
$$

If $f(\sigma_{\mathrm{upper}}) < 0$, the model price is still below the
market price, so the required implied volatility must be higher. The upper
bound is therefore doubled until $f(\sigma_{\mathrm{upper}}) \geq 0$ or the
safety cap of $20$ is reached.

Failure to bracket by $20$ raises an error. Positive implied volatilities below the numerical lower bound of $10^{-8}$ are outside the supported search interval.

Once the objective has opposite signs at the two bounds, Brent's method
solves

$$
f(\sigma)=0.
$$

### Tests

`tests/test_implied_vol.py` covers call and put recovery at $0.25$, $8$, and the cap of $20$; rejection of valid prices generated at $\sigma=30$ with $\tau=0.01$; invalid market prices; zero-volatility returns at intrinsic value; and model input validation.

## `src/iv_panel.py`

### Implemented

- `build_iv_panel(...)` keeps positive-DTE cleaned option rows and joins their expiry-level forward and discount factor.
- Joined `forward`, `discount_factor`, `strike`, and `time_to_expiry` values must be finite and strictly positive before IV inversion.
- It computes log-moneyness, $k=\log(K/F)$, and midpoint implied volatility using the existing call and put solvers.
- Vendor IV is retained as `vendor_iv`; calculated midpoint IV is stored as `mid_iv`.
- `is_otm` identifies puts with $K<F$ and calls with $K>F$. `use_for_surface` is true only when the observation is OTM and `mid_iv` is valid.
- Missing expiry-level forward data raises an error instead of dropping rows silently. A midpoint outside the admissible Black bounds produces `mid_iv = NaN` for that observation.
- The script entry point saves the panel to `data/processed/spx_iv_panel_2025-08-29.csv`.

### Mathematical logic

For each option, log-moneyness is:

$$
k=\log\left(\frac{K}{F}\right).
$$

The calculated midpoint IV is the volatility that solves the relevant Black pricing equation using the joined expiry-level $F$ and $D$. OTM surface eligibility follows:

$$
\text{put if }K<F,
\qquad
\text{call if }K>F.
$$

An exact $K=F$ observation is not marked OTM.

### Tests

`tests/test_iv_panel.py` covers synthetic call and put IV recovery, log-moneyness, invalid midpoint handling, and missing forward-data rejection.

## `notebooks/iv_surface_diagnostics.ipynb`

### Implemented

- Reads the local IV panel and summarises inversion failures by option type, DTE, and log-moneyness.
- Compares midpoint and vendor IVs for both the available comparison sample and the OTM surface sample.
- Plots raw OTM smiles for expiries nearest 7, 30, 90, and 365 DTE, without fitting or smoothing.
- Summarises quote counts, zero bids, midpoint prices, relative spreads, and IVs by moneyness/DTE bucket.
- Creates a notebook-only `use_for_fit` flag, redraws the smiles, and compares expiry-level coverage with the previous spread screen and a 0.25 midpoint-floor alternative. The original rows and saved CSV are unchanged.

### Mathematical logic

Vendor comparisons use $\Delta\sigma=\sigma_{\mathrm{mid}}-\sigma_{\mathrm{vendor}}$. With midpoint $M=(\mathrm{bid}+\mathrm{ask})/2$, the provisional fitting screen requires `use_for_surface`, finite bid, ask, midpoint, midpoint IV, and relative spread, plus:

$$
M>0,\qquad 0<\mathrm{bid}<\mathrm{ask},\qquad
0<\frac{\mathrm{ask}-\mathrm{bid}}{M}\leq 0.50.
$$

Locked (`bid == ask`) and crossed quotes are excluded from fitting; a zero recorded spread is not treated as evidence of precise IV. No premium floor, IV-level cap, volume, or open-interest filter is applied. These are provisional sample-selection rules, not a proof that excluded quotes are erroneous or the retained sample is arbitrage-free. Current coverage is recorded in [Data](data.md#notebook-fitting-sample).

### Tests

The notebook has been executed end-to-end. Separate, one-off checks cover the spread boundary, locked/crossed and zero-bid quotes, non-finite inputs, retention of inexpensive and high-IV observations, and preservation of the original panel fields. These checks are not part of the committed pytest suite. The current sample retains calls and puts for every positive-DTE expiry.

Fitting stability across screening choices and additional quote dates remains untested. The exploratory SSVI calibration below now uses this screening rule.

## `notebooks/single_expiry_SSVI_slice.ipynb`

### Implemented

- Applies the existing notebook fitting screen and selects the 2025-11-28 expiry from the filtered OTM IV sample.
- Converts midpoint implied volatility to total variance and plots it against log-moneyness.
- Fixes $\theta_{\mathrm{ATM}}$ from the filtered observation nearest $k=0$, then fits $\rho$ and $\varphi$ by bounded nonlinear least squares.
- Reports the fitted $\rho$ and $\varphi$, fixed ATM total variance, and total-variance RMSE.
- Shows the separate effects of $\rho$, $\theta$, and $\varphi$ using the fitted slice as the baseline.
- Provides manual sliders initialized at the fitted $\rho$ and $\varphi$ and fixed $\theta_{\mathrm{ATM}}$. Slider changes replace one managed plot instead of appending figures.

### Mathematical logic

For time to expiry $\tau$, implied volatility is represented as total variance:

$$
w(k)=\sigma_{\mathrm{IV}}(k)^2\tau,
\qquad
k=\log(K/F).
$$

The exploratory slice uses:

$$
w(k)=\frac{\theta}{2}
\left(
1+\rho\varphi k
+\sqrt{(\varphi k+\rho)^2+1-\rho^2}
\right).
$$

Here $\theta$ controls the variance level, $\rho$ controls the direction and strength of skew, and $\varphi$ controls curvature and wing steepness. ATM total variance is fixed as:

$$
\theta_{\mathrm{ATM}}=\sigma_{\mathrm{ATM}}^2\tau.
$$

Using the filtered observation nearest $k=0$ as the ATM observation, the notebook fits only $\rho$ and $\varphi$:

$$
\min_{\rho,\varphi}
\sum_i
\left[
w_{\mathrm{SSVI}}(k_i;\theta_{\mathrm{ATM}},\rho,\varphi)
-w_i^{\mathrm{market}}
\right]^2.
$$

### Validation and limitations

The notebook has been run, and its cell outputs are saved in the notebook, including the filtered-sample summary, fitted parameters, and plots. The filtered sample uses the same provisional `use_for_fit` rules documented for the IV diagnostics notebook. The nearest-to-forward quote is currently used as the ATM proxy rather than interpolating $w(0)$. The fitted values are exploratory defaults for understanding parameter effects. The fit is not yet used by production code and has not been checked for static arbitrage or extended jointly across expiries.

## `src/ssvi.py`

### Implemented

- `ssvi_phi(...)` and `ssvi_total_variance(...)` evaluate the global SSVI parameterisation.
- `estimate_atm_theta(...)` takes ATM total variance from the filtered observation nearest $k=0$ for one expiry.
- `fit_ssvi_surface(...)` applies the existing OTM quote screen by default to one quote date and security, estimates ATM variance for each expiry, and fits shared $\rho$, $\eta$, and $\gamma$ by least squares. `apply_quote_screen=False` retains every valid OTM IV for comparison.
- The result includes the parameters, total-variance RMSE, a table of expiry ATM variances, and the retained observations with market variance, fitted variance, residuals, and fitted IV.

### Mathematical logic

For each expiry $j$, $\theta_j$ is held fixed during the global fit. With

$$
\varphi(\theta)=\eta\theta^{-\gamma},
$$

the surface is

$$
w_{\mathrm{SSVI}}(k,\theta)
=\frac{\theta}{2}
\left[1+\rho\varphi(\theta)k
+\sqrt{(\varphi(\theta)k+\rho)^2+1-\rho^2}\right].
$$

The fit minimizes $\sum_i[w_{\mathrm{SSVI}}(k_i,\theta_{j(i)})-w_i^{\mathrm{market}}]^2$ over the shared $\rho$, $\eta$, and $\gamma$.

### Tests

`tests/test_ssvi.py` checks recovery of known shared parameters and expiry ATM variances from synthetic smiles, including exclusion of a zero-bid quote.

## `src/raw_svi.py`

### Implemented

- `raw_svi_total_variance(...)` evaluates the five-parameter Raw SVI slice.
- `fit_raw_svi_surface(...)` applies the same OTM quote screen as the SSVI fit and calibrates each expiry independently. It returns per-expiry parameters, maturity, and residuals. `enforce_arbitrage=True` accepts a feasible unconstrained fit or falls back to multiple constrained starts, retaining the best sampled-arbitrage-feasible result.
- `raw_svi_butterfly_diagnostic(...)` reports the exact minimum total variance, sampled Durrleman $g(k)$ minimum, and both wing conditions for a fitted slice.

### Mathematical logic

Each expiry $j$ has its own parameters $(a_j,b_j,\rho_j,m_j,\sigma_j)$ and total variance

$$
w_j(k)=a_j+b_j\left[\rho_j(k-m_j)+\sqrt{(k-m_j)^2+\sigma_j^2}\right].
$$

By default, the parameters are fitted by least squares to that expiry's observed total variance. With `enforce_arbitrage=True`, calibration imposes $b\geq0$, $|\rho|<1$, $\sigma>0$, nonnegative minimum variance, both wing bounds,

$$
b(1+\rho)<2,\qquad b(1-\rho)<2,
$$

and $g(k)\geq0$ on an adaptively refined grid covering the quoted range and $[-5,5]$. Unlike SSVI, Raw SVI does not share parameters across expiries.

### Limitations

The default independent fits are unconstrained with respect to arbitrage. The optional constrained mode samples $g(k)$ on a finite grid, then rechecks a denser grid; it is not a proof over all $k$ and does not test calendar-spread arbitrage. Each expiry needs at least five screened observations. Some calibrated parameters approach their numerical bounds.

On the screened 2025-08-29 panel, holding out every fifth strike by expiry gives equal-expiry mean test RMSEs of $0.000173$ for Raw SVI and $0.002648$ for a single-expiry SSVI-shaped fit. Four starts on four representative expiries converge to curves within $2.6\times10^{-7}$ total variance of the default-start curves. Of 38 unconstrained slices, 36 pass the sampled $g(k)$ check inside the quoted range, but only 10 pass it on the wide $[-5,5]$ grid; 26 fail only in extrapolation and 2 fail inside the quoted range. The left-wing condition passes all 38, while the right-wing condition passes 20. The arbitrage-aware fits converge for all 38 expiries using up to five starts; equal-expiry mean RMSE rises from $0.000127$ to $0.000221$ versus the current fit. The finite-grid checks do not prove arbitrage freedom.

### Tests

`tests/test_raw_svi.py` checks the total-variance formula, synthetic per-expiry and multi-start arbitrage-aware fits, the minimum five-observation requirement, both wing bounds, and quote-range versus extrapolation diagnostics.

## `src/vol_surface.py`

### Implemented

- `build_total_variance_grid(...)` evaluates each fitted Raw SVI slice only where $k$ is inside that expiry's observed range, retaining unsupported cells as missing values.
- `enforce_calendar_monotonicity(...)` applies equal-weight isotonic regression at each $k$ using only the maturities supported there.
- `evaluate_surface(...)` uses the fixed surrounding expiries for a target maturity and only their shared sampled support. Exact fitted maturities use their own support. It rejects unsupported requests rather than switching to more distant expiries as $k$ changes.
- `count_calendar_crossings(...)` compares consecutive supported maturities at each $k$, including across missing slices.
- `check_surface_quality(...)` returns `use_for_skew`, a failure reason, checked moneyness bounds, sampled call-price diagnostics, and min/max call slopes with their $k$ locations. It checks full shared support by default; an optional `k_range` must be entirely supported.

### Mathematical logic

At every grid point $k_\ell$, let $J(k_\ell)$ be the expiries whose observed strike range contains that point. Only their Raw SVI values are projected to the closest nondecreasing sequence in squared-error distance:

$$
\tilde w_{j+1}(k_\ell)\geq \tilde w_j(k_\ell),\qquad j\in J(k_\ell).
$$

Between maturities, total variance is interpolated linearly and converted back using $\sigma_{\mathrm{IV}}=\sqrt{w/\tau}$. The repair does not alter or refit the Raw SVI slices. Strike interpolation remains linear in total variance and is not butterfly-safe: convexity violations occur between grid points and around some knots. The notebook now checks normalized Black call prices from the actual evaluator for slope bounds and convexity, including fitted expiries and 30D/60D/90D targets. A change to the surface construction is deferred; the diagnostic does not repair these violations.

The quality gate requires sampled normalized call-price slopes between $-1$ and $0$ and nondecreasing slopes in strike, with tolerance $10^{-8}$. Unsupported maturities or requested moneyness ranges, invalid variances, and failed price checks receive `use_for_skew=False`. The Raw SVI notebook records every 30D/60D/90D target by quote date, retains evaluable failed smiles for inspection, and provides `usable_target_smiles` containing only passing tenors. No skew feature has been defined yet. Historical failure rates and their concentration in volatile periods remain to be assessed; a pass is not a proof of arbitrage freedom.

On the screened 2025-08-29 panel, support-aware projection on the common $k\in[-0.5,0.25]$ grid removes all 20 sampled crossings where adjacent expiries share quoted support. Counting consecutive supported maturities across missing slices gives 22 crossings before repair and zero afterward. The maximum supported $|\Delta w|$ is $0.000030$ (median nonzero adjustment $0.000007$), and the maximum IV adjustment is $0.26$ percentage points. The former finite-difference $g(k)$ check at grid nodes missed violations in the evaluated surface. The call-price check in the Raw SVI notebook flags 15 of 41 evaluated smiles; the later skew diagnostics apply the checks at metric-specific ranges, as described below. These sampled checks are diagnostics, not proofs of arbitrage freedom.

### Tests

`tests/test_vol_surface.py` covers maturity ordering, support-aware isotonic projection, crossing counts across missing slices, variance-first interpolation, IV conversion, fixed expiry brackets, preservation of exact-expiry grid values, and rejection of unsupported requests or extrapolation. Quality-gate tests cover a passing flat smile, a butterfly failure, unavailable maturities, missing support, requested moneyness ranges, and unchanged input values.

## `src/skew_metrics.py`

### Implemented

- `calculate_skew_metrics(...)` extracts 30D/60D/90D repaired smiles and calculates ATM IV, a centered ATM downside-skew slope, and a 25-delta downside risk reversal.
- It reports independent `atm_valid`, `slope_valid`, and `rr25_valid` flags and reasons. Full-smile `surface_valid` and `convexity_valid` are reported separately; they do not suppress otherwise valid metric values.
- 25-delta strikes use unadjusted Black forward deltas and require a unique OTM root on each wing. ATM slope uses the centered difference at $k=\pm0.01$, reported as volatility points per 1% change in $k$.

The downside-skew slope is $-[\sigma(0.01)-\sigma(-0.01)]/0.02$. The downside 25-delta risk reversal is $\sigma_{25\Delta\,put}-\sigma_{25\Delta\,call}$. A positive value for either indicates higher put-side volatility.

### Tests and diagnostics

`tests/test_skew_metrics.py` checks flat-smile metrics, positive downside-skew signs, and that an unsupported wing does not invalidate supported ATM IV or slope. `notebooks/skew_metrics_diagnostics.ipynb` plots the target smiles with 25-delta markers and reports metrics and quality flags. On the 2025-08-29 sample, all three ATM, local-slope, and 25-delta metrics pass; the full-support convexity check fails for 30D and 90D. The refined call-slope diagnostic reports in-range extrema for investigation.

## `notebooks/ssvi_surface_diagnostics.ipynb`

### Implemented

- Loads the 2025-08-29 IV panel and calls `fit_ssvi_surface(...)`.
- Shows fitted parameters and expiry ATM variances, overlays observed and fitted total variance for representative maturities, plots total-variance residuals, and compares observed IV with fitted expiry slices in 3D. A Plotly mesh of the fitted slices can be rotated in the notebook.
- Repeats the calibration with all valid OTM IVs before the bid/spread screen, compares the parameters and representative slices, and plots both observed and fitted IV at every unscreened quote location with their fitted surface.
- Fits Raw SVI separately at each screened expiry and overlays its representative total-variance slices with the shared SSVI fit.
- Compares interleaved held-out-strike train/test RMSE by expiry, checks multi-start calibration stability, distinguishes quoted-range from extrapolation butterfly violations, and compares ordinary and arbitrage-aware Raw SVI fit errors. The held-out SSVI-shaped comparator is calibrated separately for each expiry, not with the global SSVI fit.
- Projects the arbitrage-aware Raw SVI slices onto a support-aware calendar-monotone total-variance grid, compares pre/post calendar crossings and adjustment sizes, checks repaired-slice butterfly behavior, and demonstrates supported 30D/60D/90D interpolation. The 3D repaired mesh is masked outside quote support.

### Limitations

The notebook has been run and includes a saved, rotatable Plotly surface output. This is an exploratory fit for one quote date. It does not establish absence of static arbitrage or stability across quote dates and screening choices.

## `src/data_cleaning.py`

### Implemented

- `clean_option_data(...)` converts OptionMetrics rows to the project schema.
- The script entry point filters `am_settlement == 0` before cleaning and saves the processed CSV.
- Dates, strikes, option types, quote metrics, and expiry-day flags are prepared as described in [Data](data.md).

### Mathematical logic

Strike is divided by $1000$. Spread is ask minus bid; relative spread is spread divided by midpoint, with zero midpoints producing `NaN`. Remaining calendar days are divided by $365$, and `is_expiry_day` identifies zero remaining days.

### Tests

The current SPX file has been checked through cleaning and matching. There are no dedicated automated cleaning tests yet.

## `src/option_matching.py`

### Implemented

- `match_calls_and_puts(...)` inner-joins calls and puts on `security_id`, `quote_date`, `expiry_date`, and `strike`.
- Missing matching keys and duplicate keys within either side raise errors; unmatched rows are excluded.
- Contract identifiers, quotes, volume, and open interest are retained separately for calls and puts. Expiry times and `is_expiry_day` are retained from the call.
- The script writes the matched CSV described in [Data](data.md).

### Mathematical logic

Matching aligns $C(K)$ and $P(K)$ for the same security, quote date, and expiry. Subsequent forward inference must be performed separately for each security, quote date, and expiry.

### Tests

`tests/test_option_matching.py` covers output schema, value preservation, complete pairs, exclusion of mismatched keys, duplicate and missing key rejection, preservation of the expiry-day flag, and unchanged input data. Its small in-memory fixture is test-only.

## `src/historical_pipeline.py`

### Implemented

- Reads annual raw CSVs in chunks, retaining complete dates across chunk boundaries.
- Validates SPX constants, selects PM rows, drops crossed quotes, then calls the existing
  cleaning, matching, forward, IV, arbitrage-aware Raw SVI, calendar-repair, and skew APIs.
- Saves three metric rows and one diagnostics row per date. Date and expiry failures
  are recorded. Target tenors use nearest accepted fits with a maximum 30-calendar-day
  gap; exact accepted maturities have zero gap. Records bracket dates, gap and skipped
  expiry count. Both endpoints must still support each metric's required strikes.
- Keeps large intermediates in memory and checkpoints the two small CSV outputs after
  every date. Supports resume, failed-date retries, and selected date ranges.
- Uses 14-180 DTE expiries for SVI and calendar repair. Rejects fits whose ATM IV differs
  from the interpolated screened-quote benchmark by more than the larger of 0.03 and 20%.
- Records raw/repaired target ATM IVs and invalidates all three research metrics for a
  tenor if calendar repair changes its ATM IV by more than 0.05. Pre-guard values remain
  in diagnostics. Rejected expiries are excluded from brackets. Older checkpoints are invalidated.

### Mathematical logic

Reuses the existing definitions and default settings throughout. ATM downside skew slope
is the primary target, 25-delta downside RR is the trading-oriented robustness measure,
and ATM IV is a state/control variable. Metric-specific validity remains independent of
full-smile QC. See [Historical processing](historical_processing.md) for commands,
failure policy, output fields, and checkpoint limitations.

### Tests

`tests/test_historical_pipeline.py` covers chunk boundaries, input ordering, SPX constants,
early quote filtering, a synthetic run through the existing models, failed SVI expiry
handling, continuation after a date failure, resume/retry, and interrupted checkpoints.
It also checks maturity bounds, ATM benchmarks/tolerances, skipping rejected expiries,
the inclusive 30-day bracket limit, exact maturities, support checks, unchanged metrics
on accepted brackets, tenor-specific repair guards, and checkpoint versioning.
Seven targeted real dates, including the two prior outliers and 2025-08-29, retain
all 21 tenor rows with valid metrics and numerical agreement within $10^{-12}$ with
the preceding guarded results. Coverage recovery across failed fits is tested synthetically.

## `src/time_series.py`

### Implemented

- `load_skew_metrics(...)` loads the long-form historical metrics CSV with parsed quote dates.
- `build_daily_time_series(metrics, spx_prices)` aligns metric rows to the SPX
  security-price trading dates before calculating changes. It masks ATM skew slope,
  downside 25-delta RR, and ATM IV using their respective validity flags; nonfinite
  values become missing. It adds one-session changes and 30D–60D / 60D–90D
  differences for each metric. Missing values are not forward-filled.
- `add_forward_targets(daily, spx_prices)` adds fixed 1-, 5-, and 10-session
  endpoint changes:

  $$
  y^{\mathrm{spread}}_{t,h}=(S^{30}_{t+h}-S^{60}_{t+h})-(S^{30}_t-S^{60}_t),
  \qquad
  y^{30}_{t,h}=S^{30}_{t+h}-S^{30}_t,
  $$
  $$
  y^{\mathrm{RRspread}}_{t,h}=(RR^{25,30}_{t+h}-RR^{25,60}_{t+h})
  -(RR^{25,30}_t-RR^{25,60}_t).
  $$

  SPX trading dates supply the session sequence. Missing metric dates are inserted
  before calculating changes and endpoints, so an absent date cannot bridge a change
  or shorten a horizon. Only valid
  start/end values are required; intermediate missing observations do not invalidate an
  endpoint-to-endpoint target. The final $h$ dates have missing targets. No values are filled.

The two headline 5-session targets are the change in 30D skew (statistical) and the
change in the 30D–60D skew spread (relative value). The 5-session 30D–60D RR25 spread
change is a tradability robustness check; 1- and 10-session versions are robustness
horizons. See [Forecasting protocol](forecasts.md) for the frozen experiment.

### Tests

`tests/test_time_series.py` covers QC masking and long-to-wide construction, one-session
changes, term structure differences, missing observations and whole missing sessions, duplicate keys, input
preservation, CSV loading, target arithmetic at all three horizons, missing endpoints,
missing intermediate values, final-horizon rows, and absent metric dates on the SPX
session calendar.

## `src/market_state.py`

### Implemented

- `load_market_state(spx_path, vix_path)` reads separate per-security price CSVs.
  Inputs require `date` and `close`; an optional OptionMetrics `return` is retained
  as `spx_vendor_return` for comparison only. The SPX extract must have secid 108105
  and ticker SPX; the VIX extract must have ticker VIX. A VVIX extract is rejected.
- `spx_trading_dates(spx_prices)` provides the SPX session calendar used by the
  daily time-series functions.
- `build_market_state(...)` sorts and aligns prices by the union of SPX and VIX dates.
  It calculates `spx_return` from adjacent SPX closes as
  $r_t=\log(S_t/S_{t-1})$, then adds $|r_t|$ and $r_t^2$. The optional vendor return
  does not enter any calculated feature.
- `rv_5` and `rv_20` are annualized rolling realized volatility:

  $$
  RV_{h,t}=\sqrt{\frac{252}{h}\sum_{j=0}^{h-1}r_{t-j}^2},
  \qquad h\in\{5,20\}.
  $$

  Each window requires all of its returns. A missing SPX date present in the VIX extract
  remains an empty SPX row, so returns and rolling windows do not bridge it. VIX closes
  are joined on the exact date and never forward-filled.
- `merge_market_state(daily_surface, market_state)` left-joins on date while retaining
  the surface date rows. Thus SPX dates used only for pre-sample RV warm-up do not appear
  in the merged surface dataset.

### Tests

`tests/test_market_state.py` checks close-based log-return arithmetic, RV5/RV20 windows,
missing-SPX-date handling, VIX ticker identity, exact-date SPX/VIX alignment without VIX filling, CSV loading,
and preservation of surface dates when warm-up-only rows are merged.

## `src/forecasting.py`

### Implemented

- Splits development (2023–2024) and confirmation (2025) dates without altering source rows.
- Builds the development dataset on every SPX session, with QC-masked states and five-session targets; labels whose endpoint falls in 2025 remain unavailable.
- Calculates the two headline states' z-scores from 60 prior valid observations and produces expanding, five-session-purged M0 persistence, M1 historical-mean, and M2 mean-reversion forecasts.
- Adds the frozen M3 own-dynamics, M4 surface-state, M5a realised-market, and M5b VIX predictor blocks. The spread's one-session change is calculated on the full SPX session calendar; market state uses the existing exact-date SPX/VIX join. Each OLS fit uses only complete-case, matured training rows, and a missing current predictor causes that model to abstain. Coefficients are retained by origin.
- M6 elastic net and M7 shallow histogram boosting use the unchanged M5b features. Each development origin selects from the frozen grids using two latest 30-session chronological validation blocks, with expanding fits, five-session label-maturity purging, and at least 100 complete labels before the first block. Elastic-net scaling is fitted separately inside each training fold. Candidate RMSE, selection, and training counts are retained by origin.
- Scores M0–M7 on one strict common set of realised development dates, including $R^2$ versus persistence and M2 where applicable.
- The development selections are frozen: M6 elastic net for `y_skew_30_5d` and M2 mean reversion for `y_skew_spread_5d`. `fit_locked_models(...)` selects M6 hyperparameters with the final two purged development-only validation blocks, then fits M6 and both M2 benchmarks on labels matured by 2024-12-31. The scaler, coefficients, and selected hyperparameters are retained for static confirmation forecasts.
- `build_locked_feature_panel(...)` constructs causal, target-free features on the complete SPX session calendar. Past-only z-score moments continue updating from observed surface states during 2025. `forecast_locked_holdout(...)` uses the frozen fits without accepting outcome columns; separate `build_locked_outcomes(...)`, `score_locked_holdout(...)`, and `cumulative_locked_gains(...)` prepare later evaluation. No recursive refitting is implemented.
- The 2025 period is a locked temporal confirmation sample, not a perfectly pristine unseen holdout: earlier full-sample descriptive EDA included it. The frozen models were evaluated once in `notebooks/forecast_locked_2025.ipynb`; no recursive refitting or post-result retuning was performed.

### Tests

`tests/test_forecasting.py` checks date splitting, past-only z-scores, the 2024/2025 target boundary, missing sessions, matured-label purging, exact M3–M7 feature sets/settings, market-date alignment, missing-predictor abstention, future-value isolation, coefficient recovery, purged inner folds, training-only elastic-net scaling, deterministic tie-breaking, insufficient-history abstention, and strict common-date score arithmetic. Synthetic locked-confirmation tests also check frozen end-development fits, continued causal 2025 z-scores, no target dependence of predictions, static parameters, exact session horizons, missing-data abstention, and both benchmark-relative scoring formulas.

## `notebooks/forecast_baselines.ipynb` and `notebooks/forecast_linear_models.ipynb`

### Implemented

The executed baseline notebook preserves the original M0–M2 development checkpoint. The separate linear-model notebook evaluates M0–M5b on strict common 2024 forecast dates, shows model coverage, cumulative gains versus M0/M2, five non-overlapping offsets, and end-development coefficient/sign-stability diagnostics. It does not evaluate 2025.

## `notebooks/forecast_ml_models.ipynb`

### Implemented

The executed development notebook retains the previous M0–M5b checkpoint and separately compares M0–M7 on 238 strict common 2024 origins per target. It reports coverage, cumulative and quarterly squared-error gains, all five non-overlapping offsets, and M6/M7 hyperparameter-selection frequencies. Per-origin candidate scores and selections are saved under gitignored `data/processed/forecast_ml_selection_dev_2023_2024.csv`. M6/M7 each issued 243 forecasts per target; seven origins lacked current required features. No 2025 forecast was made.

For 30D outright skew change, M2 RMSE/MAE are 0.0987/0.0752. M6 gives 0.0968/0.0742 and $R^2$ versus M2 of 0.0389. Its gain is positive in three of four quarters and four of five non-overlapping offsets. M7 gives RMSE 0.0964 but MAE 0.0754; most of its net gain is concentrated in Q3. Under the frozen rule, M6 was selected for locked confirmation. For the 30D–60D spread, M2 RMSE/MAE are 0.0524/0.0404; M6 gives 0.0545/0.0419 and M7 gives 0.0543/0.0417, so M2 was selected. These overlapping-target development comparisons are descriptive, not iid significance claims or trading results.

## `notebooks/forecast_locked_2025.ipynb`

### Frozen confirmation result

The notebook was executed once after the full test suite passed. It fits the selected models using only development labels matured by 2024-12-31, then makes static 2025 predictions before constructing outcomes. The final M6 choice from the frozen two-block validation is `alpha=0.01`, `l1_ratio=0.25`; final M2 coefficients are intercept 0.007266 and slope -0.040897 for 30D skew, and intercept 0.004940 and slope -0.027615 for the 30D–60D spread.

There are 165 SPX confirmation sessions (2025-01-02 through 2025-08-29) and 158 common scored dates per target (through 2025-08-22). On the 30D target, persistence, frozen M2, and selected M6 have RMSEs 0.1012, 0.0922, and 0.1081; M6 has $R^2$ of -0.3731 versus M2 and **does not confirm incremental value**. On the spread target, persistence and frozen M2 have RMSEs 0.0544 and 0.0492; M2 has $R^2$ of 0.1824 versus persistence and **does confirm improvement on that benchmark**. The notebook retains the complete MAE, correlation, directional-accuracy and $R^2$ tables and cumulative squared-error gain plots.

No missing values were filled: M6 issued 163 of 165 outright forecasts, the spread state was unavailable on one session, and the final five sessions lack within-sample five-session outcomes. These are overlapping-target descriptive results, not iid significance evidence or a trading backtest. No model was changed or rerun after viewing confirmation performance.

## `notebooks/forecast_robustness.ipynb`

### Post-confirmation robustness analysis

This executed notebook holds the frozen 5D headline result fixed and tests only static M2 mean reversion. Each 1D/5D/10D outright or spread model uses the same 60-prior-valid-observation causal z-score; its intercept and slope are fitted once on labels matured in 2023–2024 and then frozen for 2025. The 5D M0/M2 scores reproduce the 158-date locked-confirmation checkpoint to reported precision. The 30D 5D comparison retains the original common-date mask without refitting M6. The RR25 spread uses the existing QC-masked downside convention. No alternative model, threshold, or z-score window was selected.

For outright 30D skew, M2-versus-persistence 2025 OOS $R^2$ is 0.037 at 1D (164 dates), 0.169 at 5D (158), and 0.288 at 10D (155). For the 30D–60D skew spread it is 0.106 (162), 0.182 (158), and 0.317 (153). All six development M2 slopes are negative. The 5D RR25-spread check is also positive, with 155 dates and OOS $R^2$ of 0.225. These are different horizons/outcomes and not a post-hoc replacement for the 5D headline.

M2 improves on M0 in all five fixed, original-session-index 5D offsets for both headline targets. The **development-fixed extreme-move threshold** (99th percentile of eligible development absolute changes) removes 5 outright and 1 spread confirmation observations; the remaining OOS $R^2$s are 0.255 and 0.204. Separately, the **confirmation top-1%-removed diagnostic** ranks each target's 158 realised confirmation changes by absolute size, breaks ties by earliest quote date, and removes exactly 2 observations. On the remaining 156 dates, outright M0/M2 RMSEs are 0.094375/0.087088, MAEs 0.070273/0.063124, and OOS $R^2$ 0.148469; spread RMSEs are 0.051507/0.046976, MAEs 0.040562/0.035778, and OOS $R^2$ 0.168198. This second, outcome-based trim is post-confirmation analysis, not an untouched test; neither trim changes the headline result. VIX-regime results qualify the breadth of the finding: below 20 (109 dates), OOS $R^2$ is only 0.005 outright and 0.023 spread; at or above 20 (49 dates), it is 0.254 and 0.296. These are unchanged predictions split descriptively, not regime-specific models.

For the overlapping 5D loss differential $e^2_{M0}-e^2_{M2}$, fixed-lag-4 Newey–West mean/SE/95% intervals are $0.001736/0.001451/[-0.001109,0.004580]$ outright and $0.000540/0.000457/[-0.000356,0.001437]$ spread. Both intervals include zero; the finite confirmation sample does not establish a precise positive mean loss gain. The saved notebook includes all score tables, offset/extreme/regime diagnostics, and cumulative M2-versus-M0 plots. The frozen headline finding and failed M6 confirmation remain unchanged; no trading backtest was run.

## `notebooks/skew_time_series_analysis.ipynb`

### Implemented

Uses `build_daily_time_series(...)` to plot skew levels, daily changes, and skew term
structure; summarizes level/change distributions; reports lag 1–10 autocorrelation and
paired counts; compares tenor correlations; checks contemporaneous 30D skew / ATM IV and
skew / RR25 relationships; and reports QC coverage by metric and tenor. It contains no
forecasting model. Its saved outputs cover all 667 quote dates from 2023-01-03 through
2025-08-29.

## Listed-option trading translation (mechanics only)

### Frozen protocol

`docs/trading_protocol.md` freezes the existing RR25-spread M2 signal, next-session entry/original fifth-session exit, seven-calendar-day expiry tolerance, signed forward 25-delta wings with 0.05 tolerance, equal wing vega/unit gross entry vega, observed bid/ask execution, fixed contract identities, one-session common-basket exit fallback, and previous-close SPX hedge accounting. It was written before inspecting any 2025 option selections or strategy P&L. Forecasting results were already known: this is a translation test, not a pristine new economic hypothesis. No forecasting specification or conclusion changes.

### Implemented

- `src/trade_selection.py` prepares PM quotes using existing cleaning, parity forward inference and IV inversion; selects the minimum-DTE-error expiry pair and closest eligible wings deterministically; constructs fixed fractional contract quantities with the 100 contract multiplier. It also fits the existing RR25 M2 once on development-matured labels and generates causal sign signals without accepting target outcomes.
- `src/trade_backtest.py` separates quote-only lifecycle checks from bid/ask and hedge accounting. It retains the same four contracts, liquidates the basket on one common date, earns each interval's hedge P&L from the previous EOD hedge, then sets the next hedge. Missing daily hedge inputs or SPX closes are explicit failures, never stale-value fills. Accounting has been run only on synthetic tests.
- `src/trading_feasibility.py` streams the explicit 2023 and 2024 annual files, retains the authoritative SPX session index and selected-contract availability flags, and never calls real-data P&L accounting. Boundary entries are reported separately. Position statistics use a canonical positive direction independently of forecast availability, not simulated development trades.
- Greeks distinguish RR25 forward delta from the carry-adjusted SPX hedge delta. Vega is per decimal volatility, quantities are fractional theoretical contracts, and the index hedge is frictionless with zero financing in v1; no fully executable futures/integer-order or trading-profitability claim is made.

### Tests

`tests/test_trade_selection.py`, `tests/test_trade_backtest.py` and `tests/test_trading_feasibility.py` cover expiry-pair minimisation/order/tolerances, delta conventions and distance limits, invalid quotes, future-information isolation, RR direction and unit gross vega, development-only static signal fitting, exact session offsets, all execution sides, fixed quantities/identities, basket exit fallback, previous-close hedge ordering/turnover, missing-mark failures, uncompressed missing sessions, and development-only feasibility guards. The feasibility tests forbid invoking the P&L function. No 2025 raw chain or actual strategy P&L has been evaluated.

### Development-only feasibility

The executed `notebooks/trading_feasibility.ipynb` checks all 502 SPX sessions from 2023-01-03 through 2024-12-31, streaming 10,659,814 raw rows (7,268,276 PM rows). Expiry pairs exist on 492 sessions (98.01%); all four wings on 481 (95.82%). Ten sessions lack an eligible pair; eleven fail a wing's 0.05 distance limit (ten 30D calls, one 60D call). No tolerance was relaxed. Pairwise median/95th-percentile absolute DTE errors are 0/1 days for 30D and 3/7 days for 60D. Across all nearest-wing candidates, median/95th-percentile delta distance is 0.002077/0.019276; failed candidates remain visible in the table.

The 1,924 selected options have median/95th-percentile quoted width of 0.40/0.90 SPX points and relative width of 1.395%/2.667%. The whole PM chain contains 377,894 zero-bid rows and two invalid/crossed rows; none is an executable entry. Candidate/held IV and forward estimation failures are zero. Every available basket has gross entry vega one and net entry vega zero to rounding. Canonical-positive-direction median absolute SPX hedge size is 0.000082 per unit gross vega; fractional quantities and zero-cost hedge assumptions remain explicit.

Six selected entry dates are lifecycle-boundary observations (no within-sample origin, or exit/fallback outside development). All 475 eligible baskets, comprising 1,900 fixed contract legs, have valid executable quotes through the scheduled exit and complete daily hedge inputs. All exit on schedule; fallback and unevaluable rates are zero. Examples are the first valid entry of each calendar quarter. This is adequate mechanical coverage under the frozen rules, not trading-performance evidence. The protocol and machinery are ready for a separately authorised locked 2025 trading run; **2025 option-chain selections and trading P&L have not been evaluated**, and no real-data strategy P&L was computed in development either.

## Test suite

The current test suite covers:

- Black call and put pricing.
- Put-call parity and pricing input validation.
- Synthetic forward and discount-factor recovery.
- Expiry-level forward inference, expiry-day exclusion, and saved CSV output.
- Call and put implied-volatility recovery and adaptive bracketing.
- Call and put implied-volatility safety-cap handling.
- IV-panel construction, midpoint IV recovery, surface eligibility, and missing-forward validation.
- Synthetic global SSVI parameter and expiry ATM-variance recovery.
- Call-put matching and key validation.
- QC-aware daily time-series construction and fixed 1/5/10-session forward targets.
- Leakage-safe development M0–M7 forecasts, purged inner CV, temporal boundaries, and common-date scoring.
- SPX/VIX market-state returns, rolling realized volatility, and date alignment.
- Deterministic listed-option selection, static RR25 signals, synthetic bid/ask and delta-hedge accounting, and development-only lifecycle coverage.

Run the suite with:

```bash
pytest
```

## Project configuration

- `pytest.ini` configures the project root on the pytest import path.
