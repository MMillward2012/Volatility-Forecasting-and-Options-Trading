# Project progress

## `src/pricing.py`

### Implemented

- `black_scholes_call_price(...)` prices a European call using the Black formula.
- `black_scholes_put_price(...)` prices the corresponding European put.
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

`tests/test_pricing.py` covers call-put parity, call-price monotonicity with respect to strike, and rejection of non-positive or non-finite inputs.

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
- `build_daily_time_series(...)` creates one sorted row per quote date. It masks ATM skew
  slope, downside 25-delta RR, and ATM IV using their respective validity flags; nonfinite
  values become missing. It adds adjacent-row daily changes and 30D–60D / 60D–90D
  differences for each metric. Missing values are not forward-filled.
- `add_forward_targets(daily, canonical_quote_dates)` adds fixed 1-, 5-, and 10-session
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

  The caller supplies the complete, sorted quote-date sequence. Missing dates are inserted
  before calculating endpoints, so an absent date cannot shorten a horizon. Only valid
  start/end values are required; intermediate missing observations do not invalidate an
  endpoint-to-endpoint target. The final $h$ dates have missing targets. No values are filled.

The primary target is the 5-session change in 30D–60D skew spread. The 5-session change in
30D skew is the secondary benchmark, and the 5-session 30D–60D RR25 spread change is the
tradability check. The 1- and 10-session versions are robustness horizons.

### Tests

`tests/test_time_series.py` covers QC masking and long-to-wide construction, adjacent-row
changes, term structure differences, missing observations, duplicate keys, input
preservation, CSV loading, target arithmetic at all three horizons, missing endpoints,
missing intermediate values, final-horizon rows, and absent trading dates in the supplied
canonical sequence.

## `notebooks/skew_time_series_analysis.ipynb`

### Implemented

Uses `build_daily_time_series(...)` to plot skew levels, daily changes, and skew term
structure; summarizes level/change distributions; reports lag 1–10 autocorrelation and
paired counts; compares tenor correlations; checks contemporaneous 30D skew / ATM IV and
skew / RR25 relationships; and reports QC coverage by metric and tenor. It contains no
forecasting model. The saved notebook outputs currently show a partial 73-date sample
through 2023-04-18; rerun the notebook after the historical pipeline completes to refresh
the results from the full CSV.

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

Run the suite with:

```bash
pytest
```

## Project configuration

- `pytest.ini` configures the project root on the pytest import path.
