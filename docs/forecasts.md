# Forecasting protocol

Frozen 2026-10-01, before forecasting-model performance was evaluated. This is the core experiment, not a list of models to try until one works. Changes after this freeze must be recorded as post-hoc deviations, with their reason and date, rather than replacing the headline specification.

## Samples, information and targets

The full processed SPX dataset remains unchanged. Split modelling rows by `quote_date`: development is 2023-01-03 through 2024-12-31; locked temporal confirmation is 2025-01-01 through 2025-08-29. Use date filters, not physically shortened source CSVs. Any saved modelling datasets belong under gitignored `data/processed/`, for example `forecast_dataset_dev_2023_2024.csv` and `forecast_dataset_holdout_2025.csv`.

The headline horizon is **five SPX trading sessions**. The two co-primary questions have fixed targets:

| Role | Existing target | Definition |
| --- | --- | --- |
| Statistical: outright 30D downside skew | `y_skew_30_5d` | $S^{30}_{t+5}-S^{30}_t$ |
| Relative value: 30D–60D skew spread | `y_skew_spread_5d` | $Z_{t+5}-Z_t$, where $Z_t=S^{30}_t-S^{60}_t$ |

$S^j_t$ is the QC-approved fixed-$j$D ATM downside-skew slope. The relative-value target is primary for any eventual cross-tenor interpretation. The 1D/10D horizons, 60D/90D outright skew, 60D–90D spread, and RR25 analogues are **robustness targets only**; none may replace a headline target because it performs better. Some of these robustness targets do not yet exist in `src/time_series.py` and must be constructed and tested before use.

Full-sample descriptive EDA has already found in-sample mean reversion: for the 5D spread change, the raw-spread coefficient was about $-0.507$, the past-only z-score coefficient about $-0.0273$, and in-sample $R^2$ about $0.247$. Several descriptive checks retained the negative sign. These are context, not out-of-sample results or coefficients to hard-code. They make simple mean reversion a serious mandatory benchmark; they do not license tuning thresholds, standardisation windows or models against those observations.

## Causal feature construction

Use the SPX security-price dates as the authoritative session calendar. Preserve missing sessions and QC-masked values; do not forward-fill surface metrics or bridge a one-session change. `build_daily_time_series(metrics, spx_prices)` already implements these rules, and `add_forward_targets(daily, spx_prices)` creates exact 1/5/10-session endpoints. A 5D label needs valid start and end metrics, not necessarily valid intermediate metrics. The last five sessions cannot have a 5D label.

For M2 and above, the own-state z-score uses an **expanding 60-valid-observation warm-up**, fixed now. At origin $t$, for the relevant QC-masked state $X_t$ (either $S^{30}_t$ or $Z_t$), compute

$$
z_t=\frac{X_t-\bar X_{<t}}{s_{<t}},
$$

where the mean and sample standard deviation use only earlier valid SPX-session observations. Shift before expanding; do not emit $z_t$ until 60 prior valid observations exist, or if the historical standard deviation is zero. The current $X_t$ enters only the numerator. No z-score window search is permitted. Other scaling or preprocessing must be fitted within the applicable training window or inner-CV fold only. In the strict locked confirmation, freeze fitted preprocessing, including the z-score reference moments, at the end of 2024; apply those frozen moments to 2025 states.

Same-date market-state variables are joined by exact date using `merge_market_state`; no VIX filling. The available variables are `spx_return`, `spx_abs_return`, `spx_return_sq`, `rv_5`, `rv_20` and `vix_close`. They are known only after the relevant close, so later execution assumptions must not place a trade at that same close.

## Chronological development experiment

Evaluate within 2023–2024 using an expanding estimation window. The first eligible forecast origin follows **252 prior SPX sessions**; missing labels remain missing rather than compressing the clock. At origin $t$, a training row originating at $i$ is eligible only if its endpoint has already occurred: $i+5\leq t$, with indices on the SPX session calendar. Apply that maturity/purge rule to fitting, mean changes, scaling and all inner validation. Do not use development-origin labels that mature in 2025 when fitting the final end-2024 model.

Generate and record predictions in chronological origin order before advancing to later labels. No random split or random KFold. For M6/M7, tune *inside each origin's eligible history* using the latest two non-overlapping chronological validation blocks of 30 SPX sessions, expanding training before each block, at least 100 earlier complete training labels before the first block, and the same 5-session label-maturity purge at each validation origin. If an origin cannot support those folds, M6/M7 issue no forecast there. Choose grid points by lowest pooled inner-validation RMSE; ties choose the simpler/stronger-regularised setting. Do not use later development outcomes to choose hyperparameters for an earlier pseudo-OOS prediction.

## Fixed model ladder

Fit separate versions for the two headline targets. Apart from M0, fit only on QC-valid, fully observed rows for that model's declared predictors and matured target. Intercepts are included in fitted linear models.

| Model | Fixed specification |
| --- | --- |
| M0 persistence | Predict zero 5D change. Mandatory benchmark. |
| M1 historical mean | Expanding mean of eligible, matured 5D target changes. |
| M2 mean reversion | OLS on the target's own past-only z-score: $\hat y_t=\alpha+\beta z_t$. No thresholds, polynomial terms or window search. |
| M3 own dynamics | M2 plus the immediately preceding one-session change in that same state, ending at $t$ and missing unless both $t$ and $t-1$ are valid. For the spread this is $Z_t-Z_{t-1}$; no additional lag in the core specification. |
| M4 surface state | M3 plus current `skew_60`, `skew_30_60`, `skew_60_90`, `atm_iv_30`, `atm_iv_30_60`, `atm_iv_60_90`. This nonredundant level/spread basis spans the three skew and ATM-IV tenors; do not also add all component levels. |
| M5a realised market state | M4 plus `spx_return`, `spx_abs_return`, `spx_return_sq`, `rv_5`, `rv_20`. |
| M5b implied-volatility state | M5a plus `vix_close`. No $\Delta\mathrm{VIX}$ or technical-indicator expansion in the core experiment. |
| M6 elastic net | scikit-learn `ElasticNet` on the M5b feature universe, with predictors standardised on each training fold only. Fixed grid: $\alpha\in\{10^{-4},10^{-3},10^{-2}\}$ and `l1_ratio` $\in\{0.25,0.75\}$. |
| M7 nonlinear challenger | scikit-learn `HistGradientBoostingRegressor` on the same M5b features: squared-error loss, depth 2, minimum leaf size 30, learning rate 0.05, boosting rounds $\in\{50,100\}$, `early_stopping=False`, `random_state=0`. No neural network/model zoo. |

The M5a/M5b distinction isolates whether VIX adds information beyond SPX returns and realised volatility. M6 tests correlated linear signals; M7 tests whether restrained nonlinear thresholds add anything. Do not add predictors or alter these grids after seeing results. Any required forecasting dependency will be added when the modelling code is implemented, not as part of this protocol.

## Scoring and evidence

For each target, compute headline comparisons on the **same OOS origin dates and realised outcomes** across the models being compared. Predeclare the common-date set before examining scores; report model coverage separately so missingness is visible. Report RMSE, MAE, forecast/realised correlation, and directional accuracy, plus

$$
R^2_{\mathrm{OOS,persist}}=1-\frac{\mathrm{SSE}_{\mathrm{model}}}{\mathrm{SSE}_{M0}},
\qquad
R^2_{\mathrm{OOS,MR}}=1-\frac{\mathrm{SSE}_{\mathrm{model}}}{\mathrm{SSE}_{M2}}
$$

for models richer than M2. A constant-zero forecast has undefined correlation and directional hit rate; report these as not applicable, not as zero or 50%. Directional accuracy is secondary. Plot cumulative squared-error improvement $\sum_{s\leq t}(e^2_{\mathrm{benchmark},s}-e^2_{\mathrm{model},s})$ against both relevant benchmarks to see whether gains are persistent or driven by a short episode.

Five-session targets overlap. For later uncertainty/statistical comparisons use horizon-consistent HAC/Newey–West treatment and repeat key results on all five non-overlapping origin offsets. Do not use iid t-statistics. Incremental predictive value requires improvement over **both** M0 and M2 for the same target and dates, with lower RMSE, no offsetting MAE deterioration, and broadly accumulating error reduction rather than one isolated episode. Select the simplest model meeting that standard; these are development-period decisions only.

## Locked confirmation and pre-specified checks

Use 2023–2024 pseudo-OOS results to choose one final specification per headline target. For `locked_holdout`, fit it using only development labels matured by 2024-12-31, then freeze coefficients, hyperparameters, feature definitions and fitted transformations. Predict 2025 without adapting to its forecast errors or labels. Report separately, if later desired, `recursive_deployment`: the same frozen model family and hyperparameters may re-estimate coefficients using newly matured 2025 labels as they become available. Never conflate the two or use 2025 outcomes for model/feature selection.

After headline evaluation, run the pre-specified robustness matrix: 1D/5D/10D; 30D/60D/90D outright; 30D–60D/60D–90D spreads; skew slope/RR25; overlapping/all five non-overlapping 5D offsets; endpoint-valid/complete-path-valid targets; VIX below/at-or-above a **fixed level of 20**; and excluding the most extreme 1% of absolute target moves. Fix the extreme-move cutoff from development outcomes only and use it for the 2025 diagnostic; outcome-based trimming is *evaluation only*, never a live predictor or training-time rule. Robustness cells cannot replace the headline result.

Check whether prediction errors or apparent gains cluster around bracket gaps, rejected Raw SVI slices, limited maturity support or large calendar repairs. Use existing surface diagnostics as quality controls first, not a new predictor block. Keep metric-specific QC masking and exact-date joins. Options-trading simulation is outside this stage; only after OOS forecasting beats persistence and simple mean reversion should a vega-aware, delta-hedged cross-tenor trade with executable bid/ask assumptions be designed.

When forecasting code is implemented, unit tests must cover: date split and 2025 exclusion from development fitting; SPX-session horizon alignment and missing final-five labels; $i+h\leq t$ purging, including the 2024/2025 boundary; past-only z-scores and rolling features; training-only scalers; chronological purged CV and no random splits; metric/date missingness; common-date scoring; and frozen 2025 model, hyperparameters and preprocessing.

## Frozen decisions

| Decision | Frozen choice |
| --- | --- |
| Development / confirmation | 2023-01-03–2024-12-31 / 2025-01-01–2025-08-29; date-filtered, source CSV unchanged |
| Headline targets | `y_skew_30_5d` and `y_skew_spread_5d`; 5 SPX sessions |
| Z-score | Expanding, 60 prior QC-valid observations; no window search |
| Origin and label eligibility | First forecast after 252 prior SPX sessions; train only on $i+5\leq t$ |
| Model comparison | M0–M7 above, with M5a/M5b ablation; smallest qualifying model wins |
| CV / tuning | Two expanding purged 30-session blocks; stated M6/M7 grids; inner RMSE |
| Evaluation | Common OOS dates; RMSE, MAE, two benchmark-relative $R^2$s, correlation, directional accuracy, cumulative squared-error gains |
| Confirmation | Static development-only `locked_holdout`; optional separately labelled `recursive_deployment` |
| Robustness | Declared matrix only; VIX cut 20; 1% move cutoff fixed from development |
| Scope | Forecasting first; no options trading backtest yet |

Any departure from this table must be labelled post-hoc with its date, reason, affected results and original-protocol comparison.
