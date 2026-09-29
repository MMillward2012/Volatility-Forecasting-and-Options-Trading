# Historical SPX processing

Run from the repository root with the project environment activated:

```bash
python -m src.historical_pipeline
```

The default inputs are the local, gitignored `data/raw/spx_option_prices_2023.csv`,
`spx_option_prices_2024.csv`, and `spx_option_prices_2025.csv`. The available extract
covers 667 quote dates from 2023-01-03 through 2025-08-29; 2025 is a partial year.

The reader uses 100,000-row chunks and carries the last date into the next chunk,
so each date is processed in full. Files must be supplied in chronological order,
with sorted, nonmissing quote dates and no overlapping dates between files.
Malformed files or out-of-order dates stop the reader with an explicit error.
Errors within an identified quote date are logged and processing continues.

## Daily processing

1. Validate `secid=108105`, `ticker=SPX`, `exercise_style=E`, `contract_size=100`,
   `cfadj=1`, `ss_flag=0`, and `index_flag=1` on all raw rows, including AM rows.
2. Retain `am_settlement == 0` and remove crossed PM quotes (`best_bid > best_offer`).
   Zero bids, zero volume/open interest, and missing vendor IV/Greeks remain at this stage.
3. Call the existing cleaning, matching, expiry forward inference, and IV-panel functions.
4. Retain only expiries with $14\leq\mathrm{DTE}\leq180$ for SVI and surface construction.
   Full-chain cleaning, forward and IV counts remain available; `research_window_expiries`
   records the retained count. Call `fit_raw_svi_surface(..., enforce_arbitrage=True)`
   separately for each retained expiry, preserving its quote screen, objective and constraints.
   Interpolate the nearest screened midpoint IV strictly below and strictly above $k=0$
   to get $\sigma_{\mathrm{local}}$. Reject the fit when either ATM estimate is unavailable
   or nonfinite, or when
   $|\sigma_{\mathrm{SVI}}-\sigma_{\mathrm{local}}|>\max(0.03,0.20\sigma_{\mathrm{local}})$.
   `svi_atm_rejections` records expiry, DTE, both estimates, signed difference, and reason.
   Record every failed expiry and its exception, and exclude it from the fitted grid.
   A date with some failed slices is `partial`.
   If all slices fail, the date is `failed`.
5. Use the existing supported $k\in[-0.5,0.25]$ grid with 151 nodes and calendar projection.
6. Bracket each 30D, 60D, and 90D target using its nearest accepted fitted expiries.
   Require at most 30 calendar days between the two expiries; an exact accepted expiry
   has a zero-day gap. Do not extrapolate beyond accepted maturities. Larger gaps give
   `excessive_maturity_gap`, missing metrics and false validity flags. This applies even
   when no failed expiries lie between the fits. The 30-day limit is an initial research
   choice, not a mathematical guarantee of interpolation quality.
   Call `calculate_skew_metrics(...)` with its existing defaults on allowed brackets.
   Both endpoints must support each metric's required strikes; do not switch to more
   distant expiries to obtain wider strike support. Existing metric QC still applies.
   Evaluate ATM IV on the raw and repaired grids using the same surrounding expiries.
   If the absolute change exceeds $0.05$ (five volatility percentage points), clear all
   three research metric values, set their validity flags false, and set their failure
   reasons to `excessive_calendar_repair`. Retain their pre-guard calculated values in
   diagnostics. This check does not alter the calendar projection or the metric formulas.

The model definitions and public APIs are unchanged. ATM downside skew slope is the
primary research target; downside 25-delta RR is the trading-oriented robustness
measure, and ATM IV is a state/control variable. Metric validity is independent of
the full-smile QC flag. A successful processing status means the calculation completed,
not that every metric passed QC. Partial dates can still contain valid metrics at tenors
whose surrounding expiries fitted successfully. Sampled QC does not prove arbitrage freedom.

## Outputs

Only two small CSVs are persisted by default:

- `data/processed/spx_skew_metrics_2023_2025.csv`: three rows per processed quote date,
  preserving the existing metric values, independent flags and failure reasons, delta
  locations, and full-smile diagnostics. Failed dates retain three rows with missing
  values, false flags, and the processing failure reason.
- `data/processed/spx_daily_processing_diagnostics_2023_2025.csv`: one row per date,
  with raw/AM/PM/cleaned counts, dropped crossed quotes, matched pairs, positive-DTE and
  inferred expiries, IV counts, fitted/failed SVI expiries and their reasons, supported
  calendar crossing counts before/after repair, maximum absolute variance/IV adjustments,
  tenor-specific metric availability and validity, processing status, failure stage/reason,
  and processing runtime in seconds.

Metric rows and diagnostics include `raw_atm_iv`, `repaired_atm_iv`, and the signed
`calendar_adjustment_atm_iv` (prefixed by tenor in daily diagnostics). Diagnostics also
retain each tenor's `calculated_atm_iv`, `calculated_atm_skew_slope`, and
`calculated_rr25_downside` before the repair guard, plus `calendar_repair_valid`.
ATM sanity rejections count as failed SVI expiries. Both outputs record
`bracket_lower_expiry`, `bracket_upper_expiry`, `bracket_gap_days`,
`bracket_skipped_expiries`, and `bracket_valid` (prefixed by tenor in daily diagnostics).
Skipped expiries count listed research-window maturities strictly between the accepted
endpoints, including a rejected expiry exactly at the target. Bracket validity describes
the maturity range and gap only; metric-specific strike support and QC remain separate.

IVs and IV adjustments use decimal volatility. `atm_skew_slope` retains the existing
$-\partial\sigma/\partial k$ units; its numeric value is also volatility percentage
points per 1% change in $k$. The primary downside sign convention is positive.
`max_calendar_adjustment_w` is in total variance, and crossing counts compare consecutive
supported maturities at each grid node. Unreached stage diagnostics are missing rather
than reported as zero. Large daily quote panels and surfaces remain in memory.

## Resume and targeted runs

Each date replaces its previous rows. Each CSV is replaced atomically, with diagnostics
written last. Matching checkpoint IDs in both files identify completed writes; an
interruption between the writes causes that date to be recomputed. Resume also checks
the source file's path, byte size, modification time, and surface-guard version. Checkpoints
from before these guards are recomputed when encountered. Completed failed/partial dates
are skipped by default and can be retried explicitly:

```bash
python -m src.historical_pipeline --retry-failed
```

For a selected period:

```bash
python -m src.historical_pipeline \
  --raw-files data/raw/spx_option_prices_2023.csv \
  --start-date 2023-01-03 --end-date 2023-01-03
```

Use `--output-dir` for isolated experiments and `--chunksize` to adjust reading memory.
`--no-resume` starts fresh results for the requested selection. After changing model
code or settings, use fresh outputs or `--no-resume`; source signatures do not detect
arbitrary code changes. Do not run concurrent writers against the same output directory.

## Validation

`tests/test_historical_pipeline.py` covers dates split across chunks, rejection of
unsorted/overlapping input, validation of every constant including AM rows, early quote
filtering, an end-to-end synthetic date using the real models, failed SVI expiries,
continuation after a failed date, idempotent resume/retry, and recovery from interrupted
checkpoint writes. Existing module tests continue to cover the mathematical definitions.
Guard tests cover inclusive maturity bounds, the screened ATM interpolation, missing
benchmarks, absolute/relative tolerances, skipping rejected fits, the inclusive 30-day gap
limit, exact accepted expiries, no extrapolation, continued strike-support checks,
positive/negative excessive repairs, preservation of diagnostics and unaffected tenors,
and invalidation of old checkpoints.

A complete reader/constant-validation pass covers all 14,726,287 raw rows and 667 dates,
with PM contracts on every date. `expiry_indicator` is read as nullable text because
chunks containing only blank values would otherwise infer a different dtype.

Before these guards, real-data smoke runs were completed for 2023-01-03, 2024-01-02, and 2025-08-29.
All nine target-tenor rows have valid ATM IV, local slope, and 25-delta RR. The first
two dates are partial: respectively two short expiries (2023-01-05 and 2023-01-09)
and one short expiry (2024-01-03) fail the existing SVI calibration. Their unsupported
slots do not bracket the target tenors. Sampled calendar crossings fall from 188, 81,
and 22 to zero, respectively. These results describe the original implementation.

### Targeted guard validation

Compared the guarded runner with the original runner at commit `cd9fdde` on
2023-02-27, 2023-02-28, 2023-03-01, 2023-05-16, 2023-05-17, 2023-05-18, and
2025-08-29. Outputs went to a fresh temporary directory. Unchanged expiry calibrations
were reused between the old/new calculations to isolate the guards' effects; reconstructed
old metrics matched the previously saved values. No full historical rerun was performed.

| Date | Tenor | Old ATM IV | Guarded ATM IV |
|---|---:|---:|---:|
| 2023-02-28 | 30D | 65.6427% | 18.7249% |
| 2023-02-28 | 60D | 46.4164% | 18.0512% |
| 2023-02-28 | 90D | 37.8988% | 18.0224% |
| 2023-05-17 | 30D | 80.6204% | 14.5910% |
| 2023-05-17 | 60D | 57.0072% | 14.9539% |
| 2023-05-17 | 90D | 46.5462% | 15.6678% |

The February 28 culprit (2023-03-06, six DTE) is outside the research window. The May 17
culprit (2023-06-07, 21 DTE) is rejected: fitted ATM IV is 4.169047 versus local ATM IV
0.127920, a signed difference of 4.041127 in decimal volatility. These results used the
original guard policy of retaining unsupported failed slots. May 18 retained one
pre-existing SVI calibration failure.

All 21 guarded tenor rows pass ATM, slope and RR25 checks. The four adjacent control
dates and 2025-08-29 have identical metric values to the original results. The maximum
absolute target ATM calendar adjustment is 0.0000611855 (0.00611855 volatility percentage
points), at February 28's 30D tenor; every other target adjustment is zero. All sampled
calendar crossings are removed. The repair guard is exercised separately by tests with
both excessive positive and negative adjustments. These checks validate this sample;
they do not establish historical coverage or fit quality away from ATM.

### Accepted-expiry bracket validation

Reprocessed the same seven dates after enabling nearest-accepted brackets and the
30-day gap limit, using fresh temporary outputs. All 21 tenor rows retained valid ATM,
slope and RR25 metrics; changes from the preceding guarded results were below $10^{-12}$.
Their bracket gaps ranged from zero to 18 days. None required skipping a failed expiry
inside a target bracket, so these dates establish regression stability rather than a
historical coverage improvement. Synthetic tests verify recovery across rejected fits,
including a rejected expiry exactly at the target, and rejection of gaps above 30 days.
The full test suite passed. No full historical rerun was performed.
