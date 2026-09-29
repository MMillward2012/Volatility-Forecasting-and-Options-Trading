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
4. Call `fit_raw_svi_surface(..., enforce_arbitrage=True)` separately for each positive-DTE
   expiry. This preserves its quote screen, least-squares objective, and multi-start fallback.
   Record every failed expiry and its exception. Preserve each failed expiry as an
   unsupported slot on the maturity grid, so interpolation cannot bridge it. A date with
   some failed slices is `partial`; metrics requiring those slices remain unavailable.
   If all slices fail, the date is `failed`.
5. Use the existing supported $k\in[-0.5,0.25]$ grid with 151 nodes and calendar projection.
6. Call `calculate_skew_metrics(...)` for 30D, 60D, and 90D with its existing defaults.

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
the source file's path, byte size, and modification time. Completed failed/partial dates
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
code changes. Do not run concurrent writers against the same output directory.

## Validation

`tests/test_historical_pipeline.py` covers dates split across chunks, rejection of
unsorted/overlapping input, validation of every constant including AM rows, early quote
filtering, an end-to-end synthetic date using the real models, failed SVI expiries,
continuation after a failed date, idempotent resume/retry, and recovery from interrupted
checkpoint writes. Existing module tests continue to cover the mathematical definitions.

A complete reader/constant-validation pass covers all 14,726,287 raw rows and 667 dates,
with PM contracts on every date. `expiry_indicator` is read as nullable text because
chunks containing only blank values would otherwise infer a different dtype.

Real-data smoke runs have been completed for 2023-01-03, 2024-01-02, and 2025-08-29.
All nine target-tenor rows have valid ATM IV, local slope, and 25-delta RR. The first
two dates are partial: respectively two short expiries (2023-01-05 and 2023-01-09)
and one short expiry (2024-01-03) fail the existing SVI calibration. Their unsupported
slots do not bracket the target tenors. Sampled calendar crossings fall from 188, 81,
and 22 to zero, respectively. These are validation dates, not a completed historical
calibration; the default command resumes from the small output files.
