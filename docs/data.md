# Data

## Source data

The current raw dataset is:

```text
data/raw/spx_option_prices_2025-08-29.csv
```

It contains OptionMetrics end-of-day SPX option data. The raw file has 24,072
rows for the 29 August 2025 quote date.

## Current cleaning step

`src/data_cleaning.py` currently keeps rows where:

```text
am_settlement == 0
```

For the current file, this produces 15,670 rows. The cleaned output is written
to:

```text
data/processed/spx_option_prices_cleaned_2025-08-29.csv
```

No other row filters are applied by this cleaning step.

Run from the repository root with `python -m src.data_cleaning`. The settlement filter lives in the script entry point; direct calls to `clean_option_data(...)` transform the supplied rows without that filter.

## Cleaned schema

| Cleaned column | Source / derivation | Meaning |
| --- | --- | --- |
| `security_id` | `secid` | Underlying security ID |
| `quote_date` | `date` | End-of-day quote date |
| `option_id` | `optionid` | Individual option contract identifier |
| `symbol` | `symbol` | Original OptionMetrics contract symbol |
| `expiry_date` | `exdate` | Contract expiration date |
| `option_type` | `cp_flag` | `C` becomes `call`; `P` becomes `put` |
| `strike` | `strike_price / 1000` | Strike price in index points |
| `best_bid` | `best_bid` | Closing best bid |
| `best_ask` | `best_offer` | Closing best ask |
| `mid_price` | `(best_bid + best_ask) / 2` | Quote midpoint |
| `spread` | `best_ask - best_bid` | Absolute bid-ask spread |
| `relative_spread` | `spread / mid_price` | Spread relative to midpoint |
| `volume` | `volume` | Contracts traded that day |
| `open_interest` | `open_interest` | Outstanding contracts |
| `days_to_expiry` | `expiry_date - quote_date` | Calendar days remaining |
| `time_to_expiry` | `days_to_expiry / 365` | ACT/365 time to expiry in years |
| `is_expiry_day` | `days_to_expiry == 0` | Flag for expiry-day observations |
| `expiry_indicator` | `expiry_indicator` | OptionMetrics expiry category |
| `vendor_iv` | `impl_volatility` | OptionMetrics implied volatility benchmark |
| `vendor_delta` | `delta` | OptionMetrics delta benchmark |
| `vendor_gamma` | `gamma` | OptionMetrics gamma benchmark |
| `vendor_vega` | `vega` | OptionMetrics vega benchmark |
| `vendor_theta` | `theta` | OptionMetrics theta benchmark |

## Matched call-put data

`src/option_matching.py` reads the cleaned CSV and inner-joins calls and puts on:

```text
security_id, quote_date, expiry_date, strike
```

Matching keys must be present and unique within each option type. Missing keys raise `ValueError`; duplicate keys raise pandas `MergeError`. Unmatched rows are excluded.

The matched columns, in order, are:

```text
security_id, quote_date, expiry_date, strike,
days_to_expiry, time_to_expiry, is_expiry_day,
call_option_id, call_symbol, call_bid, call_ask, call_mid, call_volume, call_open_interest,
put_option_id, put_symbol, put_bid, put_ask, put_mid, put_volume, put_open_interest
```

Expiry times and the expiry-day flag are retained from the call row. Prices and contract fields are copied from their respective sides.

Run from the repository root with `python -m src.option_matching`. The current file produces 7,835 pairs, including 409 expiry-day pairs, at:

```text
data/processed/spx_option_prices_matched_2025-08-29.csv
```

## Forward estimates

`src/forward_inference.py` reads the matched data, excludes the 409 expiry-day pairs, and estimates one forward and discount factor for each positive-DTE `(security_id, quote_date, expiry_date)` group using the midpoint call-put differences.

The output is written to:

```text
data/processed/spx_forward_estimates_2025-08-29.csv
```

Its expiry-level fields are:

```text
security_id, quote_date, expiry_date,
days_to_expiry, time_to_expiry,
forward, discount_factor, n_strikes, parity_rmse
```

The estimates use:

$$
C_K-P_K=D(F-K).
$$

The current diagnostic notebook also compares the OLS fit with a spread-weighted fit and checks whether a single common $F,D$ can satisfy the executable parity interval for every strike in each expiry. These checks are exploratory and do not currently filter the matched data or replace the baseline estimates.

## IV-analysis panel

`src/iv_panel.py` joins the cleaned option rows to the expiry-level forward estimates after excluding expiry-day rows. It writes:

```text
data/processed/spx_iv_panel_2025-08-29.csv
```

The panel retains the cleaned option fields and adds:

| Column | Meaning |
| --- | --- |
| `forward` | Joined expiry-level forward estimate |
| `discount_factor` | Joined expiry-level discount factor |
| `log_moneyness` | $\log(K/F)$ |
| `mid_iv` | Implied volatility inverted from the option midpoint |
| `is_otm` | OTM put for $K<F$ or OTM call for $K>F$ |
| `use_for_surface` | `True` only for OTM observations with valid `mid_iv` |

If a midpoint lies outside the admissible Black bounds, `mid_iv` is `NaN` for that observation. Joined `forward`, `discount_factor`, `strike`, and `time_to_expiry` values must be finite and strictly positive. Missing expiry-level forward data raises an error if the merge would drop any positive-DTE option rows.

## Notebook fitting sample

`notebooks/iv_surface_diagnostics.ipynb` creates `use_for_fit` in memory. This flag is **not** currently written by `src/iv_panel.py` or saved in the panel CSV. It requires:

- `use_for_surface == True`;
- finite bid, ask, midpoint, midpoint IV, and relative spread;
- a positive midpoint and `0 < best_bid < best_ask`;
- `0 < relative_spread <= 0.50`.

The notebook retains every original row and uses the flag only to select the fitting-sample plots. It imposes no premium floor or IV-level cap. Locked quotes are excluded as a fitting-screen choice, not classified as necessarily erroneous.

For the **2025-08-29** snapshot:

| Sample | All positive-DTE expiries | 1–7 DTE |
| --- | ---: | ---: |
| Valid OTM IVs (`use_for_surface`) | 7,426 | 871 |
| Previous positive-bid / 50% relative-spread screen | 6,598 | 607 |
| Previous screen plus midpoint at least 0.25 | 6,466 | 506 |
| Selected positive-spread screen (`use_for_fit`) | 6,550 | 559 |

The selected screen removes 48 locked quotes from the previous sample while retaining 84 observations rejected by the premium-floor alternative. All 38 positive-DTE expiries retain both calls and puts. The notebook reports per-expiry coverage; the 50% spread cutoff remains provisional and has not been validated through surface fitting or across multiple quote dates.

## Current caveats

- Rows with zero midpoint have an undefined `relative_spread`, represented as `NaN`.
- Expiry-day rows are retained and flagged in the cleaned and matched outputs, but excluded from forward inference and the IV panel.
- Vendor Greeks and implied volatility are retained as benchmarks, not as model outputs.
- Matching does not apply liquidity or model-estimation filters. Forward and discount-factor inference must use one security, quote date, and expiry at a time.
