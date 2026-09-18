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

No other row filters are currently applied.

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

## Current caveats

- Rows with zero midpoint have an undefined `relative_spread`, represented as `NaN`.
- Expiry-day rows are retained and flagged in both outputs. Exclude or handle them before pricing or IV inversion, which require positive time to expiry.
- Vendor Greeks and implied volatility are retained as benchmarks, not as model outputs.
- Matching does not apply liquidity or model-estimation filters. Forward and discount-factor inference must use one security, quote date, and expiry at a time.
