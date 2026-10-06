# Notebook guide

## Main research sequence

1. Run `python -m src.historical_pipeline` with licensed local annual option files; see [historical processing](../docs/historical_processing.md).
2. `skew_time_series_analysis.ipynb`: descriptive dynamics of the QC-masked daily series.
3. `forecast_baselines.ipynb`, `forecast_linear_models.ipynb`, `forecast_ml_models.ipynb`: 2023–2024 chronological development comparisons.
4. `forecast_locked_2025.ipynb`: frozen confirmation; `forecast_robustness.ipynb`: subsequent pre-specified robustness.
5. `trading_feasibility.ipynb`: development-only listed-contract coverage and lifecycle checks.
6. `trading_locked_2025.ipynb`: saved single-run trading result. Its evaluator checkpoint and exclusive manifest prevent accidental repetition; view its committed outputs rather than rerunning it to redraw figures.
7. `trading_attribution.ipynb`: explicitly post-result direction counts and attribution from the saved positions. This requires the local trade audit plus licensed exit quotes and does not select trades or refit forecasts.

The historical source files are `data/raw/spx_option_prices_2023.csv`, `spx_option_prices_2024.csv`, and `spx_option_prices_2025.csv`. Time-series/forecast construction also needs `spx_security_prices.csv` and `VIX_security_prices.csv`; generated metrics and audits are under gitignored `data/processed/`.

## Surface development and exploratory notebooks

`forward_inference_diagnostics.ipynb` and `iv_surface_diagnostics.ipynb` document quote diagnostics. `raw_svi_diagnostics.ipynb` and `skew_metrics_diagnostics.ipynb` explain the production Raw-SVI/calendar-repair method on a representative date; they are not historical acceptance reports.

`single_expiry_SSVI_slice.ipynb` and `ssvi_surface_diagnostics.ipynb`, together with `src/ssvi.py`, are **exploratory comparators**. Production calibration uses `src/raw_svi.py` and `src/vol_surface.py`. Their paths remain unchanged to preserve notebook imports and historical links. The single-expiry notebook's manual slider cell is interactive and optional; an unexecuted widget cell is not an incomplete production step.

## Public outputs

Quote-bearing development outputs, including observed-IV Plotly payloads, are intentionally cleared. Aggregate forecasting, trading and feasibility results remain saved. Private feasibility examples write to a local CSV rather than display original contract quotes. Run `python -m scripts.prepare_public_notebooks --check` before publishing. Previously committed outputs remain in Git history.
