# SPX Volatility Forecasting and Options Trading

An end-to-end quantitative research project using **14.7 million OptionMetrics SPX option quotes across 667 sessions**, from January 2023 to August 2025. It connects volatility-surface construction, chronological forecasting and a listed-options execution study.

Simple mean reversion improved five-session skew forecasts; the execution study quantified the additional risks and costs of trading that signal.

## Research findings

The two headline targets are five-SPX-session changes in **30D ATM downside skew** and the **30D–60D skew spread**. Development used 2023–2024; frozen models were evaluated on 2025-01-02–2025-08-29.

| Target | Persistence RMSE | Mean-reversion RMSE | RMSE reduction | OOS R² |
| --- | ---: | ---: | ---: | ---: |
| 30D downside skew | 0.1012 | 0.0922 | **8.9%** | 0.1695 |
| 30D–60D skew spread | 0.0544 | 0.0492 | **9.6%** | 0.1824 |

These are **158 common, overlapping forecast origins**. Spread correlation was 0.5619 and directional accuracy 62.03%. Earlier descriptive analysis included 2025; this is locked temporal confirmation, not wholly unseen data.

![Frozen mean reversion versus persistence](docs/figures/locked_2025_m2_gain.png)

*Positive cumulative squared-error gain favours mean reversion on identical scored dates.*

The pre-specified model ladder:

| Model | Specification |
| --- | --- |
| M0 / M1 | Persistence / historical mean change |
| M2 / M3 | Mean reversion / plus recent own change |
| M4 | Add surface-state levels and term spreads |
| M5a / M5b | Add SPX returns and realised volatility / then VIX |
| M6 / M7 | Elastic Net / shallow histogram gradient boosting |

The development-selected Elastic Net did not retain its advantage: confirmation RMSE was 0.1081, with R² of −0.3731 versus M2. Results were preserved without retuning.

## From quotes to fixed-tenor states

PM-settled SPX quotes → quote QC → matched calls/puts → parity-implied forwards and discount factors → midpoint IV → **arbitrage-aware Raw SVI per expiry** → support-aware calendar repair in total variance → 30D/60D/90D metrics.

Production uses 14–180 DTE expiries and metric-specific QC. SSVI is an exploratory comparator.

With $k=\log(K/F)$, downside skew is the local $-\partial\sigma/\partial k$ around ATM. Downside RR25 is 25-delta put IV minus 25-delta call IV.

![Calendar-repaired fixed-tenor smiles](docs/figures/repaired_smiles_30_60_90.png)

*Existing 2025-08-29 diagnostic. Dashed curves fail full-smile QC; ATM and RR25 have separate checks. Sampled checks do not establish global arbitrage freedom.*

Forecasts use expanding windows after 252 sessions, exact session horizons, causal z-scores and matured labels only ($i+h\leq t$). Comparisons share dates; no random splits or surface forward-fill. See the [frozen protocol](docs/forecasts.md).

## Listed-options translation test

The frozen RR25-spread mean-reversion signal was translated into four actual listed options: equal absolute wing vega, fixed quantities, next-session entry, original fifth-session exit, observed bid/ask and a daily SPX delta-hedge proxy.

The primary total was **−0.195946 per unit of entry gross vega**, versus −0.029551 at midpoint. Predeclared controls use identical contracts and origins:

| Position, same 149 origins/contracts | Midpoint total | Execution drag | Executable total |
| --- | ---: | ---: | ---: |
| Forecast direction | −0.029551 | −0.166395 | −0.195946 |
| Reversed direction control | +0.029551 | −0.166395 | −0.136844 |
| Constant-positive control | +0.078964 | −0.166395 | −0.087431 |

All 149 baskets exited on schedule: **89 positive / 60 negative** directions. Every fixed origin offset had a negative executable total.

Constant-tenor, constant-delta RR changes differ from the exposure of ageing fixed contracts. Entry delay, changing exposures and bid/ask matter. The [attribution notebook](notebooks/trading_attribution.ipynb) reconciles the result through time/discount, spot, forward basis, IV level and held-wing shape. Its allocation is path-dependent, not causal skew alpha.

Rules were committed before trading P&L was opened, following known forecasting results. This is a locked translation test, not prospective evidence. The SPX hedge is frictionless; quantities are fractional and financing/margin omitted. Trading profitability is not established. See [protocol](docs/trading_protocol.md) and [locked results](notebooks/trading_locked_2025.ipynb).

## Robustness and interpretation

Mean-reversion slopes were negative at 1D/5D/10D; OOS R² was 0.037/0.169/0.288 outright and 0.106/0.182/0.317 for the spread. RR25-spread R² was ≈0.225. All five fixed non-overlapping offsets and both outlier diagnostics retained forecasting gains.

![Horizon robustness](docs/figures/horizon_robustness.png)

*Squared-error magnitudes are not scale-neutral across horizons.*

Gains were stronger at VIX ≥ 20. Fixed-lag-4 HAC intervals include zero; overlap and the short sample limit inference. The VIX result is descriptive, not a fitted regime rule.

Two executed trades had unavailable QC-approved forecasting endpoints and correctly remain in trading P&L. Surface-estimation noise remains a research limitation. Any redesigned strategy needs a fresh confirmation period; already-examined data are development information.

## Code and reproduction

| Location | Purpose |
| --- | --- |
| `src/historical_pipeline.py`, `raw_svi.py`, `vol_surface.py`, `skew_metrics.py` | Quote-to-surface pipeline and metric QC |
| `src/forecasting.py`, `forecast_robustness.py` | Frozen forecasting and robustness checks |
| `src/trade_selection.py`, `trade_backtest.py`, `trading_locked.py` | Listed contracts, accounting and frozen evaluation |
| `src/trade_attribution.py` | Post-result analysis of saved positions |
| [Notebook guide](notebooks/README.md) | Research order, exploratory work and output policy |
| [Processing guide](docs/historical_processing.md), [progress](docs/progress.md) | Reproduction details and audit trail |

Python 3.13 was used for the saved results. Install the pinned direct dependencies and run the deterministic tests:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
pytest
python -m src.historical_pipeline
```

Licensed SPX option/SPX/VIX extracts are required locally; raw and processed data are gitignored. Guides list filenames and notebook order. Aggregate results remain saved; quote-bearing outputs are cleared:

```bash
python -m scripts.prepare_public_notebooks --check
```

Earlier Git history still contains cleared outputs; this cleanup does not establish licence permission. The locked evaluator has a one-attempt guard; attribution reads its saved audit.
