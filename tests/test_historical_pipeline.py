import numpy as np
import pandas as pd
import pytest

import src.historical_pipeline as pipeline
from src.pricing import black_scholes_call_price, black_scholes_put_price


def raw_options(date="2025-01-02", expiry_days=(20, 100)):
    date = pd.Timestamp(date)
    rows = []
    for days in expiry_days:
        for strike in 100 * np.exp(np.linspace(-0.18, 0.18, 13)):
            for cp, price_function in [("C", black_scholes_call_price), ("P", black_scholes_put_price)]:
                price = price_function(100, strike, 0.98, days / 365, 0.2)
                rows.append({
                    **pipeline.CONSTANTS, "date": date,
                    "exdate": date + pd.Timedelta(days=days), "symbol": "SPXW test",
                    "optionid": len(rows), "cp_flag": cp, "strike_price": strike * 1000,
                    "best_bid": 0.95 * price, "best_offer": 1.05 * price,
                    "volume": 0, "open_interest": 0, "am_settlement": 0,
                    "expiry_indicator": "w", "impl_volatility": np.nan,
                    "delta": np.nan, "gamma": np.nan, "vega": np.nan, "theta": np.nan,
                })
    return pd.DataFrame(rows)


def test_reader_retains_complete_dates_across_chunk_boundaries(tmp_path):
    first = raw_options()
    second = raw_options("2025-01-03")
    path = tmp_path / "raw.csv"
    pd.concat([first, second]).to_csv(path, index=False)

    groups = list(pipeline.iter_quote_dates([path], chunksize=17))

    assert [date for _, date, _ in groups] == list(pd.to_datetime(["2025-01-02", "2025-01-03"]))
    assert [len(rows) for _, _, rows in groups] == [len(first), len(second)]
    assert [rows.optionid.tolist() for _, _, rows in groups] == [first.optionid.tolist(), second.optionid.tolist()]


def test_reader_rejects_unsorted_and_overlapping_dates(tmp_path):
    path = tmp_path / "raw.csv"
    pd.concat([raw_options("2025-01-03"), raw_options()]).to_csv(path, index=False)
    with pytest.raises(ValueError, match="sorted"):
        list(pipeline.iter_quote_dates([path], chunksize=30))
    raw_options().to_csv(path, index=False)
    with pytest.raises(ValueError, match="overlap"):
        list(pipeline.iter_quote_dates([path, path]))


@pytest.mark.parametrize("column", pipeline.CONSTANTS)
def test_constants_are_validated_even_on_am_rows(column):
    raw = raw_options()
    raw.loc[0, "am_settlement"] = 1
    raw.loc[0, column] = "wrong" if isinstance(pipeline.CONSTANTS[column], str) else -1

    metrics, diagnostics = pipeline.process_quote_date(raw, "2025-01-02")

    assert diagnostics["processing_status"] == "failed"
    assert diagnostics["failure_stage"] == "validation"
    assert column in diagnostics["failure_reason"]
    assert len(metrics) == 3
    assert metrics[list(pipeline.METRICS.values())].isna().all().all()
    assert not metrics[["atm_valid", "slope_valid", "rr25_valid"]].any().any()


def test_only_am_and_crossed_quotes_are_removed_before_cleaning(monkeypatch):
    raw = raw_options()
    raw.loc[0, "am_settlement"] = 1
    raw.loc[1, "best_bid"] = raw.loc[1, "best_offer"] + 1
    raw.loc[2, "best_bid"] = 0
    observed = []

    def inspect_cleaned(cleaned):
        observed.append(cleaned)
        raise ValueError("stop after cleaning")

    monkeypatch.setattr(pipeline, "match_calls_and_puts", inspect_cleaned)
    _, diagnostics = pipeline.process_quote_date(raw, "2025-01-02")

    cleaned = observed[0]
    assert len(cleaned) == len(raw) - 2
    assert cleaned.best_bid.eq(0).sum() == 1
    assert cleaned.volume.eq(0).all() and cleaned.open_interest.eq(0).all()
    assert cleaned.vendor_iv.isna().all()
    assert diagnostics["raw_rows"] == len(raw)
    assert diagnostics["pm_rows"] == len(raw) - 1
    assert diagnostics["crossed_pm_rows_dropped"] == 1
    assert diagnostics["failure_stage"] == "matching"


def test_synthetic_day_runs_existing_models_end_to_end():
    metrics, diagnostics = pipeline.process_quote_date(raw_options(), "2025-01-02")

    assert diagnostics["processing_status"] == "success", diagnostics["failure_reason"]
    assert metrics.target_days.tolist() == [30, 60, 90]
    np.testing.assert_allclose(metrics.atm_iv, 0.2, atol=1e-4)
    assert metrics.atm_valid.all()
    assert diagnostics["matched_pairs"] == 26
    assert diagnostics["positive_dte_expiries"] == diagnostics["svi_expiries_fitted"] == 2
    assert diagnostics["svi_expiries_failed"] == 0
    assert diagnostics["calendar_crossings_after"] == 0
    assert diagnostics["30d_atm_available"]


def test_failed_svi_expiry_is_recorded_without_bridging_it(monkeypatch):
    original_fit = pipeline.fit_raw_svi_surface

    def fit(rows, **kwargs):
        assert kwargs == {"enforce_arbitrage": True}
        if rows.time_to_expiry.iloc[0] == 20 / 365:
            raise ValueError("synthetic convergence failure")
        return original_fit(rows, **kwargs)

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    metrics, diagnostics = pipeline.process_quote_date(
        raw_options(expiry_days=(20, 60, 100)), "2025-01-02"
    )

    assert diagnostics["processing_status"] == "partial"
    assert diagnostics["failure_stage"] == "raw_svi"
    assert diagnostics["svi_expiries_attempted"] == 3
    assert diagnostics["svi_expiries_fitted"] == 2
    assert diagnostics["svi_expiries_failed"] == 1
    assert "synthetic convergence failure" in diagnostics["svi_failures"]
    assert diagnostics["calendar_crossings_after"] == 0
    assert metrics.atm_valid.tolist() == [False, True, True]
    assert np.isnan(metrics.atm_iv.iloc[0])


def test_all_failed_svi_expiries_fail_date(monkeypatch):
    def fit(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    metrics, diagnostics = pipeline.process_quote_date(raw_options(), "2025-01-02")

    assert diagnostics["processing_status"] == "failed"
    assert diagnostics["failure_stage"] == "raw_svi"
    assert diagnostics["svi_expiries_failed"] == 2
    assert diagnostics["svi_expiries_fitted"] == 0
    assert np.isnan(diagnostics["calendar_crossings_after"])
    assert not metrics.atm_valid.any()


def test_runner_continues_after_failure_and_resumes_without_duplicates(tmp_path, monkeypatch):
    raw = pd.concat([raw_options(), raw_options("2025-01-03")])
    path = tmp_path / "raw.csv"
    output = tmp_path / "processed"
    raw.to_csv(path, index=False)
    calls = []
    original = pipeline.process_quote_date

    def process(rows, date):
        calls.append(date)
        if date == pd.Timestamp("2025-01-02"):
            rows = rows.copy()
            rows["secid"] = 0
        return original(rows, date)

    monkeypatch.setattr(pipeline, "process_quote_date", process)
    result = pipeline.run_historical_pipeline([path], output, chunksize=19)
    assert result["diagnostics"].processing_status.tolist() == ["failed", "success"]
    assert len(result["metrics"]) == 6
    assert len(calls) == 2
    pipeline.run_historical_pipeline([path], output)
    assert len(calls) == 2

    pipeline.run_historical_pipeline([path], output, retry_failed=True)
    assert len(calls) == 3
    saved = pd.read_csv(output / pipeline.METRICS_FILENAME)
    assert len(saved) == 6
    assert not saved.duplicated(["quote_date", "target_days"]).any()
    assert {p.name for p in output.iterdir()} == {pipeline.METRICS_FILENAME, pipeline.DIAGNOSTICS_FILENAME}

    # Simulate a crash after replacing metrics, before committing diagnostics.
    saved.loc[saved.quote_date.eq("2025-01-03"), "checkpoint_id"] = "uncommitted"
    saved.to_csv(output / pipeline.METRICS_FILENAME, index=False)
    pipeline.run_historical_pipeline([path], output)
    assert len(calls) == 4
    assert calls[-1] == pd.Timestamp("2025-01-03")
