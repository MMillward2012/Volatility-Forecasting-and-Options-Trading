import json

import numpy as np
import pandas as pd
import pytest

import src.historical_pipeline as pipeline
from src.pricing import black_scholes_call_price, black_scholes_put_price


def raw_options(date="2025-01-02", expiry_days=(20, 40, 70, 100)):
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
    assert diagnostics["matched_pairs"] == 52
    assert diagnostics["positive_dte_expiries"] == diagnostics["svi_expiries_fitted"] == 4
    assert diagnostics["svi_expiries_failed"] == 0
    assert diagnostics["calendar_crossings_after"] == 0
    assert diagnostics["30d_atm_available"]


def test_failed_edge_expiry_does_not_allow_extrapolation(monkeypatch):
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
    assert metrics.atm_valid.tolist() == [False, True, False]
    assert metrics.failure_reason_atm.iloc[2] == "excessive_maturity_gap"
    assert np.isnan(metrics.atm_iv.iloc[0])


def test_all_failed_svi_expiries_fail_date(monkeypatch):
    def fit(*args, **kwargs):
        raise ValueError("synthetic failure")

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    metrics, diagnostics = pipeline.process_quote_date(raw_options(), "2025-01-02")

    assert diagnostics["processing_status"] == "failed"
    assert diagnostics["failure_stage"] == "raw_svi"
    assert diagnostics["svi_expiries_failed"] == 4
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


def atm_fit(local_iv=0.2, svi_iv=0.2):
    return {
        "parameters_by_expiry": pd.DataFrame([{
            "expiry_date": pd.Timestamp("2025-01-22"), "time_to_expiry": 20 / 365,
            "a": svi_iv**2 * 20 / 365, "b": 0.0, "rho": 0.0, "m": 0.0, "sigma": 0.1,
        }]),
        "observations": pd.DataFrame({
            "days_to_expiry": [20, 20], "log_moneyness": [-0.01, 0.01],
            "mid_iv": [local_iv, local_iv],
        }),
    }


def test_atm_benchmark_interpolates_closest_screened_quotes():
    fit = atm_fit(svi_iv=0.24)
    fit["observations"] = pd.DataFrame({
        "days_to_expiry": [20] * 5,
        "log_moneyness": [-0.1, -0.02, 0.0, 0.01, 0.1],
        "mid_iv": [2.0, 0.2, 3.0, 0.26, 2.0],
    })
    check = pipeline._check_svi_atm(fit)
    assert check["local_atm_iv"] == pytest.approx(0.24)
    assert check["atm_iv_difference"] == pytest.approx(0.0)
    assert not check["reason"]


@pytest.mark.parametrize("local,svi,rejected", [
    (0.1, 0.1299, False), (0.1, 0.1301, True),
    (0.4, 0.4799, False), (0.4, 0.4801, True), (0.4, 0.3199, True),
    (0.3125, 0.375, False),
])
def test_atm_sanity_uses_absolute_and_relative_tolerances(local, svi, rejected):
    check = pipeline._check_svi_atm(atm_fit(local, svi))
    assert (check["reason"] == "svi_atm_iv_mismatch") == rejected


@pytest.mark.parametrize("case,reason", [
    ("one_sided", "local_atm_iv_unavailable"),
    ("missing_quote_iv", "local_atm_iv_unavailable"),
    ("nonfinite_fit", "svi_atm_iv_unavailable"),
])
def test_atm_sanity_rejects_unavailable_benchmarks(case, reason):
    fit = atm_fit()
    if case == "one_sided":
        fit["observations"]["log_moneyness"] = [0.01, 0.02]
    elif case == "missing_quote_iv":
        fit["observations"].loc[0, "mid_iv"] = np.nan
    else:
        fit["parameters_by_expiry"].loc[0, "a"] = np.nan
    assert pipeline._check_svi_atm(fit)["reason"] == reason


def test_research_window_is_inclusive_and_leaves_full_chain_counts(monkeypatch):
    original = pipeline.fit_raw_svi_surface
    attempted = []

    def fit(rows, **kwargs):
        attempted.append(int(rows.days_to_expiry.iloc[0]))
        return original(rows, **kwargs)

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    _, diagnostics = pipeline.process_quote_date(
        raw_options(expiry_days=(13, 14, 180, 181)), "2025-01-02"
    )
    assert attempted == [14, 180]
    assert diagnostics["positive_dte_expiries"] == diagnostics["forward_expiries"] == 4
    assert diagnostics["research_window_expiries"] == diagnostics["svi_expiries_attempted"] == 2


def test_atm_rejected_slice_is_skipped_within_gap_limit(monkeypatch):
    original = pipeline.fit_raw_svi_surface

    def fit(rows, **kwargs):
        result = original(rows, **kwargs)
        if rows.days_to_expiry.iloc[0] == 30:
            result["parameters_by_expiry"].loc[:, "a"] = 1.0
        return result

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    metrics, diagnostics = pipeline.process_quote_date(
        raw_options(expiry_days=(20, 30, 40, 60, 90)), "2025-01-02"
    )
    assert diagnostics["processing_status"] == "partial"
    assert diagnostics["svi_expiries_rejected_atm"] == diagnostics["svi_expiries_failed"] == 1
    assert metrics.atm_valid.all()
    assert metrics.bracket_gap_days.tolist() == [20, 0, 0]
    assert metrics.bracket_skipped_expiries.tolist() == [1, 0, 0]
    assert diagnostics["30d_bracket_lower_expiry"] == "2025-01-22"
    assert diagnostics["30d_bracket_upper_expiry"] == "2025-02-11"
    np.testing.assert_allclose(metrics.atm_iv, 0.2, atol=1e-4)
    rejected = json.loads(diagnostics["svi_atm_rejections"])[0]
    assert rejected["expiry_date"] == "2025-02-01"
    assert rejected["days_to_expiry"] == 30
    assert rejected["local_atm_iv"] == pytest.approx(0.2)
    assert rejected["svi_atm_iv"] > 1
    assert rejected["atm_iv_difference"] > 1
    assert rejected["reason"] == "svi_atm_iv_mismatch"


@pytest.mark.parametrize("repaired_first", [0.1, 0.3])
def test_repair_guard_invalidates_only_affected_tenor_and_preserves_diagnostics(repaired_first):
    tau = np.array([30, 60, 90]) / 365
    k = np.linspace(-0.3, 0.3, 121)
    raw_iv = np.array([0.2, 0.2, 0.2])
    repaired_iv = np.array([repaired_first, 0.249, 0.2])
    grid = {
        "time_to_expiry": tau, "log_moneyness": k,
        "sampled_support_min": np.full(3, -0.3), "sampled_support_max": np.full(3, 0.3),
        "raw_total_variance": (tau * raw_iv**2)[:, None] * np.ones_like(k),
        "repaired_total_variance": (tau * repaired_iv**2)[:, None] * np.ones_like(k),
    }
    metrics = pipeline.calculate_skew_metrics(grid, "2025-01-02")["metrics"]
    diagnostics = {}
    guarded = pipeline._apply_calendar_repair_guard(metrics, grid, diagnostics)
    assert guarded.atm_valid.tolist() == [False, True, True]
    for name, column in pipeline.METRICS.items():
        assert not guarded.loc[0, f"{name}_valid"]
        assert guarded.loc[0, f"failure_reason_{name}"] == "excessive_calendar_repair"
        assert np.isnan(guarded.loc[0, column])
        assert diagnostics[f"30d_calculated_{column}"] == metrics.loc[0, column]
    assert guarded.loc[0, "raw_atm_iv"] == pytest.approx(0.2)
    assert guarded.loc[0, "repaired_atm_iv"] == pytest.approx(repaired_first)
    assert guarded.loc[0, "calendar_adjustment_atm_iv"] == pytest.approx(repaired_first - 0.2)
    np.testing.assert_allclose(guarded.atm_skew_slope.iloc[1:], metrics.atm_skew_slope.iloc[1:])


@pytest.mark.parametrize("old_version", [None, 1])
def test_resume_reprocesses_checkpoints_from_before_guards(tmp_path, monkeypatch, old_version):
    path = tmp_path / "raw.csv"
    output = tmp_path / "processed"
    raw_options().to_csv(path, index=False)
    pipeline.run_historical_pipeline([path], output)
    diagnostics_path = output / pipeline.DIAGNOSTICS_FILENAME
    old = pd.read_csv(diagnostics_path)
    if old_version is None:
        old = old.drop(columns="surface_guard_version")
    else:
        old["surface_guard_version"] = old_version
    old.to_csv(diagnostics_path, index=False)
    calls = []
    original = pipeline.process_quote_date

    def process(rows, date):
        calls.append(date)
        return original(rows, date)

    monkeypatch.setattr(pipeline, "process_quote_date", process)
    result = pipeline.run_historical_pipeline([path], output)
    assert len(calls) == 1
    assert len(result["metrics"]) == 3
    assert result["diagnostics"].surface_guard_version.iloc[0] == pipeline.SURFACE_GUARD_VERSION


@pytest.mark.parametrize("far_days,valid", [(50, True), (51, False)])
def test_bracket_gap_boundary_and_unchanged_accepted_metrics(far_days, valid):
    tau = np.array([20, far_days, 90]) / 365
    k = np.linspace(-0.2, 0.2, 81)
    grid = {
        "time_to_expiry": tau, "log_moneyness": k,
        "sampled_support_min": np.full(3, -0.2),
        "sampled_support_max": np.full(3, 0.2),
        "repaired_total_variance": (tau * 0.2**2)[:, None] * np.ones_like(k),
    }
    grid["raw_total_variance"] = grid["repaired_total_variance"].copy()
    metrics = pipeline._calculate_bracketed_metrics(grid, "2025-01-02", tau)
    assert bool(metrics.loc[0, "bracket_valid"]) == valid
    assert metrics.loc[0, "bracket_gap_days"] == far_days - 20
    if valid:
        original = pipeline.calculate_skew_metrics(grid, "2025-01-02")["metrics"]
        np.testing.assert_allclose(
            metrics.loc[0, list(pipeline.METRICS.values())].to_numpy(dtype=float),
            original.loc[0, list(pipeline.METRICS.values())].to_numpy(dtype=float),
        )
    else:
        guarded = pipeline._apply_calendar_repair_guard(metrics, grid, {})
        for metric, column in pipeline.METRICS.items():
            assert not guarded.loc[0, f"{metric}_valid"]
            assert np.isnan(guarded.loc[0, column])
            assert guarded.loc[0, f"failure_reason_{metric}"] == "excessive_maturity_gap"
        assert np.isnan(guarded.loc[0, "raw_atm_iv"])
    assert metrics.loc[2, "atm_valid"]
    assert metrics.loc[2, "bracket_gap_days"] == 0


def test_skipping_calibration_failure_preserves_strike_support_checks(monkeypatch):
    original = pipeline.fit_raw_svi_surface

    def fit(rows, **kwargs):
        if rows.days_to_expiry.iloc[0] == 30:
            raise ValueError("synthetic convergence failure")
        result = original(rows, **kwargs)
        result["parameters_by_expiry"]["k_max"] = 0.0
        return result

    monkeypatch.setattr(pipeline, "fit_raw_svi_surface", fit)
    metrics, diagnostics = pipeline.process_quote_date(
        raw_options(expiry_days=(20, 30, 40, 60, 90)), "2025-01-02"
    )
    assert diagnostics["processing_status"] == "partial"
    assert metrics.loc[0, "bracket_skipped_expiries"] == 1
    assert metrics.loc[0, "atm_valid"]
    assert not metrics.loc[0, "slope_valid"]
    assert not metrics.loc[0, "rr25_valid"]
    assert metrics.loc[0, "failure_reason_slope"] == "atm_slope_points_unsupported"
    assert metrics.loc[0, "failure_reason_rr25"] == "25_delta_points_unsupported"
