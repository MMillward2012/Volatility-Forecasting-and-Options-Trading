"""Run the existing SPX models one quote date at a time, with CSV checkpoints."""

import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from uuid import uuid4

import numpy as np
import pandas as pd

from src.data_cleaning import clean_option_data
from src.forward_inference import infer_expiry_forwards
from src.iv_panel import build_iv_panel
from src.option_matching import match_calls_and_puts
from src.raw_svi import fit_raw_svi_surface, raw_svi_total_variance
from src.skew_metrics import calculate_skew_metrics
from src.vol_surface import (
    build_total_variance_grid,
    count_calendar_crossings,
    enforce_calendar_monotonicity,
    evaluate_surface,
)


RAW_FILES = tuple(Path(f"data/raw/spx_option_prices_{year}.csv") for year in (2023, 2024, 2025))
METRICS_FILENAME = "spx_skew_metrics_2023_2025.csv"
DIAGNOSTICS_FILENAME = "spx_daily_processing_diagnostics_2023_2025.csv"
TARGET_DAYS = (30, 60, 90)
RESEARCH_MIN_DTE = 14
RESEARCH_MAX_DTE = 180
MAX_BRACKET_GAP_DAYS = 30
SURFACE_GUARD_VERSION = 2
CONSTANTS = {
    "secid": 108105, "ticker": "SPX", "exercise_style": "E",
    "contract_size": 100, "cfadj": 1, "ss_flag": 0, "index_flag": 1,
}
RAW_COLUMNS = list(CONSTANTS) + [
    "date", "exdate", "symbol", "optionid", "cp_flag", "strike_price",
    "best_bid", "best_offer", "volume", "open_interest", "am_settlement",
    "expiry_indicator", "impl_volatility", "delta", "gamma", "vega", "theta",
]
METRICS = {"atm": "atm_iv", "slope": "atm_skew_slope", "rr25": "rr25_downside"}


def iter_quote_dates(raw_files, chunksize=100_000):
    """Yield complete dates from chronologically sorted, non-overlapping CSVs."""
    previous_file_end = None
    for raw_file in raw_files:
        path = Path(raw_file)
        carry = pd.DataFrame()
        previous_chunk_end = None
        with pd.read_csv(
            path, usecols=RAW_COLUMNS, chunksize=chunksize,
            dtype={"expiry_indicator": "string"},
        ) as chunks:
            for chunk in chunks:
                if chunk.empty:
                    continue
                chunk["date"] = pd.to_datetime(chunk["date"], errors="raise")
                dates = chunk["date"]
                if dates.isna().any() or not dates.is_monotonic_increasing:
                    raise ValueError(f"{path}: quote dates must be nonmissing and sorted.")
                if previous_chunk_end is not None and dates.iloc[0] < previous_chunk_end:
                    raise ValueError(f"{path}: quote dates are not sorted across chunks.")
                if previous_file_end is not None and dates.iloc[0] <= previous_file_end:
                    raise ValueError(f"{path}: input files overlap or are out of date order.")
                previous_chunk_end = dates.iloc[-1]
                combined = pd.concat([carry, chunk], ignore_index=True) if not carry.empty else chunk
                final_date = combined["date"].eq(previous_chunk_end)
                for date, rows in combined.loc[~final_date].groupby("date", sort=False):
                    yield path, date, rows
                carry = combined.loc[final_date].copy()
        if not carry.empty:
            yield path, previous_chunk_end, carry
            previous_file_end = previous_chunk_end


def _failed_metrics(quote_date, reason):
    rows = []
    for days in TARGET_DAYS:
        row = {
            "quote_date": quote_date, "target_days": days,
            "bracket_lower_expiry": "", "bracket_upper_expiry": "",
            "bracket_gap_days": np.nan, "bracket_skipped_expiries": 0,
            "bracket_valid": False,
        }
        for column in [
            *METRICS.values(), "k_25_put", "k_25_call", "iv_25_put", "iv_25_call",
            "minimum_call_slope", "maximum_call_slope",
            "k_at_minimum_call_slope", "k_at_maximum_call_slope",
            "raw_atm_iv", "repaired_atm_iv", "calendar_adjustment_atm_iv",
        ]:
            row[column] = np.nan
        for metric in (*METRICS, "surface", "convexity"):
            row[f"{metric}_valid"] = False
        for metric in (*METRICS, "surface"):
            row[f"failure_reason_{metric}"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _calculate_bracketed_metrics(grid, quote_date, listed_maturities):
    """Use nearest accepted fits, with at most 30 calendar days between them."""
    maturities = grid["time_to_expiry"]
    listed_maturities = np.asarray(listed_maturities)
    results = []
    for days in TARGET_DAYS:
        tau = days / 365
        bracket = {
            "bracket_lower_expiry": "", "bracket_upper_expiry": "",
            "bracket_gap_days": np.nan, "bracket_skipped_expiries": 0,
            "bracket_valid": False,
        }
        reason = "outside_maturity_range"
        if maturities[0] <= tau <= maturities[-1]:
            right = np.searchsorted(maturities, tau)
            left = right if maturities[right] == tau else right - 1
            lower, upper = maturities[left], maturities[right]
            gap = int(round((upper - lower) * 365))
            bracket.update(
                bracket_lower_expiry=str((pd.Timestamp(quote_date) + pd.Timedelta(days=round(lower * 365))).date()),
                bracket_upper_expiry=str((pd.Timestamp(quote_date) + pd.Timedelta(days=round(upper * 365))).date()),
                bracket_gap_days=gap,
                bracket_skipped_expiries=int(((listed_maturities > lower) & (listed_maturities < upper)).sum()),
                bracket_valid=gap <= MAX_BRACKET_GAP_DAYS,
            )
            reason = "excessive_maturity_gap"
        if bracket["bracket_valid"]:
            rows = calculate_skew_metrics(grid, quote_date, target_days=(days,))["metrics"]
        else:
            rows = _failed_metrics(quote_date, reason)
            rows = rows.loc[rows["target_days"].eq(days)].copy()
        for column, value in bracket.items():
            rows[column] = value
        results.append(rows)
    return pd.concat(results, ignore_index=True)


def _check_svi_atm(fit):
    """Compare fitted ATM IV with the closest screened quotes straddling k=0."""
    parameters = fit["parameters_by_expiry"].iloc[0]
    quotes = fit["observations"]
    check = {
        "expiry_date": str(pd.Timestamp(parameters["expiry_date"]).date()),
        "days_to_expiry": int(quotes["days_to_expiry"].iloc[0]),
        "svi_atm_iv": np.nan, "local_atm_iv": np.nan,
        "atm_iv_difference": np.nan, "reason": "",
    }
    variance = float(raw_svi_total_variance(
        0.0, *(parameters[c] for c in ["a", "b", "rho", "m", "sigma"])
    ))
    tau = float(parameters["time_to_expiry"])
    if np.isfinite([variance, tau]).all() and variance > 0 and tau > 0:
        check["svi_atm_iv"] = float(np.sqrt(variance / tau))

    below = quotes.loc[quotes["log_moneyness"] < 0]
    above = quotes.loc[quotes["log_moneyness"] > 0]
    if not below.empty and not above.empty:
        left = below.iloc[below["log_moneyness"].argmax()]
        right = above.iloc[above["log_moneyness"].argmin()]
        values = [left["log_moneyness"], right["log_moneyness"], left["mid_iv"], right["mid_iv"]]
        if np.isfinite(values).all():
            weight = -values[0] / (values[1] - values[0])
            check["local_atm_iv"] = float((1 - weight) * values[2] + weight * values[3])

    if not np.isfinite(check["svi_atm_iv"]):
        check["reason"] = "svi_atm_iv_unavailable"
    elif not np.isfinite(check["local_atm_iv"]):
        check["reason"] = "local_atm_iv_unavailable"
    else:
        difference = check["svi_atm_iv"] - check["local_atm_iv"]
        check["atm_iv_difference"] = float(difference)
        if abs(difference) > max(0.03, 0.20 * check["local_atm_iv"]):
            check["reason"] = "svi_atm_iv_mismatch"
    return check


def _apply_calendar_repair_guard(metrics, grid, diagnostics):
    """Keep raw/repaired ATM comparisons and invalidate excessive tenor adjustments."""
    metrics = metrics.copy()
    raw_surface = dict(grid, repaired_total_variance=grid["raw_total_variance"])
    for index, row in metrics.iterrows():
        prefix = f"{int(row['target_days'])}d"
        raw_atm = repaired_atm = np.nan
        try:
            if not row.get("bracket_valid", True):
                raise ValueError("No admissible expiry bracket.")
            tau = row["target_days"] / 365
            raw_atm = evaluate_surface(0.0, tau, raw_surface)["implied_volatility"]
            repaired_atm = evaluate_surface(0.0, tau, grid)["implied_volatility"]
        except ValueError:
            pass  # Existing metric QC records unsupported maturity/strike requests.
        adjustment = repaired_atm - raw_atm
        for column, value in [
            ("raw_atm_iv", raw_atm), ("repaired_atm_iv", repaired_atm),
            ("calendar_adjustment_atm_iv", adjustment),
        ]:
            metrics.loc[index, column] = value
            diagnostics[f"{prefix}_{column}"] = value
        diagnostics[f"{prefix}_calendar_repair_valid"] = bool(
            np.isfinite(adjustment) and abs(adjustment) <= 0.05
        )
        for metric, column in METRICS.items():
            diagnostics[f"{prefix}_calculated_{column}"] = row[column]
            if np.isfinite(adjustment) and abs(adjustment) > 0.05:
                metrics.loc[index, f"{metric}_valid"] = False
                metrics.loc[index, f"failure_reason_{metric}"] = "excessive_calendar_repair"
                metrics.loc[index, column] = np.nan
    return metrics


def process_quote_date(raw_options, quote_date, retain_surface=False):
    """Return three metric rows and diagnostics; a failed date is an explicit result."""
    started = perf_counter()
    date = pd.Timestamp(quote_date).strftime("%Y-%m-%d")
    diagnostics = {
        "quote_date": date, "raw_rows": len(raw_options),
        "am_rows": np.nan, "pm_rows": np.nan, "crossed_pm_rows_dropped": np.nan,
        "cleaned_rows": np.nan, "matched_pairs": np.nan,
        "positive_dte_expiries": np.nan, "forward_expiries": np.nan,
        "research_window_expiries": np.nan,
        "iv_rows": np.nan, "valid_iv_rows": np.nan,
        "svi_expiries_attempted": 0, "svi_expiries_fitted": 0, "svi_expiries_failed": 0,
        "svi_failures": "[]", "calendar_crossings_before": np.nan,
        "svi_expiries_rejected_atm": 0, "svi_atm_rejections": "[]",
        "calendar_crossings_after": np.nan, "max_calendar_adjustment_w": np.nan,
        "max_calendar_adjustment_iv": np.nan,
        "processing_status": "failed", "failure_stage": "", "failure_reason": "",
    }
    stage = "validation"
    grid = None
    forwards = None
    for days in TARGET_DAYS:
        for column in ["raw_atm_iv", "repaired_atm_iv", "calendar_adjustment_atm_iv"]:
            diagnostics[f"{days}d_{column}"] = np.nan
        diagnostics[f"{days}d_calendar_repair_valid"] = False
        for column in METRICS.values():
            diagnostics[f"{days}d_calculated_{column}"] = np.nan
    try:
        missing = set(RAW_COLUMNS) - set(raw_options.columns)
        if missing:
            raise ValueError(f"Missing raw columns: {sorted(missing)}")
        diagnostics["am_rows"] = int(raw_options["am_settlement"].eq(1).sum())
        diagnostics["pm_rows"] = int(raw_options["am_settlement"].eq(0).sum())
        dates = pd.to_datetime(raw_options["date"])
        if raw_options.empty or dates.isna().any() or not dates.eq(pd.Timestamp(date)).all():
            raise ValueError("Input must contain exactly the requested quote date.")
        for column, expected in CONSTANTS.items():
            if not raw_options[column].eq(expected).fillna(False).all():
                raise ValueError(f"{column} must equal {expected!r} on every raw row.")
        if not raw_options["am_settlement"].isin([0, 1]).all():
            raise ValueError("am_settlement must contain only 0 or 1.")

        stage = "cleaning"
        pm = raw_options.loc[raw_options["am_settlement"].eq(0)]
        crossed = pm["best_bid"].gt(pm["best_offer"])
        diagnostics["crossed_pm_rows_dropped"] = int(crossed.sum())
        cleaned = clean_option_data(pm.loc[~crossed])
        diagnostics["cleaned_rows"] = len(cleaned)
        diagnostics["positive_dte_expiries"] = cleaned.loc[
            cleaned["days_to_expiry"].gt(0), "expiry_date"
        ].nunique()
        if not diagnostics["positive_dte_expiries"]:
            raise ValueError("No positive-DTE PM expiries remain.")

        stage = "matching"
        matched = match_calls_and_puts(cleaned)
        diagnostics["matched_pairs"] = len(matched)
        stage = "forward_inference"
        forwards = infer_expiry_forwards(matched)
        diagnostics["forward_expiries"] = len(forwards)
        if forwards.empty:
            raise ValueError("No positive-DTE expiry forwards could be inferred.")
        stage = "iv_panel"
        panel = build_iv_panel(cleaned, forwards)
        diagnostics["iv_rows"] = len(panel)
        diagnostics["valid_iv_rows"] = int(panel["mid_iv"].notna().sum())

        stage = "raw_svi"
        research_panel = panel.loc[
            panel["days_to_expiry"].between(RESEARCH_MIN_DTE, RESEARCH_MAX_DTE)
        ]
        diagnostics["research_window_expiries"] = research_panel["expiry_date"].nunique()
        if research_panel.empty:
            raise ValueError("No expiries remain in the 14-180 DTE research window.")
        parameters = []
        failures = []
        atm_rejections = []
        for expiry, rows in research_panel.groupby("expiry_date", sort=True):
            diagnostics["svi_expiries_attempted"] += 1
            try:
                fit = fit_raw_svi_surface(rows, enforce_arbitrage=True)
                atm_check = _check_svi_atm(fit)
                if atm_check["reason"]:
                    atm_rejections.append(atm_check)
                    raise ValueError(atm_check["reason"])
                parameters.append(fit["parameters_by_expiry"])
                diagnostics["svi_expiries_fitted"] += 1
            except Exception as error:
                failures.append({"expiry_date": str(expiry.date()), "reason": f"{type(error).__name__}: {error}"})
                diagnostics["svi_expiries_failed"] += 1
        diagnostics["svi_failures"] = json.dumps(failures)
        diagnostics["svi_expiries_rejected_atm"] = len(atm_rejections)
        diagnostics["svi_atm_rejections"] = json.dumps([
            {key: None if isinstance(value, float) and not np.isfinite(value) else value
             for key, value in rejection.items()}
            for rejection in atm_rejections
        ], allow_nan=False)
        if not parameters:
            raise ValueError(f"All {len(failures)} SVI expiries failed.")
        if failures:
            diagnostics["failure_stage"] = "raw_svi"
            diagnostics["failure_reason"] = f"{len(failures)} SVI expiries failed; excluded from tenor brackets."

        stage = "calendar_repair"
        grid = build_total_variance_grid(pd.concat(parameters, ignore_index=True))
        raw_w = grid["raw_total_variance"]
        repaired_w = enforce_calendar_monotonicity(raw_w)
        grid["repaired_total_variance"] = repaired_w
        diagnostics["calendar_crossings_before"] = count_calendar_crossings(raw_w)
        diagnostics["calendar_crossings_after"] = count_calendar_crossings(repaired_w)
        diagnostics["max_calendar_adjustment_w"] = float(np.nanmax(np.abs(repaired_w - raw_w)))
        tau = grid["time_to_expiry"][:, None]
        diagnostics["max_calendar_adjustment_iv"] = float(np.nanmax(
            np.abs(np.sqrt(repaired_w / tau) - np.sqrt(raw_w / tau))
        ))

        stage = "skew_metrics"
        metrics = _calculate_bracketed_metrics(
            grid, date, research_panel["time_to_expiry"].unique()
        )
        metrics = _apply_calendar_repair_guard(metrics, grid, diagnostics)
        diagnostics["processing_status"] = "partial" if failures else "success"
    except Exception as error:
        diagnostics["failure_stage"] = stage
        diagnostics["failure_reason"] = f"{type(error).__name__}: {error}"
        metrics = _failed_metrics(date, f"{stage}: {diagnostics['failure_reason']}")

    for row in metrics.to_dict("records"):
        prefix = f"{row['target_days']}d"
        for column in ["bracket_lower_expiry", "bracket_upper_expiry", "bracket_gap_days",
                       "bracket_skipped_expiries", "bracket_valid"]:
            diagnostics[f"{prefix}_{column}"] = row.get(column, np.nan)
        for metric, column in METRICS.items():
            diagnostics[f"{prefix}_{metric}_available"] = bool(np.isfinite(row[column]))
            diagnostics[f"{prefix}_{metric}_valid"] = bool(row[f"{metric}_valid"])
            diagnostics[f"{prefix}_{metric}_failure_reason"] = row[f"failure_reason_{metric}"]
        diagnostics[f"{prefix}_surface_valid"] = bool(row["surface_valid"])
        diagnostics[f"{prefix}_convexity_valid"] = bool(row["convexity_valid"])
    diagnostics["runtime_seconds"] = perf_counter() - started
    metrics["processing_status"] = diagnostics["processing_status"]
    if retain_surface:
        return metrics, diagnostics, {"quote_date": pd.Timestamp(date), "grid": grid,
                                      "forwards": forwards, "metrics": metrics}
    return metrics, diagnostics


def _write_csv_atomic(frame, path):
    temporary_path = None
    try:
        with NamedTemporaryFile(mode="w", dir=path.parent, suffix=".tmp", delete=False) as file:
            temporary_path = Path(file.name)
            frame.to_csv(file, index=False)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def run_historical_pipeline(
    raw_files=RAW_FILES, output_dir="data/processed", chunksize=100_000,
    start_date=None, end_date=None, resume=True, retry_failed=False,
):
    """Checkpoint each date; resume skips complete dates with unchanged source files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / METRICS_FILENAME
    diagnostics_path = output_dir / DIAGNOSTICS_FILENAME
    metrics = pd.read_csv(metrics_path) if resume and metrics_path.exists() else pd.DataFrame()
    diagnostics = pd.read_csv(diagnostics_path) if resume and diagnostics_path.exists() else pd.DataFrame()
    start = pd.Timestamp(start_date) if start_date is not None else None
    end = pd.Timestamp(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise ValueError("start_date must be no later than end_date.")

    for path, quote_date, raw in iter_quote_dates(raw_files, chunksize):
        if end is not None and quote_date > end:
            break
        if start is not None and quote_date < start:
            continue
        date = quote_date.strftime("%Y-%m-%d")
        stat = path.stat()
        signature = {
            "source_file": str(path.resolve()), "source_size_bytes": stat.st_size,
            "source_mtime_ns": str(stat.st_mtime_ns),
            "surface_guard_version": SURFACE_GUARD_VERSION,
        }
        previous = diagnostics.loc[diagnostics["quote_date"].eq(date)] if not diagnostics.empty else pd.DataFrame()
        previous_metrics = metrics.loc[metrics["quote_date"].eq(date)] if not metrics.empty else pd.DataFrame()
        if len(previous) == 1 and len(previous_metrics) == len(TARGET_DAYS):
            old = previous.iloc[0]
            complete = set(previous_metrics["target_days"]) == set(TARGET_DAYS)
            complete = complete and previous_metrics.get(
                "checkpoint_id", pd.Series(dtype=str)
            ).eq(old.get("checkpoint_id", "missing")).sum() == len(TARGET_DAYS)
            unchanged = all(str(old.get(key, "")) == str(value) for key, value in signature.items())
            if complete and unchanged and not (retry_failed and old["processing_status"] != "success"):
                continue

        daily_metrics, daily_diagnostics = process_quote_date(raw, quote_date)
        daily_diagnostics.update(signature)
        checkpoint_id = uuid4().hex
        daily_metrics["checkpoint_id"] = checkpoint_id
        daily_diagnostics["checkpoint_id"] = checkpoint_id
        if not metrics.empty:
            metrics = metrics.loc[~metrics["quote_date"].eq(date)]
        if not diagnostics.empty:
            diagnostics = diagnostics.loc[~diagnostics["quote_date"].eq(date)]
        metrics = pd.concat([metrics, daily_metrics], ignore_index=True).sort_values(["quote_date", "target_days"])
        diagnostics = pd.concat([diagnostics, pd.DataFrame([daily_diagnostics])], ignore_index=True).sort_values("quote_date")
        # Diagnostics is the commit marker: an interruption between writes reruns the date.
        _write_csv_atomic(metrics, metrics_path)
        _write_csv_atomic(diagnostics, diagnostics_path)
        print(f"{date}: {daily_diagnostics['processing_status']} "
              f"({daily_diagnostics['runtime_seconds']:.1f}s) "
              f"{daily_diagnostics['failure_stage']}", flush=True)
    return {"metrics": metrics, "diagnostics": diagnostics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-files", nargs="+", type=Path, default=RAW_FILES)
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--start-date")
    parser.add_argument("--end-date")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    run_historical_pipeline(
        args.raw_files, args.output_dir, args.chunksize, args.start_date, args.end_date,
        resume=not args.no_resume, retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
