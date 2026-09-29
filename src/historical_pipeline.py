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
from src.raw_svi import fit_raw_svi_surface
from src.skew_metrics import calculate_skew_metrics
from src.vol_surface import (
    build_total_variance_grid,
    count_calendar_crossings,
    enforce_calendar_monotonicity,
)


RAW_FILES = tuple(Path(f"data/raw/spx_option_prices_{year}.csv") for year in (2023, 2024, 2025))
METRICS_FILENAME = "spx_skew_metrics_2023_2025.csv"
DIAGNOSTICS_FILENAME = "spx_daily_processing_diagnostics_2023_2025.csv"
TARGET_DAYS = (30, 60, 90)
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
        row = {"quote_date": quote_date, "target_days": days}
        for column in [
            *METRICS.values(), "k_25_put", "k_25_call", "iv_25_put", "iv_25_call",
            "minimum_call_slope", "maximum_call_slope",
            "k_at_minimum_call_slope", "k_at_maximum_call_slope",
        ]:
            row[column] = np.nan
        for metric in (*METRICS, "surface", "convexity"):
            row[f"{metric}_valid"] = False
        for metric in (*METRICS, "surface"):
            row[f"failure_reason_{metric}"] = reason
        rows.append(row)
    return pd.DataFrame(rows)


def _retain_failed_expiries(grid, maturities):
    """Keep unsupported expiry slots so tenor evaluation cannot bridge failed fits."""
    fitted_indices = np.searchsorted(maturities, grid["time_to_expiry"])
    for name in [
        "support_min", "support_max", "sampled_support_min", "sampled_support_max",
        "support_mask", "raw_total_variance",
    ]:
        values = grid[name]
        shape = (len(maturities), *values.shape[1:])
        expanded = np.zeros(shape, dtype=bool) if name == "support_mask" else np.full(shape, np.nan)
        expanded[fitted_indices] = values
        grid[name] = expanded
    grid["time_to_expiry"] = maturities
    return grid


def process_quote_date(raw_options, quote_date):
    """Return three metric rows and diagnostics; a failed date is an explicit result."""
    started = perf_counter()
    date = pd.Timestamp(quote_date).strftime("%Y-%m-%d")
    diagnostics = {
        "quote_date": date, "raw_rows": len(raw_options),
        "am_rows": np.nan, "pm_rows": np.nan, "crossed_pm_rows_dropped": np.nan,
        "cleaned_rows": np.nan, "matched_pairs": np.nan,
        "positive_dte_expiries": np.nan, "forward_expiries": np.nan,
        "iv_rows": np.nan, "valid_iv_rows": np.nan,
        "svi_expiries_attempted": 0, "svi_expiries_fitted": 0, "svi_expiries_failed": 0,
        "svi_failures": "[]", "calendar_crossings_before": np.nan,
        "calendar_crossings_after": np.nan, "max_calendar_adjustment_w": np.nan,
        "max_calendar_adjustment_iv": np.nan,
        "processing_status": "failed", "failure_stage": "", "failure_reason": "",
    }
    stage = "validation"
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
        parameters = []
        failures = []
        for expiry, rows in panel.groupby("expiry_date", sort=True):
            diagnostics["svi_expiries_attempted"] += 1
            try:
                fit = fit_raw_svi_surface(rows, enforce_arbitrage=True)
                parameters.append(fit["parameters_by_expiry"])
                diagnostics["svi_expiries_fitted"] += 1
            except Exception as error:
                failures.append({"expiry_date": str(expiry.date()), "reason": f"{type(error).__name__}: {error}"})
                diagnostics["svi_expiries_failed"] += 1
        diagnostics["svi_failures"] = json.dumps(failures)
        if not parameters:
            raise ValueError(f"All {len(failures)} SVI expiries failed.")
        if failures:
            diagnostics["failure_stage"] = "raw_svi"
            diagnostics["failure_reason"] = f"{len(failures)} SVI expiries failed; retained as unsupported slots."

        stage = "calendar_repair"
        grid = build_total_variance_grid(pd.concat(parameters, ignore_index=True))
        if failures:
            grid = _retain_failed_expiries(grid, np.sort(panel["time_to_expiry"].unique()))
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
        metrics = calculate_skew_metrics(grid, date, target_days=TARGET_DAYS)["metrics"]
        diagnostics["processing_status"] = "partial" if failures else "success"
    except Exception as error:
        diagnostics["failure_stage"] = stage
        diagnostics["failure_reason"] = f"{type(error).__name__}: {error}"
        metrics = _failed_metrics(date, f"{stage}: {diagnostics['failure_reason']}")

    for row in metrics.to_dict("records"):
        prefix = f"{row['target_days']}d"
        for metric, column in METRICS.items():
            diagnostics[f"{prefix}_{metric}_available"] = bool(np.isfinite(row[column]))
            diagnostics[f"{prefix}_{metric}_valid"] = bool(row[f"{metric}_valid"])
            diagnostics[f"{prefix}_{metric}_failure_reason"] = row[f"failure_reason_{metric}"]
        diagnostics[f"{prefix}_surface_valid"] = bool(row["surface_valid"])
        diagnostics[f"{prefix}_convexity_valid"] = bool(row["convexity_valid"])
    diagnostics["runtime_seconds"] = perf_counter() - started
    metrics["processing_status"] = diagnostics["processing_status"]
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
