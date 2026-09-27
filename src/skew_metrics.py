import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from src.vol_surface import check_surface_quality, evaluate_surface


def _forward_delta(k, time_to_expiry, surface_grid, option_type):
    """Return unadjusted Black forward delta at log-forward-moneyness k."""
    variance = evaluate_surface(k, time_to_expiry, surface_grid)["total_variance"]
    d1 = (-k + 0.5 * variance) / np.sqrt(variance)
    return norm.cdf(d1 if option_type == "call" else -d1)


def _find_delta_strike(surface_grid, time_to_expiry, delta, option_type, lower, upper):
    """Find the unique supported OTM strike for the requested forward delta."""
    k_grid = np.linspace(lower, upper, 501)
    deltas = np.array([
        _forward_delta(k, time_to_expiry, surface_grid, option_type) for k in k_grid
    ])
    errors = deltas - delta
    brackets = [
        (k_grid[i], k_grid[i + 1])
        for i in range(len(k_grid) - 1)
        if errors[i] * errors[i + 1] < 0
    ]
    exact_roots = k_grid[errors == 0].tolist()
    if len(brackets) + len(exact_roots) != 1:
        raise ValueError(f"no_unique_25_delta_{option_type}")
    if exact_roots:
        return float(exact_roots[0])
    return float(
        brentq(
            lambda k: _forward_delta(k, time_to_expiry, surface_grid, option_type) - delta,
            *brackets[0],
        )
    )


def calculate_skew_metrics(
    surface_grid,
    quote_date,
    target_days=(30, 60, 90),
    rr_delta=0.25,
    atm_slope_step=0.01,
    smile_points=151,
):
    """Extract target-tenor smiles, ATM IV/slope, and 25-delta downside risk reversal."""
    if not 0 < rr_delta < 0.5:
        raise ValueError("rr_delta must be between zero and one half.")
    if not np.isfinite(atm_slope_step) or atm_slope_step <= 0:
        raise ValueError("atm_slope_step must be finite and positive.")
    if smile_points < 2:
        raise ValueError("smile_points must be at least two.")

    maturities = np.asarray(surface_grid["time_to_expiry"], dtype=float)
    support_min = np.asarray(surface_grid["sampled_support_min"], dtype=float)
    support_max = np.asarray(surface_grid["sampled_support_max"], dtype=float)
    metrics = []
    smiles = []

    for days in target_days:
        tau = float(days) / 365
        if not np.isfinite(tau) or tau <= 0:
            raise ValueError("target_days must contain finite positive maturities.")
        row = {
            "quote_date": quote_date,
            "target_days": int(days),
            "atm_iv": np.nan,
            "atm_skew_slope": np.nan,
            "rr25_downside": np.nan,
            "k_25_put": np.nan,
            "k_25_call": np.nan,
            "iv_25_put": np.nan,
            "iv_25_call": np.nan,
            "atm_valid": False,
            "slope_valid": False,
            "rr25_valid": False,
            "surface_valid": False,
            "convexity_valid": False,
            "failure_reason_atm": "",
            "failure_reason_slope": "",
            "failure_reason_rr25": "",
            "failure_reason_surface": "",
            "minimum_call_slope": np.nan,
            "maximum_call_slope": np.nan,
            "k_at_minimum_call_slope": np.nan,
            "k_at_maximum_call_slope": np.nan,
        }

        if tau < maturities[0] or tau > maturities[-1]:
            for metric in ("atm", "slope", "rr25", "surface"):
                row[f"failure_reason_{metric}"] = "outside_maturity_range"
            metrics.append(row)
            continue

        right = np.searchsorted(maturities, tau)
        left = right if maturities[right] == tau else right - 1
        lower = np.max(support_min[left:right + 1])
        upper = np.min(support_max[left:right + 1])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            for metric in ("atm", "slope", "rr25", "surface"):
                row[f"failure_reason_{metric}"] = "insufficient_shared_support"
            metrics.append(row)
            continue

        smile_k = np.linspace(lower, upper, smile_points)
        smile_iv = evaluate_surface(smile_k, tau, surface_grid)["implied_volatility"]
        full_quality = check_surface_quality(tau, surface_grid)
        row.update(
            surface_valid=full_quality["use_for_skew"],
            convexity_valid=(
                np.isfinite(full_quality["convexity_violations"])
                and full_quality["convexity_violations"] == 0
            ),
            failure_reason_surface=full_quality["failure_reason"],
            minimum_call_slope=full_quality.get("minimum_call_slope", np.nan),
            maximum_call_slope=full_quality.get("maximum_call_slope", np.nan),
            k_at_minimum_call_slope=full_quality.get("k_at_minimum_call_slope", np.nan),
            k_at_maximum_call_slope=full_quality.get("k_at_maximum_call_slope", np.nan),
        )

        if lower <= 0 <= upper:
            atm = evaluate_surface(0.0, tau, surface_grid)["implied_volatility"]
            if np.isfinite(atm) and atm > 0:
                row.update(atm_iv=float(atm), atm_valid=True)
            else:
                row["failure_reason_atm"] = "invalid_atm_iv"
        else:
            row["failure_reason_atm"] = "atm_unsupported"

        if lower <= -atm_slope_step and upper >= atm_slope_step:
            local_quality = check_surface_quality(
                tau,
                surface_grid,
                k_range=(-atm_slope_step, atm_slope_step),
            )
            local_iv = evaluate_surface(
                np.array([-atm_slope_step, atm_slope_step]), tau, surface_grid
            )["implied_volatility"]
            if local_quality["use_for_skew"] and np.isfinite(local_iv).all():
                row.update(
                    atm_skew_slope=float(
                        -(local_iv[1] - local_iv[0]) / (2 * atm_slope_step)
                    ),
                    slope_valid=True,
                )
            else:
                row["failure_reason_slope"] = local_quality["failure_reason"]
        else:
            row["failure_reason_slope"] = "atm_slope_points_unsupported"

        try:
            put_upper = min(upper, 0.0)
            call_lower = max(lower, 0.0)
            if lower >= put_upper or call_lower >= upper:
                raise ValueError("25_delta_points_unsupported")
            k_put = _find_delta_strike(
                surface_grid, tau, rr_delta, "put", lower, put_upper
            )
            k_call = _find_delta_strike(
                surface_grid, tau, rr_delta, "call", call_lower, upper
            )
            rr_quality = check_surface_quality(
                tau, surface_grid, k_range=(k_put, k_call)
            )
            iv_put = evaluate_surface(k_put, tau, surface_grid)["implied_volatility"]
            iv_call = evaluate_surface(k_call, tau, surface_grid)["implied_volatility"]
            row.update(k_25_put=k_put, k_25_call=k_call, iv_25_put=iv_put, iv_25_call=iv_call)
            if rr_quality["use_for_skew"] and np.isfinite([iv_put, iv_call]).all():
                row.update(rr25_downside=float(iv_put - iv_call), rr25_valid=True)
            else:
                row["failure_reason_rr25"] = rr_quality["failure_reason"]
        except ValueError as error:
            row["failure_reason_rr25"] = str(error)

        metrics.append(row)
        smiles.extend(
            {
                "quote_date": quote_date,
                "target_days": int(days),
                "log_moneyness": float(k),
                "implied_volatility": float(iv),
                "surface_valid": row["surface_valid"],
                "convexity_valid": row["convexity_valid"],
            }
            for k, iv in zip(smile_k, smile_iv)
        )

    return {"metrics": pd.DataFrame(metrics), "smiles": pd.DataFrame(smiles)}
