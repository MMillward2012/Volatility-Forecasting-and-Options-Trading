import numpy as np
import pandas as pd

from src.vol_surface import check_surface_quality, evaluate_surface


def calculate_skew_metrics(
    surface_grid,
    quote_date,
    target_days=(30, 60, 90),
    skew_moneyness=0.10,
    atm_slope_step=0.01,
    smile_points=151,
):
    """Calculate target-tenor IV smiles and fixed-k skew with surface quality flags."""
    if skew_moneyness <= 0 or not np.isfinite(skew_moneyness):
        raise ValueError("skew_moneyness must be finite and positive.")
    if atm_slope_step <= 0 or atm_slope_step > skew_moneyness:
        raise ValueError("atm_slope_step must be positive and no greater than skew_moneyness.")
    if smile_points < 2:
        raise ValueError("smile_points must be at least two.")

    maturities = np.asarray(surface_grid["time_to_expiry"], dtype=float)
    support_min = np.asarray(surface_grid["sampled_support_min"], dtype=float)
    support_max = np.asarray(surface_grid["sampled_support_max"], dtype=float)
    k_grid = np.asarray(surface_grid["log_moneyness"], dtype=float)
    metrics = []
    smiles = []

    for days in target_days:
        tau = float(days) / 365
        quality = check_surface_quality(
            tau,
            surface_grid,
            k_range=(-skew_moneyness, skew_moneyness),
        )
        row = {
            "quote_date": quote_date,
            "target_days": int(days),
            "atm_iv": np.nan,
            "fixed_k_skew": np.nan,
            "atm_slope": np.nan,
            "surface_valid": quality["use_for_skew"],
            "convexity_valid": (
                np.isfinite(quality["convexity_violations"])
                and quality["convexity_violations"] == 0
            ),
            "failure_reason": quality["failure_reason"],
        }
        metrics.append(row)

        if tau < maturities[0] or tau > maturities[-1]:
            continue
        right = np.searchsorted(maturities, tau)
        left = right if maturities[right] == tau else right - 1
        lower = np.max(support_min[left:right + 1])
        upper = np.min(support_max[left:right + 1])
        if not np.isfinite([lower, upper]).all() or lower >= upper:
            continue

        smile_k = np.linspace(lower, upper, smile_points)
        smile_iv = evaluate_surface(smile_k, tau, surface_grid)["implied_volatility"]
        smiles.extend(
            {
                "quote_date": quote_date,
                "target_days": int(days),
                "log_moneyness": float(k),
                "implied_volatility": float(iv),
                "surface_valid": row["surface_valid"],
                "convexity_valid": row["convexity_valid"],
                "failure_reason": row["failure_reason"],
            }
            for k, iv in zip(smile_k, smile_iv)
        )

        if row["surface_valid"]:
            sample_k = np.array(
                [-skew_moneyness, -atm_slope_step, 0.0, atm_slope_step, skew_moneyness]
            )
            sample_iv = evaluate_surface(sample_k, tau, surface_grid)["implied_volatility"]
            row["atm_iv"] = float(sample_iv[2])
            row["fixed_k_skew"] = float(sample_iv[0] - sample_iv[4])
            row["atm_slope"] = float(
                (sample_iv[3] - sample_iv[1]) / (2 * atm_slope_step)
            )

    return {
        "metrics": pd.DataFrame(metrics),
        "smiles": pd.DataFrame(smiles),
    }
