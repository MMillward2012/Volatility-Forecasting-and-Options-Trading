import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def raw_svi_total_variance(k, a, b, rho, m, sigma):
    """Evaluate raw SVI total variance for log-moneyness k."""
    shifted_k = np.asarray(k) - m
    return a + b * (rho * shifted_k + np.sqrt(shifted_k**2 + sigma**2))


def fit_raw_svi_surface(panel, apply_quote_screen=True):
    """Fit an independent five-parameter raw SVI slice for each expiry."""
    if panel.empty or panel["quote_date"].nunique() != 1:
        raise ValueError("panel must contain one nonempty quote date.")
    if panel["security_id"].nunique() != 1:
        raise ValueError("panel must contain one security_id.")

    use_for_fit = panel["use_for_surface"].fillna(False)
    if apply_quote_screen:
        quote_columns = [
            "best_bid", "best_ask", "mid_price", "mid_iv", "relative_spread"
        ]
        finite_quotes = np.isfinite(
            panel[quote_columns].to_numpy(dtype=float)
        ).all(axis=1)
        use_for_fit = (
            use_for_fit
            & finite_quotes
            & panel["best_bid"].gt(0)
            & panel["best_ask"].gt(panel["best_bid"])
            & panel["mid_price"].gt(0)
            & panel["relative_spread"].between(0, 0.50, inclusive="right")
        )

    rows = panel.loc[use_for_fit].copy()
    if rows.empty:
        raise ValueError("No OTM quotes are available for raw SVI fitting.")
    for column in ["log_moneyness", "mid_iv", "time_to_expiry"]:
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for fitted quotes.")
    if not (rows["mid_iv"] > 0).all() or not (rows["time_to_expiry"] > 0).all():
        raise ValueError("Fitted quotes need positive IV and time to expiry.")

    parameter_rows = []
    fitted_slices = []
    for expiry, expiry_rows in rows.groupby("expiry_date", sort=True):
        if len(expiry_rows) < 5:
            raise ValueError(f"Expiry {expiry} needs at least five quotes to fit raw SVI.")

        k = expiry_rows["log_moneyness"].to_numpy(dtype=float)
        market_w = (
            expiry_rows["mid_iv"].to_numpy(dtype=float) ** 2
            * expiry_rows["time_to_expiry"].to_numpy(dtype=float)
        )
        fit = least_squares(
            lambda params: raw_svi_total_variance(k, *params) - market_w,
            x0=[max(float(market_w.min()) - 0.01, -1.0), 0.1, -0.3, 0.0, 0.1],
            bounds=(
                [-1.0, 1e-8, -0.999, -5.0, 1e-8],
                [1.0, 10.0, 0.999, 5.0, 5.0],
            ),
            max_nfev=10000,
        )
        if not fit.success:
            raise ValueError(f"Raw SVI calibration failed for {expiry}: {fit.message}")

        a, b, rho, m, sigma = fit.x
        expiry_rows = expiry_rows.copy()
        expiry_rows["market_w"] = market_w
        expiry_rows["fitted_w"] = raw_svi_total_variance(k, *fit.x)
        expiry_rows["residual_w"] = expiry_rows["fitted_w"] - market_w
        fitted_slices.append(expiry_rows)
        parameter_rows.append(
            {
                "quote_date": expiry_rows["quote_date"].iloc[0],
                "expiry_date": expiry,
                "a": a,
                "b": b,
                "rho": rho,
                "m": m,
                "sigma": sigma,
                "rmse": float(np.sqrt(np.mean(expiry_rows["residual_w"] ** 2))),
            }
        )

    return {
        "parameters_by_expiry": pd.DataFrame(parameter_rows),
        "observations": pd.concat(fitted_slices, ignore_index=True),
    }
