import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def raw_svi_total_variance(k, a, b, rho, m, sigma):
    """Evaluate raw SVI total variance for log-moneyness k."""
    shifted_k = np.asarray(k) - m
    return a + b * (rho * shifted_k + np.sqrt(shifted_k**2 + sigma**2))


def raw_svi_butterfly_diagnostic(a, b, rho, m, sigma, k_grid=None):
    """Check positivity, Durrleman g(k), and the right-wing slope on a grid."""
    if b < 0 or abs(rho) >= 1 or sigma <= 0:
        raise ValueError("Raw SVI parameters need b >= 0, |rho| < 1, and sigma > 0.")
    if k_grid is None:
        k_grid = np.linspace(-5.0, 5.0, 4001)

    k = np.asarray(k_grid, dtype=float)
    x = k - m
    w = raw_svi_total_variance(k, a, b, rho, m, sigma)
    w_k = b * (rho + x / np.sqrt(x**2 + sigma**2))
    w_kk = b * sigma**2 / (x**2 + sigma**2) ** 1.5
    if (w <= 0).any():
        g = np.full_like(w, np.nan)
    else:
        g = (
            (1 - k * w_k / (2 * w)) ** 2
            - (w_k**2 / 4) * (1 / w + 1 / 4)
            + w_kk / 2
        )

    minimum_w = a + b * sigma * np.sqrt(1 - rho**2)
    minimum_g_index = int(np.nanargmin(g)) if np.isfinite(g).any() else None
    minimum_g = float(g[minimum_g_index]) if minimum_g_index is not None else np.nan
    right_wing_slope = b * (1 + rho)
    return {
        "minimum_total_variance": float(minimum_w),
        "minimum_g_on_grid": minimum_g,
        "k_at_minimum_g": float(k[minimum_g_index]) if minimum_g_index is not None else np.nan,
        "right_wing_slope": float(right_wing_slope),
        "nonnegative_total_variance": bool(minimum_w >= 0),
        "g_nonnegative_on_grid": bool(minimum_g >= -1e-10),
        "right_wing_condition": bool(right_wing_slope < 2),
    }


def fit_raw_svi_surface(panel, apply_quote_screen=True, initial_guess=None):
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
        if initial_guess is None:
            x0 = [max(float(market_w.min()) - 0.01, -1.0), 0.1, -0.3, 0.0, 0.1]
        else:
            x0 = np.asarray(initial_guess, dtype=float)
            if x0.shape != (5,) or not np.isfinite(x0).all():
                raise ValueError("initial_guess must contain five finite Raw SVI parameters.")

        fit = least_squares(
            lambda params: raw_svi_total_variance(k, *params) - market_w,
            x0=x0,
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
