import numpy as np
import pandas as pd
from scipy.optimize import least_squares


def ssvi_phi(theta, eta, gamma):
    """Return the SSVI curvature parameter for ATM total variance theta."""
    return eta * np.asarray(theta) ** (-gamma)


def ssvi_total_variance(k, theta, rho, eta, gamma):
    """Evaluate SSVI total variance at log-moneyness k."""
    phi = ssvi_phi(theta, eta, gamma)
    return 0.5 * theta * (
        1 + rho * phi * k + np.sqrt((phi * k + rho) ** 2 + 1 - rho**2)
    )


def estimate_atm_theta(expiry_rows):
    """Use the filtered quote nearest the forward as the ATM variance proxy."""
    if expiry_rows.empty:
        raise ValueError("Each expiry needs at least one filtered quote.")

    atm_row = expiry_rows.iloc[expiry_rows["log_moneyness"].abs().argmin()]
    theta = atm_row["mid_iv"] ** 2 * atm_row["time_to_expiry"]
    if not np.isfinite(theta) or theta <= 0:
        raise ValueError("ATM total variance must be finite and positive.")
    return float(theta)


def fit_ssvi_surface(panel):
    """Fit one SSVI surface to the screened OTM quotes for one date/security."""
    if panel.empty or panel["quote_date"].nunique() != 1:
        raise ValueError("panel must contain one nonempty quote date.")
    if panel["security_id"].nunique() != 1:
        raise ValueError("panel must contain one security_id.")

    quote_columns = ["best_bid", "best_ask", "mid_price", "mid_iv", "relative_spread"]
    finite_quotes = np.isfinite(
        panel[quote_columns].to_numpy(dtype=float)
    ).all(axis=1)
    use_for_fit = (
        panel["use_for_surface"].fillna(False)
        & finite_quotes
        & panel["best_bid"].gt(0)
        & panel["best_ask"].gt(panel["best_bid"])
        & panel["mid_price"].gt(0)
        & panel["relative_spread"].between(0, 0.50, inclusive="right")
    )
    rows = panel.loc[use_for_fit].copy()
    if rows.empty:
        raise ValueError("No quotes pass the SSVI fitting screen.")
    for column in ["log_moneyness", "mid_iv", "time_to_expiry"]:
        values = rows[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite for filtered quotes.")
    if not (rows["mid_iv"] > 0).all() or not (rows["time_to_expiry"] > 0).all():
        raise ValueError("Filtered quotes need positive IV and time to expiry.")

    theta_by_expiry = pd.DataFrame(
        [
            {
                "quote_date": group["quote_date"].iloc[0],
                "expiry_date": expiry,
                "time_to_expiry": group["time_to_expiry"].iloc[0],
                "theta_atm": estimate_atm_theta(group),
            }
            for expiry, group in rows.groupby("expiry_date", sort=True)
        ]
    )
    rows = rows.merge(
        theta_by_expiry[["expiry_date", "theta_atm"]],
        on="expiry_date",
        how="left",
        validate="many_to_one",
    )
    rows["market_w"] = rows["mid_iv"] ** 2 * rows["time_to_expiry"]

    k = rows["log_moneyness"].to_numpy(dtype=float)
    theta = rows["theta_atm"].to_numpy(dtype=float)
    market_w = rows["market_w"].to_numpy(dtype=float)
    fit = least_squares(
        lambda params: ssvi_total_variance(k, theta, *params) - market_w,
        x0=[-0.5, 1.0, 0.5],
        bounds=([-0.999, 1e-8, 0.0], [0.999, 100.0, 1.0]),
    )
    if not fit.success:
        raise ValueError(f"SSVI calibration failed: {fit.message}")

    rho, eta, gamma = fit.x
    rows["fitted_w"] = ssvi_total_variance(k, theta, rho, eta, gamma)
    rows["residual_w"] = rows["fitted_w"] - rows["market_w"]
    rows["fitted_iv"] = np.sqrt(rows["fitted_w"] / rows["time_to_expiry"])

    return {
        "rho": float(rho),
        "eta": float(eta),
        "gamma": float(gamma),
        "rmse": float(np.sqrt(np.mean(rows["residual_w"] ** 2))),
        "theta_by_expiry": theta_by_expiry,
        "observations": rows,
    }
