from pathlib import Path

import numpy as np
import pandas as pd

from src.implied_vol import implied_volatility_call, implied_volatility_put


CLEANED_DATA_PATH = Path(
    "data/processed/spx_option_prices_cleaned_2025-08-29.csv"
)
FORWARD_DATA_PATH = Path(
    "data/processed/spx_forward_estimates_2025-08-29.csv"
)
IV_PANEL_DATA_PATH = Path(
    "data/processed/spx_iv_panel_2025-08-29.csv"
)

GROUP_KEYS = ["security_id", "quote_date", "expiry_date"]


def _calculate_implied_volatility(row):
    solver = (
        implied_volatility_call
        if row["option_type"] == "call"
        else implied_volatility_put
    )
    try:
        return solver(
            row["mid_price"],
            row["forward"],
            row["strike"],
            row["discount_factor"],
            row["time_to_expiry"],
        )
    except ValueError:
        return np.nan


def _validate_panel_inputs(panel):
    for column in ["forward", "discount_factor", "strike", "time_to_expiry"]:
        values = panel[column].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(values > 0):
            raise ValueError(f"{column} must contain only finite positive values.")


def build_iv_panel(cleaned_options, forward_estimates):
    """Build an implied-volatility analysis panel from cleaned option rows."""
    options = cleaned_options[cleaned_options["days_to_expiry"] > 0].copy()

    if not options["option_type"].isin(["call", "put"]).all():
        raise ValueError("option_type must contain only call and put values.")

    panel = options.merge(
        forward_estimates[GROUP_KEYS + ["forward", "discount_factor"]],
        on=GROUP_KEYS,
        how="inner",
        validate="many_to_one",
    )
    if len(panel) != len(options):
        raise ValueError("Every positive-DTE option must have expiry-level forward data.")

    _validate_panel_inputs(panel)
    panel["log_moneyness"] = np.log(panel["strike"] / panel["forward"])
    panel["mid_iv"] = panel.apply(_calculate_implied_volatility, axis=1)
    panel["is_otm"] = (
        ((panel["option_type"] == "put") & (panel["strike"] < panel["forward"]))
        | ((panel["option_type"] == "call") & (panel["strike"] > panel["forward"]))
    )
    panel["use_for_surface"] = panel["is_otm"] & panel["mid_iv"].notna()

    return panel


def main():
    cleaned_options = pd.read_csv(
        CLEANED_DATA_PATH,
        parse_dates=["quote_date", "expiry_date"],
    )
    forward_estimates = pd.read_csv(
        FORWARD_DATA_PATH,
        parse_dates=["quote_date", "expiry_date"],
    )
    iv_panel = build_iv_panel(cleaned_options, forward_estimates)
    IV_PANEL_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    iv_panel.to_csv(IV_PANEL_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
