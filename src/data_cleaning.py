from pathlib import Path

import numpy as np
import pandas as pd

from src.options_chain import calculate_mid_price


RAW_DATA_PATH = Path("data/raw/spx_option_prices_2025-08-29.csv")
PROCESSED_DATA_PATH = Path("data/processed/spx_option_prices_cleaned_2025-08-29.csv")


def clean_option_data(option_data):
    """Clean raw OptionMetrics option data into the project schema."""
    option_data = option_data.copy()

    option_data["quote_date"] = pd.to_datetime(option_data["date"])
    option_data["expiry_date"] = pd.to_datetime(option_data["exdate"])
    option_data["option_type"] = option_data["cp_flag"].map(
        {"C": "call", "P": "put"}
    )
    option_data["strike"] = option_data["strike_price"] / 1000
    option_data["best_ask"] = option_data["best_offer"]
    option_data["mid_price"] = calculate_mid_price(
        option_data["best_bid"],
        option_data["best_ask"],
    )
    option_data["spread"] = option_data["best_ask"] - option_data["best_bid"]
    option_data["relative_spread"] = option_data["spread"].div(
        option_data["mid_price"].replace(0, np.nan)
    )
    option_data["days_to_expiry"] = (
        option_data["expiry_date"] - option_data["quote_date"]
    ).dt.days
    option_data["time_to_expiry"] = option_data["days_to_expiry"] / 365
    option_data["is_expiry_day"] = option_data["days_to_expiry"] == 0

    return option_data[
        [
            "secid",
            "quote_date",
            "optionid",
            "symbol",
            "expiry_date",
            "option_type",
            "strike",
            "best_bid",
            "best_ask",
            "mid_price",
            "spread",
            "relative_spread",
            "volume",
            "open_interest",
            "days_to_expiry",
            "time_to_expiry",
            "is_expiry_day",
            "expiry_indicator",
            "impl_volatility",
            "delta",
            "gamma",
            "vega",
            "theta",
        ]
    ].rename(
        columns={
            "secid": "security_id",
            "optionid": "option_id",
            "impl_volatility": "vendor_iv",
            "delta": "vendor_delta",
            "gamma": "vendor_gamma",
            "vega": "vendor_vega",
            "theta": "vendor_theta",
        }
    )


def main():
    raw_data = pd.read_csv(RAW_DATA_PATH)
    raw_data = raw_data[raw_data["am_settlement"] == 0]
    cleaned_data = clean_option_data(raw_data)
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    cleaned_data.to_csv(PROCESSED_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
