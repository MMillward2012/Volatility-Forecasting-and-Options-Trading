from pathlib import Path

import numpy as np
import pandas as pd

from src.forward import infer_forward_and_discount_factor


MATCHED_DATA_PATH = Path(
    "data/processed/spx_option_prices_matched_2025-08-29.csv"
)

FORWARD_DATA_PATH = Path(
    "data/processed/spx_forward_estimates_2025-08-29.csv"
)

GROUP_KEYS = [
    "security_id",
    "quote_date",
    "expiry_date",
]


def infer_expiry_forwards(matched_options):
    matched_options = matched_options[
        matched_options["time_to_expiry"] > 0
    ]

    results = []

    for keys, group in matched_options.groupby(GROUP_KEYS):
        forward, discount_factor = infer_forward_and_discount_factor(
            group["call_mid"],
            group["put_mid"],
            group["strike"],
        )

        parity_values = group["call_mid"] - group["put_mid"]
        fitted_values = discount_factor * (forward - group["strike"])
        residuals = parity_values - fitted_values

        results.append({
            "security_id": keys[0],
            "quote_date": keys[1],
            "expiry_date": keys[2],
            "days_to_expiry": group["days_to_expiry"].iloc[0],
            "time_to_expiry": group["time_to_expiry"].iloc[0],
            "forward": forward,
            "discount_factor": discount_factor,
            "n_strikes": len(group),
            "parity_rmse": np.sqrt(np.mean(residuals**2)),
        })

    return pd.DataFrame(results)


def main():
    matched_options = pd.read_csv(
        MATCHED_DATA_PATH,
        parse_dates=["quote_date", "expiry_date"],
    )
    forward_estimates = infer_expiry_forwards(matched_options)
    FORWARD_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    forward_estimates.to_csv(FORWARD_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
