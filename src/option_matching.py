from pathlib import Path

import pandas as pd


CLEANED_DATA_PATH = Path("data/processed/spx_option_prices_cleaned_2025-08-29.csv")
MATCHED_DATA_PATH = Path("data/processed/spx_option_prices_matched_2025-08-29.csv")

MATCH_KEYS = ["security_id", "quote_date", "expiry_date", "strike"]


def match_calls_and_puts(option_chain):
    """Match call and put rows with the same security, date, expiry, and strike."""
    option_chain = option_chain[option_chain["option_type"].isin(["call", "put"])]
    if option_chain[MATCH_KEYS].isna().any().any():
        raise ValueError("Call and put matching keys must not be missing.")

    calls = option_chain[option_chain["option_type"] == "call"][
        MATCH_KEYS
        + [
            "days_to_expiry",
            "time_to_expiry",
            "is_expiry_day",
            "option_id",
            "symbol",
            "best_bid",
            "best_ask",
            "mid_price",
            "volume",
            "open_interest",
        ]
    ].rename(
        columns={
            "option_id": "call_option_id",
            "symbol": "call_symbol",
            "best_bid": "call_bid",
            "best_ask": "call_ask",
            "mid_price": "call_mid",
            "volume": "call_volume",
            "open_interest": "call_open_interest",
        }
    )

    puts = option_chain[option_chain["option_type"] == "put"][
        MATCH_KEYS
        + [
            "option_id",
            "symbol",
            "best_bid",
            "best_ask",
            "mid_price",
            "volume",
            "open_interest",
        ]
    ].rename(
        columns={
            "option_id": "put_option_id",
            "symbol": "put_symbol",
            "best_bid": "put_bid",
            "best_ask": "put_ask",
            "mid_price": "put_mid",
            "volume": "put_volume",
            "open_interest": "put_open_interest",
        }
    )

    return calls.merge(puts, on=MATCH_KEYS, how="inner", validate="one_to_one")


def main():
    cleaned_data = pd.read_csv(
        CLEANED_DATA_PATH,
        parse_dates=["quote_date", "expiry_date"],
    )
    matched_data = match_calls_and_puts(cleaned_data)
    MATCHED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    matched_data.to_csv(MATCHED_DATA_PATH, index=False)


if __name__ == "__main__":
    main()
