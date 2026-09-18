import pandas as pd


def calculate_mid_price(bid, ask):
    """Calculate the midpoint between bid and ask prices."""
    return (bid + ask) / 2


def calculate_time_to_expiry(quote_date, expiry_date):
    """Calculate time to expiry in years using the ACT/365 convention."""
    quote_date = pd.to_datetime(quote_date)
    expiry_date = pd.to_datetime(expiry_date)

    return (expiry_date - quote_date) / pd.Timedelta(days=365)


def match_calls_and_puts(option_chain):
    """Match calls and puts with the same quote date, expiry, and strike."""
    option_chain = option_chain.copy()

    option_chain["mid_price"] = calculate_mid_price(
        option_chain["bid"],
        option_chain["ask"],
    )

    match_columns = ["quote_date", "expiry_date", "strike"]

    calls = option_chain[
        option_chain["option_type"] == "call"
    ][
        match_columns + ["mid_price"]
    ].rename(
        columns={"mid_price": "call_mid"}
    )

    puts = option_chain[
        option_chain["option_type"] == "put"
    ][
        match_columns + ["mid_price"]
    ].rename(
        columns={"mid_price": "put_mid"}
    )

    return calls.merge(
        puts,
        on=match_columns,
        how="inner",
    )
