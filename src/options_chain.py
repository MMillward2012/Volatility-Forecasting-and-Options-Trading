import pandas as pd


def calculate_mid_price(bid, ask):
    """Calculate the midpoint between bid and ask prices."""
    return (bid + ask) / 2


def calculate_time_to_expiry(quote_date, expiry_date):
    """Calculate time to expiry in years using the ACT/365 convention."""
    quote_date = pd.to_datetime(quote_date)
    expiry_date = pd.to_datetime(expiry_date)

    return (expiry_date - quote_date) / pd.Timedelta(days=365)
