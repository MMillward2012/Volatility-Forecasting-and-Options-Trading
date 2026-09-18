import numpy as np
import pandas as pd
import pytest

from src.options_chain import calculate_mid_price, calculate_time_to_expiry
from helper import synthetic_option_chain


def test_calculate_mid_price_supports_scalars_and_arrays():
    assert calculate_mid_price(100.0, 102.0) == 101.0

    result = calculate_mid_price(
        np.array([100.0, 102.0]),
        np.array([102.0, 104.0]),
    )

    np.testing.assert_array_equal(result, np.array([101.0, 103.0]))


def test_calculate_time_to_expiry_uses_act_365():
    assert calculate_time_to_expiry("2026-01-01", "2026-07-01") == pytest.approx(
        181 / 365
    )

    quote_dates = pd.Series(["2026-01-01", "2026-04-01"])
    expiry_dates = pd.Series(["2026-07-01", "2026-10-01"])

    result = calculate_time_to_expiry(quote_dates, expiry_dates)

    np.testing.assert_allclose(result, np.array([181 / 365, 183 / 365]))


def test_synthetic_option_chain_helper():
    option_chain = synthetic_option_chain()
    expected_columns = [
        "quote_date",
        "expiry_date",
        "strike",
        "option_type",
        "bid",
        "ask",
    ]

    assert list(option_chain.columns) == expected_columns
    assert len(option_chain) == 10
    assert set(option_chain["option_type"]) == {"call", "put"}
    np.testing.assert_array_equal(
        option_chain["bid"],
        option_chain["ask"],
    )
