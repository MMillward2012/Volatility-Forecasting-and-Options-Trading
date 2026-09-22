import numpy as np
import pandas as pd
import pytest

from src.iv_panel import build_iv_panel
from src.pricing import black_scholes_call_price, black_scholes_put_price


FORWARD = 100.0
DISCOUNT_FACTOR = 0.98
TIME_TO_EXPIRY = 0.5
VOLATILITY = 0.25


def expiry_estimate():
    return pd.DataFrame([{
        "security_id": 108105,
        "quote_date": "2025-08-29",
        "expiry_date": "2025-09-19",
        "forward": FORWARD,
        "discount_factor": DISCOUNT_FACTOR,
    }])


def option_rows():
    return pd.DataFrame([
        {
            "security_id": 108105,
            "quote_date": "2025-08-29",
            "expiry_date": "2025-09-19",
            "days_to_expiry": 21,
            "time_to_expiry": TIME_TO_EXPIRY,
            "option_type": "put",
            "strike": 90.0,
            "mid_price": black_scholes_put_price(
                FORWARD,
                90.0,
                DISCOUNT_FACTOR,
                TIME_TO_EXPIRY,
                VOLATILITY,
            ),
            "vendor_iv": 0.2,
        },
        {
            "security_id": 108105,
            "quote_date": "2025-08-29",
            "expiry_date": "2025-09-19",
            "days_to_expiry": 21,
            "time_to_expiry": TIME_TO_EXPIRY,
            "option_type": "call",
            "strike": 110.0,
            "mid_price": black_scholes_call_price(
                FORWARD,
                110.0,
                DISCOUNT_FACTOR,
                TIME_TO_EXPIRY,
                VOLATILITY,
            ),
            "vendor_iv": 0.2,
        },
    ])


def test_build_iv_panel_recovers_iv_and_log_moneyness():
    panel = build_iv_panel(option_rows(), expiry_estimate())

    assert np.allclose(panel["mid_iv"], VOLATILITY)
    assert np.allclose(
        panel["log_moneyness"],
        np.log(panel["strike"] / FORWARD),
    )
    assert panel["use_for_surface"].tolist() == [True, True]


def test_build_iv_panel_marks_invalid_midpoint():
    options = option_rows()
    options.loc[1, "mid_price"] = DISCOUNT_FACTOR * FORWARD

    panel = build_iv_panel(options, expiry_estimate())

    assert pd.isna(panel.loc[1, "mid_iv"])
    assert not panel.loc[1, "use_for_surface"]


def test_build_iv_panel_rejects_missing_forward_data():
    with pytest.raises(ValueError, match="Every positive-DTE"):
        build_iv_panel(option_rows(), expiry_estimate().iloc[0:0])


@pytest.mark.parametrize(
    "column",
    ["forward", "discount_factor", "strike", "time_to_expiry"],
)
def test_build_iv_panel_rejects_invalid_positive_inputs(column):
    options = option_rows()
    forwards = expiry_estimate()

    if column == "strike":
        options[column] = 0.0
    elif column == "time_to_expiry":
        options[column] = np.nan
    else:
        forwards.loc[0, column] = 0.0

    with pytest.raises(ValueError, match=column):
        build_iv_panel(options, forwards)
