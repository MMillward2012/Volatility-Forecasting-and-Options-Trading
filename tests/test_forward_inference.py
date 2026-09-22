import numpy as np
import pandas as pd
import pytest

import src.forward_inference as forward_inference


def matched_options(include_expiry_day=False):
    strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    put_prices = np.array([2.0, 4.0, 8.0, 15.0, 25.0])
    rows = []

    groups = [("2025-09-19", 21, 21 / 365, 100.0, 0.98)]
    if include_expiry_day:
        groups.append(("2025-08-29", 0, 0.0, 100.0, 0.98))

    for expiry_date, days, time, forward, discount_factor in groups:
        call_prices = put_prices + discount_factor * (forward - strikes)
        for strike, call_mid, put_mid in zip(strikes, call_prices, put_prices):
            rows.append({
                "security_id": 108105,
                "quote_date": "2025-08-29",
                "expiry_date": expiry_date,
                "days_to_expiry": days,
                "time_to_expiry": time,
                "strike": strike,
                "call_mid": call_mid,
                "put_mid": put_mid,
            })

    return pd.DataFrame(rows)


def test_infer_expiry_forwards_recovers_synthetic_values():
    estimates = forward_inference.infer_expiry_forwards(matched_options())

    assert len(estimates) == 1
    assert estimates.loc[0, "forward"] == pytest.approx(100.0)
    assert estimates.loc[0, "discount_factor"] == pytest.approx(0.98)
    assert estimates.loc[0, "n_strikes"] == 5
    assert estimates.loc[0, "parity_rmse"] == pytest.approx(0.0)


def test_infer_expiry_forwards_excludes_expiry_day_rows():
    estimates = forward_inference.infer_expiry_forwards(
        matched_options(include_expiry_day=True)
    )

    assert estimates["expiry_date"].tolist() == ["2025-09-19"]
    assert estimates["days_to_expiry"].tolist() == [21]


def test_main_saves_expiry_estimates(tmp_path, monkeypatch):
    input_path = tmp_path / "matched.csv"
    output_path = tmp_path / "forward.csv"
    matched_options().to_csv(input_path, index=False)

    monkeypatch.setattr(forward_inference, "MATCHED_DATA_PATH", input_path)
    monkeypatch.setattr(forward_inference, "FORWARD_DATA_PATH", output_path)

    forward_inference.main()

    saved = pd.read_csv(output_path)
    assert saved.loc[0, "forward"] == pytest.approx(100.0)
    assert saved.loc[0, "discount_factor"] == pytest.approx(0.98)
