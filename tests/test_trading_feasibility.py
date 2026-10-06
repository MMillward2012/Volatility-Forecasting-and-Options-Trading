import numpy as np
import pandas as pd

from src.trading_feasibility import run_development_feasibility
from tests.test_trade_selection import synthetic_chain, synthetic_prices


def mocked_development(monkeypatch, missing_date=None):
    dates = pd.bdate_range("2023-01-03", periods=10)
    prices = synthetic_prices(dates)
    base = synthetic_chain(dates[1])
    files = ["spx_option_prices_2023.csv", "spx_option_prices_2024.csv"]
    def stream(*args):
        for date in dates:
            if date != missing_date:
                yield files[0], date, pd.DataFrame({"date": [date]})
    def prepare(raw, spot, held_option_ids=()):
        chain = base.copy()
        chain["quote_date"] = raw.date.iloc[0]
        chain["days_to_expiry"] = (chain.expiry_date - chain.quote_date).dt.days
        return {"chain": chain, "quotes": chain,
                "diagnostics": {"raw_rows": 12, "pm_rows": 12, "zero_bid_rows": 0,
                                "invalid_quote_rows": 0, "iv_failures": 0, "forward_failures": []}}
    monkeypatch.setattr("src.trading_feasibility.iter_quote_dates", stream)
    monkeypatch.setattr("src.trading_feasibility.prepare_executable_chain", prepare)
    def forbidden(*args, **kwargs):
        raise AssertionError("Feasibility must never evaluate P&L.")
    monkeypatch.setattr("src.trade_backtest.account_trade", forbidden)
    return dates, prices, files


def test_feasibility_stream_keeps_only_coverage_and_boundary_slots(monkeypatch):
    dates, prices, files = mocked_development(monkeypatch)
    result = run_development_feasibility(files, prices)
    assert len(result["daily"]) == 10
    # The fixed fixture expiries eventually age outside the seven-day window.
    assert result["daily"].four_wings_available.sum() == 7
    assert result["daily"].lifecycle_boundary.sum() == 3
    lifecycle = result["lifecycle"]
    assert len(lifecycle) == 4
    assert lifecycle.status.eq("scheduled").all()
    assert lifecycle.complete_quote_path.all()
    assert lifecycle.complete_hedge_inputs.all()
    np.testing.assert_array_equal(lifecycle.session_index, [0, 1, 2, 3])
    assert not any("pnl" in column.lower() or "return" in column.lower() for table in result.values() for column in table)


def test_absent_option_session_stays_in_calendar_and_breaks_path(monkeypatch):
    dates = pd.bdate_range("2023-01-03", periods=10)
    _, prices, files = mocked_development(monkeypatch, missing_date=dates[3])
    result = run_development_feasibility(files, prices)
    missing = result["daily"].iloc[3]
    assert missing.quote_date == dates[3]
    assert missing.raw_rows == 0
    assert not missing.four_wings_available
    assert result["lifecycle"].session_index.to_list() == [0, 1, 3]
    assert not result["lifecycle"].iloc[0].complete_quote_path
    assert not result["lifecycle"].iloc[0].complete_hedge_inputs
    assert result["lifecycle"].iloc[0].status == "scheduled"
