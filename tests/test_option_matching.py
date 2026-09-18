import pandas as pd
import pytest

from src.option_matching import MATCH_KEYS, match_calls_and_puts


@pytest.fixture
def option_chain():
    return pd.DataFrame({
        "security_id": [108105, 108105],
        "quote_date": ["2025-08-29", "2025-08-29"],
        "expiry_date": ["2025-09-19", "2025-09-19"],
        "strike": [6500.0, 6500.0],
        "option_type": ["call", "put"],
        "days_to_expiry": [21, 21],
        "time_to_expiry": [21 / 365, 21 / 365],
        "is_expiry_day": [False, False],
        "option_id": [1, 2],
        "symbol": ["call-6500", "put-6500"],
        "best_bid": [100.0, 90.0],
        "best_ask": [102.0, 92.0],
        "mid_price": [101.0, 91.0],
        "volume": [10, 20],
        "open_interest": [100, 200],
    })


def test_match_calls_and_puts_returns_expected_schema(option_chain):
    matched = match_calls_and_puts(option_chain)

    assert len(matched) == 1
    assert list(matched.columns) == [
        "security_id",
        "quote_date",
        "expiry_date",
        "strike",
        "days_to_expiry",
        "time_to_expiry",
        "is_expiry_day",
        "call_option_id",
        "call_symbol",
        "call_bid",
        "call_ask",
        "call_mid",
        "call_volume",
        "call_open_interest",
        "put_option_id",
        "put_symbol",
        "put_bid",
        "put_ask",
        "put_mid",
        "put_volume",
        "put_open_interest",
    ]


def test_match_calls_and_puts_preserves_values(option_chain):
    original = option_chain.copy(deep=True)
    matched = match_calls_and_puts(option_chain).iloc[0]

    for column in MATCH_KEYS + ["days_to_expiry", "time_to_expiry", "is_expiry_day"]:
        assert matched[column] == option_chain.iloc[0][column]
    for side, index in [("call", 0), ("put", 1)]:
        for source, target in [
            ("option_id", "option_id"), ("symbol", "symbol"),
            ("best_bid", "bid"), ("best_ask", "ask"), ("mid_price", "mid"),
            ("volume", "volume"), ("open_interest", "open_interest"),
        ]:
            assert matched[f"{side}_{target}"] == option_chain.iloc[index][source]

    pd.testing.assert_frame_equal(option_chain, original)


@pytest.mark.parametrize("key,value", [
    ("security_id", 123456),
    ("quote_date", "2025-08-28"),
    ("expiry_date", "2025-10-17"),
    ("strike", 6600.0),
])
def test_match_calls_and_puts_excludes_different_keys(option_chain, key, value):
    option_chain.loc[1, key] = value

    assert match_calls_and_puts(option_chain).empty


def test_match_calls_and_puts_keeps_only_complete_pairs(option_chain):
    second_pair = option_chain.copy()
    second_pair["strike"] = 6600.0
    second_pair["option_id"] += 2
    unmatched_call = option_chain.iloc[[0]].copy()
    unmatched_call["strike"] = 6700.0
    unmatched_call["option_id"] = 5
    chain = pd.concat([option_chain, second_pair, unmatched_call], ignore_index=True)

    matched = match_calls_and_puts(chain)

    assert matched["strike"].tolist() == [6500.0, 6600.0]
    assert matched[["call_option_id", "put_option_id"]].values.tolist() == [[1, 2], [3, 4]]
    assert not matched.duplicated(MATCH_KEYS).any()


@pytest.mark.parametrize("side", ["call", "put"])
def test_match_calls_and_puts_rejects_duplicate_keys(option_chain, side):
    duplicate = option_chain[option_chain["option_type"] == side]
    chain = pd.concat([option_chain, duplicate], ignore_index=True)

    with pytest.raises(pd.errors.MergeError):
        match_calls_and_puts(chain)


@pytest.mark.parametrize("key", MATCH_KEYS)
@pytest.mark.parametrize("side", ["call", "put", "both"])
def test_match_calls_and_puts_rejects_missing_keys(option_chain, key, side):
    rows = option_chain.index if side == "both" else option_chain.index[
        option_chain["option_type"] == side
    ]
    option_chain.loc[rows, key] = None

    with pytest.raises(ValueError, match="matching keys must not be missing"):
        match_calls_and_puts(option_chain)


def test_match_calls_and_puts_preserves_expiry_day_flag(option_chain):
    option_chain["expiry_date"] = option_chain["quote_date"]
    option_chain["days_to_expiry"] = 0
    option_chain["time_to_expiry"] = 0.0
    option_chain["is_expiry_day"] = True

    matched = match_calls_and_puts(option_chain)

    assert len(matched) == 1
    assert matched["is_expiry_day"].all()
    assert (matched["time_to_expiry"] == 0).all()
