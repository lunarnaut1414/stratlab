"""Indicator facade smoke test.

We don't unit-test indicator math itself — that's `ta`'s job and they
do it well. But a smoke test catches:

1. The `ta` library renaming a function (breaks our import).
2. Our facade renames drifting from the documented surface.
3. A new `ta` release accidentally changing input/output shapes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from stratlab import indicators


def _ohlcv(n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    close = 100 + np.cumsum(rng.standard_normal(n))
    return pd.DataFrame({
        "open": close,
        "high": close + 1,
        "low": close - 1,
        "close": close,
        "volume": rng.integers(1_000_000, 10_000_000, n).astype(float),
    }, index=pd.bdate_range("2024-01-01", periods=n))


def test_facade_exposes_documented_primitives():
    """Every name in __all__ must actually be importable and callable."""
    expected = {
        "sma", "ema", "wma",
        "macd", "macd_signal", "macd_diff",
        "adx", "aroon_up", "aroon_down", "cci",
        "rsi", "roc", "stoch", "stoch_signal",
        "atr",
        "bb_upper", "bb_lower", "bb_middle", "bb_pband",
        "donchian_upper", "donchian_lower",
        "obv", "mfi", "cmf", "vwap",
    }
    assert expected <= set(indicators.__all__)
    for name in expected:
        fn = getattr(indicators, name)
        assert callable(fn), f"{name} is not callable"


def test_close_only_indicators_return_aligned_series():
    df = _ohlcv()
    for fn in (indicators.sma, indicators.ema, indicators.wma,
               indicators.rsi, indicators.roc, indicators.macd,
               indicators.bb_middle):
        out = fn(df["close"])
        assert isinstance(out, pd.Series)
        assert len(out) == len(df)
        assert (out.index == df.index).all()


def test_ohlc_indicators_return_aligned_series():
    df = _ohlcv()
    for fn_kwargs in (
        (indicators.atr, dict(high=df["high"], low=df["low"], close=df["close"])),
        (indicators.adx, dict(high=df["high"], low=df["low"], close=df["close"])),
        (indicators.cci, dict(high=df["high"], low=df["low"], close=df["close"])),
        (indicators.donchian_upper, dict(high=df["high"], low=df["low"], close=df["close"])),
    ):
        fn, kwargs = fn_kwargs
        out = fn(**kwargs)
        assert isinstance(out, pd.Series)
        assert len(out) == len(df)


def test_volume_indicators_return_aligned_series():
    df = _ohlcv()
    assert len(indicators.obv(df["close"], df["volume"])) == len(df)
    assert len(indicators.mfi(df["high"], df["low"], df["close"], df["volume"])) == len(df)
    assert len(indicators.cmf(df["high"], df["low"], df["close"], df["volume"])) == len(df)
    assert len(indicators.vwap(df["high"], df["low"], df["close"], df["volume"])) == len(df)


def test_rsi_in_zero_to_hundred_range():
    """Sanity: RSI must be bounded [0, 100] after warmup."""
    df = _ohlcv()
    out = indicators.rsi(df["close"], window=14).dropna()
    assert (out >= 0).all() and (out <= 100).all()


def test_bollinger_upper_above_middle_above_lower():
    """Structural invariant: upper band ≥ middle ≥ lower band."""
    df = _ohlcv()
    upper = indicators.bb_upper(df["close"]).dropna()
    middle = indicators.bb_middle(df["close"]).dropna()
    lower = indicators.bb_lower(df["close"]).dropna()
    common = upper.index.intersection(middle.index).intersection(lower.index)
    assert (upper.loc[common] >= middle.loc[common]).all()
    assert (middle.loc[common] >= lower.loc[common]).all()
