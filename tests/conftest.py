"""Shared fixtures: deterministic synthetic OHLCV so tests don't hit yfinance."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def _ohlcv(prices: np.ndarray, start: str = "2024-01-01") -> pd.DataFrame:
    """Build a clean OHLCV frame from a 1-D close-price array.

    Open == High == Low == Close so fills are deterministic regardless of
    which side of the bar the engine reads. Volume is constant.
    """
    idx = pd.bdate_range(start=start, periods=len(prices))
    return pd.DataFrame(
        {
            "open": prices,
            "high": prices,
            "low": prices,
            "close": prices,
            "volume": np.full_like(prices, 1_000_000, dtype=float),
        },
        index=idx,
    )


@pytest.fixture
def linear_ramp() -> pd.DataFrame:
    """100 → 199, +1 per bar over 100 bars. Buying at i, selling at j → PnL = j - i."""
    return _ohlcv(np.arange(100.0, 200.0))


@pytest.fixture
def flat_price() -> pd.DataFrame:
    """100 bars at $100 — pure no-drift. Validates cash conservation."""
    return _ohlcv(np.full(100, 100.0))


@pytest.fixture
def two_assets():
    """Two assets with overlapping but non-identical date ranges."""
    a = _ohlcv(np.arange(100.0, 150.0), start="2024-01-01")          # 50 bars
    b = _ohlcv(np.arange(200.0, 230.0), start="2024-01-15")          # 30 bars, later start
    return {"A": a, "B": b}
