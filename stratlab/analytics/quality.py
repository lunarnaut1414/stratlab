"""Reusable per-stock quality filter and cross-sectional regime helpers.

These helpers were independently reimplemented inline by multiple arena agents
across rounds (gen_6, gen_8, gen_9, gen_10). Centralizing them removes
duplication and makes the implementations vectorizable / consistent.

Usage in a Strategy::

    from stratlab.analytics.quality import rolling_max_drawdown, cross_sectional_dispersion

    # Inside on_bar:
    dd = rolling_max_drawdown(close_window, window=21)
    candidates = symbols[dd > -0.12]   # exclude stocks in >12% drawdown

    disp = cross_sectional_dispersion(stock_returns_today)
    if disp > self.threshold:
        # high-dispersion regime = stock-picking environment
        ...

Vectorize where possible. Avoid Python-level for-loops over symbols in
on_bar: the helpers below accept pandas DataFrames keyed by symbol when
the operation is cross-sectional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rolling_max_drawdown(close: pd.Series, window: int) -> pd.Series:
    """Per-bar trailing-window max drawdown of a price series.

    Returns a Series of the same index. At each bar, the value is the largest
    drawdown observed within the trailing ``window`` bars, computed as
    ``min(close / cummax(close) - 1)`` over the window. Values are <= 0; -0.12
    means a 12% peak-trough decline within the last ``window`` bars.

    NaN where lookback is unsatisfied (first ``window-1`` bars).

    Asked for by sonnet-6 (gen_10) after inline reimplementation in
    ``sp500_maxdd_quality_momentum.py``.
    """
    if window < 2:
        raise ValueError(f"window must be >= 2, got {window}")
    # Vectorized: at each bar i, look back `window` bars; running peak / current.
    cummax = close.rolling(window, min_periods=window).max()
    # Drawdown at THE current bar relative to the trailing peak:
    # (close - rolling_max) / rolling_max. Always <= 0.
    return (close - cummax) / cummax


def quality_filter_maxdd(
    close: pd.Series | pd.DataFrame,
    window: int,
    threshold: float,
) -> pd.Series | pd.DataFrame:
    """Boolean mask: True where the trailing-window max drawdown is SHALLOWER
    than ``threshold`` (i.e. ``rolling_max_drawdown >= threshold``).

    ``threshold`` should be a negative number (e.g. -0.12 for "exclude stocks
    in >12% trailing drawdown"). Values of NaN propagate to False.

    Accepts a Series (one symbol) or DataFrame (one column per symbol — applied
    columnwise for cross-sectional screening).
    """
    if threshold > 0:
        raise ValueError(
            f"threshold should be a negative number (drawdown is <= 0); got {threshold}"
        )
    if isinstance(close, pd.DataFrame):
        dd = close.apply(lambda col: rolling_max_drawdown(col, window))
    else:
        dd = rolling_max_drawdown(close, window)
    return (dd >= threshold).fillna(False)


def cross_sectional_dispersion(
    returns: pd.Series | pd.DataFrame,
    method: str = "std",
) -> float | pd.Series:
    """Cross-sectional dispersion of stock returns at a point in time.

    Accepts either:
      - a Series indexed by ticker (one bar's returns across the universe) → returns a scalar
      - a DataFrame indexed by date, columns by ticker → returns a Series of
        per-date dispersion values

    ``method``:
      - ``"std"`` (default): population standard deviation across stocks
      - ``"iqr"``: 75th-25th percentile spread
      - ``"mad"``: median absolute deviation from the cross-sectional median

    Use as a "stock-picking environment" regime signal: high dispersion =
    stock-level alpha worth chasing; low dispersion = mega-cap-led / correlated
    regime where active stock selection underperforms broad ETF allocators.

    Asked for by opus-2 (gen_10) after inline reimplementation across three
    independent strategies.
    """
    if isinstance(returns, pd.DataFrame):
        if method == "std":
            return returns.std(axis=1, ddof=0)
        if method == "iqr":
            return returns.quantile(0.75, axis=1) - returns.quantile(0.25, axis=1)
        if method == "mad":
            median = returns.median(axis=1)
            return (returns.sub(median, axis=0)).abs().median(axis=1)
        raise ValueError(f"unknown method: {method!r}")
    # Series — single-bar value
    if method == "std":
        return float(returns.std(ddof=0))
    if method == "iqr":
        return float(returns.quantile(0.75) - returns.quantile(0.25))
    if method == "mad":
        return float((returns - returns.median()).abs().median())
    raise ValueError(f"unknown method: {method!r}")
