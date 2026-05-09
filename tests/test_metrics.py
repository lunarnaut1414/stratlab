"""Performance metric calculations.

Sharpe / CAGR / max drawdown have well-known closed-form values on
synthetic inputs — easy to verify exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratlab.analytics.metrics import compute_metrics


def test_compute_metrics_flat_equity_zeros_out():
    """Constant equity ⇒ zero return, zero vol, zero Sharpe."""
    eq = pd.Series([100.0] * 252, index=pd.bdate_range("2024-01-01", periods=252))
    rets = eq.pct_change().fillna(0.0)
    m = compute_metrics(eq, rets)
    assert m["total_return"] == 0.0
    assert m["cagr"] == 0.0
    assert m["sharpe"] == 0.0
    assert m["max_drawdown"] == 0.0
    assert m["annual_volatility"] == 0.0


def test_compute_metrics_total_return_matches_simple_diff():
    """Equity 100 → 150 ⇒ total_return 0.5 exactly."""
    eq = pd.Series(np.linspace(100, 150, 252),
                   index=pd.bdate_range("2024-01-01", periods=252))
    rets = eq.pct_change().fillna(0.0)
    m = compute_metrics(eq, rets)
    assert m["total_return"] == pytest.approx(0.5, rel=1e-6)


def test_compute_metrics_max_drawdown_finds_deepest_trough():
    """100 → 120 → 60 → 90: peak 120, trough 60 ⇒ max_drawdown = -0.5."""
    eq = pd.Series([100, 110, 120, 90, 60, 75, 90],
                   index=pd.bdate_range("2024-01-01", periods=7))
    rets = eq.pct_change().fillna(0.0)
    m = compute_metrics(eq, rets)
    assert m["max_drawdown"] == pytest.approx(-0.5, rel=1e-6)


def test_compute_metrics_sharpe_positive_for_steady_uptrend():
    """Pure uptrend with low vol ⇒ Sharpe should be positive."""
    eq = pd.Series(np.linspace(100, 110, 252),
                   index=pd.bdate_range("2024-01-01", periods=252))
    rets = eq.pct_change().fillna(0.0)
    m = compute_metrics(eq, rets)
    assert m["sharpe"] > 0
    assert m["cagr"] > 0


def test_compute_metrics_calmar_uses_max_drawdown():
    """Calmar = |CAGR / MaxDD|. Verify the relationship."""
    eq = pd.Series([100, 90, 110],
                   index=pd.bdate_range("2024-01-01", periods=3))
    rets = eq.pct_change().fillna(0.0)
    m = compute_metrics(eq, rets)
    assert m["calmar"] == pytest.approx(abs(m["cagr"] / m["max_drawdown"]), rel=1e-3)
