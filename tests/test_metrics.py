"""Performance metric calculations.

Sharpe / CAGR / max drawdown have well-known closed-form values on
synthetic inputs — easy to verify exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from stratlab.analytics.metrics import compute_metrics, compute_subperiod_metrics


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


def test_compute_subperiod_metrics_smooth_curve():
    """A linearly-rising equity curve has no drawdown in either half;
    Calmar is unbounded by definition (max_dd == 0), so the helper
    returns 0.0 — that's the documented contract."""
    idx = pd.bdate_range("2010-01-04", periods=2000)
    eq = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
    rets = eq.pct_change().fillna(0.0)
    m = compute_subperiod_metrics(eq, rets)
    assert m["is_calmar_h1"] == 0.0
    assert m["is_calmar_h2"] == 0.0
    assert m["is_calmar_min"] == 0.0
    # Linear growth across many years → top-2 years' share is < 50% by construction
    assert 0.0 < m["is_pnl_top2y_pct"] < 0.5


def test_compute_subperiod_metrics_concentrated_year():
    """Curve flat for 5y then doubles in year 6 → top-2y share ≈ 100%."""
    idx = pd.bdate_range("2010-01-04", "2018-12-31")
    eq = pd.Series(100.0, index=idx).copy()
    eq.loc[eq.index >= "2015-01-01"] = 200.0
    rets = eq.pct_change().fillna(0.0)
    m = compute_subperiod_metrics(eq, rets)
    assert m["is_pnl_top2y_pct"] >= 0.95


def test_compute_subperiod_metrics_h2_drawdown_dominates():
    """Both halves have noisy returns; H2 has an extra deep shock so its
    Calmar is materially lower. is_calmar_min must reflect the worse half."""
    rng = np.random.RandomState(7)
    idx = pd.bdate_range("2010-01-04", periods=2000)
    rets = pd.Series(rng.normal(0.0006, 0.005, len(idx)), index=idx)
    # Inject a sustained -30% drawdown over a 100-day window in H2
    midpoint = len(idx) // 2
    rets.iloc[midpoint + 100 : midpoint + 200] -= 0.005
    eq = (1 + rets).cumprod() * 100.0
    m = compute_subperiod_metrics(eq, rets)
    assert m["is_calmar_h1"] > m["is_calmar_h2"]
    assert m["is_calmar_min"] == m["is_calmar_h2"]
