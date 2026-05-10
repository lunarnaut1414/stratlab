from __future__ import annotations

import numpy as np
import pandas as pd


def compute_metrics(
    equity: pd.Series,
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Compute standard performance metrics from an equity curve."""
    total_return = (equity.iloc[-1] / equity.iloc[0]) - 1.0

    n_years = len(equity) / periods_per_year
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0 if n_years > 0 else 0.0

    excess = returns - risk_free_rate / periods_per_year
    sharpe = (
        float(np.sqrt(periods_per_year) * excess.mean() / excess.std())
        if excess.std() > 0
        else 0.0
    )

    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_drawdown = float(drawdown.min())

    downside = returns[returns < 0]
    sortino = (
        float(np.sqrt(periods_per_year) * excess.mean() / downside.std())
        if len(downside) > 0 and downside.std() > 0
        else 0.0
    )

    annual_vol = float(returns.std() * np.sqrt(periods_per_year))

    win_rate = float((returns > 0).sum() / len(returns)) if len(returns) > 0 else 0.0

    calmar = abs(cagr / max_drawdown) if max_drawdown != 0 else 0.0

    return {
        "total_return": round(total_return, 4),
        "cagr": round(cagr, 4),
        "sharpe": round(sharpe, 4),
        "sortino": round(sortino, 4),
        "max_drawdown": round(max_drawdown, 4),
        "annual_volatility": round(annual_vol, 4),
        "calmar": round(calmar, 4),
        "win_rate": round(win_rate, 4),
        "n_trades": 0,
    }


PERIOD_RETURN_KEYS = (
    "return_ytd",
    "return_1y",
    "return_3y_ann",
    "return_5y_ann",
    "return_10y_ann",
    "return_since_inception_ann",
)


def compute_period_returns(equity: pd.Series) -> dict[str, float]:
    """Trailing-period returns relative to the equity curve's last date.

    Mirrors a PIMCO / Morningstar-style fund factsheet: YTD and 1y are total
    returns; 3y / 5y / 10y / since-inception are annualized (geometric CAGR
    over the actual elapsed calendar window). Returns NaN for any period
    longer than the curve's available history — investors should see "—",
    not a misleadingly-annualized partial-period figure.
    """
    nan = float("nan")
    out = {k: nan for k in PERIOD_RETURN_KEYS}
    if len(equity) < 2:
        return out

    end_date = equity.index[-1]
    end_val = float(equity.iloc[-1])
    start_date = equity.index[0]
    start_val = float(equity.iloc[0])
    if start_val <= 0:
        return out

    total_days = (end_date - start_date).days
    if total_days > 0:
        n_years = total_days / 365.25
        out["return_since_inception_ann"] = round(
            (end_val / start_val) ** (1.0 / n_years) - 1.0, 4
        )

    def _trailing(years: int, annualize: bool) -> float:
        cutoff = end_date - pd.DateOffset(years=years)
        if cutoff < start_date:
            return nan
        sliced = equity[equity.index >= cutoff]
        if len(sliced) < 2:
            return nan
        s_val = float(sliced.iloc[0])
        if s_val <= 0:
            return nan
        total = end_val / s_val - 1.0
        if not annualize:
            return round(total, 4)
        elapsed = (sliced.index[-1] - sliced.index[0]).days
        if elapsed <= 0:
            return nan
        return round((end_val / s_val) ** (365.25 / elapsed) - 1.0, 4)

    out["return_1y"] = _trailing(1, annualize=False)
    out["return_3y_ann"] = _trailing(3, annualize=True)
    out["return_5y_ann"] = _trailing(5, annualize=True)
    out["return_10y_ann"] = _trailing(10, annualize=True)

    ytd_cutoff = pd.Timestamp(year=end_date.year, month=1, day=1)
    ytd_sliced = equity[equity.index >= ytd_cutoff]
    if len(ytd_sliced) >= 2:
        ytd_start = float(ytd_sliced.iloc[0])
        if ytd_start > 0:
            out["return_ytd"] = round(end_val / ytd_start - 1.0, 4)

    return out


def _calmar_from_equity(equity: pd.Series, periods_per_year: int = 252) -> float:
    """Calmar from a sliced equity curve. Assumes equity is positive throughout."""
    if len(equity) < 2 or equity.iloc[0] <= 0:
        return 0.0
    n_years = len(equity) / periods_per_year
    if n_years <= 0:
        return 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / n_years) - 1.0
    cummax = equity.cummax()
    drawdown = (equity - cummax) / cummax
    max_dd = float(drawdown.min())
    return float(abs(cagr / max_dd)) if max_dd != 0 else 0.0


def compute_subperiod_metrics(
    equity: pd.Series,
    returns: pd.Series,
    periods_per_year: int = 252,
) -> dict[str, float]:
    """Sub-period stability metrics over a single equity curve.

    These exist because IS-window Calmar can hide regime concentration —
    e.g., a strategy whose returns come almost entirely from 2 of 9 years
    will headline at Calmar 0.8 but degrade catastrophically OOS. Surfacing
    sub-period Calmar and PnL-year concentration at submit time lets the
    leaderboard flag the issue *before* OOS evaluation burns compute.

    Returns:
        is_calmar_h1: Calmar on the first half of the equity curve.
        is_calmar_h2: Calmar on the second half.
        is_calmar_min: min(h1, h2) — single number for ranking by stability.
        is_pnl_top2y_pct: fraction of total log-PnL contributed by the best
            2 calendar years. High values (>0.6) indicate the strategy is
            essentially a bet on a small number of regime windows.
    """
    if len(equity) < 4:
        return {
            "is_calmar_h1": 0.0,
            "is_calmar_h2": 0.0,
            "is_calmar_min": 0.0,
            "is_pnl_top2y_pct": 0.0,
        }
    mid = len(equity) // 2
    h1 = equity.iloc[: mid + 1]
    h2 = equity.iloc[mid:]
    calmar_h1 = _calmar_from_equity(h1, periods_per_year)
    calmar_h2 = _calmar_from_equity(h2, periods_per_year)

    log_returns = np.log1p(returns.fillna(0.0))
    yearly_log = log_returns.groupby(log_returns.index.year).sum()
    if len(yearly_log) == 0 or yearly_log.sum() <= 0:
        top2_pct = 0.0
    else:
        top2 = yearly_log.nlargest(min(2, len(yearly_log))).sum()
        total = yearly_log.sum()
        top2_pct = float(top2 / total) if total > 0 else 0.0

    return {
        "is_calmar_h1": round(float(calmar_h1), 4),
        "is_calmar_h2": round(float(calmar_h2), 4),
        "is_calmar_min": round(min(float(calmar_h1), float(calmar_h2)), 4),
        "is_pnl_top2y_pct": round(top2_pct, 4),
    }
